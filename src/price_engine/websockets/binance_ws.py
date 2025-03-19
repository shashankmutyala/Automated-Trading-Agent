import logging
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import requests
from .db.mongodb_handler import MongoDBHandler
from .websockets.binance_ws import BinanceWebSocket

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BINANCE_API_URL = "https://api.binance.com/api/v3"


class PriceDataManager:
    def __init__(self, db_handler=None):
        """
        Initialize price data manager.

        Args:
            db_handler: MongoDB handler for storing and retrieving data
        """
        self.db_handler = db_handler or MongoDBHandler()
        self.websocket = None

    async def start_live_data(self, symbols=None):
        """
        Start collecting live price data from Binance WebSocket.

        Args:
            symbols (list): List of symbols to track (e.g., ["btcusdt", "ethusdt"])
        """
        symbols = symbols or ["btcusdt", "ethusdt"]
        self.websocket = BinanceWebSocket(symbols)

        # Add callback to store data in MongoDB
        self.websocket.add_callback(self.store_live_price)

        # Connect to WebSocket
        await self.websocket.connect()

    async def stop_live_data(self):
        """Stop live data collection."""
        if self.websocket:
            await self.websocket.disconnect()

    async def store_live_price(self, symbol, price, timestamp):
        """
        Callback to store live price data in MongoDB.

        Args:
            symbol (str): Trading pair symbol
            price (float): Current price
            timestamp (int): Unix timestamp in milliseconds
        """
        try:
            self.db_handler.store_price("price_data", symbol, price, timestamp)
            logger.debug(f"Stored live price: {symbol} = {price}")
        except Exception as e:
            logger.error(f"Failed to store live price: {e}")

    async def get_price_history(self, symbol, start_time, end_time=None,
                                use_api_if_needed=True):
        """
        Get price history from database, optionally fetching missing data from API.

        Args:
            symbol (str): Trading pair symbol
            start_time (datetime): Start time
            end_time (datetime): End time (default: current time)
            use_api_if_needed (bool): Whether to fetch missing data from API

        Returns:
            pandas.DataFrame: Historical price data
        """
        # Convert datetime objects to milliseconds timestamp if needed
        if isinstance(start_time, datetime):
            start_time_ms = int(start_time.timestamp() * 1000)
        else:
            start_time_ms = start_time
            start_time = datetime.fromtimestamp(start_time / 1000)

        if end_time is None:
            end_time = datetime.now()
            end_time_ms = int(end_time.timestamp() * 1000)
        elif isinstance(end_time, datetime):
            end_time_ms = int(end_time.timestamp() * 1000)
        else:
            end_time_ms = end_time
            end_time = datetime.fromtimestamp(end_time / 1000)

        # First, check our database for the requested data
        df = self.db_handler.get_price_dataframe(
            symbol=symbol.upper(),
            collection_name="price_data"
        )

        # Filter by time range
        if not df.empty:
            df = df[(df['timestamp'] >= start_time_ms) & (df['timestamp'] <= end_time_ms)]

        # Check if we have complete data
        if df.empty or use_api_if_needed:
            have_complete_data = not df.empty

            if have_complete_data:
                # Check for gaps in data by analyzing timestamp differences
                df = df.sort_values('timestamp')
                time_diffs = df['timestamp'].diff()

                # Assuming trades happen frequently, gaps > 5 minutes indicate missing data
                max_acceptable_gap = 5 * 60 * 1000  # 5 minutes in milliseconds

                if time_diffs.max() > max_acceptable_gap:
                    have_complete_data = False
                    logger.info(f"Found gaps in stored data for {symbol}. Will fetch from API.")

            # If we don't have complete data and are allowed to use API
            if not have_complete_data and use_api_if_needed:
                logger.info(f"Fetching missing data from Binance API: {symbol} from {start_time} to {end_time}")
                api_data = await self.fetch_historical_klines(
                    symbol=symbol,
                    interval="1m",  # Use 1-minute candles for detailed data
                    start_time=start_time_ms,
                    end_time=end_time_ms
                )

                if not api_data.empty:
                    # If we have some data, merge with API data to fill gaps
                    if not df.empty:
                        df = pd.concat([df, api_data])
                        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                    else:
                        df = api_data

        return df

    async def fetch_historical_klines(self, symbol, interval, start_time, end_time=None, limit=1000):
        """
        Fetch historical kline/candlestick data from Binance API.

        Args:
            symbol (str): Trading pair symbol
            interval (str): Kline interval (1m, 3m, 5m, 15m, etc.)
            start_time (int): Start time in milliseconds
            end_time (int): End time in milliseconds
            limit (int): Max number of records per request

        Returns:
            pandas.DataFrame: DataFrame with historical data
        """
        # Handle parameters
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'startTime': start_time,
            'limit': limit
        }

        if end_time:
            params['endTime'] = end_time

        all_klines = []
        current_start = start_time

        while True:
            try:
                url = f"{BINANCE_API_URL}/klines"
                response = requests.get(url, params=params)
                response.raise_for_status()
                klines = response.json()

                if not klines:
                    break

                # Store in database as we fetch
                for kline in klines:
                    timestamp = kline[0]  # Open time
                    open_price = float(kline[1])
                    high = float(kline[2])
                    low = float(kline[3])
                    close = float(kline[4])
                    volume = float(kline[5])

                    # Store candle in database
                    await asyncio.to_thread(
                        self.db_handler.create_candle,
                        symbol=symbol.upper(),
                        interval=interval,
                        open_price=open_price,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume,
                        timestamp=timestamp
                    )

                    # Store trade price point as well (using close price)
                    await asyncio.to_thread(
                        self.db_handler.store_price,
                        "price_data",
                        symbol.upper(),
                        close,
                        timestamp
                    )

                all_klines.extend(klines)

                # Update start time for next batch
                current_start = klines[-1][0] + 1
                params['startTime'] = current_start

                # Check if we've reached the end time
                if end_time and current_start >= end_time:
                    break

                # Respect rate limits
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to fetch historical data: {e}")
                break

        if not all_klines:
            return pd.DataFrame()

        # Convert to DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                   'close_time', 'quote_asset_volume', 'number_of_trades',
                   'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

        df = pd.DataFrame(all_klines, columns=columns)

        # Convert types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                           'quote_asset_volume', 'taker_buy_base_asset_volume',
                           'taker_buy_quote_asset_volume']

        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['symbol'] = symbol.upper()
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df

