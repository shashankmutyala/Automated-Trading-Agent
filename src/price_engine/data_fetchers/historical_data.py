import pandas as pd
import logging
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """
    Fetches historical cryptocurrency market data from Binance.
    Provides methods to get historical klines/candlesticks for a specified time range.
    """

    def __init__(self, api_key=None, api_secret=None):
        """
        Initialize the historical data fetcher.

        Args:
            api_key (str, optional): Binance API key
            api_secret (str, optional): Binance API secret
        """
        try:
            self.client = Client(api_key, api_secret) if api_key and api_secret else Client()
            # Test the connection
            self.client.get_system_status()
            logger.info("Initialized historical data fetcher")
        except BinanceAPIException as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing Binance client: {e}")
            raise

    def fetch_historical_data(self, symbol, interval='1h', start_date=None, end_date=None, limit=1000):
        """
        Fetch historical kline data from Binance

        Args:
            symbol (str): Trading pair (e.g., 'BNBUSDT')
            interval (str): Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            limit (int): Maximum number of records to return per API call

        Returns:
            pandas.DataFrame: DataFrame with historical data or None if error
        """
        try:
            if not start_date:
                # Default to 30 days ago
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')

            logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")

            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

            # For large date ranges, we might need to make multiple API calls
            all_klines = []
            current_start = start_ts

            while current_start < end_ts:
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=current_start,
                    end_str=end_ts,
                    limit=limit
                )

                if not klines:
                    break

                all_klines.extend(klines)

                # Update start time for next batch
                # Add 1ms to the last timestamp to avoid duplicates
                current_start = klines[-1][0] + 1

                # If we got fewer than limit, we've reached the end
                if len(klines) < limit:
                    break

                logger.debug(f"Fetched batch of {len(klines)} klines, total so far: {len(all_klines)}")

            if not all_klines:
                logger.warning(f"No historical data retrieved for {symbol}")
                return None

            # Convert to pandas DataFrame
            df = pd.DataFrame(all_klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                df[col] = df[col].astype(float)

            # Convert timestamp to datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            # Sort by time to ensure chronological order
            df = df.sort_values('open_time')

            logger.info(f"Successfully fetched {len(df)} historical data points for {symbol}")
            return df

        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching historical data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    def fetch_historical_trades(self, symbol, start_time=None, end_time=None, limit=1000):
        """
        Fetch historical trades for a symbol

        Args:
            symbol (str): Trading pair (e.g., 'BNBUSDT')
            start_time (int, optional): Start timestamp in milliseconds
            end_time (int, optional): End timestamp in milliseconds
            limit (int): Maximum number of trades to return

        Returns:
            list: List of trades or None if error
        """
        try:
            trades = self.client.get_aggregate_trades(
                symbol=symbol,
                startTime=start_time,
                endTime=end_time,
                limit=limit
            )

            logger.info(f"Fetched {len(trades)} historical trades for {symbol}")
            return trades
        except Exception as e:
            logger.error(f"Error fetching historical trades for {symbol}: {e}")
            return None