import requests
import logging
import pandas as pd
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)


class BinanceLiveDataFetcher:
    """
    Fetches live cryptocurrency market data from Binance.
    Provides methods to get current prices and recent klines/candlesticks.
    """

    def __init__(self, api_key=None, api_secret=None):
        """
        Initialize the Binance data fetcher.

        Args:
            api_key (str, optional): Binance API key
            api_secret (str, optional): Binance API secret
        """
        try:
            self.client = Client(api_key, api_secret) if api_key and api_secret else Client()
            # Test the connection
            self.client.get_system_status()
            logger.info("Initialized Binance live data fetcher")
        except BinanceAPIException as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing Binance client: {e}")
            raise

    def get_current_price(self, symbol):
        """
        Get current ticker price for a symbol.

        Args:
            symbol (str): Trading pair (e.g., 'BNBUSDT')

        Returns:
            float: Current price or None if error
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            logger.info(f"Fetched current price for {symbol}: {price}")
            return price
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching price for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    def get_latest_klines(self, symbol, interval='1m', limit=100):
        """
        Get latest klines/candlesticks for a symbol.

        Args:
            symbol (str): Trading pair (e.g., 'BNBUSDT')
            interval (str): Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit (int): Number of klines to retrieve (max 1000)

        Returns:
            pandas.DataFrame: DataFrame with kline data or None if error
        """
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )

            # Convert to pandas DataFrame
            df = pd.DataFrame(klines, columns=[
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

            logger.info(f"Fetched {len(df)} klines for {symbol} with {interval} interval")
            return df
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching klines for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return None

    def get_market_depth(self, symbol, limit=100):
        """
        Get market depth (order book) for a symbol.

        Args:
            symbol (str): Trading pair (e.g., 'BNBUSDT')
            limit (int): Number of bids/asks to retrieve (max 1000)

        Returns:
            dict: Dictionary with bids and asks or None if error
        """
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            logger.info(f"Fetched market depth for {symbol} with {limit} levels")
            return depth
        except Exception as e:
            logger.error(f"Error fetching market depth for {symbol}: {e}")
            return None

    def get_24hr_stats(self, symbol):
        """
        Get 24-hour statistics for a symbol.

        Args:
            symbol (str): Trading pair (e.g., 'BNBUSDT')

        Returns:
            dict: Dictionary with 24hr statistics or None if error
        """
        try:
            stats = self.client.get_ticker(symbol=symbol)
            logger.info(f"Fetched 24hr stats for {symbol}")
            return stats
        except Exception as e:
            logger.error(f"Error fetching 24hr stats for {symbol}: {e}")
            return None