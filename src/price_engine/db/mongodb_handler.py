from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MongoDBHandler:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="price_data"):
        """
        Initialize MongoDB connection and set up collections.

        Args:
            uri (str): MongoDB connection string
            db_name (str): Database name to use
        """
        try:
            self.client = MongoClient(uri)
            # Test the connection
            self.client.admin.command('ping')
            logger.info("Connected to MongoDB successfully")

            self.db = self.client[db_name]

            # Create indexes for better query performance
            self.db.price_data.create_index([("symbol", ASCENDING), ("timestamp", DESCENDING)])
            self.db.candles.create_index([("symbol", ASCENDING), ("interval", ASCENDING), ("timestamp", DESCENDING)])
            self.db.signals.create_index([("symbol", ASCENDING), ("timestamp", DESCENDING)])

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except OperationFailure as e:
            logger.error(f"MongoDB operation failed: {e}")
            raise

    def store_price(self, collection_name, symbol, price, timestamp):
        """
        Store a price point in MongoDB.

        Args:
            collection_name (str): Collection to store the price in
            symbol (str): Trading pair symbol (e.g., BTCUSDT)
            price (float): Current price
            timestamp (int): Unix timestamp in milliseconds

        Returns:
            str: ID of inserted document or None if operation failed
        """
        try:
            collection = self.db[collection_name]
            data = {
                "symbol": symbol,
                "price": float(price),
                "timestamp": int(timestamp),
                "stored_at": datetime.utcnow()
            }
            result = collection.insert_one(data)
            logger.debug(f"Stored price for {symbol}: {price} at timestamp {timestamp}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to store price data: {e}")
            return None

    def create_candle(self, symbol, interval, open_price, high, low, close, volume, timestamp):
        """
        Store OHLCV candle data.

        Args:
            symbol (str): Trading pair symbol
            interval (str): Candle interval (1m, 5m, 15m, 1h, etc.)
            open_price (float): Opening price
            high (float): Highest price
            low (float): Lowest price
            close (float): Closing price
            volume (float): Trading volume
            timestamp (int): Candle timestamp in milliseconds

        Returns:
            str: ID of inserted document or None if operation failed
        """
        try:
            candle = {
                "symbol": symbol,
                "interval": interval,
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
                "timestamp": int(timestamp),
                "stored_at": datetime.utcnow()
            }
            result = self.db.candles.insert_one(candle)
            logger.debug(f"Stored candle for {symbol} {interval} at {timestamp}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to store candle data: {e}")
            return None

    def get_price_history(self, symbol, start_time=None, end_time=None, limit=1000, collection_name="price_data"):
        """
        Get historical price data for a symbol.

        Args:
            symbol (str): Trading pair symbol
            start_time (int): Optional start timestamp in milliseconds
            end_time (int): Optional end timestamp in milliseconds
            limit (int): Maximum number of records to return
            collection_name (str): Collection to query

        Returns:
            list: List of price documents
        """
        try:
            collection = self.db[collection_name]
            query = {"symbol": symbol}

            # Add time range filter if provided
            if start_time is not None or end_time is not None:
                query["timestamp"] = {}
                if start_time is not None:
                    query["timestamp"]["$gte"] = start_time
                if end_time is not None:
                    query["timestamp"]["$lte"] = end_time

            cursor = collection.find(
                query,
                sort=[("timestamp", DESCENDING)],
                limit=limit
            )
            return list(cursor)
        except Exception as e:
            logger.error(f"Failed to retrieve price history: {e}")
            return []

    def get_price_dataframe(self, symbol, start_time=None, end_time=None, collection_name="price_data", limit=10000):
        """
        Get historical price data as a pandas DataFrame.

        Args:
            symbol (str): Trading pair symbol
            start_time (int): Optional start timestamp in milliseconds
            end_time (int): Optional end timestamp in milliseconds
            collection_name (str): Collection to query
            limit (int): Maximum number of records to return

        Returns:
            pandas.DataFrame: DataFrame with price data
        """
        try:
            data = self.get_price_history(symbol, start_time, end_time, limit, collection_name)
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            # Sort by timestamp
            df = df.sort_values('timestamp')
            return df
        except Exception as e:
            logger.error(f"Failed to create DataFrame: {e}")
            return pd.DataFrame()

    def get_candle_dataframe(self, symbol, interval, start_time=None, end_time=None, limit=1000):
        """
        Get historical candle data as a pandas DataFrame.

        Args:
            symbol (str): Trading pair symbol
            interval (str): Candle interval (1m, 5m, 15m, 1h, etc.)
            start_time (int): Optional start timestamp in milliseconds
            end_time (int): Optional end timestamp in milliseconds
            limit (int): Maximum number of records to return

        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        try:
            query = {
                "symbol": symbol,
                "interval": interval
            }

            # Add time range filter if provided
            if start_time is not None or end_time is not None:
                query["timestamp"] = {}
                if start_time is not None:
                    query["timestamp"]["$gte"] = start_time
                if end_time is not None:
                    query["timestamp"]["$lte"] = end_time

            cursor = self.db.candles.find(
                query,
                sort=[("timestamp", ASCENDING)],
                limit=limit
            )

            data = list(cursor)
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df

        except Exception as e:
            logger.error(f"Failed to retrieve candle data: {e}")
            return pd.DataFrame()

    def get_latest_price(self, symbol, collection_name="price_data"):
        """
        Get the latest price for a symbol.

        Args:
            symbol (str): Trading pair symbol
            collection_name (str): Collection to query

        Returns:
            dict: Latest price document or None if not found
        """
        try:
            collection = self.db[collection_name]
            result = collection.find_one(
                {"symbol": symbol},
                sort=[("timestamp", DESCENDING)]
            )
            return result
        except Exception as e:
            logger.error(f"Failed to retrieve latest price: {e}")
            return None

    def store_signal(self, symbol, signal_type, metrics, timestamp):
        """
        Store a trading signal in MongoDB.

        Args:
            symbol (str): Trading pair symbol
            signal_type (str): Type of signal (BUY, SELL, etc.)
            metrics (dict): Additional metrics that generated the signal
            timestamp (int): Unix timestamp in milliseconds

        Returns:
            str: ID of inserted document or None if operation failed
        """
        try:
            data = {
                "symbol": symbol,
                "signal": signal_type,
                "metrics": metrics,
                "timestamp": timestamp,
                "created_at": datetime.utcnow()
            }
            result = self.db.signals.insert_one(data)
            logger.info(f"Stored signal {signal_type} for {symbol}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to store signal data: {e}")
            return None

    def close(self):
        """Close the MongoDB connection."""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
                logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")