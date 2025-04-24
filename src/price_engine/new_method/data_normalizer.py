# new_method/data_normalizer.py
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataNormalizer:
    @staticmethod
    def normalize_data(data):
        """
        Normalize data to a consistent format.
        :param data: Raw data from Binance or Uniswap
        :return: Normalized data dictionary
        """
        price = data.get("price", data.get("close"))
        if price is None:
            raise ValueError(f"No price or close value found in data: {data}")
        normalized = {
            "timestamp": data["timestamp"],
            "price": float(price),
            "volume": float(data.get("volume", 0.0)),
            "source": data.get("source", "unknown")
        }
        return normalized

    @staticmethod
    def handle_missing_data(data_list):
        """
        Handle missing data by filling in gaps or using fallback values.
        :param data_list: List of data dictionaries
        :return: List of reconciled data dictionaries
        """
        reconciled = []
        last_price = None
        for data in data_list:
            if data["price"] is None:
                if last_price is not None:
                    data["price"] = last_price
                else:
                    logger.warning(f"Missing price data at {data['timestamp']}, no fallback available")
            else:
                last_price = data["price"]
            reconciled.append(data)
        return reconciled