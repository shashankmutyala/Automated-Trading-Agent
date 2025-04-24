import unittest
from datetime import datetime, timedelta
from src.api_client import BinanceAPIClient

class TestBinanceAPIClient(unittest.TestCase):
    def setUp(self):
        self.client = BinanceAPIClient()

    def test_fetch_historical_data(self):
        # Test fetching historical data for the last 1 day
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        data = self.client.fetch_historical_data("BTCUSDT", "1h", start_time, end_time)
        
        # Verify that data is returned and has the expected structure
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("timestamp", data[0])
        self.assertIn("open", data[0])
        self.assertIn("high", data[0])
        self.assertIn("low", data[0])
        self.assertIn("close", data[0])
        self.assertIn("volume", data[0])

    def test_fetch_historical_data_invalid_symbol(self):
        # Test fetching data for an invalid symbol (should raise an exception)
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        with self.assertRaises(Exception):
            self.client.fetch_historical_data("INVALID_SYMBOL", "1h", start_time, end_time)

if __name__ == "__main__":
    unittest.main()