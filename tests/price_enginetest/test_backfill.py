# tests/test_backfill.py
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from src.websocket_client import BinanceWebSocketClient
from src.api_client import BinanceAPIClient

class TestBackfill(unittest.TestCase):
    def setUp(self):
        self.received_data = []
        self.api_client = MagicMock(spec=BinanceAPIClient)
        self.client = BinanceWebSocketClient(symbol="btcusdt", on_message_callback=self.store_data)
        self.client.last_data_time = datetime.now() - timedelta(minutes=5)

    def store_data(self, data):
        self.received_data.append(data)

    def test_backfill_missed_data(self):
        # Mock the API client's fetch_historical_data method
        mock_data = [
            {"timestamp": datetime.now(), "close": 50000.0, "volume": 100.0},
            {"timestamp": datetime.now(), "close": 51000.0, "volume": 200.0}
        ]
        self.api_client.fetch_historical_data.return_value = mock_data

        # Run backfill
        self.client.backfill_missed_data(self.api_client)

        # Verify that backfilled data was processed
        self.assertEqual(len(self.received_data), 2)
        self.assertEqual(self.received_data[0]["price"], 50000.0)
        self.assertEqual(self.received_data[1]["price"], 51000.0)

if __name__ == "__main__":
    unittest.main()