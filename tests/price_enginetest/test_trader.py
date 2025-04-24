# tests/test_trader.py
import unittest
from datetime import datetime, timedelta
from src.trader import Trader
from src.database import Database

class TestTrader(unittest.TestCase):
    def setUp(self):
        self.db = Database()
        self.trader = Trader()

        # Insert sample data into the database for testing
        data = {
            "timestamp": datetime.now() - timedelta(hours=1),
            "price": 50000.0,
            "volume": 100.0,
            "source": "binance"
        }
        self.db.store_price(data, "BTCUSDT")

    def tearDown(self):
        # Clean up the database
        self.db.pg_cursor.execute("DELETE FROM prices;")
        self.db.pg_conn.commit()
        self.db.close()

    def test_get_historical_data(self):
        # Test fetching historical data from the database
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        data = self.trader.get_historical_data("BTCUSDT", start_time, end_time)
        self.assertGreater(len(data), 0)
        self.assertEqual(data[0][1], 50000.0)  # Price

if __name__ == "__main__":
    unittest.main()