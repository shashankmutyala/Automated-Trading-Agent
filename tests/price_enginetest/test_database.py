# tests/test_database.py
import unittest
from datetime import datetime
from src.database import Database

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db = Database()

    def tearDown(self):
        # Clean up the database after each test
        self.db.pg_cursor.execute("DELETE FROM prices;")
        self.db.pg_conn.commit()
        self.db.mongo_collection.delete_many({})
        self.db.close()

    def test_store_price(self):
        # Test storing a price in PostgreSQL
        data = {
            "timestamp": datetime.now(),
            "price": 50000.0,
            "volume": 100.0,
            "source": "binance"
        }
        self.db.store_price(data, "BTCUSDT")

        # Verify the data was stored
        self.db.pg_cursor.execute("SELECT * FROM prices WHERE symbol = 'BTCUSDT';")
        result = self.db.pg_cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[2], "BTCUSDT")
        self.assertEqual(result[3], 50000.0)

    def test_store_raw_data(self):
        # Test storing raw data in MongoDB
        raw_data = {"test": "data"}
        self.db.store_raw_data(raw_data)

        # Verify the data was stored
        result = self.db.mongo_collection.find_one({"test": "data"})
        self.assertIsNotNone(result)
        self.assertEqual(result["test"], "data")

if __name__ == "__main__":
    unittest.main()