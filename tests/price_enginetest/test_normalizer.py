# tests/test_normalizer.py
import unittest
from datetime import datetime
from src.data_normalizer import DataNormalizer

class TestDataNormalizer(unittest.TestCase):
    def test_normalize_binance_data(self):
        # Test normalizing a sample Binance data point
        raw_data = {
            "timestamp": datetime.now(),
            "close": 50000.0,
            "volume": 100.0
        }
        normalized = DataNormalizer.normalize_binance_data(raw_data)
        
        self.assertEqual(normalized["timestamp"], raw_data["timestamp"])
        self.assertEqual(normalized["price"], 50000.0)
        self.assertEqual(normalized["volume"], 100.0)
        self.assertEqual(normalized["source"], "binance")

    def test_handle_missing_data(self):
        # Test handling missing data
        data_list = [
            {"timestamp": datetime.now(), "price": None, "volume": 100},
            {"timestamp": datetime.now(), "price": 50000, "volume": 200}
        ]
        reconciled = DataNormalizer.handle_missing_data(data_list)
        
        self.assertEqual(len(reconciled), 2)
        self.assertIsNone(reconciled[0]["price"])  # Missing price remains None (fallback not implemented)
        self.assertEqual(reconciled[1]["price"], 50000)

if __name__ == "__main__":
    unittest.main()