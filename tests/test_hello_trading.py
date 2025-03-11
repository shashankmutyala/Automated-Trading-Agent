"""Tests for the HelloTrading simulation."""
import unittest
import os
import yaml
import tempfile
from src.main import HelloTrading

class TestHelloTrading(unittest.TestCase):
    """Test case for the HelloTrading class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "assets": {"BTC": 0.5, "ETH": 5.0, "USD": 10000.0},
            "prices": {"BTC": 40000.0, "ETH": 2500.0},
            "volatility": {"BTC": 0.02, "ETH": 0.03}
        }

        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "test_config.yaml")

        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)

        self.simulator = HelloTrading(config_file=self.config_path)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_init(self):
        """Test initialization of HelloTrading."""
        self.assertIn("BTC", self.simulator.assets)
        self.assertIn("ETH", self.simulator.assets)
        self.assertIn("USD", self.simulator.assets)

    def test_portfolio_value(self):
        """Test portfolio value calculation."""
        expected = (
            self.simulator.assets["USD"] +
            self.simulator.assets["BTC"] * self.simulator.prices["BTC"] +
            self.simulator.assets["ETH"] * self.simulator.prices["ETH"]
        )
        actual = self.simulator.get_portfolio_value()
        self.assertEqual(expected, actual)

    def test_market_movement(self):
        """Test that market movement changes prices."""
        original_btc = self.simulator.prices["BTC"]
        original_eth = self.simulator.prices["ETH"]

        self.simulator.simulate_market_movement()

        self.assertNotEqual(original_btc, self.simulator.prices["BTC"])
        self.assertNotEqual(original_eth, self.simulator.prices["ETH"])

    def test_price_change_within_volatility(self):
        """Test that price changes stay within volatility limits."""
        original_btc = self.simulator.prices["BTC"]
        original_eth = self.simulator.prices["ETH"]

        self.simulator.simulate_market_movement()

        btc_pct_change = abs((self.simulator.prices["BTC"] - original_btc) / original_btc)
        eth_pct_change = abs((self.simulator.prices["ETH"] - original_eth) / original_eth)

        self.assertLessEqual(btc_pct_change, self.simulator.volatility["BTC"])
        self.assertLessEqual(eth_pct_change, self.simulator.volatility["ETH"])


if __name__ == "__main__":
    unittest.main()
