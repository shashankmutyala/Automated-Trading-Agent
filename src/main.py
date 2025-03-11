"""
---- Automated Trading Agent ----
"""

import logging
import time
import random
import yaml

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HelloTrading:
    """Simple trading simulation for demonstration purposes."""

    def __init__(self, config_file="E:\Automated-Trading\default.yaml"):
        # Load configurations
        self.config = self.load_config(config_file)

        self.assets = self.config["assets"]
        self.prices = self.config["prices"]
        self.volatility = self.config["volatility"]

    @staticmethod
    def load_config(file_path):
        """Load configuration from YAML file."""
        try:
            with open(file_path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {file_path}")
            raise

    def get_portfolio_value(self):
        """Calculating total portfolio value in USD."""
        value = self.assets["USD"]
        for asset, amount in self.assets.items():
            if asset != "USD" and asset in self.prices:
                value += amount * self.prices[asset]
        return value

    def simulate_market_movement(self):
        """Simulate simple market price changes."""
        btc_change = random.uniform(-self.volatility["BTC"], self.volatility["BTC"])
        eth_change = random.uniform(-self.volatility["ETH"], self.volatility["ETH"])

        self.prices["BTC"] *= (1 + btc_change)
        self.prices["ETH"] *= (1 + eth_change)

        logger.info(f"BTC change: {btc_change * 100:.2f}% | ETH change: {eth_change * 100:.2f}%")

    def run_simulation(self, iterations=10):
        """Running a simple trading simulation."""
        logger.info("Starting trading simulation...")
        logger.info(f"Initial portfolio: {self.assets}")
        logger.info(f"Initial value: ${self.get_portfolio_value():.2f}")

        for i in range(iterations):
            logger.info(f"--- Simulation step {i + 1} ---")
            self.simulate_market_movement()
            logger.info(f"BTC price: ${self.prices['BTC']:.2f}")
            logger.info(f"ETH price: ${self.prices['ETH']:.2f}")
            logger.info(f"Portfolio value: ${self.get_portfolio_value():.2f}")
            time.sleep(1)  # Simulate passing time

        logger.info("Simulation completed!")
        logger.info(f"Final portfolio value: ${self.get_portfolio_value():.2f}")


def main():
    """Main entry point of the application."""
    logger.info("Hello Trader! Starting Automated Trading Agent...")

    # Initialize and run simple trading simulation
    simulator = HelloTrading()
    simulator.run_simulation(iterations=5)

    logger.info("System shutdown.")


if __name__ == "__main__":
    main()
