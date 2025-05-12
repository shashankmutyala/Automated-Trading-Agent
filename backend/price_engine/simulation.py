# backend/simulation.py
import time
from datetime import datetime, timedelta
import logging
from .api_client import BinanceAPIClient
from .websocket_client import BinanceWebSocketClient
from .database import Database
from .trader import Trader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Simulation:
    def __init__(self, symbol="BTCUSDT", simulation_duration_minutes=10):
        self.symbol = symbol
        self.simulation_duration = timedelta(minutes=simulation_duration_minutes)
        self.api_client = BinanceAPIClient()
        self.db = Database()
        self.trader = Trader()
        self.received_data = []

    def store_data(self, data):
        """Callback to store real-time data in both databases."""
        # Store normalized price data in PostgreSQL
        normalized_data = {
            "timestamp": data["timestamp"],
            "price": data["price"],
            "volume": 0,  # WebSocket data doesn't include volume in this stream
            "source": "binance"
        }
        self.db.store_price(normalized_data, self.symbol)

        # Store raw WebSocket message in MongoDB for auditing
        raw_data = {
            "timestamp": str(data["timestamp"]),
            "price": data["price"],
            "source": "binance_websocket"
        }
        self.db.store_raw_data(raw_data)

        self.received_data.append(data)

    def fetch_initial_data(self):
        """Fetch historical data for the last 7 days to populate the database."""
        logger.info("Fetching initial historical data...")
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
        historical_data = self.api_client.fetch_historical_data(self.symbol, "1h", start_time, end_time)
        for data in historical_data:
            # Store normalized data in PostgreSQL
            self.db.store_price(data, self.symbol)
            # Store raw data in MongoDB
            raw_data = {
                "timestamp": str(data["timestamp"]),
                "open": data["open"],
                "high": data["high"],
                "low": data["low"],
                "close": data["close"],
                "volume": data["volume"],
                "source": "binance_api"
            }
            self.db.store_raw_data(raw_data)
        logger.info(f"Fetched {len(historical_data)} historical data points")

    def run(self):
        """Run the simulation."""
        # Step 1: Fetch initial historical data
        self.fetch_initial_data()

        # Step 2: Start real-time data streaming
        logger.info("Starting real-time data streaming...")
        ws_client = BinanceWebSocketClient(symbol=self.symbol.lower(), on_message_callback=self.store_data)
        ws_client.start()

        # Step 3: Run the simulation for the specified duration
        start_time = datetime.now()
        end_time = start_time + self.simulation_duration
        current_time = start_time

        logger.info(f"Starting simulation from {start_time} to {end_time}")
        while current_time < end_time:
            # Make a trading decision every minute
            decision, price = self.trader.make_decision(self.symbol, current_time)
            if price is not None:
                logger.info(f"Simulation time: {current_time}, Price: {price}, Decision: {decision}")
            else:
                logger.warning(f"No price data available at {current_time}")

            # Simulate time passing (1 minute per iteration)
            time.sleep(60)  # Sleep for 60 seconds to simulate 1-minute intervals
            current_time += timedelta(minutes=1)

            # Simulate a WebSocket disconnection (as per test criterion)
            if (current_time - start_time).seconds == 120:  # At 2 minutes
                logger.info("Simulating WebSocket disconnection for 10 seconds...")
                ws_client.stop()
                time.sleep(10)
                logger.info("Reconnecting WebSocket...")
                ws_client.start()
                ws_client.backfill_missed_data(self.api_client)

        # Step 4: Clean up
        logger.info("Simulation complete. Stopping WebSocket and closing database...")
        ws_client.stop()
        self.trader.close()
        self.db.close()

if __name__ == "__main__":
    simulation = Simulation(symbol="BTCUSDT", simulation_duration_minutes=10)
    simulation.run()