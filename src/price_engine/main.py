import asyncio
import logging
import json
import os
import signal
from datetime import datetime, timedelta
import argparse
import sys
import os
# Add the src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now use absolute imports
from price_engine.db.mongodb_handler import MongoDBHandler
from price_engine.data_manager import PriceDataManager
from price_engine.processor import PriceProcessor
from price_engine.risk_management import RiskManager
from price_engine.order_executor import OrderExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")


class TradingEngine:
    """Main trading engine that coordinates all components."""

    def __init__(self, config=None):
        """
        Initialize the trading engine with configuration.

        Args:
            config (dict): Configuration parameters
        """
        # Load configuration
        self.config = self._load_config(config)

        # Initialize shutdown flag
        self.is_running = False
        self.shutdown_event = asyncio.Event()

        # Initialize components
        self.db_handler = None
        self.data_manager = None
        self.processor = None
        self.risk_manager = None
        self.order_executor = None

    def _load_config(self, config=None):
        """Load configuration from file or use provided config."""
        default_config = {
            "mongodb": {
                "uri": "mongodb://localhost:27017/",
                "db_name": "crypto_trading"
            },
            "trading": {
                "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                "test_mode": True,
                "api_key": "",
                "api_secret": ""
            },
            "risk": {
                "max_risk_per_trade": 0.02,
                "max_open_trades": 3
            }
        }

        if config:
            return {**default_config, **config}

        # Try to load from file
        try:
            config_path = os.path.join("config", "trading_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                logger.info("Loaded configuration from file")
                return {**default_config, **file_config}
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")

        return default_config

    async def initialize(self):
        """Initialize all components of the trading engine."""
        try:
            # Initialize MongoDB handler
            logger.info("Initializing MongoDB connection...")
            self.db_handler = MongoDBHandler(
                uri=self.config["mongodb"]["uri"],
                db_name=self.config["mongodb"]["db_name"]
            )

            # Initialize risk manager
            logger.info("Initializing risk manager...")
            self.risk_manager = RiskManager(config=self.config["risk"])

            # Initialize order executor
            logger.info("Initializing order executor...")
            self.order_executor = OrderExecutor(
                api_key=self.config["trading"]["api_key"],
                api_secret=self.config["trading"]["api_secret"],
                testnet=self.config["trading"]["test_mode"],
                db_handler=self.db_handler,
                risk_manager=self.risk_manager
            )

            # Initialize price processor
            logger.info("Initializing price processor...")
            self.processor = PriceProcessor(db_handler=self.db_handler)

            # Wire up signal callbacks
            self.processor.add_signal_callback(self.handle_trading_signal)

            # Initialize data manager
            logger.info("Initializing data manager...")
            self.data_manager = PriceDataManager(db_handler=self.db_handler)

            # Set up price callback from data manager to processor
            self.data_manager.add_price_callback(self.processor.process_price)

            logger.info("All components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing trading engine: {e}")
            return False

    async def start(self):
        """Start the trading engine."""
        if self.is_running:
            logger.warning("Trading engine is already running")
            return

        if not await self.initialize():
            logger.error("Failed to initialize trading engine. Cannot start.")
            return

        self.is_running = True
        logger.info("Starting trading engine...")

        # Convert symbol list to lowercase for WebSocket
        symbols = [s.lower() for s in self.config["trading"]["symbols"]]

        try:
            # Start live data collection
            logger.info(f"Starting live data collection for symbols: {symbols}")
            await self.data_manager.start_live_data(symbols)

            # Process some historical data to warm up indicators
            await self.process_historical_data()

            # Set up signal handlers for graceful shutdown
            for sig in (signal.SIGTERM, signal.SIGINT):
                asyncio.get_running_loop().add_signal_handler(
                    sig, lambda: asyncio.create_task(self.shutdown())
                )

            logger.info("Trading engine started successfully")

            # Keep the engine running until shutdown is requested
            await self.shutdown_event.wait()

        except Exception as e:
            logger.error(f"Error in trading engine: {e}")
            await self.shutdown()

    async def process_historical_data(self):
        """Process historical data to warm up indicators."""
        try:
            # Get data from the last 7 days for each symbol
            start_time = datetime.now() - timedelta(days=7)

            for symbol in self.config["trading"]["symbols"]:
                logger.info(f"Loading historical data for {symbol}...")

                # Get historical price data
                df = await self.data_manager.get_price_history(
                    symbol=symbol,
                    start_time=start_time,
                    use_api_if_needed=True
                )

                if df.empty:
                    logger.warning(f"No historical data available for {symbol}")
                    continue

                # Process historical data to warm up indicators
                signals = await self.processor.process_historical_data(symbol, df)

                logger.info(f"Processed {len(df)} historical data points for {symbol}")
                logger.info(f"Generated {len(signals)} signals from historical data")

                # In backtest mode, we could execute these signals
                # For live mode, we just log them
                for signal in signals:
                    logger.info(f"Historical signal: {signal['symbol']} - {signal['signal']}")

        except Exception as e:
            logger.error(f"Error processing historical data: {e}")

    async def handle_trading_signal(self, signal):
        """
        Handle a trading signal generated by the processor.

        Args:
            signal (dict): Trading signal
        """
        try:
            if signal["signal"] in ["BUY", "SELL"]:
                logger.info(f"Received {signal['signal']} signal for {signal['symbol']}")
                logger.info(f"Signal reason: {signal['reason']}")
                logger.info(f"Confidence: {signal['confidence']}")

                # Execute the order if not in paper trading mode
                if self.order_executor:
                    logger.info("Sending signal to order executor...")
                    order_result = await self.order_executor.process_signal(signal)

                    if order_result:
                        logger.info(f"Order executed: {order_result}")
                    else:
                        logger.info("Order was not placed")
        except Exception as e:
            logger.error(f"Error handling trading signal: {e}")

    async def shutdown(self):
        """Gracefully shut down the trading engine."""
        if not self.is_running:
            return

        logger.info("Shutting down trading engine...")
        self.is_running = False

        # Stop the WebSocket data stream
        if self.data_manager:
            logger.info("Stopping live data collection...")
            await self.data_manager.stop_live_data()

        # Close database connection
        if self.db_handler:
            logger.info("Closing database connection...")
            self.db_handler.close()

        # Set shutdown event
        self.shutdown_event.set()
        logger.info("Trading engine shutdown complete")


async def main():
    """Main entry point for the trading engine."""
    parser = argparse.ArgumentParser(description="Crypto Trading Engine")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()

    # Load config from file if specified
    config = None
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return

    # Override test mode if specified
    if args.test and config:
        config["trading"]["test_mode"] = True

    # Create and start the trading engine
    engine = TradingEngine(config)

    try:
        await engine.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected")
        await engine.shutdown()


if __name__ == "__main__":
    # Create config directory if it doesn't exist
    os.makedirs("config", exist_ok=True)

    # Run the main function
    asyncio.run(main())