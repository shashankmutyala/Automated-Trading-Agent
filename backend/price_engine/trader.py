from datetime import datetime, timedelta
import logging
from src.database import Database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trader:
    def __init__(self):
        self.db = Database()
        self.position = None  # Current position: "long", "short", or None
        self.entry_price = None  # Price at which the position was entered

    def get_historical_data(self, symbol, start_time, end_time):
        """
        Fetch historical data from the database (PostgreSQL).
        :param symbol: e.g., "BTCUSDT"
        :param start_time: Start datetime
        :param end_time: End datetime
        :return: List of (timestamp, price, volume) tuples
        """
        query = """
        SELECT timestamp, price, volume
        FROM prices
        WHERE symbol = %s AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp;
        """
        self.db.pg_cursor.execute(query, (symbol, start_time, end_time))
        return self.db.pg_cursor.fetchall()

    def get_latest_price(self, symbol):
        """
        Fetch the most recent price for a symbol from the database (PostgreSQL).
        :param symbol: e.g., "BTCUSDT"
        :return: Latest price or None if no data is available
        """
        query = """
        SELECT price
        FROM prices
        WHERE symbol = %s
        ORDER BY timestamp DESC
        LIMIT 1;
        """
        self.db.pg_cursor.execute(query, (symbol,))
        result = self.db.pg_cursor.fetchone()
        return result[0] if result else None

    def calculate_momentum(self, data, window=5):
        """
        Calculate momentum based on the last `window` data points.
        :param data: List of (timestamp, price, volume) tuples
        :param window: Number of data points to consider
        :return: Momentum value (positive for upward trend, negative for downward)
        """
        if len(data) < window:
            return 0
        prices = [row[1] for row in data[-window:]]
        momentum = prices[-1] - prices[0]
        return momentum

    def log_decision(self, symbol, decision, price, momentum):
        """
        Log the trading decision to MongoDB for auditing.
        :param symbol: Symbol (e.g., "BTCUSDT")
        :param decision: Trading decision ("buy", "sell", "hold")
        :param price: Current price
        :param momentum: Calculated momentum
        """
        log_entry = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "decision": decision,
            "price": price,
            "momentum": momentum,
            "position": self.position,
            "entry_price": self.entry_price
        }
        self.db.store_raw_data(log_entry)

    def make_decision(self, symbol, current_time):
        """
        Make a trading decision based on historical and real-time data.
        :param symbol: e.g., "BTCUSDT"
        :param current_time: Current timestamp for the simulation
        :return: Decision ("buy", "sell", "hold") and current price
        """
        # Fetch historical data for the last 1 hour to calculate momentum
        start_time = current_time - timedelta(hours=1)
        historical_data = self.get_historical_data(symbol, start_time, current_time)

        if not historical_data:
            logger.warning(f"No historical data available for {symbol} at {current_time}")
            return "hold", None

        # Calculate momentum
        momentum = self.calculate_momentum(historical_data)
        current_price = historical_data[-1][1]  # Latest price from historical data

        # Simple momentum-based strategy
        decision = "hold"
        if momentum > 0 and self.position != "long":
            # Price is increasing: buy if not already in a long position
            decision = "buy"
            self.position = "long"
            self.entry_price = current_price
            logger.info(f"Buy {symbol} at {current_price} (Momentum: {momentum})")
        elif momentum < 0 and self.position == "long":
            # Price is decreasing: sell if in a long position
            decision = "sell"
            profit = current_price - self.entry_price
            logger.info(f"Sell {symbol} at {current_price} (Profit: {profit})")
            self.position = None
            self.entry_price = None
        else:
            logger.info(f"Hold {symbol} at {current_price} (Momentum: {momentum})")

        # Log the decision to MongoDB
        self.log_decision(symbol, decision, current_price, momentum)

        return decision, current_price

    def close(self):
        """Close the database connection."""
        self.db.close()