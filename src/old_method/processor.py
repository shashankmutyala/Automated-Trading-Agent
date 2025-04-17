import numpy as np
import pandas as pd
import logging
import asyncio
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class PriceProcessor:
    """
    Processes price data to generate trading signals and perform technical analysis.
    """

    def __init__(self, db_handler=None):
        """
        Initialize the price processor.

        Args:
            db_handler: Optional database handler for storing signals
        """
        self.db_handler = db_handler
        self.price_history = defaultdict(list)  # Stores recent prices by symbol
        self.signal_callbacks = []  # Callbacks to notify when signals are generated

        # Default settings for technical indicators
        self.short_window = 20  # Short moving average window (e.g., 20 periods)
        self.medium_window = 50  # Medium moving average window
        self.long_window = 200  # Long moving average window
        self.rsi_window = 14  # RSI calculation window
        self.volatility_window = 20  # For calculating price volatility
        self.bollinger_window = 20  # Bollinger bands window
        self.bollinger_std = 2  # Standard deviations for Bollinger bands
        self.macd_fast = 12  # MACD fast EMA period
        self.macd_slow = 26  # MACD slow EMA period
        self.macd_signal = 9  # MACD signal line period

        # Maximum number of price points to keep in memory per symbol
        self.max_history_size = max(1000, self.long_window * 2)

    def add_signal_callback(self, callback):
        """Add a callback to be notified when trading signals are generated"""
        if callback not in self.signal_callbacks:
            self.signal_callbacks.append(callback)

    def remove_signal_callback(self, callback):
        """Remove a signal callback"""
        if callback in self.signal_callbacks:
            self.signal_callbacks.remove(callback)

    async def process_price(self, symbol, price, timestamp):
        """
        Process a new price point and generate signals.

        Args:
            symbol (str): Trading pair symbol
            price (float): Current price
            timestamp (int): Unix timestamp in milliseconds

        Returns:
            dict: Generated signal and metrics
        """
        # Add new price to history
        self.price_history[symbol].append({
            'price': float(price),
            'timestamp': int(timestamp)
        })

        # Trim history if needed
        if len(self.price_history[symbol]) > self.max_history_size:
            self.price_history[symbol] = self.price_history[symbol][-self.max_history_size:]

        # Only analyze when we have enough data
        if len(self.price_history[symbol]) < self.short_window:
            return {
                'symbol': symbol,
                'signal': 'INSUFFICIENT_DATA',
                'price': price,
                'timestamp': timestamp
            }

        # Calculate indicators
        prices = [p['price'] for p in self.price_history[symbol]]

        # Calculate moving averages
        ma_short = self.calculate_ma(prices, self.short_window)
        ma_medium = self.calculate_ma(prices, self.medium_window) if len(prices) >= self.medium_window else None
        ma_long = self.calculate_ma(prices, self.long_window) if len(prices) >= self.long_window else None

        # Calculate additional indicators
        rsi = self.calculate_rsi(prices) if len(prices) >= self.rsi_window + 1 else None
        volatility = self.calculate_volatility(prices) if len(prices) >= self.volatility_window else None

        # Calculate Bollinger Bands
        bollinger = None
        if len(prices) >= self.bollinger_window:
            ma = self.calculate_ma(prices, self.bollinger_window)
            std = np.std(prices[-self.bollinger_window:])
            upper = ma + self.bollinger_std * std
            lower = ma - self.bollinger_std * std
            bollinger = {
                'middle': ma,
                'upper': upper,
                'lower': lower,
                'width': (upper - lower) / ma  # Normalized width
            }

        # Calculate MACD
        macd = None
        if len(prices) >= self.macd_slow + self.macd_signal:
            macd_line, signal_line, histogram = self.calculate_macd(
                prices,
                self.macd_fast,
                self.macd_slow,
                self.macd_signal
            )
            macd = {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }

        # Create metrics dictionary
        metrics = {
            'price': price,
            'ma_short': ma_short,
            'ma_medium': ma_medium,
            'ma_long': ma_long,
            'rsi': rsi,
            'volatility': volatility
        }

        if bollinger:
            metrics['bollinger'] = bollinger

        if macd:
            metrics['macd'] = macd

        # Generate trading signal
        signal = self.generate_signal(symbol, price, metrics)

        # Store signal in database if it's actionable and handler is available
        if self.db_handler and signal['signal'] not in ['HOLD', 'INSUFFICIENT_DATA']:
            try:
                await asyncio.to_thread(
                    self.db_handler.store_signal,
                    symbol=symbol,
                    signal_type=signal['signal'],
                    metrics=metrics,
                    timestamp=timestamp
                )
                logger.info(f"Generated {signal['signal']} signal for {symbol} at price {price}")
            except Exception as e:
                logger.error(f"Failed to store signal: {e}")

        # Notify callbacks about the signal
        signal_data = {**signal, 'metrics': metrics}
        for callback in self.signal_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal_data)
                else:
                    callback(signal_data)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")

        return signal_data

    def calculate_ma(self, prices, window):
        """Calculate simple moving average."""
        if len(prices) < window:
            return None
        return np.mean(prices[-window:])

    def calculate_ema(self, prices, window):
        """Calculate exponential moving average."""
        if len(prices) < window:
            return None
        return pd.Series(prices).ewm(span=window, adjust=False).mean().iloc[-1]

    def calculate_rsi(self, prices):
        """Calculate Relative Strength Index."""
        if len(prices) <= self.rsi_window:
            return None

        # Get price changes
        deltas = np.diff(prices)

        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Calculate average gains and losses
        avg_gain = np.mean(gains[-self.rsi_window:])
        avg_loss = np.mean(losses[-self.rsi_window:])

        # Calculate RS and RSI
        if avg_loss == 0:
            return 100  # If no losses, RSI is 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_volatility(self, prices):
        """Calculate price volatility (standard deviation of returns)."""
        if len(prices) < self.volatility_window:
            return None

        # Calculate returns
        returns = np.diff(prices[-self.volatility_window - 1:]) / prices[-self.volatility_window - 1:-1]
        return np.std(returns) * 100  # Convert to percentage

    def calculate_macd(self, prices, fast_period, slow_period, signal_period):
        """Calculate MACD (Moving Average Convergence Divergence)."""
        # Convert to pandas Series for easy calculation
        price_series = pd.Series(prices)

        # Calculate EMAs
        fast_ema = price_series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = price_series.ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # Calculate signal line (EMA of MACD line)
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Calculate histogram (MACD line - signal line)
        histogram = macd_line - signal_line

        # Return the latest values
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

    def generate_signal(self, symbol, price, metrics):
        """
        Generate trading signal based on technical indicators.

        Args:
            symbol (str): Trading pair symbol
            price (float): Current price
            metrics (dict): Technical indicator values

        Returns:
            dict: Signal dictionary with signal type and reason
        """
        signal = "HOLD"
        reason = []
        confidence = 0

        # 1. Moving Average Crossovers
        ma_short = metrics.get('ma_short')
        ma_medium = metrics.get('ma_medium')
        ma_long = metrics.get('ma_long')

        if ma_short and ma_medium:
            # Short-term MA crosses above medium-term MA (Golden Cross)
            if (ma_short > ma_medium and
                    self.previous_cross_below(symbol, 'ma_short', 'ma_medium', lookback=3)):
                signal = "BUY"
                reason.append("Golden Cross (short MA crossed above medium MA)")
                confidence += 0.3

            # Short-term MA crosses below medium-term MA (Death Cross)
            elif (ma_short < ma_medium and
                  self.previous_cross_above(symbol, 'ma_short', 'ma_medium', lookback=3)):
                signal = "SELL"
                reason.append("Death Cross (short MA crossed below medium MA)")
                confidence += 0.3

        # 2. RSI Signals
        rsi = metrics.get('rsi')
        if rsi is not None:
            if rsi < 30:  # Oversold
                if signal != "SELL":  # Don't contradict a strong sell signal
                    signal = "BUY"
                    reason.append(f"RSI oversold ({rsi:.1f})")
                    confidence += 0.2
            elif rsi > 70:  # Overbought
                if signal != "BUY":  # Don't contradict a strong buy signal
                    signal = "SELL"
                    reason.append(f"RSI overbought ({rsi:.1f})")
                    confidence += 0.2

        # 3. Bollinger Bands
        bollinger = metrics.get('bollinger')
        if bollinger:
            # Price breaks below lower band (potential buy)
            if price < bollinger['lower']:
                if signal != "SELL":  # Don't contradict a strong sell signal
                    signal = "BUY"
                    reason.append("Price below lower Bollinger Band")
                    confidence += 0.2

            # Price breaks above upper band (potential sell)
            elif price > bollinger['upper']:
                if signal != "BUY":  # Don't contradict a strong buy signal
                    signal = "SELL"
                    reason.append("Price above upper Bollinger Band")
                    confidence += 0.2

        # 4. MACD Signals
        macd_data = metrics.get('macd')
        if macd_data:
            # MACD line crosses above signal line (bullish)
            if (macd_data['macd'] > macd_data['signal'] and
                    macd_data['histogram'] > 0 and
                    self.previous_histogram_negative(symbol, lookback=2)):
                if signal != "SELL":  # Don't contradict a strong sell signal
                    signal = "BUY"
                    reason.append("MACD crossed above signal line")
                    confidence += 0.25

            # MACD line crosses below signal line (bearish)
            elif (macd_data['macd'] < macd_data['signal'] and
                  macd_data['histogram'] < 0 and
                  self.previous_histogram_positive(symbol, lookback=2)):
                if signal != "BUY":  # Don't contradict a strong buy signal
                    signal = "SELL"
                    reason.append("MACD crossed below signal line")
                    confidence += 0.25

        # Store the updated metrics for this symbol (for future reference)
        self.store_metrics_history(symbol, metrics)

        # Create the signal response
        return {
            'symbol': symbol,
            'signal': signal,
            'reason': reason,
            'confidence': confidence,
            'price': price,
            'timestamp': int(datetime.now().timestamp() * 1000)
        }

    def store_metrics_history(self, symbol, metrics):
        """Store metrics history for trend analysis."""
        # Initialize if needed
        if not hasattr(self, 'metrics_history'):
            self.metrics_history = defaultdict(list)

        # Add current metrics
        self.metrics_history[symbol].append(metrics)

        # Keep the history size reasonable
        max_metrics_history = 50
        if len(self.metrics_history[symbol]) > max_metrics_history:
            self.metrics_history[symbol] = self.metrics_history[symbol][-max_metrics_history:]

    def previous_cross_above(self, symbol, indicator1, indicator2, lookback=3):
        """Check if indicator1 crossed above indicator2 in recent history."""
        if not hasattr(self, 'metrics_history') or symbol not in self.metrics_history:
            return False

        history = self.metrics_history[symbol]
        if len(history) <= lookback:
            return False

        # Check if the cross occurred in recent lookback periods
        for i in range(min(lookback, len(history) - 1)):
            current = history[-(i + 1)]
            previous = history[-(i + 2)]

            # Skip if metrics are missing
            if (indicator1 not in current or indicator2 not in current or
                    indicator1 not in previous or indicator2 not in previous or
                    current[indicator1] is None or current[indicator2] is None or
                    previous[indicator1] is None or previous[indicator2] is None):
                continue

            # Check for cross above
            if (current[indicator1] > current[indicator2] and
                    previous[indicator1] <= previous[indicator2]):
                return True

        return False

    def previous_cross_below(self, symbol, indicator1, indicator2, lookback=3):
        """Check if indicator1 crossed below indicator2 in recent history."""
        if not hasattr(self, 'metrics_history') or symbol not in self.metrics_history:
            return False

        history = self.metrics_history[symbol]
        if len(history) <= lookback:
            return False

        # Check if the cross occurred in recent lookback periods
        for i in range(min(lookback, len(history) - 1)):
            current = history[-(i + 1)]
            previous = history[-(i + 2)]

            # Skip if metrics are missing
            if (indicator1 not in current or indicator2 not in current or
                    indicator1 not in previous or indicator2 not in previous or
                    current[indicator1] is None or current[indicator2] is None or
                    previous[indicator1] is None or previous[indicator2] is None):
                continue

            # Check for cross below
            if (current[indicator1] < current[indicator2] and
                    previous[indicator1] >= previous[indicator2]):
                return True

        return False

    def previous_histogram_negative(self, symbol, lookback=2):
        """Check if MACD histogram was negative in previous periods."""
        if not hasattr(self, 'metrics_history') or symbol not in self.metrics_history:
            return False

        history = self.metrics_history[symbol]
        if len(history) <= lookback:
            return False

        # Check previous periods
        for i in range(1, min(lookback + 1, len(history))):
            if (history[-i].get('macd') and
                    history[-i]['macd'].get('histogram') is not None and
                    history[-i]['macd']['histogram'] < 0):
                return True

        return False

    def previous_histogram_positive(self, symbol, lookback=2):
        """Check if MACD histogram was positive in previous periods."""
        if not hasattr(self, 'metrics_history') or symbol not in self.metrics_history:
            return False

        history = self.metrics_history[symbol]
        if len(history) <= lookback:
            return False

        # Check previous periods
        for i in range(1, min(lookback + 1, len(history))):
            if (history[-i].get('macd') and
                    history[-i]['macd'].get('histogram') is not None and
                    history[-i]['macd']['histogram'] > 0):
                return True

        return False

    async def process_historical_data(self, symbol, df):
        """
        Process historical price data to generate signals.

        Args:
            symbol (str): Trading pair symbol
            df (pandas.DataFrame): DataFrame with price data

        Returns:
            list: Generated signals
        """
        signals = []

        # Reset history for this symbol
        self.price_history[symbol] = []

        # Process each price point
        for _, row in df.iterrows():
            signal = await self.process_price(
                symbol=symbol,
                price=row['price'] if 'price' in row else row['close'],
                timestamp=int(row['timestamp'])
            )

            # Only add actionable signals
            if signal['signal'] not in ['HOLD', 'INSUFFICIENT_DATA']:
                signals.append(signal)

        return signals


