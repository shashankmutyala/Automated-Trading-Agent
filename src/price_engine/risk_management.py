import logging
import asyncio
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
import json

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk management for cryptocurrency trading.
    Handles position sizing, stops, and risk limits.
    """

    def __init__(self, config=None):
        """
        Initialize risk manager with configuration parameters.

        Args:
            config (dict): Risk management configuration
        """
        # Default configuration
        default_config = {
            'max_risk_per_trade': 0.02,  # 2% of account per trade
            'max_open_trades': 3,  # Maximum concurrent open trades
            'default_stop_loss': 0.03,  # 3% stop loss
            'default_take_profit': 0.06,  # 6% take profit
            'max_risk_per_symbol': 0.10,  # Max 10% of account in one symbol
            'max_daily_drawdown': 0.05  # Max 5% daily drawdown
        }

        # Load config from file if not provided
        if config is None:
            try:
                with open("config/risk_config.json", "r") as f:
                    config = json.load(f)
                logger.info("Loaded risk configuration from file")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load risk config: {e}. Using defaults.")
                config = {}

        # Merge with defaults
        self.config = {**default_config, **config}

        # Track current positions and risk
        self.open_positions = {}
        self.daily_pl = 0
        self.starting_balance = 0
        self.current_balance = 0

    def set_balance(self, balance):
        """Set the current account balance."""
        self.current_balance = Decimal(str(balance))
        if self.starting_balance == 0:
            self.starting_balance = self.current_balance

    def calculate_position_size(self, symbol, price, signal_strength=0.5, stop_loss_pct=None):
        """
        Calculate appropriate position size for a trade.

        Args:
            symbol (str): Trading pair symbol
            price (float): Current price
            signal_strength (float): Signal confidence from 0.0-1.0
            stop_loss_pct (float): Stop loss percentage, or None for default

        Returns:
            dict: Position details including size and risk metrics
        """
        if self.current_balance <= 0:
            logger.error("Cannot calculate position size: balance not set or zero")
            return {
                "size": 0,
                "value": 0,
                "error": "No balance available"
            }

        # Use default stop loss if not provided
        stop_loss_pct = stop_loss_pct or self.config['default_stop_loss']

        # Calculate maximum risk amount for this trade
        max_risk_amount = Decimal(str(self.current_balance)) * Decimal(str(self.config['max_risk_per_trade']))

        # Adjust based on signal strength
        adjusted_risk = max_risk_amount * Decimal(str(signal_strength))

        # Determine base currency amount at risk using stop loss
        price_at_risk = Decimal(str(price)) * Decimal(str(stop_loss_pct))

        # Calculate position size
        position_size = adjusted_risk / price_at_risk

        # Calculate position value
        position_value = position_size * Decimal(str(price))

        # Check if the position would exceed max per-symbol risk
        symbol_max_value = Decimal(str(self.current_balance)) * Decimal(str(self.config['max_risk_per_symbol']))

        # Get existing position for this symbol if any
        existing_position = self.open_positions.get(symbol, {"size": 0, "value": 0})
        total_symbol_exposure = existing_position["value"] + position_value

        if total_symbol_exposure > symbol_max_value:
            logger.warning(f"Position for {symbol} would exceed max risk per symbol")
            # Scale position down to meet limit
            scale_factor = (symbol_max_value - existing_position["value"]) / position_value
            position_size = position_size * scale_factor
            position_value = position_value * scale_factor

        # Check if we would exceed max open trades
        if (len(self.open_positions) >= self.config['max_open_trades'] and
                symbol not in self.open_positions):
            logger.warning("Maximum number of open trades reached")
            return {
                "size": 0,
                "value": 0,
                "error": "Maximum number of open trades reached"
            }

        # Round to 8 decimal places (common precision for crypto)
        position_size = position_size.quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

        # Calculate stop loss and take profit prices
        stop_price = Decimal(str(price)) * (1 - Decimal(str(stop_loss_pct)))
        take_profit_price = Decimal(str(price)) * (1 + Decimal(str(self.config['default_take_profit'])))

        return {
            "symbol": symbol,
            "size": float(position_size),
            "value": float(position_value),
            "stop_price": float(stop_price),
            "take_profit": float(take_profit_price),
            "risk_amount": float(adjusted_risk),
            "risk_pct": float(self.config['max_risk_per_trade'] * signal_strength)
        }

    def register_trade(self, symbol, size, entry_price, trade_type="BUY"):
        """Register a new open position."""
        value = Decimal(str(size)) * Decimal(str(entry_price))

        self.open_positions[symbol] = {
            "size": Decimal(str(size)),
            "entry_price": Decimal(str(entry_price)),
            "value": value,
            "type": trade_type,
            "timestamp": datetime.now()
        }

        logger.info(f"Registered {trade_type} position for {symbol}: {size} at {entry_price}")
        return True

    def close_trade(self, symbol, exit_price):
        """Close an existing position and calculate P&L."""
        if symbol not in self.open_positions:
            logger.warning(f"Attempted to close non-existing position for {symbol}")
            return None

        position = self.open_positions[symbol]
        exit_value = position["size"] * Decimal(str(exit_price))

        # Calculate P&L
        if position["type"] == "BUY":
            pl = exit_value - position["value"]
        else:  # SELL (short)
            pl = position["value"] - exit_value

        # Update daily P&L
        self.daily_pl += float(pl)

        # Check if daily drawdown limit reached
        daily_drawdown = self.daily_pl / float(self.starting_balance)
        if daily_drawdown < -self.config['max_daily_drawdown']:
            logger.warning(f"Daily drawdown limit reached: {daily_drawdown:.2%}")

        # Remove the position
        del self.open_positions[symbol]

        logger.info(f"Closed position for {symbol} at {exit_price}. P&L: {float(pl)}")
        return {
            "symbol": symbol,
            "pl_value": float(pl),
            "pl_percentage": float(pl / position["value"]) if position["value"] > 0 else 0,
            "position": position
        }

    def should_trade(self, symbol, signal):
        """
        Determine whether a trade should be executed based on risk parameters.

        Args:
            symbol (str): Trading pair symbol
            signal (dict): Signal data including confidence

        Returns:
            bool: True if trade should be executed, False otherwise
        """
        # Check if we're already at max open trades
        if (len(self.open_positions) >= self.config['max_open_trades'] and
                symbol not in self.open_positions):
            return False

        # Check signal confidence threshold
        if signal.get('confidence', 0) < 0.3:
            logger.info(f"Signal confidence for {symbol} too low: {signal.get('confidence')}")
            return False

        # Don't trade if we already have a position for this symbol in the opposite direction
        if symbol in self.open_positions:
            position_type = self.open_positions[symbol]["type"]
            if (position_type == "BUY" and signal.get('signal') == "SELL") or \
                    (position_type == "SELL" and signal.get('signal') == "BUY"):
                # We should close the position instead of opening a new one
                return True

        return True

    def reset_daily_stats(self):
        """Reset daily performance statistics."""
        self.daily_pl = 0
        self.starting_balance = self.current_balance

    def get_active_positions(self):
        """Get all active positions."""
        return self.open_positions

    def get_risk_exposure(self):
        """Calculate current risk exposure."""
        total_value = sum(pos["value"] for pos in self.open_positions.values())
        return {
            "total_exposure": float(total_value),
            "exposure_pct": float(total_value / self.current_balance) if self.current_balance > 0 else 0,
            "positions_count": len(self.open_positions),
            "daily_pl": self.daily_pl,
            "daily_pl_pct": self.daily_pl / float(self.starting_balance) if self.starting_balance > 0 else 0
        }

