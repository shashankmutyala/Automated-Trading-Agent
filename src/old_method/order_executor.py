import logging
import asyncio
import time
import uuid
import hmac
import hashlib
import requests
from urllib.parse import urlencode
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Executes trading orders based on signals and risk parameters."""

    def __init__(self, api_key=None, api_secret=None, testnet=True, db_handler=None, risk_manager=None):
        """Initialize the order executor."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.db_handler = db_handler
        self.risk_manager = risk_manager

        # Base URLs for API calls
        self.base_url = "https://testnet.binance.vision/api" if testnet else "https://api.binance.com/api"
        logger.info(f"Using Binance {'TESTNET' if testnet else 'PRODUCTION'} environment")

        self.order_callbacks = []

    def add_order_callback(self, callback):
        """Add a callback for order notifications."""
        if callback not in self.order_callbacks:
            self.order_callbacks.append(callback)

    def remove_order_callback(self, callback):
        """Remove an order callback."""
        if callback in self.order_callbacks:
            self.order_callbacks.remove(callback)

    async def process_signal(self, signal):
        """Process a trading signal and execute an order if appropriate."""
        try:
            # Extract signal information
            symbol = signal['symbol']
            signal_type = signal['signal']
            price = signal['price']
            confidence = signal.get('confidence', 0.5)

            # Skip non-actionable signals
            if signal_type in ['HOLD', 'INSUFFICIENT_DATA']:
                return None

            # Get account balance
            account_info = await self.get_account_info()
            if not account_info:
                logger.error("Failed to get account information")
                return None

            # Update risk manager with current balance
            if self.risk_manager and account_info.get('balance'):
                self.risk_manager.set_balance(account_info.get('balance'))

            # Check if we should trade according to risk rules
            if self.risk_manager and not self.risk_manager.should_trade(symbol, signal):
                logger.info(f"Risk management prevented {signal_type} for {symbol}")
                return None

            side = "BUY" if signal_type == "BUY" else "SELL"

            # Calculate position size based on risk
            if self.risk_manager:
                position_details = self.risk_manager.calculate_position_size(
                    symbol=symbol,
                    price=price,
                    signal_strength=confidence
                )

                quantity = position_details.get('size', 0)
                stop_price = position_details.get('stop_price')
                take_profit = position_details.get('take_profit')

                if quantity <= 0:
                    logger.info(f"Position size calculation resulted in zero quantity")
                    return None
            else:
                # Conservative default with no risk manager (0.5% of balance)
                usdt_amount = float(account_info.get('balance', 0)) * 0.005
                quantity = usdt_amount / price
                stop_price = None
                take_profit = None

            # Place the order
            order_result = await self.place_order(symbol, side, quantity, price)
            if not order_result:
                return None

            logger.info(f"Placed {side} order for {symbol}: {quantity} @ {price}")

            # Register with risk manager if order filled
            if self.risk_manager and order_result.get('status') == 'FILLED':
                self.risk_manager.register_trade(
                    symbol=symbol,
                    size=quantity,
                    entry_price=price,
                    trade_type=side
                )

                # Add stop loss and take profit if available
                if stop_price:
                    await self.place_stop_loss(symbol, "SELL" if side == "BUY" else "BUY",
                                               quantity, stop_price)

                if take_profit:
                    await self.place_take_profit(symbol, "SELL" if side == "BUY" else "BUY",
                                                 quantity, take_profit)

            # Store order in database
            if self.db_handler:
                await asyncio.to_thread(
                    self.db_handler.store_order,
                    symbol=symbol,
                    order_type=side,
                    quantity=quantity,
                    price=price,
                    status=order_result.get('status', 'UNKNOWN'),
                    order_id=order_result.get('orderId', str(uuid.uuid4())),
                    timestamp=int(datetime.now().timestamp() * 1000)
                )

            # Notify callbacks
            await self._notify_callbacks({
                "type": "ORDER_PLACED",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_result": order_result
            })

            return order_result

        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return None

    async def place_order(self, symbol, side, quantity, price=None, order_type="MARKET"):
        """Place an order on the exchange."""
        try:
            # For simulated trading without API keys
            if not self.api_key or not self.api_secret:
                return self._simulate_order_result(symbol, side, quantity, price, order_type)

            # Prepare order parameters
            endpoint = "/v3/order"
            params = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": quantity,
                "timestamp": int(time.time() * 1000)
            }

            if order_type == "LIMIT" and price:
                params["price"] = price
                params["timeInForce"] = "GTC"

            # Sign and send request
            query_string = self._generate_signature(params)
            headers = {"X-MBX-APIKEY": self.api_key}
            url = f"{self.base_url}{endpoint}?{query_string}"

            response = await asyncio.to_thread(requests.post, url, headers=headers)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Order API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    async def place_stop_loss(self, symbol, side, quantity, stop_price):
        """Place a stop loss order."""
        try:
            if not self.api_key or not self.api_secret:
                result = self._simulate_order_result(
                    symbol, side, quantity, stop_price, "STOP_LOSS_LIMIT"
                )
                logger.info(f"Placed simulated stop loss for {symbol} at {stop_price}")
                return result

            params = {
                "symbol": symbol,
                "side": side,
                "type": "STOP_LOSS_LIMIT",
                "quantity": quantity,
                "stopPrice": stop_price,
                "price": stop_price,  # Limit price same as stop
                "timeInForce": "GTC",
                "timestamp": int(time.time() * 1000)
            }

            query_string = self._generate_signature(params)
            headers = {"X-MBX-APIKEY": self.api_key}
            url = f"{self.base_url}/v3/order?{query_string}"

            response = await asyncio.to_thread(requests.post, url, headers=headers)

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Placed stop loss for {symbol} at {stop_price}")
                return result
            else:
                logger.error(f"Stop loss API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error placing stop loss: {e}")
            return None

    async def place_take_profit(self, symbol, side, quantity, price):
        """Place a take profit (limit) order."""
        # Similar implementation to place_stop_loss but with LIMIT order type
        try:
            if not self.api_key or not self.api_secret:
                result = self._simulate_order_result(
                    symbol, side, quantity, price, "LIMIT"
                )
                logger.info(f"Placed simulated take profit for {symbol} at {price}")
                return result

            params = {
                "symbol": symbol,
                "side": side,
                "type": "LIMIT",
                "quantity": quantity,
                "price": price,
                "timeInForce": "GTC",
                "timestamp": int(time.time() * 1000)
            }

            query_string = self._generate_signature(params)
            headers = {"X-MBX-APIKEY": self.api_key}
            url = f"{self.base_url}/v3/order?{query_string}"

            response = await asyncio.to_thread(requests.post, url, headers=headers)

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Placed take profit for {symbol} at {price}")
                return result
            else:
                logger.error(f"Take profit API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error placing take profit: {e}")
            return None

    async def close_position(self, symbol):
        """Close a position by creating an opposing order."""
        try:
            # Get position from risk manager
            if not self.risk_manager or symbol not in self.risk_manager.get_active_positions():
                logger.warning(f"No active position found for {symbol}")
                return None

            position = self.risk_manager.get_active_positions()[symbol]
            position_size = float(position.get("size", 0))
            position_type = position.get("type")

            # Get current price
            ticker = await self.get_ticker(symbol)
            if not ticker:
                logger.error(f"Failed to get current price for {symbol}")
                return None

            current_price = float(ticker.get("price", 0))

            # Place closing order
            close_side = "SELL" if position_type == "BUY" else "BUY"
            close_order = await self.place_order(
                symbol=symbol,
                side=close_side,
                quantity=position_size,
                price=current_price
            )

            if not close_order:
                return None

            # Update risk manager
            close_result = self.risk_manager.close_trade(symbol, current_price)
            if close_result:
                logger.info(f"Closed position for {symbol} with P&L: {close_result.get('pl_value')}")

                # Notify callbacks about position close
                await self._notify_callbacks({
                    "type": "POSITION_CLOSED",
                    "symbol": symbol,
                    "side": close_side,
                    "quantity": position_size,
                    "price": current_price,
                    "pl": close_result.get('pl_value'),
                    "pl_percentage": close_result.get('pl_percentage')
                })

            return close_order

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None

    async def get_account_info(self):
        """Get account information from exchange."""
        try:
            if not self.api_key or not self.api_secret:
                return {"balance": 10000.0}  # Simulate $10,000 USDT balance

            endpoint = "/v3/account"
            params = {"timestamp": int(time.time() * 1000)}

            query_string = self._generate_signature(params)
            headers = {"X-MBX-APIKEY": self.api_key}
            url = f"{self.base_url}{endpoint}?{query_string}"

            response = await asyncio.to_thread(requests.get, url, headers=headers)

            if response.status_code == 200:
                account_data = response.json()
                usdt_balance = next((float(bal.get("free", 0)) for bal in account_data.get("balances", [])
                                     if bal.get("asset") == "USDT"), 0)
                return {"balance": usdt_balance}
            else:
                logger.error(f"Account API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    async def get_ticker(self, symbol):
        """Get current price ticker for a symbol."""
        try:
            url = f"{self.base_url}/v3/ticker/price?symbol={symbol}"

            response = await asyncio.to_thread(requests.get, url)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Ticker API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting ticker: {e}")
            return None

    async def _notify_callbacks(self, data):
        """Notify all registered callbacks."""
        for callback in self.order_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")

    def _generate_signature(self, params):
        """Generate signature for authenticated API requests."""
        if not self.api_secret:
            return urlencode(params)

        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return f"{query_string}&signature={signature}"

    def _simulate_order_result(self, symbol, side, quantity, price, order_type):
        """Simulate order result for paper trading."""
        order_id = str(uuid.uuid4())
        return {
            "symbol": symbol,
            "orderId": order_id,
            "clientOrderId": f"simulated_{order_id}",
            "transactTime": int(time.time() * 1000),
            "price": str(price),
            "origQty": str(quantity),
            "executedQty": str(quantity),
            "status": "FILLED",
            "type": order_type,
            "side": side
        }