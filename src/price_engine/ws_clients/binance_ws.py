import json
import logging
import asyncio
import websockets
from datetime import datetime
import requests

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BinanceWebSocket:
    """
    WebSocket client for Binance real-time price data.
    """

    def __init__(self, symbols=None):
        """
        Initialize the WebSocket client.

        Args:
            symbols (list): List of symbols to track (e.g., ["btcusdt", "ethusdt"])
        """
        self.symbols = symbols
        self.callbacks = []
        self.ws = None
        self.running = False
        self.task = None

        # Track connection status and timing for backfill
        self.last_message_time = None
        self.disconnect_time = None
        self.reconnect_time = None
        self.connection_status = "disconnected"
        self.missed_data_callbacks = []

    def add_callback(self, callback):
        """
        Add a callback function to be called when price data is received.

        Args:
            callback: Function to call with (symbol, price, timestamp)
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def remove_callback(self, callback):
        """Remove a callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def add_missed_data_callback(self, callback):
        """
        Add a callback for missed data during reconnection.

        Args:
            callback: Function to call with (symbol, missed_data)
        """
        if callback not in self.missed_data_callbacks:
            self.missed_data_callbacks.append(callback)

    async def connect(self):
        """Connect to Binance WebSocket and start listening for price data."""
        if self.running:
            logger.warning("WebSocket is already running")
            return

        self.running = True
        self.connection_status = "connecting"

        # Format: symbol@trade for each symbol
        streams = [f"{symbol.lower()}@trade" for symbol in self.symbols]

        # Single stream or multiple streams
        if len(streams) == 1:
            ws_url = f"wss://stream.binance.com:9443/ws/{streams[0]}"
        else:
            streams_param = "/".join(streams)
            ws_url = f"wss://stream.binance.com:9443/stream?streams={streams_param}"

        logger.info(f"Connecting to Binance WebSocket: {ws_url}")

        try:
            # Use the standard websockets.connect method
            self.ws = await websockets.connect(ws_url)
            self.connection_status = "connected"
            self.reconnect_time = datetime.now()

            # Start the message handling task
            self.task = asyncio.create_task(self._handle_messages())

            logger.info("Connected to Binance WebSocket")
        except Exception as e:
            self.running = False
            self.connection_status = "disconnected"
            logger.error(f"Failed to connect to Binance WebSocket: {e}")
            raise

    async def disconnect(self):
        """Disconnect from Binance WebSocket."""
        if not self.running:
            return

        self.running = False
        self.connection_status = "disconnected"
        self.disconnect_time = datetime.now()

        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        if self.ws:
            await self.ws.close()
            self.ws = None

        logger.info("Disconnected from Binance WebSocket")

    async def _handle_messages(self):
        """Handle incoming WebSocket messages."""
        if not self.ws:
            logger.error("WebSocket not connected")
            return

        try:
            while self.running:
                message = await self.ws.recv()
                # Record the time this message was received for backfill tracking
                current_time = datetime.now()
                self.last_message_time = current_time

                try:
                    data = json.loads(message)

                    # Handle multi-stream format
                    if 'data' in data and 'stream' in data:
                        stream_data = data['data']
                        symbol = stream_data['s'].upper()  # Symbol is in 's' field
                        price = float(stream_data['p'])  # Price is in 'p' field
                        timestamp = stream_data['T']  # Trade timestamp is in 'T' field
                    # Handle single stream format
                    else:
                        symbol = data['s'].upper()  # Symbol is in 's' field
                        price = float(data['p'])  # Price is in 'p' field
                        timestamp = data['T']  # Trade timestamp is in 'T' field

                    # Call all registered callbacks
                    for callback in self.callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(symbol, price, timestamp)
                            else:
                                callback(symbol, price, timestamp)
                        except Exception as e:
                            logger.error(f"Error in callback: {e}")

                except json.JSONDecodeError:
                    logger.error(f"Failed to parse message: {message}")
                except KeyError as e:
                    logger.error(f"Missing expected field in message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except asyncio.CancelledError:
            # Task was cancelled - normal during shutdown
            pass
        except websockets.ConnectionClosed:
            logger.error("WebSocket connection closed unexpectedly")
            self.connection_status = "disconnected"
            self.disconnect_time = datetime.now()

            # Try to reconnect with backfill
            if self.running:
                logger.info("Attempting to reconnect with backfill...")
                await asyncio.sleep(5)  # Wait before reconnecting
                await self.reconnect_with_backfill()
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.connection_status = "disconnected"
            self.disconnect_time = datetime.now()

            if self.running:
                # Try to reconnect with backfill
                logger.info("Attempting to reconnect after error...")
                await asyncio.sleep(5)
                await self.reconnect_with_backfill()

    async def reconnect_with_backfill(self):
        """
        Reconnect to the WebSocket and fetch any data missed during the disconnection.
        This addresses Test Criteria #3: Backfill data after WebSocket disconnect.
        """
        # Only attempt backfill if we know when we were disconnected
        has_timing_info = (self.disconnect_time is not None and
                           self.last_message_time is not None)

        if not has_timing_info:
            logger.warning("Cannot backfill: missing disconnection timing information")
            await self.connect()  # Just reconnect without backfill
            return

        # Record reconnection time
        self.reconnect_time = datetime.now()

        # Connect first to resume the real-time data flow
        await self.connect()

        # Calculate the time window of missed data
        start_ms = int(self.last_message_time.timestamp() * 1000)
        end_ms = int(self.reconnect_time.timestamp() * 1000)

        logger.info(f"Fetching missed data from {self.last_message_time} to {self.reconnect_time}")

        # Fetch and process missed data for each symbol
        for symbol in self.symbols:
            symbol_upper = symbol.upper()
            # Fetch missed klines (candlestick data)
            missed_data = await self._fetch_missed_klines(
                symbol=symbol_upper,
                start_time=start_ms,
                end_time=end_ms
            )

            if not missed_data:
                continue

            logger.info(f"Backfilled {len(missed_data)} data points for {symbol_upper}")

            # Process the missed data through our regular callbacks
            for kline in missed_data:
                # Extract the data we need
                timestamp = kline[0]  # Open time
                close_price = float(kline[4])  # Close price

                # Call the regular callbacks with the backfilled data
                for callback in self.callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(symbol_upper, close_price, timestamp)
                        else:
                            callback(symbol_upper, close_price, timestamp)
                    except Exception as e:
                        logger.error(f"Error in callback with backfilled data: {e}")

            # Also notify specialized missed data callbacks
            for callback in self.missed_data_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(symbol_upper, missed_data)
                    else:
                        callback(symbol_upper, missed_data)
                except Exception as e:
                    logger.error(f"Error in missed data callback: {e}")

    async def _fetch_missed_klines(self, symbol, start_time, end_time, interval="2m"):
        """
        Fetch klines (candlestick data) for the period of disconnection.

        Args:
            symbol (str): Trading pair symbol (e.g., "BTCUSDT")
            start_time (int): Start time in milliseconds
            end_time (int): End time in milliseconds
            interval (str): Kline interval (default: "1m")

        Returns:
            list: List of klines or None if error
        """
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
                "limit": 2000  # Maximum allowed
            }

            # Make request asynchronously
            response = await asyncio.to_thread(
                requests.get,
                url,
                params=params
            )

            response.raise_for_status()
            klines = response.json()

            return klines
        except Exception as e:
            logger.error(f"Failed to fetch missed klines: {e}")
            return None


# Simplified function-based interface
async def binance_ws(symbols, callback):
    """
    Connect to Binance WebSocket and process price updates with a callback.

    Args:
        symbols (list): List of symbols to track
        callback: Function to call with (symbol, price, timestamp)
    """
    client = BinanceWebSocket(symbols)
    client.add_callback(callback)

    await client.connect()

    return client  # Return client so caller can disconnect when done