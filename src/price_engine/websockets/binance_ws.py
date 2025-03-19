import json
import logging
import asyncio
import websockets
from datetime import datetime

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
        self.symbols = symbols or ["btcusdt"]
        self.callbacks = []
        self.ws = None
        self.running = False
        self.task = None

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

    async def connect(self):
        """Connect to Binance WebSocket and start listening for price data."""
        if self.running:
            logger.warning("WebSocket is already running")
            return

        self.running = True

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
            # Create WebSocket connection
            self.ws = await websockets.connect(ws_url)

            # Start the message handling task
            self.task = asyncio.create_task(self._handle_messages())

            logger.info("Connected to Binance WebSocket")
        except Exception as e:
            self.running = False
            logger.error(f"Failed to connect to Binance WebSocket: {e}")
            raise

    async def disconnect(self):
        """Disconnect from Binance WebSocket."""
        if not self.running:
            return

        self.running = False

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
            # Try to reconnect
            if self.running:
                logger.info("Attempting to reconnect...")
                await asyncio.sleep(5)  # Wait before reconnecting
                await self.connect()
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if self.running:
                # Try to reconnect
                logger.info("Attempting to reconnect after error...")
                await asyncio.sleep(5)
                await self.connect()


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
