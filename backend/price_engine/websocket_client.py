# src/websocket_client.py
import websocket
import json
import threading
import time
import logging
from datetime import datetime
from web3 import Web3

logger = logging.getLogger(__name__)

class BinanceWebSocketClient:
    def __init__(self, symbol="btcusdt", on_message_callback=None):
        self.symbol = symbol.lower()
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@trade"
        self.ws = None
        self.on_message_callback = on_message_callback
        self.running = False
        self.last_data_time = None
        self.missed_data = []

    def on_message(self, ws, message):
        data = json.loads(message)
        price = float(data["p"])
        timestamp = datetime.utcfromtimestamp(data["T"] / 1000)
        self.last_data_time = timestamp
        if self.on_message_callback:
            self.on_message_callback({
                "timestamp": timestamp,
                "price": price,
                "volume": float(data["q"]),
                "source": "binance"
            })

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket closed")
        self.running = False

    def on_open(self, ws):
        logger.info("WebSocket opened")
        self.running = True

    def start(self):
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def stop(self):
        if self.ws:
            self.ws.close()

    def backfill_missed_data(self, api_client):
        if not self.last_data_time:
            logger.warning("No last data time available for backfilling.")
            return

        start_time = int(self.last_data_time.timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        logger.info(f"Backfilling data from {self.last_data_time} ({start_time}) to now ({end_time})")

        try:
            missed_trades = api_client.fetch_historical_trades(self.symbol.upper(), start_time, end_time)
            if not missed_trades:
                logger.warning("No historical trades returned for the backfill period.")
                return

            logger.info(f"Fetched {len(missed_trades)} trades for backfilling.")
            self.missed_data.extend(missed_trades)
            for trade in missed_trades:
                if self.on_message_callback:
                    self.on_message_callback({
                        "timestamp": trade["timestamp"],
                        "price": trade["price"],
                        "volume": trade["volume"],
                        "source": trade["source"]
                    })
        except Exception as e:
            logger.error(f"Error during backfill: {e}")

class UniswapWebSocketClient:
    def __init__(self, w3, pair, on_message_callback=None):
        self.w3 = w3
        self.token0, self.token1 = pair
        self.on_message_callback = on_message_callback
        self.running = False
        self.last_data_time = None
        self.missed_data = []

        factory_address = self.w3.to_checksum_address("0x1f98431c8ad98523631ae4a59f267346ea31f984")
        factory_abi = [
            {
                "inputs": [
                    {"internalType": "address", "name": "token0", "type": "address"},
                    {"internalType": "address", "name": "token1", "type": "address"},
                    {"internalType": "uint24", "name": "fee", "type": "uint24"}
                ],
                "name": "getPool",
                "outputs": [{"internalType": "address", "name": "pool", "type": "address"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        factory_contract = self.w3.eth.contract(address=factory_address, abi=factory_abi)
        # Ensure token addresses are in checksum format
        token0_checksum = self.w3.to_checksum_address(self.token0)
        token1_checksum = self.w3.to_checksum_address(self.token1)
        self.pool_address = factory_contract.functions.getPool(token0_checksum, token1_checksum, 3000).call()

        self.pool_abi = [
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "internalType": "address", "name": "sender", "type": "address"},
                    {"indexed": True, "internalType": "address", "name": "recipient", "type": "address"},
                    {"indexed": False, "internalType": "int256", "name": "amount0", "type": "int256"},
                    {"indexed": False, "internalType": "int256", "name": "amount1", "type": "int256"},
                    {"indexed": False, "internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
                    {"indexed": False, "internalType": "uint128", "name": "liquidity", "type": "uint128"},
                    {"indexed": False, "internalType": "int24", "name": "tick", "type": "int24"}
                ],
                "name": "Swap",
                "type": "event"
            }
        ]
        self.pool_contract = self.w3.eth.contract(address=self.pool_address, abi=self.pool_abi)

    def on_message(self, event):
        amount0 = float(event["args"]["amount0"]) / 1e8  # WBTC (8 decimals)
        amount1 = float(event["args"]["amount1"]) / 1e6  # USDC (6 decimals)
        if self.token0 == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2":  # WETH (18 decimals)
            amount0 = float(event["args"]["amount0"]) / 1e18
        elif self.token0 == "0x570a5d26f7765ecb712c0924e4de2af7e458c554":  # Wrapped SOL (9 decimals)
            amount0 = float(event["args"]["amount0"]) / 1e9
        price = abs(amount1 / amount0) if amount0 != 0 else 0
        timestamp = datetime.utcfromtimestamp(self.w3.eth.get_block(event["blockNumber"])["timestamp"])
        self.last_data_time = timestamp
        if self.on_message_callback:
            self.on_message_callback({
                "timestamp": timestamp,
                "price": price,
                "volume": abs(amount1),
                "source": "uniswap"
            })

    def on_error(self, error):
        logger.error(f"WebSocket error: {error}")

    def on_close(self):
        logger.info("Event listener stopped")
        self.running = False

    def on_open(self):
        logger.info("Event listener started")
        self.running = True

    def start(self):
        self.running = True
        event_filter = self.pool_contract.events.Swap.create_filter(from_block="latest")  # Updated to from_block
        threading.Thread(target=self._listen_for_events, args=(event_filter,), daemon=True).start()

    def _listen_for_events(self, event_filter):
        self.on_open()
        try:
            while self.running:
                for event in event_filter.get_new_entries():
                    self.on_message(event)
                time.sleep(1)
        except Exception as e:
            self.on_error(e)
        finally:
            self.on_close()

    def stop(self):
        self.running = False

    def backfill_missed_data(self, api_client):
        if not self.last_data_time:
            logger.warning("No last data time available for backfilling.")
            return

        start_time = int(self.last_data_time.timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        logger.info(f"Backfilling data from {self.last_data_time} ({start_time}) to now ({end_time})")

        try:
            missed_trades = api_client.fetch_historical_trades((self.token0, self.token1), start_time, end_time)
            if not missed_trades:
                logger.warning("No historical trades returned for the backfill period.")
                return

            logger.info(f"Fetched {len(missed_trades)} trades for backfilling.")
            self.missed_data.extend(missed_trades)
            for trade in missed_trades:
                if self.on_message_callback:
                    self.on_message_callback({
                        "timestamp": trade["timestamp"],
                        "price": trade["price"],
                        "volume": trade["volume"],
                        "source": trade["source"]
                    })
        except Exception as e:
            logger.error(f"Error during backfill: {e}")

def get_websocket_client(exchange, w3=None, symbol=None, pair=None, on_message_callback=None):
    if exchange.lower() == "binance":
        if not symbol:
            raise ValueError("Symbol required for Binance WebSocket client")
        return BinanceWebSocketClient(symbol=symbol, on_message_callback=on_message_callback)
    elif exchange.lower() == "uniswap":
        if not w3 or not pair:
            raise ValueError("Web3 instance and token pair required for Uniswap WebSocket client")
        return UniswapWebSocketClient(w3, pair, on_message_callback)
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")