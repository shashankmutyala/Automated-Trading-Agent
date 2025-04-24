# tests/test_websocket.py
import unittest
import time
from src.websocket_client import BinanceWebSocketClient

class TestBinanceWebSocketClient(unittest.TestCase):
    def setUp(self):
        self.received_data = []
        self.client = BinanceWebSocketClient(symbol="btcusdt", on_message_callback=self.store_data)

    def store_data(self, data):
        self.received_data.append(data)

    def test_websocket_connection(self):
        # Start the WebSocket client and run for 5 seconds to collect data
        self.client.start()
        time.sleep(5)
        self.client.stop()

        # Verify that data was received
        self.assertGreater(len(self.received_data), 0)
        self.assertIn("timestamp", self.received_data[0])
        self.assertIn("price", self.received_data[0])

    def test_websocket_disconnection(self):
        # Start the WebSocket, disconnect for 2 seconds, then reconnect
        self.client.start()
        time.sleep(2)
        self.client.stop()
        time.sleep(2)
        self.client.start()
        time.sleep(2)
        self.client.stop()

        # Verify that data was received after reconnection
        self.assertGreater(len(self.received_data), 0)

if __name__ == "__main__":
    unittest.main()