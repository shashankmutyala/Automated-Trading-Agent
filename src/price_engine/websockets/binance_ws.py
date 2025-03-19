import asyncio
import websockets
import json

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"

async def binance_ws(symbol, callback):
    """Connect to Binance WebSocket for live price data."""
    stream_name = f"{symbol.lower()}@trade"  # Example: btcusdt@trade
    url = f"{BINANCE_WS_URL}/{stream_name}"

    async with websockets.connect(url) as ws:
        print(f"Connected to Binance WebSocket for {symbol}")
        async for message in ws:
            data = json.loads(message)
            price = float(data['p'])  # Trade price
            timestamp = data['T']  # Trade timestamp
            await callback(symbol.upper(), price, timestamp)
