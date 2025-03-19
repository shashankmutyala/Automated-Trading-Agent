import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from price_engine.websockets.binance_ws import binance_ws
from price_engine.db.mongodb_handler import MongoDBHandler
from price_engine.processor import calculate_moving_average
import asyncio


async def main():
    db_handler = MongoDBHandler()
    prices = []  # Store recent prices for analysis

    async def handle_data(symbol, price, timestamp):
        # Store the price in the database
        db_handler.store_price("price_data", symbol, price, timestamp)

        # Add price to the list and calculate moving average
        prices.append(price)
        moving_avg = calculate_moving_average(prices, window_size=5)

        print(f"Symbol: {symbol}, Price: {price}, Moving Average: {moving_avg}")

    # WebSocket tasks for BTC and ETH
    tasks = [
        binance_ws("BTCUSDT", handle_data),
        binance_ws("ETHUSDT", handle_data),
    ]

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
