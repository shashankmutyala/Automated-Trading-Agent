import asyncio
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from price_engine.data_manager import PriceDataManager

async def fetch_all_symbols(symbols, start_time, end_time):
    data_manager = PriceDataManager()

    # Convert string times to datetime objects
    start_time = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    end_time = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")

    # Iterate over all symbols and fetch data
    for symbol in symbols:
        print(f"Fetching historical data for {symbol}...")
        df = await data_manager.get_price_history(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            use_api_if_needed=True  # Automatically fetch missing data from Binance
        )
        if not df.empty:
            print(f"Fetched {len(df)} records for {symbol}:")
            print(df)  # Print the data as a DataFrame
        else:
            print(f"No data available for {symbol}")

# List of symbols to fetch data for
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

# Run the function
asyncio.run(fetch_all_symbols(symbols, "2025-03-19T19:33:00", "2025-03-19T19:34:00"))
