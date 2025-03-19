import asyncio
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from price_engine.data_manager import PriceDataManager

async def fetch_historical_data(symbol, start_time, end_time):
    data_manager = PriceDataManager()

    # Convert string times to datetime objects
    start_time = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    end_time = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")

    # Fetch historical data for the given symbol and time range
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
        print("No data available")

# Run the function
asyncio.run(fetch_historical_data("BTCUSDT", "2025-03-19T14:36:00", "2025-03-19T14:38:00"))
