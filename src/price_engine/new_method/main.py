# new_method/main.py
from api_client import get_api_client
from websocket_client import get_websocket_client
from data_normalizer import DataNormalizer
from database import Database
import logging
from datetime import datetime, timedelta
import time
import argparse
from web3 import Web3
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def store_and_print(data, symbol):
    normalized_data = DataNormalizer.normalize_data(data)
    db.store_price(normalized_data, symbol)
    logger.info(f"Stored - Timestamp: {normalized_data['timestamp']}, Price: {normalized_data['price']}, Volume: {normalized_data.get('volume', 0.0)}")

if __name__ == "__main__":
    # Initialize Web3 for Uniswap
    infura_api_key = os.getenv("INFURA_API_KEY")
    if not infura_api_key:
        raise ValueError("INFURA_API_KEY not found in environment variables")
    w3 = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{infura_api_key}"))
    if not w3.is_connected():
        raise ConnectionError("Failed to connect to Ethereum node via Infura")

    # Record the start time of the run
    run_start_time = datetime.now()
    logger.info(f"Starting run at: {run_start_time}")

    # Initialize database
    db = Database()

    # Token addresses for Uniswap (in lowercase to match api_client.py)
    wbtc_address = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"  # WBTC
    weth_address = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"  # WETH
    usdc_address = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"  # USDC

    # Define trading pairs for both exchanges
    pairs = [
        # Binance pairs
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "pair": None,
            "display_name": "BTC/USDT (Binance)"
        },
        {
            "exchange": "binance",
            "symbol": "ETHUSDT",
            "pair": None,
            "display_name": "ETH/USDT (Binance)"
        },
        {
            "exchange": "binance",
            "symbol": "SOLUSDT",
            "pair": None,
            "display_name": "SOL/USDT (Binance)"
        },
        # Uniswap pairs
        {
            "exchange": "uniswap",
            "symbol": "WBTCUSDC",
            "pair": (wbtc_address, usdc_address),
            "display_name": "WBTC/USDC (Uniswap)"
        },
        {
            "exchange": "uniswap",
            "symbol": "WETHUSDC",
            "pair": (weth_address, usdc_address),
            "display_name": "WETH/USDC (Uniswap)"
        }
    ]

    # Test Criterion 1: Fetch historical data for all pairs
    logger.info("Fetching historical data...")
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)  # Reduced to 1 day
    for pair_info in pairs:
        exchange = pair_info["exchange"]
        symbol = pair_info["symbol"]
        pair = pair_info["pair"]
        display_name = pair_info["display_name"]
        
        logger.info(f"Fetching historical data for {display_name}...")
        api_client = get_api_client(exchange)
        historical_data = api_client.fetch_historical_data(
            symbol if exchange == "binance" else pair,
            "1h",
            start_time,
            end_time
        )
        for data in historical_data:
            store_and_print(data, symbol)

    # Test Criterion 2 & 3: Real-time data with disconnection for all pairs
    ws_clients = []
    for pair_info in pairs:
        exchange = pair_info["exchange"]
        symbol = pair_info["symbol"]
        pair = pair_info["pair"]
        display_name = pair_info["display_name"]
        
        logger.info(f"Starting WebSocket for real-time data for {display_name}...")
        ws_client = get_websocket_client(
            exchange,
            w3=w3 if exchange == "uniswap" else None,
            symbol=symbol if exchange == "binance" else None,
            pair=pair if exchange == "uniswap" else None,
            on_message_callback=lambda data, s=symbol: store_and_print(data, s)
        )
        ws_clients.append(ws_client)
        ws_client.start()

    time.sleep(5)  # Run for 5 seconds
    logger.info("Disconnecting WebSockets for 10 seconds...")
    for ws_client in ws_clients:
        ws_client.stop()
    time.sleep(10)  # Simulate 10-second disconnection
    logger.info("Reconnecting WebSockets...")
    for ws_client in ws_clients:
        ws_client.start()
        api_client = get_api_client(pair_info["exchange"])
        ws_client.backfill_missed_data(api_client)
    time.sleep(5)  # Run for another 5 seconds
    for ws_client in ws_clients:
        ws_client.stop()

    # Test Criterion 4: Handle missing data
    sample_data = [
        {"timestamp": datetime.now(), "price": None, "volume": 100},
        {"timestamp": datetime.now(), "price": 50000, "volume": 200}
    ]
    reconciled_data = DataNormalizer.handle_missing_data(sample_data)
    for data in reconciled_data:
        logger.info(f"Reconciled - Timestamp: {data['timestamp']}, Price: {data['price']}")

    # Close the database connection
    db.close()

    # Save the start time of this run to a file (last_run.txt)
    with open("last_run.txt", "w") as f:
        f.write(run_start_time.isoformat())
    logger.info(f"Saved run start time to last_run.txt: {run_start_time}")