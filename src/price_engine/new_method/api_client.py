# new_method/api_client.py
import requests
import logging
from datetime import datetime
from web3 import Web3
import os
from dotenv import load_dotenv
import time

load_dotenv()

logger = logging.getLogger(__name__)

class BinanceAPIClient:
    def __init__(self):
        self.base_url = "https://api.binance.com"

    def fetch_historical_data(self, symbol, interval, start_time, end_time):
        endpoint = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        return [
            {
                "timestamp": datetime.fromtimestamp(candle[0] / 1000),
                "price": float(candle[4]),
                "volume": float(candle[5]),
                "source": "binance"
            }
            for candle in data
        ]

    def fetch_historical_trades(self, symbol, start_time, end_time):
        endpoint = f"{self.base_url}/api/v3/historicalTrades"
        params = {
            "symbol": symbol,
            "limit": 1000
        }
        trades = []
        last_id = None

        while True:
            if last_id:
                params["fromId"] = last_id

            try:
                response = requests.get(endpoint, params=params)
                response.raise_for_status()
                new_trades = response.json()

                if not new_trades:
                    break

                for trade in new_trades:
                    trade_time = int(trade["time"])
                    if trade_time < start_time:
                        continue
                    if trade_time > end_time:
                        break
                    trades.append({
                        "timestamp": datetime.utcfromtimestamp(trade_time / 1000),
                        "price": float(trade["price"]),
                        "volume": float(trade["qty"]),
                        "source": "binance"
                    })

                last_id = new_trades[-1]["id"] if new_trades else None
                if len(new_trades) < 1000:
                    break

            except requests.RequestException as e:
                logger.error(f"Error fetching historical trades: {e}")
                break

        logger.info(f"Fetched {len(trades)} historical trades for {symbol} from {start_time} to {end_time}")
        return trades

class UniswapAPIClient:
    def __init__(self):
        infura_api_key = os.getenv("INFURA_API_KEY")
        if not infura_api_key:
            raise ValueError("INFURA_API_KEY not found in environment variables")
        self.w3 = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{infura_api_key}"))
        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum node via Infura")

        self.factory_address = self.w3.to_checksum_address("0x1f98431c8ad98523631ae4a59f267346ea31f984")
        self.factory_abi = [
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
        self.factory_contract = self.w3.eth.contract(address=self.factory_address, abi=self.factory_abi)

        # Token addresses (mainnet, in lowercase)
        self.wbtc_address = self.w3.to_checksum_address("0x2260fac5e5542a773aa44fbcfedf7c193bc2c599")  # WBTC
        self.weth_address = self.w3.to_checksum_address("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2")  # WETH
        self.usdc_address = self.w3.to_checksum_address("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48")  # USDC
        self.fee = 3000  # 0.3% fee tier

        # ABI for Uniswap V3 Pool contract (for fetching Swap events)
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

        # Cache for block timestamps
        self.block_timestamp_cache = {}

    def get_block_timestamp(self, block_number):
        """Fetch and cache the timestamp of a block."""
        if block_number in self.block_timestamp_cache:
            return self.block_timestamp_cache[block_number]
        try:
            block = self.w3.eth.get_block(block_number)
            timestamp = block["timestamp"]
            self.block_timestamp_cache[block_number] = timestamp
            return timestamp
        except Exception as e:
            logger.error(f"Error fetching block {block_number} timestamp: {e}")
            raise

    def get_block_number_by_timestamp(self, timestamp):
        """
        Find the block number closest to the given timestamp using binary search.
        :param timestamp: Unix timestamp in seconds
        :return: Block number
        """
        logger.debug(f"Searching for block at timestamp {timestamp}")
        target_timestamp = int(timestamp)
        latest_block = self.w3.eth.block_number
        latest_timestamp = self.get_block_timestamp(latest_block)

        # Estimate the starting block based on average block time (12 seconds)
        seconds_diff = latest_timestamp - target_timestamp
        estimated_blocks_back = seconds_diff // 12  # Approximate blocks (12 seconds per block)
        estimated_block = max(0, latest_block - estimated_blocks_back)

        # Narrow the search range around the estimated block
        low = max(0, estimated_block - 10000)  # Search 10,000 blocks before the estimate
        high = min(latest_block, estimated_block + 10000)  # Search 10,000 blocks after the estimate

        logger.debug(f"Starting binary search: low={low}, high={high}, target_timestamp={target_timestamp}")

        while low <= high:
            mid = (low + high) // 2
            block_timestamp = self.get_block_timestamp(mid)
            logger.debug(f"Checking block {mid}: timestamp={block_timestamp}")

            if block_timestamp < target_timestamp:
                low = mid + 1
            elif block_timestamp > target_timestamp:
                high = mid - 1
            else:
                logger.debug(f"Found exact block {mid} for timestamp {target_timestamp}")
                return mid

        # Return the closest block
        if high < 0:
            logger.debug(f"Returning block {low} (high < 0)")
            return low
        if low > latest_block:
            logger.debug(f"Returning block {high} (low > latest_block)")
            return high
        low_timestamp = self.get_block_timestamp(low)
        high_timestamp = self.get_block_timestamp(high)
        if abs(low_timestamp - target_timestamp) < abs(high_timestamp - target_timestamp):
            logger.debug(f"Returning block {low} (closer timestamp)")
            return low
        logger.debug(f"Returning block {high} (closer timestamp)")
        return high

    def get_pool_address(self, token0, token1):
        """Get the Uniswap V3 pool address for a token pair."""
        # Ensure token addresses are in checksum format
        token0 = self.w3.to_checksum_address(token0)
        token1 = self.w3.to_checksum_address(token1)
        pool_address = self.factory_contract.functions.getPool(
            token0, token1, self.fee
        ).call()
        return pool_address

    def fetch_historical_data(self, pair, interval, start_time, end_time):
        """
        Fetch historical swap data directly from the Uniswap V3 pool contract.
        :param pair: Tuple of (token0_address, token1_address), e.g., (WBTC, USDC)
        """
        token0, token1 = pair
        pool_address = self.get_pool_address(token0, token1)
        if pool_address == "0x0000000000000000000000000000000000000000":
            logger.error(f"No pool found for {token0}/{token1}")
            return []

        # Create a contract instance for the pool
        pool_contract = self.w3.eth.contract(address=pool_address, abi=self.pool_abi)

        # Convert timestamps to block numbers
        logger.info(f"Converting start_time {start_time // 1000} to block number...")
        start_block = self.get_block_number_by_timestamp(start_time // 1000)
        logger.info(f"Converting end_time {end_time // 1000} to block number...")
        end_block = self.get_block_number_by_timestamp(end_time // 1000)

        logger.info(f"Fetching Swap events from block {start_block} to {end_block} for pool {pool_address}...")

        # Fetch Swap events in chunks to avoid Infura rate limits
        block_range = 10000  # Fetch events in chunks of 10,000 blocks
        current_block = start_block
        historical_data = []

        while current_block <= end_block:
            to_block = min(current_block + block_range - 1, end_block)
            logger.info(f"Fetching Swap events from block {current_block} to {to_block}...")
            try:
                swap_events = pool_contract.events.Swap.get_logs(
                    from_block=current_block,
                    to_block=to_block
                )
                logger.info(f"Fetched {len(swap_events)} Swap events from block {current_block} to {to_block}")
                for event in swap_events:
                    timestamp = self.w3.eth.get_block(event["blockNumber"])["timestamp"]
                    timestamp_dt = datetime.utcfromtimestamp(timestamp)
                    amount0 = float(event["args"]["amount0"])
                    amount1 = float(event["args"]["amount1"])

                    # Adjust for token decimals
                    if token0 == self.wbtc_address:  # WBTC (8 decimals)
                        amount0 /= 1e8
                        amount1 /= 1e6  # USDC (6 decimals)
                    elif token0 == self.weth_address:  # WETH (18 decimals)
                        amount0 /= 1e18
                        amount1 /= 1e6  # USDC (6 decimals)

                    price = abs(amount1 / amount0) if amount0 != 0 else 0
                    historical_data.append({
                        "timestamp": timestamp_dt,
                        "price": price,
                        "volume": abs(amount1),
                        "source": "uniswap"
                    })
            except Exception as e:
                logger.error(f"Error fetching swap events from block {current_block} to {to_block}: {e}")
                # Retry after a delay
                time.sleep(5)
                continue

            current_block = to_block + 1

        logger.info(f"Fetched {len(historical_data)} historical swaps for pool {pool_address}")
        return historical_data

    def fetch_historical_trades(self, pair, start_time, end_time):
        """
        Fetch historical trades (swaps) for backfilling.
        """
        historical_data = self.fetch_historical_data(pair, None, start_time, end_time)
        return [
            {
                "timestamp": data["timestamp"],
                "price": data["price"],
                "volume": data["volume"],
                "source": data["source"]
            }
            for data in historical_data
        ]

def get_api_client(exchange):
    if exchange.lower() == "binance":
        return BinanceAPIClient()
    elif exchange.lower() == "uniswap":
        return UniswapAPIClient()
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")