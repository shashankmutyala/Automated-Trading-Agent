# backend/analytics/price_ingestion.py

import requests
import datetime

def fetch_binance_prices(symbol="BTCUSDT", interval="1h", limit=10):
    """
    Fetch historical kline/candlestick data from Binance.

    Args:
        symbol (str): Trading pair like 'BTCUSDT'.
        interval (str): Timeframe e.g. '1h', '15m', '1d'.
        limit (int): Number of data points to retrieve.

    Returns:
        list of dict: Each entry contains timestamp, price (close), and volume.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    data = response.json()

    return [
        {
            "timestamp": datetime.datetime.fromtimestamp(item[0] / 1000),
            "price": float(item[4]),      # Closing price
            "volume": float(item[5])      # Volume
        }
        for item in data
    ]
