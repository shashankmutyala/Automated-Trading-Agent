import numpy as np

def calculate_moving_average(prices, window_size=5):
    """Calculate a simple moving average."""
    if len(prices) >= window_size:
        return np.mean(prices[-window_size:])
    return None

def generate_signal(price, moving_average):
    """Generate buy/sell signals based on price and moving average."""
    if price > moving_average:
        return "BUY"
    elif price < moving_average:
        return "SELL"
    return "HOLD"
