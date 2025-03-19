"""
WebSocket module for real-time price data collection and management.
Provides functionality for live price data via Binance WebSocket API and historical data management.

Exposed components:
- PriceDataManager: Manages live and historical price data.
- BinanceWebSocket: WebSocket client for live data streaming.
"""

from .binance_ws import BinanceWebSocket

__all__ = ['BinanceWebSocket']
