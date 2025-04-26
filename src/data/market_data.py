from typing import Dict, List, Optional, Any, Callable
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
from ..trading.market_data import MarketDataStreamer

class MarketDataCollector:
    """Collects and manages market data from Hyperliquid."""

    def __init__(
        self,
        market_data_streamer: MarketDataStreamer,
        candle_interval: str = "1m",
        max_candles: int = 1000
    ):
        """Initialize the market data collector.
        
        Args:
            market_data_streamer: MarketDataStreamer instance for data streaming
            candle_interval: Interval for candle data (e.g., "1m", "5m", "1h")
            max_candles: Maximum number of candles to keep in memory
        """
        self.streamer = market_data_streamer
        self.candle_interval = candle_interval
        self.max_candles = max_candles
        
        self.candles: List[Dict[str, Any]] = market_data_streamer.get_candles(interval=self.candle_interval, limit=self.max_candles)
        self.trades: List[Dict[str, Any]] = []
        self.orderbook: Optional[Dict[str, Any]] = None
        
    def start(self) -> None:
        """Start the market data collector."""        
        self.streamer.subscribe("candle", candle_interval=self.candle_interval, callback=self.handle_candles)

    def stop(self) -> None:
        """Stop the market data collector."""
        self.streamer.unsubscribe("candle")

    def merge_candle(self, data: Dict[str, Any]) -> None:
        """Merge new candle data with existing candles."""
        if not self.candles:
            self.candles = [data]
            return

        latest_candle = self.candles[-1]
        if latest_candle['t'] == data['t']:
            self.candles[-1] = data
        else:
            self.candles.append(data)
    
    def check_max_candles(self) -> None:
        """Ensure we don't exceed the maximum number of candles."""
        if len(self.candles) > self.max_candles:
            self.candles = self.candles[-self.max_candles:]
            logger.debug(f"Trimmed candles to {self.max_candles}")

    def handle_candles(self, data: Dict[str, Any]) -> None:
        """Handle incoming candle updates."""
        try:
            if 'data' in data:
                self.merge_candle(data['data'])
                self.check_max_candles()
                logger.debug(f"Updated candles at {datetime.fromtimestamp(data['data']['t']/1000)}")
        except Exception as e:
            logger.error(f"Error handling candles: {str(e)}")

    def get_latest_candle(self) -> Optional[Dict[str, Any]]:
        """Get the latest candle data."""
        return self.candles[-1] if self.candles else None

    def get_latest_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the latest trades."""
        return self.trades[-limit:] if self.trades else []

    def get_latest_orderbook(self) -> Optional[Dict[str, Any]]:
        """Get the latest orderbook snapshot."""
        return self.orderbook


        