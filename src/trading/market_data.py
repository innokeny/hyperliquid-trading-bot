import time
from typing import Callable, Dict, Any, Optional, List
from src.config import settings
from hyperliquid.info import Info
from hyperliquid.utils.types import Subscription
from loguru import logger

class MarketDataStreamer:
    """Handles real-time market data streaming from Hyperliquid."""
    
    def __init__(self, info: Info, coin: str = settings.COIN):
        """Initialize the market data streamer.
        
        Args:
            info: Hyperliquid Info object for API access
            coin: The coin to stream data for (e.g., "BTC")
        """
        self.coin = coin
        self.info = info
    
    def _subscribe(self, event_type: str, candle_interval: Optional[str] = None, callback: Callable = logger.info) -> None:
        """Subscribe to a specific event type."""
        subscribe_message: Subscription = {
                "type": event_type,
                "coin": self.coin
        } # type: ignore

        if event_type == "candle":
            subscribe_message["interval"] = candle_interval # type: ignore
        
        try:
            self.info.subscribe(subscribe_message, callback)
            logger.success(f"Subscribed to {event_type} for {self.coin}")
        except Exception as e:
            logger.error(f"Failed to subscribe to {event_type}: {e}")
            raise
    
    def get_orderbook(self) -> Dict[str, Any]:
        """Get the current orderbook snapshot."""
        try:
            return self.info.l2_snapshot(self.coin)
        except Exception as e:
            logger.error(f"Failed to get orderbook: {e}")
            raise
     
    
    def get_candles(
        self,
        interval: str = "1m",
        limit: int = 50,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get historical candles."""
        try:
            if end_time is None:
                end_time = int(time.time() * 1000)
                
            interval_ms = {
                "1m": 60 * 1000,
                "5m": 5 * 60 * 1000,
                "15m": 15 * 60 * 1000,
                "1h": 60 * 60 * 1000,
                "4h": 4 * 60 * 60 * 1000,
                "1d": 24 * 60 * 60 * 1000
            }.get(interval, 60 * 1000)
            
            start_time = end_time - (interval_ms * limit)
            
            return self.info.candles_snapshot(self.coin, interval, start_time, end_time)
        except Exception as e:
            logger.error(f"Failed to get candles: {e}")
            raise