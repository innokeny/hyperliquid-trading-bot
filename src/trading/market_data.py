import asyncio
import json
import logging
import time
from typing import Callable, Dict, Any, Optional, Union, List
import websockets
import aiohttp
from src.config import settings

logger = logging.getLogger(__name__)

class MarketDataStreamer:
    """Handles real-time market data streaming from Hyperliquid."""
    
    def __init__(self, trading_pair: str = settings.TRADING_PAIR):
        """Initialize the market data streamer.
        
        Args:
            trading_pair: The trading pair to stream data for (e.g., "BTC-PERP")
        """
        self.trading_pair = trading_pair
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        self.rest_url = "https://api.hyperliquid.xyz"
        self.websocket: Optional[Any] = None
        self.running = False
        self.callbacks: Dict[str, Callable] = {}
        self.subscriptions: Dict[str, bool] = {}
        
    async def connect(self) -> None:
        """Establish WebSocket connection to Hyperliquid."""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            logger.info(f"Connected to Hyperliquid WebSocket at {self.ws_url}")
            
            # Subscribe to all registered event types
            for event_type in self.callbacks.keys():
                await self._subscribe(event_type)
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise
    
    async def _subscribe(self, event_type: str) -> None:
        """Subscribe to a specific event type.
        
        Args:
            event_type: The type of event to subscribe to
        """
        if not self.websocket:
            return
            
        subscribe_message = {
            "method": "subscribe",
            "subscription": {
                "type": event_type,
                "coin": self.trading_pair.split("-")[0]
            }
        }
        
        try:
            await self.websocket.send(json.dumps(subscribe_message))
            self.subscriptions[event_type] = True
            logger.info(f"Subscribed to {event_type} for {self.trading_pair}")
        except Exception as e:
            logger.error(f"Failed to subscribe to {event_type}: {e}")
    
    async def start(self) -> None:
        """Start streaming market data."""
        if self.running:
            logger.warning("Market data streamer is already running")
            return
            
        self.running = True
        try:
            await self.connect()
            
            while self.running and self.websocket:
                try:
                    message = await self.websocket.recv()
                    data = json.loads(message)
                    await self._handle_message(data)
                    
                except websockets.ConnectionClosed:
                    logger.warning("WebSocket connection closed, attempting to reconnect...")
                    await self.connect()
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await asyncio.sleep(1)  # Prevent tight loop on errors
                    
        except Exception as e:
            logger.error(f"Error in market data streamer: {e}")
            self.running = False
            raise
    
    async def stop(self) -> None:
        """Stop streaming market data."""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        logger.info("Market data streamer stopped")
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for a specific event type.
        
        Args:
            event_type: The type of event to register for (e.g., "l2Book", "trades")
            callback: The function to call when the event occurs
        """
        self.callbacks[event_type] = callback
        logger.debug(f"Registered callback for {event_type}")
        
        # If already connected, subscribe to the new event type
        if self.websocket and self.running:
            asyncio.create_task(self._subscribe(event_type))
    
    async def _handle_message(self, data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages.
        
        Args:
            data: The parsed message data
        """
        try:
            # Check if this is a subscription confirmation
            if "subscription" in data and "type" in data["subscription"]:
                event_type = data["subscription"]["type"]
                if event_type in self.callbacks:
                    logger.info(f"Subscription confirmed for {event_type}")
                return
                
            # Handle regular data messages
            channel = data.get("channel")
            if channel and channel in self.callbacks:
                await self.callbacks[channel](data)
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def get_orderbook(self) -> Dict[str, Any]:
        """Get the current orderbook snapshot.
        
        Returns:
            Dict containing the orderbook data
        """
        if not self.websocket:
            raise RuntimeError("Not connected to WebSocket")
            
        message = {
            "method": "getOrderbook",
            "params": {
                "coin": self.trading_pair.split("-")[0]
            }
        }
        
        await self.websocket.send(json.dumps(message))
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def get_trades(self, limit: int = 100) -> Dict[str, Any]:
        """Get recent trades.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            Dict containing recent trades
        """
        if not self.websocket:
            raise RuntimeError("Not connected to WebSocket")
            
        message = {
            "method": "getTrades",
            "params": {
                "coin": self.trading_pair.split("-")[0],
                "limit": limit
            }
        }
        
        await self.websocket.send(json.dumps(message))
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def get_candles(
        self,
        interval: str = "1m",
        limit: int = 50,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get historical candles.
        
        Args:
            interval: Candlestick interval (e.g., "1m", "5m", "15m", "1h", "4h", "1d")
            limit: Number of candles to retrieve
            end_time: End time in milliseconds (defaults to current time)
            
        Returns:
            List of candle data:
            [
                {
                    "T": int,      # End time
                    "c": str,      # Close price
                    "h": str,      # High price
                    "i": str,      # Interval
                    "l": str,      # Low price
                    "n": int,      # Number of trades
                    "o": str,      # Open price
                    "s": str,      # Start time
                    "t": int,      # Start time
                    "v": str       # Volume
                },
                ...
            ]
        """
        if end_time is None:
            end_time = int(time.time() * 1000)
            
        # Calculate start time based on interval and limit
        interval_ms = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000
        }.get(interval, 60 * 1000)
        
        start_time = end_time - (interval_ms * limit)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.rest_url}/info",
                json={
                    "type": "candleSnapshot",
                    "req": {
                        "coin": self.trading_pair.split("-")[0],
                        "interval": interval,
                        "startTime": start_time,
                        "endTime": end_time
                    }
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    raise RuntimeError(f"Failed to fetch candles: {response.status}") 