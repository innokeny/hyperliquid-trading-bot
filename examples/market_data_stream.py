import asyncio
import json
from datetime import datetime
from src.trading.market_data import MarketDataStreamer
from src.trading.connection import HyperliquidConnection
from hyperliquid.utils.constants import MAINNET_API_URL
from src.config import settings

import logging

logging.basicConfig(level=logging.INFO)

def handle_orderbook(data: dict) -> None:
    """Handle orderbook updates."""
    print("Received orderbook data")

def handle_trades(data: dict) -> None:
    """Handle trade updates."""
    print("Received trades data")

def handle_candles(data: dict) -> None:
    """Handle candle updates."""
    print(f"Received candles data: {data}")


def main():
    """Main function to demonstrate market data streaming."""
    connection = HyperliquidConnection()
    connection.setup(base_url=MAINNET_API_URL)
    info, _ = connection.get_connection()
    streamer = MarketDataStreamer(info)
    
    try:
        print("Starting market data stream...")
        print(f"Trading pair: {streamer.coin}")
        print("Press Ctrl+C to stop")

        candles = streamer.get_candles(interval="1m", limit=50)
        print(f"Retrieved {len(candles)} candles")
    
        streamer.subscribe("trades", callback=handle_trades)
        streamer.subscribe("candle", candle_interval="1m", callback=handle_candles)
        
    except KeyboardInterrupt:
        print("\nStopping market data stream...")
    except Exception as e:
        print(f"Error: {e}")
        

if __name__ == "__main__":
    main()