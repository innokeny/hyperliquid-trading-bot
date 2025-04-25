import asyncio
import json
from datetime import datetime
from src.trading.market_data import MarketDataStreamer
from src.settings import settings

async def handle_orderbook(data: dict) -> None:
    """Handle orderbook updates."""
    if "data" in data:
        orderbook = data["data"]
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Orderbook Update:")
        print(f"Bids: {orderbook.get('bids', [])[:5]}")  # Show top 5 bids
        print(f"Asks: {orderbook.get('asks', [])[:5]}")  # Show top 5 asks

async def handle_trades(data: dict) -> None:
    """Handle trade updates."""
    if "data" in data:
        trades = data["data"]
        for trade in trades:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New Trade:")
            print(f"Price: {trade.get('px')}")
            print(f"Size: {trade.get('sz')}")
            print(f"Side: {trade.get('side')}")

async def main():
    """Main function to demonstrate market data streaming."""
    # Initialize the market data streamer
    streamer = MarketDataStreamer()
    
    # Register callbacks for different event types
    streamer.register_callback("l2Book", handle_orderbook)
    streamer.register_callback("trades", handle_trades)
    
    try:
        print("Starting market data stream...")
        print(f"Trading pair: {streamer.trading_pair}")
        print("Press Ctrl+C to stop")
        
        # Start the stream
        await streamer.start()
        
    except KeyboardInterrupt:
        print("\nStopping market data stream...")
        await streamer.stop()
    except Exception as e:
        print(f"Error: {e}")
        await streamer.stop()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 