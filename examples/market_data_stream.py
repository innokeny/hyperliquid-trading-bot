import asyncio
import json
from datetime import datetime
from src.trading.market_data import MarketDataStreamer
from src.config import settings

async def handle_orderbook(data: dict) -> None:
    """Handle orderbook updates."""
    print("\nRaw orderbook data:", json.dumps(data, indent=2))
    
    if "data" in data:
        orderbook_data = data["data"]
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Orderbook Update:")
        print("Orderbook data:", orderbook_data)

async def handle_trades(data: dict) -> None:
    """Handle trade updates."""
    print("\nRaw trades data:", json.dumps(data, indent=2))
    
    if "data" in data:
        trades_data = data["data"]
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Trades Update:")
        print("Trades data:", trades_data)

async def main():
    """Main function to demonstrate market data streaming."""
    # Initialize the market data streamer
    streamer = MarketDataStreamer()
    
    # Register callbacks for different event types
    streamer.register_callback("orderbook", handle_orderbook)
    streamer.register_callback("trades", handle_trades)
    
    try:
        print("Starting market data stream...")
        print(f"Trading pair: {streamer.trading_pair}")
        print("Press Ctrl+C to stop")

        # Get historical candlestick data
        print("\nFetching candlestick data...")
        candles = await streamer.get_candles(interval="1m", limit=50)
        print(f"Retrieved {len(candles)} candles")
        
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