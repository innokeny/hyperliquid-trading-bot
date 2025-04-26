import asyncio
import time
from src.trading.market_data import MarketDataStreamer
from src.data.market_data import MarketDataCollector
from hyperliquid.utils.constants import MAINNET_API_URL
from loguru import logger


async def main():
    """Example usage of MarketDataCollector with threading."""
    from src.trading.connection import HyperliquidConnection
    
    connection = HyperliquidConnection()
    _, info, _ = connection.setup(base_url=MAINNET_API_URL)
    
    streamer = MarketDataStreamer(info)
    collector = MarketDataCollector(streamer, candle_interval="1m", max_candles=100)
    
    try:
        logger.info("Started market data collector")
        collector.start()
        # Example: Monitor data for 60 seconds
        for _ in range(120):

            print(f"len: {len(collector.candles)}, min: {min(collector.candles, key=lambda x: x['t'])['t']}, max: {max(collector.candles, key=lambda x: x['t'])['t']}")
            
            
            # if latest_candle:
            #     logger.info(f"Latest candle: {latest_candle}")
            # if latest_trades:
            #     logger.info(f"Latest trades: {len(latest_trades)}")
            # if orderbook:
            #     logger.info(f"Orderbook depth: {len(orderbook.get('bids', []))} bids, {len(orderbook.get('asks', []))} asks")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        collector.stop()
    finally:
        logger.info("Market data collector stopped")
        collector.stop()

if __name__ == "__main__":
    asyncio.run(main())