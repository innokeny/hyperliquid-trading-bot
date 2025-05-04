import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from loguru import logger
from hyperliquid.utils.constants import MAINNET_API_URL
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info

from src.trading.connection import HyperliquidConnection
from src.trading.market_data import MarketDataStreamer
from src.data.market_data import MarketDataCollector
from src.strategy import DonchianStrategy
from src.trading.order_manager import OrderManager
from src.settings import settings

class TradingBot:
    """Main trading bot class that integrates all components."""
    
    def __init__(self):
        """Initialize the trading bot with all required components."""
        # Initialize connection
        self.connection = HyperliquidConnection()
        self.connection.setup(base_url=MAINNET_API_URL)
        self.info, self.exchange = self.connection.get_connection()
        
        # Initialize components
        self.streamer = MarketDataStreamer(self.info)
        self.collector = MarketDataCollector(self.streamer, candle_interval="1m", max_candles=1000)
        self.strategy = DonchianStrategy()
        self.order_manager = OrderManager(self.exchange, self.info)
        
        logger.info("Trading bot initialized successfully")
    
    async def start(self):
        """Start the trading bot."""
        try:
            self.collector.start()
            logger.info("Started market data collection")
            
            while True:
                latest_candle = self.collector.candles[:-500]
                if not latest_candle:
                    await asyncio.sleep(1)
                    continue
                
                w = self.strategy.get_weights(latest_candle).iloc[-1]['w_combo']
                
                if abs(w) > 1e-6:
                    await self._process_signal(w)
                
                await asyncio.sleep(600)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        finally:
            self.stop()
    
    async def _process_signal(self, signal: float):
        """Process trading signal and execute orders."""
        try:
            if signal['signal'] == 'BUY' and self.strategy.current_position is None:
                # Place buy order
                order = self.order_manager.place_order(
                    name=settings.COIN,
                    is_buy=True,
                    sz=signal['position_size'],
                    limit_px=signal['current_price']
                )
                
                if order.get('status') == 'ok':
                    position: Dict[str, Any] = {
                        'side': 'LONG',
                        'entry_price': signal['current_price'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'size': signal['position_size']
                    }
                    self.strategy.update_position(position)
                    logger.info(f"Opened LONG position: {position}")
            
            elif signal['signal'] == 'SELL' and self.strategy.current_position is not None:
                # Place sell order
                order = self.order_manager.place_order(
                    name=settings.COIN,
                    is_buy=False,
                    sz=self.strategy.current_position['size'],
                    limit_px=signal['current_price']
                )
                
                if order.get('status') == 'ok':
                    logger.info(f"Closed position: {self.strategy.current_position}")
                    self.strategy.update_position({})  # Empty dict to clear position
        
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
    
    async def _check_stop_conditions(self, market_data: Dict[str, Any]):
        """Check if stop conditions are met for existing positions."""
        try:
            if self.strategy.current_position is None:
                return
                
            current_price = market_data['c']
            position = self.strategy.current_position
            
            if position['side'] == 'LONG':
                # Check stop loss
                if current_price <= position['stop_loss']:
                    await self._process_signal({'signal': 'SELL', 'current_price': current_price})
                    logger.info("Stop loss triggered")
                
                # Check take profit
                elif current_price >= position['take_profit']:
                    await self._process_signal({'signal': 'SELL', 'current_price': current_price})
                    logger.info("Take profit reached")
                
                # Check trailing stop
                elif self.strategy.trailing_stop_price is not None and current_price <= self.strategy.trailing_stop_price:
                    await self._process_signal({'signal': 'SELL', 'current_price': current_price})
                    logger.info("Trailing stop triggered")
        
        except Exception as e:
            logger.error(f"Error checking stop conditions: {str(e)}")
    
    def stop(self):
        """Stop the trading bot and clean up resources."""
        try:
            self.collector.stop()
            logger.info("Stopped market data collection")
            
            # Cancel all open orders
            self.order_manager.cancel_all_orders(settings.COIN)
            logger.info("Cancelled all open orders")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

async def main():
    """Main function to run the trading bot."""
    bot = TradingBot()
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main()) 