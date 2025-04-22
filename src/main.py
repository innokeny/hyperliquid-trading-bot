import asyncio
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

from src.config.ml_config import MLConfig
from src.trading.hyperliquid_client import HyperliquidClient
from src.trading.strategy import TradingStrategy
from src.ml.model_inference import ModelInference
from src.data.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)

class TradingBot:
    """Main trading bot class that orchestrates all components."""

    def __init__(self, config: MLConfig):
        """
        Initialize the trading bot.

        Args:
            config: MLConfig instance containing all configuration parameters
        """
        self.config = config
        self.client = HyperliquidClient(
            api_url=config.api_url,
            account_address=config.account_address,
            secret_key=config.secret_key
        )
        self.data_preprocessor = DataPreprocessor()
        self.model_inference = ModelInference(model_path=config.model_path)
        self.strategy = TradingStrategy(self.model_inference, config)
        self.is_running = False
        self.last_trade_time = None
        self.position = None
        logger.info("Trading bot initialized")

    async def start(self):
        """Start the trading bot."""
        try:
            self.is_running = True
            logger.info("Starting trading bot")
            
            # Initialize connection
            await self.client.connect()
            
            # Main trading loop
            while self.is_running:
                try:
                    # Get market data
                    market_data = await self._get_market_data()
                    
                    # Preprocess data
                    df = pd.DataFrame([market_data])
                    processed_data = self.data_preprocessor.clean_market_data(df)
                    processed_data = self.data_preprocessor.calculate_technical_indicators(processed_data)
                    processed_data = self.data_preprocessor.create_features(processed_data)
                    processed_data = self.data_preprocessor.normalize_features(processed_data)
                    
                    # Generate trading signals
                    signal = self.strategy.generate_signals(processed_data.iloc[-1].to_dict())
                    
                    # Execute trading logic
                    await self._execute_trading_logic(signal, processed_data.iloc[-1].to_dict())
                    
                    # Update position
                    self.position = await self.client.get_position()
                    if self.position:
                        self.strategy.update_position(self.position)
                    
                    # Log performance metrics
                    self._log_performance_metrics()
                    
                    # Sleep for the configured interval
                    await asyncio.sleep(self.config.trading_interval)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {str(e)}")
                    await asyncio.sleep(5)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Fatal error in trading bot: {str(e)}")
            self.is_running = False
            raise

    async def stop(self):
        """Stop the trading bot."""
        self.is_running = False
        await self.client.disconnect()
        logger.info("Trading bot stopped")

    async def _get_market_data(self) -> Dict[str, Any]:
        """Get current market data."""
        try:
            # Get order book data
            order_book = await self.client.get_order_book()
            
            # Get recent trades
            trades = await self.client.get_recent_trades()
            
            # Get current price
            price = await self.client.get_current_price()
            
            # Get account balance
            balance = await self.client.get_balance()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'order_book': order_book,
                'trades': trades,
                'price': price,
                'balance': balance,
                'position': self.position
            }
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            raise

    async def _execute_trading_logic(self, signal: Dict[str, Any], market_data: Dict[str, Any]):
        """Execute trading logic based on signals."""
        try:
            if signal['signal'] == "HOLD":
                return
                
            # Check if we have enough balance
            if market_data['balance'] < signal['position_size']:
                logger.warning("Insufficient balance for trade")
                return
                
            # Check if we're already in a position
            if self.position is not None:
                if signal['signal'] == "EXIT":
                    # Close position
                    await self.client.close_position()
                    self.last_trade_time = datetime.now()
                    logger.info("Position closed")
                return
                
            # Execute new trade
            if signal['signal'] in ["BUY", "SELL"]:
                order = {
                    'side': signal['signal'],
                    'size': signal['position_size'],
                    'price': market_data['price'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit']
                }
                
                # Place order
                order_id = await self.client.place_order(order)
                self.last_trade_time = datetime.now()
                logger.info(f"Order placed: {order_id}")
                
        except Exception as e:
            logger.error(f"Error executing trading logic: {str(e)}")
            raise

    def _log_performance_metrics(self):
        """Log performance metrics."""
        try:
            if self.position is None:
                return
                
            # Calculate current P&L
            current_price = self.position['current_price']
            entry_price = self.position['entry_price']
            size = self.position['size']
            
            if self.position['side'] == "LONG":
                pnl = (current_price - entry_price) * size
            else:
                pnl = (entry_price - current_price) * size
                
            # Log metrics
            logger.info(f"Position: {self.position['side']} | Size: {size} | P&L: {pnl}")
            logger.info(f"Stop Loss: {self.position['stop_loss']} | Take Profit: {self.position['take_profit']}")
            
        except Exception as e:
            logger.error(f"Error logging performance metrics: {str(e)}")

async def main():
    """Main entry point for the trading bot."""
    try:
        # Load configuration
        config = MLConfig(
            api_url="https://api.hyperliquid.xyz",
            account_address="your_account_address",
            secret_key="your_secret_key",
            model_path="models/model.pth"
        )
        
        # Initialize and start trading bot
        bot = TradingBot(config)
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        if 'bot' in locals():
            await bot.stop()

if __name__ == "__main__":
    asyncio.run(main()) 