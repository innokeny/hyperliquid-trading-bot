from src.ml.model_inference import ModelInference
from src.trading.strategy import TradingStrategy
from src.settings import settings
from loguru import logger
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

async def main():
    """Example usage of TradingStrategy."""
    try:
        # Initialize model inference
        model_path = str(settings.MODEL_PATH) if settings.MODEL_PATH else "models/model.pth"
        model_inference = ModelInference(model_path)
        
        # Initialize trading strategy
        strategy = TradingStrategy(model_inference)
        
        # Example market data
        market_data = {
            't': 123123123,
            'o': 50000.0,
            'h': 51000.0,
            'l': 49500.0,
            'c': 50500.0,
            'v': 1000.0,
            'rsi': 45.0,
            'macd': -50.0,
            'signal': -30.0,
            'volatility': 0.02,
            'sma_20': 50200.0,
            'atr': 200.0,
            'available_capital': 10000.0
        }
        
        # Generate trading signals
        signal = strategy.generate_signals(market_data)
        logger.info(f"Generated signal: {signal}")
        
        # Example position update
        if signal['signal'] == "BUY":
            position: Dict[str, Any] = {
                'side': 'LONG',
                'entry_price': market_data['close'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'size': signal['position_size']
            }
            strategy.update_position(position)
            logger.info(f"Opened position: {position}")
        
        # Example market data update
        market_data['close'] = 51000.0
        market_data['rsi'] = 60.0
        market_data['volatility'] = 0.03
        
        # Generate new signals
        signal = strategy.generate_signals(market_data)
        logger.info(f"Updated signal: {signal}")
        
        # Example position update
        if signal['signal'] == "EXIT":
            logger.info(f"Exit signal generated: {signal.get('reason', 'Unknown reason')}")
            strategy.update_position({})  # Empty dict to clear position
        
        # Get trade history
        history = strategy.get_trade_history()
        logger.info(f"Trade history: {history}")
        
    except Exception as e:
        logger.error(f"Error in strategy example: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 