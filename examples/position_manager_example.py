import asyncio
from loguru import logger
from src.trading.connection import HyperliquidConnection
from src.trading.position_manager import PositionManager

# Configure loguru logger
logger.add(
    "position_manager.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

async def main():
    try:
        # Initialize connection
        connection = HyperliquidConnection()
        address, info, exchange = connection.setup()
        
        # Initialize PositionManager
        position_manager = PositionManager(info, exchange)
        
        # Update and get all positions
        positions = position_manager.update_positions()
        logger.info(f"Current positions: {positions}")
        
        # Example: Get position for a specific coin
        coin = "BTC"  # Example coin
        position = position_manager.get_position(coin)
        if position:
            logger.info(f"Position for {coin}:")
            logger.info(f"  Size: {position.size}")
            logger.info(f"  Entry Price: {position.entry_price}")
            logger.info(f"  Leverage: {position.leverage}")
            logger.info(f"  Unrealized PnL: {position.unrealized_pnl}")
            logger.info(f"  Realized PnL: {position.realized_pnl}")
            logger.info(f"  Liquidation Price: {position.liquidation_price}")
        else:
            logger.info(f"No position found for {coin}")
        
        # Example: Get position size
        position_size = position_manager.get_position_size(coin)
        logger.info(f"Position size for {coin}: {position_size}")
        
        # Example: Get position value
        position_value = position_manager.get_position_value(coin)
        logger.info(f"Position value for {coin}: {position_value}")
        
        # Example: Get position PnL
        pnl = position_manager.get_position_pnl(coin)
        logger.info(f"PnL for {coin}: {pnl}")
        
        # Example: Get position risk metrics
        risk_metrics = position_manager.get_position_risk(coin)
        logger.info(f"Risk metrics for {coin}: {risk_metrics}")
        
        # Example: Get position history
        history = position_manager.get_position_history(coin=coin)
        logger.info(f"Position history for {coin}: {len(history)} entries")
        
    except Exception as e:
        logger.error(f"Error in position manager example: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 