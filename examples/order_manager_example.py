import asyncio
import time
from loguru import logger
from src.trading.order_manager import OrderManager
from src.trading.connection import HyperliquidConnection
from hyperliquid.utils.constants import TESTNET_API_URL
from hyperliquid.utils.types import Side, Cloid
from hyperliquid.utils.signing import OrderType, LimitOrderType, TriggerOrderType, Tpsl

async def main():
    """Example usage of OrderManager with live market data."""
    # Initialize connection
    connection = HyperliquidConnection()
    _, info, exchange = connection.setup(base_url=TESTNET_API_URL)
    
    # Initialize order manager
    order_manager = OrderManager(exchange, info)
    
    try:
        # Example 1: Place a limit order
        logger.info("Placing a limit order...")
        limit_order = order_manager.place_order(
            name="BTC",
            is_buy=True,
            sz=0.001,  # 0.001 BTC
            limit_px=50000.0,  # $50,000
            order_type=OrderType(limit=LimitOrderType(tif="Alo")),  # Good till cancelled
            reduce_only=False
        )
        logger.info(f"Limit order placed: {limit_order}")
        
        # Wait for 5 seconds
        await asyncio.sleep(10)
        
        # Example 2: Get open orders
        logger.info("Getting open orders...")
        open_orders = order_manager.get_open_orders("BTC")
        logger.info(f"Open orders: {open_orders}")
        
        # Example 3: Modify an order
        if open_orders:
            logger.info("Modifying the first open order...")
            modified_order = order_manager.modify_order(
                oid=open_orders[0]["oid"],
                name="BTC",
                is_buy=True,
                sz=0.002,  # Increase size to 0.002 BTC
                limit_px=49000.0,  # Lower price to $49,000
                order_type=OrderType(limit=LimitOrderType(tif="Alo"))
            )
            logger.info(f"Order modified: {modified_order}")
        
        # Wait for 5 seconds
        await asyncio.sleep(5)
        
        # Example 4: Cancel all orders
        logger.info("Cancelling all orders...")
        cancel_result = order_manager.cancel_all_orders("BTC")
        logger.info(f"Cancel result: {cancel_result}")
        
        # Example 5: Place a stop loss order
        logger.info("Placing a stop loss order...")
        stop_loss_order = order_manager.place_order(
            name="BTC",
            is_buy=False,
            sz=0.001,
            limit_px=48000.0,
            order_type=OrderType(trigger=TriggerOrderType(triggerPx=48500.0, isMarket=True, tpsl="sl")),
            reduce_only=True
        )
        logger.info(f"Stop loss order placed: {stop_loss_order}")
        
        # Wait for 5 seconds
        await asyncio.sleep(5)
        
        # Example 6: Cancel specific order
        if stop_loss_order.get("order_id"):
            logger.info("Cancelling stop loss order...")
            cancel_result = order_manager.cancel_order("BTC", stop_loss_order["order_id"])
            logger.info(f"Cancel result: {cancel_result}")
        

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        # Cancel any remaining orders
        try:
            order_manager.cancel_all_orders("BTC")
        except Exception as e:
            logger.error(f"Error cancelling remaining orders: {str(e)}")
        logger.info("Order manager example completed")

if __name__ == "__main__":
    asyncio.run(main()) 