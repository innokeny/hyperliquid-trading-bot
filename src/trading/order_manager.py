from typing import Dict, Optional, Union, List, Literal
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.types import Side, Cloid
from hyperliquid.utils.signing import OrderType, LimitOrderType, TriggerOrderType
import logging

logger = logging.getLogger(__name__)

class OrderManager:
    """Manages order operations for the trading bot."""
    
    def __init__(self, exchange: Exchange, info: Info):
        """
        Initialize the OrderManager.
        
        Args:
            exchange: Hyperliquid Exchange instance
            info: Hyperliquid Info instance
        """
        self.exchange = exchange
        self.info = info
        self.open_orders: Dict[str, Dict] = {}  # Track open orders by order ID
        
    async def place_order(
        self,
        name: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        order_type: Union[LimitOrderType, TriggerOrderType] = LimitOrderType(tif="Alo"),
        reduce_only: bool = False,
        cloid: Optional[Cloid] = None
    ) -> Dict:
        """
        Place an order on Hyperliquid.
        
        Args:
            name: The coin to trade (e.g., "BTC")
            is_buy: True for buy orders, False for sell orders
            sz: Size of the order
            limit_px: Price for limit orders
            order_type: Type of order (limit or trigger)
            reduce_only: Whether the order is reduce-only
            cloid: Client order ID (optional)
            
        Returns:
            Dict containing the order response
        """
        try:
            # Validate order parameters
            if not name or sz <= 0:
                raise ValueError("Invalid order parameters")
            
            # For limit orders, price is required
            if hasattr(order_type, "tif") and limit_px <= 0:
                raise ValueError("Price is required for limit orders")
            
            # Prepare order parameters
            order_params = {
                "name": name,
                "is_buy": is_buy,
                "sz": sz,
                "limit_px": limit_px,
                "order_type": order_type,
                "reduce_only": reduce_only,
                "cloid": cloid
            }
            
            # Place the order
            response = await self.exchange.order(**order_params)
            
            if not response.get("success"):
                raise Exception(f"Failed to place order: {response.get('error')}")
            
            # Track the order if successful
            if "order_id" in response:
                self.open_orders[response["order_id"]] = order_params
                logger.info(f"Order placed successfully: {response['order_id']}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise
            
    async def cancel_order(self, name: str, oid: int) -> Dict:
        """
        Cancel an existing order.
        
        Args:
            name: The coin of the order
            oid: The ID of the order to cancel
            
        Returns:
            Dict containing the cancellation response
        """
        try:
            response = await self.exchange.cancel(name, oid)
            
            if not response.get("success"):
                raise Exception(f"Failed to cancel order: {response.get('error')}")
            
            # Remove from tracked orders if successful
            if str(oid) in self.open_orders:
                del self.open_orders[str(oid)]
                logger.info(f"Order cancelled successfully: {oid}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            raise
            
    async def modify_order(
        self,
        oid: int,
        name: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        order_type: Union[LimitOrderType, TriggerOrderType] = LimitOrderType(tif="Alo"),
        reduce_only: bool = False,
        cloid: Optional[Cloid] = None
    ) -> Dict:
        """
        Modify an existing order.
        
        Args:
            oid: The ID of the order to modify
            name: The coin of the order
            is_buy: True for buy orders, False for sell orders
            sz: New size for the order
            limit_px: New price for the order
            order_type: Type of order (limit or trigger)
            reduce_only: Whether the order is reduce-only
            cloid: Client order ID (optional)
            
        Returns:
            Dict containing the modification response
        """
        try:
            if str(oid) not in self.open_orders:
                raise ValueError(f"Order {oid} not found")
            
            # Prepare modification parameters
            modify_params = {
                "oid": oid,
                "name": name,
                "is_buy": is_buy,
                "sz": sz,
                "limit_px": limit_px,
                "order_type": order_type,
                "reduce_only": reduce_only,
                "cloid": cloid
            }
            
            response = await self.exchange.modify_order(**modify_params)
            
            if not response.get("success"):
                raise Exception(f"Failed to modify order: {response.get('error')}")
            
            # Update tracked order if successful
            if "order_id" in response:
                self.open_orders[str(oid)].update(modify_params)
                logger.info(f"Order modified successfully: {oid}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error modifying order: {str(e)}")
            raise
            
    async def get_open_orders(self, name: Optional[str] = None) -> List[Dict]:
        """
        Get all open orders, optionally filtered by coin.
        
        Args:
            name: Optional coin to filter orders by
            
        Returns:
            List of open orders
        """
        try:
            orders = await self.info.open_orders(self.exchange.wallet.address)
            
            if name:
                orders = [order for order in orders if order["name"] == name]
            
            return orders
            
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            raise
            
    async def cancel_all_orders(self, name: Optional[str] = None) -> Dict:
        """
        Cancel all open orders, optionally filtered by coin.
        
        Args:
            name: Optional coin to filter orders by
            
        Returns:
            Dict containing the cancellation response
        """
        try:
            open_orders = await self.get_open_orders(name)
            
            for order in open_orders:
                await self.cancel_order(order["name"], order["order_id"])
            
            return {"success": True, "message": "All orders cancelled successfully"}
            
        except Exception as e:
            logger.error(f"Error cancelling all orders: {str(e)}")
            raise 