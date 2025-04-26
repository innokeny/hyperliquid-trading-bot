from typing import Dict, Optional, Union, List, Literal
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.types import Side, Cloid
from hyperliquid.utils.signing import OrderType, LimitOrderType, TriggerOrderType, Tpsl
from hyperliquid.utils.types import BuilderInfo
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
        self.open_orders: Dict[str, Dict] = {}
        
    def place_order(
        self,
        name: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        order_type: OrderType = OrderType({'limit': LimitOrderType(tif="Alo")}),
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
            if not name or sz <= 0:
                raise ValueError("Invalid order parameters")
            
            if hasattr(order_type, "tif") and limit_px <= 0:
                raise ValueError("Price is required for limit orders")
            
            order_params = {
                "name": name,
                "is_buy": is_buy,
                "sz": sz,
                "limit_px": limit_px,
                "order_type": order_type,
                "reduce_only": reduce_only,
                "cloid": cloid
            }
            
            response = self.exchange.order(**order_params)
            if response.get("status", 'failed') != 'ok':
                raise Exception(f"Failed to place order: {response.get('error')}")
            
            if "response" in response and "data" in response["response"]:
                statuses = response["response"]["data"].get("statuses", [])
                if statuses and "resting" in statuses[0]:
                    oid = statuses[0]["resting"]["oid"]
                    self.open_orders[str(oid)] = order_params
                    logger.info(f"Order placed successfully: {oid}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise
            
    def cancel_order(self, name: str, oid: int) -> Dict:
        """
        Cancel an existing order.
        
        Args:
            name: The coin of the order
            oid: The ID of the order to cancel
            
        Returns:
            Dict containing the cancellation response
        """
        try:
            response = self.exchange.cancel(name, oid)
            if not response.get("status", 'failed') == 'ok':
                raise Exception(f"Failed to cancel order: {response.get('error')}")
            
            if str(oid) in self.open_orders:
                del self.open_orders[str(oid)]
                logger.info(f"Order cancelled successfully: {oid}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            raise
            
    def modify_order(
        self,
        oid: int,
        name: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        order_type: OrderType = OrderType({'limit': LimitOrderType(tif="Alo")}),
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
            
            response = self.exchange.modify_order(**modify_params)
            if not response.get("status", 'failed') == 'ok':
                raise Exception(f"Failed to modify order: {response.get('error')}")
            
            if "oid" in response:
                self.open_orders[str(oid)].update(modify_params)
                logger.info(f"Order modified successfully: {oid}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error modifying order: {str(e)}")
            raise
            
    def get_open_orders(self, name: Optional[str] = None) -> List[Dict]:
        """
        Get all open orders, optionally filtered by coin.
        
        Args:
            name: Optional coin to filter orders by
            
        Returns:
            List of open orders
        """
        try:
            if not self.exchange.account_address:
                raise ValueError("Exchange account address not set")
                
            orders = self.info.open_orders(self.exchange.account_address)
            
            if name:
                orders = [order for order in orders if order["coin"] == name]
            
            return orders
            
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            raise
            
    def cancel_all_orders(self, name: Optional[str] = None) -> Dict:
        """
        Cancel all open orders, optionally filtered by coin.
        
        Args:
            name: Optional coin to filter orders by
            
        Returns:
            Dict containing the cancellation response
        """
        try:
            open_orders = self.get_open_orders(name)
            if len(open_orders) > 0:
                for order in open_orders:
                    self.cancel_order(order["coin"], order["oid"])
            
            return {"success": True, "message": "All orders cancelled successfully"}
            
        except Exception as e:
            logger.error(f"Error cancelling all orders: {str(e)}")
            raise 