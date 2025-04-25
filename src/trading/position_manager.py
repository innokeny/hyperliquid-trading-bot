from typing import Dict, Optional, List, Union
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from loguru import logger
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Position:
    """Represents a trading position."""
    coin: str
    size: float  # Positive for long, negative for short
    entry_price: float
    leverage: float
    unrealized_pnl: float
    realized_pnl: float
    last_updated: datetime
    liquidation_price: Optional[float] = None
    margin_used: Optional[float] = None
    margin_available: Optional[float] = None

class PositionManager:
    """Manages position tracking and operations."""
    
    def __init__(self, info: Info, exchange: Exchange):
        """
        Initialize the PositionManager.
        
        Args:
            info: Hyperliquid Info instance
            exchange: Hyperliquid Exchange instance
        """
        self.info = info
        self.exchange = exchange
        self.positions: Dict[str, Position] = {}  # Track positions by coin
        self.position_history: List[Dict] = []  # Track position history
        
    def update_positions(self) -> Dict[str, Position]:
        """
        Update all positions from the exchange.
        
        Returns:
            Dict of current positions by coin
        """
        try:
            # Get user state which includes positions
            user_state = self.info.user_state(self.exchange.wallet.address)
            
            # Clear existing positions
            self.positions.clear()
            
            # Update positions from user state
            for position_data in user_state.get("assetPositions", []):
                position = position_data["position"]
                coin = position["coin"]
                
                # Calculate position metrics
                size = float(position["szi"])
                entry_price = float(position["entryPx"])
                leverage = float(position["leverage"])
                unrealized_pnl = float(position["unrealizedPnl"])
                realized_pnl = float(position["realizedPnl"])
                
                # Calculate liquidation price if possible
                liquidation_price = None
                if size != 0:
                    # Simplified liquidation price calculation
                    # This should be adjusted based on your risk parameters
                    liquidation_price = entry_price * (1 - (1 / leverage))
                
                # Create Position object
                self.positions[coin] = Position(
                    coin=coin,
                    size=size,
                    entry_price=entry_price,
                    leverage=leverage,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=realized_pnl,
                    liquidation_price=liquidation_price,
                    last_updated=datetime.now()
                )
                
                # Add to position history
                self.position_history.append({
                    "timestamp": datetime.now(),
                    "coin": coin,
                    "size": size,
                    "entry_price": entry_price,
                    "leverage": leverage,
                    "unrealized_pnl": unrealized_pnl,
                    "realized_pnl": realized_pnl
                })
            
            logger.info(f"Updated positions: {self.positions}")
            return self.positions
            
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
            raise
            
    def get_position(self, coin: str) -> Optional[Position]:
        """
        Get position for a specific coin.
        
        Args:
            coin: The coin to get position for
            
        Returns:
            Position object if exists, None otherwise
        """
        try:
            # Update positions first to ensure we have latest data
            self.update_positions()
            return self.positions.get(coin)
            
        except Exception as e:
            logger.error(f"Error getting position for {coin}: {str(e)}")
            raise
            
    def get_position_size(self, coin: str) -> float:
        """
        Get the current position size for a coin.
        
        Args:
            coin: The coin to get position size for
            
        Returns:
            Position size (positive for long, negative for short)
        """
        try:
            position = self.get_position(coin)
            return position.size if position else 0.0
            
        except Exception as e:
            logger.error(f"Error getting position size for {coin}: {str(e)}")
            raise
            
    def get_position_value(self, coin: str) -> float:
        """
        Get the current position value for a coin.
        
        Args:
            coin: The coin to get position value for
            
        Returns:
            Position value in base currency
        """
        try:
            position = self.get_position(coin)
            if not position:
                return 0.0
                
            # Get current market price
            market_data = self.info.all_mids()
            current_price = float(market_data[coin])
            
            return abs(position.size * current_price)
            
        except Exception as e:
            logger.error(f"Error getting position value for {coin}: {str(e)}")
            raise
            
    def get_position_pnl(self, coin: str) -> Dict[str, float]:
        """
        Get the PnL for a position.
        
        Args:
            coin: The coin to get PnL for
            
        Returns:
            Dict containing unrealized and realized PnL
        """
        try:
            position = self.get_position(coin)
            if not position:
                return {"unrealized": 0.0, "realized": 0.0}
                
            return {
                "unrealized": position.unrealized_pnl,
                "realized": position.realized_pnl
            }
            
        except Exception as e:
            logger.error(f"Error getting PnL for {coin}: {str(e)}")
            raise
            
    def get_position_risk(self, coin: str) -> Dict[str, Optional[float]]:
        """
        Get risk metrics for a position.
        
        Args:
            coin: The coin to get risk metrics for
            
        Returns:
            Dict containing risk metrics
        """
        try:
            position = self.get_position(coin)
            if not position:
                return {
                    "liquidation_price": None,
                    "margin_used": None,
                    "margin_available": None
                }
                
            return {
                "liquidation_price": position.liquidation_price,
                "margin_used": position.margin_used,
                "margin_available": position.margin_available
            }
            
        except Exception as e:
            logger.error(f"Error getting risk metrics for {coin}: {str(e)}")
            raise
            
    def get_position_history(
        self,
        coin: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get position history with optional filtering.
        
        Args:
            coin: Optional coin to filter by
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            
        Returns:
            List of position history entries
        """
        try:
            filtered_history = self.position_history
            
            if coin:
                filtered_history = [entry for entry in filtered_history if entry["coin"] == coin]
                
            if start_time:
                filtered_history = [entry for entry in filtered_history if entry["timestamp"] >= start_time]
                
            if end_time:
                filtered_history = [entry for entry in filtered_history if entry["timestamp"] <= end_time]
                
            return filtered_history
            
        except Exception as e:
            logger.error(f"Error getting position history: {str(e)}")
            raise 