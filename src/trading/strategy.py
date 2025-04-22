from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from ..ml.model_inference import ModelInference
from src.config.ml_config import MLConfig

logger = logging.getLogger(__name__)

class TradingStrategy:
    """Implements trading strategy with signal generation, position sizing, and risk management."""

    def __init__(self, model_inference: ModelInference, config: MLConfig):
        """
        Initialize the trading strategy.

        Args:
            model_inference: ModelInference instance for making predictions
            config: MLConfig instance containing strategy parameters
        """
        self.model_inference = model_inference
        self.config = config
        self.current_position = None
        self.trade_history = []
        self.entry_signals = []
        self.exit_signals = []
        self.trailing_stop_price = None
        self.highest_price = None
        self.lowest_price = None
        logger.info("Initialized trading strategy")

    def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on market data and model predictions.

        Args:
            market_data: Dictionary containing current market data

        Returns:
            Dictionary containing trading signals and metadata
        """
        try:
            # Convert market data to DataFrame
            df = self._prepare_market_data(market_data)
            
            # Get model prediction
            features = self.model_inference.prepare_features(df)
            prediction = self.model_inference.make_prediction(features)
            
            # Process prediction into trading signal
            signal = self._process_signal(prediction, market_data)
            
            # Apply risk management rules
            signal = self._apply_risk_management(signal, market_data)
            
            # Calculate position size
            signal = self._calculate_position_size(signal, market_data)
            
            # Generate entry/exit signals
            signal = self._generate_entry_exit_signals(signal, market_data)
            
            logger.debug(f"Generated trading signal: {signal['signal']}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise

    def _generate_entry_exit_signals(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate entry and exit signals based on market conditions."""
        try:
            # Initialize entry/exit conditions
            entry_conditions = []
            exit_conditions = []
            
            # Get current price and indicators
            current_price = market_data['close']
            rsi = market_data.get('rsi', 50)
            macd = market_data.get('macd', 0)
            macd_signal = market_data.get('signal', 0)
            volatility = market_data.get('volatility', 0)
            
            # Entry conditions for LONG positions
            if signal['signal'] == "BUY":
                # RSI oversold condition
                if rsi < 30:
                    entry_conditions.append("RSI oversold")
                
                # MACD crossover
                if macd > macd_signal and macd < 0:
                    entry_conditions.append("MACD bullish crossover")
                
                # Volatility breakout
                if volatility > self.config.volatility_threshold:
                    entry_conditions.append("Volatility breakout")
                
                # Price above moving average
                if current_price > market_data.get('sma_20', current_price):
                    entry_conditions.append("Price above SMA20")
            
            # Entry conditions for SHORT positions
            elif signal['signal'] == "SELL":
                # RSI overbought condition
                if rsi > 70:
                    entry_conditions.append("RSI overbought")
                
                # MACD crossover
                if macd < macd_signal and macd > 0:
                    entry_conditions.append("MACD bearish crossover")
                
                # Volatility breakout
                if volatility > self.config.volatility_threshold:
                    entry_conditions.append("Volatility breakout")
                
                # Price below moving average
                if current_price < market_data.get('sma_20', current_price):
                    entry_conditions.append("Price below SMA20")
            
            # Exit conditions for existing positions
            if self.current_position is not None:
                # Take profit condition
                if (self.current_position['side'] == "LONG" and current_price >= signal['take_profit']) or \
                   (self.current_position['side'] == "SHORT" and current_price <= signal['take_profit']):
                    exit_conditions.append("Take profit reached")
                
                # Stop loss condition
                if (self.current_position['side'] == "LONG" and current_price <= signal['stop_loss']) or \
                   (self.current_position['side'] == "SHORT" and current_price >= signal['stop_loss']):
                    exit_conditions.append("Stop loss triggered")
                
                # Trailing stop condition
                if self._check_trailing_stop(current_price):
                    exit_conditions.append("Trailing stop triggered")
                
                # RSI divergence
                if self._check_rsi_divergence(market_data, current_price):
                    exit_conditions.append("RSI divergence")
            
            # Update signal with entry/exit conditions
            signal['entry_conditions'] = entry_conditions
            signal['exit_conditions'] = exit_conditions
            
            # Determine final signal based on conditions
            if exit_conditions:
                signal['signal'] = "EXIT"
            elif not entry_conditions and signal['signal'] != "HOLD":
                signal['signal'] = "HOLD"
                signal['reason'] = "Entry conditions not met"
            
            return signal
        except Exception as e:
            logger.error(f"Error generating entry/exit signals: {str(e)}")
            raise

    def _check_trailing_stop(self, current_price: float) -> bool:
        """Check if trailing stop condition is met."""
        if self.current_position is None:
            return False
            
        # Calculate trailing stop price
        if self.current_position['side'] == "LONG":
            trailing_stop = self.current_position['entry_price'] * (1 + self.config.trailing_stop_percentage)
            return current_price <= trailing_stop
        else:
            trailing_stop = self.current_position['entry_price'] * (1 - self.config.trailing_stop_percentage)
            return current_price >= trailing_stop

    def _check_rsi_divergence(self, market_data: Dict[str, Any], current_price: float) -> bool:
        """Check for RSI divergence."""
        if self.current_position is None:
            return False
            
        current_rsi = market_data.get('rsi', 50)
        previous_rsi = market_data.get('previous_rsi', 50)
        
        if self.current_position['side'] == "LONG":
            # Bearish divergence
            return current_price > self.current_position['entry_price'] and current_rsi < previous_rsi
        else:
            # Bullish divergence
            return current_price < self.current_position['entry_price'] and current_rsi > previous_rsi

    def _prepare_market_data(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare market data for feature generation."""
        # Convert market data to DataFrame format
        df = pd.DataFrame([market_data])
        return df

    def _calculate_dynamic_stop_loss(self, entry_price: float, market_data: Dict[str, Any]) -> float:
        """Calculate dynamic stop loss based on market conditions."""
        try:
            # Base stop loss percentage
            base_stop_loss = entry_price * (1 - self.config.stop_loss_percentage)
            
            # Adjust for volatility
            volatility = market_data.get('volatility', 0)
            volatility_adjustment = volatility * self.config.volatility_multiplier
            stop_loss = base_stop_loss * (1 - volatility_adjustment)
            
            # Adjust for ATR if available
            atr = market_data.get('atr', 0)
            if atr > 0:
                atr_adjustment = atr * self.config.atr_multiplier
                stop_loss = min(stop_loss, entry_price - atr_adjustment)
            
            # Ensure minimum stop loss distance
            min_stop_distance = entry_price * self.config.min_stop_distance_percentage
            stop_loss = max(stop_loss, entry_price - min_stop_distance)
            
            return stop_loss
        except Exception as e:
            logger.error(f"Error calculating dynamic stop loss: {str(e)}")
            return entry_price * (1 - self.config.stop_loss_percentage)

    def _calculate_dynamic_take_profit(self, entry_price: float, market_data: Dict[str, Any]) -> float:
        """Calculate dynamic take profit based on market conditions."""
        try:
            # Base take profit percentage
            base_take_profit = entry_price * (1 + self.config.take_profit_percentage)
            
            # Adjust for volatility
            volatility = market_data.get('volatility', 0)
            volatility_adjustment = volatility * self.config.volatility_multiplier
            take_profit = base_take_profit * (1 + volatility_adjustment)
            
            # Adjust for ATR if available
            atr = market_data.get('atr', 0)
            if atr > 0:
                atr_adjustment = atr * self.config.atr_multiplier
                take_profit = max(take_profit, entry_price + atr_adjustment)
            
            # Ensure minimum take profit distance
            min_profit_distance = entry_price * self.config.min_profit_distance_percentage
            take_profit = min(take_profit, entry_price + min_profit_distance)
            
            return take_profit
        except Exception as e:
            logger.error(f"Error calculating dynamic take profit: {str(e)}")
            return entry_price * (1 + self.config.take_profit_percentage)

    def _update_trailing_stop(self, current_price: float, market_data: Dict[str, Any]) -> None:
        """Update trailing stop based on current price and market conditions."""
        try:
            if self.current_position is None:
                return
                
            # Update highest/lowest price
            if self.current_position['side'] == "LONG":
                if self.highest_price is None or current_price > self.highest_price:
                    self.highest_price = current_price
                    # Calculate new trailing stop
                    trailing_stop = self.highest_price * (1 - self.config.trailing_stop_percentage)
                    if self.trailing_stop_price is None or trailing_stop > self.trailing_stop_price:
                        self.trailing_stop_price = trailing_stop
            else:
                if self.lowest_price is None or current_price < self.lowest_price:
                    self.lowest_price = current_price
                    # Calculate new trailing stop
                    trailing_stop = self.lowest_price * (1 + self.config.trailing_stop_percentage)
                    if self.trailing_stop_price is None or trailing_stop < self.trailing_stop_price:
                        self.trailing_stop_price = trailing_stop
        except Exception as e:
            logger.error(f"Error updating trailing stop: {str(e)}")

    def _check_stop_conditions(self, current_price: float, market_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if any stop conditions are met."""
        try:
            if self.current_position is None:
                return False, None
                
            # Check stop loss
            if self.current_position['side'] == "LONG":
                if current_price <= self.current_position['stop_loss']:
                    return True, "Stop loss triggered"
                if self.trailing_stop_price and current_price <= self.trailing_stop_price:
                    return True, "Trailing stop triggered"
            else:
                if current_price >= self.current_position['stop_loss']:
                    return True, "Stop loss triggered"
                if self.trailing_stop_price and current_price >= self.trailing_stop_price:
                    return True, "Trailing stop triggered"
            
            # Check take profit
            if self.current_position['side'] == "LONG":
                if current_price >= self.current_position['take_profit']:
                    return True, "Take profit reached"
            else:
                if current_price <= self.current_position['take_profit']:
                    return True, "Take profit reached"
            
            # Check volatility stop
            volatility = market_data.get('volatility', 0)
            if volatility > self.config.max_volatility:
                return True, "Volatility stop triggered"
            
            return False, None
        except Exception as e:
            logger.error(f"Error checking stop conditions: {str(e)}")
            return False, None

    def _process_signal(self, prediction: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process model prediction into trading signal."""
        try:
            # Get raw prediction and confidence
            raw_prediction = prediction['prediction']
            confidence = prediction['confidence']
            
            # Map prediction to trading signal
            signal = self.config.signal_mapping.get(str(raw_prediction), "HOLD")
            
            # Calculate price targets based on current price
            current_price = market_data['close']
            stop_loss = self._calculate_dynamic_stop_loss(current_price, market_data)
            take_profit = self._calculate_dynamic_take_profit(current_price, market_data)
            
            # Update trailing stop if we have a position
            if self.current_position is not None:
                self._update_trailing_stop(current_price, market_data)
                # Check stop conditions
                should_exit, exit_reason = self._check_stop_conditions(current_price, market_data)
                if should_exit:
                    signal = "EXIT"
                    if exit_reason:
                        logger.info(f"Exit signal generated: {exit_reason}")
            
            return {
                'timestamp': datetime.now().isoformat(),
                'signal': signal,
                'confidence': confidence,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop': self.trailing_stop_price,
                'raw_prediction': raw_prediction,
                'metadata': prediction['metadata']
            }
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
            raise

    def _apply_risk_management(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply risk management rules to trading signal."""
        try:
            # Skip risk management for HOLD signals
            if signal['signal'] == "HOLD":
                return signal

            # Check confidence threshold
            if signal['confidence'] < self.config.min_confidence:
                signal['signal'] = "HOLD"
                signal['reason'] = "Low confidence"
                return signal

            # Check if we already have a position
            if self.current_position is not None:
                # Check if signal contradicts current position
                if (self.current_position['side'] == "LONG" and signal['signal'] == "SELL") or \
                   (self.current_position['side'] == "SHORT" and signal['signal'] == "BUY"):
                    signal['signal'] = "HOLD"
                    signal['reason'] = "Contradicts current position"
                    return signal

            # Check volatility
            volatility = market_data.get('volatility', 0)
            if volatility > self.config.max_volatility:
                signal['signal'] = "HOLD"
                signal['reason'] = "High volatility"
                return signal

            return signal
        except Exception as e:
            logger.error(f"Error applying risk management: {str(e)}")
            raise

    def _calculate_position_size(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position size based on risk parameters and market conditions."""
        try:
            if signal['signal'] == "HOLD":
                signal['position_size'] = 0
                return signal

            # Get available capital
            available_capital = market_data.get('available_capital', 0)
            
            # Calculate base position size
            base_size = available_capital * self.config.max_position_size
            
            # Adjust position size based on confidence
            confidence_factor = signal['confidence'] / self.config.prediction_threshold
            position_size = base_size * confidence_factor
            
            # Adjust for volatility
            volatility = market_data.get('volatility', 0)
            volatility_factor = 1 - (volatility / self.config.max_volatility)
            position_size = position_size * volatility_factor
            
            # Ensure position size is within limits
            position_size = min(position_size, available_capital)
            position_size = max(position_size, 0)
            
            signal['position_size'] = position_size
            return signal
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            raise

    def update_position(self, position: Dict[str, Any]):
        """Update current position information."""
        self.current_position = position
        self.trade_history.append({
            'timestamp': datetime.now().isoformat(),
            'position': position
        })
        # Reset trailing stop tracking
        self.trailing_stop_price = None
        self.highest_price = None
        self.lowest_price = None
        logger.info(f"Updated position: {position}")

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history."""
        return self.trade_history 