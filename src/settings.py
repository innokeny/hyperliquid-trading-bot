from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Hyperliquid API settings
    HYPERLIQUID_SECRET_KEY: str
    HYPERLIQUID_ACCOUNT_ADDRESS: Optional[str] = None
    
    # Trading parameters
    COIN: str = "BTC"
    MAX_POSITION_SIZE: float = 0.1
    LEVERAGE: int = 1
    
    # Strategy parameters
    STOP_LOSS_PERCENTAGE: float = 0.02
    TAKE_PROFIT_PERCENTAGE: float = 0.04
    TRAILING_STOP_PERCENTAGE: float = 0.01
    MIN_CONFIDENCE: float = 0.7
    
    # Volatility parameters
    VOLATILITY_THRESHOLD: float = 0.02
    MAX_VOLATILITY: float = 0.05
    VOLATILITY_MULTIPLIER: float = 1.5
    
    # ATR parameters
    ATR_MULTIPLIER: float = 2.0
    
    # Position sizing parameters
    MIN_STOP_DISTANCE_PERCENTAGE: float = 0.01
    MIN_PROFIT_DISTANCE_PERCENTAGE: float = 0.02
    PREDICTION_THRESHOLD: float = 0.8
    BASE_POSITION_SIZE: float = 0.1
    MIN_POSITION_SIZE: float = 0.01
    RISK_REWARD_RATIO: float = 2.0
    
    # RSI parameters
    RSI_OVERSOLD: int = 30
    RSI_OVERBOUGHT: int = 70
    
    # Signal mapping
    SIGNAL_MAPPING: Dict[str, str] = {
        "0": "HOLD",
        "1": "BUY",
        "2": "SELL"
    }
    
    # Data and model paths
    DATA_DIR: Path = Path("data")
    CACHE_DIR: Path = Path("cache")
    MODEL_PATH: Optional[Path] = Path("models/model.pth")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore" 
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DATA_DIR.mkdir(exist_ok=True)
        self.CACHE_DIR.mkdir(exist_ok=True)
    
    def reload(self) -> None:
        """Reload settings from environment variables."""
        load_dotenv()
        self.__init__()
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get strategy configuration as a dictionary."""
        return {
            # Signal mapping
            "signal_mapping": self.SIGNAL_MAPPING,
            
            # Risk management
            "stop_loss_percentage": self.STOP_LOSS_PERCENTAGE,
            "take_profit_percentage": self.TAKE_PROFIT_PERCENTAGE,
            "trailing_stop_percentage": self.TRAILING_STOP_PERCENTAGE,
            
            # Position sizing
            "max_position_size": self.MAX_POSITION_SIZE,
            "base_position_size": self.BASE_POSITION_SIZE,
            "min_position_size": self.MIN_POSITION_SIZE,
            "risk_reward_ratio": self.RISK_REWARD_RATIO,
            
            # Volatility parameters
            "volatility_threshold": self.VOLATILITY_THRESHOLD,
            "max_volatility": self.MAX_VOLATILITY,
            "volatility_multiplier": self.VOLATILITY_MULTIPLIER,
            
            # ATR parameters
            "atr_multiplier": self.ATR_MULTIPLIER,
            
            # Distance parameters
            "min_stop_distance_percentage": self.MIN_STOP_DISTANCE_PERCENTAGE,
            "min_profit_distance_percentage": self.MIN_PROFIT_DISTANCE_PERCENTAGE,
            
            # Confidence and prediction
            "min_confidence": self.MIN_CONFIDENCE,
            "prediction_threshold": self.PREDICTION_THRESHOLD,
            
            # RSI parameters
            "rsi_oversold": self.RSI_OVERSOLD,
            "rsi_overbought": self.RSI_OVERBOUGHT
        }

settings = Settings() 