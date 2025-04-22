from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class MLConfig(BaseModel):
    """Configuration for ML model settings."""
    
    # API Settings
    api_url: str = Field(..., description="Hyperliquid API URL")
    account_address: str = Field(..., description="Account address for trading")
    secret_key: str = Field(..., description="Secret key for API authentication")
    
    # Trading Settings
    trading_interval: int = Field(
        default=60,
        description="Interval between trading iterations in seconds"
    )
    
    # Model settings
    model_path: str = Field(..., description="Path to the trained model file")
    model_type: str = Field(..., description="Type of the ML model (e.g., 'regression', 'classification')")
    model_version: str = Field(..., description="Version of the ML model")
    
    # Feature settings
    feature_columns: List[str] = Field(
        default=[
            'sma_20', 'sma_50', 'ema_20', 'rsi', 'macd', 'signal',
            'bb_middle', 'bb_upper', 'bb_lower', 'atr',
            'price_change', 'price_change_1d', 'price_change_1w',
            'volume_ma', 'volume_std', 'volume_change',
            'volatility', 'price_ma_ratio'
        ],
        description="List of feature columns used by the model"
    )
    
    # Prediction settings
    prediction_threshold: float = Field(
        default=0.7,
        description="Confidence threshold for accepting predictions"
    )
    min_confidence: float = Field(
        default=0.6,
        description="Minimum confidence required for trading signals"
    )
    
    # Signal processing settings
    signal_mapping: Dict[str, str] = Field(
        default={
            "0": "SELL",
            "1": "HOLD",
            "2": "BUY"
        },
        description="Mapping of model predictions to trading signals"
    )
    
    # Risk management settings
    max_position_size: float = Field(
        default=0.1,
        description="Maximum position size as a fraction of total capital"
    )
    stop_loss_percentage: float = Field(
        default=0.02,
        description="Stop loss percentage for positions"
    )
    take_profit_percentage: float = Field(
        default=0.04,
        description="Take profit percentage for positions"
    )
    
    # Performance monitoring settings
    performance_window: int = Field(
        default=100,
        description="Number of predictions to consider for performance monitoring"
    )
    min_accuracy: float = Field(
        default=0.55,
        description="Minimum accuracy required to continue using the model"
    )
    
    class Config:
        """Pydantic model configuration."""
        env_prefix = "ML_"
        case_sensitive = False 