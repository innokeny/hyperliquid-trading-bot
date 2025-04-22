from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Hyperliquid API settings
    HYPERLIQUID_API_KEY: str = Field(..., env="HYPERLIQUID_API_KEY")
    HYPERLIQUID_API_SECRET: str = Field(..., env="HYPERLIQUID_API_SECRET")
    HYPERLIQUID_API_URL: str = Field("https://api.hyperliquid.xyz", env="HYPERLIQUID_API_URL")
    
    # Trading settings
    TRADING_PAIR: str = Field("BTC-PERP", env="TRADING_PAIR")
    MAX_POSITION_SIZE: float = Field(0.1, env="MAX_POSITION_SIZE")  # in BTC
    LEVERAGE: int = Field(1, env="LEVERAGE")
    
    # Model settings
    MODEL_PATH: Path = Field(Path("models/model.pth"), env="MODEL_PATH")
    PREDICTION_THRESHOLD: float = Field(0.7, env="PREDICTION_THRESHOLD")
    
    # Risk management
    STOP_LOSS_PERCENTAGE: float = Field(0.02, env="STOP_LOSS_PERCENTAGE")
    TAKE_PROFIT_PERCENTAGE: float = Field(0.04, env="TAKE_PROFIT_PERCENTAGE")
    
    # Data settings
    DATA_DIR: Path = Field(Path("data"), env="DATA_DIR")
    CACHE_DIR: Path = Field(Path("cache"), env="CACHE_DIR")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()

# Create necessary directories
settings.DATA_DIR.mkdir(exist_ok=True)
settings.CACHE_DIR.mkdir(exist_ok=True) 