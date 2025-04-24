from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    HYPERLIQUID_SECRET_KEY: str
    HYPERLIQUID_ACCOUNT_ADDRESS: Optional[str] = None
    
    COIN: str = "BTC"
    MAX_POSITION_SIZE: float = 0.1
    LEVERAGE: int = 1
    STOP_LOSS_PERCENTAGE: float = 0.02
    TAKE_PROFIT_PERCENTAGE: float = 0.04
    
    DATA_DIR: Path = Path("data")
    CACHE_DIR: Path = Path("cache")
    
    MODEL_PATH: Optional[Path] = Path("models/model.pth")
    PREDICTION_THRESHOLD: float = 0.7
    
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

settings = Settings() 