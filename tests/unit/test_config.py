import os
from pathlib import Path
import pytest
from pydantic import ValidationError
from src.settings import Settings

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to set up mock environment variables."""
    env_vars = {
        "HYPERLIQUID_SECRET_KEY": "test_private_key",
        "HYPERLIQUID_ACCOUNT_ADDRESS": "test_address",
        "HYPERLIQUID_API_URL": "https://test.api.hyperliquid.xyz",
        "TRADING_PAIR": "BTC-PERP",
        "MAX_POSITION_SIZE": "0.1",
        "LEVERAGE": "1",
        "STOP_LOSS_PERCENTAGE": "0.02",
        "TAKE_PROFIT_PERCENTAGE": "0.04",
        "DATA_DIR": "test_data",
        "CACHE_DIR": "test_cache"
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars

def test_settings_initialization(mock_env_vars):
    """Test that settings are correctly initialized from environment variables."""
    settings = Settings()
    
    assert settings.HYPERLIQUID_SECRET_KEY == mock_env_vars["HYPERLIQUID_SECRET_KEY"]
    assert settings.HYPERLIQUID_ACCOUNT_ADDRESS == mock_env_vars["HYPERLIQUID_ACCOUNT_ADDRESS"]
    assert settings.HYPERLIQUID_API_URL == mock_env_vars["HYPERLIQUID_API_URL"]
    assert settings.TRADING_PAIR == mock_env_vars["TRADING_PAIR"]
    assert settings.MAX_POSITION_SIZE == float(mock_env_vars["MAX_POSITION_SIZE"])
    assert settings.LEVERAGE == int(mock_env_vars["LEVERAGE"])
    assert settings.STOP_LOSS_PERCENTAGE == float(mock_env_vars["STOP_LOSS_PERCENTAGE"])
    assert settings.TAKE_PROFIT_PERCENTAGE == float(mock_env_vars["TAKE_PROFIT_PERCENTAGE"])
    assert str(settings.DATA_DIR) == mock_env_vars["DATA_DIR"]
    assert str(settings.CACHE_DIR) == mock_env_vars["CACHE_DIR"]


def test_directory_creation(mock_env_vars, tmp_path):
    """Test that data and cache directories are created."""
    # Override directory paths with temporary paths
    data_dir = tmp_path / "test_data"
    cache_dir = tmp_path / "test_cache"
    
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["CACHE_DIR"] = str(cache_dir)
    
    settings = Settings()
    settings.DATA_DIR.mkdir(exist_ok=True)
    settings.CACHE_DIR.mkdir(exist_ok=True)
    
    assert data_dir.exists()
    assert cache_dir.exists()
    assert data_dir.is_dir()
    assert cache_dir.is_dir()

def test_default_values():
    """Test that default values are used when environment variables are not set."""
    # Clear all environment variables
    for key in os.environ:
        if key.startswith(("HYPERLIQUID_", "TRADING_", "MAX_", "LEVERAGE_", "STOP_", "TAKE_", "DATA_", "CACHE_")):
            os.environ.pop(key)
    
    settings = Settings()
    
    # Test default values
    assert settings.HYPERLIQUID_API_URL == "https://api.hyperliquid-testnet.xyz"
    assert settings.TRADING_PAIR == "BTC-PERP"
    assert settings.MAX_POSITION_SIZE == 0.1
    assert settings.LEVERAGE == 1
    assert settings.STOP_LOSS_PERCENTAGE == 0.02
    assert settings.TAKE_PROFIT_PERCENTAGE == 0.04 