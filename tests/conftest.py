import os
import sys
import pytest
from pathlib import Path
from loguru import logger

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Configure logging for tests
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="{message}",
        level="WARNING",  # Only show warnings and errors during tests
    )
    
    # Create test directories
    test_dirs = ["data", "cache", "logs", "models"]
    for dir_name in test_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    yield
    
    # Cleanup after each test
    for dir_name in test_dirs:
        test_dir = Path(dir_name)
        if test_dir.exists():
            for file in test_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            test_dir.rmdir()

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to set up mock environment variables."""
    env_vars = {
        "HYPERLIQUID_API_KEY": "test_api_key",
        "HYPERLIQUID_API_SECRET": "test_api_secret",
        "HYPERLIQUID_API_URL": "https://test.api.hyperliquid.xyz",
        "TRADING_PAIR": "BTC-PERP",
        "MAX_POSITION_SIZE": "0.1",
        "LEVERAGE": "1",
        "MODEL_PATH": "models/test_model.pth",
        "PREDICTION_THRESHOLD": "0.7",
        "STOP_LOSS_PERCENTAGE": "0.02",
        "TAKE_PROFIT_PERCENTAGE": "0.04",
        "DATA_DIR": "data",
        "CACHE_DIR": "cache"
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars 