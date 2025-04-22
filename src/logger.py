import sys
from pathlib import Path
from loguru import logger

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging
logger.remove()  # Remove default handler

# Add console handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Add file handler
logger.add(
    log_dir / "trading_bot_{time}.log",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
)

# Export logger
__all__ = ["logger"] 