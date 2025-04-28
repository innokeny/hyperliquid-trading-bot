# Hyperliquid Trading Bot

A machine learning-based trading bot for the Hyperliquid exchange, using PyTorch and CatBoost for predictions. The bot implements real-time market data streaming, ML-based signal generation, and automated order execution with risk management.

## Project Structure

```
.
├── src/                    # Source code
│   ├── trading/           # Trading components
│   │   ├── connection.py  # Hyperliquid API connection
│   │   ├── market_data.py # Market data streaming
│   │   ├── strategy.py    # Trading strategy implementation
│   │   └── order_manager.py # Order management
│   ├── data/              # Data collection and processing
│   │   └── market_data.py # Market data collector
│   ├── ml/                # Machine learning components
│   │   └── model_inference.py # Model inference
│   └── settings.py        # Configuration settings
├── data/                  # Data storage
├── models/                # ML models
├── logs/                  # Log files
├── cache/                 # Cache files
├── tests/                 # Test files
├── notebooks/             # Jupyter notebooks
├── docs/                  # Documentation
└── scripts/               # Utility scripts
```

## Features

- Real-time market data streaming from Hyperliquid
- ML-based trading signal generation
- Automated order execution
- Risk management with stop-loss and take-profit
- Position tracking and management
- Comprehensive logging

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd hyperliquid-trading-bot
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy environment variables:
```bash
cp .env.example .env
```

5. Edit `.env` file with your Hyperliquid API credentials and other settings.

## Configuration

The bot can be configured through environment variables in the `.env` file:

```env
# Required Settings
HYPERLIQUID_SECRET_KEY=your_private_key_here

# Optional Settings
HYPERLIQUID_ACCOUNT_ADDRESS=your_account_address_here
HYPERLIQUID_API_URL=https://api.hyperliquid.xyz

# Trading Settings
COIN=BTC-PERP
MAX_POSITION_SIZE=0.1
LEVERAGE=1

# Risk Management
STOP_LOSS_PERCENTAGE=0.02
TAKE_PROFIT_PERCENTAGE=0.04

# Model Settings
MODEL_PATH=models/your_model.pt

# Data Settings
DATA_DIR=data
CACHE_DIR=cache
```

## Usage

1. Start the trading bot:
```bash
python3 main.py
```

2. Monitor logs in the `logs/` directory.

## Development

- Code style is enforced using Black and Ruff
- Type checking is done using mypy
- Tests can be run using pytest

## Security Notes

- The `HYPERLIQUID_SECRET_KEY` is your private key. Keep it secure and never share it.
- If you don't provide `HYPERLIQUID_ACCOUNT_ADDRESS`, the bot will use the address derived from your private key.
- Make sure your `.env` file has proper file permissions (600) to prevent unauthorized access.
- Consider using a separate account for trading with limited funds.

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]