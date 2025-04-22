# ML Trading Bot

A machine learning-based trading bot for the Hyperliquid exchange, using PyTorch and CatBoost for predictions.

## Project Structure

```
.
├── src/                    # Source code
├── data/                   # Data storage
├── models/                 # ML models
├── logs/                   # Log files
├── cache/                  # Cache files
├── tests/                  # Test files
├── notebooks/              # Jupyter notebooks
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-trading-bot
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

- `HYPERLIQUID_API_KEY`: Your Hyperliquid API key
- `HYPERLIQUID_API_SECRET`: Your Hyperliquid API secret
- `TRADING_PAIR`: Trading pair to trade (default: BTC-PERP)
- `MAX_POSITION_SIZE`: Maximum position size in BTC
- `LEVERAGE`: Trading leverage
- `MODEL_PATH`: Path to your ML model
- `PREDICTION_THRESHOLD`: Confidence threshold for trading signals
- `STOP_LOSS_PERCENTAGE`: Stop loss percentage
- `TAKE_PROFIT_PERCENTAGE`: Take profit percentage

## Usage

1. Start the trading bot:
```bash
python src/main.py
```

2. Monitor logs in the `logs/` directory.

## Development

- Code style is enforced using Black and Ruff
- Type checking is done using mypy
- Tests can be run using pytest

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]