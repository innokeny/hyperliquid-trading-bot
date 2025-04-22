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

## Environment Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your Hyperliquid credentials and settings:

   ```env
   # Required Settings
   HYPERLIQUID_SECRET_KEY=your_private_key_here
   
   # Optional Settings
   HYPERLIQUID_ACCOUNT_ADDRESS=your_account_address_here
   HYPERLIQUID_API_URL=https://api.hyperliquid.xyz
   
   # Trading Settings
   TRADING_PAIR=BTC-PERP
   MAX_POSITION_SIZE=0.1
   LEVERAGE=1
   
   # Risk Management
   STOP_LOSS_PERCENTAGE=0.02
   TAKE_PROFIT_PERCENTAGE=0.04
   
   # Data Settings
   DATA_DIR=data
   CACHE_DIR=cache
   ```

3. Make sure to:
   - Keep your `.env` file secure and never commit it to version control
   - Use a strong private key for `HYPERLIQUID_SECRET_KEY`
   - Set appropriate risk management parameters based on your strategy
   - Create the data and cache directories if they don't exist

## Security Notes

- The `HYPERLIQUID_SECRET_KEY` is your private key. Keep it secure and never share it.
- If you don't provide `HYPERLIQUID_ACCOUNT_ADDRESS`, the bot will use the address derived from your private key.
- Make sure your `.env` file has proper file permissions (600) to prevent unauthorized access.
- Consider using a separate account for trading with limited funds.

## Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run tests:
   ```bash
   pytest
   ```