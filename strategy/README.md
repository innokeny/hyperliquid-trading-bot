# Trading Strategies

This directory contains various trading strategies for cryptocurrency markets, with a focus on the Donchian Channels strategy enhanced with machine learning capabilities.

## Files Overview

### Core Strategy Files

- `donchian_channels_ml_strategy.py`: Main strategy implementation combining Donchian Channels with ML predictions
- `donchian_channels_strategy.py`: Basic Donchian Channels strategy without ML
- `strategy.py`: Base strategy class with common functionality

### Backtesting and Training

- `run_backtest.py`: Script for running backtests on historical data
- `ml_model_trainer.py`: Script for training ML models on historical data
- `run_all_models_backtest.py`: Script for running backtests across all trained models

## Strategy Components

### Donchian Channels ML Strategy

The main strategy combines traditional Donchian Channels with machine learning predictions:

1. **Technical Indicators**:
   - RSI (14 periods)
   - Momentum (10 and 30 days)
   - Stochastic Oscillator
   - MACD
   - ADX
   - Moving Averages (SMA 50, EMA 20)
   - ATR
   - Bollinger Bands
   - Volatility measures
   - OBV (On-Balance Volume)

2. **ML Features**:
   - All technical indicators are used as features
   - Features are normalized using StandardScaler
   - Feature selection based on correlation and mutual information

3. **Model Training**:
   - Uses LightGBM classifier
   - Hyperparameter optimization with Optuna
   - Time series cross-validation
   - Target variable based on ATR-based profit targets

4. **Trading Logic**:
   - Entry signals from Donchian Channel breakouts
   - ML model filters potential trades
   - Position sizing based on ATR
   - Multiple timeframe analysis (5d to 360d)

## Usage

### Training Models

```bash
python ml_model_trainer.py --data_path <path_to_data> --output_dir models --train_test_split 0.8
```
```
python3 stategy/ml_model_trainer.py --data_path data/1d_timeframe/binance/binance_ETH_USDT_1d_20150427_20250427.csv --output_dir models --train_test_split 0.8 --n_optuna_trials 200 --n_cv_splits 5 --feature_selection_threshold 0.001 --high_correlation_threshold 0.95
```

### Running Backtest

```bash
python run_backtest.py --data_path <path_to_data> --strategy_type ml --model_path <path_to_model>
```

```
python3 stategy/run_backtest.py --data_path data/1d_timeframe/binance/binance_ETH_USDT_1d_20150427_20250427.csv --strategy_type ml --model_path models/model_5d.joblib --train_test_split 0.8 --output_dir backtest_logs
```

```
python3 run_backtest.py --data_path ../data/1d_timeframe/binance/binance_ETH_USDT=USDT_1d_20150427_20250427.csv --strategy basic --output_path backtest_logs/binance_eth_1d_basic.csv
```

### Running All Models Backtest

```bash
python run_all_models_backtest.py --data_path <path_to_data>
```

```
python3 stategy/run_all_models_backtest.py
```

## Performance Metrics

The strategy calculates and reports various performance metrics:

- CAGR (Compound Annual Growth Rate)
- Annualized Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Win Rate
- Gain/Loss Ratio

## Data Requirements

The strategy expects data in the following format:
- OHLCV data (Open, High, Low, Close, Volume)
- Daily timeframe
- Clean data without missing values
- Sufficient historical data for feature calculation

## Dependencies

- pandas
- numpy
- pandas_ta
- lightgbm
- scikit-learn
- optuna
- joblib

## Notes

- The strategy is designed for cryptocurrency markets but can be adapted for other markets
- ML models are trained separately for each timeframe
- The strategy includes risk management through position sizing
- Performance metrics are calculated both for the strategy and buy-and-hold approach 