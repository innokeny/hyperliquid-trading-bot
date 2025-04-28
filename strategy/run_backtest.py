import argparse
import logging
import os
import pandas as pd
from datetime import datetime
from donchian_channels_strategy import DonchianChannelsStrategy
from donchian_channels_ml_strategy import DonchianChannelsMLStrategy
from backtest import Backtest
import numpy as np
from strategy import Strategy

def setup_logging(log_file):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def generate_unique_filename(data_path, strategy_type, output_dir="strategy/backtest_logs"):
    """Generate unique filenames for results and logs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    results_file = f"{output_dir}/{data_name}_{strategy_type}_{timestamp}_results.csv"
    log_file = f"{output_dir}/{data_name}_{strategy_type}_{timestamp}.log"
    return results_file, log_file

def main():
    parser = argparse.ArgumentParser(description='Run backtest with specified strategy')
    parser.add_argument('--data_path', type=str, default='data/processed/btc_1h_2024-01-01_2024-02-01.csv',
                      help='Path to the data file')
    parser.add_argument('--strategy_type', type=str, choices=['basic', 'ml'], default='basic',
                      help='Type of strategy to use')
    parser.add_argument('--output_dir', type=str, default='strategy/backtest_logs',
                      help='Directory to save results and logs')
    parser.add_argument('--train_test_split', type=float, default=0.8,
                      help='Proportion of data used for training (default: 0.8)')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model file (required for ML strategy)')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate unique filenames
    results_file, log_file = generate_unique_filename(args.data_path, args.strategy_type, args.output_dir)
    
    # Set up logging
    logger = setup_logging(log_file)
    logger.info(f"Starting backtest with {args.strategy_type} strategy")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Train/test split: {args.train_test_split}")
    
    if args.strategy_type == 'ml':
        logger.info(f"Using model from: {args.model_path}")

    # Check if data file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return

    try:
        # Load data
        data = pd.read_csv(args.data_path)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Create strategy instance
        if args.strategy_type == 'ml':
            strategy = DonchianChannelsMLStrategy(
                ml_enabled=True,
                train_test_split=args.train_test_split,
                model_path=args.model_path
            )
        else:
            strategy = DonchianChannelsStrategy()

        # Run backtest
        results = strategy.run(data)
        
        # Calculate metrics
        # Check if we have the new format or the old format
        if 'returns' in results.columns:
            # New format
            returns = results['returns']
            asset_returns = data['close'].pct_change()
        else:
            # Old format
            returns = results['strategy_return_Combo']
            asset_returns = results['asset_return']
            
        cagr = (1 + returns.mean()) ** 252 - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = cagr / annual_vol if annual_vol != 0 else 0
        
        # Calculate Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = cagr / downside_vol if downside_vol != 0 else 0
        
        # Calculate max drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calculate Calmar ratio
        calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
        
        # Calculate cumulative returns
        strategy_return = (1 + returns).prod() - 1
        buy_hold_return = (1 + asset_returns).prod() - 1
        
        # Calculate win rate and gain/loss ratio
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
        gain_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0

        # Log results
        logger.info("\nBacktest Results:")
        logger.info(f"CAGR: {cagr:.2%}")
        logger.info(f"Annual Volatility: {annual_vol:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Sortino Ratio: {sortino:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Calmar Ratio: {calmar:.2f}")
        logger.info(f"Strategy Return: {strategy_return:.2%}")
        logger.info(f"Buy & Hold Return: {buy_hold_return:.2%}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Gain/Loss Ratio: {gain_loss_ratio:.2f}")

        # Save results
        results.to_csv(results_file, index=False)
        logger.info(f"Results saved to: {results_file}")

    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 