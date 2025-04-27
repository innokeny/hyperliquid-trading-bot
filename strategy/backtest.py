import pandas as pd
import numpy as np
import argparse
from strategy import Strategy
from donchian_channels_strategy import DonchianChannelsStrategy
from donchian_channels_ml_strategy import DonchianChannelsMLStrategy
from datetime import datetime
import logging

class Backtest:
    """
    Class for running backtests on trading strategies.
    """
    def __init__(self, strategy):
        """
        Initialize Backtest class
        
        Args:
            strategy: Strategy object that implements the run() method
        """
        self.strategy = strategy
        self.data = None
        self.results = None
        self.metrics = {}
        
    def load_data(self, data_path):
        """
        Load data from CSV file
        
        Args:
            data_path: Path to CSV file with OHLCV data
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            self.data = pd.read_csv(data_path)
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data.set_index('timestamp', inplace=True)
            logging.info(f"Data loaded successfully from {data_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load data: {str(e)}")
            return False
            
    def run(self):
        """
        Run backtest using the provided strategy
        
        Returns:
            bool: True if backtest completed successfully, False otherwise
        """
        try:
            if self.data is None:
                logging.error("No data loaded. Please load data first.")
                return False
                
            self.results = self.strategy.run(self.data)
            self.calculate_metrics()
            logging.info("Backtest completed successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to run backtest: {str(e)}")
            return False
            
    def calculate_metrics(self):
        """Calculate performance metrics from backtest results"""
        if self.results is None:
            logging.error("No results to calculate metrics from")
            return
            
        # Check for the correct column name
        if 'strategy_return' in self.results.columns:
            returns = self.results['strategy_return']
        elif 'strategy_returns' in self.results.columns:
            returns = self.results['strategy_returns']
        else:
            # Try to find any column with 'return' in the name
            return_cols = [col for col in self.results.columns if 'return' in col.lower()]
            if return_cols:
                returns = self.results[return_cols[0]]
                logging.info(f"Using column '{return_cols[0]}' for returns calculation")
            else:
                logging.error("No return column found in results")
                return
        
        # Basic metrics
        self.metrics['total_return'] = (1 + returns).prod() - 1
        self.metrics['annualized_return'] = (1 + self.metrics['total_return']) ** (252/len(returns)) - 1
        self.metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
        self.metrics['sharpe_ratio'] = self.metrics['annualized_return'] / self.metrics['annualized_volatility']
        
        # Calculate downside volatility and Sortino ratio
        downside_returns = returns[returns < 0]
        self.metrics['downside_volatility'] = downside_returns.std() * np.sqrt(252)
        self.metrics['sortino_ratio'] = self.metrics['annualized_return'] / self.metrics['downside_volatility'] if self.metrics['downside_volatility'] > 0 else 0
        
        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        self.metrics['max_drawdown'] = drawdowns.min()
        
        # Calmar ratio
        self.metrics['calmar_ratio'] = self.metrics['annualized_return'] / abs(self.metrics['max_drawdown']) if self.metrics['max_drawdown'] < 0 else 0
        
        # Win rate
        self.metrics['win_rate'] = (returns > 0).mean()
        
        # Average win/loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        self.metrics['avg_win'] = wins.mean() if len(wins) > 0 else 0
        self.metrics['avg_loss'] = losses.mean() if len(losses) > 0 else 0
        
        # Gain/Loss ratio
        if self.metrics['avg_loss'] < 0:
            self.metrics['gain_loss_ratio'] = abs(self.metrics['avg_win'] / self.metrics['avg_loss'])
        else:
            self.metrics['gain_loss_ratio'] = 0
        
        logging.info("Metrics calculated successfully")
        
    def print_metrics(self):
        """Print calculated metrics in a formatted table"""
        if not self.metrics:
            logging.error("No metrics to print")
            return
            
        print("\nBacktest Results:")
        print("-" * 50)
        for metric, value in self.metrics.items():
            if isinstance(value, float):
                print(f"{metric:20s}: {value:.4f}")
            else:
                print(f"{metric:20s}: {value}")
                
    def save_results(self, output_path):
        """
        Save backtest results to CSV file
        
        Args:
            output_path: Path to save results
        """
        if self.results is None:
            logging.error("No results to save")
            return
            
        try:
            self.results.to_csv(output_path)
            logging.info(f"Results saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save results: {str(e)}")


def main():
    """
    Main function to run the backtest from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Run backtest on Donchian Channels strategy."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/ETH_USDT_1d.csv",
        help="Path to the CSV file containing OHLCV data with a 'timestamp' index.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["basic", "ml"],
        default="basic",
        help="Strategy to use for backtesting (basic or ml).",
    )
    parser.add_argument(
        "--disable_ml",
        action="store_true",
        help="Disable ML training and filtering (only applies to ML strategy).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the backtest results (optional).",
    )

    args = parser.parse_args()

    # Create strategy based on command line arguments
    if args.strategy == "basic":
        strategy = DonchianChannelsStrategy()
    else:  # ml
        strategy = DonchianChannelsMLStrategy(ml_enabled=not args.disable_ml)

    # Create and run backtest
    backtest = Backtest(strategy)
    
    if not backtest.load_data(args.data_path):
        return
    
    if not backtest.run():
        return
    
    # Print metrics
    backtest.print_metrics()
    
    # Save results if output path is provided
    if args.output_path:
        backtest.save_results(args.output_path)
    else:
        # Generate default output path
        output_path = args.data_path.replace(".csv", "_strategy_output.csv")
        if args.strategy == "ml":
            output_path = output_path.replace(".csv", "_ml_enhanced.csv")
        backtest.save_results(output_path)


if __name__ == "__main__":
    main() 