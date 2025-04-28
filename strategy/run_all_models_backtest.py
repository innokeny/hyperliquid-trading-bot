import os
import subprocess
import logging
from datetime import datetime

def setup_logging(output_dir):
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{output_dir}/run_all_models_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    # Configuration
    data_path = "data/1d_timeframe/binance/binance_ETH_USDT_1d_20150427_20250427.csv"
    output_dir = "strategy/backtest_logs"
    train_test_split = 0.8  # 80% training, 20% testing
    
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting backtest with all models from models directory")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Train/test split: {train_test_split}")
    
    # Get all model files
    models_dir = "models"
    model_files = [
        "model_5d.joblib",
        "model_10d.joblib",
        "model_20d.joblib",
        "model_30d.joblib",
        "model_60d.joblib",
        "model_90d.joblib",
        "model_150d.joblib",
        "model_250d.joblib",
        "model_360d.joblib"
    ]
    
    if not all(os.path.exists(os.path.join(models_dir, model)) for model in model_files):
        logger.error(f"Some model files are missing in {models_dir}")
        return
    
    logger.info(f"Found {len(model_files)} model files: {model_files}")
    
    # Run backtest for each model
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        logger.info(f"Running backtest with model: {model_path}")
        
        try:
            # Run the backtest script
            cmd = [
                "python", "strategy/run_backtest.py",
                "--data_path", data_path,
                "--strategy_type", "ml",
                "--model_path", model_path,
                "--train_test_split", str(train_test_split),
                "--output_dir", output_dir
            ]
            
            logger.info(f"Executing command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Backtest completed successfully for {model_file}")
                logger.info(result.stdout)
            else:
                logger.error(f"Backtest failed for {model_file}")
                logger.error(result.stderr)
        
        except Exception as e:
            logger.error(f"Error running backtest for {model_file}: {str(e)}", exc_info=True)
    
    logger.info("All backtests completed")

if __name__ == "__main__":
    main() 