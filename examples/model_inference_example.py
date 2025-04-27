import asyncio
import time
from datetime import datetime, timedelta
from loguru import logger
from src.ml.model_inference import ModelInference
from src.data.market_data import MarketDataCollector
from src.trading.market_data import MarketDataStreamer
from src.trading.connection import HyperliquidConnection
from hyperliquid.utils.constants import MAINNET_API_URL

async def main():
    """Example usage of ModelInference with live market data."""
    # Initialize connection and data collection
    connection = HyperliquidConnection()
    _, info, _ = connection.setup(base_url=MAINNET_API_URL)
    
    streamer = MarketDataStreamer(info)
    collector = MarketDataCollector(streamer, candle_interval="1m", max_candles=100)
    
    # Initialize model inference
    model_path = "models/trained_model.pkl"  # Replace with your actual model path
    model_inference = ModelInference(model_path)
    
    try:
        # Start data collection
        collector.start()
        logger.info("Started market data collector")
        
        # Example: Monitor and make predictions for 60 seconds
        for _ in range(60):
            # Get latest candles
            latest_candles = collector.candles[-10:]
            if latest_candles:
                # Prepare features from the latest candle
                features = model_inference.prepare_features(latest_candles)
                
                # Make prediction
                prediction = model_inference.make_prediction(features)
                
                # Process prediction into trading signal
                signal = model_inference.process_prediction(prediction)
                
                # Log the results
                logger.info(f"Prediction: {prediction['prediction']}, Signal: {signal['signal']}" + (f", Price Target: {signal['price_target']}, Stop Loss: {signal['stop_loss']}, Take Profit: {signal['take_profit']}" if signal['signal'] != "HOLD" else ""))
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        # Stop the collector
        collector.stop()
        logger.info("Market data collector stopped")

if __name__ == "__main__":
    asyncio.run(main()) 