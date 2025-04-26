from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from ..data.preprocessing import DataPreprocessor
import lightgbm as lgb
from scipy.sparse import spmatrix, csr_matrix
logger = logging.getLogger(__name__)

class ModelInference:
    """Handles model inference and prediction processing."""

    def __init__(self, model_path: str):
        """
        Initialize the model inference system.

        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = self._load_model()
        self.preprocessor = DataPreprocessor()
        logger.info(f"Initialized model inference with model at {model_path}")

    def _load_model(self) -> lgb.Booster:
        """
        Load the trained LightGBM model from disk.

        Returns:
            Loaded LightGBM model
        """
        try:
            logger.info(f"Loading LightGBM model from {self.model_path}")
            model = lgb.Booster(model_file=self.model_path)
            logger.info("Successfully loaded LightGBM model")
            return model
        except Exception as e:
            logger.error(f"Error loading LightGBM model: {str(e)}")
            raise

    def prepare_features(self, candles: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare features for model inference.

        Args:
            candles: List of candles

        Returns:
            Processed features DataFrame
        """
        try:
            preprocessed_candles = self.preprocessor.preprocess_candles(candles)
            normalized_candles = self.preprocessor.normalize_candles(preprocessed_candles)
            
            logger.debug(f"Prepared features for inference: {len(normalized_candles)} samples")
            return normalized_candles
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def make_prediction(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using the loaded LightGBM model.

        Args:
            features: Processed features DataFrame

        Returns:
            Dictionary containing prediction results
        """
        try:
            required_features = [
                'o', 'h', 'l', 'c', 'v', 't'
            ]
            
            missing_features = [f for f in required_features if f not in features.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

            X = features[required_features].values
            
            raw_prediction = self.model.predict(X)
            
            if isinstance(raw_prediction, np.ndarray):
                prediction = float(raw_prediction[-1])
                confidence = float(np.abs(raw_prediction[-1]))
            elif isinstance(raw_prediction, (spmatrix, csr_matrix)):
                dense_pred = np.asarray(raw_prediction)
                prediction = float(dense_pred[-1])
                confidence = float(np.abs(dense_pred[-1]))
            else:
                try:
                    prediction = float(raw_prediction)
                    confidence = float(np.abs(raw_prediction))
                except (TypeError, ValueError) as e:
                    logger.error(f"Failed to convert prediction to float: {str(e)}")
                    raise ValueError("Invalid prediction output format")
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'confidence': confidence,
                'features_used': required_features,
                'raw_features': features.iloc[-1].to_dict()
            }
            
            logger.debug(f"Made prediction with confidence: {result['confidence']}")
            return result
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def process_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process prediction results into trading signals.

        Args:
            prediction: Raw prediction results

        Returns:
            Processed trading signals
        """
        try:
            signal = 'BUY' if prediction['prediction'] > 0 else 'SELL'
            
            confidence = prediction['confidence']
            current_price = prediction['raw_features']['c']
            
            stop_loss = current_price * (1 - 0.02) if signal == 'BUY' else current_price * (1 + 0.02)
            take_profit = current_price * (1 + 0.04) if signal == 'BUY' else current_price * (1 - 0.04)
            
            position_size = min(1.0, confidence)
            
            processed_signal = {
                'timestamp': prediction['timestamp'],
                'signal': signal,
                'confidence': prediction['confidence'],
                'price_target': take_profit,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'metadata': {
                    'model_version': '1.0',
                    'prediction_time': prediction['timestamp'],
                    'raw_prediction': prediction['prediction']
                }
            }
            
            logger.debug(f"Processed prediction into signal: {processed_signal['signal']}")
            return processed_signal
        except Exception as e:
            logger.error(f"Error processing prediction: {str(e)}")
            raise 