from typing import Dict, List, Optional, Any, Union, Tuple, cast, Literal
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pandas import Series

logger = logging.getLogger(__name__)

class DataPreprocessor:

    @staticmethod
    def preprocess_candles(candles: List[Dict[str, Any]]) -> pd.DataFrame:
        """Preprocess the candles data."""
        # TODO: preprocess the candles data to get the features
        df = pd.DataFrame(candles)
        df['o'] = df['o'].astype(float)
        df['h'] = df['h'].astype(float)
        df['l'] = df['l'].astype(float)
        df['c'] = df['c'].astype(float)
        df['v'] = df['v'].astype(float)
        df['t'] = df['t'].astype(int)
        return df
    
    @staticmethod
    def normalize_candles(candles: pd.DataFrame) -> pd.DataFrame:
        """Normalize the candles data."""
        # TODO: normalize the candles data
        candles['c'] = candles['c'] / candles['c'].max()
        return candles
    
