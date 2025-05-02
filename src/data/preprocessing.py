from typing import Dict, List, Optional, Any, Union, Tuple, cast, Literal
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'rsi_14', 'mom_10d', 'mom_30d', 'stochk_14_3_3', 'stochd_14_3_3',
    'macd_hist', 'adx_14', 'plus_di_14', 'minus_di_14', 'sma_50_ratio', 'ema_20_ratio',
    'atr_14_norm', 'bbands_width_20_2', 'volatility_30d', 'volatility_90d',
    'obv_pct_change_10d',
    'donchian_width_rel_60'
]

TRADING_DAYS_PER_YEAR = 252

class DataPreprocessor:
    """Preprocesses the data."""
    def __init__(self, TRADING_DAYS_PER_YEAR: int):
        self.TRADING_DAYS_PER_YEAR = TRADING_DAYS_PER_YEAR


    @staticmethod
    def preprocess_candles(candles: List[Dict[str, Any]]) -> pd.DataFrame:
        """Preprocess the candles data."""
        df = pd.DataFrame(candles)
        df['open'] = df['o'].astype(float)
        df['high'] = df['h'].astype(float)
        df['low'] = df['l'].astype(float)
        df['close'] = df['c'].astype(float)
        df['volume'] = df['v'].astype(float)
        df['timestamp'] = df['t'].astype(int)
        return df.drop(columns=['o', 'h', 'l', 'c', 'v', 't'])
