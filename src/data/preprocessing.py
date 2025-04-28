from typing import Dict, List, Optional, Any, Union, Tuple, cast, Literal
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pandas_ta as ta


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
    def __init__(self, scaler: Optional[Any] = None):
        """Initialize the DataPreprocessor."""
        self.scaler = scaler

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

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the features."""
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['mom_10d'] = df['close'].pct_change(periods=10)
        df['mom_30d'] = df['close'].pct_change(periods=30)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        if stoch is not None:
            df['stochk_14_3_3'] = stoch['STOCHk_14_3_3']
            df['stochd_14_3_3'] = stoch['STOCHd_14_3_3']

        macd = ta.macd(df['close'])
        if macd is not None and 'MACDh_12_26_9' in macd.columns:
            df['macd_hist'] = macd['MACDh_12_26_9']
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['adx_14'] = adx[f'ADX_{14}']
            df['plus_di_14'] = adx[f'DMP_{14}']
            df['minus_di_14'] = adx[f'DMN_{14}']
        sma50 = ta.sma(df['close'], length=50)
        ema20 = ta.ema(df['close'], length=20)
        df['sma_50_ratio'] = (df['close'] / sma50) - 1
        df['ema_20_ratio'] = (df['close'] / ema20) - 1

        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_14_norm'] = df['atr_14'] / df['close']
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None:
            df['bbands_width_20_2'] = (bbands[f'BBU_{20}_2.0'] - bbands[f'BBL_{20}_2.0']) / bbands[f'BBM_{20}_2.0']
        df['log_returns_1d'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_30d'] = df['log_returns_1d'].rolling(window=30).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        df['volatility_90d'] = df['log_returns_1d'].rolling(window=90).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        if 'volume' in df.columns and df['volume'].nunique() > 1 and df['volume'].sum() > 0:
            obv = ta.obv(df['close'], df['volume'])
            if obv is not None:
                df['obv'] = obv
                df['obv_pct_change_10d'] = df['obv'].pct_change(periods=10)
            df = df.drop(columns=['obv'], errors='ignore')
        else:
            df['obv_pct_change_10d'] = 0.0
            if 'obv_pct_change_10d' in FEATURE_COLS:
                print("Warning: Volume data seems unreliable. OBV feature set to 0.")

        dc_temp = ta.donchian(high=df['close'], low=df['close'], length=60)
        dcl_temp_col = f"DCL_{60}"
        dcm_temp_col = f"DCM_{60}"
        dcu_temp_col = f"DCU_{60}"
        dc_temp.columns = [dcl_temp_col, dcm_temp_col, dcu_temp_col]

        if dc_temp is not None:
            df['donchian_width_rel_60'] = (dc_temp['DCU_60'] - dc_temp['DCL_60']) / df['close']
        else:
            df['donchian_width_rel_60'] = 0.0

        df = df.drop(columns=['log_returns_1d'], errors='ignore')

        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        df[FEATURE_COLS] = df[FEATURE_COLS].shift(1)

        df = df.ffill().bfill()
        df = df.dropna(subset=['close', 'high', 'low', 'asset_return'])
        return df

    def normalize_candles(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Normalize the candles data."""
        if self.scaler:
            candles[FEATURE_COLS] = self.scaler.transform(candles[FEATURE_COLS])
        return candles
