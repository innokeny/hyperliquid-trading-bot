import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta

from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

class MLInference:
    FEATURE_COLS = [
        'rsi_14', 'mom_10d', 'mom_30d', 'stochk_14_3_3', 'stochd_14_3_3',
        'macd_hist', 'adx_14', 'plus_di_14', 'minus_di_14', 'sma_50_ratio', 'ema_20_ratio',
        'atr_14_norm', 'bbands_width_20_2', 'volatility_30d', 'volatility_90d',
        'obv_pct_change_10d',
        'donchian_width_rel_60'
    ]

    def __init__(self, path: str):
        self.path = path
        self.model, self.scaler, self.features, self.threshold, self.window = self._load_model(path)
        self._scaled_features = [f'{name}_scaled_by_{self.window}' for name in self.features]
    
    @staticmethod
    def _load_model(path: str) -> tuple[LGBMClassifier, StandardScaler, list[str], float, int]:
        file: dict = joblib.load(path)
        return file['model'], file['scaler'], file['selected_features'], file['threshold'], file['timeframe']

    @classmethod
    def calc_shared_features(cls, data: pd.DataFrame, TRADING_DAYS_PER_YEAR: int) -> pd.DataFrame:
        data['rsi_14'] = ta.rsi(data['close'], length=14)
        
        data['mom_10d'] = data['close'].pct_change(periods=10)
        data['mom_30d'] = data['close'].pct_change(periods=30)
        
        stoch = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3, smooth_k=3)
        if stoch is not None:
            data['stochk_14_3_3'] = stoch['STOCHk_14_3_3']
            data['stochd_14_3_3'] = stoch['STOCHd_14_3_3']

        macd = ta.macd(data['close'])
        if macd is not None and 'MACDh_12_26_9' in macd.columns:
            data['macd_hist'] = macd['MACDh_12_26_9']
        
        adx = ta.adx(data['high'], data['low'], data['close'], length=14)
        if adx is not None:
            data['adx_14'] = adx['ADX_14']
            data['plus_di_14'] = adx['DMP_14']
            data['minus_di_14'] = adx['DMN_14']
        
        sma50: np.float64 = ta.sma(data['close'], length=50) # type: ignore
        ema20: np.float64 = ta.ema(data['close'], length=20) # type: ignore
        data['sma_50_ratio'] = (data['close'] / sma50) - 1
        data['ema_20_ratio'] = (data['close'] / ema20) - 1

        data['atr_14'] = ta.atr(data['high'], data['low'], data['close'], length=14)
        data['atr_14_norm'] = data['atr_14'] / data['close']
        
        bbands = ta.bbands(data['close'], length=20, std=2)
        if bbands is not None:
            data['bbands_width_20_2'] = (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / bbands[f'BBM_{20}_2.0']
        
        data['log_returns_1d'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility_30d'] = data['log_returns_1d'].rolling(window=30).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        data['volatility_90d'] = data['log_returns_1d'].rolling(window=90).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        if 'volume' in data.columns and data['volume'].nunique() > 1 and data['volume'].sum() > 0:
            obv = ta.obv(data['close'], data['volume'])
            if obv is not None:
                data['obv'] = obv
                data['obv_pct_change_10d'] = data['obv'].pct_change(periods=10)
            data = data.drop(columns=['obv'], errors='ignore')
        else:
            data['obv_pct_change_10d'] = 0.0
            if 'obv_pct_change_10d' in cls.FEATURE_COLS:
                print("Warning: Volume data seems unreliable. OBV feature set to 0.")

        dc_temp = ta.donchian(high=data['close'], low=data['close'], length=60)
        if dc_temp is None:
            raise ValueError('')
        dc_temp.rename({
            'DCL_20_20': 'DCL_60',
            'DCM_20_20': 'DCM_60',
            'DCU_20_20': 'DCU_60'
        })
        
        if dc_temp is not None:
            data['donchian_width_rel_60'] = (dc_temp['DCU_60'] - dc_temp['DCL_60']) / data['close']
        else:
            data['donchian_width_rel_60'] = 0.0

        data = data.drop(columns=['log_returns_1d'], errors='ignore')

        for col in cls.FEATURE_COLS:
            if col not in data.columns:
                data[col] = 0.0

        data[cls.FEATURE_COLS] = data[cls.FEATURE_COLS].shift(1)

        data = data.ffill().bfill()
        data = data.dropna(subset=['close', 'high', 'low', 'asset_return'])
        return data
        

    def _transform_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self._scaled_features] = self.scaler.transform(data[self.features])
        return data

    def _predict_model(self, data: pd.DataFrame) -> pd.DataFrame:
        prediction: np.ndarray = self.model.predict_proba(data[self._scaled_features]) # type: ignore
        data[f'pred_{self.window}'] = prediction.reshape(-1)
        data[f'ml_pass_{self.window}'] = data[f'pred_{self.window}'] > self.threshold
        return data
       
    def predict(self, data: pd.DataFrame):
        data = self._transform_features(data)
        data = self._predict_model(data)
        return data