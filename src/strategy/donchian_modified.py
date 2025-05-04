from typing import TypedDict
import os
import numpy as np
import pandas as pd
import pandas_ta as ta
from src.strategy.donchian import DonchianStrategy
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib

class Model(TypedDict):
    clf: lgb.LGBMClassifier
    scaler: StandardScaler
    threshold: float
    columns: list[str]

class ModifiedDonchianStrategy(DonchianStrategy):
    FEATURE_COLS = [
        'rsi_14',
        'mom_10d',
        'mom_30d',
        'stochk_14_3_3',
        'stochd_14_3_3',
        'macd_hist',
        'adx_14',
        'plus_di_14',
        'minus_di_14',
        'sma_50_ratio',
        'ema_20_ratio',
        'atr_14_norm',
        'bbands_width_20_2',
        'volatility_30d',
        'volatility_90d',
        'obv_pct_change_10d',
        'donchian_width_rel_60',
    ]

    def __init__(self, dir: str, **kwargs):
        super().__init__(**kwargs)
        self.models = self._load_models(dir)
        self.TRADING_DAYS_PER_YEAR
    
    def _load_models(self, dir: str) -> dict[int, Model]:
        models = {}
        for window in self.LOOK_BACK_WINDOWS:
            path = os.path.join(dir, f'model_{window}.joblib')
            models[window]: Model = joblib.load(path) # type: ignore
        return models
    

    def _calc_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data['rsi_14'] = ta.rsi(data['close'], length=14)

        data['mom_10d'] = data['close'].pct_change(periods=10)
        data['mom_30d'] = data['close'].pct_change(periods=30)

        data[['stochk_14_3_3', 'stochd_14_3_3']] = ta.stoch(
            data['high'], data['low'], data['close'], k=14, d=3, smooth_k=3)

        data['macd_hist'] = ta.macd(data['close'])['MACDh_12_26_9'] # type: ignore[reportOptionalMemberAccess]

        data[['adx_14', 'plus_di_14', 'minus_di_14']] = ta.adx(
            data['high'], data['low'], data['close'], length=14)
        
        data['sma_50_ratio'] = data['close'] / ta.sma(data['close'], length=50) - 1  # type: ignore[reportOptionalMemberAccess]
        data['ema_20_ratio'] = data['close'] / ta.ema(data['close'], length=50) - 1  # type: ignore[reportOptionalMemberAccess]

        data['atr_14_norm'] = ta.atr(data['high'], data['low'], data['close'], length=14) / data['close'] # type: ignore[reportOptionalMemberAccess]

        _bbands = ta.bbands(data['close'], length=20, std=2)
        data['bbands_width_20_2'] = (_bbands['BBU_20_2.0'] - _bbands['BBL_20_2.0']) / _bbands['BBM_20_2.0'] # type: ignore[reportOptionalMemberAccess]

        data['log_returns_1d'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility_30d'] = data['log_returns_1d'].rolling(window=30).std()
        data['volatility_90d'] = data['log_returns_1d'].rolling(window=90).std()

        data['obv_pct_change_10d'] = ta.obv(data['close'], data['volume']).pct_change(periods=10)

        _dc = ta.donchian(high=data['high'], low=data['low'], lower_length=60, upper_length=60)
        data['donchian_width_rel_60'] = (_dc['DCU_60_60'] - _dc['DCL_60_60']) / data['close'] # type: ignore[reportOptionalMemberAccess]

        data.drop(columns=['log_returns_1d'], errors='ignore', inplace=True)

        data[self.FEATURE_COLS] = data[self.FEATURE_COLS].shift(1)
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        return data    

    # override
    def _calc_pos_and_trailing_stop(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        model = self.models[window]
        data = self._calc_features(data)
        x = data[model['columns']]
        x_scaled = model['scaler'].transform(x)
        POS_COL = f'POS_{window}'
        TRAILING_STOP_COL = f'TRL_STOP_{window}'
        
        data[POS_COL] = 0
        data[TRAILING_STOP_COL] = np.nan
        idx = data.index
        for i in range(1, len(data)):
            curr_data_idx = idx[i]
            pred_data_idx = idx[i-1]

            current_close: np.float64 = data.loc[curr_data_idx, "close"] # type: ignore
            prev_pos: np.float64 = data.loc[pred_data_idx, POS_COL] # type: ignore
            prev_stop: np.float64 = data.loc[pred_data_idx, TRAILING_STOP_COL] # type: ignore
            current_dcm: np.float64 = data.loc[curr_data_idx, f'DCM_{window}'] # type: ignore
            current_dcu: np.float64 = data.loc[curr_data_idx, f'DCU_{window}'] # type: ignore
            prev_dcm: np.float64 = data.loc[pred_data_idx, f'DCM_{window}'] # type: ignore

            ml_pass = False
            potential_entry = (prev_pos == 0 and not np.isnan(current_dcu) and current_close >= current_dcu)
    
            curr_features: np.ndarray = x_scaled[i].reshape(1, -1)
            probability = model['clf'].predict_proba(curr_features)[0, 1] # type: ignore

            if potential_entry:
                if probability >= model['threshold']:
                    ml_pass = True

        
            if potential_entry and ml_pass:
                data.loc[curr_data_idx, POS_COL] = 1.0
                data.loc[curr_data_idx, TRAILING_STOP_COL] = current_dcm
            
            elif prev_pos == 1 and not np.isnan(prev_stop) and current_close <= prev_stop:
                data.loc[curr_data_idx, POS_COL] = 0.
                data.loc[curr_data_idx, TRAILING_STOP_COL] = np.nan
            
            # Hold condition: previous position was long and exit not triggered
            elif prev_pos == 1:
                data.loc[curr_data_idx, POS_COL] = 1.
                # Update trailing stop: max of previous stop and previous day midpoint
                if not np.isnan(prev_stop) and not np.isnan(prev_dcm):
                    data.loc[curr_data_idx, TRAILING_STOP_COL] = np.minimum(prev_stop, prev_dcm)
                else:
                    data.loc[curr_data_idx, TRAILING_STOP_COL] = prev_stop
            
            # No position condition (otherwise)
            else:
                data.loc[curr_data_idx, POS_COL] = 0.
                data.loc[curr_data_idx, TRAILING_STOP_COL] = np.nan

        return data


