import os
import numpy as np
import pandas as pd
from src.strategy.donchian import DonchianStrategy
from src.ml.model_inference import MLInference


class ModifiedDonchianStrategy(DonchianStrategy):
    def __init__(self, dir: str, **kwargs):
        super().__init__(**kwargs)
        self.models = self._load_models(dir)
        self.TRADING_DAYS_PER_YEAR
    
    @staticmethod
    def _load_models(dir: str) -> dict[int, MLInference]:
        models = {}
        for path in os.listdir(dir):
            if path.endswith('.joblib'):
                clf = MLInference(os.path.join(dir, path))
                models[clf.window] = clf
        return models
    
    # override
    def _calc_pos_and_trailing_stop(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        data = MLInference.calc_shared_features(data, self.TRADING_DAYS_PER_YEAR)
        data = self.models[window].predict(data)
        
        POS_COL = f'POS_{window}'
        TRAILING_STOP_COL = f'TRL_STOP_{window}'
        ML_PASS_COL = f'ml_pass_{window}'
        
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

            ml_pass = data.loc[curr_data_idx, ML_PASS_COL]
            potential_entry = (prev_pos == 0 and not np.isnan(current_dcu) and current_close >= current_dcu)
            if potential_entry and ml_pass:
                data.loc[curr_data_idx, POS_COL] = 1.
                data.loc[curr_data_idx, TRAILING_STOP_COL] = current_dcm

            # Exit condition: previous position was long AND close hits trailing stop from previous day
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


