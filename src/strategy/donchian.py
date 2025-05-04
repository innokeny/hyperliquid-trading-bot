import pandas as pd
import pandas_ta as ta
import numpy as np


class DonchianStrategy:
    def __init__(
        self,
        LOOK_BACK_WINDOWS: list[int] = [5, 10, 20, 30, 60, 90, 150, 250, 360],
        TARGET_VOLATILITY: float = .25,
        MAX_ALLOCATION: float = 2.,
        VOLATILITY_WINDOW: int = 90,
        TRADING_DAYS_PER_YEAR: int = 252,
        RISK_FREE_RATE: float = .0
    ):
        self.LOOK_BACK_WINDOWS = LOOK_BACK_WINDOWS
        self.TARGET_VOLATILITY = TARGET_VOLATILITY
        self.MAX_ALLOCATION = MAX_ALLOCATION
        self.VOLATILITY_WINDOW = VOLATILITY_WINDOW
        self.TRADING_DAYS_PER_YEAR = TRADING_DAYS_PER_YEAR
        self.RISK_FREE_RATE = RISK_FREE_RATE

        self._w_cols = [f'w_{window}' for window in self.LOOK_BACK_WINDOWS]
    
    def _calc_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates {VOLATILITY_WINDOW}-day annualized volatility of log returns."""
        df = data.copy()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        rolling_std = (
            df["log_returns"]
            .rolling(window=self.VOLATILITY_WINDOW, min_periods=self.VOLATILITY_WINDOW // 2)
            .std()
        )
        df["sigma_t"] = rolling_std * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        df["sigma_t"] = df["sigma_t"].bfill().ffill()
        df["sigma_t"] = df["sigma_t"].replace(0, 1e-6)
        df = df.drop(columns=["log_returns"])
        return df

    @staticmethod
    def _calc_donchian(data: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Calculates Donchian Channels (based on close), Trailing Stop, and Trading Signal (Pos).
        """

        donchian = ta.donchian(high=data["close"], low=data["close"], lower_length=window, upper_length=window)
        if donchian is None:
            raise ValueError(f'Cant calculate ta.donchian')
        
        donchian.rename({
            f'DCL_{window}_{window}': f'DCL_{window}',
            f'DCM_{window}_{window}': f'DCM_{window}',
            f'DCU_{window}_{window}': f'DCU_{window}'
        }, inplace=True, axis=1)
        
        data = pd.concat([data, donchian], axis=1)
        
        return data

    def _calc_pos_and_trailing_stop(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
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
            # Entry condition: previous position was flat AND close hits upper band
            if prev_pos == 0 and current_close >= current_dcu:
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
                    data.loc[curr_data_idx, TRAILING_STOP_COL] = np.maximum(prev_stop, prev_dcm)
                else:
                    data.loc[curr_data_idx, TRAILING_STOP_COL] = prev_stop
            
            # No position condition (otherwise)
            else:
                data.loc[curr_data_idx, POS_COL] = 0.
                data.loc[curr_data_idx, TRAILING_STOP_COL] = np.nan
        return data

    def _calc_position(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        W_COL = f'w_{window}'
        POS_COL = f'POS_{window}'
        weight = np.minimum(self.TARGET_VOLATILITY / data['sigma_t'], self.MAX_ALLOCATION)
        data[W_COL] = weight * data[POS_COL]
        data[W_COL] = data[W_COL].fillna(0)
        return data

    def _aggregate_windows(self, data: pd.DataFrame) -> pd.DataFrame:
        data['w_combo'] = data[self._w_cols].mean(axis=1)
        return data


    def get_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._calc_volatility(data)
        for window in self.LOOK_BACK_WINDOWS:
            data = self._calc_donchian(data, window)
            data = self._calc_pos_and_trailing_stop(data, window)
            data = self._calc_position(data, window)
        data = self._aggregate_windows(data)        
        return data


