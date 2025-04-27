import pandas as pd
import pandas_ta as ta
import numpy as np

class Strategy:
    """
    Base class for trading strategies.
    """
    def __init__(self, look_back_windows=None, target_volatility=0.25, max_allocation=2.0, 
                 volatility_window=90, trading_days_per_year=252, risk_free_rate=0.0):
        """
        Initialize the strategy with parameters.
        
        Args:
            look_back_windows (list): List of lookback periods for Donchian channels
            target_volatility (float): Target volatility for position sizing
            max_allocation (float): Maximum allocation per position
            volatility_window (int): Window for volatility calculation
            trading_days_per_year (int): Number of trading days in a year
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
        """
        self.look_back_windows = look_back_windows or [5, 10, 20, 30, 60, 90, 150, 250, 360]
        self.target_volatility = target_volatility
        self.max_allocation = max_allocation
        self.volatility_window = volatility_window
        self.trading_days_per_year = trading_days_per_year
        self.risk_free_rate = risk_free_rate
        
    def calculate_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates annualized volatility of log returns.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added volatility column
        """
        df = data.copy()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        rolling_std = (
            df["log_returns"]
            .rolling(window=self.volatility_window, min_periods=self.volatility_window // 2)
            .std()
        )
        # Annualize the volatility
        df["sigma_t"] = rolling_std * np.sqrt(self.trading_days_per_year)
        df["sigma_t"] = df["sigma_t"].bfill().ffill()
        df["sigma_t"] = df["sigma_t"].replace(0, 1e-6)
        df = df.drop(columns=["log_returns"])
        return df
    
    def calculate_donchian_and_signals(self, data: pd.DataFrame, timeframe: int) -> pd.DataFrame:
        """
        Calculates Donchian Channels, Trailing Stop, and Trading Signal for a given timeframe.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            timeframe (int): Lookback period for Donchian channels
            
        Returns:
            pd.DataFrame: DataFrame with added Donchian channels and signals
        """
        df = data.copy()

        # Calculate Donchian Channels using Close prices
        donchian = ta.donchian(high=df["close"], low=df["close"], length=timeframe)
        
        dcl_col = f"DCL_{timeframe}"
        dcm_col = f"DCM_{timeframe}"
        dcu_col = f"DCU_{timeframe}"
        donchian.columns = [dcl_col, dcm_col, dcu_col]
        df = pd.concat([df, donchian], axis=1)

        # Calculate Trailing Stop and Trading Signal (Pos)
        pos_col = f"Pos_{timeframe}"
        stop_col = f"TrailingStop_{timeframe}"

        df[pos_col] = 0.0
        df[stop_col] = np.nan

        df_index = df.index
        for i in range(1, len(df)):
            idx_curr = df_index[i]
            idx_prev = df_index[i - 1]

            current_close = df.loc[idx_curr, "close"]
            prev_pos = df.loc[idx_prev, pos_col]
            prev_stop = df.loc[idx_prev, stop_col]
            current_dcm = df.loc[idx_curr, dcm_col]
            current_dcu = df.loc[idx_curr, dcu_col]
            prev_dcm = df.loc[idx_prev, dcm_col]

            # Entry condition: previous position was flat AND close hits upper band
            if prev_pos == 0 and current_close >= current_dcu:
                df.loc[idx_curr, pos_col] = 1.0
                df.loc[idx_curr, stop_col] = current_dcm  # Initial stop at midpoint

            # Exit condition: previous position was long AND close hits trailing stop
            elif prev_pos == 1 and not np.isnan(prev_stop) and current_close <= prev_stop:
                df.loc[idx_curr, pos_col] = 0.0
                df.loc[idx_curr, stop_col] = np.nan

            # Hold condition: previous position was long and exit not triggered
            elif prev_pos == 1:
                df.loc[idx_curr, pos_col] = 1.0
                # Update trailing stop: max of previous stop and previous day midpoint
                if not np.isnan(prev_stop) and not np.isnan(prev_dcm):
                    df.loc[idx_curr, stop_col] = max(prev_stop, prev_dcm)
                else:
                    df.loc[idx_curr, stop_col] = prev_stop

            # No position condition (otherwise)
            else:
                df.loc[idx_curr, pos_col] = 0.0
                df.loc[idx_curr, stop_col] = np.nan

        return df
    
    def calculate_position_size(self, data: pd.DataFrame, timeframe: int) -> pd.DataFrame:
        """
        Calculates the position weight for a given timeframe.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data and signals
            timeframe (int): Lookback period for Donchian channels
            
        Returns:
            pd.DataFrame: DataFrame with added position weights
        """
        df = data.copy()
        pos_col = f"Pos_{timeframe}"
        weight_col = f"w_{timeframe}"

        # Calculate raw weight based on volatility target
        raw_weight = self.target_volatility / df["sigma_t"]

        # Apply maximum allocation cap
        capped_weight = np.minimum(raw_weight, self.max_allocation)

        # Apply the position signal (0 or 1)
        df[weight_col] = capped_weight * df[pos_col]
        df[weight_col] = df[weight_col].fillna(0)

        return df
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the strategy on the provided data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with strategy signals and weights
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement run()") 