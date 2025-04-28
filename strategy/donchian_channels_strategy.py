import pandas as pd
import numpy as np
from strategy import Strategy

class DonchianChannelsStrategy(Strategy):
    """
    Implementation of the Donchian Channels strategy.
    """
    def __init__(self, **kwargs):
        """
        Initialize the Donchian Channels strategy.
        
        Args:
            **kwargs: Keyword arguments to pass to the parent Strategy class
        """
        super().__init__(**kwargs)
        
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the Donchian Channels strategy on the provided data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with strategy signals and weights
        """
        # Calculate volatility (needed for all lookbacks)
        data_with_vol = self.calculate_volatility(data)
        
        all_data = data_with_vol.copy()
        weight_columns = []

        # Process each lookback window
        for look_back_window in self.look_back_windows:
            # Calculate Donchian, Signals, and Stops for this window
            all_data = self.calculate_donchian_and_signals(all_data, look_back_window)

            # Calculate Position Size (Weight) for this window
            all_data = self.calculate_position_size(all_data, look_back_window)
            weight_columns.append(f"w_{look_back_window}")

        # Calculate Ensemble (Combo) Weight
        all_data["w_Combo"] = all_data[weight_columns].mean(axis=1)
        
        # Calculate Returns
        all_data["asset_return"] = all_data["close"].pct_change()
        all_data["strategy_return"] = all_data["w_Combo"].shift(1) * all_data["asset_return"]
        
        return all_data 