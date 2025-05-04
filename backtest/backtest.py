from typing import Optional, TypedDict
import pandas as pd
import numpy as np
from src.strategy.donchian import DonchianStrategy

class BackTestResults(TypedDict):
    StartDate: str
    EndDate: str
    DurationInYears: float
    CAGR: float
    CumulativeReturns: float
    AnnualizedVolatility: float
    SharpeRatio: Optional[float]
    SortinoRatio: Optional[float]
    MaxDrawdown: float
    CalmarRatio: Optional[float]
    BuyAndHold: Optional[float]

class BackTest:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def run(self, strategy: DonchianStrategy) -> tuple[pd.DataFrame, dict[str, BackTestResults]]:
        data = strategy.get_weights(self.data)
        metrics: dict[str, BackTestResults] = {}
        for w_col in strategy._w_cols:
            data = self._calc_returns(data, w_col)
            metrics[w_col] = self._calc_metrics(data, strategy, w_col)
        data = self._calc_returns(data, 'w_combo')
        metrics['w_combo'] = self._calc_metrics(data, strategy, 'w_combo')
        
        return data, metrics
    
    def _calc_asset_returns(self, data: pd.DataFrame):
        _data = data.copy()
        _data['asset_returns'] = _data['close'].pct_change()
        return _data

    def _calc_returns(self, data: pd.DataFrame, w_col: str = 'w_combo') -> pd.DataFrame:
        _data = data.copy()
        if 'asset_returns' not in _data.columns:
            _data = self._calc_asset_returns(_data)
        _data[f'strategy_return({w_col})'] = _data[w_col].shift(1) * _data['asset_returns']
        return _data
    
    def _calc_metrics(self, data: pd.DataFrame, strategy: DonchianStrategy, w_col: str = 'w_combo', ) -> BackTestResults:
        asset_return = data['asset_returns']
        strategy_return = data[f'strategy_return({w_col})']
        
        start_date = data.index[0].strftime("%Y-%m-%d")
        end_date = data.index[-1].strftime("%Y-%m-%d")
        duration = (data.index[-1] - data.index[0]).days / 365.25

        cumulative_returns = (1 + strategy_return).cumprod()
        
        cagr = 0.
        if duration > 0:
            cagr = ((cumulative_returns.iloc[-1]) ** (1. / duration) - 1) * 100


        annualized_volatility = strategy_return.std() * np.sqrt(strategy.TRADING_DAYS_PER_YEAR) * 100
        
        sharpe_ratio = None
        if annualized_volatility > 1e-6:
            mean_annual_return = strategy_return.mean() * strategy.TRADING_DAYS_PER_YEAR
            sharpe_ratio = (mean_annual_return - strategy.RISK_FREE_RATE) / annualized_volatility * 100

        sortino_ratio = None
        negative_returns = strategy_return[strategy_return < 0]
        if not negative_returns.empty:
            downside_std = negative_returns.std() * np.sqrt(strategy.TRADING_DAYS_PER_YEAR)
            if downside_std > 1e-6:
                sortino_ratio = (mean_annual_return - strategy.RISK_FREE_RATE) / downside_std
        else:  # No losing days
            sortino_ratio = np.inf

        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        calmar_ratio = None
        if abs(max_drawdown) > 1e-6:
            calmar_ratio = cagr / abs(max_drawdown)
        
        buy_and_hold = None
        asset_return = asset_return.loc[strategy_return.index].dropna()
        if not asset_return.empty:
            buy_and_hold = ((1 + asset_return).cumprod().iloc[-1] - 1) * 100

        return {
            'StartDate': start_date,
            'EndDate': end_date,
            'DurationInYears': duration,
            'CAGR': cagr,
            'CumulativeReturns': (cumulative_returns.iloc[-1] - 1) * 100,
            'AnnualizedVolatility': annualized_volatility, 
            'SharpeRatio': sharpe_ratio,
            'SortinoRatio': sortino_ratio,
            'MaxDrawdown': max_drawdown,
            'CalmarRatio': calmar_ratio,
            'BuyAndHold': buy_and_hold
        }
