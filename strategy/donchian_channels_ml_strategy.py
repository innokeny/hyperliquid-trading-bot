import pandas as pd
import numpy as np
import pandas_ta as ta
from strategy import Strategy
import joblib
import os
from datetime import datetime

# Constants
LOOK_BACK_WINDOWS = [5, 10, 20, 30, 60, 90, 150, 250, 360]

class DonchianChannelsMLStrategy(Strategy):
    """
    Implementation of the Donchian Channels strategy with ML filtering.
    """
    def __init__(self, ml_enabled=True, train_test_split=0.8, model_path=None, **kwargs):
        """
        Initialize the ML-enhanced Donchian Channels strategy.
        
        Args:
            ml_enabled (bool): Whether to use ML filtering
            train_test_split (float): Proportion of data used for training (default: 0.8)
            model_path (str): Path to the pre-trained model file
            **kwargs: Keyword arguments to pass to the parent Strategy class
        """
        super().__init__(**kwargs)
        self.ml_enabled = ml_enabled
        self.train_test_split = train_test_split
        
        # ML models and parameters
        self.ml_models = {}
        self.ml_scalers = {}
        self.ml_thresholds = {}
        self.ml_selected_features = {}
        
        # Load pre-trained model if provided
        if ml_enabled and model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a pre-trained model from file.
        
        Args:
            model_path (str): Path to the model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Extract timeframe from model filename
            model_name = os.path.basename(model_path)
            if 'model_' in model_name and 'd.joblib' in model_name:
                timeframe_str = model_name.split('_')[1].split('d')[0]
                timeframe = int(timeframe_str)
                print(f"Extracted timeframe {timeframe}d from model filename")
            else:
                # If we can't extract timeframe, use the first one in LOOK_BACK_WINDOWS
                timeframe = LOOK_BACK_WINDOWS[0]
                print(f"Could not extract timeframe from model filename, using {timeframe}d")
            
            model_data = joblib.load(model_path)
            self.ml_models[timeframe] = model_data['model']
            self.ml_scalers[timeframe] = model_data['scaler']
            self.ml_thresholds[timeframe] = model_data['threshold']
            self.ml_selected_features[timeframe] = model_data['selected_features']
            print(f"Loaded model for timeframe {timeframe}d from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise
    
    def calculate_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ML features for the strategy.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added ML features
        """
        df = data.copy()
        print("Calculating ML features...")

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
        df['volatility_30d'] = df['log_returns_1d'].rolling(window=30).std() * np.sqrt(252)
        df['volatility_90d'] = df['log_returns_1d'].rolling(window=90).std() * np.sqrt(252)

        if 'volume' in df.columns and df['volume'].nunique() > 1 and df['volume'].sum() > 0:
            obv = ta.obv(df['close'], df['volume'])
            if obv is not None:
                df['obv'] = obv
                df['obv_pct_change_10d'] = df['obv'].pct_change(periods=10)
            df = df.drop(columns=['obv'], errors='ignore')
        else:
            df['obv_pct_change_10d'] = 0.0

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

        # Add any missing features that might be required by the model
        feature_cols = [
            'rsi_14', 'mom_10d', 'mom_30d', 'stochk_14_3_3', 'stochd_14_3_3',
            'macd_hist', 'adx_14', 'plus_di_14', 'minus_di_14', 'sma_50_ratio', 'ema_20_ratio',
            'atr_14_norm', 'bbands_width_20_2', 'volatility_30d', 'volatility_90d',
            'obv_pct_change_10d', 'donchian_width_rel_60'
        ]
        
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        df[feature_cols] = df[feature_cols].shift(1)

        df = df.ffill().bfill()
        df = df.dropna(subset=['close', 'high', 'low'])

        print(f"ML features calculated. Shape after feature engineering: {df.shape}")
        return df
    
    def calculate_donchian_and_signals_ml(self, data: pd.DataFrame, timeframe: int, ml_model, scaler, 
                                         ml_threshold: float, selected_features: list) -> pd.DataFrame:
        """
        Calculate Donchian Channels and signals with ML filtering.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            timeframe (int): Lookback period for Donchian channels
            ml_model: Trained ML model
            scaler: Fitted scaler
            ml_threshold (float): Threshold for ML predictions
            selected_features (list): List of selected features
            
        Returns:
            pd.DataFrame: DataFrame with added Donchian channels and signals
        """
        df = data.copy()
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required selected ML features: {missing_features}")

        donchian = ta.donchian(high=df["close"], low=df["close"], length=timeframe)
        dcl_col, dcm_col, dcu_col = f'DCL_{timeframe}', f'DCM_{timeframe}', f'DCU_{timeframe}'
        donchian.columns = [dcl_col, dcm_col, dcu_col]
        df = pd.concat([df, donchian], axis=1)

        if scaler is None:
            print(f"Warning: Scaler is None for timeframe {timeframe}. Cannot apply ML filter.")
            ml_model = None
        else:
            try:
                features_scaled = scaler.transform(df[selected_features])
                scaled_feature_cols = [f"{f}_scaled" for f in selected_features]
                df[scaled_feature_cols] = features_scaled
            except Exception as e:
                print(f"Error scaling features for prediction ({timeframe}d): {e}. Disabling ML filter.")
                ml_model = None

        pos_col = f'Pos_{timeframe}'
        stop_col = f'TrailingStop_{timeframe}'
        df[pos_col] = 0.0
        df[stop_col] = np.nan
        
        # Calculate Donchian channel signals
        df[pos_col] = np.where(df['close'] > df[dcu_col], 1, 
                              np.where(df['close'] < df[dcl_col], -1, 0))
        
        # Apply ML filtering if model is available
        if ml_model is not None:
            try:
                # Get probability predictions
                proba = ml_model.predict_proba(df[selected_features])[:, 1]
                
                # Filter signals based on ML predictions
                df[pos_col] = np.where(proba > ml_threshold, df[pos_col], 0)
                
                # Add ML probability as a column
                df[f'ML_Prob_{timeframe}'] = proba
            except Exception as e:
                print(f"Error applying ML filter ({timeframe}d): {e}")
        
        return df
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the Donchian Channels strategy with ML filtering on the provided data.
        """
        results = pd.DataFrame(index=data.index)
        results['close'] = data['close']
        results['position'] = 0
        results['returns'] = 0.0

        # Calculate features for all timeframes
        data = self.calculate_ml_features(data)

        # Process each timeframe that we have a model for
        for timeframe in self.ml_models.keys():
            print(f"Processing timeframe {timeframe}d...")
            
            # Calculate Donchian channels for this timeframe
            data = self.calculate_donchian_and_signals_ml(data, timeframe, self.ml_models[timeframe], self.ml_scalers[timeframe], self.ml_thresholds[timeframe], self.ml_selected_features[timeframe])
            
            # Get column names for this timeframe
            dcl_col = f'DCL_{timeframe}'
            dcu_col = f'DCU_{timeframe}'
            pos_col = f'Pos_{timeframe}'
            signal_event_col = f'signal_event_{timeframe}'
            
            # Identify signal events (when price crosses above upper channel)
            data[signal_event_col] = (data['close'] >= data[dcu_col]) & \
                                    (data[pos_col].shift(1).fillna(0) == 0)
            data[signal_event_col] = data[signal_event_col].fillna(False)
            
            # Get signal events for this timeframe
            signal_events = data[data[signal_event_col]].index
            
            if len(signal_events) == 0:
                continue
                
            # Prepare features for prediction
            X = data.loc[signal_events, self.ml_selected_features[timeframe]]
            
            # Scale features
            X_scaled = self.ml_scalers[timeframe].transform(X)
            
            # Make predictions
            y_pred_proba = self.ml_models[timeframe].predict_proba(X_scaled)[:, 1]
            
            # Apply threshold to get final predictions
            y_pred = (y_pred_proba >= self.ml_thresholds[timeframe]).astype(int)
            
            # Update positions based on predictions
            for idx, pred in zip(signal_events, y_pred):
                if pred == 1:  # If ML model predicts a good entry
                    results.loc[idx:, 'position'] = 1  # Take long position
                else:
                    results.loc[idx:, 'position'] = 0  # Stay in cash
            
            # Calculate returns
            results['returns'] = results['position'].shift(1) * data['close'].pct_change()
            results['returns'] = results['returns'].fillna(0)
            
            # Calculate cumulative returns
            results['cumulative_returns'] = (1 + results['returns']).cumprod()
            
            # Calculate buy and hold returns for comparison
            results['buy_hold_returns'] = data['close'].pct_change()
            results['buy_hold_cumulative_returns'] = (1 + results['buy_hold_returns']).cumprod()
            
            # Calculate metrics
            total_return = results['cumulative_returns'].iloc[-1] - 1
            buy_hold_return = results['buy_hold_cumulative_returns'].iloc[-1] - 1
            
            # Calculate period
            try:
                if isinstance(data.index[0], (pd.Timestamp, datetime)):
                    start_date = data.index[0].strftime('%Y-%m-%d')
                    end_date = data.index[-1].strftime('%Y-%m-%d')
                else:
                    start_date = str(data.index[0])
                    end_date = str(data.index[-1])
            except:
                start_date = "Unknown"
                end_date = "Unknown"
            
            years = len(data) / 365
            
            # Calculate annualized metrics
            cagr = (1 + total_return) ** (1 / years) - 1
            buy_hold_cagr = (1 + buy_hold_return) ** (1 / years) - 1
            
            # Calculate Sharpe ratio
            annual_volatility = results['returns'].std() * np.sqrt(365)
            sharpe_ratio = (cagr - 0.02) / annual_volatility if annual_volatility != 0 else 0
            
            # Calculate Sortino ratio
            downside_returns = results['returns'][results['returns'] < 0]
            downside_vol = downside_returns.std() * np.sqrt(365)
            sortino_ratio = cagr / downside_vol if downside_vol != 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = results['cumulative_returns']
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # Calculate Calmar ratio
            calmar_ratio = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
            
            # Calculate win rate and trades
            winning_trades = (results['returns'] > 0).sum()
            total_trades = (results['position'].diff() != 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate gain/loss ratio
            avg_win = results['returns'][results['returns'] > 0].mean() if len(results['returns'][results['returns'] > 0]) > 0 else 0
            avg_loss = abs(results['returns'][results['returns'] < 0].mean()) if len(results['returns'][results['returns'] < 0]) > 0 else 0
            gain_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
            
            # Format and print results in the requested style
            print(f"\nLookback Window: {timeframe}d days")
            print(f"Period: {start_date} to {end_date} ({years:.2f} years)")
            print(f"CAGR:                          {cagr:.2%}")
            print(f"Annualized Volatility:         {annual_volatility:.2%}")
            print(f"Sharpe Ratio:                  {sharpe_ratio:.2f}")
            print(f"Sortino Ratio:                 {sortino_ratio:.2f}")
            print(f"Max Drawdown:                  {max_drawdown:.2%}")
            print(f"Calmar Ratio:                  {calmar_ratio:.2f}")
            print(f"----------------------------------------")
            print(f"Cumulative Return (Strategy):  {total_return:.2%}")
            print(f"Cumulative Return (Buy & Hold):{buy_hold_return:.2%}")
            print(f"----------------------------------------")
            print(f"Total Trades:                  {total_trades}")
            print(f"Win Rate:                      {win_rate:.2%}")
            print(f"Gain:Loss Ratio:               {gain_loss_ratio:.2f}")
            
            # Save results to CSV
            results.to_csv(f'backtest_results_{timeframe}d.csv')
            
            return results 