import pandas as pd
import pandas_ta as ta
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, roc_auc_score, precision_recall_curve, f1_score
from sklearn.feature_selection import mutual_info_classif
import joblib
import os
import argparse
from typing import Tuple, Dict, Optional, List
from datetime import datetime
import warnings
import logging

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Constants
LOOK_BACK_WINDOWS = [5, 10, 20, 30, 60, 90, 150, 250, 360]
FORWARD_LOOK_PERIOD = 20
TARGET_PROFIT_ATR = 3.0
STOP_LOSS_ATR = 1.5
ATR_PERIOD_TARGET = 14
TRADING_DAYS_PER_YEAR = 252

FEATURE_COLS = [
    'rsi_14', 'mom_10d', 'mom_30d', 'stochk_14_3_3', 'stochd_14_3_3',
    'macd_hist', 'adx_14', 'plus_di_14', 'minus_di_14', 'sma_50_ratio', 'ema_20_ratio',
    'atr_14_norm', 'bbands_width_20_2', 'volatility_30d', 'volatility_90d',
    'obv_pct_change_10d', 'donchian_width_rel_60'
]

def setup_logging(output_dir: str) -> logging.Logger:
    """Set up logging to both console and file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger('ml_model_trainer')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    log_file = os.path.join(output_dir, 'lightgbm_training.log')
    file_handler = logging.FileHandler(log_file)
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def calculate_ml_features(data: pd.DataFrame) -> pd.DataFrame:
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
    df = df.dropna(subset=['close', 'high', 'low'])

    print(f"ML features calculated. Shape after feature engineering: {df.shape}")
    return df

def create_target_variable_atr(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    target_col = 'target'
    df[target_col] = 0

    print("Creating ATR-based target variable...")

    if f'atr_{ATR_PERIOD_TARGET}' not in df.columns:
        print(f"Calculating ATR_{ATR_PERIOD_TARGET} for target definition.")
        df[f'atr_{ATR_PERIOD_TARGET}'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD_TARGET)
        df[f'atr_{ATR_PERIOD_TARGET}'] = df[f'atr_{ATR_PERIOD_TARGET}'].fillna(method='bfill').fillna(method='ffill')

    atr_values = df[f'atr_{ATR_PERIOD_TARGET}'].values
    close_prices = df['close'].values
    low_prices = df['low'].values
    high_prices = df['high'].values

    for i in range(len(df) - FORWARD_LOOK_PERIOD):
        entry_price = close_prices[i]
        entry_atr = atr_values[i]

        if entry_atr <= 1e-9 or np.isnan(entry_atr):
            continue

        profit_target_price = entry_price + TARGET_PROFIT_ATR * entry_atr
        stop_loss_price = entry_price - STOP_LOSS_ATR * entry_atr

        hit_target = False
        hit_stop = False

        for j in range(1, FORWARD_LOOK_PERIOD + 1):
            future_low = low_prices[i + j]
            future_high = high_prices[i + j]

            if future_low <= stop_loss_price:
                hit_stop = True
                break
            if future_high >= profit_target_price:
                hit_target = True
                break

        if hit_target and not hit_stop:
            df.iloc[i, df.columns.get_loc(target_col)] = 1

    print(f"Target variable created. Distribution:\n{df[target_col].value_counts(normalize=True)}")
    return df

def calculate_donchian_channels(data: pd.DataFrame, timeframe: int) -> pd.DataFrame:
    df = data.copy()
    donchian = ta.donchian(high=df["close"], low=df["close"], length=timeframe)
    dcl_col, dcm_col, dcu_col = f'DCL_{timeframe}', f'DCM_{timeframe}', f'DCU_{timeframe}'
    donchian.columns = [dcl_col, dcm_col, dcu_col]
    df = pd.concat([df, donchian], axis=1)
    
    pos_col = f'Pos_{timeframe}'
    df[pos_col] = np.where(df['close'] > df[dcu_col], 1, 
                          np.where(df['close'] < df[dcl_col], -1, 0))
    
    return df

def train_ml_model(data: pd.DataFrame, timeframe: int, output_dir: str, n_optuna_trials: int, n_cv_splits: int, feature_selection_threshold: float, high_correlation_threshold: float, logger: logging.Logger) -> None:
    logger.info(f"\n--- Training ML Model for Timeframe {timeframe} ---")
    features_df = data.copy()
    target_col = 'target'
    
    # Calculate Donchian channels for the current timeframe
    features_df = calculate_donchian_channels(features_df, timeframe)
    signal_event_col = f'signal_event_{timeframe}'
    dcu_col = f'DCU_{timeframe}'

    features_df[signal_event_col] = (features_df['close'] >= features_df[dcu_col]) & \
                                   (features_df[f'Pos_{timeframe}'].shift(1).fillna(0) == 0)
    features_df[signal_event_col] = features_df[signal_event_col].fillna(False)

    if target_col not in features_df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Run create_target_variable_atr first.")

    ml_data = features_df[features_df[signal_event_col]].copy()
    ml_data = ml_data.dropna(subset=FEATURE_COLS + [target_col])

    if ml_data.empty or len(ml_data) < 10:
        logger.info(f"Not enough signal events ({len(ml_data)}) for training. Cannot train ML model.")
        return None

    logger.info(f"Total signal events for ML ({timeframe}d): {len(ml_data)}")
    X = ml_data[FEATURE_COLS]
    y = ml_data[target_col]

    logger.info("Performing feature selection...")
    selected_features = list(X.columns)

    # Correlation-based feature selection
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] > high_correlation_threshold)]
    selected_features = [f for f in selected_features if f not in to_drop_corr]
    logger.info(f"Dropped due to high correlation: {to_drop_corr}")
    logger.info(f"Features remaining after correlation filter: {len(selected_features)}")

    X = X[selected_features]

    # Mutual Information based feature selection
    if len(selected_features) > 1 and len(X) > 0:
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            mi_series = pd.Series(mi_scores, index=selected_features).sort_values(ascending=False)
            to_keep_mi = mi_series[mi_series > feature_selection_threshold].index.tolist()
            if len(to_keep_mi) < 2:
                logger.info("Warning: MI threshold too high, keeping top 5 features instead.")
                to_keep_mi = mi_series.head(min(5, len(selected_features))).index.tolist()

            to_drop_mi = [f for f in selected_features if f not in to_keep_mi]
            selected_features = to_keep_mi
            logger.info(f"Dropped due to low mutual information: {to_drop_mi}")
            logger.info(f"Final selected features: {len(selected_features)}")

            X = X[selected_features]
        except Exception as e:
            logger.info(f"Warning: Error during MI feature selection - {e}. Keeping all features after correlation.")

    if not selected_features:
        logger.info("Error: No features selected. Cannot train.")
        return None

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scale_pos_weight = len(y[y == 0]) / (len(y[y == 1]) + 1e-9)
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Hyperparameter tuning with Optuna
    logger.info("Starting hyperparameter tuning with Optuna...")
    study = optuna.create_study(direction='maximize')

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'seed': 42,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'scale_pos_weight': scale_pos_weight
        }

        cv_scores = []
        tscv = TimeSeriesSplit(n_splits=n_cv_splits)

        for train_idx, val_idx in tscv.split(X_scaled):
            X_fold_train, y_fold_train = X_scaled[train_idx], y.iloc[train_idx]
            X_fold_val, y_fold_val = X_scaled[val_idx], y.iloc[val_idx]

            if len(np.unique(y_fold_train)) < 2 or len(np.unique(y_fold_val)) < 2:
                continue

            model_cv = lgb.LGBMClassifier(**params)
            model_cv.fit(X_fold_train, y_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)],
                        eval_metric='auc',
                        callbacks=[lgb.early_stopping(10, verbose=False)])
            y_pred_proba_cv = model_cv.predict_proba(X_fold_val)[:, 1]
            cv_scores.append(roc_auc_score(y_fold_val, y_pred_proba_cv))

        return np.mean(cv_scores) if cv_scores else 0.0

    try:
        study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=True)
        best_params = study.best_params
        logger.info(f"Optuna best AUC: {study.best_value:.4f}")
        logger.info(f"Optuna best params: {best_params}")
    except Exception as e:
        logger.info(f"Warning: Optuna optimization failed - {e}. Using default parameters.")
        best_params = {}

    # Train final model
    logger.info("Training final LightGBM model with best parameters...")
    final_params = {
        'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
        'boosting_type': 'gbdt', 'seed': 42, 'scale_pos_weight': scale_pos_weight
    }
    final_params.update(best_params)

    lgb_clf_final = lgb.LGBMClassifier(**final_params)
    lgb_clf_final.fit(X_scaled, y)

    # Save model and components
    model_data = {
        'model': lgb_clf_final,
        'scaler': scaler,
        'threshold': 0.5,  # Default threshold since we're not using validation set
        'selected_features': selected_features,
        'timeframe': timeframe
    }

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'model_{timeframe}d.joblib')
    joblib.dump(model_data, model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("--- ML Model Training Complete ---")

def main():
    parser = argparse.ArgumentParser(description='Train ML models for the Donchian Channels strategy')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the data file')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Directory to save the trained models')
    parser.add_argument('--n_optuna_trials', type=int, default=100,
                      help='Number of Optuna trials for hyperparameter tuning (default: 100)')
    parser.add_argument('--n_cv_splits', type=int, default=5,
                      help='Number of cross-validation splits (default: 5)')
    parser.add_argument('--feature_selection_threshold', type=float, default=0.001,
                      help='Threshold for feature selection (default: 0.001)')
    parser.add_argument('--high_correlation_threshold', type=float, default=0.95,
                      help='Threshold for correlation-based feature selection (default: 0.95)')
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.output_dir)
    
    # Load and prepare data
    logger.info(f"Loading data from {args.data_path}")
    data = pd.read_csv(args.data_path, index_col=0, parse_dates=True)
    
    logger.info(f"Total data points: {len(data)}")
    
    # Prepare features and target
    data = calculate_ml_features(data)
    data = create_target_variable_atr(data)
    
    # Train models for each timeframe
    for timeframe in LOOK_BACK_WINDOWS:
        logger.info(f"\nTraining model for timeframe {timeframe} days...")
        train_ml_model(data, timeframe, args.output_dir, 
                      n_optuna_trials=args.n_optuna_trials,
                      n_cv_splits=args.n_cv_splits,
                      feature_selection_threshold=args.feature_selection_threshold,
                      high_correlation_threshold=args.high_correlation_threshold,
                      logger=logger)

    logger.info("\nAll models trained successfully!")

if __name__ == "__main__":
    main() 