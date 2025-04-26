import argparse
import pandas as pd
import pandas_ta as ta
import numpy as np
import warnings
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, roc_auc_score, precision_recall_curve, f1_score
from sklearn.feature_selection import mutual_info_classif
from baseline import calculate_donchian_and_signals, calculate_volatility, calculate_position_size, calculate_backtest_metrics, print_metrics_table


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')
optuna.logging.set_verbosity(optuna.logging.WARNING)


LOOK_BACK_WINDOWS = [5, 10, 20, 30, 60, 90, 150, 250, 360]
TARGET_VOLATILITY = 0.25
MAX_ALLOCATION = 2.0
VOLATILITY_WINDOW = 90
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0


ML_ENABLED = True
ML_TRAIN_END_DATE = "2023-01-01"
ML_VALID_END_DATE = "2024-01-01"
N_OPTUNA_TRIALS = 100
N_CV_SPLITS = 5
FEATURE_SELECTION_THRESHOLD = 0.001
HIGH_CORRELATION_THRESHOLD = 0.95


FORWARD_LOOK_PERIOD = 20
TARGET_PROFIT_ATR = 3.0
STOP_LOSS_ATR = 1.5
ATR_PERIOD_TARGET = 14


FEATURE_COLS = [
    'rsi_14', 'mom_10d', 'mom_30d', 'stochk_14_3_3', 'stochd_14_3_3',
    'macd_hist', 'adx_14', 'plus_di_14', 'minus_di_14', 'sma_50_ratio', 'ema_20_ratio',
    'atr_14_norm', 'bbands_width_20_2', 'volatility_30d', 'volatility_90d',
    'obv_pct_change_10d',
    'donchian_width_rel_60'
]

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
    df = df.dropna(subset=['close', 'high', 'low', 'asset_return'])


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

def train_ml_model(data: pd.DataFrame, timeframe: int):
    print(f"\n--- Training ML Model for Timeframe {timeframe} ---")
    features_df = data.copy()
    target_col = 'target'
    signal_event_col = f'signal_event_{timeframe}'

    dcu_col = f'DCU_{timeframe}'

    features_df[signal_event_col] = (features_df['close'] >= features_df[dcu_col]) & \
                                   (features_df[f'Pos_{timeframe}'].shift(1).fillna(0) == 0)
    features_df[signal_event_col] = features_df[signal_event_col].fillna(False)

    if target_col not in features_df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Run create_target_variable_atr first.")

    ml_data = features_df[features_df[signal_event_col]].copy()
    ml_data = ml_data.dropna(subset=FEATURE_COLS + [target_col])

    if ml_data.empty or len(ml_data) < (N_CV_SPLITS + 2) * 5:
        print(f"Not enough signal events ({len(ml_data)}) for training/validation/testing. Cannot train ML model.")
        return None, None, 0.5, FEATURE_COLS

    print(f"Total signal events for ML ({timeframe}d): {len(ml_data)}")
    X = ml_data[FEATURE_COLS]
    y = ml_data[target_col]

    train_indices = X.index < ML_TRAIN_END_DATE
    valid_indices = (X.index >= ML_TRAIN_END_DATE) & (X.index < ML_VALID_END_DATE)
    test_indices = X.index >= ML_VALID_END_DATE

    X_train, y_train = X[train_indices], y[train_indices]
    X_valid, y_valid = X[valid_indices], y[valid_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print(f"Train samples: {len(X_train)}, Valid samples: {len(X_valid)}, Test samples: {len(X_test)}")

    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        print("Not enough training data or only one class present. Cannot train.")
        return None, None, 0.5, FEATURE_COLS

    print("Performing feature selection...")
    selected_features = list(X_train.columns)

    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] > HIGH_CORRELATION_THRESHOLD)]
    selected_features = [f for f in selected_features if f not in to_drop_corr]
    print(f"Dropped due to high correlation: {to_drop_corr}")
    print(f"Features remaining after correlation filter: {len(selected_features)}")

    X_train = X_train[selected_features]
    X_valid = X_valid[selected_features] if not X_valid.empty else X_valid
    X_test = X_test[selected_features] if not X_test.empty else X_test

    if len(selected_features) > 1 and len(X_train) > 0:
         try:
            mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
            mi_series = pd.Series(mi_scores, index=selected_features).sort_values(ascending=False)
            to_keep_mi = mi_series[mi_series > FEATURE_SELECTION_THRESHOLD].index.tolist()
            if len(to_keep_mi) < 2:
                 print("Warning: MI threshold too high, keeping top 5 features instead.")
                 to_keep_mi = mi_series.head(min(5, len(selected_features))).index.tolist()

            to_drop_mi = [f for f in selected_features if f not in to_keep_mi]
            selected_features = to_keep_mi
            print(f"Dropped due to low mutual information: {to_drop_mi}")
            print(f"Final selected features: {len(selected_features)}")

            X_train = X_train[selected_features]
            X_valid = X_valid[selected_features] if not X_valid.empty else X_valid
            X_test = X_test[selected_features] if not X_test.empty else X_test
         except Exception as e:
             print(f"Warning: Error during MI feature selection - {e}. Keeping all features after correlation.")


    if not selected_features:
        print("Error: No features selected. Cannot train.")
        return None, None, 0.5, FEATURE_COLS

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid) if not X_valid.empty else np.array([])
    X_test_scaled = scaler.transform(X_test) if not X_test.empty else np.array([])

    scale_pos_weight = len(y_train[y_train == 0]) / (len(y_train[y_train == 1]) + 1e-9)
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    print("Starting hyperparameter tuning with Optuna...")
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

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

        for train_idx, val_idx in tscv.split(X_train_scaled):
             X_fold_train, y_fold_train = X_train_scaled[train_idx], y_train.iloc[train_idx]
             X_fold_val, y_fold_val = X_train_scaled[val_idx], y_train.iloc[val_idx]

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

    study = optuna.create_study(direction='maximize')
    try:
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
        best_params = study.best_params
        print(f"Optuna best AUC: {study.best_value:.4f}")
        print(f"Optuna best params: {best_params}")
    except Exception as e:
         print(f"Warning: Optuna optimization failed - {e}. Using default parameters.")
         best_params = {}

    print("Training final LightGBM model with best parameters...")
    final_params = {
        'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
        'boosting_type': 'gbdt', 'seed': 42, 'scale_pos_weight': scale_pos_weight
    }
    final_params.update(best_params)

    lgb_clf_final = lgb.LGBMClassifier(**final_params)
    lgb_clf_final.fit(X_train_scaled, y_train,
                     eval_set=[(X_valid_scaled, y_valid)] if not X_valid.empty else None,
                     eval_metric='auc',
                     callbacks=[lgb.early_stopping(10, verbose=False)] if not X_valid.empty else None)

    optimal_threshold = 0.5
    if not X_valid.empty and len(np.unique(y_valid)) > 1:
        print("Evaluating on validation set to find threshold...")
        y_pred_proba_valid = lgb_clf_final.predict_proba(X_valid_scaled)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_valid, y_pred_proba_valid)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-9)
        valid_f1_scores = f1_scores[:-1]
        valid_thresholds = thresholds

        if len(valid_thresholds) > 0:
            optimal_idx = np.argmax(valid_f1_scores)
            optimal_threshold = valid_thresholds[optimal_idx]
            print(f"Optimal threshold (Max F1 on Valid) found: {optimal_threshold:.4f}")
            print(f"Validation F1 @ optimal threshold: {valid_f1_scores[optimal_idx]:.4f}")
            print(f"Validation Precision @ optimal threshold: {precision[optimal_idx]:.4f}")
            print(f"Validation Recall @ optimal threshold: {recall[optimal_idx]:.4f}")
        else:
             print("Warning: Could not find threshold from validation set. Using 0.5.")

    if not X_test.empty and len(np.unique(y_test)) > 1:
        y_pred_proba_test = lgb_clf_final.predict_proba(X_test_scaled)[:, 1]
        y_pred_test = (y_pred_proba_test >= optimal_threshold).astype(int)

        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
        test_auc = roc_auc_score(y_test, y_pred_proba_test)
        test_logloss = log_loss(y_test, y_pred_proba_test)

        print(f"\n--- Test Set Evaluation Metrics (Threshold = {optimal_threshold:.4f}) ---")
        print(f"Accuracy:  {test_accuracy:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall:    {test_recall:.4f}")
        print(f"F1 Score:  {test_f1:.4f}")
        print(f"AUC:       {test_auc:.4f}")
        print(f"Log Loss:  {test_logloss:.4f}")
        print("----------------------------------------------------")


    print("--- ML Model Training Complete ---")


    return lgb_clf_final, scaler, optimal_threshold, selected_features


def calculate_donchian_and_signals_ml(
    data: pd.DataFrame, timeframe: int, ml_model, scaler, ml_threshold: float, selected_features: list
) -> pd.DataFrame:
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

    df_index = df.index
    for i in range(1, len(df)):
        idx_curr = df_index[i]
        idx_prev = df_index[i-1]
        current_close = df.loc[idx_curr, 'close']
        prev_pos = df.loc[idx_prev, pos_col]
        prev_stop = df.loc[idx_prev, stop_col]
        current_dcm = df.loc[idx_curr, dcm_col]
        current_dcu = df.loc[idx_curr, dcu_col]
        prev_dcm = df.loc[idx_prev, dcm_col]

        ml_passes = False
        potential_entry = (prev_pos == 0 and not np.isnan(current_dcu) and current_close >= current_dcu)

        if potential_entry and ml_model is not None:
            try:
                current_features_scaled = df.loc[[idx_curr], scaled_feature_cols].values
                if not np.isnan(current_features_scaled).any():
                     probability = ml_model.predict_proba(current_features_scaled)[0, 1]
                     if probability >= ml_threshold:
                         ml_passes = True
            except Exception as e:
                 ml_passes = False

        if potential_entry and ml_passes:
            df.loc[idx_curr, pos_col] = 1.0
            df.loc[idx_curr, stop_col] = current_dcm

        elif prev_pos == 1 and not np.isnan(prev_stop) and current_close <= prev_stop:
             df.loc[idx_curr, pos_col] = 0.0
             df.loc[idx_curr, stop_col] = np.nan

        elif prev_pos == 1:
             df.loc[idx_curr, pos_col] = 1.0
             if not np.isnan(prev_stop) and not np.isnan(prev_dcm):
                 df.loc[idx_curr, stop_col] = max(prev_stop, prev_dcm)
             elif not np.isnan(prev_stop):
                 df.loc[idx_curr, stop_col] = prev_stop

        else:
            df.loc[idx_curr, pos_col] = 0.0
            df.loc[idx_curr, stop_col] = np.nan

    if 'scaled_feature_cols' in locals() and scaled_feature_cols:
        df = df.drop(columns=scaled_feature_cols, errors='ignore')
    return df

def main(args):

    try:
        base_data = pd.read_csv(args.data_path, index_col='timestamp', parse_dates=True)
        base_data.index = pd.to_datetime(base_data.index)
        base_data = base_data.dropna(subset=['open', 'high', 'low', 'close'])
        if 'volume' not in base_data.columns: base_data['volume'] = 0
        base_data['volume'] = base_data['volume'].fillna(0)
    except Exception as e:
        print(f"Error loading or parsing data: {e}")
        return

    if args.start_date:
        base_data = base_data[base_data.index >= pd.to_datetime(args.start_date)]
    if args.end_date:
        base_data = base_data[base_data.index <= pd.to_datetime(args.end_date)]
    if base_data.empty:
        print("Error: No data remaining after date filtering.")
        return
    print(f"Data shape after filtering: {base_data.shape}")

    data_with_indicators = calculate_volatility(base_data)
    data_with_indicators['asset_return'] = data_with_indicators['close'].pct_change()
    data_with_indicators = calculate_ml_features(data_with_indicators)
    data_with_indicators = create_target_variable_atr(data_with_indicators)

    ml_models, ml_scalers, ml_thresholds, ml_selected_features = {}, {}, {}, {}

    if ML_ENABLED:
        print("\n--- Starting ML Model Training Phase ---")
        temp_data_for_training = data_with_indicators.copy()
        for look_back_window in LOOK_BACK_WINDOWS:
             print(f"Pre-calculating Donchian/Pos for ML training ({look_back_window}d)...")
             temp_data_for_training = calculate_donchian_and_signals(temp_data_for_training, look_back_window)

        for look_back_window in LOOK_BACK_WINDOWS:
             model, scaler, threshold, selected_features = train_ml_model(temp_data_for_training, look_back_window)
             ml_models[look_back_window] = model
             ml_scalers[look_back_window] = scaler
             ml_thresholds[look_back_window] = threshold
             ml_selected_features[look_back_window] = selected_features
        print("--- ML Model Training Phase Complete ---")
    else:
        print("\nML Training and Filtering is disabled.")
        for look_back_window in LOOK_BACK_WINDOWS:
             ml_selected_features[look_back_window] = FEATURE_COLS

    print("\n--- Starting Backtesting Phase ---")
    all_data = data_with_indicators.copy()
    all_metrics = {}
    weight_columns = []

    for look_back_window in LOOK_BACK_WINDOWS:
        print(f"Backtesting lookback window: {look_back_window} days...")
        window_str = f"{look_back_window}d"

        model = ml_models.get(look_back_window) if ML_ENABLED else None
        scaler = ml_scalers.get(look_back_window) if ML_ENABLED else None
        threshold = ml_thresholds.get(look_back_window, 0.5) if ML_ENABLED else 0.0
        selected_features = ml_selected_features.get(look_back_window, FEATURE_COLS)

        if ML_ENABLED and (model is None or scaler is None):
             print(f"Warning: ML Model/Scaler not available for {look_back_window}d. Running without ML filter.")
             all_data = calculate_donchian_and_signals(all_data, look_back_window)
        elif ML_ENABLED:
            all_data = calculate_donchian_and_signals_ml(
                data=all_data,
                timeframe=look_back_window,
                ml_model=model,
                scaler=scaler,
                ml_threshold=threshold,
                selected_features=selected_features
            )
        else:
             all_data = calculate_donchian_and_signals(all_data, look_back_window)

        all_data = calculate_position_size(all_data, look_back_window)

        weight_col = f"w_{look_back_window}"
        weight_columns.append(weight_col)
        returns_col = f"strategy_return_{look_back_window}"
        all_data[returns_col] = all_data[weight_col].shift(1) * all_data['asset_return']
        current_returns = all_data[returns_col].dropna()
        if not current_returns.empty:
             metrics = calculate_backtest_metrics(
                 returns=current_returns,
                 asset_returns=all_data['asset_return']
             )
             all_metrics[window_str] = metrics
             print(f"Finished backtesting & metrics for window: {look_back_window}")
        else:
             print(f"Skipping metrics for window {look_back_window}: No returns.")
             all_metrics[window_str] = {"error": "No returns generated"}

    all_data['w_Combo'] = all_data[weight_columns].mean(axis=1)
    all_data['strategy_return_Combo'] = all_data['w_Combo'].shift(1) * all_data['asset_return']
    combo_returns = all_data['strategy_return_Combo'].dropna()

    if not combo_returns.empty:
        combo_metrics = calculate_backtest_metrics(
            returns=combo_returns,
            asset_returns=all_data['asset_return']
        )
        all_metrics['Combo'] = combo_metrics
    else:
        all_metrics['Combo'] = {"error": "No returns generated"}

    print("\n--- Backtest Metrics Summary ---")
    print_metrics_table(all_metrics, ml_flag=True)


    output_path = args.data_path.replace(".csv", "_strategy_output_ml_enhanced.csv")
    try:
        cols_to_save = ['open', 'high', 'low', 'close', 'volume', 'sigma_t', 'asset_return', 'w_Combo', 'strategy_return_Combo', 'target']
        cols_to_save.extend(weight_columns)
        cols_to_save.extend([f'strategy_return_{lw}' for lw in LOOK_BACK_WINDOWS])
        cols_to_save = [col for col in cols_to_save if col in all_data.columns]
        all_data[cols_to_save].to_csv(output_path)
        print(f"\nSubset of results saved to {output_path}")
    except Exception as e:
        print(f"\nError saving results to CSV: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implement and Backtest Donchian Ensemble Strategy with ML Signal Filtering and Tuning.")
    parser.add_argument( "--data_path", type=str,
                         default="/Users/a.tikhonov/ml_dl_projects/applied_ai_blockchain/data/1d_timeframe/spot/binance_spot_ETH_USDT_1d_20150425_20250425.csv",
                         help="Path to CSV data.")
    parser.add_argument("--start_date", type=str, default=None, help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default=None, help="Backtest end date (YYYY-MM-DD).")

    parser.add_argument("--disable_ml", action="store_true", help="Disable ML training and filtering.")

    args = parser.parse_args()
    if args.disable_ml:
        ML_ENABLED = False

    main(args)
