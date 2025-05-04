from typing import NamedTuple
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import pandas_ta as ta
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, roc_auc_score, precision_recall_curve, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from src.strategy.donchian import DonchianStrategy

class TrainArgs(NamedTuple):
    LOOK_BACK_WINDOWS: list[int] = [5, 10, 20, 30, 60, 90, 150, 250, 360]
    TARGET_VOLATILITY: float = 0.25
    MAX_ALLOCATION: float = 2.0
    VOLATILITY_WINDOW: int = 90
    TRADING_DAYS_PER_YEAR: int = 252
    RISK_FREE_RATE: float = 0.0


    ML_ENABLED = True
    ML_TRAIN_END_DATE: str = "2023-01-01"
    ML_VALID_END_DATE: str = "2024-01-01"
    N_OPTUNA_TRIALS: int = 100
    N_CV_SPLITS: int = 5
    HIGH_CORRELATION_THRESHOLD: float = 0.95


    FORWARD_LOOK_PERIOD: int = 20
    TARGET_PROFIT_ATR: float = 3.0
    STOP_LOSS_ATR: float = 1.5
    ATR_PERIOD_TARGET: int = 14

class Trainer:
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

    def __init__(self, args: TrainArgs):
        self.args = args
        self.strategy = DonchianStrategy(
            LOOK_BACK_WINDOWS = args.LOOK_BACK_WINDOWS,
            TARGET_VOLATILITY = args.TARGET_VOLATILITY,
            MAX_ALLOCATION = args.MAX_ALLOCATION,
            VOLATILITY_WINDOW = args.VOLATILITY_WINDOW,
            TRADING_DAYS_PER_YEAR = args.TRADING_DAYS_PER_YEAR,
            RISK_FREE_RATE = args.RISK_FREE_RATE,
        )

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
    
    def _calc_target(self, data: pd.DataFrame) -> pd.DataFrame:
        TARGET_COL = 'target'
        data[TARGET_COL] = 0
        data[f'atr_{self.args.ATR_PERIOD_TARGET}'] = ta.atr(data['high'], data['low'], data['close'], length=self.args.ATR_PERIOD_TARGET).fillna(method='bfill').fillna(method='ffill') # type: ignore

        for i in range(len(data) - self.args.FORWARD_LOOK_PERIOD):
            entry_price = data['close'].iloc[i]
            entry_atr = data[f'atr_{self.args.ATR_PERIOD_TARGET}'].iloc[i]

            if entry_atr <= 1e-9 or np.isnan(entry_atr):
                continue

            profit_target_price = entry_price + self.args.TARGET_PROFIT_ATR * entry_atr
            stop_loss_price = entry_price - self.args.STOP_LOSS_ATR * entry_atr
         
            hit_target = False
            hit_stop = False

            for j in range(1, self.args.FORWARD_LOOK_PERIOD + 1):
                future_low = data['low'].iloc[i + j]
                future_high = data['high'].iloc[i + j]

                if future_low <= stop_loss_price:
                    hit_stop = True
                    break
                if future_high >= profit_target_price:
                    hit_target = True
                    break
            
            if hit_target and not hit_stop:
                data.iloc[i, data.columns.get_loc(TARGET_COL)] = 1 # type: ignore

        return data

    def _make_dataset(self, data: pd.DataFrame, window: int):
        TARGET_COL = 'target'
        SIGNAL_COL = f'signal_{window}'
        DCU_COL = f'DCU_{window}'
        POS_COL = f'POS_{window}'

        data[SIGNAL_COL] = ((data['close'] >= data[DCU_COL]) & (data[POS_COL].shift(1).fillna(0) == 0)).fillna(False)
        # data[SIGNAL_COL] = (data[POS_COL].fillna(0) != 0).fillna(False)

        dataset = data[data[SIGNAL_COL]].copy().dropna() # type: ignore
        X, y = dataset[self.FEATURE_COLS], dataset[TARGET_COL]
        # X, y = data[self.FEATURE_COLS], data[TARGET_COL]

        train_indices = X.index < self.args.ML_TRAIN_END_DATE
        valid_indices = (X.index >= self.args.ML_TRAIN_END_DATE) & (X.index < self.args.ML_VALID_END_DATE)
        test_indices = X.index >= self.args.ML_VALID_END_DATE

        X_train, y_train = X[train_indices], y[train_indices]
        X_valid, y_valid = X[valid_indices], y[valid_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        # drop features with high correlation
        corr_matrix = X_train.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_with_corr = [column for column in upper_tri.columns if any(upper_tri[column] > self.args.HIGH_CORRELATION_THRESHOLD)]
        
        X_train = X_train.drop(columns=drop_with_corr)
        X_valid = X_valid.drop(columns=drop_with_corr)
        X_test = X_test.drop(columns=drop_with_corr)

        bad = np.where(np.isinf(X_train.values))[0]
        X_train.iloc[bad] = 0.

        bad = np.where(np.isinf(X_test.values))[0]
        X_test.iloc[bad] = 0.

        bad = np.where(np.isinf(X_valid.values))[0]
        X_valid.iloc[bad] = 0.

        X_train.fillna(0)

        return X_train, y_train, X_test, y_test, X_valid, y_valid
    
    def _train_window_filter(self, data: pd.DataFrame, window: int):
        X_train, y_train, X_test, y_test, X_valid, y_valid = self._make_dataset(data, window)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid) if not X_valid.empty else np.array([])
        X_test_scaled = scaler.transform(X_test) if not X_test.empty else np.array([])

        scale_pos_weight = len(y_train[y_train == 0]) / (len(y_train[y_train == 1]) + 1e-9)
        tscv = TimeSeriesSplit(n_splits=self.args.N_CV_SPLITS)

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
                y_pred_proba_cv = model_cv.predict_proba(X_fold_val)[:, 1] # type: ignore
                cv_scores.append(roc_auc_score(y_fold_val, y_pred_proba_cv, average='weighted'))

            return np.mean(cv_scores) if cv_scores else 0.0

        study = optuna.create_study(direction='maximize')
        try:
            study.optimize(objective, n_trials=self.args.N_OPTUNA_TRIALS, show_progress_bar=True) # type: ignore
            best_params = study.best_params
            # print(f"Optuna best AUC: {study.best_value:.4f}")
            # print(f"Optuna best params: {best_params}")
        except Exception as e:
            # print(f"Warning: Optuna optimization failed - {e}. Using default parameters.")
            best_params = {}
        
        final_params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'boosting_type': 'gbdt', 'seed': 42, 'scale_pos_weight': scale_pos_weight
        }

        final_params.update(best_params)

        lgb_clf_final = lgb.LGBMClassifier(**final_params).fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_valid_scaled, y_valid)] if not X_valid.empty else None,
            eval_metric='auc',
            callbacks=[lgb.early_stopping(10, verbose=False)] if not X_valid.empty else None
        )
        optimal_threshold = 0.5
        if not X_valid.empty and len(np.unique(y_valid)) > 1:
            print("Evaluating on validation set to find threshold...")
            y_pred_proba_valid = lgb_clf_final.predict_proba(X_valid_scaled)[:, 1] # type: ignore
            precision, recall, thresholds = precision_recall_curve(y_valid, y_pred_proba_valid)
            f1_scores = 2 * recall * precision / (recall + precision + 1e-9)
            valid_f1_scores = f1_scores[:-1]
            valid_thresholds = thresholds

            if len(valid_thresholds) > 0:
                optimal_idx = np.argmax(valid_f1_scores)
                optimal_threshold = valid_thresholds[optimal_idx]
                # print(f"Optimal threshold (Max F1 on Valid) found: {optimal_threshold:.4f}")
                # print(f"Validation F1 @ optimal threshold: {valid_f1_scores[optimal_idx]:.4f}")
                # print(f"Validation Precision @ optimal threshold: {precision[optimal_idx]:.4f}")
                # print(f"Validation Recall @ optimal threshold: {recall[optimal_idx]:.4f}")
            else:
                
                print("Warning: Could not find threshold from validation set. Using 0.5.")

        if not X_test.empty and len(np.unique(y_test)) > 1:
            y_pred_proba_test = lgb_clf_final.predict_proba(X_test_scaled)[:, 1] # type: ignore
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

        return lgb_clf_final, scaler, optimal_threshold, X_train.columns
    
    def train(self, data: pd.DataFrame):
        data = self.strategy.get_weights(data)
        data = self._calc_features(data)
        data = self._calc_target(data)
        models = {}
        for window in self.strategy.LOOK_BACK_WINDOWS:
            clf, scaler, threshold, columns = self._train_window_filter(data, window)
            models[window] = dict(
                scaler=scaler,
                columns=columns,
                clf=clf,
                threshold=threshold
            )
        
        return models


