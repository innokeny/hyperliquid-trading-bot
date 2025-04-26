import argparse
import pandas as pd
import pandas_ta as ta
import numpy as np

LOOK_BACK_WINDOWS = [5, 10, 20, 30, 60, 90, 150, 250, 360]
TARGET_VOLATILITY = 0.25
MAX_ALLOCATION = 2.0
VOLATILITY_WINDOW = 90
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0


def get_pretty_print(metrics: dict):
    print(
        f"Period: {metrics['Start Date']} to {metrics['End Date']} ({metrics['Duration (Years)']:.2f} years)"
    )
    print(f"CAGR:                          {metrics['CAGR (%)']:.2f}%")
    print(f"Annualized Volatility:         {metrics['Annualized Volatility (%)']:.2f}%")
    print(f"Sharpe Ratio:                  {metrics['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio:                 {metrics['Sortino Ratio']:.2f}")
    print(f"Max Drawdown:                  {metrics['Max Drawdown (%)']:.2f}%")
    print(f"Calmar Ratio:                  {metrics['Calmar Ratio']:.2f}")
    print("-" * 40)
    print(f"Cumulative Return (Strategy):  {metrics['Cumulative Return (%)']:.2f}%")
    if "Buy & Hold Return (%)" in metrics:
        print(f"Cumulative Return (Buy & Hold):{metrics['Buy & Hold Return (%)']:.2f}%")
    print("-" * 40)
    print(f"Total Trades:                  {metrics['Total Trades']}")
    print(f"Win Rate:                      {metrics['Win Rate (%)']}")
    print(f"Gain:Loss Ratio:               {metrics['Gain:Loss Ratio']}")


def print_metrics_table(strategy_metrics: dict, ml_flag: bool = False):
    """Prints a formatted table of backtest metrics."""

    if "error" in strategy_metrics:
        print(f"Error calculating metrics: {strategy_metrics['error']}")
    else:
        if ml_flag:
            print("ML Strategy Metrics:")
            for look_back_window in strategy_metrics.keys():
                print(f"Lookback Window: {look_back_window} days")
                get_pretty_print(strategy_metrics[look_back_window])
                print("\n")

        else:
            get_pretty_print(strategy_metrics)


def calculate_volatility(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates 90-day annualized volatility of log returns."""
    df = data.copy()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    rolling_std = (
        df["log_returns"]
        .rolling(window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW // 2)
        .std()
    )
    # Annualize the volatility
    df["sigma_t"] = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    df["sigma_t"] = df["sigma_t"].bfill().ffill()
    df["sigma_t"] = df["sigma_t"].replace(0, 1e-6)
    df = df.drop(columns=["log_returns"])
    return df


def calculate_donchian_and_signals(data: pd.DataFrame, timeframe: int) -> pd.DataFrame:
    """
    Calculates Donchian Channels (based on close), Trailing Stop, and Trading Signal (Pos)
    for a given timeframe.
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
            df.loc[idx_curr, stop_col] = (
                current_dcm  # Initial stop at midpoint of *current* day entry occurs
            )

        # Exit condition: previous position was long AND close hits trailing stop from previous day
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


def calculate_position_size(data: pd.DataFrame, timeframe: int) -> pd.DataFrame:
    """Calculates the position weight (w_n_t) for a given timeframe."""
    df = data.copy()
    pos_col = f"Pos_{timeframe}"
    weight_col = f"w_{timeframe}"

    # Calculate raw weight based on volatility target
    raw_weight = TARGET_VOLATILITY / df["sigma_t"]

    # Apply maximum allocation cap
    capped_weight = np.minimum(raw_weight, MAX_ALLOCATION)

    # Apply the position signal (0 or 1)
    df[weight_col] = capped_weight * df[pos_col]
    df[weight_col] = df[weight_col].fillna(0)

    return df


def calculate_backtest_metrics(
    returns: pd.Series, asset_returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE
) -> dict:
    """Calculates performance metrics for a given return series."""
    metrics = {}

    returns = returns.dropna()
    if returns.empty:
        return {"error": "Return series is empty after dropna."}

    metrics["Start Date"] = returns.index[0].strftime("%Y-%m-%d")
    metrics["End Date"] = returns.index[-1].strftime("%Y-%m-%d")
    metrics["Duration (Years)"] = (returns.index[-1] - returns.index[0]).days / 365.25

    cumulative_returns = (1 + returns).cumprod()
    metrics["Cumulative Return (%)"] = (cumulative_returns.iloc[-1] - 1) * 100

    total_years = metrics["Duration (Years)"]
    if total_years > 0:
        metrics["CAGR (%)"] = (cumulative_returns.iloc[-1]) ** (1 / total_years) - 1
        metrics["CAGR (%)"] *= 100
    else:
        metrics["CAGR (%)"] = 0

    metrics["Annualized Volatility (%)"] = (
        returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    )

    if metrics["Annualized Volatility (%)"] > 1e-6:
        mean_annual_return = returns.mean() * TRADING_DAYS_PER_YEAR
        metrics["Sharpe Ratio"] = (mean_annual_return - risk_free_rate) / (
            metrics["Annualized Volatility (%)"] / 100
        )
    else:
        metrics["Sharpe Ratio"] = np.nan

    # Sortino Ratio
    negative_returns = returns[returns < 0]
    if not negative_returns.empty:
        downside_std = negative_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if downside_std > 1e-6:
            metrics["Sortino Ratio"] = (
                mean_annual_return - risk_free_rate
            ) / downside_std
        else:
            metrics["Sortino Ratio"] = np.nan
    else:  # No losing days
        metrics["Sortino Ratio"] = np.inf

    # Max Drawdown
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    metrics["Max Drawdown (%)"] = drawdown.min() * 100

    # Calmar Ratio
    if abs(metrics["Max Drawdown (%)"]) > 1e-6:
        # Use CAGR for Calmar ratio calculation
        metrics["Calmar Ratio"] = (metrics["CAGR (%)"] / 100) / abs(
            metrics["Max Drawdown (%)"] / 100
        )
    else:
        metrics["Calmar Ratio"] = np.nan

    if asset_returns is not None:
        asset_returns = asset_returns.loc[returns.index].dropna()
        if not asset_returns.empty:
            metrics["Buy & Hold Return (%)"] = (
                (1 + asset_returns).cumprod().iloc[-1] - 1
            ) * 100
            metrics["Beta (vs Asset)"] = "Not Calculated"

    # --- Trade Metrics (Placeholder - Requires Trade Identification Logic) ---
    # These metrics require identifying individual trades (entry/exit points)
    # which is complex for the ensemble signal. We omit them here.
    metrics["Total Trades"] = "Not Calculated (Ensemble)"
    metrics["Win Rate (%)"] = "Not Calculated (Ensemble)"
    metrics["Avg PnL x Unit"] = "Not Calculated (Ensemble)"
    metrics["Gain:Loss Ratio"] = "Not Calculated (Ensemble)"

    return metrics


def main(args):
    try:
        base_data = pd.read_csv(args.data_path, index_col="timestamp", parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.data_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if not all(
        col in base_data.columns for col in ["open", "high", "low", "close", "volume"]
    ):
        raise ValueError(
            "CSV must contain 'timestamp', 'open', 'high', 'low', 'close', 'volume' columns."
        )

    print(
        f"Loaded data from {base_data.index.min()} to {base_data.index.max()} ({len(base_data)} rows)"
    )

    if args.start_date:
        base_data = base_data[base_data.index >= pd.to_datetime(args.start_date)]
    if args.end_date:
        base_data = base_data[base_data.index <= pd.to_datetime(args.end_date)]

    if base_data.empty:
        print("Error: No data remaining after date filtering.")
        return
    print(f"Data shape after filtering: {base_data.shape}")

    # Calculate Volatility (needed for all lookbacks)
    data_with_vol = calculate_volatility(base_data)
    print("Volatility calculated.")

    all_data = data_with_vol.copy()
    weight_columns = []

    for look_back_window in LOOK_BACK_WINDOWS:
        print(f"Processing lookback window: {look_back_window} days...")
        # Calculate Donchian, Signals, and Stops for this window
        all_data = calculate_donchian_and_signals(all_data, look_back_window)

        # Calculate Position Size (Weight) for this window
        all_data = calculate_position_size(all_data, look_back_window)
        weight_columns.append(f"w_{look_back_window}")
        print(f"Finished processing window: {look_back_window}")

    # Calculate Ensemble (Combo) Weight
    all_data["w_Combo"] = all_data[weight_columns].mean(axis=1)
    print("Ensemble 'w_Combo' weight calculated.")

    # Calculate Returns
    all_data["asset_return"] = all_data["close"].pct_change()
    all_data["strategy_return"] = (
        all_data["w_Combo"].shift(1) * all_data["asset_return"]
    )
    all_data = all_data.dropna(subset=["asset_return", "strategy_return"], how="any")

    print("Returns calculated.")

    print("\n--- Backtest Metrics (Ensemble Strategy) ---")
    strategy_metrics = calculate_backtest_metrics(
        returns=all_data["strategy_return"], asset_returns=all_data["asset_return"]
    )

    print_metrics_table(strategy_metrics)

    output_path = args.data_path.replace(".csv", "_strategy_output.csv")
    try:
        all_data.to_csv(output_path)
        print(f"\nFull results saved to {output_path}")
    except Exception as e:
        print(f"\nError saving results to CSV: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Implement and Backtest Donchian Ensemble Trend Following Strategy."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/Users/a.tikhonov/ml_dl_projects/applied_ai_blockchain/data/1d_timeframe/perps/bybit_ETH_USDT:USDT_1d_20150425_20250425.csv",
        help="Path to the CSV file containing OHLCV data with a 'timestamp' index.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,  # e.g., "2018-01-01"
        help="Optional start date for backtesting (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,  # e.g., "2023-12-31"
        help="Optional end date for backtesting (YYYY-MM-DD).",
    )

    args = parser.parse_args()

    try:
        main(args)
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")
        import traceback

        traceback.print_exc()  # Print detailed traceback for debugging
