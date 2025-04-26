import sys
from hyperliquid.info import Info
from hyperliquid.utils.constants import TESTNET_API_URL, MAINNET_API_URL
import pandas as pd
import argparse
from datetime import datetime, timedelta


def fetch_candles_df(token: str, interval: str = '1h', info: Info = Info(TESTNET_API_URL)) -> pd.DataFrame:
    """
    Fetch candle data for a given token and construct a DataFrame.

    Args:
        token (str): Token symbol (e.g. 'BTC', 'ETH')
        interval (str): Candle interval ('5m','15m','1h','4h','1d')
        info (Info): Hyperliquid Info instance for API access

    Returns:
        pd.DataFrame: DataFrame with columns:
            - timestamp (datetime): Start time of the candle
            - end (datetime): End time of the candle
            - open (float): Opening price
            - close (float): Closing price
            - high (float): Highest price
            - low (float): Lowest price
            - volume (float): Trading volume
    """
    start_time = int(datetime.now().timestamp()*1000) - int(timedelta(days=5000).total_seconds()*1000)
    end_time = int(datetime.now().timestamp()*1000)

    candles = info.candles_snapshot(token, interval, start_time, end_time)

    df = pd.DataFrame(candles)

    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df['T'] = pd.to_datetime(df['T'], unit='ms')

    for col in ['o', 'c', 'h', 'l', 'v']:
        df[col] = df[col].astype(float)

    df = df.sort_values('t')
    df.rename(columns={
        'T': 'end',
        'o': 'open',
        'c': 'close',
        'h': 'high',
        'l': 'low',
        'v': 'volume',
        't': 'timestamp',
    }, inplace=True)
    return df

def main(token: str, interval: str, output: str, testnet: bool):
    info = Info(TESTNET_API_URL) if testnet else Info(MAINNET_API_URL)
    df = fetch_candles_df(token, interval, info)
    df.to_csv(output, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--interval", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--testnet", action="store_true", default=False)
    args = parser.parse_args()
    main(args.token, args.interval, args.output, args.testnet)