import asyncio
import argparse
from dateutil.relativedelta import relativedelta
from datetime import datetime, timezone

import pandas as pd
import ccxt.async_support as ccxt


MILLISECONDS_IN = {
    "minute": 60 * 1000,
    "hour": 60 * 60 * 1000,
    "day": 24 * 60 * 60 * 1000,
}


async def get_bybit_spot_candles_ccxt(
    exchange: str = "bybit",
    symbol: str = "ETH/USDT",
    timeframe: str = "1d",
    start_dt: datetime = None,
    end_dt: datetime = None,
    limit_per_request: int = 1000,
) -> pd.DataFrame:
    """
    Fetches historical OHLCV data for a spot pair from Bybit using CCXT,
    handling pagination.

    Args:
        symbol: Trading pair symbol in CCXT format (e.g., 'ETH/USDT').
        timeframe: CCXT standard timeframe (e.g., '1m', '5m', '1h', '1d').
        start_dt: Start datetime (timezone-aware UTC recommended). If None, fetches recent data.
        end_dt: End datetime (timezone-aware UTC recommended). If None, fetches up to now.
        limit_per_request: Number of candles per API call (max 1000 for Bybit).

    Returns:
        Pandas DataFrame with OHLCV data, or None if an error occurs.
    """

    if exchange in ["bybit", "binance"]:
        if exchange == "bybit":
            exchange = ccxt.bybit(
                {
                    "enableRateLimit": True,
                }
            )
        elif exchange == "binance":
            exchange = ccxt.binance(
                {
                    "enableRateLimit": True,
                }
            )
    else:
        raise ValueError(f"Exchange {exchange} not supported")

    if not exchange.has["fetchOHLCV"]:
        print(f"Error: The exchange {exchange.id} does not support fetchOHLCV.")
        await exchange.close()
        return None

    if timeframe not in exchange.timeframes:
        print(f"Error: Timeframe '{timeframe}' not supported by {exchange.id}.")
        print(f"Supported timeframes: {list(exchange.timeframes.keys())}")
        await exchange.close()
        return None

    all_ohlcv = []

    since_ms = int(start_dt.timestamp() * 1000) if start_dt else None
    end_ms = int(end_dt.timestamp() * 1000) if end_dt else None

    timeframe_duration_ms = exchange.parse_timeframe(timeframe) * 1000

    print(f"Fetching {symbol} {timeframe} candles from {exchange.id}...")

    try:
        while True:
            print(
                f"  Fetching batch starting around: {datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc) if since_ms else 'latest available'}"
            )
            ohlcv = await exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=limit_per_request,
            )

            if not ohlcv:
                print("  No more data returned.")
                break

            print(f"  Received {len(ohlcv)} candles.")
            last_candle_ts = ohlcv[-1][0]

            all_ohlcv.extend(ohlcv)

            next_since_ms = last_candle_ts + timeframe_duration_ms

            if end_ms is not None and next_since_ms >= end_ms:
                print("  Reached or exceeded end date.")
                break

            if len(ohlcv) < limit_per_request:
                print(
                    "  Received fewer candles than limit, assuming end of available data."
                )
                break

            since_ms = next_since_ms

    except ccxt.NetworkError as e:
        print(f"Network Error: {e}")
        return None
    except ccxt.ExchangeError as e:
        print(f"Exchange Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    finally:
        print("Closing exchange connection.")
        await exchange.close()

    if not all_ohlcv:
        print("No data fetched.")
        return None

    df = pd.DataFrame(
        all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    if start_dt:
        df = df[df.index >= start_dt]
    if end_dt:
        df = df[df.index <= end_dt]

    print(f"Finished fetching. Total unique candles: {len(df)}")
    return df


async def main(args: argparse.Namespace):
    end_date = datetime.now(timezone.utc)
    start_date = end_date - relativedelta(years=args.years)

    eth_candles_d = await get_bybit_spot_candles_ccxt(
        exchange=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_dt=start_date,
        end_dt=end_date,
    )

    if eth_candles_d is not None:
        print("\n--- Fetched ETH/USDT Daily Candles via CCXT (Last 5 rows) ---")
        print(eth_candles_d.tail())

        filename = f"{args.output_dir}/{args.exchange}_{args.symbol.replace('/', '_')}_{args.timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        eth_candles_d.to_csv(filename)
        print(f"\nData saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", type=str, default="bybit")
    parser.add_argument("--symbol", type=str, default="ETH/USDT")
    parser.add_argument("--timeframe", type=str, default="1d")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    args = parser.parse_args()
    asyncio.run(main(args))
