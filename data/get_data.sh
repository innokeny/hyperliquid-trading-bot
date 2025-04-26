#! /bin/bash

python fetch_spot_data_cexs.py \
    --exchange bybit \
    --symbol ETH/USDT:USDT \
    --timeframe 1d \
    --years 10 \
    --output_dir /Users/a.tikhonov/ml_dl_projects/applied_ai_blockchain/data/1d_timeframe/perps

python fetch_spot_data_cexs.py \
    --exchange binance \
    --symbol ETH/USDT:USDT \
    --timeframe 1d \
    --years 10 \
    --output_dir /Users/a.tikhonov/ml_dl_projects/applied_ai_blockchain/data/1d_timeframe/perps

