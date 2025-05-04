#!/bin/bash

python fetch_spot_data_cexs.py \
    --exchange bybit \
    --symbol "ETH/USDT:USDT" \
    --timeframe 1d \
    --years 10 \
    --output_dir ../data/1d_timeframe/bybit

python fetch_spot_data_cexs.py \
    --exchange binance \
    --symbol "ETH/USDT:USDT" \
    --timeframe 1d \
    --years 10 \
    --output_dir ../data/1d_timeframe/binance

python market.py \
    --token ETH \
    --interval 1d \
    --output ../data/1d_timeframe/hyperliquid/hyperliquid_ETH_USDT_1d_20150425_20250425.csv 

