import pandas as pd
import os

# Read the data
df = pd.read_csv('data/1d_timeframe/binance/binance_ETH_USDT_1d_20150427_20250427.csv')

# Calculate the split point (80% of the data)
split_point = int(len(df) * 0.8)

# Split the data
train_data = df[:split_point]
test_data = df[split_point:]

# Create output directory if it doesn't exist
os.makedirs('data/1d_timeframe/binance/train', exist_ok=True)
os.makedirs('data/1d_timeframe/binance/test', exist_ok=True)

# Save the split datasets
train_data.to_csv('data/1d_timeframe/binance/train/binance_ETH_USDT_1d_train.csv', index=False)
test_data.to_csv('data/1d_timeframe/binance/test/binance_ETH_USDT_1d_test.csv', index=False)

print(f"Total data points: {len(df)}")
print(f"Training set size: {len(train_data)} ({len(train_data)/len(df)*100:.1f}%)")
print(f"Testing set size: {len(test_data)} ({len(test_data)/len(df)*100:.1f}%)") 