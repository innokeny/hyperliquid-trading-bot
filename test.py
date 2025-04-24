from hyperliquid.info import Info
from hyperliquid.utils.constants import TESTNET_API_URL, MAINNET_API_URL


info = Info(base_url=TESTNET_API_URL, )
# info.subscribe({"type": "l2Book", "coin": "ETH"}, print)
info.subscribe({"type": "candle", "coin": "ETH", "interval": "1h"}, print)
# info.subscribe({"type": "candle", "coin": "ETH", "interval": "3m"}, print)
# info.subscribe({"type": "candle", "coin": "ETH", "interval": "5m"}, print)
# info.subscribe({"type": "candle", "coin": "ETH", "interval": "15m"}, print)
print('done')