from typing import Any
from src.utils.token import Token
from .etherscan import EtherscanFetcher
from .hyperliquid import HyperliquidFetcher

class Fetcher:
    def __init__(self, hyperliquid_token: Token, etherscan_token: Token):
        self.hyperliquid_token = hyperliquid_token
        self.etherscan_token = etherscan_token
    

    def fetch(self) -> dict[str, Any]:
        pass