from src.utils.token import Token

class EtherscanFetcher:
    def __init__(self, etherscan_token: Token):
        self.etherscan_token = etherscan_token