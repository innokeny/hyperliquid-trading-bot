import os
from typing import Optional, Tuple
from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.constants import TESTNET_API_URL
from loguru import logger
from src.settings import settings

class HyperliquidConnection:
    """Handles authentication and connection to Hyperliquid exchange."""
    
    def __init__(self):
        """Initialize the connection handler."""
        self.account: Optional[LocalAccount] = None
        self.address: Optional[str] = None
        self.exchange: Optional[Exchange] = None
        self.info: Optional[Info] = None
        
    def setup(self, base_url: str = TESTNET_API_URL, skip_ws: bool = False) -> Tuple[str, Info, Exchange]:
        """
        Set up the connection to Hyperliquid.
        
        Args:
            base_url: Optional base URL for the API. If None, uses default.
            skip_ws: Whether to skip WebSocket connection.
            
        Returns:
            Tuple containing (address, info, exchange)
        """
        try:
            self.account = Account.from_key(settings.HYPERLIQUID_SECRET_KEY)
            self.address = settings.HYPERLIQUID_ACCOUNT_ADDRESS
            
            if not self.address:
                self.address = self.account.address
                logger.info(f"Using account address: {self.address}")
            elif self.address != self.account.address:
                logger.info(f"Using agent address: {self.account.address}")
                
            self.info = Info(base_url, skip_ws)
            self.exchange = Exchange(self.account, base_url, account_address=self.address)
            
            self._verify_account_equity()
            
            logger.success("Successfully connected to Hyperliquid")
            return self.address, self.info, self.exchange
            
        except Exception as e:
            logger.error(f"Failed to setup connection: {str(e)}")
            raise
            
    def _verify_account_equity(self) -> None:
        """Verify that the account has sufficient equity."""
        if not self.info or not self.address:
            raise RuntimeError("Connection not established")
            
        user_state = self.info.user_state(self.address)
        spot_user_state = self.info.spot_user_state(self.address)
        margin_summary = user_state["marginSummary"]
        
        if float(margin_summary["accountValue"]) == 0 and len(spot_user_state["balances"]) == 0:
            url = self.info.base_url.split(".", 1)[1]
            error_msg = (
                f"No account value found.\n"
                f"If you think this is a mistake, make sure that {self.address} has a balance on {url}.\n"
                f"If the address shown is your API wallet address, update the config to specify the address "
                f"of your account, not the address of the API wallet."
            )
            raise ValueError(error_msg)
            
    def get_connection(self) -> Tuple[Info, Exchange]:
        """
        Get the current connection objects.
        
        Returns:
            Tuple containing (info, exchange)
            
        Raises:
            RuntimeError: If connection is not established
        """
        if not self.info or not self.exchange:
            raise RuntimeError("Connection not established. Call setup() first.")
        return self.info, self.exchange 