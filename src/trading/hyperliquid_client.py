import aiohttp
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class HyperliquidClient:
    """Client for interacting with the Hyperliquid API."""

    def __init__(self, api_url: str, account_address: str, secret_key: str):
        """
        Initialize the Hyperliquid client.

        Args:
            api_url: Base URL for the Hyperliquid API
            account_address: Account address for authentication
            secret_key: Secret key for authentication
        """
        self.api_url = api_url
        self.account_address = account_address
        self.secret_key = secret_key
        self.session = None
        logger.info("Hyperliquid client initialized")

    async def connect(self):
        """Establish connection to the API."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            logger.info("Connected to Hyperliquid API")

    async def disconnect(self):
        """Close the API connection."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Disconnected from Hyperliquid API")

    async def _ensure_session(self):
        """Ensure the session is connected."""
        if self.session is None:
            await self.connect()

    async def get_order_book(self) -> Dict[str, Any]:
        """Get the current order book."""
        try:
            await self._ensure_session()
            async with self.session.get(f"{self.api_url}/orderbook") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error getting order book: {str(e)}")
            raise

    async def get_recent_trades(self) -> Dict[str, Any]:
        """Get recent trades."""
        try:
            await self._ensure_session()
            async with self.session.get(f"{self.api_url}/trades") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error getting recent trades: {str(e)}")
            raise

    async def get_current_price(self) -> float:
        """Get the current market price."""
        try:
            await self._ensure_session()
            async with self.session.get(f"{self.api_url}/price") as response:
                response.raise_for_status()
                data = await response.json()
                return float(data['price'])
        except Exception as e:
            logger.error(f"Error getting current price: {str(e)}")
            raise

    async def get_balance(self) -> float:
        """Get the account balance."""
        try:
            await self._ensure_session()
            headers = {
                'X-API-Key': self.secret_key,
                'X-API-Address': self.account_address
            }
            async with self.session.get(f"{self.api_url}/balance", headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                return float(data['balance'])
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            raise

    async def get_position(self) -> Optional[Dict[str, Any]]:
        """Get the current position."""
        try:
            await self._ensure_session()
            headers = {
                'X-API-Key': self.secret_key,
                'X-API-Address': self.account_address
            }
            async with self.session.get(f"{self.api_url}/position", headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                return data if data else None
        except Exception as e:
            logger.error(f"Error getting position: {str(e)}")
            raise

    async def place_order(self, order: Dict[str, Any]) -> str:
        """Place a new order."""
        try:
            await self._ensure_session()
            headers = {
                'X-API-Key': self.secret_key,
                'X-API-Address': self.account_address
            }
            async with self.session.post(f"{self.api_url}/order", json=order, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                return data['order_id']
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise

    async def close_position(self) -> str:
        """Close the current position."""
        try:
            await self._ensure_session()
            headers = {
                'X-API-Key': self.secret_key,
                'X-API-Address': self.account_address
            }
            async with self.session.post(f"{self.api_url}/close", headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                return data['order_id']
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            raise 