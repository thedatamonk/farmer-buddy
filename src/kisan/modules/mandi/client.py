"""Mandi API client for fetching market prices."""

from datetime import datetime, timedelta

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from kisan.core.config import Settings
from kisan.core.exceptions import MandiAPIError
from kisan.core.logging import logger
from kisan.modules.mandi.parser import format_prices_summary, parse_mandi_response
from kisan.schemas.mandi import MandiPriceResult


class MandiClient:
    """Client for fetching mandi prices from government API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_url = settings.mandi_api_url
        self.api_key = settings.mandi_api_key
        self._cache: dict[str, tuple[datetime, MandiPriceResult]] = {}
        self._cache_duration = timedelta(minutes=30)

    def _get_cache_key(
        self,
        commodity: str,
        state: str | None,
        district: str | None,
    ) -> str:
        """Generate cache key for a query."""
        return f"{commodity.lower()}:{state or ''}:{district or ''}"

    def _get_cached(self, cache_key: str) -> MandiPriceResult | None:
        """Get cached result if still valid."""
        if cache_key in self._cache:
            cached_time, result = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_duration:
                logger.debug(f"Cache hit for {cache_key}")
                return result
            else:
                del self._cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, result: MandiPriceResult) -> None:
        """Cache a result."""
        self._cache[cache_key] = (datetime.now(), result)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def get_prices(
        self,
        commodity: str,
        state: str | None = None,
        district: str | None = None,
        market: str | None = None,
    ) -> MandiPriceResult:
        """Fetch mandi prices for a commodity."""
        cache_key = self._get_cache_key(commodity, state, district)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            params = {
                "api-key": self.api_key,
                "format": "json",
                "limit": 100,
            }

            # Add filters
            filters = []
            if commodity:
                params["filters[commodity]"] = commodity
            if state:
                params["filters[state]"] = state
            if district:
                params["filters[district]"] = district
            if market:
                params["filters[market]"] = market

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.api_url, params=params)

                if response.status_code == 200:
                    data = response.json()
                    prices = parse_mandi_response(data)

                    result = MandiPriceResult(
                        query_commodity=commodity,
                        query_location=state or district or market,
                        prices=prices,
                        total_results=len(prices),
                        message=None if prices else "No prices found for the specified criteria",
                    )

                    self._set_cache(cache_key, result)
                    return result

                elif response.status_code == 401:
                    logger.error("Mandi API authentication failed")
                    raise MandiAPIError("API authentication failed. Check API key.")

                else:
                    logger.error(f"Mandi API error: {response.status_code}")
                    raise MandiAPIError(f"API returned status {response.status_code}")

        except httpx.RequestError as e:
            logger.error(f"Mandi API request failed: {e}")
            raise MandiAPIError(f"Failed to connect to mandi API: {e}") from e

    async def get_prices_formatted(
        self,
        commodity: str,
        state: str | None = None,
        district: str | None = None,
    ) -> str:
        """Get formatted price summary for a commodity."""
        result = await self.get_prices(commodity, state, district)
        return format_prices_summary(result.prices, commodity)

    async def get_available_commodities(self) -> list[str]:
        """Get list of commonly traded commodities."""
        # Static list of common commodities
        return [
            "Wheat", "Rice", "Maize", "Bajra", "Jowar",
            "Cotton", "Sugarcane", "Groundnut", "Soybean",
            "Mustard", "Onion", "Potato", "Tomato",
            "Chilli", "Turmeric", "Garlic", "Ginger",
            "Apple", "Banana", "Mango", "Orange",
        ]

    def clear_cache(self) -> None:
        """Clear the price cache."""
        self._cache.clear()
        logger.debug("Mandi price cache cleared")
