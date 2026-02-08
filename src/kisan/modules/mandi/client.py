"""Mandi API client for fetching market prices."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from kisan.core.config import Settings
from kisan.core.exceptions import MandiAPIError
from kisan.core.logging import logger
from kisan.modules.mandi.parser import format_prices_summary, parse_mandi_response
from kisan.schemas.mandi import MandiPriceResult

if TYPE_CHECKING:
    from kisan.modules.mandi.repository import MandiPriceRepository


class MandiClient:
    """Client for fetching mandi prices from government API."""

    def __init__(
        self,
        settings: Settings,
        repository: MandiPriceRepository | None = None,
    ):
        self.settings = settings
        self.api_url = settings.mandi_api_url
        self.api_key = settings.mandi_api_key
        self.repository = repository
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
                logger.debug(f"In-memory cache hit for {cache_key}")
                return result
            else:
                del self._cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, result: MandiPriceResult) -> None:
        """Cache a result."""
        self._cache[cache_key] = (datetime.now(), result)

    async def get_prices(
        self,
        commodity: str,
        state: str | None = None,
        district: str | None = None,
        market: str | None = None,
    ) -> MandiPriceResult:
        """
        Fetch mandi prices for a commodity.

        Query flow:
        1. Check in-memory cache
        2. Check database cache (if repository available)
        3. Fall back to live API
        4. If API fails, return stale DB data if available
        """
        cache_key = self._get_cache_key(commodity, state, district)

        # 1. Check in-memory cache first (fastest)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Variables to track DB results for potential stale fallback
        db_prices = []
        db_last_fetched = None

        # 2. Try database cache if repository available
        if self.repository:
            try:
                db_prices, db_last_fetched = await self.repository.get_prices(
                    commodity=commodity,
                    state=state,
                    district=district,
                    market=market,
                )

                if db_prices and self.repository.is_data_fresh(db_last_fetched):
                    result = MandiPriceResult(
                        query_commodity=commodity,
                        query_location=state or district or market,
                        prices=db_prices,
                        total_results=len(db_prices),
                        message=None,
                        cache_hit=True,
                        last_updated=db_last_fetched,
                    )
                    self._set_cache(cache_key, result)
                    logger.info(f"Database cache hit for {commodity}")
                    return result

            except Exception as e:
                logger.warning(f"Database lookup failed, falling back to API: {e}")

        # 3. Fall back to live API
        try:
            result = await self._fetch_from_api(
                commodity=commodity,
                state=state,
                district=district,
                market=market,
            )
            self._set_cache(cache_key, result)
            return result

        except MandiAPIError:
            # 4. If API fails and we have stale DB data, return it
            if db_prices:
                logger.warning(f"API failed, returning stale data for {commodity}")
                return MandiPriceResult(
                    query_commodity=commodity,
                    query_location=state or district or market,
                    prices=db_prices,
                    total_results=len(db_prices),
                    message="Note: This data may be outdated (live API unavailable)",
                    cache_hit=True,
                    last_updated=db_last_fetched,
                )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _fetch_from_api(
        self,
        commodity: str,
        state: str | None = None,
        district: str | None = None,
        market: str | None = None,
    ) -> MandiPriceResult:
        """Fetch prices directly from the government API."""
        try:
            params = {
                "api-key": self.api_key,
                "format": "json",
                "limit": 100,
            }

            # Add filters
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

                    return MandiPriceResult(
                        query_commodity=commodity,
                        query_location=state or district or market,
                        prices=prices,
                        total_results=len(prices),
                        message=None if prices else "No prices found for the specified criteria",
                        cache_hit=False,
                        last_updated=datetime.now(),
                    )

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
