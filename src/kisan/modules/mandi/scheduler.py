"""Background job scheduler for mandi price fetching."""

import asyncio

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from kisan.core.config import Settings
from kisan.core.logging import logger
from kisan.modules.mandi.parser import parse_mandi_response
from kisan.modules.mandi.repository import MandiPriceRepository


class MandiPriceFetchScheduler:
    """Background scheduler for fetching and caching mandi prices."""

    # Common commodities to fetch prices for
    # TODO: Bajra, Jowar, and Garlic are getting errors from API, need to investigate
    COMMODITIES = [
        "Wheat", "Rice", "Maize", "Bajra", "Jowar",
        "Cotton", "Sugarcane", "Groundnut", "Soybean",
        "Mustard", "Onion", "Potato", "Tomato",
        "Chilli", "Turmeric", "Garlic", "Ginger",
        "Apple", "Banana", "Mango", "Orange",
    ]

    def __init__(
        self,
        repository: MandiPriceRepository,
        settings: Settings,
    ):
        self.repository = repository
        self.settings = settings
        self.scheduler = AsyncIOScheduler()
        self._running = False

    async def start(self) -> None:
        """Start the background scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        if not self.repository.db.is_connected():
            logger.warning("Database not connected, scheduler not starting")
            return

        # Schedule periodic fetch (every N hours)
        interval_hours = self.settings.mandi_fetch_interval_hours
        self.scheduler.add_job(
            self._fetch_all_prices,
            trigger=IntervalTrigger(hours=interval_hours),
            id="mandi_price_fetch",
            name="Fetch mandi prices",
            replace_existing=True,
        )

        # Schedule daily cleanup at 3 AM
        self.scheduler.add_job(
            self._cleanup_old_data,
            trigger=CronTrigger(hour=3, minute=0),
            id="mandi_cleanup",
            name="Cleanup old mandi data",
            replace_existing=True,
        )

        self.scheduler.start()
        self._running = True
        logger.info(f"Mandi price scheduler started (fetch every {interval_hours}h)")

        # Initial fetch after 10 seconds delay
        asyncio.create_task(self._delayed_initial_fetch())

    async def stop(self) -> None:
        """Stop the background scheduler."""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Mandi price scheduler stopped")

    async def _delayed_initial_fetch(self) -> None:
        """Perform initial fetch after a short delay."""
        await asyncio.sleep(10)
        logger.info("Starting initial mandi price fetch")
        await self._fetch_all_prices()

    async def _fetch_all_prices(self) -> None:
        """Fetch prices for all commodities."""
        logger.info("Starting scheduled mandi price fetch")
        total_records = 0
        errors = []

        for commodity in self.COMMODITIES:
            try:
                records = await self._fetch_commodity_prices(commodity)
                total_records += records
                # Small delay between API calls to avoid rate limiting
                await asyncio.sleep(1)
            except Exception as e:
                errors.append(f"{commodity}: {e}")
                logger.error(f"Failed to fetch prices for {commodity}: {e}")

        if errors:
            await self.repository.update_fetch_metadata(
                success=False,
                records_fetched=total_records,
                error="; ".join(errors[:5]),  # Limit error message size
            )
        else:
            await self.repository.update_fetch_metadata(
                success=True,
                records_fetched=total_records,
            )

        logger.info(f"Completed mandi price fetch: {total_records} records, {len(errors)} errors")

    async def _fetch_commodity_prices(self, commodity: str) -> int:
        """Fetch and store prices for a single commodity."""
        # TODO: How come I am able to fetch prices even though there is no API key?
        params = {
            "api-key": self.settings.mandi_api_key,
            "format": "json",
            "limit": 500,  # Fetch more records for caching
            "filters[commodity]": commodity,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(
                    self.settings.mandi_api_url,
                    params=params,
                )

                if response.status_code == 200:
                    data = response.json()
                    prices = parse_mandi_response(data)

                    if prices:
                        count = await self.repository.upsert_prices(prices)
                        logger.debug(f"Fetched {count} prices for {commodity}")
                        return count
                    return 0

                elif response.status_code == 401:
                    logger.error("Mandi API authentication failed")
                    raise Exception("API authentication failed")

                else:
                    logger.warning(f"API returned {response.status_code} for {commodity}")
                    return 0

        except httpx.RequestError as e:
            logger.error(f"Request failed for {commodity}: {e}")
            raise

    async def _cleanup_old_data(self) -> None:
        """Remove old price records."""
        logger.info("Starting scheduled cleanup of old mandi data")
        deleted = await self.repository.cleanup_old_data(days=7)
        logger.info(f"Cleanup complete: {deleted} old records removed")

    async def trigger_fetch(self) -> None:
        """Manually trigger a price fetch (for testing/admin)."""
        logger.info("Manual mandi price fetch triggered")
        await self._fetch_all_prices()
