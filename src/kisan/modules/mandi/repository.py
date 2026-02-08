"""Data access layer for mandi prices."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from kisan.core.config import Settings
from kisan.core.logging import logger
from kisan.schemas.mandi import MandiPrice
from kisan.services.database import DatabaseService


class MandiPriceRepository:
    """Repository for mandi price data access."""

    def __init__(self, db: DatabaseService, settings: Settings):
        self.db = db
        self.settings = settings
        self._cache_ttl = timedelta(hours=settings.mandi_cache_ttl_hours)

    async def get_prices(
        self,
        commodity: str,
        state: str | None = None,
        district: str | None = None,
        market: str | None = None,
        limit: int = 100,
    ) -> tuple[list[MandiPrice], datetime | None]:
        """
        Query prices with filters.
        Returns tuple of (prices, last_fetched_time).
        """
        if not self.db.is_connected():
            return [], None

        # Build query with filters
        conditions = ["commodity ILIKE $1"]
        params: list = [f"%{commodity}%"]
        param_idx = 2

        if state:
            conditions.append(f"state ILIKE ${param_idx}")
            params.append(f"%{state}%")
            param_idx += 1

        if district:
            conditions.append(f"district ILIKE ${param_idx}")
            params.append(f"%{district}%")
            param_idx += 1

        if market:
            conditions.append(f"market ILIKE ${param_idx}")
            params.append(f"%{market}%")
            param_idx += 1

        where_clause = " AND ".join(conditions)
        params.append(limit)

        query = f"""
            SELECT commodity, variety, state, district, market,
                   min_price, max_price, modal_price, arrival_date, fetched_at
            FROM mandi_prices
            WHERE {where_clause}
            ORDER BY fetched_at DESC, arrival_date DESC NULLS LAST
            LIMIT ${param_idx}
        """

        try:
            rows = await self.db.fetch(query, *params)

            prices = []
            last_fetched = None

            for row in rows:
                prices.append(MandiPrice(
                    commodity=row["commodity"],
                    variety=row["variety"],
                    state=row["state"],
                    district=row["district"],
                    market=row["market"],
                    min_price=float(row["min_price"]),
                    max_price=float(row["max_price"]),
                    modal_price=float(row["modal_price"]),
                    arrival_date=row["arrival_date"],
                ))
                if last_fetched is None and row["fetched_at"]:
                    last_fetched = row["fetched_at"]

            logger.debug(f"Fetched {len(prices)} prices from database for {commodity}")
            return prices, last_fetched

        except Exception as e:
            logger.error(f"Error fetching prices from database: {e}")
            return [], None

    async def upsert_prices(self, prices: list[MandiPrice]) -> int:
        """
        Insert or update price records.
        Returns number of records upserted.
        """
        if not self.db.is_connected() or not prices:
            return 0

        query = """
            INSERT INTO mandi_prices
                (commodity, variety, state, district, market,
                 min_price, max_price, modal_price, arrival_date, fetched_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            ON CONFLICT (commodity, state, district, market, arrival_date)
            DO UPDATE SET
                variety = EXCLUDED.variety,
                min_price = EXCLUDED.min_price,
                max_price = EXCLUDED.max_price,
                modal_price = EXCLUDED.modal_price,
                fetched_at = NOW()
        """

        try:
            count = 0
            async with self.db.pool.acquire() as conn:
                for price in prices:
                    await conn.execute(
                        query,
                        price.commodity,
                        price.variety,
                        price.state,
                        price.district,
                        price.market,
                        Decimal(str(price.min_price)),
                        Decimal(str(price.max_price)),
                        Decimal(str(price.modal_price)),
                        price.arrival_date,
                    )
                    count += 1

            logger.info(f"Upserted {count} price records")
            return count

        except Exception as e:
            logger.error(f"Error upserting prices: {e}")
            return 0

    async def get_last_fetch_time(self) -> datetime | None:
        """Get the last successful fetch timestamp."""
        if not self.db.is_connected():
            return None

        try:
            result = await self.db.fetchval(
                "SELECT last_successful_fetch FROM fetch_metadata WHERE id = 1"
            )
            return result
        except Exception as e:
            logger.error(f"Error getting last fetch time: {e}")
            return None

    async def update_fetch_metadata(
        self,
        success: bool,
        records_fetched: int = 0,
        error: str | None = None,
    ) -> None:
        """Update fetch tracking metadata."""
        if not self.db.is_connected():
            return

        try:
            now = datetime.now(timezone.utc)
            if success:
                await self.db.execute(
                    """
                    UPDATE fetch_metadata SET
                        last_successful_fetch = $1,
                        last_fetch_attempt = $1,
                        last_error = NULL,
                        records_fetched = $2
                    WHERE id = 1
                    """,
                    now,
                    records_fetched,
                )
            else:
                await self.db.execute(
                    """
                    UPDATE fetch_metadata SET
                        last_fetch_attempt = $1,
                        last_error = $2
                    WHERE id = 1
                    """,
                    now,
                    error,
                )
        except Exception as e:
            logger.error(f"Error updating fetch metadata: {e}")

    async def cleanup_old_data(self, days: int = 7) -> int:
        """
        Remove records older than specified days.
        Returns number of records deleted.
        """
        if not self.db.is_connected():
            return 0

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            result = await self.db.execute(
                "DELETE FROM mandi_prices WHERE fetched_at < $1",
                cutoff,
            )
            # Parse "DELETE N" to get count
            deleted = int(result.split()[-1]) if result else 0
            logger.info(f"Cleaned up {deleted} old price records")
            return deleted
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0

    def is_data_fresh(self, last_fetched: datetime | None) -> bool:
        """Check if data is within cache TTL."""
        if last_fetched is None:
            return False
        # Make sure we compare timezone-aware datetimes
        now = datetime.now(timezone.utc)
        if last_fetched.tzinfo is None:
            last_fetched = last_fetched.replace(tzinfo=timezone.utc)
        return (now - last_fetched) < self._cache_ttl
