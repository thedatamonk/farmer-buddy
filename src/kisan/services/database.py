"""PostgreSQL database connection manager using asyncpg."""

from typing import Any

import asyncpg

from kisan.core.config import Settings
from kisan.core.exceptions import DatabaseError
from kisan.core.logging import logger

# SQL schema for mandi prices
SCHEMA_SQL = """
-- Price records table
CREATE TABLE IF NOT EXISTS mandi_prices (
    id SERIAL PRIMARY KEY,
    commodity TEXT NOT NULL,
    variety TEXT,
    state TEXT NOT NULL,
    district TEXT NOT NULL,
    market TEXT NOT NULL,
    min_price DECIMAL(12,2) NOT NULL,
    max_price DECIMAL(12,2) NOT NULL,
    modal_price DECIMAL(12,2) NOT NULL,
    arrival_date DATE,
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(commodity, state, district, market, arrival_date)
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_mandi_commodity ON mandi_prices(commodity);
CREATE INDEX IF NOT EXISTS idx_mandi_commodity_state ON mandi_prices(commodity, state);
CREATE INDEX IF NOT EXISTS idx_mandi_fetched_at ON mandi_prices(fetched_at);

-- Fetch tracking table
CREATE TABLE IF NOT EXISTS fetch_metadata (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    last_successful_fetch TIMESTAMP WITH TIME ZONE,
    last_fetch_attempt TIMESTAMP WITH TIME ZONE,
    last_error TEXT,
    records_fetched INTEGER DEFAULT 0
);

-- Insert default row if not exists
INSERT INTO fetch_metadata (id) VALUES (1) ON CONFLICT (id) DO NOTHING;
"""


class DatabaseService:
    """PostgreSQL connection manager with connection pooling."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        """Get the connection pool, raising if not connected."""
        if self._pool is None:
            raise DatabaseError("Database not connected. Call connect() first.")
        return self._pool

    async def connect(self) -> None:
        """Initialize connection pool and create schema."""
        if not self.settings.database_url:
            logger.warning("DATABASE_URL not configured, database features disabled")
            return

        try:
            self._pool = await asyncpg.create_pool(
                self.settings.database_url,
                min_size=1,
                max_size=10,
                command_timeout=30,
            )
            logger.info("Database connection pool created")

            # Initialize schema
            await self._init_schema()
            logger.info("Database schema initialized")

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise DatabaseError(f"Database connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed")

    async def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a query and return status."""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """Fetch multiple rows."""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        """Fetch a single row."""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Fetch a single value."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._pool is not None
