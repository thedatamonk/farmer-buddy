"""Unit tests for MandiPriceRepository."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from kisan.core.config import Settings
from kisan.modules.mandi.repository import MandiPriceRepository
from kisan.schemas.mandi import MandiPrice
from kisan.services.database import DatabaseService


@pytest.fixture
def mock_settings():
    """Create test settings."""
    return Settings(
        openai_api_key="test-key",
        database_url="postgresql://test:test@localhost:5432/test",
        mandi_cache_ttl_hours=24,
    )


@pytest.fixture
def mock_db_service(mock_settings):
    """Create a mock database service."""
    db = MagicMock(spec=DatabaseService)
    db.settings = mock_settings
    db.is_connected.return_value = True
    db.pool = MagicMock()
    return db


@pytest.fixture
def repository(mock_db_service, mock_settings):
    """Create repository with mock database."""
    return MandiPriceRepository(mock_db_service, mock_settings)


@pytest.fixture
def sample_prices():
    """Create sample MandiPrice objects."""
    return [
        MandiPrice(
            commodity="Wheat",
            variety="HD-2967",
            state="Punjab",
            district="Ludhiana",
            market="Ludhiana Mandi",
            min_price=2100.0,
            max_price=2300.0,
            modal_price=2200.0,
            arrival_date=date(2024, 1, 15),
        ),
        MandiPrice(
            commodity="Wheat",
            variety="PBW-343",
            state="Haryana",
            district="Karnal",
            market="Karnal Grain Market",
            min_price=2050.0,
            max_price=2250.0,
            modal_price=2150.0,
            arrival_date=date(2024, 1, 15),
        ),
    ]


class TestMandiPriceRepository:
    """Tests for MandiPriceRepository."""

    async def test_get_prices_returns_empty_when_not_connected(self, mock_settings):
        """Test that get_prices returns empty list when DB not connected."""
        db = MagicMock(spec=DatabaseService)
        db.is_connected.return_value = False
        repo = MandiPriceRepository(db, mock_settings)

        prices, last_fetched = await repo.get_prices("Wheat")

        assert prices == []
        assert last_fetched is None

    async def test_get_prices_with_commodity_filter(self, repository, mock_db_service):
        """Test fetching prices with commodity filter."""
        mock_rows = [
            {
                "commodity": "Wheat",
                "variety": "HD-2967",
                "state": "Punjab",
                "district": "Ludhiana",
                "market": "Ludhiana Mandi",
                "min_price": Decimal("2100.00"),
                "max_price": Decimal("2300.00"),
                "modal_price": Decimal("2200.00"),
                "arrival_date": date(2024, 1, 15),
                "fetched_at": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            }
        ]
        mock_db_service.fetch = AsyncMock(return_value=mock_rows)

        prices, last_fetched = await repository.get_prices("Wheat")

        assert len(prices) == 1
        assert prices[0].commodity == "Wheat"
        assert prices[0].modal_price == 2200.0
        assert last_fetched is not None

    async def test_get_prices_with_state_filter(self, repository, mock_db_service):
        """Test fetching prices with state filter."""
        mock_db_service.fetch = AsyncMock(return_value=[])

        await repository.get_prices("Wheat", state="Punjab")

        # Verify the query includes state filter
        call_args = mock_db_service.fetch.call_args
        assert call_args is not None
        query = call_args[0][0]
        assert "state ILIKE" in query

    async def test_upsert_prices_empty_list(self, repository):
        """Test upserting empty list returns 0."""
        count = await repository.upsert_prices([])
        assert count == 0

    async def test_upsert_prices_not_connected(self, mock_settings, sample_prices):
        """Test upserting when DB not connected returns 0."""
        db = MagicMock(spec=DatabaseService)
        db.is_connected.return_value = False
        repo = MandiPriceRepository(db, mock_settings)

        count = await repo.upsert_prices(sample_prices)
        assert count == 0

    async def test_upsert_prices_success(self, repository, mock_db_service, sample_prices):
        """Test successful price upsert."""
        mock_conn = AsyncMock()
        mock_db_service.pool.acquire.return_value.__aenter__.return_value = mock_conn

        count = await repository.upsert_prices(sample_prices)

        assert count == 2
        assert mock_conn.execute.call_count == 2

    async def test_get_last_fetch_time(self, repository, mock_db_service):
        """Test getting last fetch time."""
        expected_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        mock_db_service.fetchval = AsyncMock(return_value=expected_time)

        result = await repository.get_last_fetch_time()

        assert result == expected_time

    async def test_get_last_fetch_time_not_connected(self, mock_settings):
        """Test get_last_fetch_time returns None when not connected."""
        db = MagicMock(spec=DatabaseService)
        db.is_connected.return_value = False
        repo = MandiPriceRepository(db, mock_settings)

        result = await repo.get_last_fetch_time()

        assert result is None

    async def test_update_fetch_metadata_success(self, repository, mock_db_service):
        """Test updating fetch metadata on success."""
        mock_db_service.execute = AsyncMock()

        await repository.update_fetch_metadata(success=True, records_fetched=100)

        mock_db_service.execute.assert_called_once()
        call_args = mock_db_service.execute.call_args
        query = call_args[0][0]
        assert "last_successful_fetch" in query

    async def test_update_fetch_metadata_failure(self, repository, mock_db_service):
        """Test updating fetch metadata on failure."""
        mock_db_service.execute = AsyncMock()

        await repository.update_fetch_metadata(success=False, error="API timeout")

        mock_db_service.execute.assert_called_once()
        call_args = mock_db_service.execute.call_args
        query = call_args[0][0]
        assert "last_error" in query

    async def test_cleanup_old_data(self, repository, mock_db_service):
        """Test cleaning up old data."""
        mock_db_service.execute = AsyncMock(return_value="DELETE 50")

        deleted = await repository.cleanup_old_data(days=7)

        assert deleted == 50
        mock_db_service.execute.assert_called_once()

    async def test_cleanup_old_data_not_connected(self, mock_settings):
        """Test cleanup returns 0 when not connected."""
        db = MagicMock(spec=DatabaseService)
        db.is_connected.return_value = False
        repo = MandiPriceRepository(db, mock_settings)

        deleted = await repo.cleanup_old_data()

        assert deleted == 0

    def test_is_data_fresh_with_fresh_data(self, repository):
        """Test is_data_fresh returns True for fresh data."""
        now = datetime.now(timezone.utc)
        assert repository.is_data_fresh(now) is True

    def test_is_data_fresh_with_stale_data(self, repository):
        """Test is_data_fresh returns False for stale data."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        assert repository.is_data_fresh(old_time) is False

    def test_is_data_fresh_with_none(self, repository):
        """Test is_data_fresh returns False for None."""
        assert repository.is_data_fresh(None) is False

    def test_is_data_fresh_handles_naive_datetime(self, repository):
        """Test is_data_fresh handles naive datetime."""
        now = datetime.now()  # Naive datetime
        assert repository.is_data_fresh(now) is True
