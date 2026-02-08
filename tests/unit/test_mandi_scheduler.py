"""Unit tests for MandiPriceFetchScheduler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kisan.core.config import Settings
from kisan.modules.mandi.repository import MandiPriceRepository
from kisan.modules.mandi.scheduler import MandiPriceFetchScheduler
from kisan.services.database import DatabaseService


@pytest.fixture
def mock_settings():
    """Create test settings."""
    return Settings(
        openai_api_key="test-key",
        mandi_api_key="test-mandi-key",
        mandi_api_url="https://api.data.gov.in/resource/test",
        database_url="postgresql://test:test@localhost:5432/test",
        mandi_cache_ttl_hours=24,
        mandi_fetch_interval_hours=6,
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
def mock_repository(mock_db_service, mock_settings):
    """Create a mock repository."""
    repo = MagicMock(spec=MandiPriceRepository)
    repo.db = mock_db_service
    repo.settings = mock_settings
    repo.upsert_prices = AsyncMock(return_value=10)
    repo.update_fetch_metadata = AsyncMock()
    repo.cleanup_old_data = AsyncMock(return_value=5)
    return repo


@pytest.fixture
def scheduler(mock_repository, mock_settings):
    """Create scheduler with mocks."""
    return MandiPriceFetchScheduler(mock_repository, mock_settings)


class TestMandiPriceFetchScheduler:
    """Tests for MandiPriceFetchScheduler."""

    async def test_scheduler_init(self, scheduler, mock_repository, mock_settings):
        """Test scheduler initialization."""
        assert scheduler.repository == mock_repository
        assert scheduler.settings == mock_settings
        assert scheduler._running is False

    async def test_start_when_db_not_connected(self, mock_settings):
        """Test scheduler doesn't start when DB not connected."""
        db = MagicMock(spec=DatabaseService)
        db.is_connected.return_value = False
        repo = MagicMock(spec=MandiPriceRepository)
        repo.db = db

        sched = MandiPriceFetchScheduler(repo, mock_settings)
        await sched.start()

        assert sched._running is False

    async def test_start_and_stop(self, scheduler):
        """Test scheduler start and stop lifecycle."""
        await scheduler.start()
        assert scheduler._running is True

        await scheduler.stop()
        assert scheduler._running is False

    async def test_start_twice_does_not_duplicate(self, scheduler):
        """Test starting scheduler twice doesn't create duplicate jobs."""
        await scheduler.start()
        await scheduler.start()  # Should log warning but not fail

        assert scheduler._running is True
        await scheduler.stop()

    @patch("httpx.AsyncClient")
    async def test_fetch_commodity_prices_success(self, mock_client_class, scheduler):
        """Test successful commodity price fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "records": [
                {
                    "commodity": "Wheat",
                    "state": "Punjab",
                    "district": "Ludhiana",
                    "market": "Ludhiana",
                    "min_price": "2100",
                    "max_price": "2300",
                    "modal_price": "2200",
                    "arrival_date": "15/01/2024",
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        count = await scheduler._fetch_commodity_prices("Wheat")

        assert count == 10  # From mock upsert
        scheduler.repository.upsert_prices.assert_called_once()

    @patch("httpx.AsyncClient")
    async def test_fetch_commodity_prices_api_error(self, mock_client_class, scheduler):
        """Test handling API errors during fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        count = await scheduler._fetch_commodity_prices("Wheat")

        assert count == 0

    @patch("httpx.AsyncClient")
    async def test_fetch_commodity_prices_auth_failure(self, mock_client_class, scheduler):
        """Test handling auth failure during fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        with pytest.raises(Exception, match="authentication failed"):
            await scheduler._fetch_commodity_prices("Wheat")

    async def test_cleanup_old_data(self, scheduler):
        """Test cleanup job."""
        await scheduler._cleanup_old_data()

        scheduler.repository.cleanup_old_data.assert_called_once_with(days=7)

    @patch.object(MandiPriceFetchScheduler, "_fetch_commodity_prices")
    async def test_fetch_all_prices_success(self, mock_fetch, scheduler):
        """Test fetching all commodity prices."""
        mock_fetch.return_value = 10

        await scheduler._fetch_all_prices()

        # Should be called for each commodity
        assert mock_fetch.call_count == len(scheduler.COMMODITIES)
        scheduler.repository.update_fetch_metadata.assert_called_once()

        # Should indicate success
        call_args = scheduler.repository.update_fetch_metadata.call_args
        assert call_args[1]["success"] is True

    @patch.object(MandiPriceFetchScheduler, "_fetch_commodity_prices")
    async def test_fetch_all_prices_with_errors(self, mock_fetch, scheduler):
        """Test fetch_all_prices handles errors gracefully."""
        mock_fetch.side_effect = [
            10,  # First succeeds
            Exception("API error"),  # Second fails
            10,  # Third succeeds
        ] + [10] * (len(scheduler.COMMODITIES) - 3)

        await scheduler._fetch_all_prices()

        # Should still call update_fetch_metadata
        scheduler.repository.update_fetch_metadata.assert_called_once()
        call_args = scheduler.repository.update_fetch_metadata.call_args
        assert call_args[1]["success"] is False

    async def test_trigger_fetch(self, scheduler):
        """Test manual fetch trigger."""
        with patch.object(scheduler, "_fetch_all_prices", new_callable=AsyncMock) as mock_fetch:
            await scheduler.trigger_fetch()
            mock_fetch.assert_called_once()

    def test_commodities_list(self, scheduler):
        """Test that commodities list is populated."""
        assert len(scheduler.COMMODITIES) > 0
        assert "Wheat" in scheduler.COMMODITIES
        assert "Rice" in scheduler.COMMODITIES
        assert "Onion" in scheduler.COMMODITIES
