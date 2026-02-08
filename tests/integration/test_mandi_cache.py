"""Integration tests for mandi price caching system."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kisan.core.config import Settings
from kisan.core.exceptions import MandiAPIError
from kisan.modules.mandi.client import MandiClient
from kisan.modules.mandi.repository import MandiPriceRepository
from kisan.schemas.mandi import MandiPrice
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
    return db


@pytest.fixture
def mock_repository(mock_db_service, mock_settings):
    """Create a mock repository."""
    repo = MagicMock(spec=MandiPriceRepository)
    repo.db = mock_db_service
    repo.settings = mock_settings
    repo._cache_ttl = timedelta(hours=mock_settings.mandi_cache_ttl_hours)
    return repo


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
        ),
    ]


class TestMandiCacheFlow:
    """Test the cache hit/miss flow."""

    async def test_cache_hit_from_database(self, mock_settings, mock_repository, sample_prices):
        """Test cache hit when data is in database and fresh."""
        last_fetched = datetime.now(timezone.utc)
        mock_repository.get_prices = AsyncMock(return_value=(sample_prices, last_fetched))
        mock_repository.is_data_fresh = MagicMock(return_value=True)

        client = MandiClient(mock_settings, repository=mock_repository)
        result = await client.get_prices("Wheat")

        assert result.cache_hit is True
        assert result.total_results == 1
        assert result.prices[0].commodity == "Wheat"
        mock_repository.get_prices.assert_called_once()

    @patch("httpx.AsyncClient")
    async def test_cache_miss_falls_back_to_api(
        self, mock_client_class, mock_settings, mock_repository
    ):
        """Test that cache miss triggers API call."""
        mock_repository.get_prices = AsyncMock(return_value=([], None))
        mock_repository.is_data_fresh = MagicMock(return_value=False)

        # Mock API response
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
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        client = MandiClient(mock_settings, repository=mock_repository)
        result = await client.get_prices("Wheat")

        assert result.cache_hit is False
        assert result.total_results == 1

    @patch("httpx.AsyncClient")
    async def test_stale_data_returned_on_api_failure(
        self, mock_client_class, mock_settings, mock_repository, sample_prices
    ):
        """Test that stale data is returned when API fails."""
        last_fetched = datetime.now(timezone.utc) - timedelta(hours=25)  # Stale
        mock_repository.get_prices = AsyncMock(return_value=(sample_prices, last_fetched))
        mock_repository.is_data_fresh = MagicMock(return_value=False)

        # Mock API failure with httpx.RequestError
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.RequestError("Connection error")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        client = MandiClient(mock_settings, repository=mock_repository)
        result = await client.get_prices("Wheat")

        assert result.cache_hit is True
        assert "outdated" in result.message.lower()
        assert result.total_results == 1

    @patch("httpx.AsyncClient")
    async def test_api_error_with_no_stale_data_raises(
        self, mock_client_class, mock_settings, mock_repository
    ):
        """Test that API error with no stale data raises exception."""
        mock_repository.get_prices = AsyncMock(return_value=([], None))
        mock_repository.is_data_fresh = MagicMock(return_value=False)

        # Mock API failure
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        client = MandiClient(mock_settings, repository=mock_repository)

        with pytest.raises(MandiAPIError):
            await client.get_prices("Wheat")

    async def test_in_memory_cache_hit(self, mock_settings, mock_repository, sample_prices):
        """Test in-memory cache hit on second request."""
        last_fetched = datetime.now(timezone.utc)
        mock_repository.get_prices = AsyncMock(return_value=(sample_prices, last_fetched))
        mock_repository.is_data_fresh = MagicMock(return_value=True)

        client = MandiClient(mock_settings, repository=mock_repository)

        # First request - hits DB
        result1 = await client.get_prices("Wheat")
        assert mock_repository.get_prices.call_count == 1

        # Second request - hits in-memory cache
        result2 = await client.get_prices("Wheat")
        assert mock_repository.get_prices.call_count == 1  # No additional DB call

        assert result1.total_results == result2.total_results

    async def test_no_repository_falls_back_to_api(self, mock_settings):
        """Test that client without repository uses API directly."""
        client = MandiClient(mock_settings, repository=None)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"records": []}

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await client.get_prices("Wheat")

            assert result.cache_hit is False
            mock_client.get.assert_called_once()

    async def test_database_error_handled_gracefully(
        self, mock_settings, mock_repository
    ):
        """Test that database errors are handled and API is used as fallback."""
        mock_repository.get_prices = AsyncMock(side_effect=Exception("DB connection lost"))

        with patch("httpx.AsyncClient") as mock_client_class:
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
                    }
                ]
            }

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            client = MandiClient(mock_settings, repository=mock_repository)
            result = await client.get_prices("Wheat")

            # Should fall back to API successfully
            assert result.cache_hit is False
            assert result.total_results == 1

    async def test_result_includes_last_updated(
        self, mock_settings, mock_repository, sample_prices
    ):
        """Test that cache hit result includes last_updated timestamp."""
        last_fetched = datetime.now(timezone.utc)
        mock_repository.get_prices = AsyncMock(return_value=(sample_prices, last_fetched))
        mock_repository.is_data_fresh = MagicMock(return_value=True)

        client = MandiClient(mock_settings, repository=mock_repository)
        result = await client.get_prices("Wheat")

        assert result.last_updated == last_fetched
