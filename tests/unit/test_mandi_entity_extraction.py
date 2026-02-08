"""Phase 1: Mandi entity extraction and tool execution tests.

Tests that the mandi price tool correctly handles various argument patterns
and edge cases. These are deterministic — mock the MandiClient.
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kisan.agent.tools import ToolExecutor
from kisan.core.config import Settings
from kisan.schemas.mandi import MandiPrice, MandiPriceResult


@pytest.fixture
def mock_settings():
    return Settings(openai_api_key="test-key", qdrant_url="http://localhost:6333")


@pytest.fixture
def tool_executor(mock_settings):
    with patch("kisan.agent.tools.DiseaseDetector"), \
         patch("kisan.agent.tools.MandiClient"), \
         patch("kisan.agent.tools.SchemeRetriever"):
        executor = ToolExecutor(MagicMock(), MagicMock(), mock_settings)
        # Store reference to the mock mandi_client
        return executor


def _make_price_result(commodity="Wheat", district="New Delhi", state="Delhi", n_prices=2):
    """Helper to create a MandiPriceResult with sample data."""
    prices = [
        MandiPrice(
            commodity=commodity,
            variety="FAQ",
            state=state,
            district=district,
            market=f"Market {i + 1}",
            min_price=2000.0 + i * 100,
            max_price=2500.0 + i * 100,
            modal_price=2250.0 + i * 100,
            arrival_date=date(2024, 1, 15),
        )
        for i in range(n_prices)
    ]
    return MandiPriceResult(
        query_commodity=commodity,
        prices=prices,
        total_results=n_prices,
    )


class TestMandiArgumentHandling:
    """Test that mandi tool handles various argument patterns correctly."""

    async def test_commodity_only(self, tool_executor):
        tool_executor.mandi_client.get_prices = AsyncMock(
            return_value=_make_price_result(commodity="Wheat")
        )
        result = await tool_executor._execute_mandi_prices({"commodity": "wheat"})
        tool_executor.mandi_client.get_prices.assert_called_once_with(
            commodity="wheat", state=None, district=None
        )
        assert "Wheat" in result

    async def test_commodity_and_state(self, tool_executor):
        tool_executor.mandi_client.get_prices = AsyncMock(
            return_value=_make_price_result(commodity="Rice", state="Punjab")
        )
        result = await tool_executor._execute_mandi_prices(
            {"commodity": "rice", "state": "Punjab"}
        )
        tool_executor.mandi_client.get_prices.assert_called_once_with(
            commodity="rice", state="Punjab", district=None
        )
        assert "Rice" in result and "Punjab" in result

    async def test_commodity_state_and_district(self, tool_executor):
        tool_executor.mandi_client.get_prices = AsyncMock(
            return_value=_make_price_result(commodity="Onion", state="Maharashtra", district="Nashik")
        )
        result = await tool_executor._execute_mandi_prices(
            {"commodity": "onion", "state": "Maharashtra", "district": "Nashik"}
        )
        tool_executor.mandi_client.get_prices.assert_called_once_with(
            commodity="onion", state="Maharashtra", district="Nashik"
        )

        assert "Onion" in result and "Maharashtra" in result and "Nashik" in result

    async def test_empty_commodity_returns_message(self, tool_executor):
        result = await tool_executor._execute_mandi_prices({"commodity": ""})
        # Basically we're checking whether the LLM asked the user to specify a
        # specific commodity.
        assert "specify" in result.lower()

    async def test_missing_commodity_returns_message(self, tool_executor):
        result = await tool_executor._execute_mandi_prices({})
        assert "specify" in result.lower()

    async def test_no_results_found(self, tool_executor):
        tool_executor.mandi_client.get_prices = AsyncMock(
            return_value=MandiPriceResult(
                query_commodity="Saffron", prices=[], total_results=0
            )
        )
        result = await tool_executor._execute_mandi_prices({"commodity": "saffron"})
        assert "no price data" in result.lower()


class TestMandiOutputFormatting:
    """Test that mandi results are formatted correctly for the LLM."""

    async def test_output_contains_price_range(self, tool_executor):
        tool_executor.mandi_client.get_prices = AsyncMock(
            return_value=_make_price_result(commodity="Wheat")
        )
        result = await tool_executor._execute_mandi_prices({"commodity": "wheat"})
        assert "2,000" in result or "2000" in result
        assert "quintal" in result.lower()

    async def test_output_contains_market_name(self, tool_executor):
        tool_executor.mandi_client.get_prices = AsyncMock(
            return_value=_make_price_result(commodity="Wheat")
        )
        result = await tool_executor._execute_mandi_prices({"commodity": "wheat"})
        assert "Market 1" in result

    async def test_output_truncated_at_10_results(self, tool_executor):
        tool_executor.mandi_client.get_prices = AsyncMock(
            return_value=_make_price_result(commodity="Wheat", n_prices=15)
        )
        result = await tool_executor._execute_mandi_prices({"commodity": "wheat"})
        # Should show "Showing 10 of 15"
        assert "15" in result
        # Count 📍 (pin emoji markers) — one per displayed market entry
        assert result.count("\U0001f4cd") == 10

    async def test_output_includes_commodity_title(self, tool_executor):
        tool_executor.mandi_client.get_prices = AsyncMock(
            return_value=_make_price_result(commodity="Tomato")
        )
        result = await tool_executor._execute_mandi_prices({"commodity": "tomato"})
        assert "Tomato" in result
