"""Phase 1: Tool selection accuracy tests.

Tests that the orchestrator's tool dispatch routes to the correct handler
based on tool_name, and that TOOL_DEFINITIONS are well-formed.
These are deterministic tests — no LLM calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kisan.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from kisan.core.config import Settings
from kisan.core.exceptions import ToolExecutionError


@pytest.fixture
def mock_settings():
    return Settings(openai_api_key="test-key", qdrant_url="http://localhost:6333")


@pytest.fixture
def mock_llm_service():
    return MagicMock()


@pytest.fixture
def mock_vectordb_service():
    return MagicMock()


@pytest.fixture
def tool_executor(mock_llm_service, mock_vectordb_service, mock_settings):
    with patch("kisan.agent.tools.DiseaseDetector"), \
         patch("kisan.agent.tools.MandiClient"), \
         patch("kisan.agent.tools.SchemeRetriever"):
        return ToolExecutor(mock_llm_service, mock_vectordb_service, mock_settings)


class TestToolDefinitions:
    """Verify TOOL_DEFINITIONS structure is valid for OpenAI function calling."""

    def test_all_tools_have_required_fields(self):
        for tool_def in TOOL_DEFINITIONS:
            assert tool_def["type"] == "function"
            func = tool_def["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"
            assert "properties" in func["parameters"]

    def test_tool_names_are_unique(self):
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        assert len(names) == len(set(names))

    def test_expected_tools_exist(self):
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        assert "detect_disease" in names
        assert "get_mandi_prices" in names
        assert "search_schemes" in names

    def test_mandi_tool_requires_commodity(self):
        mandi_tool = next(
            t for t in TOOL_DEFINITIONS if t["function"]["name"] == "get_mandi_prices"
        )
        assert "commodity" in mandi_tool["function"]["parameters"]["required"]

    def test_scheme_tool_requires_query(self):
        scheme_tool = next(
            t for t in TOOL_DEFINITIONS if t["function"]["name"] == "search_schemes"
        )
        assert "query" in scheme_tool["function"]["parameters"]["required"]

    def test_disease_tool_has_no_required_params(self):
        disease_tool = next(
            t for t in TOOL_DEFINITIONS if t["function"]["name"] == "detect_disease"
        )
        assert disease_tool["function"]["parameters"].get("required", []) == []

class TestToolDispatch:
    """Verify ToolExecutor routes to the correct handler."""

    async def test_dispatch_detect_disease(self, tool_executor):
        tool_executor._execute_disease_detection = AsyncMock(return_value="disease result")
        result = await tool_executor.execute("detect_disease", {"crop_hint": "wheat"})
        tool_executor._execute_disease_detection.assert_called_once_with({"crop_hint": "wheat"})
        assert result == "disease result"

    async def test_dispatch_get_mandi_prices(self, tool_executor):
        tool_executor._execute_mandi_prices = AsyncMock(return_value="price result")
        result = await tool_executor.execute("get_mandi_prices", {"commodity": "wheat"})
        tool_executor._execute_mandi_prices.assert_called_once_with({"commodity": "wheat"})
        assert result == "price result"

    async def test_dispatch_search_schemes(self, tool_executor):
        tool_executor._execute_scheme_search = AsyncMock(return_value="scheme result")
        result = await tool_executor.execute("search_schemes", {"query": "PM-KISAN"})
        tool_executor._execute_scheme_search.assert_called_once_with({"query": "PM-KISAN"})
        assert result == "scheme result"

    async def test_dispatch_unknown_tool_raises(self, tool_executor):
        with pytest.raises(ToolExecutionError):
            await tool_executor.execute("nonexistent_tool", {})

    async def test_dispatch_wraps_unexpected_errors(self, tool_executor):
        tool_executor._execute_mandi_prices = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(ToolExecutionError):
            await tool_executor.execute("get_mandi_prices", {"commodity": "rice"})


# --- Tool selection dataset: query -> expected tool name + args ---
# These encode what tool *should* be called for each query.
# In production, the LLM picks the tool; here we test the assumption.
TOOL_SELECTION_CASES = [
    {
        "query": "What is the price of wheat in Delhi?",
        "expected_tool": "get_mandi_prices",
        "expected_args_keys": ["commodity"],
    },
    {
        "query": "Show me onion prices in Maharashtra",
        "expected_tool": "get_mandi_prices",
        "expected_args_keys": ["commodity", "state"],
    },
    {
        "query": "Rice rates in Karnal district Haryana",
        "expected_tool": "get_mandi_prices",
        "expected_args_keys": ["commodity"],
    },
    {
        "query": "What is PM-KISAN scheme?",
        "expected_tool": "search_schemes",
        "expected_args_keys": ["query"],
    },
    {
        "query": "How to apply for crop insurance?",
        "expected_tool": "search_schemes",
        "expected_args_keys": ["query"],
    },
    {
        "query": "Tell me about government subsidies for irrigation",
        "expected_tool": "search_schemes",
        "expected_args_keys": ["query"],
    },
    {
        "query": "My wheat plant has yellow spots on the leaves",
        "expected_tool": "detect_disease",
        "expected_args_keys": [],
    },
]


class TestToolSelectionDataset:
    """Verify the tool selection dataset is consistent with TOOL_DEFINITIONS."""

    def test_all_expected_tools_are_defined(self):
        defined_tools = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        for case in TOOL_SELECTION_CASES:
            assert case["expected_tool"] in defined_tools, (
                f"Expected tool '{case['expected_tool']}' not in TOOL_DEFINITIONS"
            )

    def test_expected_args_keys_match_tool_properties(self):
        tool_properties = {
            t["function"]["name"]: set(t["function"]["parameters"]["properties"].keys())
            for t in TOOL_DEFINITIONS
        }
        for case in TOOL_SELECTION_CASES:
            tool_name = case["expected_tool"]
            available_keys = tool_properties[tool_name]
            for key in case["expected_args_keys"]:
                assert key in available_keys, (
                    f"Expected arg '{key}' not in {tool_name} properties: {available_keys}"
                )
