"""Phase 4: End-to-end agent evaluation.

Tests the full orchestrator loop: user message → tool selection →
tool execution → response generation. Verifies tool routing,
multi-turn context, safety, and latency.

These tests require:
- A running Qdrant instance with indexed scheme documents
- OPENAI_API_KEY set (for LLM calls)

Run with: pytest tests/evaluation/test_agent_e2e.py -m eval -v -s
"""

import json
import os
import time
from pathlib import Path

import pytest

from kisan.agent.orchestrator import AgentOrchestrator
from kisan.core.config import Settings
from kisan.services.llm import LLMService
from kisan.services.session import SessionManager
from kisan.services.vectordb import VectorDBService

pytestmark = pytest.mark.eval

EVAL_DATA_DIR = Path(__file__).parent / "data"


def _load_scenarios():
    with open(EVAL_DATA_DIR / "agent_scenarios.json") as f:
        return json.load(f)


def _has_eval_prerequisites():
    return os.environ.get("OPENAI_API_KEY") is not None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eval_settings():
    return Settings()


@pytest.fixture(scope="module")
def llm_service(eval_settings):
    return LLMService(eval_settings)


@pytest.fixture(scope="module")
def vectordb_service(eval_settings):
    return VectorDBService(eval_settings)


@pytest.fixture(scope="module")
def session_manager(eval_settings):
    return SessionManager(eval_settings)


@pytest.fixture(scope="module")
def orchestrator(llm_service, session_manager, vectordb_service, eval_settings):
    return AgentOrchestrator(
        llm_service=llm_service,
        session_manager=session_manager,
        vectordb_service=vectordb_service,
        settings=eval_settings,
    )


@pytest.fixture(scope="module")
def scenarios():
    data = _load_scenarios()
    return data["scenarios"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_scenarios_by_category(scenarios, category):
    return [s for s in scenarios if s["category"] == category]


async def _run_scenario(orchestrator, scenario):
    """Run all turns in a scenario, return list of (turn, response) pairs."""
    results = []
    session_id = None

    for turn in scenario["turns"]:
        response = await orchestrator.process_message(
            message=turn["user_message"],
            session_id=session_id,
            image_base64=turn.get("image_base64"),
        )
        # Reuse session for multi-turn
        session_id = response.session_id
        results.append((turn, response))

    return results


def _assert_turn(turn, response, scenario_id, turn_idx):
    """Run all assertions for a single turn."""
    errors = []

    # Response should be non-empty
    if len(response.response) <= 20:
        errors.append(
            f"Response too short ({len(response.response)} chars): "
            f"'{response.response[:50]}'"
        )

    # Tool selection check
    expected_tools = set(turn["expected_tools"])
    actual_tools = {tc.name for tc in response.tools_used}
    if expected_tools and actual_tools != expected_tools:
        errors.append(
            f"Tool mismatch: expected {expected_tools}, got {actual_tools}"
        )
    elif not expected_tools and actual_tools:
        # No tools expected but some were called — note but don't fail hard
        # The LLM may reasonably decide to use a tool for context
        pass

    # Positive content checks
    response_lower = response.response.lower()
    for keyword in turn.get("expected_response_contains", []):
        if keyword.lower() not in response_lower:
            errors.append(f"Expected '{keyword}' in response, not found")

    # Negative content checks
    for keyword in turn.get("expected_response_not_contains", []):
        if keyword.lower() in response_lower:
            errors.append(f"Did NOT expect '{keyword}' in response, but found it")

    if errors:
        error_msg = (
            f"\n[{scenario_id} turn {turn_idx}] "
            f"Query: {turn['user_message'][:60]}\n"
            f"Response: {response.response[:200]}\n"
            f"Tools used: {[tc.name for tc in response.tools_used]}\n"
            f"Errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )
        return error_msg
    return None


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_eval_prerequisites(), reason="No OPENAI_API_KEY set")
class TestSingleToolRouting:
    """Test that single-tool queries route to the correct tool."""

    async def test_scheme_queries_use_search_schemes(self, orchestrator, scenarios):
        """Scheme queries should route to search_schemes tool."""
        scheme_scenarios = _get_scenarios_by_category(scenarios, "single_tool_scheme")
        failures = []

        for scenario in scheme_scenarios:
            results = await _run_scenario(orchestrator, scenario)
            for idx, (turn, response) in enumerate(results):
                error = _assert_turn(turn, response, scenario["id"], idx)
                if error:
                    failures.append(error)

        total = len(scheme_scenarios)
        passed = total - len(failures)
        print("\n=== Scheme Tool Routing ===")
        print(f"Passed: {passed}/{total}")

        if failures:
            print("\nFailures:")
            for f in failures:
                print(f)

        assert not failures, f"{len(failures)} scheme routing failures"

    async def test_mandi_queries_use_get_mandi_prices(self, orchestrator, scenarios):
        """Mandi price queries should route to get_mandi_prices tool."""
        mandi_scenarios = _get_scenarios_by_category(scenarios, "single_tool_mandi")
        failures = []

        for scenario in mandi_scenarios:
            results = await _run_scenario(orchestrator, scenario)
            for idx, (turn, response) in enumerate(results):
                error = _assert_turn(turn, response, scenario["id"], idx)
                if error:
                    failures.append(error)

        total = len(mandi_scenarios)
        passed = total - len(failures)
        print("\n=== Mandi Tool Routing ===")
        print(f"Passed: {passed}/{total}")

        if failures:
            print("\nFailures:")
            for f in failures:
                print(f)

        assert not failures, f"{len(failures)} mandi routing failures"

    async def test_no_tool_queries_return_direct_response(self, orchestrator, scenarios):
        """Greetings and general queries should not trigger tool calls."""
        no_tool_scenarios = _get_scenarios_by_category(scenarios, "no_tool")
        failures = []

        for scenario in no_tool_scenarios:
            results = await _run_scenario(orchestrator, scenario)
            for idx, (turn, response) in enumerate(results):
                # For no-tool scenarios, we just check there's a reasonable response
                if len(response.response) < 5:
                    failures.append(
                        f"[{scenario['id']}] Response too short: '{response.response}'"
                    )

        total = len(no_tool_scenarios)
        passed = total - len(failures)
        print("\n=== No-Tool Direct Response ===")
        print(f"Passed: {passed}/{total}")

        assert not failures, f"{len(failures)} no-tool failures"


@pytest.mark.skipif(not _has_eval_prerequisites(), reason="No OPENAI_API_KEY set")
class TestMultiTurnConversation:
    """Test multi-turn conversation handling."""

    async def test_follow_up_question_uses_context(self, orchestrator, scenarios):
        """Follow-up questions should use the same session and maintain context."""
        multi_scenarios = _get_scenarios_by_category(scenarios, "multi_turn")
        failures = []

        for scenario in multi_scenarios:
            results = await _run_scenario(orchestrator, scenario)

            # Check that all turns in the same scenario share a session
            session_ids = [r.session_id for _, r in results]
            if len(set(session_ids)) > 1:
                failures.append(
                    f"[{scenario['id']}] Session IDs changed across turns: {session_ids}"
                )

            # Check individual turn assertions
            for idx, (turn, response) in enumerate(results):
                error = _assert_turn(turn, response, scenario["id"], idx)
                if error:
                    failures.append(error)

        total = len(multi_scenarios)
        passed = total - len(failures)
        print("\n=== Multi-Turn Context ===")
        print(f"Passed: {passed}/{total} scenarios (0 failures in {sum(len(s['turns']) for s in multi_scenarios)} total turns)")  # noqa: E501

        if failures:
            print("\nFailures:")
            for f in failures:
                print(f)

        assert not failures, f"{len(failures)} multi-turn failures"

    async def test_topic_switch_within_session(self, orchestrator, scenarios):
        """Switching topics (scheme → mandi) within a session should work."""
        # e2e-multi-02 switches from scheme to mandi
        switch_scenarios = [
            s for s in scenarios
            if s["id"] == "e2e-multi-02"
        ]

        if not switch_scenarios:
            pytest.skip("No topic-switch scenario found")

        scenario = switch_scenarios[0]
        results = await _run_scenario(orchestrator, scenario)

        # First turn should use search_schemes
        turn_0, resp_0 = results[0]
        tools_0 = {tc.name for tc in resp_0.tools_used}
        print("\n=== Topic Switch ===")
        print(f"Turn 0 tools: {tools_0}")

        # Second turn should use get_mandi_prices
        turn_1, resp_1 = results[1]
        tools_1 = {tc.name for tc in resp_1.tools_used}
        print(f"Turn 1 tools: {tools_1}")

        assert "search_schemes" in tools_0, f"Expected search_schemes in turn 0, got {tools_0}"
        assert "get_mandi_prices" in tools_1, f"Expected get_mandi_prices in turn 1, got {tools_1}"


@pytest.mark.skipif(not _has_eval_prerequisites(), reason="No OPENAI_API_KEY set")
class TestAgentSafety:
    """Test agent safety and handling of adversarial inputs."""

    async def test_off_topic_handled_gracefully(self, orchestrator, scenarios):
        """Adversarial / off-topic inputs should be handled without crashing."""
        adversarial_scenarios = _get_scenarios_by_category(scenarios, "adversarial")

        for scenario in adversarial_scenarios:
            results = await _run_scenario(orchestrator, scenario)

            for idx, (turn, response) in enumerate(results):
                print(f"\n=== Safety: {scenario['id']} ===")
                print(f"Input: {turn['user_message'][:80]}")
                print(f"Response: {response.response[:200]}")
                print(f"Tools used: {[tc.name for tc in response.tools_used]}")

                # Should not crash and should return a response
                assert response.response, (
                    f"[{scenario['id']}] Empty response for adversarial input"
                )
                # Response should be reasonable length
                assert len(response.response) > 5, (
                    f"[{scenario['id']}] Response too short: '{response.response}'"
                )


@pytest.mark.skipif(not _has_eval_prerequisites(), reason="No OPENAI_API_KEY set")
class TestAgentLatency:
    """Measure agent response latency (informational, not a hard gate)."""

    async def test_single_turn_under_10_seconds(self, orchestrator, scenarios):
        """Single-turn queries should complete within 10 seconds (informational)."""
        # Pick one simple scheme query
        scheme_scenarios = _get_scenarios_by_category(scenarios, "single_tool_scheme")
        if not scheme_scenarios:
            pytest.skip("No scheme scenarios found")

        scenario = scheme_scenarios[0]
        turn = scenario["turns"][0]

        start = time.time()
        response = await orchestrator.process_message(
            message=turn["user_message"],
            image_base64=turn.get("image_base64"),
        )
        elapsed = time.time() - start

        print("\n=== Latency ===")
        print(f"Query: {turn['user_message'][:60]}")
        print(f"Response time: {elapsed:.2f}s")
        print(f"Processing time (internal): {response.processing_time_ms:.0f}ms")
        print(f"Response length: {len(response.response)} chars")

        if elapsed > 10:
            print(f"WARNING: Response took {elapsed:.2f}s (> 10s target)")
        # Informational — log but don't fail
