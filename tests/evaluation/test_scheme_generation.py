"""Phase 3: Generator quality evaluation using DeepEval (LLM-as-judge).

Measures Faithfulness, Answer Relevancy, and Hallucination for the
scheme RAG pipeline's generated answers.

These tests require:
- A running Qdrant instance with indexed scheme documents
- OPENAI_API_KEY set (DeepEval uses GPT-4o as judge)

Run with: pytest tests/evaluation/test_scheme_generation.py -m eval -v -s
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

try:
    from deepeval.metrics import (AnswerRelevancyMetric, FaithfulnessMetric,  # noqa: I001
                                  HallucinationMetric)
    from deepeval.test_case import LLMTestCase

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

from kisan.core.config import Settings
from kisan.modules.schemes.retriever import SchemeRetriever
from kisan.services.llm import LLMService
from kisan.services.vectordb import VectorDBService

load_dotenv()

pytestmark = pytest.mark.eval

EVAL_DATA_DIR = Path(__file__).parent / "data"


def _load_dataset():
    with open(EVAL_DATA_DIR / "scheme_retrieval.json") as f:
        return json.load(f)


def _has_eval_prerequisites():
    return os.environ.get("OPENAI_API_KEY") is not None


def _is_non_adversarial(case: dict) -> bool:
    return not case["category"].startswith("adversarial") \
        and case.get("ground_truth_answer") is not None


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
def retriever(llm_service, vectordb_service, eval_settings):
    return SchemeRetriever(llm_service, vectordb_service, eval_settings)


@pytest.fixture(scope="module")
def dataset():
    return _load_dataset()


@pytest.fixture(scope="module")
def test_cases(dataset):
    return dataset["test_cases"]


@pytest.fixture(scope="module")
def generation_results(retriever, test_cases):
    """Run all non-adversarial queries once and cache (case, LLMTestCase) pairs.

    This avoids calling the LLM N x M times (N cases x M metrics).
    """
    pairs: list[tuple[dict, LLMTestCase]] = []

    async def _gather():
        for case in test_cases:
            if not _is_non_adversarial(case):
                continue

            result = await retriever.query(case["query"])
            retrieval_context = [doc.content for doc in result.documents]

            tc = LLMTestCase(
                input=case["query"],
                actual_output=result.answer or "",
                expected_output=case["ground_truth_answer"],
                retrieval_context=retrieval_context,
            )
            pairs.append((case, tc))

    asyncio.get_event_loop().run_until_complete(_gather())
    print(f"\n[setup] Generated answers for {len(pairs)} non-adversarial cases")
    return pairs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _measure_metric(metric, generation_results: list[tuple[dict, "LLMTestCase"]]):
    """Run a single DeepEval metric across all cached test cases.

    Returns list of (case, score) tuples.
    """
    scored: list[tuple[dict, float]] = []
    for case, tc in generation_results:
        metric.measure(tc)
        scored.append((case, metric.score))
    return scored


def _print_summary(metric_name: str, scored: list[tuple[dict, float]]):
    scores = [s for _, s in scored]
    avg = sum(scores) / len(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0

    print(f"\n=== {metric_name} Results ===")
    print(f"  Mean:  {avg:.3f}")
    print(f"  Min:   {min_score:.3f}")
    print(f"  Max:   {max_score:.3f}")
    print(f"  Cases: {len(scores)}")

    # Per-case breakdown
    for case, score in scored:
        status = "PASS" if score >= 0.7 else "FAIL"
        print(f"  [{status}] {case['id']} ({case['category']}): {score:.3f}")

    return avg


def _filter_by_category(scored: list[tuple[dict, float]], category: str):
    return [(c, s) for c, s in scored if c["category"] == category]


# ---------------------------------------------------------------------------
# Phase 3a: Overall Generation Quality
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DEEPEVAL_AVAILABLE, reason="deepeval not installed")
@pytest.mark.skipif(not _has_eval_prerequisites(), reason="No OPENAI_API_KEY set")
class TestSchemeGenerationQuality:
    """Evaluate generation quality across all non-adversarial cases."""

    def test_faithfulness_above_threshold(self, generation_results):
        """Average FaithfulnessMetric across all cases >= 0.7."""
        metric = FaithfulnessMetric(threshold=0.7)
        scored = _measure_metric(metric, generation_results)
        avg = _print_summary("Faithfulness", scored)

        failures = [(c, s) for c, s in scored if s < 0.7]
        if failures:
            print(f"\n  Failures ({len(failures)}):")
            for c, s in failures:
                print(f"    {c['id']}: {s:.3f} — {c['query'][:60]}")

        assert avg >= 0.7, f"Average faithfulness is {avg:.3f}, expected >= 0.7"

    def test_answer_relevancy_above_threshold(self, generation_results):
        """Average AnswerRelevancyMetric across all cases >= 0.7."""
        metric = AnswerRelevancyMetric(threshold=0.7)
        scored = _measure_metric(metric, generation_results)
        avg = _print_summary("Answer Relevancy", scored)

        failures = [(c, s) for c, s in scored if s < 0.7]
        if failures:
            print(f"\n  Failures ({len(failures)}):")
            for c, s in failures:
                print(f"    {c['id']}: {s:.3f} — {c['query'][:60]}")

        assert avg >= 0.7, f"Average answer relevancy is {avg:.3f}, expected >= 0.7"

    def test_hallucination_below_threshold(self, generation_results):
        """Average HallucinationMetric across all cases <= 0.3."""
        metric = HallucinationMetric(threshold=0.3)
        scored = _measure_metric(metric, generation_results)
        avg = _print_summary("Hallucination", scored)

        high_hallucination = [(c, s) for c, s in scored if s > 0.3]
        if high_hallucination:
            print(f"\n  High hallucination ({len(high_hallucination)}):")
            for c, s in high_hallucination:
                print(f"    {c['id']}: {s:.3f} — {c['query'][:60]}")

        assert avg <= 0.3, f"Average hallucination is {avg:.3f}, expected <= 0.3"


# ---------------------------------------------------------------------------
# Phase 3b: Generation Quality By Category
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DEEPEVAL_AVAILABLE, reason="deepeval not installed")
@pytest.mark.skipif(not _has_eval_prerequisites(), reason="No OPENAI_API_KEY set")
class TestSchemeGenerationByCategory:
    """Evaluate generation quality broken down by query category."""

    def test_factual_faithfulness(self, generation_results):
        """Factual category should have high faithfulness >= 0.8."""
        metric = FaithfulnessMetric(threshold=0.8)
        all_scored = _measure_metric(metric, generation_results)
        factual = _filter_by_category(all_scored, "factual")

        if not factual:
            pytest.skip("No factual cases found")

        scores = [s for _, s in factual]
        avg = sum(scores) / len(scores)
        print(f"\nFactual Faithfulness: {avg:.3f} ({len(factual)} cases)")
        for c, s in factual:
            print(f"  {c['id']}: {s:.3f}")

        assert avg >= 0.8, f"Factual faithfulness is {avg:.3f}, expected >= 0.8"

    def test_process_relevancy(self, generation_results):
        """Process category should have good relevancy >= 0.7."""
        metric = AnswerRelevancyMetric(threshold=0.7)
        all_scored = _measure_metric(metric, generation_results)
        process = _filter_by_category(all_scored, "process")

        if not process:
            pytest.skip("No process cases found")

        scores = [s for _, s in process]
        avg = sum(scores) / len(scores)
        print(f"\nProcess Relevancy: {avg:.3f} ({len(process)} cases)")
        for c, s in process:
            print(f"  {c['id']}: {s:.3f}")

        assert avg >= 0.7, f"Process relevancy is {avg:.3f}, expected >= 0.7"

    def test_eligibility_relevancy(self, generation_results):
        """Eligibility category should have good relevancy >= 0.7."""
        metric = AnswerRelevancyMetric(threshold=0.7)
        all_scored = _measure_metric(metric, generation_results)
        eligibility = _filter_by_category(all_scored, "eligibility")

        if not eligibility:
            pytest.skip("No eligibility cases found")

        scores = [s for _, s in eligibility]
        avg = sum(scores) / len(scores)
        print(f"\nEligibility Relevancy: {avg:.3f} ({len(eligibility)} cases)")
        for c, s in eligibility:
            print(f"  {c['id']}: {s:.3f}")

        assert avg >= 0.7, f"Eligibility relevancy is {avg:.3f}, expected >= 0.7"
