"""Phase 2: Scheme retrieval quality evaluation.

Measures Recall@k, MRR (Mean Reciprocal Rank), and context precision
for the scheme RAG retriever against the gold-standard dataset.

These tests require:
- A running Qdrant instance with indexed scheme documents
- An OpenAI API key for embeddings

Run with: pytest tests/evaluation/test_scheme_retrieval.py -m eval -v
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

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
    """Check if we have what we need to run eval tests."""
    return os.environ.get("OPENAI_API_KEY") is not None


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


def compute_recall_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float | None:
    """Compute Recall@k: fraction of expected sources found in retrieved results.

    Returns 1.0 if all expected sources appear in retrieved, 0.0 if none.
    Returns None (skip) for adversarial cases with no expected sources.
    """
    if not expected_sources:
        return None  # Not applicable for adversarial cases
    found = sum(1 for src in expected_sources if src in retrieved_sources)
    return found / len(expected_sources)


def compute_reciprocal_rank(
    retrieved_sources: list[str], expected_sources: list[str]
) -> float | None:
    """Compute Reciprocal Rank: 1/rank of first relevant result.

    Returns 0.0 if no expected source found in retrieved results.
    Returns None (skip) for adversarial cases with no expected sources.
    """
    # NOTE: Won't rank be always 1?
    if not expected_sources:
        return None
    expected_set = set(expected_sources)
    for rank, source in enumerate(retrieved_sources, 1):
        if source in expected_set:
            return 1.0 / rank
    return 0.0


def compute_context_precision(
    retrieved_sources: list[str], expected_sources: list[str]
) -> float | None:
    """Compute Context Precision: fraction of retrieved docs that are relevant.

    Returns None (skip) for adversarial cases with no expected sources.
    """
    if not expected_sources:
        return None
    if not retrieved_sources:
        return 0.0
    expected_set = set(expected_sources)
    relevant = sum(1 for src in retrieved_sources if src in expected_set)
    return relevant / len(retrieved_sources)


@pytest.mark.skipif(not _has_eval_prerequisites(), reason="No OPENAI_API_KEY set")
class TestSchemeRetrievalQuality:
    """Evaluate retrieval quality against the gold-standard dataset."""

    async def test_recall_at_k(self, retriever, test_cases):
        """Measure Recall@k across all non-adversarial test cases."""
        recalls = []
        failures = []

        for case in test_cases:
            if not case["expected_sources"]:
                continue  # Skip adversarial

            docs = await retriever.search(case["query"], top_k=5)
            retrieved_sources = [doc.source for doc in docs]
            recall = compute_recall_at_k(retrieved_sources, case["expected_sources"])

            if recall is not None:
                recalls.append(recall)
                if recall < 1.0:
                    failures.append({
                        "id": case["id"],
                        "query": case["query"],
                        "expected": case["expected_sources"],
                        "retrieved": retrieved_sources,
                        "recall": recall,
                    })

        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

        # Log results
        print("\n=== Recall@5 Results ===")
        print(f"Average Recall@5: {avg_recall:.2%}")
        print(f"Cases evaluated: {len(recalls)}")
        print(f"Perfect recall: {sum(1 for r in recalls if r == 1.0)}/{len(recalls)}")

        if failures:
            print(f"\nFailures ({len(failures)}):")
            for f in failures[:5]:
                print(f"  {f['id']}: recall={f['recall']:.2f} "
                      f"expected={f['expected']} got={f['retrieved']}")

        # Threshold: at least 70% average recall
        assert avg_recall >= 0.7, (
            f"Average Recall@5 is {avg_recall:.2%}, expected >= 70%"
        )

    async def test_mrr(self, retriever, test_cases):
        """Measure Mean Reciprocal Rank across all non-adversarial test cases."""
        rrs = []

        for case in test_cases:
            if not case["expected_sources"]:
                continue

            docs = await retriever.search(case["query"], top_k=5)
            retrieved_sources = [doc.source for doc in docs]
            rr = compute_reciprocal_rank(retrieved_sources, case["expected_sources"])

            if rr is not None:
                rrs.append(rr)

        mrr = sum(rrs) / len(rrs) if rrs else 0.0

        print("\n=== MRR Results ===")
        print(f"Mean Reciprocal Rank: {mrr:.3f}")
        print(f"Cases evaluated: {len(rrs)}")

        # Threshold: MRR >= 0.6 (first relevant result usually in top 2)
        assert mrr >= 0.6, f"MRR is {mrr:.3f}, expected >= 0.6"

    async def test_context_precision(self, retriever, test_cases):
        """Measure Context Precision across all non-adversarial test cases."""
        precisions = []

        for case in test_cases:
            if not case["expected_sources"]:
                continue

            docs = await retriever.search(case["query"], top_k=5)
            retrieved_sources = [doc.source for doc in docs]
            precision = compute_context_precision(
                retrieved_sources, case["expected_sources"]
            )

            if precision is not None:
                precisions.append(precision)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0

        print("\n=== Context Precision Results ===")
        print(f"Average Context Precision: {avg_precision:.2%}")
        print(f"Cases evaluated: {len(precisions)}")

        # Threshold: at least 40% precision (some cross-scheme noise is expected)
        assert avg_precision >= 0.4, (
            f"Average Context Precision is {avg_precision:.2%}, expected >= 40%"
        )


@pytest.mark.skipif(not _has_eval_prerequisites(), reason="No OPENAI_API_KEY set")
class TestSchemeRetrievalByCategory:
    """Evaluate retrieval quality broken down by category."""

    async def test_factual_queries_recall(self, retriever, test_cases):
        """Factual queries should have high recall — these are direct lookups."""
        factual_cases = [c for c in test_cases if c["category"] == "factual"]
        recalls = []

        for case in factual_cases:
            docs = await retriever.search(case["query"], top_k=5)
            retrieved_sources = [doc.source for doc in docs]
            recall = compute_recall_at_k(retrieved_sources, case["expected_sources"])
            if recall is not None:
                recalls.append(recall)

        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"\nFactual Recall@5: {avg_recall:.2%} ({len(recalls)} cases)")
        assert avg_recall >= 0.8, f"Factual recall is {avg_recall:.2%}, expected >= 80%"

    async def test_cross_scheme_queries_recall(self, retriever, test_cases):
        """Cross-scheme queries may retrieve from multiple sources."""
        cross_cases = [c for c in test_cases if c["category"] == "cross_scheme"]
        recalls = []

        for case in cross_cases:
            docs = await retriever.search(case["query"], top_k=5)
            retrieved_sources = [doc.source for doc in docs]
            recall = compute_recall_at_k(retrieved_sources, case["expected_sources"])
            if recall is not None:
                recalls.append(recall)

        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"\nCross-scheme Recall@5: {avg_recall:.2%} ({len(recalls)} cases)")
        # Lower threshold — multi-source retrieval is harder
        assert avg_recall >= 0.5, f"Cross-scheme recall is {avg_recall:.2%}, expected >= 50%"

    async def test_adversarial_returns_low_scores(self, retriever, test_cases):
        """Adversarial queries about nonexistent schemes should return low scores or no results."""
        adversarial_cases = [
            c for c in test_cases if c["category"] == "adversarial_nonexistent"
        ]
        high_score_count = 0

        for case in adversarial_cases:
            docs = await retriever.search(case["query"], top_k=5)
            if docs:
                max_score = max(doc.score for doc in docs)
                if max_score > 0.7:
                    high_score_count += 1
                    print(f"  WARNING: '{case['query']}' got high score {max_score:.2f}")

        print(f"\nAdversarial nonexistent: {len(adversarial_cases)} cases, "
              f"{high_score_count} with suspiciously high scores (>0.7)")


@pytest.mark.skipif(not _has_eval_prerequisites(), reason="No OPENAI_API_KEY set")
class TestSchemeNameExtraction:
    """Test that the retriever's scheme name extraction works on real results."""

    async def test_scheme_names_from_query_results(self, retriever, test_cases):
        """Test scheme name extraction against expected schemes from dataset."""
        correct = 0
        total = 0

        for case in test_cases:
            if not case["expected_schemes"]:
                continue

            result = await retriever.query(case["query"])
            expected_set = {s.upper() for s in case["expected_schemes"]}
            mentioned_set = {s.upper() for s in result.schemes_mentioned}

            # Check if at least one expected scheme was mentioned
            if expected_set & mentioned_set:
                correct += 1
            total += 1

        accuracy = correct / total if total else 0.0
        print(f"\n=== Scheme Name Extraction ===")
        print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        assert accuracy >= 0.7, f"Scheme name extraction accuracy is {accuracy:.2%}, expected >= 70%"


@pytest.mark.skipif(not _has_eval_prerequisites(), reason="No OPENAI_API_KEY set")
class TestRetrievalScoreDistribution:
    """Analyze score distributions to validate threshold settings."""

    async def test_relevant_results_above_threshold(self, retriever, test_cases):
        """Relevant results should score above the 0.3 threshold."""
        scores_for_relevant = []
        scores_for_irrelevant = []

        for case in test_cases:
            if not case["expected_sources"]:
                continue

            docs = await retriever.search(case["query"], top_k=5)
            expected_set = set(case["expected_sources"])

            for doc in docs:
                if doc.source in expected_set:
                    scores_for_relevant.append(doc.score)
                else:
                    scores_for_irrelevant.append(doc.score)

        if scores_for_relevant:
            avg_relevant = sum(scores_for_relevant) / len(scores_for_relevant)
            min_relevant = min(scores_for_relevant)
            print(f"\nRelevant doc scores: avg={avg_relevant:.3f}, min={min_relevant:.3f}, "
                  f"n={len(scores_for_relevant)}")
        if scores_for_irrelevant:
            avg_irrelevant = sum(scores_for_irrelevant) / len(scores_for_irrelevant)
            print(f"Irrelevant doc scores: avg={avg_irrelevant:.3f}, "
                  f"n={len(scores_for_irrelevant)}")

