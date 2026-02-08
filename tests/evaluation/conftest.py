"""Shared fixtures for evaluation tests."""

import json
from pathlib import Path

import pytest

EVAL_DATA_DIR = Path(__file__).parent / "data"


def load_eval_dataset(name: str) -> dict:
    """Load an evaluation dataset by name.

    Args:
        name: Filename (without extension) in tests/evaluation/data/
    """
    path = EVAL_DATA_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def scheme_retrieval_dataset():
    """Load the scheme retrieval evaluation dataset."""
    return load_eval_dataset("scheme_retrieval")


@pytest.fixture
def scheme_test_cases(scheme_retrieval_dataset):
    """Return just the test cases from the scheme dataset."""
    return scheme_retrieval_dataset["test_cases"]
