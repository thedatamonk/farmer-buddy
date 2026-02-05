"""Pytest configuration and fixtures."""

import pytest

from kisan.core.config import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        openai_api_key="test-key",
        qdrant_url="http://localhost:6333",
        log_level="DEBUG",
    )


@pytest.fixture
def sample_image_base64():
    """Return a minimal valid base64-encoded image."""
    # This is a 1x1 pixel PNG
    return (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
