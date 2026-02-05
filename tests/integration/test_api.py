"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from kisan.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "kisan"
    assert "version" in data


def test_chat_endpoint_requires_message(client):
    """Test that chat endpoint requires a message."""
    response = client.post("/api/v1/chat", json={})

    assert response.status_code == 422  # Validation error


def test_chat_endpoint_empty_message(client):
    """Test that chat endpoint rejects empty message."""
    response = client.post("/api/v1/chat", json={"message": ""})

    assert response.status_code == 422


def test_get_nonexistent_session(client):
    """Test getting a non-existent session."""
    response = client.get("/api/v1/sessions/nonexistent-id")

    assert response.status_code == 404


def test_delete_nonexistent_session(client):
    """Test deleting a non-existent session."""
    response = client.delete("/api/v1/sessions/nonexistent-id")

    assert response.status_code == 404


# Note: Full chat tests require mocking the LLM service
# These are basic smoke tests for the API structure
