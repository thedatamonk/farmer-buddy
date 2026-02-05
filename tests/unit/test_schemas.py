"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from kisan.schemas.chat import ChatRequest, ChatResponse, Message, MessageRole
from kisan.schemas.disease import DiseaseDetectionResult, Severity, Treatment
from kisan.schemas.mandi import MandiPrice, MandiPriceResult


def test_chat_request_valid():
    """Test valid chat request."""
    request = ChatRequest(message="Hello, how are you?")
    assert request.message == "Hello, how are you?"
    assert request.session_id is None
    assert request.image is None


def test_chat_request_with_session():
    """Test chat request with session ID."""
    request = ChatRequest(message="Test", session_id="abc-123")
    assert request.session_id == "abc-123"


def test_chat_request_empty_message():
    """Test that empty message is rejected."""
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_request_long_message():
    """Test that overly long message is rejected."""
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * 5000)


def test_chat_response():
    """Test chat response schema."""
    response = ChatResponse(
        response="Hello!",
        session_id="abc-123",
        tools_used=[],
    )
    assert response.response == "Hello!"
    assert response.session_id == "abc-123"


def test_message_schema():
    """Test message schema."""
    msg = Message(role=MessageRole.USER, content="Test message")
    assert msg.role == MessageRole.USER
    assert msg.content == "Test message"
    assert msg.timestamp is not None


def test_disease_detection_result():
    """Test disease detection result schema."""
    result = DiseaseDetectionResult(
        crop_identified="Wheat",
        disease_detected=True,
        disease_name="Rust",
        confidence=0.85,
        severity=Severity.MODERATE,
    )
    assert result.crop_identified == "Wheat"
    assert result.disease_detected is True
    assert result.confidence == 0.85


def test_disease_detection_confidence_bounds():
    """Test confidence must be between 0 and 1."""
    with pytest.raises(ValidationError):
        DiseaseDetectionResult(disease_detected=False, confidence=1.5)

    with pytest.raises(ValidationError):
        DiseaseDetectionResult(disease_detected=False, confidence=-0.1)


def test_treatment_schema():
    """Test treatment schema."""
    treatment = Treatment(
        method="Apply neem oil",
        description="Spray neem oil solution on affected leaves",
        is_organic=True,
    )
    assert treatment.is_organic is True


def test_mandi_price():
    """Test mandi price schema."""
    price = MandiPrice(
        commodity="Wheat",
        state="Punjab",
        district="Ludhiana",
        market="Ludhiana Mandi",
        min_price=2000.0,
        max_price=2500.0,
        modal_price=2200.0,
    )
    assert price.commodity == "Wheat"
    assert price.modal_price == 2200.0


def test_mandi_price_result():
    """Test mandi price result schema."""
    result = MandiPriceResult(
        query_commodity="Wheat",
        prices=[],
        total_results=0,
    )
    assert result.query_commodity == "Wheat"
    assert result.total_results == 0
