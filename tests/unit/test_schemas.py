"""Tests for Pydantic schemas."""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from kisan.schemas.chat import ChatRequest, ChatResponse, Message, MessageRole, ToolCall
from kisan.schemas.disease import DiseaseDetectionResult, Severity, Treatment
from kisan.schemas.mandi import MandiPrice, MandiPriceResult
from kisan.schemas.scheme import IndexingResult, SchemeDocument, SchemeResult


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


def test_mandi_price_full():
    """Test MandiPrice with all fields populated."""
    price = MandiPrice(
        commodity="Rice",
        variety="Basmati",
        state="Haryana",
        district="Karnal",
        market="Karnal Grain Market",
        min_price=3500.0,
        max_price=4200.0,
        modal_price=3800.0,
        arrival_date=date(2024, 1, 15),
    )
    assert price.variety == "Basmati"


def test_mandi_price_result_with_cache():
    """Test MandiPriceResult with cache and prices populated."""
    result = MandiPriceResult(
        query_commodity="Wheat",
        query_location="Delhi",
        prices=[
            MandiPrice(
                commodity="Wheat",
                state="Delhi",
                district="New Delhi",
                market="Azadpur",
                min_price=2000,
                max_price=2500,
                modal_price=2250,
            )
        ],
        total_results=1,
        cache_hit=True,
        last_updated=datetime(2024, 1, 15, 10, 0),
    )
    assert result.cache_hit is True
    assert len(result.prices) == 1


def test_tool_call():
    """Test ToolCall schema."""
    tc = ToolCall(
        name="get_mandi_prices",
        input={"commodity": "wheat", "state": "Delhi"},
        output="Price data...",
    )
    assert tc.name == "get_mandi_prices"


def test_chat_response_minimal():
    """Test ChatResponse with only required fields."""
    resp = ChatResponse(response="Hello!", session_id="xyz")
    assert resp.tools_used == []
    assert resp.processing_time_ms is None


def test_disease_result_serialization():
    """Test DiseaseDetectionResult serializes severity and nested models."""
    result = DiseaseDetectionResult(
        crop_identified="Tomato",
        disease_detected=True,
        disease_name="Late Blight",
        confidence=0.8,
        severity=Severity.SEVERE,
        treatments=[
            Treatment(method="Spray", description="Use Mancozeb", is_organic=False)
        ],
    )
    data = result.model_dump()
    assert data["severity"] == "severe"
    assert data["treatments"][0]["is_organic"] is False


def test_scheme_document_valid():
    """Test SchemeDocument construction and score bounds."""
    doc = SchemeDocument(
        content="PM-KISAN provides Rs 6000 per year",
        source="PMKisanSamanNidhi.pdf",
        page_number=1,
        score=0.87,
    )
    assert doc.content == "PM-KISAN provides Rs 6000 per year"
    assert doc.score == 0.87


def test_scheme_document_score_bounds():
    """Test SchemeDocument rejects out-of-range scores."""
    with pytest.raises(ValidationError):
        SchemeDocument(content="test", source="test.pdf", score=1.5)
    with pytest.raises(ValidationError):
        SchemeDocument(content="test", source="test.pdf", score=-0.1)


def test_scheme_result():
    """Test SchemeResult with defaults and populated fields."""
    minimal = SchemeResult(query="What is PM-KISAN?")
    assert minimal.documents == []
    assert minimal.answer is None

    full = SchemeResult(
        query="What is PM-KISAN?",
        documents=[SchemeDocument(content="test", source="test.pdf", score=0.8)],
        answer="PM-KISAN is a direct benefit transfer scheme.",
        schemes_mentioned=["PM-KISAN"],
    )
    assert len(full.documents) == 1
    assert "PM-KISAN" in full.schemes_mentioned


def test_indexing_result():
    """Test IndexingResult for success and failure cases."""
    success = IndexingResult(
        documents_processed=3,
        chunks_created=150,
        collection_name="kisan_schemes",
        success=True,
    )
    assert success.errors == []

    failure = IndexingResult(
        documents_processed=3,
        chunks_created=100,
        collection_name="kisan_schemes",
        success=False,
        errors=["Failed to process file.pdf"],
    )
    assert len(failure.errors) == 1
