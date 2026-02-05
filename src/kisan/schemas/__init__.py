"""Pydantic schemas for the Kisan application."""

from kisan.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    Message,
    MessageRole,
    ToolCall,
)
from kisan.schemas.disease import (
    DiseaseDetectionRequest,
    DiseaseDetectionResult,
    Severity,
    Treatment,
)
from kisan.schemas.mandi import MandiPrice, MandiPriceRequest, MandiPriceResult
from kisan.schemas.scheme import (
    IndexingResult,
    SchemeDocument,
    SchemeInfo,
    SchemeResult,
)

__all__ = [
    # Chat
    "ChatRequest",
    "ChatResponse",
    "ConversationHistory",
    "Message",
    "MessageRole",
    "ToolCall",
    # Disease
    "DiseaseDetectionRequest",
    "DiseaseDetectionResult",
    "Severity",
    "Treatment",
    # Mandi
    "MandiPrice",
    "MandiPriceRequest",
    "MandiPriceResult",
    # Scheme
    "IndexingResult",
    "SchemeDocument",
    "SchemeInfo",
    "SchemeResult",
]
