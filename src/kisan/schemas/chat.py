"""Chat-related Pydantic schemas."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    """Role of the message sender."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single message in a conversation."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    image_data: str | None = Field(default=None, description="Base64 encoded image")


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: str | None = Field(default=None, description="Session ID for continuity")
    image: str | None = Field(default=None, description="Base64 image for disease detection")


class ToolCall(BaseModel):
    """Information about a tool that was called."""

    name: str
    input: dict
    output: str | None = None


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    response: str = Field(..., description="Assistant's response")
    session_id: str = Field(..., description="Session ID for continuity")
    tools_used: list[ToolCall] = Field(default_factory=list, description="Tools used")
    processing_time_ms: float | None = Field(default=None, description="Processing time ms")


class ConversationHistory(BaseModel):
    """Full conversation history for a session."""

    session_id: str
    messages: list[Message]
    created_at: datetime
    updated_at: datetime
