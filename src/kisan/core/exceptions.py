"""Custom exceptions for the Kisan application."""


class KisanError(Exception):
    """Base exception for all Kisan errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class LLMError(KisanError):
    """Error related to LLM operations."""

    pass


class VisionError(LLMError):
    """Error related to vision/image processing."""

    pass


class EmbeddingError(LLMError):
    """Error related to embedding generation."""

    pass


class VectorDBError(KisanError):
    """Error related to vector database operations."""

    pass


class IndexingError(VectorDBError):
    """Error during document indexing."""

    pass


class RetrievalError(VectorDBError):
    """Error during document retrieval."""

    pass


class MandiAPIError(KisanError):
    """Error related to Mandi API operations."""

    pass


class SessionError(KisanError):
    """Error related to session management."""

    pass


class SessionNotFoundError(SessionError):
    """Session not found."""

    pass


class ValidationError(KisanError):
    """Error related to input validation."""

    pass


class ToolExecutionError(KisanError):
    """Error during tool execution."""

    def __init__(self, tool_name: str, message: str, details: dict | None = None):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}", details)


class ConfigurationError(KisanError):
    """Error related to configuration."""

    pass
