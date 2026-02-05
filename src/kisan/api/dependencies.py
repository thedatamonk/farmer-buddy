"""FastAPI dependency injection."""

from functools import lru_cache

from kisan.core.config import Settings, get_settings
from kisan.services.llm import LLMService
from kisan.services.session import SessionManager
from kisan.services.vectordb import VectorDBService


@lru_cache
def get_llm_service() -> LLMService:
    """Get LLM service instance."""
    settings = get_settings()
    return LLMService(settings)


@lru_cache
def get_session_manager() -> SessionManager:
    """Get session manager instance."""
    settings = get_settings()
    return SessionManager(settings)


@lru_cache
def get_vectordb_service() -> VectorDBService:
    """Get vector database service instance."""
    settings = get_settings()
    return VectorDBService(settings)


def get_settings_dep() -> Settings:
    """Get settings dependency."""
    return get_settings()
