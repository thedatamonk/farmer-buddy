"""Tests for configuration module."""



from kisan.core.config import Settings, get_settings


def test_settings_defaults():
    """Test that settings have sensible defaults."""
    settings = Settings()

    assert settings.chat_model == "gpt-4o"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "kisan_schemes"
    assert settings.log_level == "INFO"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50


def test_settings_from_env(monkeypatch):
    """Test that settings can be loaded from environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("CHUNK_SIZE", "1000")

    # Clear cache to get fresh settings
    get_settings.cache_clear()

    settings = Settings()

    assert settings.openai_api_key == "test-key"
    assert settings.log_level == "DEBUG"
    assert settings.chunk_size == 1000


def test_get_settings_cached():
    """Test that get_settings returns cached instance."""
    get_settings.cache_clear()

    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2
