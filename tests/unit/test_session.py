"""Tests for session management."""

import pytest

from kisan.core.config import Settings
from kisan.core.exceptions import SessionNotFoundError
from kisan.schemas.chat import MessageRole
from kisan.services.session import SessionManager


@pytest.fixture
def session_manager():
    """Create a session manager for testing."""
    settings = Settings()
    return SessionManager(settings)


def test_create_session(session_manager):
    """Test creating a new session."""
    session_id = session_manager.create_session()

    assert session_id is not None
    assert len(session_id) == 36  # UUID format
    assert session_manager.session_exists(session_id)


def test_get_session(session_manager):
    """Test getting an existing session."""
    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)

    assert session.session_id == session_id
    assert len(session.messages) == 0


def test_get_nonexistent_session(session_manager):
    """Test getting a non-existent session raises error."""
    with pytest.raises(SessionNotFoundError):
        session_manager.get_session("nonexistent-id")


def test_add_message(session_manager):
    """Test adding messages to a session."""
    session_id = session_manager.create_session()

    session_manager.add_message(session_id, MessageRole.USER, "Hello")
    session_manager.add_message(session_id, MessageRole.ASSISTANT, "Hi there!")

    messages = session_manager.get_messages(session_id)
    assert len(messages) == 2
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "Hello"
    assert messages[1].role == MessageRole.ASSISTANT


def test_get_or_create_session_new(session_manager):
    """Test get_or_create creates new session when needed."""
    session_id, session = session_manager.get_or_create_session(None)

    assert session_id is not None
    assert session.session_id == session_id


def test_get_or_create_session_existing(session_manager):
    """Test get_or_create returns existing session."""
    original_id = session_manager.create_session()
    session_manager.add_message(original_id, MessageRole.USER, "Test")

    session_id, session = session_manager.get_or_create_session(original_id)

    assert session_id == original_id
    assert len(session.messages) == 1


def test_clear_session(session_manager):
    """Test clearing session messages."""
    session_id = session_manager.create_session()
    session_manager.add_message(session_id, MessageRole.USER, "Test")

    session_manager.clear_session(session_id)

    messages = session_manager.get_messages(session_id)
    assert len(messages) == 0


def test_delete_session(session_manager):
    """Test deleting a session."""
    session_id = session_manager.create_session()
    session_manager.delete_session(session_id)

    assert not session_manager.session_exists(session_id)


def test_message_trimming(session_manager):
    """Test that messages are trimmed when exceeding max."""
    session_manager.max_history = 5
    session_id = session_manager.create_session()

    for i in range(10):
        session_manager.add_message(session_id, MessageRole.USER, f"Message {i}")

    messages = session_manager.get_messages(session_id)
    assert len(messages) == 5
    # Should have the last 5 messages
    assert messages[0].content == "Message 5"


def test_get_messages_for_llm(session_manager):
    """Test getting messages in LLM format."""
    session_id = session_manager.create_session()
    session_manager.add_message(session_id, MessageRole.USER, "Question")
    session_manager.add_message(session_id, MessageRole.ASSISTANT, "Answer")

    llm_messages = session_manager.get_messages_for_llm(session_id)

    assert len(llm_messages) == 2
    assert llm_messages[0] == {"role": "user", "content": "Question"}
    assert llm_messages[1] == {"role": "assistant", "content": "Answer"}
