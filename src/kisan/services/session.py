"""Session management service."""

import uuid
from datetime import datetime

from kisan.core.config import Settings
from kisan.core.exceptions import SessionNotFoundError
from kisan.core.logging import logger
from kisan.schemas.chat import ConversationHistory, Message, MessageRole


class SessionManager:
    """In-memory session storage with interface for future Redis upgrade."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.max_history = settings.max_conversation_history
        self._sessions: dict[str, ConversationHistory] = {}

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        self._sessions[session_id] = ConversationHistory(
            session_id=session_id,
            messages=[],
            created_at=now,
            updated_at=now,
        )
        logger.debug(f"Created new session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> ConversationHistory:
        """Get a session by ID."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return self._sessions[session_id]

    def get_or_create_session(self, session_id: str | None) -> tuple[str, ConversationHistory]:
        """Get existing session or create new one."""
        if session_id and session_id in self._sessions:
            return session_id, self._sessions[session_id]

        new_id = self.create_session()
        return new_id, self._sessions[new_id]

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        image_data: str | None = None,
    ) -> None:
        """Add a message to a session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        session = self._sessions[session_id]
        message = Message(
            role=role,
            content=content,
            image_data=image_data,
        )
        session.messages.append(message)
        session.updated_at = datetime.utcnow()

        # Trim history if needed
        if len(session.messages) > self.max_history:
            session.messages = session.messages[-self.max_history :]

        logger.debug(f"Added {role} message to session {session_id}")

    def get_messages(self, session_id: str) -> list[Message]:
        """Get all messages in a session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return self._sessions[session_id].messages

    def get_messages_for_llm(self, session_id: str) -> list[dict]:
        """Get messages formatted for LLM API."""
        messages = self.get_messages(session_id)
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

    def clear_session(self, session_id: str) -> None:
        """Clear a session's messages."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        self._sessions[session_id].messages = []
        self._sessions[session_id].updated_at = datetime.utcnow()
        logger.debug(f"Cleared session: {session_id}")

    def delete_session(self, session_id: str) -> None:
        """Delete a session entirely."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        del self._sessions[session_id]
        logger.debug(f"Deleted session: {session_id}")

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self._sessions
