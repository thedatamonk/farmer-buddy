"""Conversation memory management for the agent."""

from kisan.schemas.chat import Message, MessageRole


class ConversationMemory:
    """Manages conversation context for the agent."""

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._messages: list[Message] = []

    def add_user_message(self, content: str, image_data: str | None = None) -> None:
        """Add a user message to memory."""
        self._messages.append(
            Message(role=MessageRole.USER, content=content, image_data=image_data)
        )
        self._trim_if_needed()

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to memory."""
        self._messages.append(Message(role=MessageRole.ASSISTANT, content=content))
        self._trim_if_needed()

    def add_system_message(self, content: str) -> None:
        """Add a system message to memory."""
        self._messages.append(Message(role=MessageRole.SYSTEM, content=content))

    def get_messages(self) -> list[Message]:
        """Get all messages in memory."""
        return self._messages.copy()

    def get_messages_for_llm(self) -> list[dict]:
        """Get messages formatted for LLM API call."""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in self._messages
        ]

    def clear(self) -> None:
        """Clear all messages."""
        self._messages = []

    def _trim_if_needed(self) -> None:
        """Trim messages if exceeding max limit, keeping system messages."""
        if len(self._messages) <= self.max_messages:
            return

        # Separate system and non-system messages
        system_messages = [m for m in self._messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in self._messages if m.role != MessageRole.SYSTEM]

        # Keep most recent non-system messages
        keep_count = self.max_messages - len(system_messages)
        other_messages = other_messages[-keep_count:] if keep_count > 0 else []

        # Rebuild with system messages first, then others
        self._messages = system_messages + other_messages

    def get_context_summary(self) -> str:
        """Get a summary of the conversation context."""
        if not self._messages:
            return "No previous conversation."

        user_messages = [m for m in self._messages if m.role == MessageRole.USER]
        if not user_messages:
            return "No user messages yet."

        topics = []
        for msg in user_messages[-3:]:  # Last 3 user messages
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            topics.append(content)

        return f"Recent topics: {'; '.join(topics)}"
