"""Simple in-memory chat history store."""

from collections import defaultdict


class MemoryStore:
    """Store chat history per session in memory."""

    def __init__(self) -> None:
        """Initialize the internal session dictionary."""
        self._sessions: dict[str, list[dict]] = defaultdict(list)

    def get_history(self, session_id: str) -> list[dict]:
        """Return the saved conversation for a session."""
        return list(self._sessions[session_id])

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to a session's conversation history."""
        self._sessions[session_id].append({"role": role, "content": content})


memory_store = MemoryStore()
