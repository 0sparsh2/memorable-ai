"""
Conscious mode: inject a stable memory block once per session.

At the first call of a session the user's profile (most important facts) plus
whatever is relevant to the opening message is retrieved and cached. Later calls
in the session reuse it, so the system prompt stays byte-stable and
prompt caching keeps working.

Inspired by Memori's conscious_ingest mode (https://github.com/GibsonAI/Memori).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConsciousMode:
    """One-shot, per-session memory injection."""

    def __init__(self, retriever: Any, max_sessions: int = 10000):
        self.retriever = retriever
        self.max_sessions = max_sessions
        self._injected_memories: Dict[str, str] = {}

    async def get_context(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        scope: Optional[Any] = None,
    ) -> str:
        key = self._key(session_id, scope)
        if key in self._injected_memories:
            return self._injected_memories[key]
        context = await self.retriever.get_context(messages, scope=scope, profile=6)
        if len(self._injected_memories) >= self.max_sessions:
            self._injected_memories.pop(next(iter(self._injected_memories)))
        self._injected_memories[key] = context
        return context

    @staticmethod
    def _key(session_id: str, scope: Optional[Any]) -> str:
        user = getattr(scope, "user_id", None) if scope is not None else None
        return session_id if user is None else f"{user}\x00{session_id}"

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""
        lines = ["Relevant memories from previous conversations:"]
        lines += [f"- [{m.get('type', 'fact')}] {m.get('content', '')}" for m in memories]
        return "\n".join(lines)

    def clear_session(self, session_id: str, scope: Optional[Any] = None):
        self._injected_memories.pop(self._key(session_id, scope), None)
        self._injected_memories.pop(session_id, None)
