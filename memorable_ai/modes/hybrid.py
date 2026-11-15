"""
Hybrid mode: per-session profile (conscious) + per-query retrieval (auto).

Inspired by Memori's combined mode (https://github.com/GibsonAI/Memori).
"""

import logging
from typing import Any, Dict, List, Optional

from memorable_ai.modes.auto import AutoMode
from memorable_ai.modes.conscious import ConsciousMode

logger = logging.getLogger(__name__)


class HybridMode:
    """Session-level profile plus query-specific context."""

    def __init__(self, retriever: Any):
        self.conscious = ConsciousMode(retriever)
        self.auto = AutoMode(retriever)

    async def get_context(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        scope: Optional[Any] = None,
        session_summary: Optional[str] = None,
    ) -> str:
        conscious_context = await self.conscious.get_context(session_id, messages, scope=scope)
        auto_context = await self.auto.get_context(messages, scope=scope, session_summary=session_summary)
        if conscious_context and auto_context and conscious_context != auto_context:
            return f"{conscious_context}\n\n{auto_context}"
        return conscious_context or auto_context or ""

    def clear_session(self, session_id: str, scope: Optional[Any] = None):
        self.conscious.clear_session(session_id, scope)
