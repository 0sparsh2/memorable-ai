"""
Auto mode: retrieve for every call, using the latest user message as the query.

Inspired by Memori's auto_ingest mode (https://github.com/GibsonAI/Memori).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AutoMode:
    """Per-query retrieval: facts, past episodes and excerpts relevant to this turn."""

    def __init__(self, retriever: Any):
        self.retriever = retriever

    async def get_context(
        self,
        messages: List[Dict[str, Any]],
        scope: Optional[Any] = None,
        session_summary: Optional[str] = None,
    ) -> str:
        return await self.retriever.get_context(messages, scope=scope, session_summary=session_summary)

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Plain list format (compatibility)."""
        if not memories:
            return ""
        lines = ["Relevant memories:"]
        lines += [f"- [{m.get('type', 'fact')}] {m.get('content', '')}" for m in memories]
        return "\n".join(lines)
