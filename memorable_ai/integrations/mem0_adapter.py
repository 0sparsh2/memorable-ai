"""
Optional Mem0 backend adapter — delegate extraction/storage to Mem0 when configured.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Mem0Adapter:
    """Bridge Memorable API to Mem0 Memory client (optional dependency)."""

    def __init__(self, api_key: Optional[str] = None, user_id: str = "default"):
        self.user_id = user_id
        self._client = None
        try:
            from mem0 import MemoryClient

            self._client = MemoryClient(api_key=api_key)
            logger.info("Mem0 adapter initialized")
        except ImportError:
            logger.warning("mem0 package not installed; pip install mem0ai for adapter")

    @property
    def available(self) -> bool:
        return self._client is not None

    async def add(self, content: str, metadata: Optional[Dict] = None) -> Any:
        if not self._client:
            return None
        return self._client.add(content, user_id=self.user_id, metadata=metadata or {})

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self._client:
            return []
        results = self._client.search(query, user_id=self.user_id, limit=limit)
        return [
            {"content": r.get("memory", r.get("text", "")), "metadata": r}
            for r in (results or [])
        ]
