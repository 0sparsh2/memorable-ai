"""Base class for multimodal memory ingestion (Supermemory-inspired)."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class MultimodalMemoryPlugin(ABC):
    """Plugin interface for non-text memory sources."""

    @abstractmethod
    async def ingest(self, source: str, **kwargs) -> List[Dict[str, Any]]:
        """Ingest file/URL and return memory dicts."""

    @abstractmethod
    def supported_mime_types(self) -> List[str]:
        """MIME types this plugin handles."""


class TextFilePlugin(MultimodalMemoryPlugin):
    """Plain text / markdown files."""

    def supported_mime_types(self):
        return ["text/plain", "text/markdown"]

    async def ingest(self, source: str, **kwargs) -> List[Dict[str, Any]]:
        with open(source, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [{"content": content[:8000], "type": "context", "metadata": {"source": source}}]
