"""Connector plugin SDK (Notion, Slack, etc. — implement per connector)."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ConnectorPlugin(ABC):
    """Sync external sources into Memorable storage."""

    name: str = "base"

    @abstractmethod
    async def sync(self, memory_engine: Any, since: Optional[str] = None) -> int:
        """Pull documents and store as memories. Returns count synced."""

    @abstractmethod
    def required_env_vars(self) -> List[str]:
        """Environment variables required for this connector."""


class StubConnector(ConnectorPlugin):
    """Placeholder for custom connectors."""

    name = "stub"

    async def sync(self, memory_engine: Any, since: Optional[str] = None) -> int:
        return 0

    def required_env_vars(self) -> List[str]:
        return []
