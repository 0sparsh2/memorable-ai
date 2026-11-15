"""
AutoGen Integration

Inject Memorable context into AutoGen agent message flows.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AutoGenMemoryBridge:
    """Bridge Memorable MemoryEngine to AutoGen-style message lists."""

    def __init__(self, memory_engine: Any):
        self.engine = memory_engine

    async def enrich_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not hasattr(self.engine, "_inject_context"):
            return messages
        return await self.engine._inject_context(messages)

    async def record_turn(
        self, messages: List[Dict[str, Any]], response: Any
    ) -> None:
        if hasattr(self.engine, "_store_conversation"):
            await self.engine._store_conversation(messages, response)

    def enrich_messages_sync(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if hasattr(self.engine, "_inject_context_sync"):
            return self.engine._inject_context_sync(messages)
        return messages

    def record_turn_sync(self, messages: List[Dict[str, Any]], response: Any) -> None:
        if hasattr(self.engine, "_store_conversation_sync"):
            self.engine._store_conversation_sync(messages, response)
        elif hasattr(self.engine, "_store_conversation"):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.engine._store_conversation(messages, response))
                else:
                    loop.run_until_complete(
                        self.engine._store_conversation(messages, response)
                    )
            except RuntimeError:
                asyncio.run(self.engine._store_conversation(messages, response))


def setup_autogen_integration(memory_engine: Any) -> Optional[AutoGenMemoryBridge]:
    """
    Setup AutoGen integration.
    
    Args:
        memory_engine: MemoryEngine instance
        
    Returns:
        AutoGenMemoryBridge for use around agent chat turns
    """
    try:
        bridge = AutoGenMemoryBridge(memory_engine)
        logger.info("AutoGen integration ready (AutoGenMemoryBridge)")
        return bridge
    except Exception as e:
        logger.warning("AutoGen integration failed: %s", e)
        return None
