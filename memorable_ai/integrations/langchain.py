"""
LangChain Integration

LangChain callback handler for Memorable memory injection and storage.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def setup_langchain_integration(memory_engine: Any):
    """
    Setup LangChain integration.
    
    Args:
        memory_engine: MemoryEngine instance
        
    Returns:
        MemorableCallbackHandler or None if LangChain not installed
    """
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError:
        try:
            from langchain.callbacks.base import BaseCallbackHandler
        except ImportError:
            logger.warning("LangChain not available, integration skipped")
            return None

    class MemorableCallbackHandler(BaseCallbackHandler):
        """Inject memories before LLM calls and store after."""

        def __init__(self, engine: Any):
            super().__init__()
            self.engine = engine
            self._pending_messages: List[Dict[str, Any]] = []

        def on_chat_model_start(self, serialized, messages, **kwargs):
            flat = self._flatten_messages(messages)
            self._pending_messages = flat
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._inject_async(flat))
                else:
                    loop.run_until_complete(self._inject_async(flat))
            except Exception as e:
                logger.debug(f"LangChain inject skipped: {e}")

        def on_llm_end(self, response, **kwargs):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(
                        self.engine._store_conversation(self._pending_messages, response)
                    )
                else:
                    loop.run_until_complete(
                        self.engine._store_conversation(self._pending_messages, response)
                    )
            except Exception as e:
                logger.debug(f"LangChain store failed: {e}")

        async def _inject_async(self, messages: List[Dict[str, Any]]):
            if hasattr(self.engine, "_inject_context"):
                await self.engine._inject_context(messages)

        def _flatten_messages(self, messages) -> List[Dict[str, Any]]:
            flat = []
            for batch in messages:
                for msg in batch:
                    if hasattr(msg, "type") and hasattr(msg, "content"):
                        role = getattr(msg, "type", "user")
                        if role == "human":
                            role = "user"
                        elif role == "ai":
                            role = "assistant"
                        flat.append({"role": role, "content": str(msg.content)})
                    elif isinstance(msg, dict):
                        flat.append(msg)
            return flat

    logger.info("LangChain integration ready")
    return MemorableCallbackHandler(memory_engine)
