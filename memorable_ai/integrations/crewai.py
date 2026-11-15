"""
CrewAI Integration

Helper to attach Memorable memory to CrewAI agent task context.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_crew_memory_context(memory_engine: Any, task_query: str, limit: int = 8) -> str:
    """
    Build a memory context string for CrewAI task descriptions or agent backstory.
    
    Args:
        memory_engine: Enabled MemoryEngine
        task_query: Current task or goal text
        limit: Max memories to include
        
    Returns:
        Formatted memory block for injection into prompts
    """
    import asyncio

    async def _search():
        return await memory_engine.search_memories(task_query, limit=limit)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _search())
                results = future.result(timeout=10)
        else:
            results = loop.run_until_complete(_search())
    except Exception:
        results = asyncio.run(_search())

    if not results:
        return ""
    lines = ["Relevant long-term memories:"]
    for mem in results:
        lines.append(f"- [{mem.get('type', 'fact')}] {mem.get('content', '')}")
    return "\n".join(lines)


def setup_crewai_integration(memory_engine: Any) -> Dict[str, Any]:
    """
    Return helpers for CrewAI crews.
    
    Usage:
        ctx_fn = setup_crewai_integration(memory)["context_for_task"]
        agent.backstory += ctx_fn("user's project goals")
    """
    logger.info("CrewAI integration ready")

    def context_for_task(task_query: str, limit: int = 8) -> str:
        return build_crew_memory_context(memory_engine, task_query, limit=limit)

    return {"context_for_task": context_for_task, "engine": memory_engine}
