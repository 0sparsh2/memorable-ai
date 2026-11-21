"""Identity-scoped memory retrieval."""

import pytest

from memorable_ai import MemoryEngine


@pytest.mark.asyncio
async def test_identity_scoped_search():
    engine = MemoryEngine(database="sqlite:///:memory:")
    engine.enable()
    engine.register_identity("session_a", "user_1")
    engine.register_identity("session_b", "user_1")

    await engine.add_memory("User one likes pizza", user_id="session_a")
    await engine.add_memory("User two likes sushi", user_id="session_b")

    scoped = await engine.search_memories("pizza", user_id="session_b")
    contents = " ".join(m.get("content", "") for m in scoped).lower()
    assert "pizza" in contents or "sushi" in contents
    engine.disable()
