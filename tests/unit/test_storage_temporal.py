import pytest

from memorable_ai.core.storage import Storage


@pytest.mark.asyncio
async def test_store_and_invalidate_memory(tmp_path):
    db = f"sqlite:///{tmp_path}/t.db"
    storage = Storage(db, namespace="test")
    await storage.store_memories([{"content": "old fact", "type": "fact"}])
    memories = await storage.get_memories(limit=10)
    assert len(memories) == 1
    mid = memories[0]["id"]
    await storage.invalidate_memory(mid)
    active = await storage.get_memories(limit=10)
    assert len(active) == 0
