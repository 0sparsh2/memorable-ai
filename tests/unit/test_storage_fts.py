"""SQLite FTS5 keyword search."""

import pytest

from memorable_ai.core.storage import Storage


@pytest.mark.asyncio
async def test_fts5_search_finds_memory():
    storage = Storage("sqlite:///:memory:")
    assert storage._fts_enabled

    await storage.store_memories(
        [{"content": "Caroline attended an LGBTQ support group in Seattle", "type": "fact"}]
    )
    hits = await storage.search_memories_text("LGBTQ support", limit=5)
    assert len(hits) >= 1
    assert "LGBTQ" in hits[0]["content"]
