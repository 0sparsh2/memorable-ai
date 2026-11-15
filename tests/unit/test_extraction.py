import pytest

from memorable_ai.core.extraction import MemoryExtractor


@pytest.mark.asyncio
async def test_pattern_extraction_preferences():
    ext = MemoryExtractor(extraction_mode="pattern")
    memories = await ext.extract(
        [{"role": "user", "content": "I prefer Python over JavaScript."}],
        None,
    )
    types = {m["type"] for m in memories}
    assert "preference" in types
    assert any("Python" in m["content"] for m in memories)


@pytest.mark.asyncio
async def test_deduplicate_memories():
    ext = MemoryExtractor(extraction_mode="pattern")
    dupes = [
        {"content": "I work at Google", "type": "fact"},
        {"content": "I work at Google", "type": "fact"},
    ]
    assert len(ext._deduplicate_memories(dupes)) == 1
