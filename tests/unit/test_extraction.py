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


def test_parse_json_keeps_the_complete_items_of_a_cut_off_array():
    from memorable_ai.core.llm_client import parse_json, salvage_array

    full = '[{"content": "A", "relations": [["x", "likes", "y"]]}, {"content": "B", "relations": []}'
    cut = full + ', {"content": "C", "relations": [["x", "lik'
    assert parse_json(cut) == [{"content": "A", "relations": [["x", "likes", "y"]]}, {"content": "B", "relations": []}]
    assert parse_json("```json\n" + cut) == parse_json(cut), "inside a fence too"
    assert parse_json('[{"a": 1}]') == [{"a": 1}]
    assert salvage_array('[ {"a": 1} , {"b": 2}]') == [{"a": 1}, {"b": 2}]
    import pytest

    with pytest.raises(ValueError):
        parse_json('[{"content": "only one, cut')


@pytest.mark.asyncio
async def test_extraction_output_budget_grows_with_the_session():
    from datetime import datetime

    from memorable_ai.core.extraction import MemoryExtractor

    ext = MemoryExtractor(llm_model="stub")
    budgets = []

    async def fake(prompt, max_tokens=800):
        budgets.append(max_tokens)
        return []

    ext._call_llm_json = fake
    for n in (2, 20, 60):
        await ext._extract_llm([{"role": "user", "content": f"m{i}"} for i in range(n)], [], "", datetime(2026, 11, 18))
    assert budgets == [1100, 3800, 4096]
