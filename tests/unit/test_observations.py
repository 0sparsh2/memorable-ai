"""Observer and observation log (docs/observation_memory.md), with a fake LLM."""

import json

import pytest

from memorable_ai.core.context import render_observation_log
from memorable_ai.core.models import Edge, Memory
from memorable_ai.core.scope import Scope

U = Scope(user_id="u", session_id="s1")

OBS = [
    {"content": "User adopted a beagle named Biscuit", "priority": "high", "event_date": "2024-03-02",
     "entities": ["Biscuit"], "turns": [0]},
    {"content": "Assistant recommended puppy classes at Paws Academy", "priority": "medium",
     "event_date": None, "entities": ["Paws Academy"], "turns": [1]},
    {"content": "User mentioned the weather was rainy", "priority": "low", "turns": [2]},
]


class FakeLLM:
    """Answers the observer prompt with OBS and every other prompt with []."""

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(OBS if "observation log" in prompt else [])


@pytest.fixture
def eng(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORABLE_LLM__DEFAULT_MODEL", "none")
    monkeypatch.setenv("MEMORABLE_LLM__EMBEDDING_MODEL", "hash")
    monkeypatch.setenv("MEMORABLE_LLM__TURN_EMBEDDING_MODEL", "none")
    from memorable_ai import MemoryEngine

    e = MemoryEngine(database=f"sqlite:///{tmp_path / 'obs.db'}", write_mode="sync")
    e._ensure_components()
    e._extraction.llm_model, e._extraction.llm_client = None, FakeLLM()
    e._pipeline.observations, e._pipeline.facts = True, False
    yield e
    e.close()


async def session(eng):
    from datetime import datetime

    msgs = [{"role": "user", "content": "We adopted a beagle puppy yesterday, his name is Biscuit!"},
            {"role": "assistant", "content": "Congratulations! Puppy classes at Paws Academy are great."},
            {"role": "user", "content": "It rained all day so we stayed in with him."},
            {"role": "assistant", "content": "Cosy day then."}]
    await eng.ingest(msgs, None, user_id="u", session_id="s1", occurred_at=datetime(2024, 3, 3, 10))
    await eng.end_session(user_id="u", session_id="s1")


@pytest.mark.asyncio
async def test_observe_parses_items():
    from memorable_ai.core.extraction import MemoryExtractor

    ex = MemoryExtractor(llm_client=FakeLLM(), embedding_model="hash")
    turns = [{"id": 10 + i, "role": r, "content": c} for i, (r, c) in
             enumerate([("user", "a"), ("assistant", "b"), ("user", "c")])]
    out = await ex.observe(turns, user_name="Priya")
    assert [o["type"] for o in out] == ["observation"] * 3
    assert out[0]["metadata"]["priority"] == "high" and out[0]["metadata"]["event_date"] == "2024-03-02"
    assert out[0]["source_turn_ids"] == [10] and out[1]["source_turn_ids"] == [11]
    assert out[0]["importance_score"] > out[1]["importance_score"] > out[2]["importance_score"]
    assert "event_date" not in out[1]["metadata"], "a null date is not a date"
    assert 'refer to them as "Priya"' in ex.llm_client.prompts[0]
    assert "Assistant recommended" in ex.llm_client.prompts[0], "the observer records what the assistant said"


@pytest.mark.asyncio
async def test_observe_needs_an_llm():
    from memorable_ai.core.extraction import MemoryExtractor

    assert await MemoryExtractor(llm_model="none").observe([{"role": "user", "content": "hi"}]) == []


@pytest.mark.asyncio
async def test_episode_close_writes_observations_with_provenance(eng):
    await session(eng)
    s = eng._storage.get_session()
    try:
        obs = s.query(Memory).filter(Memory.memory_type == "observation", Memory.user_id == "u").all()
        assert len(obs) == 3
        assert not s.query(Memory).filter(Memory.memory_type != "observation").count(), "facts were off"
        src = s.query(Edge).filter(Edge.kind == "derived_from", Edge.src_kind == "fact",
                                   Edge.src_id.in_([o.id for o in obs])).count()
        assert src >= 3, "observations point at their turns and episode"
        about = s.query(Edge).filter(Edge.kind == "about", Edge.src_kind == "fact",
                                     Edge.src_id.in_([o.id for o in obs])).count()
        assert about >= 2, "the entities they name are in the graph"
    finally:
        s.close()


def test_render_orders_marks_and_budgets():
    obs = [
        {"id": 2, "content": "B later", "metadata": {"stated_at": "2024-03-05T10:00", "priority": "low"}},
        {"id": 1, "content": "A first", "metadata": {"stated_at": "2024-03-03T10:00", "priority": "high",
                                                     "event_date": "2024-03-02"}},
        {"id": 3, "content": "C same day", "metadata": {"stated_at": "2024-03-05T11:00", "priority": "medium"}},
    ]
    text = render_observation_log(obs)
    assert text.splitlines() == ["Date: 2024-03-03", "- 🔴 A first (on 2024-03-02)", "Date: 2024-03-05",
                                 "- 🟢 B later", "- 🟡 C same day"]
    tight = render_observation_log(obs, max_tokens=20)  # A (8) + C (4) + two day headings (8)
    assert "B later" not in tight and "A first" in tight and "C same day" in tight, "low priority goes first"
    tighter = render_observation_log(obs, max_tokens=12)  # A and its heading only
    assert "A first" in tighter and "C same day" not in tighter, "then medium, before high"


@pytest.mark.asyncio
async def test_get_context_shows_the_log_once(eng):
    await session(eng)
    r = eng._retrieval
    r.observation_log_tokens = 2000
    ctx = await r.get_context([{"role": "user", "content": "What is my beagle called?"}], Scope(user_id="u"))
    assert "Observation log (oldest first):" in ctx and "Biscuit" in ctx
    assert ctx.count("User adopted a beagle named Biscuit") == 1
