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


class BatchLLM:
    """Batched observer: one observation per date block, citing that block's first message."""

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        import re

        self.prompts.append(prompt)
        if "New messages, oldest first" not in prompt:
            return "[]"
        return json.dumps([{"content": f"User did something on {d}", "priority": "high", "turns": [int(i)]}
                           for d, i in re.findall(r"--- (\d{4}-\d{2}-\d{2}) \(\w+\) ---\n\[(\d+)\]", prompt)])


@pytest.fixture
def beng(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORABLE_LLM__DEFAULT_MODEL", "none")
    monkeypatch.setenv("MEMORABLE_LLM__EMBEDDING_MODEL", "hash")
    monkeypatch.setenv("MEMORABLE_LLM__TURN_EMBEDDING_MODEL", "none")
    from memorable_ai import MemoryEngine

    e = MemoryEngine(database=f"sqlite:///{tmp_path / 'batch.db'}", write_mode="sync")
    e._ensure_components()
    e._extraction.llm_model, e._extraction.llm_client = None, BatchLLM()
    e._pipeline.observations, e._pipeline.facts = True, False
    e._pipeline.observe_batch_tokens = 300  # each session below is ~200 tokens of messages
    yield e
    e.close()


async def long_session(eng, sid, day):
    from datetime import datetime

    pad = " We talked about it for a long while and covered many small details along the way."
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i} of session {sid}.{pad * 2}"}
            for i in range(4)]
    await eng.ingest(msgs, None, user_id="u", session_id=sid, occurred_at=datetime(2024, 3, day, 10))
    await eng.end_session(user_id="u", session_id=sid)


def observations(eng):
    s = eng._storage.get_session()
    try:
        return [(m.content, m.session_id, m.source_episode_id, (m.extra_metadata or {}).get("stated_at", "")[:10])
                for m in s.query(Memory).filter(Memory.memory_type == "observation").order_by(Memory.id)]
    finally:
        s.close()


@pytest.mark.asyncio
async def test_batched_observer_waits_for_the_threshold_then_observes_the_queue(beng):
    llm = beng._extraction.llm_client
    await long_session(beng, "s1", 3)
    assert not llm.prompts and not observations(beng), "one session is below the threshold: queued"
    await long_session(beng, "s2", 10)
    assert len(llm.prompts) == 1, "both queued sessions in one call"
    assert "--- 2024-03-03 (Sunday) ---" in llm.prompts[0] and "--- 2024-03-10 (Sunday) ---" in llm.prompts[0]
    obs = observations(beng)
    assert [(c, s, d) for c, s, _, d in obs] == [("User did something on 2024-03-03", "s1", "2024-03-03"),
                                                 ("User did something on 2024-03-10", "s2", "2024-03-10")]
    assert obs[0][2] != obs[1][2], "each observation points at the episode of the message it cites"


@pytest.mark.asyncio
async def test_flush_observes_the_rest_with_the_log_so_far(beng):
    llm = beng._extraction.llm_client
    for sid, day in (("s1", 3), ("s2", 10), ("s3", 20)):
        await long_session(beng, sid, day)
    assert len(llm.prompts) == 1 and len(observations(beng)) == 2
    assert await beng.flush_observations(user_id="u") == 1
    assert len(llm.prompts) == 2
    last = llm.prompts[1]
    assert "Message 0 of session s3" in last and "Message 0 of session s1" not in last, "only new messages"
    assert "User did something on 2024-03-03" in last.split("New messages")[0], "the log so far is shown"
    assert observations(beng)[-1][1:] [::2] == ("s3", "2024-03-20")
    assert await beng.flush_observations(user_id="u") == 0, "nothing left in the queue"


@pytest.mark.asyncio
async def test_per_episode_mode_marks_episodes_observed(eng):
    from memorable_ai.core.models import Episode

    await session(eng)
    s = eng._storage.get_session()
    try:
        metas = [e.extra_metadata or {} for e in s.query(Episode).all()]
    finally:
        s.close()
    assert metas and all(m.get("observed") for m in metas)
