"""End-to-end: conversations -> turns/episodes/facts/graph -> recall in a later session."""

from datetime import datetime, timedelta

import pytest

T0 = datetime(2026, 10, 1, 18, 0)


def _conv(*pairs):
    msgs = []
    for u, a in pairs:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return msgs


@pytest.mark.asyncio
async def test_cross_session_recall_with_supersession(engine_factory):
    eng = engine_factory()
    s1 = _conv(
        ("Hi, I'm Dana. I live in Austin and work at Dell.", "Nice to meet you, Dana!"),
        ("My brother Sam lives in Denver. I'm vegetarian.", "Got it."),
    )
    # Client resends the growing history each call; only new turns are stored.
    await eng.ingest(s1[:2], None, user_id="dana", session_id="s1", occurred_at=T0)
    report = await eng.ingest(s1, None, user_id="dana", session_id="s1", occurred_at=T0 + timedelta(minutes=1))
    assert report["new_turns"] == 2
    await eng.end_session(user_id="dana", session_id="s1")

    s2 = _conv(("Big news: I moved to Seattle last week for a job at Amazon.", "Congratulations!"))
    report = await eng.ingest(s2, None, user_id="dana", session_id="s2", occurred_at=T0 + timedelta(days=20))
    ops = {f["content"]: f["op"] for f in report["facts"]}
    assert any(op == "update" for op in ops.values()), ops
    assert report["retired_facts"], "the Austin fact must be retired"
    await eng.end_session(user_id="dana", session_id="s2")

    # A different user with overlapping facts must never surface.
    await eng.ingest(_conv(("I live in Austin too and I love BBQ.", "Nice")), None, user_id="eve", session_id="e1", occurred_at=T0)

    active = [m["content"] for m in await eng._storage.get_memories(user_id="dana")]
    assert not any("Austin" in c for c in active)
    assert any("Seattle" in c for c in active)

    res = await eng.recall("Where does my brother live?", user_id="dana", session_id="s3")
    assert any("Sam lives in Denver" in f["content"] for f in res["facts"][:3])

    ctx = await eng.get_context("Can you suggest a restaurant near my place given my diet?", user_id="dana", session_id="s3")
    assert "vegetarian" in ctx
    assert "Seattle" in ctx
    assert "BBQ" not in ctx and "Austin too" not in ctx

    eps = eng.list_episodes(user_id="dana")
    assert len(eps) == 2 and all(e["status"] == "closed" and e["key_points"] for e in eps)
    edges = {(e["subject"], e["predicate"], e["object"]) for e in eng.graph(user_id="dana")}
    assert ("Sam", "lives_in", "Denver") in edges
    assert not any(o == "Austin" for _, p, o in edges if p == "lives_in")


@pytest.mark.asyncio
async def test_episode_timed_extraction_and_session_summary(engine_factory):
    eng = engine_factory()
    eng.config.memory.extraction.timing = "episode"
    eng._ensure_components()
    eng._pipeline.timing = "episode"
    msgs = _conv(("I'm training for the Berlin marathon in September.", "Exciting!"))
    report = await eng.ingest(msgs, None, user_id="u", session_id="s", occurred_at=T0)
    assert report["facts"] == []  # deferred until the episode closes
    closed = await eng.end_session(user_id="u", session_id="s")
    assert closed["facts"] and closed["closed_episodes"]
    summary = await eng._summary_store.get_summary("s")
    assert "marathon" in summary.lower()


@pytest.mark.asyncio
async def test_add_memory_infer_handles_conflicts(engine_factory):
    eng = engine_factory()
    first = await eng.add_memory("User works at Google", user_id="z", infer=True)
    second = await eng.add_memory("User works at Stripe", user_id="z", infer=True)
    assert first != second
    chain = [m["content"] for m in eng.history(first)]
    assert chain == ["User works at Google", "User works at Stripe"]
    assert await eng.forget(second)
    assert [m["content"] for m in await eng.search_memories("work", user_id="z")] == []
