"""MemoryEngine.expand: summaries and facts stay reversible to their source turns, in scope."""

import pytest


@pytest.mark.asyncio
async def test_expand_episode_and_fact_within_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORABLE_LLM__DEFAULT_MODEL", "none")
    monkeypatch.setenv("MEMORABLE_LLM__EMBEDDING_MODEL", "hash")
    monkeypatch.setenv("MEMORABLE_LLM__TURN_EMBEDDING_MODEL", "none")
    from memorable_ai import MemoryEngine
    from memorable_ai.core.models import Turn

    eng = MemoryEngine(database=f"sqlite:///{tmp_path / 'x.db'}", write_mode="sync")
    msgs = [
        {"role": "user", "content": "I adopted a beagle puppy named Biscuit last spring."},
        {"role": "assistant", "content": "Beagles are lovely; Biscuit is a great name."},
    ]
    await eng.ingest(msgs, None, user_id="alice", session_id="s1")
    await eng.end_session(user_id="alice", session_id="s1")

    eps = eng.list_episodes(user_id="alice")
    assert eps, "ingest should close an episode"
    out = eng.expand("episode", eps[0]["id"], user_id="alice")
    assert [t["content"] for t in out["turns"]] == [m["content"] for m in msgs]
    assert eng.expand("episode", eps[0]["id"], user_id="mallory") == {}, "another user's episode leaked"

    s = eng._storage.get_session()
    try:
        tid = s.query(Turn.id).filter(Turn.user_id == "alice").order_by(Turn.seq).first()[0]
    finally:
        s.close()
    fid, _ = eng._storage.insert_memory({"content": "User has a beagle named Biscuit", "memory_type": "fact",
                                          "user_id": "alice", "source_turn_ids": [tid]})
    fact = eng.expand("fact", fid, user_id="alice")
    assert fact["fact"]["content"] == "User has a beagle named Biscuit"
    assert [t["id"] for t in fact["turns"]] == [tid]
    assert eng.expand("fact", fid, user_id="mallory") == {}
    with pytest.raises(ValueError):
        eng.expand("session", 1, user_id="alice")
    eng.close()
