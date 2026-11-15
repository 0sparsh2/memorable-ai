"""Vector-space safety: backends of one model share a space; a model change re-embeds."""

import pytest

from memorable_ai.core.embeddings import space_of
from memorable_ai.core.models import Turn


def test_space_of_groups_backends_of_one_model():
    m = "sentence-transformers/all-MiniLM-L6-v2"
    assert space_of(m) == space_of(f"onnx:{m}") == space_of(f"onnx-int8:{m}") == space_of(f"torch:{m}") == m
    assert space_of("static:minishlab/potion-base-32M") == "static:minishlab/potion-base-32M"
    assert space_of("hash") != space_of("hashing")


@pytest.mark.asyncio
async def test_changing_turn_model_reembeds_turns_and_tags_spaces(engine_factory, monkeypatch, tmp_path):
    db = f"sqlite:///{tmp_path / 'spaces.db'}"
    monkeypatch.setenv("MEMORABLE_LLM__EMBEDDING_MODEL", "hash")  # offline and deterministic in every mode
    monkeypatch.delenv("MEMORABLE_LLM__TURN_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("MEMORABLE_LLM__SEGMENT_EMBEDDING_MODEL", raising=False)
    eng = engine_factory(database=db)
    await eng.ingest([{"role": "user", "content": "I adopted a beagle named Biscuit"}], "Cute!",
                     user_id="u", session_id="s")
    storage = eng._storage
    assert storage.embedding_space("turn") == "hash" == storage.embedding_space("fact")
    eng.close()

    calls = []
    real = storage.__class__.reembed

    def spy(self, kind, encode, **kw):
        calls.append(kind)
        return real(self, kind, encode, **kw)

    monkeypatch.setattr(storage.__class__, "reembed", spy)
    monkeypatch.setenv("MEMORABLE_LLM__TURN_EMBEDDING_MODEL", "hashing")  # a different space
    eng2 = engine_factory(database=db)
    eng2._ensure_components()
    assert calls == ["turn"]  # facts/episodes kept the main model's space: untouched
    assert eng2._storage.embedding_space("turn") == "hashing"
    session = eng2._storage.get_session()
    try:
        assert all(t.embedding for t in session.query(Turn).all())
    finally:
        session.close()
    res = await eng2.recall("Tell me about my beagle", user_id="u", session_id="other")  # hash embedder: lexical
    assert any("Biscuit" in e["content"] for e in res["excerpts"])


@pytest.mark.asyncio
async def test_segmentation_uses_its_own_embedder(tmp_path):
    """Topic boundaries come from the segmentation embedder, not the turn-index model."""
    import numpy as np

    from memorable_ai.core.episodic import EpisodicMemory
    from memorable_ai.core.scope import Scope
    from memorable_ai.core.storage import Storage

    class Flat:  # turn-index model: every text looks the same (distances ~0, like leaf-ir's tight cone)
        def encode(self, texts, **_):
            return np.tile(np.eye(8, dtype=np.float32)[0], (len(texts), 1))

    class Topical:  # segmentation model: "bike" texts point elsewhere
        def encode(self, texts, **_):
            return np.stack([np.eye(8, dtype=np.float32)[1 if "bike" in t else 0] for t in texts])

    async def episodes(seg):
        st = Storage(f"sqlite:///{tmp_path / ('s%d.db' % id(seg))}")
        ep = EpisodicMemory(st, embedder=Flat(), turn_embedder=Flat(), segment_embedder=seg)
        scope = Scope(user_id="u", session_id="s")
        msgs = []
        for i in range(6):
            msgs += [{"role": "user", "content": f"Planning my Kyoto trip day {i} with temples"},
                     {"role": "assistant", "content": f"Sure, day {i} temples and gardens plan"}]
        msgs += [{"role": "user", "content": "Totally different: my bike chain keeps slipping off"}]
        await ep.ingest(scope, msgs, None)
        return [e["boundary_reason"] for e in ep.list_episodes(scope)]

    assert "topic_shift" not in await episodes(None)        # segment = turn model: no boundary
    assert "topic_shift" in await episodes(Topical())       # dedicated segmentation model: boundary found
