"""Score fusion for the turn and episode layers (docs/retrieval_round2.md)."""

import numpy as np
import pytest

from memorable_ai.core.vector_index import VectorIndex, reciprocal_rank_fusion, score_fusion


def test_score_fusion_keeps_score_margins_that_rrf_discards():
    # Dense is nearly tied between 1 and 2; BM25 strongly prefers 2. RRF only sees ranks,
    # so dense rank 1 vs 2 counts as much as BM25's large gap.
    dense = {1: 0.80, 2: 0.79, 3: 0.20, 4: 0.10}
    lexical = {2: 12.0, 1: 1.0, 3: 0.5}
    fused = [i for i, _ in score_fusion({"semantic": dense, "lexical": lexical}, {"semantic": 1.0, "lexical": 0.5})]
    assert fused[0] == 2
    rrf = [i for i, _ in reciprocal_rank_fusion({"semantic": list(dense), "lexical": list(lexical)},
                                                {"semantic": 1.0, "lexical": 0.5})]
    assert rrf[0] == 1


def test_score_fusion_pool_is_union_and_missing_scores_count_as_zero():
    out = dict(score_fusion({"semantic": {1: 0.9, 2: 0.1}, "lexical": {3: 5.0}}))
    assert set(out) == {1, 2, 3}
    # 3 has no dense score (0 < 0.1), but the only lexical match
    assert out[3] > out[2]


def test_score_fusion_edge_cases():
    assert score_fusion({}) == []
    assert score_fusion({"semantic": {}, "lexical": {}}) == []
    # constant signal carries no information; ordering falls back to the other signal
    out = [i for i, _ in score_fusion({"semantic": {1: 0.5, 2: 0.5}, "lexical": {2: 3.0, 1: 1.0}})]
    assert out == [2, 1]
    # zero weight disables a signal
    out = [i for i, _ in score_fusion({"semantic": {1: 0.9, 2: 0.1}, "lexical": {2: 9.0}}, {"lexical": 0.0})]
    assert out[0] == 1


def test_vector_index_scores_for_matches_search():
    rng = np.random.default_rng(0)
    idx = VectorIndex()
    vecs = rng.normal(size=(20, 8)).astype(np.float32)
    for i, v in enumerate(vecs):
        idx.add(100 + i, "u", v)
    q = rng.normal(size=8).astype(np.float32)
    full = dict(idx.search(q, k=20, user_id="u"))
    some = idx.scores_for(q, [100, 105, 119, 999])
    assert set(some) == {100, 105, 119}
    for i, s in some.items():
        assert s == pytest.approx(full[i], abs=1e-6)
    idx.remove(105)
    assert 105 not in idx.scores_for(q, [105, 100])
    assert idx.scores_for(np.zeros(8), [100]) == {}
    assert idx.scores_for(np.ones(3), [100]) == {}


@pytest.mark.asyncio
async def test_rank_layer_score_fusion_scores_lexical_only_candidates(tmp_path, monkeypatch):
    """Lexical-only candidates get their exact dense score; a non-semantic embedder keeps rank fusion."""
    monkeypatch.setenv("MEMORABLE_LLM__DEFAULT_MODEL", "none")
    monkeypatch.setenv("MEMORABLE_LLM__EMBEDDING_MODEL", "hash")
    monkeypatch.setenv("MEMORABLE_LLM__TURN_EMBEDDING_MODEL", "none")
    from memorable_ai import MemoryEngine
    from memorable_ai.core.scope import Scope

    eng = MemoryEngine(database=f"sqlite:///{tmp_path / 'f.db'}", write_mode="sync")
    msgs = [
        {"role": "user", "content": "I adopted a beagle puppy named Biscuit last spring."},
        {"role": "assistant", "content": "Beagles are lovely dogs."},
        {"role": "user", "content": "My sister lives in Lisbon and teaches piano."},
        {"role": "assistant", "content": "Lisbon is a beautiful city."},
    ]
    await eng.ingest(msgs, None, user_id="u1", session_id="s1")
    await eng.end_session(user_id="u1", session_id="s1")
    r = eng._retrieval
    scope = Scope(user_id="u1")
    qvec = r._encode("what is the name of my dog Biscuit", kind="turn")
    calls = []
    real = eng._storage.vector_scores
    monkeypatch.setattr(eng._storage, "vector_scores", lambda *a, **k: calls.append(a) or real(*a, **k))
    q = "what is the name of my dog Biscuit"
    for fusion in ("score", "rrf"):  # hash embedder: both take the BM25-weighted rank path
        r.fusion = fusion
        ids = r._rank_layer("turn", q, qvec, scope, k=3)
        assert "Biscuit" in eng._storage.hydrate_turns(ids[:1], user_id="u1")[0]["content"], fusion
    assert not calls

    # Pretend the embedder is semantic: score fusion runs and fills dense scores for lexical-only hits.
    monkeypatch.setattr(type(r.turn_embedder), "is_semantic", property(lambda self: True))
    monkeypatch.setattr(eng._storage, "vector_search", lambda *a, **k: [])  # no dense hits at all
    r.fusion = "score"
    ids = r._rank_layer("turn", q, qvec, scope, k=3)
    assert calls and "Biscuit" in eng._storage.hydrate_turns(ids[:1], user_id="u1")[0]["content"]
    eng.close()
