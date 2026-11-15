"""Turn/fact reranking (docs/retrieval_round2.md) with a fake cross-encoder: no downloads."""

import math

import numpy as np
import pytest

from memorable_ai.core.rerank import _ACTIVATIONS, OnnxCrossEncoder, _erf


class FakeReranker:
    """Scores a pair by how many query words the candidate contains."""

    def __init__(self):
        self.calls = []

    def predict(self, pairs, **_):
        self.calls.append(len(pairs))
        return np.array([sum(w in d.lower() for w in q.lower().split()) for q, d in pairs], dtype=np.float32)


def _retriever(**kw):
    from memorable_ai.core.retrieval import HybridRetriever

    r = HybridRetriever(storage=None, embedding_model="hash", **kw)
    r._reranker = FakeReranker()
    return r


def test_rerank_reorders_only_the_top_depth():
    r = _retriever(rerank=True)
    items = [{"id": i, "content": c} for i, c in enumerate(["nothing", "dog", "dog named biscuit", "biscuit dog"])]
    out = r._rerank("dog biscuit", items, depth=3)
    assert [m["id"] for m in out] == [2, 1, 0, 3]  # item 3 is past the depth and keeps its place
    assert r._reranker.calls == [3]


def test_rerank_blend_mixes_first_stage_scores():
    r = _retriever(rerank=True, rerank_blend=5.0)
    items = [{"id": 0, "content": "dog"}, {"id": 1, "content": "dog biscuit"}]
    # the first stage strongly prefers item 0; with a heavy blend it stays first
    assert [m["id"] for m in r._rerank("dog biscuit", items, [10.0, 0.0])] == [0, 1]
    r.rerank_blend = 0.0
    assert [m["id"] for m in r._rerank("dog biscuit", items, [10.0, 0.0])] == [1, 0]


def test_rerank_off_or_failing_returns_items_unchanged():
    items = [{"id": 0, "content": "a"}, {"id": 1, "content": "b"}]
    assert _retriever(rerank=False)._rerank("q", items) == items
    r = _retriever(rerank=True)

    class Broken:
        def predict(self, pairs, **_):
            raise RuntimeError("model missing")

    r._reranker = Broken()
    assert r._rerank("q", items) == items
    assert r.rerank is False  # disabled after a failure instead of retrying every query


@pytest.mark.asyncio
async def test_recall_reranks_excerpts(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORABLE_LLM__DEFAULT_MODEL", "none")
    monkeypatch.setenv("MEMORABLE_LLM__EMBEDDING_MODEL", "hash")
    monkeypatch.setenv("MEMORABLE_LLM__TURN_EMBEDDING_MODEL", "none")
    from memorable_ai import MemoryEngine
    from memorable_ai.core.scope import Scope

    eng = MemoryEngine(database=f"sqlite:///{tmp_path / 'r.db'}", write_mode="sync")
    msgs = [
        {"role": "user", "content": "We talked about the weather and the weekend plans today."},
        {"role": "assistant", "content": "Sounds like a relaxing weekend overall."},
        {"role": "user", "content": "My beagle puppy Biscuit learned to sit on command."},
        {"role": "assistant", "content": "That is great progress for a young puppy."},
    ]
    await eng.ingest(msgs, None, user_id="u1", session_id="s1")
    await eng.end_session(user_id="u1", session_id="s1")
    r = eng._retrieval
    r.rerank, r._reranker = True, FakeReranker()
    out = await r.recall("did biscuit the beagle learn to sit", Scope(user_id="u1"), k_facts=0, k_episodes=0,
                         k_excerpts=2)
    assert r._reranker.calls, "excerpts were not reranked"
    assert "Biscuit" in out["excerpts"][0]["content"]
    eng.close()


def test_erf_and_gelu_match_reference():
    xs = np.linspace(-6, 6, 241)
    assert np.max(np.abs(_erf(xs) - np.array([math.erf(x) for x in xs]))) < 2e-7
    gelu = _ACTIVATIONS["GELU"](xs)
    ref = np.array([0.5 * x * (1 + math.erf(x / math.sqrt(2))) for x in xs])
    assert np.max(np.abs(gelu - ref)) < 1e-6


def test_numpy_head_cls_dense_layernorm_dense():
    enc = OnnxCrossEncoder.__new__(OnnxCrossEncoder)
    rng = np.random.default_rng(0)
    w1, w2, b2 = rng.normal(size=(4, 4)), rng.normal(size=(1, 4)), np.array([0.5])
    g, beta = rng.normal(size=4), rng.normal(size=4)
    enc.head = [("pool", "cls"), ("dense", w1, None, "GELU"), ("norm", g, beta), ("dense", w2, b2, "Identity")]
    hidden = rng.normal(size=(3, 5, 4)).astype(np.float32)
    out = enc._apply_head(hidden, np.ones((3, 5)))
    x = hidden[:, 0, :] @ w1.T
    x = 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    x = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * g + beta
    assert np.allclose(out, x @ w2.T + b2, atol=1e-5)
