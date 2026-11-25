"""Query-embedding LRU: agent tool loops repeat the same query on every LLM call."""

import numpy as np

from memorable_ai.core.embeddings import Embedder


def test_query_cache_hits_and_returns_copies():
    emb = Embedder("hash")
    emb.query_cache_size = 2
    a = emb.encode_query("where does my sister live?")
    a[:] = 0  # callers may mutate; the cache must not change
    b = emb.encode_query("where does my sister live?")
    assert emb.query_cache_hits == 1 and np.linalg.norm(b) > 0
    emb.encode_query("q2")
    emb.encode_query("q3")  # evicts the oldest entry
    assert "where does my sister live?" not in emb._query_cache
