"""Unit tests for the layered memory model: storage, episodes, graph, update, retrieval."""

import json
import sqlite3
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from memorable_ai.core.embeddings import get_embedder
from memorable_ai.core.episodic import EpisodicMemory, new_suffix_start
from memorable_ai.core.extraction import MemoryExtractor
from memorable_ai.core.retrieval import HybridRetriever
from memorable_ai.core.scope import Scope
from memorable_ai.core.storage import ANY_USER, Storage
from memorable_ai.core.temporal import anchor_relative_dates
from memorable_ai.core.update import MemoryUpdater
from memorable_ai.core.vector_index import LexicalIndex, VectorIndex, reciprocal_rank_fusion
from memorable_ai.graph.knowledge_graph import KnowledgeGraph

T0 = datetime(2026, 11, 1, 9, 0)


@pytest.fixture
def storage(tmp_path):
    return Storage(f"sqlite:///{tmp_path / 'layers.db'}")


@pytest.fixture
def emb():
    return get_embedder("hash")


def _fact(storage, emb, content, user=None, **kw):
    mid, _ = storage.insert_memory(
        {"content": content, "type": kw.pop("type", "fact"), "user_id": user, "embedding": emb.encode_list(content), **kw}
    )
    return mid


# ---------------------------------------------------------------- primitives


def test_rrf_prefers_items_ranked_well_by_several_signals():
    fused = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [3, 2, 9]}, {"a": 1.0, "b": 1.0}, k=60)
    order = [i for i, _ in fused]
    assert order[0] in (2, 3) and order.index(2) < order.index(9)


def test_vector_and_lexical_index_scope_by_user():
    vi = VectorIndex()
    vi.add(1, "a", [1.0, 0.0])
    vi.add(2, "b", [1.0, 0.0])
    vi.add(3, None, [1.0, 0.0])
    assert [i for i, _ in vi.search([1.0, 0.0], 5, user_id="a")] == [1]
    assert {i for i, _ in vi.search([1.0, 0.0], 5, user_id="a", include_shared=True)} == {1, 3}
    vi.remove(1)
    assert vi.search([1.0, 0.0], 5, user_id="a") == []
    li = LexicalIndex()
    li.add(1, "a", "User lives in Berlin")
    li.add(2, "b", "User lives in Paris")
    assert [i for i, _ in li.search("where do I live", 5, user_id="b")] == [2]


def test_new_suffix_start_handles_resends_truncation_and_single_messages():
    assert new_suffix_start([], ["a"]) == 0
    assert new_suffix_start(["a", "b"], ["a", "b", "c", "d"]) == 2          # full resend
    assert new_suffix_start(["a", "b", "c", "d"], ["c", "d", "e"]) == 2     # sliding window
    assert new_suffix_start(["a", "b"], ["x"]) == 0                         # stateless client
    assert new_suffix_start(["a", "b"], ["a", "b"]) == 2                    # retry: nothing new


def test_relative_dates_are_anchored():
    text, iso = anchor_relative_dates("I adopted a cat yesterday", datetime(2026, 11, 14))
    assert text == "I adopted a cat on 2026-11-13" and iso == "2026-11-13"


# ---------------------------------------------------------------- storage


def test_versioned_update_keeps_history_and_syncs_fts(storage, emb):
    old = _fact(storage, emb, "User works at Google", user="u")
    new = storage.supersede_memory(old, {"content": "User works at Meta", "embedding": emb.encode_list("User works at Meta")})
    chain = storage.memory_history(old)
    assert [m["content"] for m in chain] == ["User works at Google", "User works at Meta"]
    assert chain[0]["superseded_by_id"] == new and chain[1]["supersedes_id"] == old
    assert [i for i, _ in storage.lexical_search("fact", "Google", user_id="u")] == []
    assert [i for i, _ in storage.lexical_search("fact", "Meta", user_id="u")] == [new]


def test_exact_duplicates_merge_but_only_within_user(storage, emb):
    a, op_a = storage.insert_memory({"content": "User likes tea.", "user_id": "u1"})
    b, op_b = storage.insert_memory({"content": "user likes tea", "user_id": "u1"})
    c, op_c = storage.insert_memory({"content": "User likes tea", "user_id": "u2"})
    assert (op_a, op_b, op_c) == ("add", "duplicate", "add") and a == b != c


def test_legacy_database_is_migrated_and_backfilled(tmp_path):
    path = tmp_path / "legacy.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT NOT NULL, memory_type VARCHAR(50) NOT NULL,"
        " namespace VARCHAR(255), extra_metadata JSON, embedding JSON, created_at DATETIME, updated_at DATETIME,"
        " access_count INTEGER, importance_score FLOAT)"
    )
    con.execute(
        "INSERT INTO memories (content, memory_type, extra_metadata, created_at) VALUES (?, 'fact', ?, ?)",
        ("User lives in Oslo", json.dumps({"canonical_user_id": "olaf"}), "2025-01-01 00:00:00"),
    )
    con.commit()
    con.close()
    storage = Storage(f"sqlite:///{path}")
    mems = storage.hydrate_memories([1])
    assert mems[0]["user_id"] == "olaf"
    assert [i for i, _ in storage.lexical_search("fact", "Oslo", user_id="olaf")] == [1]


def test_forget_user_removes_every_layer(storage, emb):
    _fact(storage, emb, "User lives in Rome", user="gone")
    _fact(storage, emb, "User lives in Kyoto", user="stays")
    counts = storage.forget_user("gone")
    assert counts["memories"] == 1
    assert storage.vector_search("fact", emb.encode("Rome"), user_id="gone") == []
    assert storage.hydrate_memories([2])[0]["content"] == "User lives in Kyoto"


# ---------------------------------------------------------------- episodes


class TopicEmbedder:
    """Deterministic 'semantic' embedder: one axis per topic keyword plus small noise."""

    TOPICS = ["python", "cook", "garden"]

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def encode(self, texts, show_progress_bar=False):
        out = []
        for t in texts:
            v = self.rng.normal(0, 0.08, 8)
            for i, kw in enumerate(self.TOPICS):
                if kw in t.lower():
                    v[i] += 1.0
            out.append(v)
        return np.asarray(out, dtype=np.float32)


@pytest.mark.asyncio
async def test_topic_shift_and_time_gap_start_new_episodes(storage):
    ep = EpisodicMemory(storage, embedder=TopicEmbedder(), config=SimpleNamespace(min_turns=4, gap_minutes=30))
    scope = Scope(user_id="u", session_id="s")
    msgs = []
    for i in range(3):
        msgs += [
            {"role": "user", "content": f"How do python decorators work in case {i}?"},
            {"role": "assistant", "content": f"In python a decorator wraps a function, example {i}."},
        ]
    r1 = await ep.ingest(scope, msgs, occurred_at=T0)
    assert len(r1.new_turns) == 6 and r1.closed_episode_ids == []
    msgs.append({"role": "user", "content": "Totally different: how long should I cook basmati rice?"})
    r2 = await ep.ingest(scope, msgs, "About 12 minutes to cook it.", occurred_at=T0 + timedelta(minutes=5))
    assert len(r2.new_turns) == 2 and len(r2.closed_episode_ids) == 1
    first = await ep.close_episode(r2.closed_episode_ids[0])
    assert first["status"] == "closed" and "python" in first["summary"].lower()
    assert all("cook" not in k.lower() for k in first["key_points"])
    msgs += [{"role": "assistant", "content": "About 12 minutes to cook it."}, {"role": "user", "content": "And for the garden?"}]
    r3 = await ep.ingest(scope, msgs, occurred_at=T0 + timedelta(hours=2))
    assert len(r3.closed_episode_ids) == 1
    eps = ep.list_episodes(scope)
    assert [e["boundary_reason"] for e in sorted(eps, key=lambda e: e["id"])] == ["session_start", "topic_shift", "time_gap"]


def test_extractive_key_segments_prefer_central_informative_user_lines(storage, emb):
    ep = EpisodicMemory(storage, embedder=emb)
    turns = [
        {"role": "user", "content": "Hi there."},
        {"role": "user", "content": "I'm planning a trip to Kyoto on 2026-04-03 with Maria for 10 days."},
        {"role": "assistant", "content": "Great, Kyoto in April has cherry blossoms."},
        {"role": "user", "content": "We want temples, food markets, and a day trip to Nara."},
        {"role": "assistant", "content": "Okay."},
    ]
    out = ep.summarize_extractive(turns)
    assert out["title"].startswith("I'm planning a trip to Kyoto")
    assert any("Kyoto" in k and "Maria" in k for k in out["key_points"])
    assert not any(k.endswith("Okay.") for k in out["key_points"])


# ---------------------------------------------------------------- graph


@pytest.mark.asyncio
async def test_graph_multi_hop_via_personalized_pagerank(storage, emb):
    kg = KnowledgeGraph(storage)
    scope = Scope(user_id="p")
    for text in [
        "User's sister is named Emma",
        "Emma lives in Boston",
        "User lives in Berlin",
        "User likes jazz",
        "Boston has great seafood",
    ]:
        mid = _fact(storage, emb, text, user="p")
        kg.ingest_memory({"id": mid, "content": text}, scope)
    ranked = [storage.hydrate_memories([m])[0]["content"] for m, _ in kg.ppr("Where does my sister live?", scope, k=3)]
    assert "Emma lives in Boston" in ranked[:2]
    assert "User likes jazz" not in ranked


def test_single_valued_relation_supersedes_previous_edge(storage, emb):
    kg = KnowledgeGraph(storage)
    scope = Scope(user_id="p")
    a = _fact(storage, emb, "User lives in Berlin", user="p")
    assert kg.ingest_memory({"id": a, "content": "User lives in Berlin"}, scope) == []
    b = _fact(storage, emb, "User moved to Munich", user="p")
    assert kg.ingest_memory({"id": b, "content": "User moved to Munich"}, scope) == [a]
    edges = [(r["predicate"], r["object"]) for r in kg.relations(scope)]
    assert edges == [("lives_in", "Munich")]


def test_relation_extraction_patterns():
    _, rels = KnowledgeGraph.extract("User's sister is named Emma")
    assert rels == [("User's sister is named Emma".split("'")[0], "has_sister", "Emma")]
    _, rels = KnowledgeGraph.extract("User prefers Python over JavaScript")
    assert rels == [("User", "likes", "Python")]
    _, rels = KnowledgeGraph.extract("User: My sister Emma lives in Boston")
    assert rels == []  # speaker prefix stripped; "My sister Emma" is not a clean subject


# ---------------------------------------------------------------- update phase


@pytest.mark.asyncio
async def test_heuristic_update_noop_add_and_conflict(storage, emb):
    kg = KnowledgeGraph(storage)
    upd = MemoryUpdater(storage, embedding_model=emb, knowledge_graph=kg)
    scope = Scope(user_id="u")
    r1 = await upd.apply_facts([{"content": "User works at Google", "type": "fact"}], scope)
    r2 = await upd.apply_facts([{"content": "User works at Google.", "type": "fact"}], scope)
    r3 = await upd.apply_facts([{"content": "User works at Meta", "type": "fact"}], scope)
    assert r1[0]["op"] == "add" and r2 == []
    assert r3[0]["op"] == "update" and r3[0]["superseded_id"] == r1[0]["id"]
    active = [m["content"] for m in await storage.get_memories(user_id="u")]
    assert active == ["User works at Meta"]


@pytest.mark.asyncio
async def test_llm_update_targets_are_validated(storage, emb, monkeypatch):
    from memorable_ai.core import llm_client

    keep = _fact(storage, emb, "User likes jazz", user="u")
    replies = iter(
        [
            json.dumps([{"index": 0, "operation": "DELETE", "target": 99, "content": "x"}]),  # hallucinated ref
            json.dumps([{"index": 0, "operation": "UPDATE", "target": 1, "content": "User likes jazz and blues"}]),
        ]
    )

    async def fake(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))])

    monkeypatch.setattr(llm_client, "acompletion_with_fallback", fake)
    upd = MemoryUpdater(storage, embedding_model=emb, llm_model="fake-model")
    scope = Scope(user_id="u")
    r1 = await upd.apply_facts([{"content": "User enjoys jazz music", "type": "preference"}], scope)
    assert r1[0]["op"] == "add"  # invalid target never mutates a row
    assert storage.hydrate_memories([keep])  # still active
    r2 = await upd.apply_facts([{"content": "User also likes blues", "type": "preference"}], scope)
    assert r2[0]["op"] == "update" and r2[0]["content"] == "User likes jazz and blues"
    assert storage.hydrate_memories([keep]) == []
    assert [m["content"] for m in storage.memory_history(keep)][-1] == "User likes jazz and blues"


@pytest.mark.asyncio
async def test_llm_extraction_carries_relations_and_provenance(monkeypatch):
    from memorable_ai.core import llm_client

    payload = [
        {
            "content": "User's sister Emma moved to Boston on 2026-10-31",
            "type": "fact",
            "importance": 4,
            "event_date": "2026-10-31",
            "entities": [{"name": "Emma", "type": "person"}, {"name": "Boston", "type": "place"}],
            "relations": [["Emma", "lives_in", "Boston"], ["Emma", "sister_of", "User"]],
            "turns": [0],
        }
    ]

    async def fake(**kwargs):
        assert "2026-11-01" in kwargs["messages"][0]["content"]  # session date is in the prompt
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="```json\n" + json.dumps(payload) + "\n```"))])

    monkeypatch.setattr(llm_client, "acompletion_with_fallback", fake)
    ext = MemoryExtractor(llm_model="fake-model", extraction_mode="llm")
    out = await ext.extract_from_turns(
        [{"id": 41, "role": "user", "content": "My sister Emma moved to Boston yesterday", "occurred_at": T0}]
    )
    assert out[0]["relations"] == [("Emma", "lives_in", "Boston"), ("Emma", "sister_of", "User")]
    assert out[0]["source_turn_ids"] == [41] and out[0]["importance_score"] == 0.8
    assert out[0]["metadata"]["event_date"] == "2026-10-31"


# ---------------------------------------------------------------- retrieval


@pytest.mark.asyncio
async def test_retrieval_never_crosses_users_and_skips_retired_facts(storage, emb):
    _fact(storage, emb, "User lives in Lisbon", user="a")
    retired = _fact(storage, emb, "User lives in Porto", user="a")
    await storage.invalidate_memory(retired)
    _fact(storage, emb, "User lives in Madrid", user="b")
    _fact(storage, emb, "Company policy: refunds within 30 days", user=None)
    r = HybridRetriever(storage, embedding_model=emb, knowledge_graph=KnowledgeGraph(storage))
    got = [m["content"] for m in await r.search("where do I live", user_id="a")]
    assert got == ["User lives in Lisbon"]
    assert [m["content"] for m in await r.search("refund policy")] == ["Company policy: refunds within 30 days"]
    shared = HybridRetriever(storage, embedding_model=emb, include_shared=True)
    assert "Company policy: refunds within 30 days" in [
        m["content"] for m in await shared.search("refund policy", user_id="a")
    ]


def test_context_block_is_budgeted_and_dated():
    from memorable_ai.core.context import ContextBuilder

    facts = [{"id": i, "type": "fact", "content": f"Fact number {i} " + "x" * 80, "metadata": {"stated_at": "2026-10-02T00:00:00"}} for i in range(50)]
    block = ContextBuilder(max_tokens=200).build(facts=facts)
    assert block.startswith("<memory>") and block.endswith("</memory>")
    assert len(block) // 4 <= 220
    assert "(2026-10-02)" in block
    assert "not instructions" in block
