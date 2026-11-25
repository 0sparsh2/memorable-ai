"""Unified edges table, GraphEngine (CSR PPR), provenance edges, tiers and agent_id (G0)."""

import random
import sqlite3
from datetime import datetime, timedelta

import networkx as nx
import pytest

from memorable_ai.core.episodic import EpisodicMemory
from memorable_ai.core.models import Edge, Memory
from memorable_ai.core.scope import Scope
from memorable_ai.core.storage import Storage
from memorable_ai.graph.engine import GraphEngine
from memorable_ai.graph.knowledge_graph import KnowledgeGraph


@pytest.fixture
def storage(tmp_path):
    return Storage(f"sqlite:///{tmp_path / 'g.db'}")


def _edges(storage, pairs, user_id="u", kind="relates"):
    return storage.add_edges(
        [
            {"src_kind": a[0], "src_id": a[1], "kind": kind, "dst_kind": b[0], "dst_id": b[1], "user_id": user_id}
            for a, b in pairs
        ]
    )


def _nx_ppr(pairs, seeds, alpha=0.85, drop=()):
    g = nx.MultiGraph()
    for a, b in pairs:
        if a in drop or b in drop:
            continue
        g.add_edge(a, b)
    simple = nx.Graph()
    for a, b, _ in g.edges(keys=True):  # multi-edges add weight, like the CSR sum
        w = simple[a][b]["weight"] + 1 if simple.has_edge(a, b) else 1
        simple.add_edge(a, b, weight=w)
    pers = {k: v for k, v in seeds.items() if k in simple}
    return nx.pagerank(simple, alpha=alpha, personalization=pers, weight="weight", tol=1e-10, max_iter=500)


def _random_pairs(n_nodes=60, n_edges=180, seed=3):
    rng = random.Random(seed)
    nodes = [("entity", i) for i in range(1, n_nodes // 2)] + [("fact", i) for i in range(1, n_nodes // 2)]
    pairs = []
    while len(pairs) < n_edges:
        a, b = rng.sample(nodes, 2)
        pairs.append((a, b))
    return pairs


def test_exact_ppr_matches_networkx(storage):
    pairs = _random_pairs()
    _edges(storage, pairs)
    eng = GraphEngine(storage, weights={"relates": 1.0}, tol=1e-9)
    seeds = {("entity", 1): 1.0, ("entity", 2): 0.5}
    res = eng.ppr(None, "u", seeds)
    ref = _nx_ppr(pairs, seeds)
    for node, val in ref.items():
        assert res.score(node) == pytest.approx(val, abs=2e-4)


def test_exclusion_is_exact_induced_subgraph(storage):
    pairs = _random_pairs(seed=5)
    _edges(storage, pairs)
    eng = GraphEngine(storage, weights={"relates": 1.0}, tol=1e-9)
    hub = ("entity", 3)
    seeds = {("entity", 1): 1.0}
    res = eng.ppr(None, "u", seeds, exclude=[hub])
    ref = _nx_ppr(pairs, seeds, drop={hub})
    assert res.score(hub) == 0.0
    for node, val in ref.items():
        assert res.score(node) == pytest.approx(val, abs=2e-4)


def test_local_push_agrees_with_exact(storage):
    pairs = _random_pairs(n_nodes=200, n_edges=800, seed=9)
    _edges(storage, pairs)
    exact = GraphEngine(storage, weights={"relates": 1.0}, tol=1e-9)
    push = GraphEngine(storage, weights={"relates": 1.0}, push_eps=1e-7)
    seeds = {("entity", 4): 1.0, ("entity", 7): 1.0}
    a = exact.ppr(None, "u", seeds).top("fact", 10)
    b = push.ppr(None, "u", seeds, method="push").top("fact", 10)
    assert len({i for i, _ in a} & {i for i, _ in b}) >= 8
    for (i, s), (j, t) in zip(a[:3], b[:3]):
        assert s == pytest.approx(t, rel=0.05)


def test_tiers_isolate_users_and_share_org(storage):
    # Alice and Bob each link entity 1 to their own fact; the org tier links entity 1 to fact 100.
    _edges(storage, [(("entity", 1), ("fact", 10))], user_id="alice", kind="about")
    _edges(storage, [(("entity", 1), ("fact", 20))], user_id="bob", kind="about")
    _edges(storage, [(("entity", 1), ("fact", 100))], user_id=None, kind="about")
    eng = GraphEngine(storage)
    seeds = {("entity", 1): 1.0}
    alice_only = {i for i, _ in eng.ppr(None, "alice", seeds).top("fact")}
    alice_org = {i for i, _ in eng.ppr(None, "alice", seeds, include_shared=True).top("fact")}
    org_only = {i for i, _ in eng.ppr(None, None, seeds).top("fact")}
    assert alice_only == {10}
    assert alice_org == {10, 100}
    assert org_only == {100}
    assert 20 not in alice_org  # Bob's data never enters Alice's view


def test_delta_sync_invalidation_and_epoch(storage):
    eng = GraphEngine(storage)
    seeds = {("entity", 1): 1.0}
    _edges(storage, [(("entity", 1), ("fact", 1))], kind="about")
    assert {i for i, _ in eng.ppr(None, "u", seeds).top("fact")} == {1}
    ids = _edges(storage, [(("entity", 1), ("fact", 2))], kind="about")  # appended after first query
    assert {i for i, _ in eng.ppr(None, "u", seeds).top("fact")} == {1, 2}
    storage.invalidate_edges(ids=ids)  # transaction-time retirement
    assert {i for i, _ in eng.ppr(None, "u", seeds).top("fact")} == {1}
    session = storage.get_session()
    try:
        storage.delete_node_edges(session, "fact", 1)  # hard delete -> graph_epoch bump
        session.commit()
    finally:
        session.close()
    assert eng.ppr(None, "u", seeds).top("fact") == []


def test_fact_provenance_supersession_and_invalidation(storage):
    kg = KnowledgeGraph(storage)
    scope = Scope(user_id="u", agent_id="ops-bot")
    mid, _ = storage.insert_memory(
        {"content": "User lives in Berlin", "user_id": "u", "agent_id": "ops-bot",
         "source_turn_ids": [5, 6], "source_episode_id": 2}
    )
    kg.ingest_memory({"id": mid, "content": "User lives in Berlin"}, scope)
    session = storage.get_session()
    try:
        prov = {(e.dst_kind, e.dst_id) for e in session.query(Edge).filter_by(src_kind="fact", src_id=mid,
                                                                               kind="derived_from")}
        about = session.query(Edge).filter_by(src_kind="fact", src_id=mid, kind="about").all()
        assert prov == {("turn", 5), ("turn", 6), ("episode", 2)}
        assert about and all(e.agent_id == "ops-bot" for e in about)
        assert session.get(Memory, mid).agent_id == "ops-bot"
    finally:
        session.close()
    new_id = storage.supersede_memory(mid, {"content": "User lives in Munich"})
    session = storage.get_session()
    try:
        sup = session.query(Edge).filter_by(src_kind="fact", src_id=new_id, kind="supersedes").one()
        assert (sup.dst_kind, sup.dst_id) == ("fact", mid)
        # The old fact left the walkable graph; its provenance survives as history.
        assert all(e.invalidated_at is not None
                   for e in session.query(Edge).filter_by(src_kind="fact", src_id=mid, kind="about"))
        assert session.query(Edge).filter_by(src_kind="fact", src_id=mid, kind="derived_from",
                                             invalidated_at=None).count() == 3
    finally:
        session.close()


def test_invalidated_fact_drops_out_of_ppr(storage):
    kg = KnowledgeGraph(storage)
    scope = Scope(user_id="u")
    ids = []
    for text in ("Emma lives in Paris", "Emma works at Acme"):
        mid, _ = storage.insert_memory({"content": text, "user_id": "u"})
        kg.ingest_memory({"id": mid, "content": text}, scope)
        ids.append(mid)
    assert {m for m, _ in kg.ppr("Tell me about Emma", scope)} == set(ids)
    storage._invalidate(ids[0], reason="test")
    assert {m for m, _ in kg.ppr("Tell me about Emma", scope)} == {ids[1]}


def test_cascade_finds_unsupported_dependents(storage):
    a, _ = storage.insert_memory({"content": "fact a", "user_id": "u", "source_turn_ids": [1]})
    b, _ = storage.insert_memory({"content": "fact b", "user_id": "u", "source_turn_ids": [1, 2]})
    assert storage.unsupported_dependents("turn", 1) == [("fact", a)]
    assert storage.unsupported_dependents("turn", 2) == []


@pytest.mark.asyncio
async def test_episodic_structure_edges(storage):
    ep = EpisodicMemory(storage, embedder="hash")
    scope = Scope(user_id="u", session_id="s1", agent_id="bot")
    t0 = datetime(2026, 1, 5, 9, 0)
    await ep.ingest(scope, [{"role": "user", "content": "Plan my Kyoto trip"}], "Sure.", t0)
    await ep.ingest(
        scope,
        [{"role": "user", "content": "Plan my Kyoto trip"}, {"role": "assistant", "content": "Sure."},
         {"role": "user", "content": "Different topic: fix my bike"}],
        "Okay.",
        t0 + timedelta(hours=3),  # beyond the 30-minute gap -> new episode
    )
    session = storage.get_session()
    try:
        turn_links = session.query(Edge).filter_by(src_kind="turn", kind="part_of").all()
        ep_links = session.query(Edge).filter_by(src_kind="episode", kind="part_of").all()
        nxt = session.query(Edge).filter_by(kind="next").all()
        assert len(turn_links) == 4 and len(ep_links) == 2 and len(nxt) == 1
        assert {e.agent_id for e in turn_links + ep_links + nxt} == {"bot"}
    finally:
        session.close()


def test_legacy_relations_and_mentions_are_backfilled(tmp_path):
    path = tmp_path / "legacy.db"
    Storage(f"sqlite:///{path}")  # create the v0.3 schema, then fake pre-v0.3 tables
    con = sqlite3.connect(path)
    con.executescript(
        """
        DELETE FROM edges;
        CREATE TABLE relations (id INTEGER PRIMARY KEY, namespace TEXT, user_id TEXT, subject_id INT,
          predicate TEXT, object_id INT, memory_id INT, episode_id INT, confidence FLOAT,
          valid_from TIMESTAMP, valid_to TIMESTAMP, invalidated_at TIMESTAMP, superseded_by_id INT,
          created_at TIMESTAMP);
        CREATE TABLE mentions (id INTEGER PRIMARY KEY, entity_id INT, memory_id INT, episode_id INT,
          turn_id INT, created_at TIMESTAMP);
        INSERT INTO entities (id, namespace, user_id, name, norm_name, entity_type, is_self, mention_count)
          VALUES (1, NULL, 'u', 'User', 'user', 'person', 1, 0), (2, NULL, 'u', 'Berlin', 'berlin', 'place', 0, 1);
        INSERT INTO memories (id, content, memory_type, user_id) VALUES (7, 'User lives in Berlin', 'fact', 'u');
        INSERT INTO relations (namespace, user_id, subject_id, predicate, object_id, memory_id, confidence)
          VALUES (NULL, 'u', 1, 'lives_in', 2, 7, 1.0);
        INSERT INTO mentions (entity_id, memory_id) VALUES (2, 7);
        INSERT INTO mentions (entity_id, episode_id) VALUES (2, 3);
        """
    )
    con.commit()
    con.close()
    storage = Storage(f"sqlite:///{path}")
    rels = KnowledgeGraph(storage).relations(Scope(user_id="u"))
    assert [(r["subject"], r["predicate"], r["object"], r["memory_id"]) for r in rels] == [
        ("User", "lives_in", "Berlin", 7)
    ]
    session = storage.get_session()
    try:
        kinds = sorted(e.kind for e in session.query(Edge).all())
        assert kinds == ["about", "mentions", "relates"]
    finally:
        session.close()


@pytest.mark.asyncio
async def test_agent_id_flows_through_engine_scope(engine_factory):
    engine = engine_factory()
    with engine.scope(user_id="alice", agent_id="support-bot"):
        mid = await engine.add_memory("Alice prefers email contact")
    session = engine._storage.get_session()
    try:
        assert session.get(Memory, mid).agent_id == "support-bot"
    finally:
        session.close()
    assert engine.resolve_scope(user_id="x", hints={"agent_id": "a1"}).agent_id == "a1"


@pytest.mark.asyncio
async def test_forget_user_removes_their_edges(engine_factory):
    engine = engine_factory()
    with engine.scope(user_id="carol"):
        await engine.add_memory("Carol lives in Lisbon")
    kg = engine._knowledge_graph
    scope = Scope(namespace=engine.config.memory.namespace, user_id="carol")
    assert kg.ppr("Where is Lisbon", scope)
    await engine.forget_user("carol")
    assert kg.ppr("Where is Lisbon", scope) == []
    session = engine._storage.get_session()
    try:
        assert session.query(Edge).filter_by(user_id="carol").count() == 0
    finally:
        session.close()


def test_out_of_order_invalidation_is_not_missed(storage):
    """Regression (found by property testing): sync must follow commit order, not timestamps."""
    writer = Storage(storage.connection_string)  # a second process on the same database
    ids = _edges(storage, [(("entity", 1), ("fact", i)) for i in (1, 2, 3)], kind="about")
    eng = GraphEngine(storage)
    facts = lambda e: sorted(i for i, _ in e.ppr(None, "u", {("entity", 1): 1.0}).top("fact"))
    assert facts(eng) == [1, 2, 3]
    writer.invalidate_edges(ids=[ids[0]])
    assert facts(eng) == [2, 3]
    writer.invalidate_edges(ids=[ids[1]], at=datetime(2020, 1, 1))  # stamped before the previous commit
    assert facts(eng) == [3] == facts(GraphEngine(storage))


def test_nodes_whose_edges_all_died_behave_as_absent(storage):
    """Regression (found by property testing): a dead seed must not keep teleport mass."""
    ids = _edges(storage, [(("entity", 1), ("fact", 7))], kind="about")
    eng = GraphEngine(storage)
    assert eng.ppr(None, "u", {("fact", 7): 1.0}).score(("fact", 7)) > 0
    storage.invalidate_edges(ids=ids)
    live = eng.ppr(None, "u", {("fact", 7): 1.0})
    assert live.score(("fact", 7)) == 0.0 and live.top("fact") == []


def test_concurrent_writers_and_readers_stay_consistent(storage):
    """Threads writing edges/invalidations while others query: no errors, and the
    long-lived engine converges to exactly what a fresh engine sees."""
    import threading

    eng = GraphEngine(storage, tol=1e-10)
    errors = []
    seed = {("entity", 1): 1.0}

    def writer(k):
        try:
            w = Storage(storage.connection_string)
            for i in range(25):
                ids = w.add_edges([{"src_kind": "entity", "src_id": 1 + (i % 3), "kind": "about",
                                    "dst_kind": "fact", "dst_id": 1000 * k + i, "user_id": "u"}])
                if i % 4 == 0:
                    w.invalidate_edges(ids=ids)
        except Exception as e:  # pragma: no cover - surfaced below
            errors.append(e)

    def reader():
        try:
            for _ in range(40):
                eng.ppr(None, "u", seed).top("fact", 10)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(k,)) for k in (1, 2, 3)] + [
        threading.Thread(target=reader) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    live = eng.ppr(None, "u", seed)
    fresh = GraphEngine(storage, tol=1e-10).ppr(None, "u", seed)
    assert {i for i, _ in live.top("fact", 1000)} == {i for i, _ in fresh.top("fact", 1000)}
