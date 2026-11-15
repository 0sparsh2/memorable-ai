"""
Property-based tests (Hypothesis) for the G0 memory graph.

  * differential: GraphEngine PPR == networkx PageRank on the same scoped,
    masked graph
  * isolation: a view never contains nodes known only from other users' edges
  * stateful: an engine kept alive across random writes, invalidations (including
    ones stamped *before* a later commit, as happens with concurrent writers) and
    hard deletes always answers exactly like a freshly built engine
  * ingest idempotence: re-sending a growing history yields exactly that history
"""

import asyncio
import atexit
import itertools
import os
import shutil
import tempfile
from datetime import datetime, timedelta

import networkx as nx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from memorable_ai.core.scope import Scope
from memorable_ai.core.storage import Storage
from memorable_ai.graph.engine import GraphEngine

# CI budget; for a deep run: MEMORABLE_PROP_EXAMPLES=500 pytest tests/property
N = int(os.getenv("MEMORABLE_PROP_EXAMPLES", "60"))

USERS = ["alice", "bob", None]  # None = shared org tier
NODES = [("entity", i) for i in range(1, 7)] + [("fact", i) for i in range(1, 7)]
KINDS = ["relates", "about"]

edge_st = st.tuples(st.sampled_from(USERS), st.sampled_from(KINDS), st.sampled_from(NODES), st.sampled_from(NODES)).filter(
    lambda e: e[2] != e[3]
)


# One temp root per test session; each example's database is removed as soon as the
# next one is created (thousands of examples must not accumulate on disk).
_ROOT = tempfile.mkdtemp(prefix="memorable_prop_")
atexit.register(shutil.rmtree, _ROOT, ignore_errors=True)
_LIVE: list = []
_COUNTER = itertools.count()


def _fresh_storage():
    while len(_LIVE) > 4:  # keep the last few (stateful machines hold two handles)
        old = _LIVE.pop(0)
        try:
            old.engine.dispose()
        finally:
            for suffix in ("", "-wal", "-shm"):
                path = old.connection_string.split("///", 1)[1] + suffix
                if os.path.exists(path):
                    os.remove(path)
    path = os.path.join(_ROOT, f"g{next(_COUNTER)}.db")
    storage = Storage(f"sqlite:///{path}")
    _LIVE.append(storage)
    return storage


def _add(storage, edges):
    return storage.add_edges(
        [{"src_kind": a[0], "src_id": a[1], "kind": k, "dst_kind": b[0], "dst_id": b[1], "user_id": u}
         for u, k, a, b in edges]
    )


def _nx_reference(edges, user, include_shared, seeds, exclude, alpha=0.85):
    allowed_users = {user} | ({None} if (include_shared or user is None) else set())
    g = nx.Graph()
    for u, _k, a, b in edges:
        if u not in allowed_users:
            continue
        for n in (a, b):  # present in the view (has an edge), even if all its neighbours are excluded
            if n not in exclude:
                g.add_node(n)
        if a in exclude or b in exclude:
            continue
        w = g[a][b]["weight"] + 1.0 if g.has_edge(a, b) else 1.0
        g.add_edge(a, b, weight=w)
    pers = {n: w for n, w in seeds.items() if n in g and n not in exclude}
    if not pers:
        return {}
    return nx.pagerank(g, alpha=alpha, personalization=pers, weight="weight", tol=1e-12, max_iter=1000)


@settings(max_examples=N, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    edges=st.lists(edge_st, min_size=1, max_size=25),
    user=st.sampled_from(USERS),
    include_shared=st.booleans(),
    seeds=st.dictionaries(st.sampled_from(NODES), st.floats(0.1, 2.0), min_size=1, max_size=3),
    exclude=st.sets(st.sampled_from(NODES), max_size=2),
)
def test_ppr_matches_networkx_on_scoped_masked_graph(edges, user, include_shared, seeds, exclude):
    storage = _fresh_storage()
    _add(storage, edges)
    eng = GraphEngine(storage, weights={"relates": 1.0, "about": 1.0}, tol=1e-12, max_iter=1000)
    res = eng.ppr(None, user, seeds, include_shared=include_shared, exclude=exclude)
    ref = _nx_reference(edges, user, include_shared, seeds, exclude)
    for node in NODES:
        assert res.score(node) == pytest.approx(ref.get(node, 0.0), abs=1e-6), node


@settings(max_examples=N, deadline=None)
@given(edges=st.lists(edge_st, min_size=1, max_size=30), user=st.sampled_from(["alice", "bob"]),
       include_shared=st.booleans())
def test_views_never_contain_other_users_nodes(edges, user, include_shared):
    storage = _fresh_storage()
    _add(storage, edges)
    eng = GraphEngine(storage)
    allowed = {user} | ({None} if include_shared else set())
    visible = {n for u, _k, a, b in edges if u in allowed for n in (a, b)}
    res = eng.ppr(None, user, {n: 1.0 for n in NODES}, include_shared=include_shared)
    for node in NODES:
        if node not in visible:
            assert res.score(node) == 0.0
            assert res._view is None or res._view.idx(node) is None


class GraphSyncMachine(RuleBasedStateMachine):
    """A long-lived engine must always agree with one rebuilt from SQL."""

    @initialize()
    def setup(self):
        self.storage = _fresh_storage()
        self.writer = Storage(self.storage.connection_string)  # a second "process"
        self.engine = GraphEngine(self.storage, tol=1e-12, max_iter=1000)
        self.ids = []

    @rule(edges=st.lists(edge_st, min_size=1, max_size=4), via_writer=st.booleans())
    def add_edges(self, edges, via_writer):
        self.ids += _add(self.writer if via_writer else self.storage, edges)

    @rule(data=st.data(), backdate=st.booleans())
    def invalidate(self, data, backdate):
        if not self.ids:
            return
        eid = data.draw(st.sampled_from(self.ids))
        # A concurrent writer may stamp invalidated_at *before* another commit lands.
        at = datetime(2020, 1, 1) if backdate else None
        self.writer.invalidate_edges(ids=[eid], at=at)

    @rule(node=st.sampled_from(NODES))
    def hard_delete(self, node):
        session = self.writer.get_session()
        try:
            self.writer.delete_node_edges(session, node[0], node[1])
            session.commit()
        finally:
            session.close()

    @rule(user=st.sampled_from(USERS), include_shared=st.booleans(), seed=st.sampled_from(NODES))
    def query_matches_fresh_engine(self, user, include_shared, seed):
        live = self.engine.ppr(None, user, {seed: 1.0}, include_shared=include_shared)
        fresh = GraphEngine(self.storage, tol=1e-12, max_iter=1000).ppr(
            None, user, {seed: 1.0}, include_shared=include_shared
        )
        for node in NODES:
            # float32 sums in a different node order differ by ~1e-7; anything structural is >> 1e-5
            assert live.score(node) == pytest.approx(fresh.score(node), abs=1e-5), (node, user, include_shared)

    @invariant()
    def engine_is_consistent(self):
        pass


GraphSyncMachine.TestCase.settings = settings(max_examples=max(N, 150), stateful_step_count=30, deadline=None)
TestGraphSync = GraphSyncMachine.TestCase


@settings(max_examples=N, deadline=None)
@given(msgs=st.lists(st.tuples(st.sampled_from(["user", "assistant"]), st.sampled_from(["ok", "yes", "hi", "no", "thanks"])),
                     min_size=1, max_size=12),
       cuts=st.lists(st.integers(1, 12), min_size=1, max_size=5))
def test_resending_growing_history_stores_exactly_that_history(msgs, cuts):
    from memorable_ai.core.episodic import EpisodicMemory
    from memorable_ai.core.models import Turn

    storage = _fresh_storage()
    ep = EpisodicMemory(storage, embedder="hash")
    scope = Scope(user_id="u", session_id="s")
    history = [{"role": r, "content": c} for r, c in msgs]
    points = sorted({min(c, len(history)) for c in cuts} | {len(history)})
    t0 = datetime(2026, 1, 1)
    for i, k in enumerate(points):
        asyncio.run(ep.ingest(scope, history[:k], None, t0 + timedelta(minutes=i)))
    session = storage.get_session()
    try:
        stored = [(t.role, t.content) for t in session.query(Turn).order_by(Turn.seq).all()]
    finally:
        session.close()
    assert stored == [(m["role"], m["content"]) for m in history]
