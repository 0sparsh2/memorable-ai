"""Benchmark graph-ranking engines for memorable's PPR retrieval (docs/memory_graph_design.md §3.3).

Run: MEMORABLE_LLM__DEFAULT_MODEL=none python benchmarks/graph/ppr_engines.py [--sizes per-user,team,org]

  legacy   the v0.2 algorithm: SQL BFS per query (2 hops, 4,000-node cap) + networkx
           pagerank; kept here only as the baseline
  engine   memorable_ai.graph.engine.GraphEngine: delta-synced CSR, exact float32
           power iteration (the production path)
  push     GraphEngine forced to forward local push (the large-graph fallback)
  ref      exact float64 PPR on the full graph (quality reference)

Graphs are synthetic: Zipf-distributed entity popularity; each fact links to 1-3
entities with ``about`` edges, and facts with 2+ entities add a ``relates`` edge.
Reports p50/p95 latency and overlap@20 with the reference. Note that many facts
attached to one entity tie on score, which caps overlap below 1.0 for every engine.
"""

import argparse
import os
import random
import sqlite3
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from memorable_ai.core.storage import Storage  # noqa: E402
from memorable_ai.graph.engine import GraphEngine  # noqa: E402

SIZES = {"per-user": (5_000, 2_000), "team": (50_000, 15_000), "org": (300_000, 80_000)}


def zipf_pick(rng_np, n, s=1.1):
    while True:
        x = int(rng_np.zipf(s)) - 1
        if x < n:
            return x


def build(n_facts, n_ents, path, seed=7):
    rng, rng_np = random.Random(seed), np.random.default_rng(seed)
    if os.path.exists(path):
        os.remove(path)
    st = Storage(f"sqlite:///{path}")
    perm = list(range(1, n_ents + 1))
    rng.shuffle(perm)
    now = datetime(2026, 1, 1).isoformat(sep=" ")
    rows = []
    for f in range(1, n_facts + 1):
        ents = list({perm[zipf_pick(rng_np, n_ents)] for _ in range(rng.choice((1, 2, 2, 3)))})
        rows += [(None, "u1", "fact", f, "about", "entity", e, 1.0, 1.0, now) for e in ents]
        if len(ents) >= 2:
            rows.append((None, "u1", "entity", ents[0], "relates", "entity", ents[1], 1.0, 1.0, now))
    con = sqlite3.connect(path)
    con.executemany(
        "INSERT INTO edges(namespace, user_id, src_kind, src_id, kind, dst_kind, dst_id, weight, confidence, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()
    return st, perm, len(rows), rng, rng_np


def legacy_ppr(path, seeds, hops=2, max_nodes=4000, k=20):
    """The v0.2 algorithm, re-expressed over the edges table."""
    import networkx as nx

    con = sqlite3.connect(path)
    G = nx.Graph()
    frontier, visited = set(seeds), set()
    for _ in range(hops):
        if not frontier or G.number_of_nodes() > max_nodes:
            break
        visited |= frontier
        q = ",".join(str(e) for e in frontier)
        nxt = set()
        for s, o in con.execute(
            f"SELECT src_id, dst_id FROM edges WHERE kind='relates' AND (src_id IN ({q}) OR dst_id IN ({q}))"
        ):
            G.add_edge(f"e{s}", f"e{o}")
            nxt.update((s, o))
        mems = set()
        for f, e in con.execute(f"SELECT src_id, dst_id FROM edges WHERE kind='about' AND dst_id IN ({q})"):
            G.add_edge(f"e{e}", f"m{f}")
            mems.add(f)
        if mems:
            mq = ",".join(str(m) for m in mems)
            for f, e in con.execute(f"SELECT src_id, dst_id FROM edges WHERE kind='about' AND src_id IN ({mq})"):
                G.add_edge(f"e{e}", f"m{f}")
                nxt.add(e)
        frontier = nxt - visited
    con.close()
    pers = {f"e{e}": w for e, w in seeds.items() if f"e{e}" in G}
    if not pers:
        return []
    sc = nx.pagerank(G, alpha=0.85, personalization=pers, max_iter=100)
    return [int(n[1:]) for n, _ in sorted(((n, s) for n, s in sc.items() if n[0] == "m"), key=lambda x: -x[1])[:k]]


def reference_top(path, n_facts, n_ents, seeds, k=20):
    con = sqlite3.connect(path)
    src, dst = [], []
    for sk, si, dk, di in con.execute("SELECT src_kind, src_id, dst_kind, dst_id FROM edges"):
        src.append(si - 1 if sk == "entity" else n_ents + si - 1)
        dst.append(di - 1 if dk == "entity" else n_ents + di - 1)
    con.close()
    N = n_ents + n_facts
    a, b = np.array(src + dst), np.array(dst + src)
    A = sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(N, N))
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    s = np.zeros(N)
    for e, w in seeds.items():
        s[e - 1] = w
    s /= s.sum()
    x = s.copy()
    for _ in range(200):
        x_new = 0.85 * (A @ (x / deg)) + 0.15 * s
        if np.abs(x_new - x).sum() < 1e-10:
            break
        x = x_new
    f = x[n_ents:]
    idx = np.argsort(-f)[:k]
    return [int(i) + 1 for i in idx if f[i] > 0]


def overlap(a, b, k=20):
    a, b = set(a[:k]), set(b[:k])
    return len(a & b) / max(1, min(k, len(a), len(b))) if a and b else float("nan")


def run(label, n_queries=30):
    n_facts, n_ents = SIZES[label]
    tmp = tempfile.mkdtemp(prefix="memorable_ppr_")
    try:
        _run(label, n_queries, n_facts, n_ents, os.path.join(tmp, f"{label}.db"))
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def _run(label, n_queries, n_facts, n_ents, path):
    st, perm, n_edges, rng, rng_np = build(n_facts, n_ents, path)
    eng = GraphEngine(st)
    t = time.perf_counter()
    eng.ppr(None, "u1", {("entity", perm[0]): 1.0})  # first query loads the partition from SQL
    load_ms = 1000 * (time.perf_counter() - t)
    times = {"legacy": [], "engine": [], "push": []}
    ov = {"legacy": [], "engine": [], "push": []}
    for _ in range(n_queries):
        seeds = {perm[zipf_pick(rng_np, n_ents, 1.3)]: 1.0, perm[rng.randrange(n_ents)]: 1.0}
        ref = reference_top(path, n_facts, n_ents, seeds) if n_queries else []
        keyed = {("entity", e): w for e, w in seeds.items()}
        for name, fn in (
            ("legacy", lambda: legacy_ppr(path, seeds)),
            ("engine", lambda: [i for i, _ in eng.ppr(None, "u1", keyed).top("fact", 20)]),
            ("push", lambda: [i for i, _ in eng.ppr(None, "u1", keyed, method="push").top("fact", 20)]),
        ):
            t = time.perf_counter()
            got = fn()
            times[name].append(time.perf_counter() - t)
            ov[name].append(overlap(got, ref))
    print(f"\n== {label}: {n_facts:,} facts, {n_ents:,} entities, {n_edges:,} edges ==")
    print(f"  engine first load from SQL: {load_ms:.0f} ms (one-off per process; deltas after)")
    for name in ("legacy", "engine", "push"):
        xs = times[name]
        print(f"  {name:7s} p50 {1000*np.median(xs):7.1f} ms  p95 {1000*np.percentile(xs, 95):7.1f} ms"
              f"   overlap@20 vs exact {np.nanmean(ov[name]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="per-user,team,org")
    ap.add_argument("--queries", type=int, default=30)
    args = ap.parse_args()
    for label in args.sizes.split(","):
        run(label.strip(), n_queries=args.queries if label != "org" else min(args.queries, 15))
