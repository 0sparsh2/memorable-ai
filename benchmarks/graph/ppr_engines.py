"""Benchmark graph-engine options for memorable's PPR retrieval (docs/memory_graph_design.md §3.3).

Run: MEMORABLE_LLM__DEFAULT_MODEL=none python benchmarks/graph/ppr_engines.py

A. current: KnowledgeGraph.ppr (SQL BFS per query + networkx pagerank on a 2-hop subgraph)
B. CSR exact: scipy sparse adjacency cached in-process, full-graph PPR by power iteration
C. CSR local push: Andersen-Chung-Lang forward push (cost independent of graph size)

Quality reference = B (exact full-graph PPR). Reports latency and overlap@20 of fact rankings.
"""
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
from memorable_ai.core.scope import Scope  # noqa: E402
from memorable_ai.core.storage import Storage  # noqa: E402
from memorable_ai.graph.knowledge_graph import KnowledgeGraph  # noqa: E402

RNG = random.Random(7)
SCRATCH = tempfile.mkdtemp(prefix="memorable_ppr_")


def zipf_pick(n, s=1.1):
    # inverse-CDF-ish Zipf sampler over [0, n)
    while True:
        x = int(np.random.zipf(s)) - 1
        if x < n:
            return x


def build(n_facts, n_ents, path):
    if os.path.exists(path):
        os.remove(path)
    st = Storage(f"sqlite:///{path}")
    con = sqlite3.connect(path)
    now = datetime(2026, 1, 1).isoformat(sep=" ")
    perm = list(range(n_ents))
    RNG.shuffle(perm)  # hubs are random ids, not low ids
    con.executemany(
        "INSERT INTO entities(id, namespace, user_id, name, norm_name, entity_type, is_self, mention_count, first_seen, last_seen)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(i + 1, None, "u1", f"ent{i}", f"ent{i}", "concept", 0, 0, now, now) for i in range(n_ents)],
    )
    mems, ments, rels = [], [], []
    for f in range(1, n_facts + 1):
        mems.append((f, f"fact {f}", "fact", None, "u1", now, now))
        k = RNG.choice((1, 2, 2, 3))
        ents = list({perm[zipf_pick(n_ents)] + 1 for _ in range(k)})
        for e in ents:
            ments.append((e, f, now))
        if len(ents) >= 2:
            rels.append((None, "u1", ents[0], "related_to", ents[1], f, 1.0, now, now))
    con.executemany(
        "INSERT INTO memories(id, content, memory_type, namespace, user_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        mems,
    )
    con.executemany("INSERT INTO mentions(entity_id, memory_id, created_at) VALUES (?,?,?)", ments)
    con.executemany(
        "INSERT INTO relations(namespace, user_id, subject_id, predicate, object_id, memory_id, confidence, valid_from, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        rels,
    )
    con.commit()
    con.close()
    return st, perm, len(ments), len(rels)


class CSRGraph:
    """Entities 0..E-1, facts E..E+F-1, undirected, row-normalised transition."""

    def __init__(self, path, n_ents, n_facts):
        t = time.perf_counter()
        con = sqlite3.connect(path)
        rows = con.execute("SELECT entity_id-1, memory_id FROM mentions").fetchall()
        rrows = con.execute("SELECT subject_id-1, object_id-1 FROM relations WHERE invalidated_at IS NULL").fetchall()
        con.close()
        E, F = n_ents, n_facts
        self.E, self.N = E, E + F
        src = [r[0] for r in rows] + [r[0] for r in rrows]
        dst = [E + r[1] - 1 for r in rows] + [r[1] for r in rrows]
        a = np.array(src + dst, dtype=np.int64)
        b = np.array(dst + src, dtype=np.int64)
        A = sp.csr_matrix((np.ones(len(a), dtype=np.float64), (a, b)), shape=(self.N, self.N))
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        self.P_T = sp.csr_matrix(sp.diags(1.0 / deg) @ A).T.tocsr()  # column-stochastic transpose for x @ P
        self.indptr, self.indices = A.indptr, A.indices
        self.deg = deg
        self.load_s = time.perf_counter() - t

    def exact(self, seeds, damping=0.85, tol=1e-8, max_iter=100):
        s = np.zeros(self.N)
        for e, w in seeds.items():
            s[e] = w
        s /= s.sum()
        x = s.copy()
        for _ in range(max_iter):
            nx_ = damping * (self.P_T @ x) + (1 - damping) * s
            if np.abs(nx_ - x).sum() < tol:
                x = nx_
                break
            x = nx_
        return x

    def push(self, seeds, damping=0.85, eps=1e-5):
        alpha = 1 - damping
        tot = sum(seeds.values())
        p, r = {}, {e: w / tot for e, w in seeds.items()}
        queue = [u for u in r if r[u] >= eps * self.deg[u]]
        indptr, indices, deg = self.indptr, self.indices, self.deg
        while queue:
            u = queue.pop()
            ru = r.get(u, 0.0)
            if ru < eps * deg[u]:
                continue
            p[u] = p.get(u, 0.0) + alpha * ru
            share = (1 - alpha) * ru / deg[u]
            r[u] = 0.0
            for v in indices[indptr[u] : indptr[u + 1]]:
                old = r.get(v, 0.0)
                new = old + share
                r[v] = new
                if old < eps * deg[v] <= new:
                    queue.append(v)
        return p

    def top_facts(self, scores, k=20):
        if isinstance(scores, dict):
            items = [(n - self.E + 1, s) for n, s in scores.items() if n >= self.E]
            items.sort(key=lambda x: -x[1])
            return [i for i, _ in items[:k]]
        f = scores[self.E :]
        idx = np.argpartition(-f, min(k, len(f) - 1))[:k]
        idx = idx[np.argsort(-f[idx])]
        return [int(i) + 1 for i in idx if f[i] > 0]


def overlap(a, b, k=20):
    a, b = set(a[:k]), set(b[:k])
    return len(a & b) / max(1, min(k, len(a), len(b))) if a and b else float("nan")


def run(label, n_facts, n_ents, n_queries=40):
    path = os.path.join(SCRATCH, f"bench_{label}.db")
    st, perm, n_ment, n_rel = build(n_facts, n_ents, path)
    kg = KnowledgeGraph(st)
    scope = Scope(user_id="u1")
    g = CSRGraph(path, n_ents, n_facts)
    queries = []
    for _ in range(n_queries):
        e1, e2 = perm[zipf_pick(n_ents, 1.3)], perm[RNG.randrange(n_ents)]
        queries.append((f"what do I know about ent{e1} and ent{e2}", {e1: 1.0, e2: 1.0}))
    tA, tB, tC, oAB, oCB = [], [], [], [], []
    for q, seeds in queries:
        t = time.perf_counter()
        a = [m for m, _ in kg.ppr(q, scope, k=20)]
        tA.append(time.perf_counter() - t)
        t = time.perf_counter()
        b = g.top_facts(g.exact(seeds))
        tB.append(time.perf_counter() - t)
        t = time.perf_counter()
        c = g.top_facts(g.push(seeds))
        tC.append(time.perf_counter() - t)
        oAB.append(overlap(a, b))
        oCB.append(overlap(c, b))
    ms = lambda xs: f"p50 {1000*np.median(xs):7.1f} ms  p95 {1000*np.percentile(xs,95):7.1f} ms"
    print(f"\n== {label}: {n_facts:,} facts, {n_ents:,} entities, {n_ment:,} mentions, {n_rel:,} relations ==")
    print(f"  CSR cache build (one-off, from SQL): {1000*g.load_s:.0f} ms")
    print(f"  A current  (SQL BFS + networkx, 2-hop): {ms(tA)}   overlap@20 vs exact {np.nanmean(oAB):.2f}")
    print(f"  B CSR exact power iteration          : {ms(tB)}   (reference)")
    print(f"  C CSR local push (eps=1e-5)          : {ms(tC)}   overlap@20 vs exact {np.nanmean(oCB):.2f}")
    st.engine.dispose() if hasattr(st, "engine") else None
    os.remove(path)


if __name__ == "__main__":
    np.random.seed(7)
    run("per-user", 5_000, 2_000)
    run("team", 50_000, 15_000)
    run("org", 300_000, 80_000, n_queries=20)
