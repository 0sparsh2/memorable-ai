"""
In-process sparse-matrix graph engine (docs/memory_graph_design.md §3.3).

SQL (the ``edges`` table) is the system of record. This engine keeps each tier
partition, ``(namespace, user_id)`` or ``(namespace, None)`` for the shared org
tier, in memory as edge arrays plus cached scipy CSR matrices, and answers
Personalized PageRank (PPR) queries with sparse matrix-vector products.

Why: the v0.2 path rebuilt a networkx graph from SQL on every query, capped at two
hops and 4,000 nodes. On synthetic graphs it took 48 / 271 / 862 ms
(per-user / team / org) and agreed with exact PPR only 26% of the time at org
scale. This engine takes about 1.4 / 16 / 71 ms for exact PPR on the whole
partition (benchmarks/graph/ppr_engines.py).

Mechanics:
  * Sync: each partition tracks the highest edge id it has loaded and the newest
    ``invalidated_at`` it has applied, so every query pulls only deltas.
    Hard deletes bump ``memorable_meta.graph_epoch``, which forces a full rebuild.
    This works across processes.
  * Views: a query for user ``u`` with the shared tier walks one matrix space,
    with shared nodes first and then ``u``-only nodes. The shared matrix is reused
    as-is, and only ``u``'s small matrix is remapped. Other users' edges are never
    loaded into the view, so isolation holds by construction.
  * Typed weights: ``A = sum_k w_k * A_k`` over edge kinds, cached per weight
    profile. ``DEFAULT_WEIGHTS`` keeps v0.2 behaviour (entity-entity + fact-entity
    only). The multi-layer profiles come in G2.
  * Masking: excluded nodes (e.g. the user's own "self" hub) are removed exactly.
    Degrees are recomputed on the induced subgraph (``deg = A @ mask``).
  * Exact float32 power iteration by default. Forward local push (Andersen,
    Chung and Lang 2006) above ``exact_max_entries``, where cost no longer grows
    with graph size.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

NodeKey = Tuple[str, int]

# G0 parity with v0.2 retrieval: walk entity<->entity relations and fact<->entity links.
DEFAULT_WEIGHTS: Dict[str, float] = {"relates": 1.0, "about": 1.0}

_INVALIDATION_SKEW = timedelta(seconds=2)  # re-check a small window for out-of-order commits

_KIND_CODES: Dict[str, int] = {}


def _kind_code(kind: str) -> int:
    code = _KIND_CODES.get(kind)
    if code is None:
        code = _KIND_CODES[kind] = len(_KIND_CODES)
    return code


def _weights_key(weights: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((k, float(v)) for k, v in weights.items() if v))


class _Partition:
    """Edges of one tier partition as growable arrays, plus cached CSR per weight profile."""

    def __init__(self, namespace: Optional[str], user_id: Optional[str]):
        self.namespace = namespace
        self.user_id = user_id
        self.index: Dict[NodeKey, int] = {}
        self.keys: List[NodeKey] = []
        self.node_kind: List[int] = []  # kind code per node (vectorised ranking)
        self.node_id: List[int] = []
        self.src: List[int] = []
        self.dst: List[int] = []
        self.kind: List[str] = []
        self.weight: List[float] = []
        self.pos: Dict[int, int] = {}
        self.alive = bytearray()
        self.max_id = 0
        self.since: Optional[datetime] = None
        self.cursor: Any = None  # table-wide (max id, max invalidated_at) at last sync
        self.loaded = False
        self.version = 0
        self._mats: Dict[Any, Any] = {}

    def node(self, key: NodeKey) -> int:
        idx = self.index.get(key)
        if idx is None:
            idx = len(self.keys)
            self.index[key] = idx
            self.keys.append(key)
            self.node_kind.append(_kind_code(key[0]))
            self.node_id.append(key[1])
        return idx

    def add(self, rows: Sequence[Tuple[int, str, int, str, str, int, float]]) -> None:
        for eid, sk, si, kind, dk, di, w in rows:
            if eid in self.pos:
                continue
            self.pos[eid] = len(self.src)
            self.src.append(self.node((sk, int(si))))
            self.dst.append(self.node((dk, int(di))))
            self.kind.append(kind)
            self.weight.append(1.0 if w is None else float(w))
            self.alive.append(1)
            self.max_id = max(self.max_id, int(eid))
        if rows:
            self.version += 1
            self._mats.clear()

    def kill(self, edge_ids: Iterable[int]) -> None:
        changed = False
        for eid in edge_ids:
            p = self.pos.get(int(eid))
            if p is not None and self.alive[p]:
                self.alive[p] = 0
                changed = True
        if changed:
            self.version += 1
            self._mats.clear()

    def coo(self, weights: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Alive, positively weighted edges as (src, dst, weight) arrays in local indices."""
        if not self.src:
            e = np.zeros(0, dtype=np.int64)
            return e, e, np.zeros(0, dtype=np.float32)
        kw = np.array([weights.get(k, 0.0) for k in self.kind], dtype=np.float32)
        w = kw * np.asarray(self.weight, dtype=np.float32) * np.frombuffer(bytes(self.alive), dtype=np.uint8)
        keep = w > 0
        return (
            np.asarray(self.src, dtype=np.int64)[keep],
            np.asarray(self.dst, dtype=np.int64)[keep],
            w[keep],
        )

    def matrix(self, weights: Dict[str, float]):
        """Symmetric weighted adjacency (n x n, float32 CSR) for this partition alone."""
        key = ("self", _weights_key(weights))
        m = self._mats.get(key)
        if m is None:
            m = _sym_csr(*self.coo(weights), n=len(self.keys))
            self._mats[key] = m
        return m


def _sym_csr(src: np.ndarray, dst: np.ndarray, w: np.ndarray, n: int):
    import scipy.sparse as sp

    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.concatenate([w, w]).astype(np.float32, copy=False)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)


class _View:
    """One matrix space for a query scope: shared nodes first, then user-only nodes."""

    def __init__(self, shared: Optional[_Partition], user: Optional[_Partition], weights: Dict[str, float]):
        self.shared, self.user = shared, user
        self.stamp = (shared.version if shared else -1, user.version if user else -1, _weights_key(weights))
        self.n_s = len(shared.keys) if shared else 0
        self.extra_keys: List[NodeKey] = []
        self.extra_index: Dict[NodeKey, int] = {}
        self.mats = []
        if shared is not None:
            self.mats.append(shared.matrix(weights))
        if user is not None:
            if shared is None:
                self.mats.append(user.matrix(weights))
                self.extra_keys, self.extra_index = user.keys, user.index
            else:
                remap = np.empty(len(user.keys), dtype=np.int64)
                for j, key in enumerate(user.keys):
                    s_idx = shared.index.get(key)
                    if s_idx is not None and s_idx < self.n_s:
                        remap[j] = s_idx
                    else:
                        self.extra_index[key] = len(self.extra_keys)
                        self.extra_keys.append(key)
                        remap[j] = self.n_s + self.extra_index[key]
                src, dst, w = user.coo(weights)
                self.mats.append(_sym_csr(remap[src], remap[dst], w, self.n_s + len(self.extra_keys)))
        # Snapshot the size: partitions keep growing, but this view's matrices don't.
        self.n = self.n_s + len(self.extra_keys)
        kinds: List[np.ndarray] = []
        ids: List[np.ndarray] = []
        if shared is not None:
            kinds.append(np.asarray(shared.node_kind[: self.n_s], dtype=np.int16))
            ids.append(np.asarray(shared.node_id[: self.n_s], dtype=np.int64))
        if user is not None:
            extra = self.extra_keys[: self.n - self.n_s]
            if shared is None:
                kinds.append(np.asarray(user.node_kind[: self.n], dtype=np.int16))
                ids.append(np.asarray(user.node_id[: self.n], dtype=np.int64))
            else:
                kinds.append(np.asarray([_kind_code(k[0]) for k in extra], dtype=np.int16))
                ids.append(np.asarray([k[1] for k in extra], dtype=np.int64))
        self.node_kind = np.concatenate(kinds) if kinds else np.zeros(0, dtype=np.int16)
        self.node_id = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)
        self._deg_cache: Dict[Any, np.ndarray] = {}

    def idx(self, key: NodeKey) -> Optional[int]:
        if self.shared is not None:
            i = self.shared.index.get(key)
            if i is not None and i < self.n_s:
                return i
        j = self.extra_index.get(key)
        if j is None:
            return None
        i = self.n_s + j
        return i if i < self.n else None

    def key(self, i: int) -> NodeKey:
        if self.shared is not None and i < self.n_s:
            return self.shared.keys[i]
        return self.extra_keys[i - self.n_s]

    def matvec(self, v: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for m in self.mats:
            k = m.shape[0]
            if k:
                out[:k] += m @ v[:k]
        return out

    def nnz(self) -> int:
        return int(sum(m.nnz for m in self.mats))


class PPRResult:
    """Scores over a view's nodes; ``top(kind, k)`` ranks nodes of one kind."""

    def __init__(self, view: Optional[_View], scores: Optional[np.ndarray] = None,
                 sparse: Optional[Dict[int, float]] = None):
        self._view = view
        self._scores = scores
        self._sparse = sparse

    def top(self, kind: str, k: int = 20) -> List[Tuple[int, float]]:
        if self._view is None:
            return []
        view = self._view
        code = _KIND_CODES.get(kind)
        if code is None:
            return []
        if self._sparse is not None:
            items = [(int(view.node_id[i]), s) for i, s in self._sparse.items()
                     if s > 0 and view.node_kind[i] == code]
            return sorted(items, key=lambda x: -x[1])[:k]
        cand = np.flatnonzero((view.node_kind == code) & (self._scores > 0))
        if cand.size == 0:
            return []
        if cand.size > k:
            cand = cand[np.argpartition(-self._scores[cand], k - 1)[:k]]
        cand = cand[np.argsort(-self._scores[cand], kind="stable")]
        return [(int(view.node_id[i]), float(self._scores[i])) for i in cand]

    def score(self, key: NodeKey) -> float:
        if self._view is None:
            return 0.0
        i = self._view.idx(key)
        if i is None:
            return 0.0
        if self._sparse is not None:
            return float(self._sparse.get(i, 0.0))
        return float(self._scores[i])


class GraphEngine:
    """Delta-synced CSR partitions with exact / local-push PPR."""

    def __init__(
        self,
        storage: Any,
        weights: Optional[Dict[str, float]] = None,
        damping: float = 0.85,
        tol: float = 1e-4,  # ranking-only use: top-k unchanged vs 1e-6, ~40% faster (measured)
        max_iter: int = 100,
        exact_max_entries: int = 4_000_000,
        push_eps: float = 1e-5,
    ):
        self.storage = storage
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.damping = damping
        self.tol = tol
        self.max_iter = max_iter
        self.exact_max_entries = exact_max_entries
        self.push_eps = push_eps
        self._parts: Dict[Tuple[Optional[str], Optional[str]], _Partition] = {}
        self._views: Dict[Tuple[Any, ...], _View] = {}
        self._epoch: Optional[int] = None
        self._lock = threading.RLock()
        self.max_views = 64

    # ------------------------------------------------------------------ sync

    def _check_state(self) -> Any:
        """One round trip: rebuild on epoch change; return the table-wide change cursor."""
        epoch, max_id, max_inv = self.storage.graph_state()
        if epoch != self._epoch:
            self._parts.clear()
            self._views.clear()
            self._epoch = epoch
        return (max_id, max_inv)

    def _partition(self, namespace: Optional[str], user_id: Optional[str], cursor: Any = None) -> _Partition:
        key = (namespace, user_id)
        part = self._parts.get(key)
        if part is None:
            part = _Partition(namespace, user_id)
            self._parts[key] = part
        if part.loaded and cursor is not None and part.cursor == cursor:
            return part  # nothing was added or invalidated anywhere since the last sync
        part.cursor = cursor
        rows = self.storage.edges_after(namespace, user_id, part.max_id)
        if rows:
            part.add(rows)
        if not part.loaded:
            # Everything loaded so far was valid; only later invalidations matter.
            part.since = self.storage.max_edge_invalidated_at(namespace, user_id)
            part.loaded = True
        elif part.max_id:
            since = part.since - _INVALIDATION_SKEW if part.since else None
            inv = self.storage.edge_invalidations_since(namespace, user_id, since, part.max_id)
            if inv:
                part.kill(i for i, _ in inv)
                newest = max(t for _, t in inv)
                part.since = max(part.since, newest) if part.since else newest
        return part

    def _view(self, namespace: Optional[str], user_id: Optional[str], include_shared: bool,
              weights: Dict[str, float], cursor: Any = None) -> _View:
        shared = self._partition(namespace, None, cursor) if (include_shared or user_id is None) else None
        user = self._partition(namespace, user_id, cursor) if user_id is not None else None
        if shared is not None and not shared.keys and user is not None:
            shared = None  # empty org tier: walk the user's partition alone
        vkey = (namespace, user_id, shared is not None, _weights_key(weights))
        stamp = (shared.version if shared else -1, user.version if user else -1, _weights_key(weights))
        view = self._views.get(vkey)
        if view is None or view.stamp != stamp:
            view = _View(shared, user, weights)
            self._views[vkey] = view
            if len(self._views) > self.max_views:
                self._views.pop(next(iter(self._views)))
        return view

    def invalidate(self) -> None:
        """Drop all cached partitions (next query reloads from SQL)."""
        with self._lock:
            self._parts.clear()
            self._views.clear()
            self._epoch = None

    # ------------------------------------------------------------------ queries

    def ppr(
        self,
        namespace: Optional[str],
        user_id: Optional[str],
        seeds: Dict[NodeKey, float],
        include_shared: bool = False,
        exclude: Iterable[NodeKey] = (),
        weights: Optional[Dict[str, float]] = None,
        method: str = "auto",
    ) -> PPRResult:
        """
        Personalized PageRank from ``seeds`` over the scope's view.

        ``user_id=None`` walks the org tier only; otherwise the user's partition,
        plus the org tier when ``include_shared``. ``exclude`` nodes are removed
        from the walk exactly (induced subgraph).
        """
        weights = dict(weights or self.weights)
        with self._lock:
            cursor = self._check_state()
            view = self._view(namespace, user_id, include_shared, weights, cursor)
            n = view.n
            if n == 0:
                return PPRResult(None)
            excluded = sorted({i for i in (view.idx(k) for k in exclude) if i is not None})
            s = np.zeros(n, dtype=np.float32)
            for key, w in seeds.items():
                i = view.idx(key)
                if i is not None and w > 0:
                    s[i] += float(w)
            if excluded:
                s[excluded] = 0.0
            total = float(s.sum())
            if total <= 0:
                return PPRResult(view, np.zeros(n, dtype=np.float32))
            s /= total
            mask_key = tuple(excluded)
            deg = view._deg_cache.get(mask_key)
            mask = np.ones(n, dtype=np.float32)
            if excluded:
                mask[excluded] = 0.0
            if deg is None:
                deg = view.matvec(mask) * mask
                view._deg_cache.clear()
                view._deg_cache[mask_key] = deg
            use_push = method == "push" or (method == "auto" and view.nnz() > self.exact_max_entries)
            if use_push:
                return PPRResult(view, sparse=self._push(view, s, mask, deg))
            return PPRResult(view, self._exact(view, s, mask, deg))

    def _exact(self, view: _View, s: np.ndarray, mask: np.ndarray, deg: np.ndarray) -> np.ndarray:
        d = np.float32(self.damping)
        inv = np.zeros_like(deg)
        nz = deg > 0
        inv[nz] = 1.0 / deg[nz]
        dangling = (~nz) & (mask > 0)
        x = s.copy()
        for _ in range(self.max_iter):
            lost = float(x[dangling].sum()) if dangling.any() else 0.0
            x_new = d * (view.matvec(x * inv) * mask) + np.float32(d * lost + (1.0 - self.damping)) * s
            if float(np.abs(x_new - x).sum()) < self.tol:
                return x_new
            x = x_new
        return x

    def _push(self, view: _View, s: np.ndarray, mask: np.ndarray, deg: np.ndarray) -> Dict[int, float]:
        alpha = 1.0 - self.damping
        eps = self.push_eps
        mats = [(m.shape[0], m.indptr, m.indices, m.data) for m in view.mats]
        p: Dict[int, float] = {}
        r: Dict[int, float] = {int(i): float(s[i]) for i in np.flatnonzero(s)}
        queue = [u for u, ru in r.items() if deg[u] > 0 and ru >= eps * deg[u]]
        while queue:
            u = queue.pop()
            ru = r.get(u, 0.0)
            du = float(deg[u])
            if du <= 0 or ru < eps * du:
                continue
            p[u] = p.get(u, 0.0) + alpha * ru
            r[u] = 0.0
            share = (1.0 - alpha) * ru / du
            for size, indptr, indices, data in mats:
                if u >= size:
                    continue
                for j in range(indptr[u], indptr[u + 1]):
                    v = int(indices[j])
                    if mask[v] == 0:
                        continue
                    old = r.get(v, 0.0)
                    new = old + share * float(data[j])
                    r[v] = new
                    dv = float(deg[v])
                    if dv > 0 and old < eps * dv <= new:
                        queue.append(v)
        for u, ru in r.items():  # isolated seeds keep their teleport mass
            if deg[u] <= 0 and ru > 0:
                p[u] = p.get(u, 0.0) + ru
        return p

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "partitions": len(self._parts),
                "views": len(self._views),
                "nodes": sum(len(p.keys) for p in self._parts.values()),
                "edges": sum(len(p.src) for p in self._parts.values()),
            }
