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
  * Sync: every graph write transaction takes the next ``graph_seq`` under the
    database write lock, so sequence order is commit order. Each partition
    remembers the sequence it is synced to and pulls only rows with a larger
    ``seq``, covering both inserts and invalidations; when nothing changed this is
    one indexed read. Syncing by timestamp was rejected: a concurrent writer can
    commit an older ``invalidated_at`` after a newer one, and a property test found
    that a time-based engine missed it forever. Hard deletes bump
    ``memorable_meta.graph_epoch``, which forces a full rebuild. All of this works
    across processes.
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
from array import array
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

NodeKey = Tuple[str, int]

# G0 parity with v0.2 retrieval: walk entity<->entity relations and fact<->entity links.
DEFAULT_WEIGHTS: Dict[str, float] = {"relates": 1.0, "about": 1.0}


_KIND_CODES: Dict[str, int] = {}


def _kind_code(kind: str) -> int:
    code = _KIND_CODES.get(kind)
    if code is None:
        code = _KIND_CODES[kind] = len(_KIND_CODES)
    return code


def _node_code(key: NodeKey) -> int:
    """Pack (kind, id) into one int: dict keys stay small ints, not tuples."""
    return (_kind_code(key[0]) << 48) | int(key[1])


def _weights_key(weights: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((k, float(v)) for k, v in weights.items() if v))


class _Partition:
    """
    Edges of one tier partition in compact typed arrays, plus cached CSR per weight profile.

    Storage is about 22 bytes per edge (int32 src/dst, uint8 kind code, float32 weight,
    alive flag) plus the id->position map and the node index. The first version kept
    Python lists of str/int objects, which cost 342 bytes per edge (measured).
    """

    def __init__(self, namespace: Optional[str], user_id: Optional[str]):
        self.namespace = namespace
        self.user_id = user_id
        self.index: Dict[int, int] = {}  # packed node code -> local index
        self.node_kind = array("B")  # kind code per node (vectorised ranking)
        self.node_id = array("q")
        self.src = array("i")
        self.dst = array("i")
        self.kind = array("B")
        self.weight = array("f")
        self.eids = array("q")
        self._pos: Optional[Dict[int, int]] = None  # only if edge ids arrive out of order
        self.alive = bytearray()
        self.max_id = 0
        self.seq = 0  # graph write sequence this partition is synced to
        self.loaded = False
        self.version = 0
        self._mats: Dict[Any, Any] = {}

    @property
    def n_nodes(self) -> int:
        return len(self.node_id)

    def node(self, key: NodeKey) -> int:
        code = _node_code(key)
        idx = self.index.get(code)
        if idx is None:
            idx = len(self.node_id)
            self.index[code] = idx
            self.node_kind.append(code >> 48)
            self.node_id.append(int(key[1]))
        return idx

    def _position(self, eid: int) -> Optional[int]:
        if self._pos is not None:
            return self._pos.get(eid)
        ids = np.frombuffer(self.eids, dtype=np.int64)
        i = int(np.searchsorted(ids, eid))
        return i if i < len(ids) and ids[i] == eid else None

    def add(self, rows: Sequence[Tuple[int, str, int, str, str, int, float]]) -> None:
        node = self.node
        added = False
        for eid, sk, si, kind, dk, di, w in rows:
            eid = int(eid)
            if eid <= self.max_id and self._position(eid) is not None:
                continue  # already loaded (deltas may repeat rows)
            if eid < self.max_id and self._pos is None:
                # Out-of-order id (e.g. Postgres sequences): switch to an explicit map.
                self._pos = {int(e): i for i, e in enumerate(self.eids)}
            if self._pos is not None:
                self._pos[eid] = len(self.eids)
            self.eids.append(eid)
            self.src.append(node((sk, si)))
            self.dst.append(node((dk, di)))
            self.kind.append(_kind_code(kind))
            self.weight.append(1.0 if w is None else float(w))
            self.alive.append(1)
            if eid > self.max_id:
                self.max_id = eid
            added = True
        if added:
            self.version += 1
            self._mats.clear()

    def kill(self, edge_ids: Iterable[int]) -> None:
        changed = False
        for eid in edge_ids:
            p = self._position(int(eid))
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
        table = np.zeros(max(_KIND_CODES.values(), default=0) + 1, dtype=np.float32)
        for k, v in weights.items():
            if k in _KIND_CODES:
                table[_KIND_CODES[k]] = v
        w = (
            table[np.frombuffer(self.kind, dtype=np.uint8)]
            * np.frombuffer(self.weight, dtype=np.float32)
            * np.frombuffer(self.alive, dtype=np.uint8)
        )
        keep = w > 0
        return (
            np.frombuffer(self.src, dtype=np.int32)[keep].astype(np.int64),
            np.frombuffer(self.dst, dtype=np.int32)[keep].astype(np.int64),
            w[keep],
        )

    def matrix(self, weights: Dict[str, float]):
        """Symmetric weighted adjacency (n x n, float32 CSR) for this partition alone."""
        key = ("self", _weights_key(weights))
        m = self._mats.get(key)
        if m is None:
            m = _sym_csr(*self.coo(weights), n=self.n_nodes)
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
        self.n_s = shared.n_nodes if shared else 0
        self.extra_index: Dict[int, int] = {}  # packed code -> offset after the shared block
        self.mats = []
        kinds: List[np.ndarray] = []
        ids: List[np.ndarray] = []
        if shared is not None:
            self.mats.append(shared.matrix(weights))
            kinds.append(np.frombuffer(shared.node_kind, dtype=np.uint8)[: self.n_s].astype(np.int16))
            ids.append(np.frombuffer(shared.node_id, dtype=np.int64)[: self.n_s].copy())
        n_extra = 0
        if user is not None:
            n_u = user.n_nodes
            u_kind = np.frombuffer(user.node_kind, dtype=np.uint8)[:n_u]
            u_id = np.frombuffer(user.node_id, dtype=np.int64)[:n_u]
            if shared is None:
                self.mats.append(user.matrix(weights))
                self.extra_index = user.index
                n_extra = n_u
                kinds.append(u_kind.astype(np.int16))
                ids.append(u_id.copy())
            else:
                remap = np.empty(n_u, dtype=np.int64)
                extra_rows = []
                for j in range(n_u):
                    code = (int(u_kind[j]) << 48) | int(u_id[j])
                    s_idx = shared.index.get(code)
                    if s_idx is not None and s_idx < self.n_s:
                        remap[j] = s_idx
                    else:
                        self.extra_index[code] = n_extra
                        remap[j] = self.n_s + n_extra
                        extra_rows.append(j)
                        n_extra += 1
                src, dst, w = user.coo(weights)
                self.mats.append(_sym_csr(remap[src], remap[dst], w, self.n_s + n_extra))
                rows = np.asarray(extra_rows, dtype=np.int64)
                kinds.append(u_kind[rows].astype(np.int16))
                ids.append(u_id[rows].copy())
        # Snapshot the size: partitions keep growing, but this view's matrices don't.
        self.n = self.n_s + n_extra
        self.node_kind = np.concatenate(kinds) if kinds else np.zeros(0, dtype=np.int16)
        self.node_id = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)
        self._deg_cache: Dict[Any, np.ndarray] = {}

    def idx(self, key: NodeKey) -> Optional[int]:
        code = _node_code(key)
        if self.shared is not None:
            i = self.shared.index.get(code)
            if i is not None and i < self.n_s:
                return i
        j = self.extra_index.get(code)
        if j is None:
            return None
        i = self.n_s + j
        return i if i < self.n else None

    def key(self, i: int) -> NodeKey:
        kind = next(k for k, c in _KIND_CODES.items() if c == int(self.node_kind[i]))
        return (kind, int(self.node_id[i]))

    def matvec(self, v: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for m in self.mats:
            k = m.shape[0]
            if k:
                out[:k] += m @ v[:k]
        return out

    def nnz(self) -> int:
        return int(sum(m.nnz for m in self.mats))

    def combined(self):
        """All matrices summed into one n x n CSR (cached; used by the push kernel)."""
        if getattr(self, "_combined", None) is None:
            import scipy.sparse as sp

            total = None
            for m in self.mats:
                k = m.shape[0]
                if k < self.n:
                    indptr = np.concatenate([m.indptr, np.full(self.n - k, m.indptr[-1], dtype=m.indptr.dtype)])
                    m = sp.csr_matrix((m.data, m.indices, indptr), shape=(self.n, self.n))
                total = m if total is None else total + m
            self._combined = total if total is not None else sp.csr_matrix((self.n, self.n), dtype=np.float32)
        return self._combined

    def present(self) -> np.ndarray:
        """
        Nodes with at least one live edge in this view. A node whose edges were all
        invalidated stays in the partition's index until the next rebuild, but it must
        behave exactly as if it were absent (found by the stateful property test).
        """
        if getattr(self, "_present", None) is None:
            self._present = self.matvec(np.ones(self.n, dtype=np.float32)) > 0
        return self._present

    def push_arrays(self):
        if getattr(self, "_push_arrays", None) is None:
            from memorable_ai.graph._push import prepare

            self._push_arrays = prepare(self.combined())
        return self._push_arrays


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

    def _check_state(self) -> int:
        """One round trip: rebuild on epoch change; return the current graph write sequence."""
        epoch, seq = self.storage.graph_state()
        if epoch != self._epoch:
            self._parts.clear()
            self._views.clear()
            self._epoch = epoch
        return seq

    def _partition(self, namespace: Optional[str], user_id: Optional[str], seq: int = 0) -> _Partition:
        key = (namespace, user_id)
        part = self._parts.get(key)
        if part is None:
            part = _Partition(namespace, user_id)
            self._parts[key] = part
        if not part.loaded:
            # ``seq`` was read before this load. Writes committed in between are
            # loaded now and seen again by the next delta; ``add`` is idempotent.
            part.add(self.storage.partition_edges(namespace, user_id))
            part.seq, part.loaded = seq, True
            return part
        if seq <= part.seq:
            return part  # no graph write committed anywhere since the last sync
        changes = self.storage.partition_changes(namespace, user_id, part.seq)
        added = [r[:7] for r in changes if not r[7]]
        dead = [r[0] for r in changes if r[7]]
        if added:
            part.add(added)
        if dead:
            part.kill(dead)
        part.seq = seq
        return part

    def _view(self, namespace: Optional[str], user_id: Optional[str], include_shared: bool,
              weights: Dict[str, float], seq: int = 0) -> _View:
        shared = self._partition(namespace, None, seq) if (include_shared or user_id is None) else None
        user = self._partition(namespace, user_id, seq) if user_id is not None else None
        if shared is not None and not shared.n_nodes and user is not None:
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
            seq = self._check_state()
            view = self._view(namespace, user_id, include_shared, weights, seq)
            n = view.n
            if n == 0:
                return PPRResult(None)
            excluded = sorted({i for i in (view.idx(k) for k in exclude) if i is not None})
            present = view.present()
            s = np.zeros(n, dtype=np.float32)
            for key, w in seeds.items():
                i = view.idx(key)
                if i is not None and w > 0 and present[i]:
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
            use_push = method in ("push", "push-python") or (method == "auto" and view.nnz() > self.exact_max_entries)
            if use_push and method != "push-python":
                from memorable_ai.graph._push import push_ppr

                scores = push_ppr(view.push_arrays(), deg, mask, s, 1.0 - self.damping, self.push_eps)
                return PPRResult(view, scores.astype(np.float32) * present)
            if use_push:
                sparse = self._push(view, s, mask, deg)
                return PPRResult(view, sparse={i: v for i, v in sparse.items() if present[i]})
            return PPRResult(view, self._exact(view, s, mask, deg) * present)

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
                "nodes": sum(p.n_nodes for p in self._parts.values()),
                "edges": sum(len(p.src) for p in self._parts.values()),
            }
