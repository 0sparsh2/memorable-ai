"""
Forward local-push PPR (Andersen, Chung and Lang 2006) over one CSR matrix.

``push_ppr`` is JIT-compiled with numba when it is installed (the optional
``memorable-ai[fast]`` extra). Otherwise it runs as plain Python, which gives the
same results more slowly. Both versions take dense arrays and return dense
estimates ``p``. Their invariant: ``p + PPR(residual) = PPR(seed)``, with every
residual ``r[u] < eps * deg[u]``.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised when numba is installed
    import numba

    _jit = numba.njit(cache=True, nogil=True)
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def _jit(f):
        return f


@_jit
def _push_kernel(indptr, indices, data, deg, mask, seed_idx, seed_val, alpha, eps):  # pragma: no cover - jitted
    n = deg.shape[0]
    p = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    queue = np.empty(n, dtype=np.int64)
    inq = np.zeros(n, dtype=np.bool_)
    head = 0
    tail = 0
    for j in range(seed_idx.shape[0]):  # only seeds start with residual mass
        u = seed_idx[j]
        r[u] += seed_val[j]
    for j in range(seed_idx.shape[0]):
        u = seed_idx[j]
        if not inq[u] and r[u] > 0.0 and deg[u] > 0.0 and r[u] >= eps * deg[u]:
            queue[tail] = u
            tail += 1
            inq[u] = True
    count = tail  # number of entries currently queued (ring buffer of size n)
    while count > 0:
        u = queue[head]
        head += 1
        if head == n:
            head = 0
        count -= 1
        inq[u] = False
        ru = r[u]
        du = deg[u]
        if du <= 0.0 or ru < eps * du:
            continue
        p[u] += alpha * ru
        r[u] = 0.0
        share = (1.0 - alpha) * ru / du
        for j in range(indptr[u], indptr[u + 1]):
            v = indices[j]
            if mask[v] == 0.0:
                continue
            r[v] += share * data[j]
            dv = deg[v]
            if dv > 0.0 and not inq[v] and r[v] >= eps * dv:
                queue[tail] = v
                tail += 1
                if tail == n:
                    tail = 0
                inq[v] = True
                count += 1
    for j in range(seed_idx.shape[0]):  # isolated seeds keep their teleport mass
        u = seed_idx[j]
        if deg[u] <= 0.0 and r[u] > 0.0:
            p[u] += r[u]
            r[u] = 0.0
    return p


def prepare(csr):
    """Kernel-ready (indptr, indices, data) arrays; cache them per matrix version."""
    return (
        np.ascontiguousarray(csr.indptr, dtype=np.int64),
        np.ascontiguousarray(csr.indices, dtype=np.int64),
        np.ascontiguousarray(csr.data, dtype=np.float64),
    )


def push_ppr(arrays, deg: np.ndarray, mask: np.ndarray, seed: np.ndarray, alpha: float, eps: float) -> np.ndarray:
    """Approximate PPR with teleport probability ``alpha``; returns a dense vector."""
    indptr, indices, data = arrays
    idx = np.flatnonzero(seed).astype(np.int64)
    return _push_kernel(
        indptr, indices, data,
        deg.astype(np.float64, copy=False),
        mask.astype(np.float64, copy=False),
        idx, seed[idx].astype(np.float64),
        float(alpha), float(eps),
    )
