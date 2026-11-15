"""
In-process indexes over SQL rows.

``VectorIndex`` keeps a normalised float32 matrix per memory kind and answers
top-k cosine queries with one matrix product. It replaces the old "load the 1000
highest-importance rows and loop in Python" search, which made newer
low-importance facts unreachable.

``LexicalIndex`` is an incremental BM25 index with an inverted index, used when
SQLite FTS5 is not available (Postgres/MySQL).

Both indexes are kept in sync by ``Storage``: rows are added on insert, and new
rows written by other processes are picked up by an ``id > max_id`` delta query
before each search. Removed or invalidated rows are dropped here when possible
and always filtered again when results are hydrated from SQL.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

_ANY_USER = object()


# Key for the shared (user_id=None) pool. Not "\x00": numpy strips trailing NULs in
# string comparisons, which made shared-pool rows unmatchable.
_SHARED = "\x1f<shared>\x1f"


def _ukey(user_id: Optional[str]) -> str:
    return _SHARED if user_id is None else str(user_id)


class VectorIndex:
    """Brute-force cosine index. Exact, and fast up to a few hundred thousand rows."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ids: List[int] = []
        self._users: List[str] = []
        self._rows: List[np.ndarray] = []
        self._matrix: Optional[np.ndarray] = None
        self._user_arr: Optional[np.ndarray] = None
        self._id_arr: Optional[np.ndarray] = None
        self._dead: Set[int] = set()
        self._id_set: Set[int] = set()
        self._dim: Optional[int] = None
        self.loaded: bool = False

    def __len__(self) -> int:
        return len(self._id_set - self._dead)

    def add(self, item_id: int, user_id: Optional[str], vector: Sequence[float]) -> None:
        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim != 1 or vec.size == 0:
            return
        item_id = int(item_id)
        with self._lock:
            if item_id in self._id_set:
                return
            if self._dim is None:
                self._dim = int(vec.size)
            if vec.size != self._dim:
                return  # embedding model changed; ignore stale-dimension rows
            norm = float(np.linalg.norm(vec))
            if norm == 0:
                return
            self._ids.append(item_id)
            self._id_set.add(item_id)
            self._users.append(_ukey(user_id))
            self._rows.append(vec / norm)
            self._matrix = None

    def remove(self, item_id: int) -> None:
        with self._lock:
            self._dead.add(int(item_id))

    def _materialize(self) -> None:
        if self._matrix is None and self._rows:
            self._matrix = np.vstack(self._rows)
            self._user_arr = np.asarray(self._users, dtype=object)
            self._id_arr = np.asarray(self._ids, dtype=np.int64)

    def search(
        self,
        query: Sequence[float],
        k: int,
        user_id: object = _ANY_USER,
        include_shared: bool = False,
    ) -> List[Tuple[int, float]]:
        """Top-k (id, cosine) restricted to ``user_id`` (``None`` = shared pool)."""
        q = np.asarray(query, dtype=np.float32)
        with self._lock:
            self._materialize()
            if self._matrix is None or self._dim is None or q.size != self._dim:
                return []
            qn = float(np.linalg.norm(q))
            if qn == 0:
                return []
            scores = self._matrix @ (q / qn)
            if user_id is not _ANY_USER:
                mask = self._user_arr == _ukey(user_id)  # type: ignore[arg-type]
                if include_shared and user_id is not None:
                    mask = mask | (self._user_arr == _ukey(None))
                scores = np.where(mask, scores, -np.inf)
            if self._dead:
                dead_mask = np.isin(self._id_arr, list(self._dead))
                scores = np.where(dead_mask, -np.inf, scores)
            n = scores.shape[0]
            if n == 0:
                return []
            k = min(k, n)
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [
                (int(self._id_arr[i]), float(scores[i]))  # type: ignore[index]
                for i in top
                if np.isfinite(scores[i])
            ]


_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """a an the and or but if then else of to in on at by for with from as is are was were be been
    being am do does did have has had i me my mine you your yours we our us he him his she her they
    them their it its this that these those what which who whom whose when where why how not no yes
    so too very can could would should will shall may might must just also about into over under
    again there here than up down out off any all some such own same other s t""".split()
)


def light_stem(word: str) -> str:
    """
    Suffix stripping + truncation stemming (6 chars). Coarse but symmetric:
    live/lives/living -> "liv", allergic/allergies -> "allerg".
    """
    w = word.lower()
    for suf in ("ing", "ies", "es", "ed", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)] + ("y" if suf == "ies" else "")
            break
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w[:6]


def tokenize(text: str, drop_stopwords: bool = True, stem: bool = False) -> List[str]:
    toks = _TOKEN_RE.findall((text or "").lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    if stem:
        toks = [light_stem(t) for t in toks]
    return toks


class LexicalIndex:
    """Incremental BM25 (Okapi) with an inverted index."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._lock = threading.RLock()
        self._postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        self._doc_len: Dict[int, int] = {}
        self._doc_user: Dict[int, str] = {}
        self._total_len = 0
        self._dead: Set[int] = set()
        self.loaded = False

    def add(self, item_id: int, user_id: Optional[str], text: str) -> None:
        toks = tokenize(text, stem=True)
        with self._lock:
            item_id = int(item_id)
            if item_id in self._doc_len:
                return
            for term, tf in Counter(toks).items():
                self._postings[term][item_id] = tf
            self._doc_len[item_id] = len(toks)
            self._doc_user[item_id] = _ukey(user_id)
            self._total_len += len(toks)

    def remove(self, item_id: int) -> None:
        with self._lock:
            self._dead.add(int(item_id))

    def search(
        self,
        query: str,
        k: int,
        user_id: object = _ANY_USER,
        include_shared: bool = False,
    ) -> List[Tuple[int, float]]:
        q_terms = list(dict.fromkeys(tokenize(query, stem=True)))
        with self._lock:
            n = len(self._doc_len)
            if not q_terms or n == 0:
                return []
            avgdl = self._total_len / n if n else 1.0
            allowed: Optional[Set[str]] = None
            if user_id is not _ANY_USER:
                allowed = {_ukey(user_id)}  # type: ignore[arg-type]
                if include_shared and user_id is not None:
                    allowed.add(_ukey(None))
            scores: Dict[int, float] = defaultdict(float)
            for term in q_terms:
                postings = self._postings.get(term)
                if not postings:
                    continue
                df = len(postings)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                for doc, tf in postings.items():
                    if doc in self._dead:
                        continue
                    if allowed is not None and self._doc_user.get(doc) not in allowed:
                        continue
                    dl = self._doc_len.get(doc, 0)
                    denom = tf + self.k1 * (1 - self.b + self.b * dl / (avgdl or 1.0))
                    scores[doc] += idf * tf * (self.k1 + 1) / denom
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return ranked[:k]


def reciprocal_rank_fusion(
    ranked_lists: Dict[str, Iterable[int]],
    weights: Optional[Dict[str, float]] = None,
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Weighted Reciprocal Rank Fusion (Cormack et al., SIGIR 2009).

    score(d) = sum_s w_s / (k + rank_s(d)). Ranks rather than raw scores are fused,
    so an unbounded signal (BM25) cannot drown out a bounded one (cosine).
    """
    weights = weights or {}
    fused: Dict[int, float] = defaultdict(float)
    for name, ids in ranked_lists.items():
        w = weights.get(name, 1.0)
        if w <= 0:
            continue
        seen: Set[int] = set()
        rank = 0
        for item_id in ids:
            if item_id in seen:
                continue
            seen.add(item_id)
            rank += 1
            fused[item_id] += w / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
