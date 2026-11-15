"""Lightweight BM25 scoring for keyword retrieval (no external deps)."""

import math
import re
from collections import Counter
from typing import Dict, List


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """In-memory BM25 index over memory documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: Dict[int, List[str]] = {}
        self._doc_lengths: Dict[int, int] = {}
        self._avg_dl = 0.0
        self._df: Counter = Counter()
        self._n = 0

    def build(self, documents: List[Dict]) -> None:
        self._docs.clear()
        self._doc_lengths.clear()
        self._df = Counter()
        self._n = len(documents)
        total_len = 0
        for doc in documents:
            doc_id = doc.get("id")
            if doc_id is None:
                continue
            tokens = _tokenize(doc.get("content", ""))
            self._docs[doc_id] = tokens
            self._doc_lengths[doc_id] = len(tokens)
            total_len += len(tokens)
            for term in set(tokens):
                self._df[term] += 1
        self._avg_dl = total_len / self._n if self._n else 0.0

    def score(self, query: str, top_k: int = 10) -> List[tuple]:
        q_terms = _tokenize(query)
        if not q_terms or not self._docs:
            return []
        scores = []
        for doc_id, tokens in self._docs.items():
            dl = self._doc_lengths.get(doc_id, 0)
            tf = Counter(tokens)
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                df = self._df.get(term, 0)
                idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avg_dl or 1))
                s += idf * (freq * (self.k1 + 1)) / denom
            if s > 0:
                scores.append((doc_id, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
