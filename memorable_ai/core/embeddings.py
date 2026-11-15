"""
Shared embedding models.

One embedder instance per model name per process (the old code loaded a
SentenceTransformer per retriever and per extracted fact). Every vector is
L2-normalised float32, so cosine similarity is a dot product.

``model_name="hash"`` selects a deterministic feature-hashing embedder with no
model download. It is used for tests and as a fallback when
sentence-transformers is unavailable. It captures lexical overlap, not meaning.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import zlib
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HASH_MODEL_NAMES = {"hash", "hashing", "hash-384"}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing over word unigrams, bigrams and character 4-grams."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _features(self, text: str) -> Iterable[str]:
        words = _TOKEN_RE.findall(text.lower())
        for w in words:
            yield "w:" + w
            # Light stemming so "likes"/"like" and "lives"/"live" collide.
            if len(w) > 4 and w.endswith("s"):
                yield "w:" + w[:-1]
            padded = f"#{w}#"
            for i in range(max(1, len(padded) - 3)):
                yield "c:" + padded[i : i + 4]
        for a, b in zip(words, words[1:]):
            yield f"b:{a}_{b}"

    def encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat in self._features(text or ""):
            h = zlib.crc32(feat.encode("utf-8"))
            idx = h % self.dim
            sign = 1.0 if (h >> 31) & 1 else -1.0
            weight = 1.0 if feat.startswith("w:") else (0.8 if feat.startswith("b:") else 0.3)
            vec[idx] += sign * weight
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec


class Embedder:
    """Thread-safe wrapper with a uniform ``encode`` API."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv(
            "MEMORABLE_LLM__EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self._lock = threading.Lock()
        self._model = None
        self._hash: Optional[HashingEmbedder] = None
        # Agent tool loops resend the same user message on every LLM call, so query
        # embeddings repeat constantly; a small LRU skips the model (~6 ms/query on CPU).
        self._query_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.query_cache_size = int(os.getenv("MEMORABLE_QUERY_EMBED_CACHE", "2048"))
        self.query_cache_hits = 0
        if self.model_name.lower() in HASH_MODEL_NAMES:
            self._hash = HashingEmbedder()
        else:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                logger.info("Loaded embedding model %s", self.model_name)
            except Exception as e:  # pragma: no cover - depends on environment
                logger.warning(
                    "Embedding model %s unavailable (%s); falling back to hashing embedder",
                    self.model_name,
                    e,
                )
                self._hash = HashingEmbedder()

    @property
    def is_semantic(self) -> bool:
        """True when backed by a neural model rather than feature hashing."""
        return self._model is not None

    @property
    def dim(self) -> int:
        if self._hash is not None:
            return self._hash.dim
        get_dim = getattr(self._model, "get_sentence_embedding_dimension", None)
        return int(get_dim()) if get_dim else 384

    def encode(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        """Encode one string (returns 1-D) or a list (returns 2-D), L2-normalised."""
        single = isinstance(texts, str)
        batch: List[str] = [texts] if single else list(texts)  # type: ignore[list-item]
        if not batch:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._hash is not None:
            mat = np.stack([self._hash.encode_one(t) for t in batch])
        else:
            with self._lock:
                mat = np.asarray(
                    self._model.encode(batch, show_progress_bar=False), dtype=np.float32
                )
            mat = _normalize_rows(mat)
        return mat[0] if single else mat

    def encode_query(self, text: str) -> np.ndarray:
        """
        Encode a search query. Asymmetric retrieval models (EmbeddingGemma, BGE, E5,
        Nomic) define a "query" prompt; documents are encoded without it.
        """
        cached = self._query_cache.get(text) if self.query_cache_size else None
        if cached is not None:
            with self._lock:
                self._query_cache.move_to_end(text)
                self.query_cache_hits += 1
            return cached.copy()
        prompts = getattr(self._model, "prompts", None) or {}
        if self._model is not None and "query" in prompts:
            with self._lock:
                vec = np.asarray(
                    self._model.encode([text], prompt_name="query", show_progress_bar=False), dtype=np.float32
                )
            out = _normalize_rows(vec)[0]
        else:
            out = self.encode(text)
        if self.query_cache_size:
            with self._lock:
                self._query_cache[text] = out.copy()
                if len(self._query_cache) > self.query_cache_size:
                    self._query_cache.popitem(last=False)
        return out

    def encode_list(self, text: str) -> List[float]:
        """Encode a single string to a plain list (for JSON storage)."""
        return self.encode(text).tolist()


_REGISTRY: Dict[str, Embedder] = {}
_REGISTRY_LOCK = threading.Lock()


def get_embedder(model_name: Optional[str] = None) -> Embedder:
    """Process-wide cached embedder for ``model_name``."""
    name = model_name or os.getenv("MEMORABLE_LLM__EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    with _REGISTRY_LOCK:
        emb = _REGISTRY.get(name)
        if emb is None:
            emb = Embedder(name)
            _REGISTRY[name] = emb
        return emb


def as_embedder(obj: object) -> Optional[Embedder]:
    """Adapt a model name, an ``Embedder`` or a raw SentenceTransformer to ``Embedder``."""
    if obj is None:
        return None
    if isinstance(obj, Embedder):
        return obj
    if isinstance(obj, str):
        return get_embedder(obj)
    if hasattr(obj, "encode"):
        wrapper = Embedder.__new__(Embedder)
        wrapper.model_name = getattr(obj, "model_name", "custom")
        wrapper._lock = threading.Lock()
        wrapper._model = obj
        wrapper._hash = None
        return wrapper
    return None
