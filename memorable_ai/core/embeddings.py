"""
Shared embedding models.

One embedder instance per model name per process (the old code loaded a
SentenceTransformer per retriever and per extracted fact). Every vector is
L2-normalised float32, so cosine similarity is a dot product.

``model_name="hash"`` selects a deterministic feature-hashing embedder with no
model download. It is used for tests and as a fallback when
sentence-transformers is unavailable. It captures lexical overlap, not meaning.

Backends (docs/embedding_study.md has the measurements):

    "sentence-transformers/all-MiniLM-L6-v2"           PyTorch (sentence-transformers)
    "onnx:sentence-transformers/all-MiniLM-L6-v2"      ONNX Runtime fp32 (same vectors, faster)
    "onnx-int8:sentence-transformers/all-MiniLM-L6-v2" ONNX Runtime dynamic int8 for this CPU
    "onnx-o3:..."                                      ONNX graph-optimised (O3) export
    "static:minishlab/potion-base-8M"                  model2vec static embeddings (no transformer)

A plain model name uses ONNX Runtime automatically when ``onnxruntime`` is
installed and the repo publishes ``onnx/model.onnx``. The vectors are identical to
PyTorch's (measured cosine 1.0000) and 2-6x faster on CPU. Set
``MEMORABLE_EMBED_BACKEND=torch`` to opt out.

``MEMORABLE_TORCH_DEVICE`` (e.g. ``cpu``) pins the PyTorch models, this embedder and
the reranker's CrossEncoder, to one device. By default sentence-transformers picks
CUDA, then Apple's MPS, then CPU. On MPS a process that embeds many haystacks keeps
growing: measured 4.5 GB after one LongMemEval haystack and about 0.4 GB more per
haystack, against a flat 0.6 GB on CPU at the same speed (2026-11-20).

The ONNX path needs only ``onnxruntime`` plus the ``tokenizers`` package that
transformers already installs. It reads the ONNX files the model repo already
publishes. sentence-transformers' own ONNX extra (optimum-onnx) cannot be installed
alongside transformers 5.x / Python 3.14.
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


def _onnx_file(variant: str) -> str:
    """Pick the published ONNX export for a variant and this CPU."""
    if variant == "onnx-o3":
        return "onnx/model_O3.onnx"
    if variant != "onnx-int8":
        return "onnx/model.onnx"
    import platform

    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "onnx/model_qint8_arm64.onnx"
    try:  # x86: prefer VNNI, then AVX-512, then AVX2
        flags = open("/proc/cpuinfo").read()
    except Exception:
        flags = ""
    if "avx512_vnni" in flags:
        return "onnx/model_qint8_avx512_vnni.onnx"
    if "avx512" in flags:
        return "onnx/model_qint8_avx512.onnx"
    return "onnx/model_quint8_avx2.onnx"


class OnnxEncoder:
    """
    Sentence-transformers-compatible encoder on ONNX Runtime: HF ``tokenizers`` +
    an exported transformer + mean pooling + L2 norm, all in NumPy.

    Inputs are sorted by length and padded per batch, so short texts don't pay for
    long ones. Threads default to ONNX Runtime's choice; ``MEMORABLE_ORT_THREADS``
    overrides that.
    """

    def __init__(
        self,
        repo: str,
        variant: str = "onnx",
        max_length: Optional[int] = None,
        providers: Optional[Sequence[str]] = None,
        threads: Optional[int] = None,
    ):
        import json

        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.repo, self.variant = repo, variant
        self.file_name = _onnx_file(variant)
        # Only reproduce module stacks we understand exactly; anything else raises so
        # load_encoder falls back to PyTorch instead of silently returning wrong vectors.
        self.dense: List[tuple] = []
        try:
            modules = json.load(open(hf_hub_download(repo, "modules.json")))
        except Exception:
            modules = []
        for mod in modules:
            kind = str(mod.get("type", "")).rsplit(".", 1)[-1]
            if kind in ("Transformer", "Pooling", "Normalize"):
                continue
            if kind == "Dense":
                self.dense.append(self._load_dense(repo, mod.get("path", "")))
                continue
            raise ValueError(f"unsupported sentence-transformers module for ONNX: {kind}")
        model_path = self._model_path(repo, self.file_name)
        try:
            cfg = json.load(open(hf_hub_download(repo, "sentence_bert_config.json")))
            default_len = int(cfg.get("max_seq_length") or 256)
        except Exception:
            default_len = 256
        try:
            pool = json.load(open(hf_hub_download(repo, "1_Pooling/config.json")))
        except Exception:
            pool = {"pooling_mode_mean_tokens": True}
        if pool.get("pooling_mode_cls_token"):
            self.pooling = "cls"
        elif pool.get("pooling_mode_mean_tokens"):
            self.pooling = "mean"
        else:
            raise ValueError(f"unsupported pooling for ONNX: {pool}")
        try:  # sentence-transformers prompts are plain prefixes ("query: ", ...)
            st_cfg = json.load(open(hf_hub_download(repo, "config_sentence_transformers.json")))
            prompts = dict(st_cfg.get("prompts") or {})
        except Exception:
            prompts = {}
        self.max_length = int(max_length or os.getenv("MEMORABLE_EMBED_MAX_LENGTH") or default_len)
        self.tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.no_padding()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 4 intra-op threads measured best on a 10-core Apple M-series (fewer contend
        # less with efficiency cores); override with MEMORABLE_ORT_THREADS.
        threads = threads or int(os.getenv("MEMORABLE_ORT_THREADS") or min(4, os.cpu_count() or 4))
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path, so, providers=list(providers or ["CPUExecutionProvider"]))
        self._inputs = {i.name for i in self.session.get_inputs()}
        self.dim = int(self.dense[-1][0].shape[0]) if self.dense else int(self.session.get_outputs()[0].shape[-1])
        self.prompts: Dict[str, str] = prompts

    @staticmethod
    def _model_path(repo: str, file_name: str) -> str:
        """
        Path to the ONNX graph. Exports with a sidecar ``*.onnx_data`` weights file are
        materialised as real files: ONNX Runtime rejects external data reached through
        the Hugging Face cache's symlinks ("path escapes model directory").
        """
        from huggingface_hub import HfApi, hf_hub_download

        try:
            has_data = HfApi().file_exists(repo, file_name + "_data")
        except Exception:
            has_data = False
        if not has_data:
            return hf_hub_download(repo, file_name)
        local = os.path.join(os.path.expanduser("~/.cache/memorable/onnx"), repo.replace("/", "--"))
        for f in (file_name, file_name + "_data"):
            hf_hub_download(repo, f, local_dir=local)
        return os.path.join(local, file_name)

    @staticmethod
    def _load_dense(repo: str, path: str) -> tuple:
        import json

        from huggingface_hub import hf_hub_download
        from safetensors.numpy import load_file

        cfg = json.load(open(hf_hub_download(repo, f"{path}/config.json")))
        act = str(cfg.get("activation_function", "")).rsplit(".", 1)[-1]
        if act not in ("Identity", "Tanh"):
            raise ValueError(f"unsupported Dense activation for ONNX: {act}")
        w = load_file(hf_hub_download(repo, f"{path}/model.safetensors"))
        weight = w.get("linear.weight")
        bias = w.get("linear.bias")
        if weight is None:
            raise ValueError("Dense weights not found")
        return weight.astype(np.float32), (bias.astype(np.float32) if bias is not None else None), act

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self, sentences: Sequence[str], batch_size: int = 64, prompt_name: Optional[str] = None, **_: object
    ) -> np.ndarray:
        texts = [sentences] if isinstance(sentences, str) else list(sentences)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self.prompts.get(prompt_name, "") if prompt_name else ""
        if prefix:
            texts = [prefix + t for t in texts]
        encs = self.tokenizer.encode_batch(texts)
        order = np.argsort([len(e.ids) for e in encs], kind="stable")
        hidden_dim = int(self.dense[0][0].shape[1]) if self.dense else self.dim
        out = np.empty((len(texts), hidden_dim), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            idx = order[start : start + batch_size]
            width = max(len(encs[i].ids) for i in idx)
            ids = np.zeros((len(idx), width), dtype=np.int64)
            mask = np.zeros((len(idx), width), dtype=np.int64)
            types = np.zeros((len(idx), width), dtype=np.int64)
            for r, i in enumerate(idx):
                e = encs[i]
                n = len(e.ids)
                ids[r, :n] = e.ids
                mask[r, :n] = 1
                types[r, :n] = e.type_ids
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = types
            hidden = self.session.run(None, feed)[0]
            if self.pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                m = mask[..., None].astype(np.float32)
                pooled = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
            out[idx] = pooled
        for weight, bias, act in self.dense:
            out = out @ weight.T
            if bias is not None:
                out = out + bias
            if act == "Tanh":
                out = np.tanh(out)
        return _normalize_rows(out)


class StaticEncoder:
    """model2vec static embeddings: a token-embedding lookup + mean, no transformer."""

    def __init__(self, repo: str):
        from model2vec import StaticModel

        self.repo = repo
        self.model = StaticModel.from_pretrained(repo)
        self.dim = int(self.model.dim)
        self.prompts: Dict[str, str] = {}

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, sentences: Sequence[str], batch_size: int = 1024, **_: object) -> np.ndarray:
        texts = [sentences] if isinstance(sentences, str) else list(sentences)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize_rows(np.asarray(self.model.encode(texts, batch_size=batch_size), dtype=np.float32))


_BACKEND_PREFIXES = ("onnx", "onnx-int8", "onnx-o3", "torch")


def space_of(spec: Optional[str]) -> str:
    """
    The vector space a backend spec writes into. PyTorch, ONNX and O3 exports of
    one model give identical vectors; int8 is within cosine 0.9965 (measured), so
    they all share the model's space. A static model or another model is a new space.
    """
    if not spec:
        return ""
    head, _, rest = spec.partition(":")
    if rest and head in _BACKEND_PREFIXES:
        return rest
    return spec


def torch_device() -> Optional[str]:
    """``MEMORABLE_TORCH_DEVICE`` for the PyTorch models, or None for sentence-transformers' own choice."""
    return os.getenv("MEMORABLE_TORCH_DEVICE") or None


def load_encoder(spec: str):
    """Build the model object for a backend spec (see module docstring)."""
    if ":" in spec and spec.split(":", 1)[0] in ("onnx", "onnx-int8", "onnx-o3"):
        variant, repo = spec.split(":", 1)
        return OnnxEncoder(repo, variant=variant)
    if spec.startswith("static:"):
        return StaticEncoder(spec.split(":", 1)[1])
    if spec.startswith("torch:"):
        spec = spec.split(":", 1)[1]
    elif os.getenv("MEMORABLE_EMBED_BACKEND", "auto").lower() == "auto" and "/" in spec:
        try:  # same vectors, 2-6x faster on CPU; falls back silently
            import onnxruntime  # noqa: F401

            return OnnxEncoder(spec, variant="onnx")
        except Exception as e:
            logger.debug("ONNX backend unavailable for %s (%s); using PyTorch", spec, e)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec, device=torch_device())
    if os.getenv("MEMORABLE_EMBED_MAX_LENGTH"):
        model.max_seq_length = int(os.environ["MEMORABLE_EMBED_MAX_LENGTH"])
    return model


class Embedder:
    """Thread-safe wrapper with a uniform ``encode`` API."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv(
            "MEMORABLE_LLM__EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self._lock = threading.Lock()
        self._model = None
        self._hash: Optional[HashingEmbedder] = None
        self._init_query_cache()
        if self.model_name.lower() in HASH_MODEL_NAMES:
            self._hash = HashingEmbedder()
        else:
            try:
                self._model = load_encoder(self.model_name)
                logger.info("Loaded embedding model %s", self.model_name)
            except Exception as e:  # pragma: no cover - depends on environment
                logger.warning(
                    "Embedding model %s unavailable (%s); falling back to hashing embedder",
                    self.model_name,
                    e,
                )
                self._hash = HashingEmbedder()

    def _init_query_cache(self) -> None:
        # Agent tool loops resend the same user message on every LLM call, so query
        # embeddings repeat constantly; a small LRU skips the model (~6 ms/query on CPU).
        self._query_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.query_cache_size = int(os.getenv("MEMORABLE_QUERY_EMBED_CACHE", "2048"))
        self.query_cache_hits = 0

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
        wrapper._init_query_cache()
        return wrapper
    return None
