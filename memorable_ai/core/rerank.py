"""
Cross-encoder rerankers for the retrieval layers.

A reranker reads (query, candidate) together, so it catches what a bi-encoder's single
vector cannot (who did what, negation, the question's focus). On LoCoMo turns it adds
6 to 12 points of recall@5 on top of score fusion (docs/retrieval_round2.md).

Specs, like the embedder specs in ``embeddings.load_encoder``:
    "cross-encoder/ms-marco-MiniLM-L6-v2"        ONNX Runtime when available, else PyTorch
    "onnx-int8:cross-encoder/ettin-reranker-17m-v1"  int8 export for this CPU
    "torch:/path/to/finetuned"                    sentence-transformers CrossEncoder
Scores are raw logits (higher = more relevant); only their order and spread matter.
Pairs are truncated to 128 tokens by default: no measured quality loss, half the time
on long turns.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Sequence, Tuple

import numpy as np

from memorable_ai.core.embeddings import OnnxEncoder, _onnx_file

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorised erf (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7), for exact GELU without scipy."""
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(
        -a * a
    )
    return s * y


_ACTIVATIONS = {
    "Identity": lambda x: x,
    "Tanh": np.tanh,
    "GELU": lambda x: 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0))),
}


def _load_head(repo: str) -> list:
    """
    The scoring head of a modular sentence-transformers CrossEncoder (Transformer ->
    Pooling -> Dense / LayerNorm ... -> score), e.g. the Ettin rerankers, whose ONNX
    export stops at the token embeddings. Unknown modules raise, so load_reranker falls
    back to PyTorch instead of returning wrong scores.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    head = []
    for mod in json.load(open(hf_hub_download(repo, "modules.json"))):
        kind = str(mod.get("type", "")).rsplit(".", 1)[-1]
        path = mod.get("path", "")
        if kind == "Transformer":
            continue
        cfg = json.load(open(hf_hub_download(repo, f"{path}/config.json"))) if path else {}
        if kind == "Pooling":
            mode = cfg.get("pooling_mode") or ("cls" if cfg.get("pooling_mode_cls_token") else "mean")
            if mode not in ("cls", "mean"):
                raise ValueError(f"unsupported pooling for ONNX reranker: {mode}")
            head.append(("pool", mode))
        elif kind == "Dense":
            act = str(cfg.get("activation_function", "")).rsplit(".", 1)[-1]
            if act not in _ACTIVATIONS:
                raise ValueError(f"unsupported Dense activation for ONNX reranker: {act}")
            w = load_file(hf_hub_download(repo, f"{path}/model.safetensors"))
            b = w.get("linear.bias")
            head.append(("dense", w["linear.weight"].astype(np.float32),
                         None if b is None else b.astype(np.float32), act))
        elif kind == "LayerNorm":
            w = load_file(hf_hub_download(repo, f"{path}/model.safetensors"))
            head.append(("norm", w["norm.weight"].astype(np.float32), w["norm.bias"].astype(np.float32)))
        else:
            raise ValueError(f"unsupported CrossEncoder module for ONNX: {kind}")
    return head


class OnnxCrossEncoder:
    """
    Cross-encoder on ONNX Runtime: HF ``tokenizers`` pair encoding + the exported model.
    Classic exports (BERT/MiniLM ``ForSequenceClassification``) output the logit
    directly; modular ones output token embeddings and the head runs in NumPy.
    """

    def __init__(self, repo: str, variant: str = "onnx", max_length: int = 128, threads: Optional[int] = None):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.repo, self.variant = repo, variant
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
        if int(cfg.get("num_labels") or len(cfg.get("id2label") or {0: 0})) != 1:
            raise ValueError("only single-logit rerankers are supported on ONNX")
        self.max_length = int(max_length)
        self.tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_length, strategy="longest_first")
        self.tokenizer.no_padding()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = threads or int(os.getenv("MEMORABLE_ORT_THREADS") or min(4, os.cpu_count() or 4))
        path = OnnxEncoder._model_path(repo, _onnx_file(variant))
        self.session = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self.session.get_inputs()}
        hidden = len(self.session.get_outputs()[0].shape or []) == 3
        self.head = _load_head(repo) if hidden else []
        if hidden and not any(step[0] == "pool" for step in self.head):
            raise ValueError("ONNX export outputs token embeddings but the repo has no pooling module")

    def _apply_head(self, hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x = hidden
        for step in self.head:
            if step[0] == "pool":
                if step[1] == "cls":
                    x = x[:, 0, :]
                else:
                    m = mask[..., None].astype(np.float32)
                    x = (x * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
            elif step[0] == "dense":
                _, w, b, act = step
                x = x @ w.T
                if b is not None:
                    x = x + b
                x = _ACTIVATIONS[act](x)
            else:  # LayerNorm, torch default eps
                _, w, b = step
                mu = x.mean(axis=-1, keepdims=True)
                var = x.var(axis=-1, keepdims=True)
                x = (x - mu) / np.sqrt(var + 1e-5) * w + b
        return x

    def predict(self, pairs: Sequence[Tuple[str, str]], batch_size: int = 32, **_: object) -> np.ndarray:
        pairs = list(pairs)
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        encs = self.tokenizer.encode_batch([(q, d) for q, d in pairs])
        order = np.argsort([len(e.ids) for e in encs], kind="stable")
        out = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), batch_size):
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
            res = self.session.run(None, feed)[0]
            if self.head:
                res = self._apply_head(np.asarray(res, dtype=np.float32), mask)
            out[idx] = np.asarray(res, dtype=np.float32).reshape(len(idx), -1)[:, -1]
        return out


def load_reranker(spec: Optional[str] = None, max_length: int = 128):
    """Build a reranker with a ``predict(pairs) -> scores`` method (see module docstring)."""
    spec = spec or DEFAULT_RERANK_MODEL
    head, _, rest = spec.partition(":")
    if rest and head in ("onnx", "onnx-int8", "onnx-o3"):
        return OnnxCrossEncoder(rest, variant=head, max_length=max_length)
    if head == "torch" and rest:
        spec = rest
    elif "/" in spec and not os.path.isdir(spec):
        try:
            return OnnxCrossEncoder(spec, max_length=max_length)
        except Exception as e:  # no ONNX export / onnxruntime missing: same scores via PyTorch
            logger.debug("ONNX reranker unavailable for %s (%s); using PyTorch", spec, e)
    from sentence_transformers import CrossEncoder

    return CrossEncoder(spec, max_length=max_length)
