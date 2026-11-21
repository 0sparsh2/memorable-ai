"""Embedding backend study: speed and retrieval quality on LoCoMo (docs/embedding_study.md).

Run: HF_HUB_OFFLINE=1 python benchmarks/embeddings/bench_backends.py [--only onnx-int8,torch-cpu]

For every backend it measures:
  * load time
  * bulk throughput: all 5,882 LoCoMo turns, batch 64
  * today's ingest call pattern: 2 texts per call
  * single-query latency p50/p95 (100 questions, one at a time)
  * quality: dense evidence recall@5 over each conversation's turns (the same hit
    rule as retrieval_eval), raw and with the pipeline's contextual blend (0.5)
  * agreement with the PyTorch fp32 baseline: mean cosine of turn vectors and
    top-5 overlap (same model family only)
"""

import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from benchmarks.locomo.loader import format_turn_text, iter_sessions  # noqa: E402

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
DATA = os.path.join(os.path.dirname(__file__), "..", "locomo", "data", "locomo10.json")


def load_locomo():
    data = json.load(open(DATA))
    convs = []
    for sample in data:
        conv = sample["conversation"]
        turns, dia, session_of = [], [], []
        for skey, _date, ts in iter_sessions(conv):
            for t in ts:
                turns.append(f"{t.get('speaker')}: {format_turn_text(t)}")
                dia.append(t.get("dia_id"))
                session_of.append(skey)
        qs = [(q["question"], set(q.get("evidence") or []), q.get("category"))
              for q in sample["qa"] if q.get("category") in (1, 2, 3, 4) and q.get("evidence")]
        convs.append({"turns": turns, "dia": dia, "session": session_of, "qs": qs})
    return convs


def make(name):
    """Return (encode_fn(list)->np.ndarray, cleanup_fn)."""
    import torch

    default_threads = torch.get_num_threads()
    from memorable_ai.core.embeddings import OnnxEncoder, StaticEncoder

    if name.startswith("torch"):
        from sentence_transformers import SentenceTransformer

        device = "mps" if "mps" in name else "cpu"
        m = SentenceTransformer(MINILM, device=device)
        if "len128" in name:
            m.max_seq_length = 128
        if "-t" in name:
            torch.set_num_threads(int(name.split("-t")[-1]))

        def enc(texts, bs=64):
            return np.asarray(m.encode(texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True),
                              dtype=np.float32)

        return enc, lambda: torch.set_num_threads(default_threads)
    if name.startswith("onnx"):
        variant = "onnx-int8" if "int8" in name else ("onnx-o3" if "o3" in name else "onnx")
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"] if "coreml" in name else None
        threads = int(name.split("-t")[-1]) if "-t" in name else None
        e = OnnxEncoder(MINILM, variant=variant, max_length=128 if "len128" in name else None,
                        providers=providers, threads=threads)
        return (lambda texts, bs=64: e.encode(texts, batch_size=bs)), (lambda: None)
    if name.startswith("static"):
        repo = {"static-potion-8M": "minishlab/potion-base-8M",
                "static-potion-retrieval-32M": "minishlab/potion-retrieval-32M"}[name]
        e = StaticEncoder(repo)
        return (lambda texts, bs=1024: e.encode(texts, batch_size=bs)), (lambda: None)
    raise ValueError(name)


def recall_at_k(convs, turn_vecs, q_vecs, k=5, blend=0.0):
    hits = defaultdict(lambda: [0, 0])
    tops = []
    qi = 0
    for c, tv in zip(convs, turn_vecs):
        if blend:
            tv = tv.copy()
            prev = None
            for i in range(len(tv)):
                v = tv[i].copy()
                if prev is not None and c["session"][i] == c["session"][i - 1]:
                    b = v + blend * prev
                    tv[i] = b / (np.linalg.norm(b) or 1.0)
                prev = v
        for q, gold, cat in c["qs"]:
            sims = tv @ q_vecs[qi]
            top = np.argpartition(-sims, k)[:k]
            top = top[np.argsort(-sims[top])]
            tops.append(top)
            hit = bool(gold & {c["dia"][i] for i in top})
            hits["overall"][0] += hit
            hits["overall"][1] += 1
            hits[str(cat)][0] += hit
            hits[str(cat)][1] += 1
            qi += 1
    return {kk: round(h / n, 4) for kk, (h, n) in hits.items()}, tops


def run(name, convs, baseline=None):
    t = time.perf_counter()
    enc, cleanup = make(name)
    load_s = time.perf_counter() - t
    all_turns = [t for c in convs for t in c["turns"]]
    all_qs = [q for c in convs for q, _, _ in c["qs"]]
    enc(all_turns[:64])  # warm-up (JIT / graph compile / caches)
    subset = "coreml" in name
    bulk_texts = all_turns[:600] if subset else all_turns
    t = time.perf_counter()
    bulk = enc(bulk_texts)
    bulk_ms = 1000 * (time.perf_counter() - t) / len(bulk_texts)
    t = time.perf_counter()
    for i in range(0, 200, 2):
        enc(all_turns[i : i + 2])
    pair_ms = 1000 * (time.perf_counter() - t) / 200
    single = []
    for q in all_qs[:100]:
        t = time.perf_counter()
        enc([q])
        single.append(1000 * (time.perf_counter() - t))
    res = {"backend": name, "load_s": round(load_s, 2), "bulk_ms_per_text": round(bulk_ms, 3),
           "pair_call_ms_per_text": round(pair_ms, 3),
           "single_query_ms_p50": round(float(np.median(single)), 2),
           "single_query_ms_p95": round(float(np.percentile(single, 95)), 2)}
    if subset:
        bulk = enc(all_turns)
    q_vecs = enc(all_qs)
    turn_vecs, off = [], 0
    for c in convs:
        turn_vecs.append(bulk[off : off + len(c["turns"])])
        off += len(c["turns"])
    raw, tops = recall_at_k(convs, turn_vecs, q_vecs)
    blended, _ = recall_at_k(convs, turn_vecs, q_vecs, blend=0.5)
    res["recall@5"] = raw
    res["recall@5_blend0.5"] = blended
    if baseline is not None and baseline["vecs"].shape == bulk.shape:
        res["cos_vs_torch_mean"] = round(float(np.mean(np.sum(baseline["vecs"] * bulk, axis=1))), 5)
        res["cos_vs_torch_min"] = round(float(np.min(np.sum(baseline["vecs"] * bulk, axis=1))), 5)
        res["top5_overlap_vs_torch"] = round(float(np.mean(
            [len(set(a) & set(b)) / 5 for a, b in zip(baseline["tops"], tops)])), 4)
    cleanup()
    return res, bulk, tops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), f"results_{platform.machine()}.json"))
    args = ap.parse_args()
    names = ["torch-cpu", "torch-cpu-t8", "torch-cpu-len128", "torch-mps",
             "onnx", "onnx-o3", "onnx-int8", "onnx-int8-len128", "onnx-int8-t4", "onnx-coreml",
             "static-potion-8M", "static-potion-retrieval-32M"]
    if args.only:
        names = ["torch-cpu"] + [n for n in args.only.split(",") if n != "torch-cpu"]
    convs = load_locomo()
    print(f"LoCoMo: {sum(len(c['turns']) for c in convs)} turns, {sum(len(c['qs']) for c in convs)} questions")
    results, baseline = [], None
    for name in names:
        try:
            res, vecs, tops = run(name, convs, baseline)
        except Exception as e:  # keep going; record the failure
            res = {"backend": name, "error": repr(e)[:300]}
            vecs = tops = None
        if name == "torch-cpu" and vecs is not None:
            baseline = {"vecs": vecs, "tops": tops}
        results.append(res)
        print(json.dumps(res))
    json.dump({"machine": platform.platform(), "python": sys.version.split()[0], "results": results},
              open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
