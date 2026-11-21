"""Reranker latency per query on CPU (docs/retrieval_round2.md): runtime x depth x max length.

Run on an otherwise idle machine:
  HF_HUB_OFFLINE=1 python -m benchmarks.embeddings.bench_rerank_latency \
      --models cross-encoder/ms-marco-MiniLM-L6-v2,cross-encoder/ettin-reranker-17m-v1

Each query scores its first-stage top-`depth` turns (score fusion, as in the pipeline):
LoCoMo turns are short chat lines; LongMemEval turns are often long assistant answers.
"""

import argparse
import json
import time

import numpy as np

from benchmarks.embeddings.turn_signals import LEAF, load_lme, load_locomo, rank_all, signal_scores


def candidate_sets(n_queries, depth):
    sets = {}
    for bench in (load_locomo(), load_lme(150)):
        scores = signal_scores(bench, [LEAF])
        orders = rank_all(scores, "zp", {LEAF: 1.0, "bm25": 0.5})
        rng = np.random.default_rng(0)
        pick = rng.choice(len(bench.queries), min(n_queries, len(bench.queries)), replace=False)
        sets[bench.name] = [
            [(bench.queries[q]["text"], bench.texts[bench.groups[bench.queries[q]["group"]]["cand"][i]])
             for i in orders[q][:depth]]
            for q in pick
        ]
    return sets


def time_model(model, sets):
    out = {}
    for name, queries in sets.items():
        model.predict(queries[0])
        lat = []
        for pairs in queries:
            t = time.perf_counter()
            model.predict(pairs)
            lat.append(1000 * (time.perf_counter() - t))
        out[name] = {"p50_ms": round(float(np.median(lat)), 1), "p95_ms": round(float(np.percentile(lat, 95)), 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--runtimes", default="torch,onnx,onnx-int8")
    ap.add_argument("--depths", default="10,20")
    ap.add_argument("--max-lengths", default="128,256")
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from memorable_ai.core.rerank import load_reranker

    results = []
    depths = [int(d) for d in args.depths.split(",")]
    sets_by_depth = {d: candidate_sets(args.queries, d) for d in depths}
    for repo in args.models.split(","):
        for rt in args.runtimes.split(","):
            for ml in (int(x) for x in args.max_lengths.split(",")):
                spec = f"torch:{repo}" if rt == "torch" else f"{rt}:{repo}"
                model = load_reranker(spec, max_length=ml)
                for d in depths:
                    r = {"model": repo, "runtime": rt, "max_length": ml, "depth": d,
                         **time_model(model, sets_by_depth[d])}
                    results.append(r)
                    print(json.dumps(r), flush=True)
                del model
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
