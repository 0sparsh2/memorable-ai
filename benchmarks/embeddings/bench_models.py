"""Compare embedding *models* on both memory benchmarks (docs/embedding_study.md).

Run: HF_HUB_OFFLINE=1 python -m benchmarks.embeddings.bench_models --models a,b --out results.json

For each model spec (anything ``load_encoder`` accepts):
  * LoCoMo dense evidence recall@5 over each conversation's turns, raw and with
    the pipeline's contextual blend
  * LongMemEval-S turn recall@10 and session recall@5 (150 stratified questions)
  * CPU encode throughput on LoCoMo turns and single-query latency
Query prompts are used when the model defines one (asymmetric retrievers).
"""

import argparse
import json
import time

import numpy as np

from benchmarks.embeddings.bench_backends import load_locomo, recall_at_k
from benchmarks.embeddings.bench_longmemeval import evaluate as lme_evaluate
from benchmarks.embeddings.bench_longmemeval import sample_questions
from memorable_ai.core.embeddings import load_encoder


def run(spec, convs, lme_qs):
    t = time.perf_counter()
    m = load_encoder(spec)
    load_s = time.perf_counter() - t
    prompts = getattr(m, "prompts", None) or {}

    def enc(texts, query=False):
        kw = {"prompt_name": "query"} if (query and "query" in prompts) else {}
        v = np.asarray(m.encode(texts, batch_size=64, show_progress_bar=False, **kw), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    turns = [t for c in convs for t in c["turns"]]
    qs = [q for c in convs for q, _, _ in c["qs"]]
    enc(turns[:64])
    t = time.perf_counter()
    tv = enc(turns)
    bulk_ms = 1000 * (time.perf_counter() - t) / len(turns)
    single = []
    for q in qs[:100]:
        t = time.perf_counter()
        enc([q], query=True)
        single.append(1000 * (time.perf_counter() - t))
    qv = enc(qs, query=True)
    per, off = [], 0
    for c in convs:
        per.append(tv[off : off + len(c["turns"])])
        off += len(c["turns"])
    raw, _ = recall_at_k(convs, per, qv)
    blend, _ = recall_at_k(convs, per, qv, blend=0.5)
    out = {"model": spec, "query_prompt": "query" in prompts, "load_s": round(load_s, 1),
           "bulk_ms_per_text": round(bulk_ms, 3), "single_query_ms_p50": round(float(np.median(single)), 2),
           "locomo_recall@5": raw["overall"], "locomo_recall@5_blend": blend["overall"],
           "locomo_by_cat_blend": blend}
    lme = lme_evaluate(spec, lme_qs)
    out.update({"lme_turn@10": lme["turn@10"], "lme_session@5": lme["session@5"], "lme_encode_s": lme["encode_s"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--questions", type=int, default=150)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    convs = load_locomo()
    lme_qs = sample_questions(args.questions)
    results = []
    for spec in args.models.split(","):
        try:
            r = run(spec.strip(), convs, lme_qs)
        except Exception as e:
            r = {"model": spec, "error": repr(e)[:300]}
        results.append(r)
        print(json.dumps({k: (v.get("overall") if isinstance(v, dict) and "overall" in v else v)
                          for k, v in r.items() if k != "locomo_by_cat_blend"}), flush=True)
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
