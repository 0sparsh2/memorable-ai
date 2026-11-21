"""
Base vs fine-tuned models against the preregistered bars (docs/retrieval_round2.md §0).

  python -m benchmarks.embeddings.ft_compare --models-dir training/data/ft_v1/models --out ft_compare.json

Same harness as turn_signals.py (contextual blend, BM25, top-50 score fusion). Every
fine-tuned model is compared with its base on identical candidate sets, with 95% paired
bootstrap CIs:
  turn model    leaf-ir vs leafir-ft: dense, and score fusion with BM25 (quality profile)
  static model  potion-32M vs potion-ft: score fusion with BM25 (fast profile)
  semantic      MiniLM vs minilm-ft: dense turn retrieval (the episode bar is end to end)
  rerankers     base vs fine-tuned on the same first stage, top 20, 128 tokens, pure and blend
"""

import argparse
import json
import time

import numpy as np

from benchmarks.embeddings.turn_signals import (LEAF, MINILM, POTION, hits, load_lme, load_locomo, paired_ci,
                                                rank_all, signal_scores, summarize)

RERANKERS = {"ettin17": "cross-encoder/ettin-reranker-17m-v1", "minilm-ce": "cross-encoder/ms-marco-MiniLM-L6-v2",
             "ettin32": "cross-encoder/ettin-reranker-32m-v1"}
FT_RERANKERS = {"ettin17": "ce_ettin17", "minilm-ce": "ce_minilm", "ettin32": "ce_ettin32"}


def rerank(bench, orders, fused, model, depth=20, blend=False):
    out, lat = [], []
    for q, order, f in zip(bench.queries, orders, fused):
        top = order[:depth]
        cand = bench.groups[q["group"]]["cand"]
        t = time.perf_counter()
        s = np.asarray(model.predict([(q["text"], bench.texts[cand[i]]) for i in top], batch_size=depth,
                                     show_progress_bar=False), dtype=np.float64).reshape(len(top), -1)[:, -1]
        lat.append(1000 * (time.perf_counter() - t))
        if blend:
            z = lambda v: (v - v.mean()) / (v.std() or 1.0)  # noqa: E731
            s = z(s) + z(f[top])
        out.append(np.concatenate([top[np.argsort(-s, kind="stable")], order[depth:]]))
    return out, float(np.median(lat))


def main():
    from sentence_transformers import CrossEncoder

    from benchmarks.embeddings.turn_signals import zpool

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--questions", type=int, default=150)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    ft = lambda name: f"torch:{args.models_dir.rstrip('/')}/{name}"  # noqa: E731
    LEAF_FT, POTION_FT, POTION_FT01, MINILM_FT = ft("leafir"), ft("potion_lr05"), ft("potion_lr01"), ft("minilm")
    report = {}
    for bench in (load_locomo(), load_lme(args.questions)):
        key = "turn@5" if bench.name == "locomo" else "turn@10"
        sc = signal_scores(bench, [LEAF, LEAF_FT, POTION, POTION_FT, POTION_FT01, MINILM, MINILM_FT])
        cfg = {
            "dense leaf-ir": ("rrf", {LEAF: 1.0}), "dense leaf-ir-ft": ("rrf", {LEAF_FT: 1.0}),
            "quality zp(leaf-ir, bm25)": ("zp", {LEAF: 1.0, "bm25": 0.5}),
            "quality zp(leaf-ir-ft, bm25)": ("zp", {LEAF_FT: 1.0, "bm25": 0.5}),
            "fast zp(potion, bm25)": ("zp", {POTION: 1.0, "bm25": 0.5}),
            "fast zp(potion-ft lr.05, bm25)": ("zp", {POTION_FT: 1.0, "bm25": 0.5}),
            "fast zp(potion-ft lr.01, bm25)": ("zp", {POTION_FT01: 1.0, "bm25": 0.5}),
            "dense minilm": ("rrf", {MINILM: 1.0}), "dense minilm-ft": ("rrf", {MINILM_FT: 1.0}),
        }
        H, rep = {}, {}
        for name, (rule, w) in cfg.items():
            H[name] = hits(bench, rank_all(sc, rule, w))
            rep[name] = summarize(bench, H[name])
        pairs = [("dense leaf-ir", "dense leaf-ir-ft"), ("quality zp(leaf-ir, bm25)", "quality zp(leaf-ir-ft, bm25)"),
                 ("fast zp(potion, bm25)", "fast zp(potion-ft lr.05, bm25)"),
                 ("fast zp(potion, bm25)", "fast zp(potion-ft lr.01, bm25)"), ("dense minilm", "dense minilm-ft")]
        for base, new in pairs:
            rep[new]["delta_vs_base"] = {m: paired_ci(H[base][m], H[new][m]) for m in H[new]}
        # rerankers on both first stages
        for s1_name, s1_w in (("zp(leaf-ir)", {LEAF: 1.0, "bm25": 0.5}), ("zp(leaf-ir-ft)", {LEAF_FT: 1.0, "bm25": 0.5})):
            fused = [zpool({k: sc[k][i] for k in s1_w}, s1_w) for i in range(len(bench.queries))]
            orders = [np.argsort(-f, kind="stable") for f in fused]
            for short, base_spec in RERANKERS.items():
                for tag, spec in (("base", base_spec), ("ft", f"{args.models_dir.rstrip('/')}/{FT_RERANKERS[short]}")):
                    model = CrossEncoder(spec, max_length=128)
                    for blend in (False, True):
                        ranked, ms = rerank(bench, orders, fused, model, blend=blend)
                        name = f"{s1_name} > {short}-{tag} {'blend' if blend else 'pure'}"
                        H[name] = hits(bench, ranked)
                        rep[name] = {**summarize(bench, H[name]), "rerank_ms_p50": round(ms, 1)}
                    del model
                for mode in ("pure", "blend"):
                    b, n = f"{s1_name} > {short}-base {mode}", f"{s1_name} > {short}-ft {mode}"
                    rep[n]["delta_vs_base"] = {m: paired_ci(H[b][m], H[n][m]) for m in H[n]}
        report[bench.name] = rep
        print(f"\n== {bench.name} ({len(bench.queries)} q)", flush=True)
        for name, r in rep.items():
            d = r.get("delta_vs_base", {}).get(key)
            extra = f" s@5 {r['session@5']:.3f}" if "session@5" in r else ""
            print(f"  {name:48s} {key} {r[key]:.3f}{extra}" + (f"  Δ {d}" if d else ""), flush=True)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
