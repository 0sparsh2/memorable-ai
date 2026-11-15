"""Turn-layer signal study: dense models, BM25, fusion rules and rerankers (docs/embedding_study.md §9).

Both benchmarks, same candidate sets as the pipeline's turn layer:
  LoCoMo        every turn of the conversation; hit = an evidence dia_id in the top k (1,536 questions)
  LongMemEval-S every turn of the question's ~50-session haystack (150 stratified questions);
                turn@k and session@k (sessions ranked by their best turn)

Dense vectors get the pipeline's contextual blend (v + 0.5 * previous turn of the same
session, renormalised). BM25 uses memorable's tokenizer. Fusion rules:
  rrf    weighted reciprocal rank fusion of each signal's top 50 (the pipeline's rule, k=60)
  z      calibrated score fusion: per-candidate-set z-scores, weighted sum

  python -m benchmarks.embeddings.turn_signals encode --models MongoDB/mdbr-leaf-ir,static:minishlab/potion-base-32M
  python -m benchmarks.embeddings.turn_signals fuse --out results_turn_fusion.json
  python -m benchmarks.embeddings.turn_signals rerank --rerankers cross-encoder/ms-marco-MiniLM-L6-v2

Embeddings are cached (fp16) under $MEMORABLE_BENCH_CACHE (default ~/.cache/memorable/bench),
so each model is encoded once.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from benchmarks.embeddings.bench_longmemeval import sample_questions  # noqa: E402
from benchmarks.locomo.loader import format_turn_text, iter_sessions  # noqa: E402

LOCOMO = os.path.join(os.path.dirname(__file__), "..", "locomo", "data", "locomo10.json")
CACHE = os.environ.get("MEMORABLE_BENCH_CACHE", os.path.expanduser("~/.cache/memorable/bench"))
LOCOMO_CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
BLEND = 0.5
DEPTH, RRF_K = 50, 60


# ------------------------------------------------------------------ data


class Bench:
    """Unique texts, candidate groups (one per conversation / haystack) and queries."""

    def __init__(self, name):
        self.name, self.texts, self.groups, self.queries = name, [], [], []
        self._index = {}

    def tid(self, text):
        i = self._index.get(text)
        if i is None:
            i = self._index[text] = len(self.texts)
            self.texts.append(text)
        return i

    def digest(self):
        return hashlib.sha1("\x00".join(self.texts).encode()).hexdigest()[:12]


def load_locomo():
    b = Bench("locomo")
    for sample in json.load(open(LOCOMO)):
        conv = sample["conversation"]
        cand, sess, dia = [], [], []
        for skey, _date, ts in iter_sessions(conv):
            for t in ts:
                cand.append(b.tid(f"{t.get('speaker')}: {format_turn_text(t)}"))
                sess.append(skey)
                dia.append(t.get("dia_id"))
        g = len(b.groups)
        b.groups.append({"cand": np.array(cand), "sess": sess})
        for q in sample["qa"]:
            if q.get("category") in LOCOMO_CATS and q.get("evidence"):
                ev = set(q["evidence"])
                b.queries.append({"text": q["question"], "group": g, "cat": LOCOMO_CATS[q["category"]],
                                  "gold": np.array([d in ev for d in dia]), "gold_sess": None})
    return b


def load_lme(n=150):
    b = Bench("lme")
    for x in sample_questions(n):
        cand, sess, gold = [], [], []
        for sid, s in zip(x["haystack_session_ids"], x["haystack_sessions"]):
            for t in s:
                cand.append(b.tid(t["content"]))
                sess.append(sid)
                gold.append(bool(t.get("has_answer")))
        b.groups.append({"cand": np.array(cand), "sess": sess})
        b.queries.append({"text": x["question"], "group": len(b.groups) - 1, "cat": x["question_type"],
                          "gold": np.array(gold), "gold_sess": set(x["answer_session_ids"])})
    return b


# ------------------------------------------------------------------ signals


def _slug(spec):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", spec)


def vectors(bench, spec, encode_if_missing=True):
    """(doc_vecs, query_vecs), float32, L2-normalised; cached as fp16."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{bench.name}__{_slug(spec)}__{bench.digest()}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["docs"].astype(np.float32), z["queries"].astype(np.float32)
    if not encode_if_missing:
        raise FileNotFoundError(path)
    from memorable_ai.core.embeddings import load_encoder

    m = load_encoder(spec)
    prompts = getattr(m, "prompts", None) or {}

    def enc(texts, query=False):
        kw = {"prompt_name": "query"} if (query and "query" in prompts) else {}
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        v = np.asarray(m.encode([texts[i] for i in order], batch_size=64, show_progress_bar=False, **kw),
                       dtype=np.float32)
        res = np.empty_like(v)
        res[order] = v
        return res / np.maximum(np.linalg.norm(res, axis=1, keepdims=True), 1e-9)

    t = time.perf_counter()
    docs = enc(bench.texts)
    queries = enc([q["text"] for q in bench.queries], query=True)
    print(f"  encoded {bench.name} with {spec}: {len(bench.texts)} texts in {time.perf_counter() - t:.0f}s",
          flush=True)
    np.savez(path, docs=docs.astype(np.float16), queries=queries.astype(np.float16))
    return docs, queries


def blended(group, docs):
    v = docs[group["cand"]]
    out = v.copy()
    sess = group["sess"]
    for i in range(1, len(v)):
        if sess[i] == sess[i - 1]:
            b = v[i] + BLEND * v[i - 1]
            out[i] = b / (np.linalg.norm(b) or 1.0)
    return out


class Lexical:
    def __init__(self, bench):
        from memorable_ai.core.vector_index import tokenize

        self.tokenize = tokenize
        self.tok = [Counter(tokenize(t, drop_stopwords=True, stem=True)) for t in bench.texts]
        self.stats = {}

    def scores(self, bench, gi, query, k1=1.2, b=0.75):
        g = bench.groups[gi]
        if gi not in self.stats:
            docs = [self.tok[i] for i in g["cand"]]
            dl = np.array([sum(c.values()) for c in docs], dtype=np.float32)
            df = Counter(w for c in docs for w in c)
            n = len(docs)
            idf = {w: float(np.log(1 + (n - f + 0.5) / (f + 0.5))) for w, f in df.items()}
            self.stats[gi] = (docs, dl, idf, float(dl.mean() or 1.0))
        docs, dl, idf, avgdl = self.stats[gi]
        out = np.zeros(len(docs), dtype=np.float32)
        for w in set(self.tokenize(query, drop_stopwords=True, stem=True)):
            if w not in idf:
                continue
            tf = np.array([c.get(w, 0) for c in docs], dtype=np.float32)
            out += idf[w] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
        return out


def signal_scores(bench, specs, lexical=True):
    """{signal: [score array per query]} over each query's candidate set."""
    out = defaultdict(list)
    for spec in specs:
        docs, qv = vectors(bench, spec)
        cache = {}
        for qi, q in enumerate(bench.queries):
            g = q["group"]
            if g not in cache:
                cache = {g: blended(bench.groups[g], docs)}
            out[spec].append(cache[g] @ qv[qi])
    if lexical:
        lex = Lexical(bench)
        out["bm25"] = [lex.scores(bench, q["group"], q["text"]) for q in bench.queries]
    return out


# ------------------------------------------------------------------ fusion + metrics


def rrf(scores, weights):
    n = len(next(iter(scores.values())))
    fused = np.zeros(n)
    for name, w in weights.items():
        if w <= 0:
            continue
        s = scores[name]
        top = np.argsort(-s, kind="stable")[:DEPTH]
        if name == "bm25":
            top = top[s[top] > 0]
        fused[top] += w / (RRF_K + np.arange(1, len(top) + 1))
    return fused


def zpool(scores, weights):
    """Score fusion the pipeline can compute: z-scores over the union of each signal's top 50.
    Dense scores are exact for the whole pool; BM25 outside its own top 50 counts as 0."""
    pool, tops = set(), {}
    for name in weights:
        s = scores[name]
        t = np.argsort(-s, kind="stable")[:DEPTH]
        if name == "bm25":
            t = t[s[t] > 0]
        tops[name] = t
        pool |= set(t.tolist())
    pool = np.array(sorted(pool))
    fused = np.full(len(next(iter(scores.values()))), -1e9)
    acc = np.zeros(len(pool))
    for name, w in weights.items():
        v = scores[name][pool].astype(np.float64)
        if name == "bm25":
            v = np.where(np.isin(pool, tops[name]), v, 0.0)
        sd = v.std()
        acc += w * ((v - v.mean()) / sd if sd > 0 else 0.0)
    fused[pool] = acc
    return fused


def zfuse(scores, weights):
    fused = 0.0
    for name, w in weights.items():
        s = scores[name].astype(np.float64)
        sd = s.std()
        fused = fused + w * ((s - s.mean()) / sd if sd > 0 else 0 * s)
    return fused


def hits(bench, rankings, ks=(5, 10)):
    """Per-query hit vectors: {metric: np.array(bool)}."""
    out = defaultdict(list)
    for q, order in zip(bench.queries, rankings):
        gold = q["gold"]
        for k in ks:
            out[f"turn@{k}"].append(bool(gold[order[:k]].any()))
        if q["gold_sess"] is not None:
            sess = bench.groups[q["group"]]["sess"]
            ranked = list(dict.fromkeys(sess[i] for i in order[:200]))
            for k in ks:
                out[f"session@{k}"].append(bool(q["gold_sess"] & set(ranked[:k])))
    return {k: np.array(v) for k, v in out.items()}


def summarize(bench, h):
    res = {m: round(float(v.mean()), 4) for m, v in h.items()}
    key = "turn@5" if bench.name == "locomo" else "turn@10"
    cats = defaultdict(list)
    for q, x in zip(bench.queries, h[key]):
        cats[q["cat"]].append(x)
    res[f"{key}_by_cat"] = {c: round(float(np.mean(v)), 3) for c, v in sorted(cats.items())}
    return res


def paired_ci(a, b, n=2000, seed=0):
    """95% bootstrap CI of mean(b - a) over queries."""
    rng = np.random.default_rng(seed)
    d = b.astype(float) - a.astype(float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return round(float(d.mean()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)


def rank_all(scores_by_signal, rule, weights):
    fn = {"rrf": rrf, "z": zfuse, "zp": zpool}[rule]
    n_q = len(next(iter(scores_by_signal.values())))
    out = []
    for qi in range(n_q):
        s = {k: v[qi] for k, v in scores_by_signal.items() if k in weights}
        out.append(np.argsort(-fn(s, weights), kind="stable"))
    return out


# ------------------------------------------------------------------ commands


LEAF, POTION, MINILM = "MongoDB/mdbr-leaf-ir", "static:minishlab/potion-base-32M", "sentence-transformers/all-MiniLM-L6-v2"


def configs(a, b):
    """Fusion configurations: today's two profiles, then dual dense signals."""
    cfg = {
        "v0.2: rrf(minilm .5, bm25 1)": ("rrf", {MINILM: 0.5, "bm25": 1.0}),
        "fast: rrf(potion .5, bm25 1)": ("rrf", {POTION: 0.5, "bm25": 1.0}),
        "quality: rrf(leaf .5, bm25 1)": ("rrf", {a: 0.5, "bm25": 1.0}),
        "dense leaf": ("rrf", {a: 1.0}),
        "dense potion": ("rrf", {b: 1.0}),
    }
    for wa, wb, wl in ((0.5, 0.25, 1.0), (0.5, 0.5, 1.0), (1.0, 0.5, 1.0), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5),
                       (1.0, 0.5, 0.5)):
        cfg[f"rrf(leaf {wa}, potion {wb}, bm25 {wl})"] = ("rrf", {a: wa, b: wb, "bm25": wl})
    for wa, wl in ((1.0, 0.3), (1.0, 0.5), (1.0, 1.0)):
        cfg[f"z(leaf {wa}, bm25 {wl})"] = ("z", {a: wa, "bm25": wl})
    for wa, wb, wl in ((1.0, 0.5, 0.3), (1.0, 0.5, 0.5), (1.0, 1.0, 0.5), (1.0, 0.3, 0.3)):
        cfg[f"z(leaf {wa}, potion {wb}, bm25 {wl})"] = ("z", {a: wa, b: wb, "bm25": wl})
    for wl in (0.3, 0.5):  # pipeline-feasible variants (top-50 pool)
        cfg[f"zp(leaf 1.0, bm25 {wl})"] = ("zp", {a: 1.0, "bm25": wl})
        cfg[f"zp(leaf 1.0, potion 0.5, bm25 {wl})"] = ("zp", {a: 1.0, b: 0.5, "bm25": wl})
    return cfg


def cmd_encode(args):
    for bench in (load_locomo(), load_lme(args.questions)):
        for spec in args.models.split(","):
            vectors(bench, spec.strip())


def cmd_fuse(args):
    a, b = args.dense_a, args.dense_b
    report = {}
    for bench in (load_locomo(), load_lme(args.questions)):
        scores = signal_scores(bench, [a, b, MINILM] if a == LEAF else [a, b, MINILM, LEAF])
        base_name = "quality: rrf(leaf .5, bm25 1)"
        res, base_hits = {}, None
        for name, (rule, w) in configs(a, b).items():
            h = hits(bench, rank_all(scores, rule, w))
            if name == base_name:
                base_hits = h
            res[name] = summarize(bench, h)
            res[name]["_hits"] = h
        key = "turn@5" if bench.name == "locomo" else "turn@10"
        for name, r in res.items():
            r[f"delta_vs_quality_{key}"] = paired_ci(base_hits[key], r.pop("_hits")[key])
        report[bench.name] = res
        print(f"\n== {bench.name} ({len(bench.queries)} questions) ==")
        for name, r in res.items():
            extra = f" session@5 {r['session@5']:.3f}" if "session@5" in r else ""
            print(f"  {name:42s} {key} {r[key]:.3f}  turn@10 {r['turn@10']:.3f}{extra}"
                  f"  Δ {r[f'delta_vs_quality_{key}']}")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)


def cmd_rerank(args):
    from sentence_transformers import CrossEncoder

    a, b = args.dense_a, args.dense_b
    first_stage = {
        "rrf": ("rrf", {a: 0.5, "bm25": 1.0}),
        "zp": ("zp", {a: 1.0, "bm25": 0.5}),
        "zp-dual": ("zp", {a: 1.0, b: 0.5, "bm25": 0.5}),
    }
    first_stage = {k: v for k, v in first_stage.items() if k in args.stage1.split(",")}
    report = {}
    for bench in (load_locomo(), load_lme(args.questions)):
        scores = signal_scores(bench, [a, b])
        key = "turn@5" if bench.name == "locomo" else "turn@10"
        rep = report[bench.name] = {}
        stage1, s1_fused = {}, {}
        for name, (rule, w) in first_stage.items():
            fn = {"rrf": rrf, "zp": zpool}[rule]
            s1_fused[name] = [fn({k: scores[k][qi] for k in w}, w) for qi in range(len(bench.queries))]
            stage1[name] = [np.argsort(-f, kind="stable") for f in s1_fused[name]]
            h = hits(bench, stage1[name])
            rep[name] = summarize(bench, h)
            rep[name]["_h"] = h
        for spec in args.rerankers.split(","):
            spec = spec.strip()
            ce = CrossEncoder(spec, max_length=args.max_length, trust_remote_code=True)
            ce.predict([("warm up", "warm up")])
            for s1 in first_stage:
                lat, ce_scores = [], []
                for q, order in zip(bench.queries, stage1[s1]):
                    top = order[: args.depth]
                    cand = bench.groups[q["group"]]["cand"]
                    pairs = [(q["text"], bench.texts[cand[i]]) for i in top]
                    t = time.perf_counter()
                    s = np.asarray(ce.predict(pairs, batch_size=args.depth, show_progress_bar=False), dtype=np.float64)
                    lat.append(1000 * (time.perf_counter() - t))
                    ce_scores.append(s.reshape(len(pairs), -1)[:, -1])
                for mode in ("pure", "rrf-mix", "z-blend"):
                    ranked = []
                    for order, s, f in zip(stage1[s1], ce_scores, s1_fused[s1]):
                        top = order[: args.depth]
                        if mode == "pure":
                            new = top[np.argsort(-s, kind="stable")]
                        elif mode == "rrf-mix":  # equal-weight RRF of reranker rank and first-stage rank
                            r = np.empty(len(top))
                            r[np.argsort(-s, kind="stable")] = np.arange(1, len(top) + 1)
                            new = top[np.argsort(-(1 / (RRF_K + r) + 1 / (RRF_K + np.arange(1, len(top) + 1))),
                                                  kind="stable")]
                        else:  # z-scored reranker score + z-scored first-stage score, over the top-k
                            f1 = f[top]
                            z = lambda v: (v - v.mean()) / (v.std() or 1.0)  # noqa: E731
                            new = top[np.argsort(-(z(s) + z(f1)), kind="stable")]
                        ranked.append(np.concatenate([new, order[args.depth:]]))
                    h = hits(bench, ranked)
                    name = f"{s1} > {spec.split('/')[-1]} top{args.depth} {mode}"
                    r = summarize(bench, h)
                    r["rerank_ms_p50"] = round(float(np.median(lat)), 1)
                    r["rerank_ms_p95"] = round(float(np.percentile(lat, 95)), 1)
                    r[f"delta_vs_{s1}_{key}"] = paired_ci(rep[s1]["_h"][key], h[key])
                    rep[name] = r
                    extra = f" s@5 {r['session@5']:.3f}" if "session@5" in r else ""
                    print(f"  [{bench.name}] {name:52s} {key} {r[key]:.3f}{extra}  p50 {r['rerank_ms_p50']:.0f} ms"
                          f"  Δ {r[f'delta_vs_{s1}_{key}']}", flush=True)
            del ce
        for s1 in first_stage:
            h = rep[s1].pop("_h")
            print(f"  [{bench.name}] first stage {s1:10s} {key} {h[key].mean():.3f}")
        if args.out:
            json.dump(report, open(args.out, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("encode", "fuse", "rerank"):
        p = sub.add_parser(name)
        p.add_argument("--questions", type=int, default=150)
        p.add_argument("--dense-a", default=LEAF)
        p.add_argument("--dense-b", default=POTION)
        p.add_argument("--out", default="")
        if name == "encode":
            p.add_argument("--models", default=f"{LEAF},{POTION},{MINILM}")
        if name == "rerank":
            p.add_argument("--rerankers", default="cross-encoder/ms-marco-MiniLM-L6-v2")
            p.add_argument("--stage1", default="zp,zp-dual")
            p.add_argument("--depth", type=int, default=20)
            p.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()
    {"encode": cmd_encode, "fuse": cmd_fuse, "rerank": cmd_rerank}[args.cmd](args)


if __name__ == "__main__":
    main()
