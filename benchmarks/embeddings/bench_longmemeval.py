"""Embedding quality on LongMemEval-S retrieval (a second, more semantic benchmark than LoCoMo).

Run: HF_HUB_OFFLINE=1 python -m benchmarks.embeddings.bench_longmemeval --questions 150 --models a,b

Per question, every turn in its ~50-session haystack is a candidate.
  * turn recall@k:    a turn labelled ``has_answer`` is in the top-k
  * session recall@k: an ``answer_session_ids`` session is in the top-k sessions,
                      where sessions are ranked by their best turn
Unique texts are embedded once per model (haystacks share sessions).
"bm25" is a lexical reference, to show how keyword-driven the benchmark is.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DATA = os.path.join(os.path.dirname(__file__), "..", "longmemeval", "data", "longmemeval_s_cleaned.json")


def sample_questions(n, seed=13):
    d = json.load(open(DATA))
    d = [x for x in d if not x["question_id"].endswith("_abs")
         and any(t.get("has_answer") for s in x["haystack_sessions"] for t in s)]
    by_type = defaultdict(list)
    for x in d:
        by_type[x["question_type"]].append(x)
    rng = random.Random(seed)
    out = []
    per = max(1, n // len(by_type))
    for t, xs in sorted(by_type.items()):
        rng.shuffle(xs)
        out.extend(xs[:per])
    return out


def encoder(name):
    if name == "bm25":
        return None
    from memorable_ai.core.embeddings import load_encoder

    m = load_encoder(name)
    prompts = getattr(m, "prompts", None) or {}

    def enc(texts, query=False):
        kw = {"prompt_name": "query"} if (query and "query" in prompts) else {}
        v = np.asarray(m.encode(texts, batch_size=64, show_progress_bar=False, **kw), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    return enc


def bm25_scores(query, docs_tok, idf, avgdl, k1=1.2, b=0.75):
    from memorable_ai.core.vector_index import tokenize

    q = tokenize(query, drop_stopwords=True, stem=True)
    out = np.zeros(len(docs_tok), dtype=np.float32)
    for i, (tf, dl) in enumerate(docs_tok):
        s = 0.0
        for w in q:
            f = tf.get(w)
            if f:
                s += idf.get(w, 0.0) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        out[i] = s
    return out


def evaluate(name, qs, ks=(5, 10)):
    from collections import Counter

    from memorable_ai.core.vector_index import tokenize

    texts = {}
    for x in qs:
        for s in x["haystack_sessions"]:
            for t in s:
                texts.setdefault(t["content"], None)
    uniq = list(texts)
    t0 = time.perf_counter()
    enc = encoder(name)
    if enc is not None:
        vecs = enc(uniq)
        qv = enc([x["question"] for x in qs], query=True)
        index = {t: i for i, t in enumerate(uniq)}
    encode_s = time.perf_counter() - t0
    res = defaultdict(lambda: [0, 0])
    for qi, x in enumerate(qs):
        turn_text, turn_sess, gold_turn = [], [], []
        for sid, s in zip(x["haystack_session_ids"], x["haystack_sessions"]):
            for t in s:
                turn_text.append(t["content"])
                turn_sess.append(sid)
                gold_turn.append(bool(t.get("has_answer")))
        if enc is not None:
            sims = vecs[[index[t] for t in turn_text]] @ qv[qi]
        else:
            toks = [Counter(tokenize(t, drop_stopwords=True, stem=True)) for t in turn_text]
            docs = [(c, sum(c.values())) for c in toks]
            df = Counter(w for c, _ in docs for w in c)
            n = len(docs)
            idf = {w: float(np.log(1 + (n - f + 0.5) / (f + 0.5))) for w, f in df.items()}
            avgdl = np.mean([dl for _, dl in docs]) or 1.0
            sims = bm25_scores(x["question"], docs, idf, avgdl)
        order = np.argsort(-sims, kind="stable")
        gold_s = set(x["answer_session_ids"])
        sess_rank = list(dict.fromkeys(turn_sess[i] for i in order))
        for k in ks:
            th = any(gold_turn[i] for i in order[:k])
            sh = bool(gold_s & set(sess_rank[:k]))
            for key, hit in ((f"turn@{k}", th), (f"session@{k}", sh)):
                res[(key, "overall")][0] += hit
                res[(key, "overall")][1] += 1
                res[(key, x["question_type"])][0] += hit
                res[(key, x["question_type"])][1] += 1
    out = {"model": name, "unique_texts": len(uniq), "encode_s": round(encode_s, 1)}
    for (key, typ), (h, n) in sorted(res.items()):
        out.setdefault(key, {})[typ] = round(h / n, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=150)
    ap.add_argument("--models", default="bm25,sentence-transformers/all-MiniLM-L6-v2,"
                    "onnx-int8:sentence-transformers/all-MiniLM-L6-v2,static:minishlab/potion-base-8M")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    qs = sample_questions(args.questions)
    print(f"{len(qs)} questions")
    results = []
    for m in args.models.split(","):
        try:
            r = evaluate(m.strip(), qs)
        except Exception as e:
            r = {"model": m, "error": repr(e)[:300]}
        results.append(r)
        print(json.dumps({k: (v.get("overall") if isinstance(v, dict) else v) for k, v in r.items()}))
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
