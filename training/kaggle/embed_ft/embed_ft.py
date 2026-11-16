"""Contrastive fine-tune of memorable's embedding models for conversational memory (Kaggle GPU job).

Input: qgen.jsonl from the memorable-qgen kernel (synthetic questions over public
dialogue corpora; nothing from LoCoMo or LongMemEval).

1. Teacher pass (Qwen3-Embedding-0.6B) over every question and turn.
   * Round-trip filter: keep a question only if an evidence turn ranks in the
     teacher's top 10 within its own conversation (drops vague or unanswerable
     generations).
   * Negatives, tiered as in HiNS (arXiv 2601.14857): "hard" = the teacher's top
     non-evidence turn of the same conversation; "medium" = a random turn of the
     same conversation; "easy" = other conversations, via in-batch negatives.
     False-negative guard (NV-Retriever style): skip candidates scoring above 95%
     of the evidence score, and the turns adjacent to an evidence turn (the
     pipeline blends the previous turn into each turn vector).
2. Train with (Cached)MultipleNegativesRankingLoss, NO_DUPLICATES batches:
     leafir   MongoDB/mdbr-leaf-ir            turn index (quality profile)
     minilm   sentence-transformers/all-MiniLM-L6-v2  semantic tier; also
              question -> turn window pairs, which look like episode text
     potion   minishlab/potion-base-32M (StaticEmbedding), two learning rates
   ce_*     cross-encoder rerankers (MiniLM-L6, Ettin 17M / 32M) with BCE on 1 positive :
            3 hard + 1 medium negatives
3. Sanity check on held-out synthetic conversations (recall@5 within a conversation).

Output in /kaggle/working: models/<name>/ (sentence-transformers format),
train_pairs.jsonl, ft_report.json.
"""

import glob
import json
import os
import random
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

DRY = os.environ.get("FT_DRYRUN") == "1"  # local CPU check: tiny data, 2 steps per model
OUT = os.environ.get("FT_OUT", "/kaggle/working")
DEVICE = "cpu" if DRY else "cuda"
SEED = 20261115
TEACHER = "MongoDB/mdbr-leaf-ir" if DRY else "Qwen/Qwen3-Embedding-0.6B"
TEACHER_QUERY = "Instruct: Given a question about a past conversation, retrieve the turn that answers it\nQuery:"
DEV_FRACTION = 0.04
RANK_KEEP = 10
FALSE_NEG = 0.95
HUMAN = ("cc", "msc", "soda")

RUNS = [
    # name, base, kind, learning rate, epochs
    ("leafir", "MongoDB/mdbr-leaf-ir", "transformer", 2e-5, 1),
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2", "transformer", 3e-5, 1),
    ("potion_lr01", "minishlab/potion-base-32M", "static", 1e-2, 1),
    ("potion_lr05", "minishlab/potion-base-32M", "static", 5e-2, 1),
]
CE_RUNS = [
    # rerankers: start from MS MARCO cross-encoders, adapt to conversation turns
    ("ce_minilm", "cross-encoder/ms-marco-MiniLM-L6-v2", 2e-5),
    ("ce_ettin17", "cross-encoder/ettin-reranker-17m-v1", 3e-5),
    ("ce_ettin32", "cross-encoder/ettin-reranker-32m-v1", 2e-5),
]


def pip(*pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs], capture_output=True, text=True)
    print("pip", pkgs, r.returncode, r.stderr[-800:], flush=True)


def turn_text(conv, t):
    return t["text"] if conv["source"] == "wildchat" else f"{t['speaker']}: {t['text']}"


def window_text(conv, idx, k=2):
    s = conv["turns"][idx]["session"]
    lo, hi = max(0, idx - k), min(len(conv["turns"]), idx + k + 1)
    return "\n".join(turn_text(conv, t) for t in conv["turns"][lo:hi] if t["session"] == s)


def load():
    paths = ([os.environ["FT_QGEN"]] if os.environ.get("FT_QGEN")
             else sorted(glob.glob("/kaggle/input/**/qgen*.jsonl", recursive=True)))
    if not paths:
        raise SystemExit("qgen.jsonl not found under /kaggle/input")
    convs = []
    for path in paths:  # every qgen*.jsonl found (Kaggle generation, salvaged chunks, NIM generation)
        convs += [json.loads(line) for line in open(path)]
    for c in convs:  # generators sometimes write a statement instead of a question
        c["questions"] = [q for q in c.get("questions") or [] if "?" in q["q"]]
    convs = [c for c in convs if c["questions"] and len(c["turns"]) >= 4]
    rng = random.Random(SEED)
    rng.shuffle(convs)
    n_dev = 6 if DRY else max(20, int(len(convs) * DEV_FRACTION))
    return convs[n_dev:], convs[:n_dev]


def encode(model, texts, bs=64, prompt=None):
    import numpy as np

    kw = {"prompt": prompt} if prompt else {}
    v = model.encode(texts, batch_size=bs, convert_to_numpy=True, normalize_embeddings=True,
                     show_progress_bar=False, **kw)
    return np.asarray(v, dtype=np.float32)


def teacher_pass(convs):
    """Per question: teacher scores against its own conversation's turns."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    m = SentenceTransformer(TEACHER, device=DEVICE, model_kwargs={"torch_dtype": torch.float16} if not DRY else {})
    m.max_seq_length = 512
    texts = [turn_text(c, t) for c in convs for t in c["turns"]]
    qs = [q["q"] for c in convs for q in c["questions"]]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    tv = np.empty((len(texts), m.get_sentence_embedding_dimension()), dtype=np.float32)
    tv[order] = encode(m, [texts[i] for i in order], bs=128)
    qv = encode(m, qs, bs=256, prompt=TEACHER_QUERY)
    del m
    torch.cuda.empty_cache()
    off, qi = 0, 0
    for c in convs:
        n = len(c["turns"])
        sims = qv[qi:qi + len(c["questions"])] @ tv[off:off + n].T
        for q, s in zip(c["questions"], sims):
            q["_scores"] = s
        off += n
        qi += len(c["questions"])
    print(f"teacher pass: {len(texts)} turns, {len(qs)} questions in {time.time() - t0:.0f}s", flush=True)


def build_pairs(convs, rng):
    import numpy as np

    rows, stats = [], {"questions": 0, "kept": 0, "no_hard_neg": 0}
    for c in convs:
        idx_of = {t["tid"]: i for i, t in enumerate(c["turns"])}
        for q in c["questions"]:
            stats["questions"] += 1
            ev = [idx_of[e] for e in q["evidence"] if e in idx_of]
            s = q.pop("_scores")
            ranks = np.argsort(-s)
            pos_rank = min(int(np.where(ranks == e)[0][0]) for e in ev)
            if pos_rank >= RANK_KEEP:
                continue
            stats["kept"] += 1
            blocked = set()
            for e in ev:
                for d in (-1, 0, 1):
                    j = e + d
                    if 0 <= j < len(c["turns"]) and c["turns"][j]["session"] == c["turns"][e]["session"]:
                        blocked.add(j)
            floor = FALSE_NEG * min(float(s[e]) for e in ev)
            cand = [int(j) for j in ranks if int(j) not in blocked and s[j] < floor]
            hard = cand[: 3]
            if not hard:
                stats["no_hard_neg"] += 1
                continue
            pool = [j for j in range(len(c["turns"])) if j not in blocked and j not in hard[:1]]
            medium = rng.choice(pool) if pool else hard[-1]
            for e in ev:
                rows.append({
                    "cid": c["cid"], "source": c["source"], "type": q["type"], "anchor": q["q"],
                    "positive": turn_text(c, c["turns"][e]),
                    "positive_window": window_text(c, e),
                    "hard": turn_text(c, c["turns"][rng.choice(hard)]),
                    "hards": [turn_text(c, c["turns"][j]) for j in hard],
                    "hard_window": window_text(c, hard[0]),
                    "medium": turn_text(c, c["turns"][medium]),
                    "teacher_rank": pos_rank,
                })
    return rows, stats


def dev_recall(model, convs, prompt=None, k=5):
    import numpy as np

    hits = []
    for c in convs:
        tv = encode(model, [turn_text(c, t) for t in c["turns"]])
        idx_of = {t["tid"]: i for i, t in enumerate(c["turns"])}
        qv = encode(model, [q["q"] for q in c["questions"]], prompt=prompt)
        for q, s in zip(c["questions"], qv @ tv.T):
            top = set(np.argsort(-s)[:k].tolist())
            hits.append(any(idx_of[e] in top for e in q["evidence"] if e in idx_of))
    return round(float(np.mean(hits)), 4)


def ce_dev_recall(model, convs, k=5, max_turns=80):
    """Reranker alone over every turn of the conversation (harder than the top-20 it sees in production)."""
    import numpy as np

    hits = []
    for c in convs:
        turns = c["turns"][:max_turns]
        texts = [turn_text(c, t) for t in turns]
        idx_of = {t["tid"]: i for i, t in enumerate(turns)}
        for q in c["questions"]:
            gold = {idx_of[e] for e in q["evidence"] if e in idx_of}
            if not gold:
                continue
            s = np.asarray(model.predict([(q["q"], t) for t in texts], batch_size=128, show_progress_bar=False))
            hits.append(bool(gold & set(np.argsort(-s)[:k].tolist())))
    return round(float(np.mean(hits)), 4)


def train_ce(name, base, lr, rows, dev):
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    t0 = time.time()
    model = CrossEncoder(base, num_labels=1, max_length=256, device=DEVICE)
    before = ce_dev_recall(model, dev)
    q, p, y = [], [], []
    for r in rows:
        negs = list(dict.fromkeys(r["hards"] + [r["medium"]]))
        for text, label in [(r["positive"], 1.0)] + [(n, 0.0) for n in negs]:
            q.append(r["anchor"])
            p.append(text)
            y.append(label)
    ds = Dataset.from_dict({"query": q, "passage": p, "label": y}).shuffle(seed=SEED)
    n_neg = (len(y) - sum(y)) / max(1, sum(y))
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(n_neg))
    args = CrossEncoderTrainingArguments(
        output_dir=f"{OUT}/ckpt_{name}", num_train_epochs=1, per_device_train_batch_size=16 if DRY else 64,
        learning_rate=lr, warmup_ratio=0.1, fp16=not DRY, max_steps=2 if DRY else -1, save_strategy="no",
        logging_steps=100, report_to="none", seed=SEED,
        dataloader_drop_last=True,
    )
    CrossEncoderTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
    after = ce_dev_recall(model, dev)
    path = f"{OUT}/models/{name}"
    model.save_pretrained(path)
    os.system(f"rm -rf {OUT}/ckpt_{name}")
    res = {"name": name, "base": base, "lr": lr, "pairs": len(ds), "dev_recall@5_before": before,
           "dev_recall@5_after": after, "secs": round(time.time() - t0)}
    print(json.dumps(res), flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def train(name, base, kind, lr, epochs, rows, dev):
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    t0 = time.time()
    if kind == "static":
        from sentence_transformers.models import StaticEmbedding

        model = SentenceTransformer(modules=[StaticEmbedding.from_model2vec(base)], device=DEVICE)
        qprompt = None
    else:
        model = SentenceTransformer(base, device=DEVICE)
        model.max_seq_length = 256
        qprompt = (model.prompts or {}).get("query")
    before = dev_recall(model, dev, prompt=qprompt)

    data = {"anchor": [], "positive": [], "negative_1": [], "negative_2": []}
    for r in rows:
        data["anchor"].append(r["anchor"])
        data["positive"].append(r["positive"])
        data["negative_1"].append(r["hard"])
        data["negative_2"].append(r["medium"])
        if name == "minilm":  # episode-like targets for the semantic tier
            data["anchor"].append(r["anchor"])
            data["positive"].append(r["positive_window"])
            data["negative_1"].append(r["hard_window"])
            data["negative_2"].append(r["medium"])
    ds = Dataset.from_dict(data).shuffle(seed=SEED)

    if kind == "static":
        loss = losses.MultipleNegativesRankingLoss(model)
        bs = 16 if DRY else 1024
    else:
        # GradCache gives 256-wide batches on a T4; it needs CUDA (its RNG context breaks on CPU/MPS)
        loss = (losses.MultipleNegativesRankingLoss(model) if DRY
                else losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=64))
        bs = 16 if DRY else 256
    args = SentenceTransformerTrainingArguments(
        output_dir=f"{OUT}/ckpt_{name}", num_train_epochs=epochs, per_device_train_batch_size=bs,
        learning_rate=lr, warmup_ratio=0.1, fp16=torch.cuda.is_available() and kind != "static",
        max_steps=2 if DRY else -1,
        batch_sampler=BatchSamplers.NO_DUPLICATES, save_strategy="no", logging_steps=25,
        report_to="none", seed=SEED, prompts={"anchor": qprompt} if qprompt else None,
    )
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
    after = dev_recall(model, dev, prompt=qprompt)
    path = f"{OUT}/models/{name}"
    model.save(path)
    os.system(f"rm -rf {OUT}/ckpt_{name}")
    res = {"name": name, "base": base, "lr": lr, "epochs": epochs, "examples": len(ds),
           "dev_recall@5_before": before, "dev_recall@5_after": after, "secs": round(time.time() - t0)}
    print(json.dumps(res), flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    t0 = time.time()
    if not DRY:
        pip("sentence-transformers>=5.1", "model2vec>=0.6")
    rng = random.Random(SEED)
    train_convs, dev = load()
    print(f"train conversations {len(train_convs)}, dev {len(dev)}", flush=True)
    teacher_pass(train_convs)
    rows, stats = build_pairs(train_convs, rng)
    print(json.dumps(stats), flush=True)
    with open(f"{OUT}/train_pairs.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(f"{OUT}/dev_convs.jsonl", "w") as f:
        for c in dev:
            f.write(json.dumps(c) + "\n")
    report = {"filter": stats, "pairs": len(rows), "runs": []}
    for run in RUNS:
        try:
            report["runs"].append(train(*run, rows, dev))
        except Exception as e:  # keep the other runs
            import traceback

            traceback.print_exc()
            report["runs"].append({"name": run[0], "error": repr(e)[:400]})
        json.dump(report, open(f"{OUT}/ft_report.json", "w"), indent=2)
    for run in CE_RUNS:
        try:
            report["runs"].append(train_ce(*run, rows, dev))
        except Exception as e:
            import traceback

            traceback.print_exc()
            report["runs"].append({"name": run[0], "error": repr(e)[:400]})
        json.dump(report, open(f"{OUT}/ft_report.json", "w"), indent=2)
    report["wall_s"] = round(time.time() - t0)
    json.dump(report, open(f"{OUT}/ft_report.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
