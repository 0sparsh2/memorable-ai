"""
prereg v5 (answer model): build the comparison's contexts from the stored v3 contexts, no new retrieval.

    python -m benchmarks.qa.v5_build --run benchmarks/qa/runs/v5             # full run
    python -m benchmarks.qa.v5_build --run benchmarks/qa/runs/v5_pilot --pilot  # 3 items per cell

Dev rows only (prereg v5): LoCoMo's dev conversations (LOCOMO_DEV) and LongMemEval-S's 150 dev
questions. Held-out LongMemEval rows in the stored file are skipped without being used or printed.
Each (arm, reader) cell is written as arm "<arm>@<reader>" with a "reader" field; the judge stays
e2e_qa.MODEL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.qa.e2e_qa import LOCOMO_DEV, READERS, V5_SEED, _public, full_context, lme_items  # noqa: E402

MAIN = ROOT / "benchmarks" / "qa" / "runs" / "main"
NO_MEMORY = "(no memories found)"  # what memory_context writes for an empty context
STRONG = ("ultra", "mistral")  # D6 (no-memory decoy) readers


def stratified(items, n: int, seed: int, key) -> list:
    """n items stratified by key, over sorted qids (fixed before any answer)."""
    by = defaultdict(list)
    for it in items:
        by[key(it)].append(it)
    total = sum(len(v) for v in by.values())
    rng = random.Random(seed)
    picked = []
    for _, group in sorted(by.items()):
        group = sorted(group, key=lambda r: r["qid"])
        rng.shuffle(group)
        picked += group[: max(1, round(n * len(group) / total))]
    return picked


def stored(bench: str, arms) -> list:
    """Stored v3 contexts for dev rows; first copy of each (arm, qid)."""
    seen, out = set(), []
    for line in open(MAIN / f"contexts_{bench}.jsonl"):
        r = json.loads(line)
        if r["arm"] not in arms or (r["arm"], r["qid"]) in seen:
            continue
        if bench == "locomo" and r["group"] not in LOCOMO_DEV:
            continue
        if bench == "lme" and r.get("split") != "dev":
            continue  # held out: never used
        seen.add((r["arm"], r["qid"]))
        out.append(r)
    return out


def build() -> dict:
    cells = {"locomo": stored("locomo", ("A0", "A3")), "lme": stored("lme", ("A3",))}
    dev = lme_items("dev")
    a0 = stratified(dev, 50, V5_SEED, key=lambda it: it["category"])
    cells["lme"] += [{"arm": "A0", **_public(it), "context": full_context(it), "retrieval_ms": 0.0} for it in a0]
    out = {"locomo": [], "lme": []}
    for bench, recs in cells.items():
        for reader in READERS:
            out[bench] += [{**r, "arm": f"{r['arm']}@{reader}", "reader": reader} for r in recs]
        # D6: the question with no memory at all; a strong reader must not score from priors
        a3 = [r for r in recs if r["arm"] == "A3"]
        for r in stratified(a3, 50, V5_SEED + 1, key=lambda r: r["category"]):
            for reader in STRONG:
                out[bench].append({**r, "arm": f"D6@{reader}", "reader": reader, "context": NO_MEMORY})
    return out


def pilot(out: dict) -> dict:
    """3 items per cell; each A0-LME cell includes the longest prompt (context-length check)."""
    rng = random.Random(V5_SEED)
    res = {}
    for bench, recs in out.items():
        by = defaultdict(list)
        for r in recs:
            by[r["arm"]].append(r)
        keep = []
        for arm, rs in sorted(by.items()):
            rs = sorted(rs, key=lambda r: r["qid"])
            if bench == "lme" and arm.startswith("A0@"):
                longest = max(rs, key=lambda r: len(r["context"]))
                keep += [longest] + rng.sample([r for r in rs if r is not longest], 2)
            else:
                keep += rng.sample(rs, 3)
        res[bench] = keep
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing contexts (before the freeze only)")
    args = ap.parse_args()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    out = build()
    if args.pilot:
        out = pilot(out)
    manifest = {}
    for bench, recs in out.items():
        path = run / f"contexts_{bench}.jsonl"
        if path.exists() and not args.force:
            raise SystemExit(f"{path} exists (frozen); pass --force only before the pilot")
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        counts = Counter(r["arm"] for r in recs)
        longest = {a: max(len(r["context"]) for r in recs if r["arm"] == a) // 4 for a in counts}
        manifest[bench] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "items": dict(sorted(counts.items())),
                           "max_context_tokens_est": dict(sorted(longest.items()))}
    (run / "contexts.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
