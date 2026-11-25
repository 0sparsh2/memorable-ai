"""
Publish prereg v4.2 contexts into the answer loop's run directory (benchmarks/qa/eval_plan/prereg-v4.2.md).

Contexts are built in a staging run (``e2e_qa retrieve --run <stage>``) so a crash never leaves a
half-written line where the running loop reads. This script:
  1. checks that A5 without its facts (A5x) equals the stored A3 context of every subset question;
  2. adds the drift check: A3's stored contexts for 40 LoCoMo / 20 LME subset questions, as arm A3r;
  3. writes A5, A6, D5 and A3r rows to <main>/contexts_<bench>_v4.jsonl atomically (temp file, then rename).

    python -m benchmarks.qa.v4_publish --stage benchmarks/qa/runs/v4 --main benchmarks/qa/runs/main --bench locomo
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

V4_ARMS = ("A5", "A6", "D5")
DRIFT_N = {"locomo": 40, "lme": 20}
SEED = 20261118


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--bench", choices=("locomo", "lme"), required=True)
    ap.add_argument("--allow-mismatch", action="store_true", help="publish even if A5x differs from A3 (report it)")
    args = ap.parse_args()
    stage, main_dir = Path(args.stage), Path(args.main)

    stored = {}
    with open(main_dir / f"contexts_{args.bench}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["arm"] == "A3":
                stored[r["qid"]] = r
    checks = [json.loads(line) for line in open(stage / f"checks_{args.bench}.jsonl")]
    same = sum(c["context"] == stored[c["qid"]]["context"] for c in checks)
    print(f"A5x == stored A3: {same}/{len(checks)}")
    if same != len(checks) and not args.allow_mismatch:
        raise SystemExit("A5 without facts differs from A3; not publishing (see prereg v4.2)")

    rows, seen = [], set()
    for line in open(stage / f"contexts_{args.bench}.jsonl"):
        r = json.loads(line)
        if r["arm"] in V4_ARMS and (r["arm"], r["qid"]) not in seen:
            seen.add((r["arm"], r["qid"]))
            rows.append(r)
    subset = sorted({c["qid"] for c in checks})
    for qid in random.Random(SEED).sample(subset, min(DRIFT_N[args.bench], len(subset))):
        rows.append({**stored[qid], "arm": "A3r"})  # A3 answered again: does the answer model drift?

    out = main_dir / f"contexts_{args.bench}_v4.jsonl"
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, out)
    counts = {}
    for r in rows:
        counts[r["arm"]] = counts.get(r["arm"], 0) + 1
    print(f"published {len(rows)} rows to {out}: {counts}")


if __name__ == "__main__":
    main()
