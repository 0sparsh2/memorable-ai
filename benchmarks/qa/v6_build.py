"""
prereg v6 (observation log): assemble the answer queue from the contexts in runs/v6.

    python -m benchmarks.qa.v6_build --reader ultra --run benchmarks/qa/runs/v6ans          # full run
    python -m benchmarks.qa.v6_build --reader ultra --run benchmarks/qa/runs/v6ans_pilot --pilot

Every row is answered by ``--reader`` (v5's B11 choice) and judged by e2e_qa.MODEL, as in v5:
arm "<arm>@<reader>" with a "reader" field.
- A3g, A7 and D7: the contexts ``e2e_qa retrieve`` built into runs/v6.
- A3r, the drift check: 40 LoCoMo-dev and 20 LME-dev A3 contexts from v5 (stratified by category,
  seed V6_SEED), answered again inside this run. A3 and A0 answers themselves are reused from v5.
Dev rows only (LoCoMo conv-48/49, LongMemEval dev 150).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.qa.e2e_qa import LOCOMO_DEV, READERS  # noqa: E402
from benchmarks.qa.v5_build import stratified  # noqa: E402

RUNS = ROOT / "benchmarks" / "qa" / "runs"
SRC, V5 = RUNS / "v6", RUNS / "v5"
V6_SEED = 20261121
ARMS = ("A3g", "A7", "D7")
EXPECTED = {"locomo": 347, "lme": 150}
DRIFT = {"locomo": 40, "lme": 20}


def dev(bench: str, r: dict) -> bool:
    return r.get("group") in LOCOMO_DEV if bench == "locomo" else r.get("split") == "dev"


def v6_rows(bench: str) -> dict:
    """arm -> [rows] from runs/v6, first copy of each (arm, qid), dev rows only."""
    out, seen = defaultdict(list), set()
    path = SRC / f"contexts_{bench}.jsonl"
    for line in open(path) if path.exists() else ():
        r = json.loads(line)
        if r["arm"] in ARMS and dev(bench, r) and (r["arm"], r["qid"]) not in seen:
            seen.add((r["arm"], r["qid"]))
            out[r["arm"]].append(r)
    return out


def drift_rows(bench: str, reader: str) -> list:
    """The A3 contexts v5 answered with this reader, sampled for the drift check."""
    a3 = [r for r in map(json.loads, open(V5 / f"contexts_{bench}.jsonl"))
          if r["arm"] == f"A3@{reader}" and dev(bench, r)]
    return stratified(a3, DRIFT[bench], V6_SEED, key=lambda r: r["category"])


def build(reader: str, pilot: bool) -> dict:
    out, missing = {}, []
    rng = random.Random(V6_SEED)
    for bench in ("locomo", "lme"):
        cells = v6_rows(bench)
        cells["A3r"] = drift_rows(bench, reader)
        recs = []
        for arm, rows in sorted(cells.items()):
            want = DRIFT[bench] if arm == "A3r" else EXPECTED[bench]
            if pilot:
                rows = rng.sample(sorted(rows, key=lambda r: r["qid"]), min(3, len(rows)))
                if len(rows) < 3:
                    missing.append(f"{bench}/{arm}: {len(rows)} of 3 (pilot)")
            elif len(rows) != want:
                missing.append(f"{bench}/{arm}: {len(rows)} of {want}")
            recs += [{**r, "arm": f"{arm}@{reader}", "reader": reader} for r in rows]
        for arm in ARMS:
            if arm not in cells:
                missing.append(f"{bench}/{arm}: not built")
        out[bench] = recs
    return {"recs": out, "missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True, choices=sorted(READERS))
    ap.add_argument("--run", required=True)
    ap.add_argument("--pilot", action="store_true", help="3 rows per cell, from whatever is built")
    ap.add_argument("--force", action="store_true", help="overwrite an existing queue")
    args = ap.parse_args()
    res = build(args.reader, args.pilot)
    if res["missing"] and not args.pilot:
        raise SystemExit("v6 contexts incomplete:\n  " + "\n  ".join(res["missing"]))
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    manifest = {"reader": args.reader, "pilot": args.pilot, "missing": res["missing"], "cells": {}}
    for bench, recs in res["recs"].items():
        path = run / f"contexts_{bench}.jsonl"
        if path.exists() and not args.force:
            raise SystemExit(f"{path} exists; pass --force to rebuild it")
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        manifest["cells"][bench] = dict(Counter(r["arm"] for r in recs))
    json.dump(manifest, open(run / "contexts.manifest.json", "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
