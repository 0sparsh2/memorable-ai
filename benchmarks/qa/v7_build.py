"""
prereg v7 (the answer prompt): queue A7's contexts under prompt variants, plus the D8 guard.

    python -m benchmarks.qa.v7_build --run benchmarks/qa/runs/v7ans            # full run
    python -m benchmarks.qa.v7_build --run benchmarks/qa/runs/v7ans_pilot --pilot

Source: the A7 rows v6 answered (runs/v6ans/contexts_*.jsonl, arm "A7@<reader>"), byte for byte.
- A7p1, A7p2: every A7 row with prompt p1 / p2.
- A7r: the drift check, 40 LoCoMo-dev and 20 LME-dev A7 rows (stratified by category, seed V7_SEED),
  prompt p0.
- D8p0, D8p1, D8p2: LME-dev questions except single-session-preference, each with the whole A7
  context of the next LME-dev haystack in sorted order, judged as unanswerable.
Dev rows only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.qa.prompts import ANSWER_PROMPTS  # noqa: E402
from benchmarks.qa.v6_build import DRIFT, dev, stratified_exact  # noqa: E402

RUNS = ROOT / "benchmarks" / "qa" / "runs"
V6ANS = RUNS / "v6ans"
V7_SEED = 20261124
READER = "ultra"
PROMPTS = ("p1", "p2")
D8_PROMPTS = ("p0", "p1", "p2")
D8_SKIP = {"single-session-preference"}
D8_EXPLANATION = ("The memories shown come from a different user's conversations and do not contain the "
                  "information this question asks about.")
EXPECTED = {"locomo": 347, "lme": 150}
EXPECTED_D8 = 125


def a7_rows(bench: str, src: Path = V6ANS) -> list:
    """v6's queued A7 rows for one benchmark, first copy per question, dev rows only."""
    out, seen = [], set()
    for r in map(json.loads, open(src / f"contexts_{bench}.jsonl")):
        if r["arm"] == f"A7@{READER}" and dev(bench, r) and r["qid"] not in seen:
            seen.add(r["qid"])
            out.append(r)
    return out


def d8_rows(lme: list) -> list:
    """Each non-preference LME-dev question with the next haystack's whole A7 context (sorted order)."""
    by_group = {r["group"]: r for r in lme}  # LME: one question per haystack
    order = sorted(by_group)
    out = []
    for r in lme:
        if r["category"] in D8_SKIP:
            continue
        other = by_group[order[(order.index(r["group"]) + 1) % len(order)]]
        out.append({**r, "context": other["context"], "answer": D8_EXPLANATION, "abstention": True,
                    "d8_from": other["group"]})
    return out


def build(pilot: bool, src: Path = V6ANS) -> dict:
    rng = random.Random(V7_SEED)
    recs, missing = {}, []
    for bench in ("locomo", "lme"):
        a7 = a7_rows(bench, src)
        if len(a7) != EXPECTED[bench]:
            missing.append(f"{bench}/A7: {len(a7)} of {EXPECTED[bench]}")
        cells = {f"A7{p}": [{**r, "prompt": p} for r in a7] for p in PROMPTS}
        cells["A7r"] = [{**r, "prompt": "p0"} for r in
                        stratified_exact(a7, DRIFT[bench], V7_SEED, key=lambda r: r["category"])]
        if bench == "lme":
            d8 = d8_rows(a7)
            if len(d8) != EXPECTED_D8:
                missing.append(f"lme/D8: {len(d8)} of {EXPECTED_D8}")
            for p in D8_PROMPTS:
                cells[f"D8{p}"] = [{**r, "prompt": p} for r in d8]
        out = []
        for arm, rows in sorted(cells.items()):
            if pilot:
                rows = rng.sample(sorted(rows, key=lambda r: r["qid"]), min(3, len(rows)))
            out += [{**r, "arm": f"{arm}@{READER}", "reader": READER} for r in rows]
        recs[bench] = out
    return {"recs": recs, "missing": missing}


def checks(recs: dict, src: Path = V6ANS) -> list:
    """The dry run's structural checks (prereg v7, Stop rules); returns the failures."""
    bad = []
    for bench, rows in recs.items():
        a7 = {r["qid"]: r for r in a7_rows(bench, src)}
        for r in rows:
            arm = r["arm"].partition("@")[0]
            base = a7.get(r["qid"])
            if base is None:
                bad.append(f"{bench} {arm} {r['qid']}: not an A7 question")
                continue
            if r.get("prompt") not in ANSWER_PROMPTS:
                bad.append(f"{bench} {arm} {r['qid']}: unknown prompt {r.get('prompt')}")
            same = ("question", "date_line") if arm.startswith("D8") else ("context", "question", "date_line", "answer")
            for k in same:
                if r.get(k) != base.get(k):
                    bad.append(f"{bench} {arm} {r['qid']}: {k} differs from v6's A7")
            if arm.startswith("D8") and (r.get("d8_from") == r["group"] or r["context"] == base["context"]):
                bad.append(f"{bench} {arm} {r['qid']}: paired with its own haystack")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pilot", action="store_true", help="3 rows per cell")
    ap.add_argument("--force", action="store_true", help="overwrite an existing queue")
    args = ap.parse_args()
    res = build(args.pilot)
    bad = res["missing"] + checks(res["recs"])
    if bad:
        raise SystemExit("v7 queue refused:\n  " + "\n  ".join(bad[:20]))
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    manifest = {"reader": READER, "pilot": args.pilot, "seed": V7_SEED, "cells": {}}
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
