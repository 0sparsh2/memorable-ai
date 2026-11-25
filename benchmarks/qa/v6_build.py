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


def stratified_exact(items, n: int, seed: int, key) -> list:
    """Exactly n items, allocated to categories by largest remainder, shuffled within each (sorted qids)."""
    by = defaultdict(list)
    for it in items:
        by[key(it)].append(it)
    total = sum(len(v) for v in by.values())
    quota = {k: n * len(v) / total for k, v in by.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - alloc[k]), str(k)))[: n - sum(alloc.values())]:
        alloc[k] += 1
    rng = random.Random(seed)
    picked = []
    for k, group in sorted(by.items(), key=lambda kv: str(kv[0])):
        group = sorted(group, key=lambda r: r["qid"])
        rng.shuffle(group)
        picked += group[: alloc[k]]
    return picked


def drift_rows(bench: str, reader: str) -> list:
    """The A3 contexts v5 answered with this reader, sampled for the drift check."""
    a3 = [r for r in map(json.loads, open(V5 / f"contexts_{bench}.jsonl"))
          if r["arm"] == f"A3@{reader}" and dev(bench, r)]
    return stratified_exact(a3, DRIFT[bench], V6_SEED, key=lambda r: r["category"])


LOG_SEP = "\n\nRetrieved excerpts:\n"  # e2e_qa.memory_context: the log, then this, then A3g's context


def base_mismatches(cells: dict) -> list:
    """A7 and D7 rows whose excerpts differ from A3g's context for the same question (prereg v6: A7 = A3g + log).

    The device can flip an extractive-summary near-tie: CPU and MPS agree on 53 of 53 dev haystacks
    but differed on 1 of 26 held-out LME haystacks (2026-11-20), so this is checked, not assumed.
    """
    a3g = {r["qid"]: r["context"] for r in cells.get("A3g", ())}
    bad = []
    for arm in ("A7", "D7"):
        for r in cells.get(arm, ()):
            head, sep, base = r["context"].rpartition(LOG_SEP)
            if not sep or base != a3g.get(r["qid"]):
                bad.append(f"{arm} {r['group']} {r['qid']}")
    return bad


def build(reader: str, pilot: bool, only=None) -> dict:
    """``only``: the arms to queue (the rest are left for a later --append)."""
    out, missing = {}, []
    rng = random.Random(V6_SEED)
    for bench in ("locomo", "lme"):
        cells = v6_rows(bench)
        missing += [f"{bench}: excerpts differ from A3g: {m}" for m in base_mismatches(cells)]
        cells["A3r"] = drift_rows(bench, reader)
        if only:
            cells = {a: rows for a, rows in cells.items() if a in only}
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
        for arm in (a for a in ARMS if not only or a in only):
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
    ap.add_argument("--arms", default="", help="only these arms (A3g,A7,D7,A3r); the others can be --append-ed later")
    ap.add_argument("--append", action="store_true", help="add these arms to an existing queue")
    args = ap.parse_args()
    only = set(args.arms.split(",")) - {""} or None
    res = build(args.reader, args.pilot, only)
    if res["missing"] and not args.pilot:
        raise SystemExit("v6 contexts incomplete:\n  " + "\n  ".join(res["missing"]))
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    manifest = {"reader": args.reader, "pilot": args.pilot, "missing": res["missing"], "cells": {}}
    for bench, recs in res["recs"].items():
        path = run / f"contexts_{bench}.jsonl"
        have = [json.loads(line) for line in open(path)] if path.exists() else []
        if have and not (args.force or args.append):
            raise SystemExit(f"{path} exists; pass --append to add arms or --force to rebuild it")
        if args.append:
            queued = {r["arm"] for r in have}
            if any(r["arm"] in queued for r in recs) or any(r.get("reader") != args.reader for r in have):
                raise SystemExit(f"{path}: these arms are already queued, or the queue has another reader")
            recs = have + recs
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        manifest["cells"][bench] = dict(Counter(r["arm"] for r in recs))
    json.dump(manifest, open(run / "contexts.manifest.json", "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
