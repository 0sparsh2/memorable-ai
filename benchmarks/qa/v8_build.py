"""
prereg v8 (the answer model): probe the candidate readers, then queue A7·P1's contexts for each arm.

    python -m benchmarks.qa.v8_build probe                                   # availability and latency only
    python -m benchmarks.qa.v8_build build --readers kimi,mistral,minimax --run benchmarks/qa/runs/v8ans [--pilot]

Source: v7's queued A7·P1 and D8·P1 rows (runs/v7ans/contexts_*.jsonl), byte for byte.
- A7p1s@ultra, D8p1s@ultra: Ultra with P1s (step-by-step).
- A7p1@<reader>, D8p1@<reader>: each other reader with P1.
- A7p1r@ultra: the drift check, 40 LoCoMo-dev and 20 LME-dev A7·P1 rows (stratified, seed V8_SEED), P1.
Dev rows only. The probe never prints or judges a response.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.qa.e2e_qa import ANSWER_MAX_TOKENS, READERS, _complete, answer_prompt  # noqa: E402
from benchmarks.qa.prompts import ANSWER_PROMPTS  # noqa: E402
from benchmarks.qa.v6_build import DRIFT, stratified_exact  # noqa: E402

RUNS = ROOT / "benchmarks" / "qa" / "runs"
V7ANS = RUNS / "v7ans"
V8_SEED = 20261125
BASE = "ultra"
EXPECTED = {"locomo": 347, "lme": 150}
EXPECTED_D8 = 125
PROBE_READERS = ("kimi", "mistral", "minimax", "step")


def v7_rows(bench: str, arm: str, src: Path = V7ANS) -> list:
    """v7's queued rows for one arm ("A7p1" or "D8p1", Ultra), first copy per question."""
    out, seen = [], set()
    path = src / f"contexts_{bench}.jsonl"
    for r in map(json.loads, open(path)) if path.exists() else ():
        if r["arm"] == f"{arm}@{BASE}" and r["qid"] not in seen:
            seen.add(r["qid"])
            out.append(r)
    return out


def build(readers: list, pilot: bool, src: Path = V7ANS) -> dict:
    rng = random.Random(V8_SEED)
    recs, missing = {}, []
    for bench in ("locomo", "lme"):
        a7 = v7_rows(bench, "A7p1", src)
        d8 = v7_rows(bench, "D8p1", src) if bench == "lme" else []
        if len(a7) != EXPECTED[bench]:
            missing.append(f"{bench}/A7p1: {len(a7)} of {EXPECTED[bench]}")
        if bench == "lme" and len(d8) != EXPECTED_D8:
            missing.append(f"lme/D8p1: {len(d8)} of {EXPECTED_D8}")
        cells = {("A7p1s", BASE, "p1s"): a7, ("D8p1s", BASE, "p1s"): d8,
                 ("A7p1r", BASE, "p1"): stratified_exact(a7, DRIFT[bench], V8_SEED, key=lambda r: r["category"])}
        for rd in readers:
            cells[("A7p1", rd, "p1")] = a7
            cells[("D8p1", rd, "p1")] = d8
        out = []
        for (arm, rd, prompt), rows in sorted(cells.items()):
            if not rows:
                continue
            if pilot:
                rows = rng.sample(sorted(rows, key=lambda r: r["qid"]), min(3, len(rows)))
            out += [{**r, "arm": f"{arm}@{rd}", "reader": rd, "prompt": prompt} for r in rows]
        recs[bench] = out
    return {"recs": recs, "missing": missing}


def checks(recs: dict, readers: list, src: Path = V7ANS) -> list:
    """The dry run's structural checks (prereg v8, Stop rules); returns the failures."""
    bad = [f"reader {rd} not in READERS" for rd in readers if rd not in READERS]
    for bench, rows in recs.items():
        base = {("A7", r["qid"]): r for r in v7_rows(bench, "A7p1", src)}
        base.update({("D8", r["qid"]): r for r in v7_rows(bench, "D8p1", src)})
        for r in rows:
            b = base.get((r["arm"][:2], r["qid"]))
            if b is None:
                bad.append(f"{bench} {r['arm']} {r['qid']}: no v7 row")
                continue
            if r["prompt"] not in ANSWER_PROMPTS:
                bad.append(f"{bench} {r['arm']} {r['qid']}: unknown prompt")
            for k in ("context", "question", "date_line", "answer", "abstention"):
                if r.get(k) != b.get(k):
                    bad.append(f"{bench} {r['arm']} {r['qid']}: {k} differs from v7")
    return bad


async def _probe_one(name: str, prompt: str) -> dict:
    t = time.perf_counter()
    try:
        text, usage = await _complete([{"role": "user", "content": prompt}], effort="low",
                                      max_tokens=ANSWER_MAX_TOKENS, model=READERS[name])
        status = "ok" if text else "empty"
    except Exception as e:  # noqa: BLE001 - the probe records any failure
        usage, status = {}, f"error: {type(e).__name__}: {str(e)[:120]}"
    return {"reader": name, "model": READERS[name].split("/", 1)[1], "status": status,
            "seconds": round(time.perf_counter() - t, 1), "in_tokens": usage.get("in"),
            "out_tokens": usage.get("out"), "served_model": usage.get("served_model")}


async def probe(readers=PROBE_READERS) -> list:
    """Every candidate gets the same A7·P1 prompt: the first LME-dev question in sorted qid order."""
    row = sorted(v7_rows("lme", "A7p1"), key=lambda r: r["qid"])[0]
    prompt = answer_prompt({**row, "prompt": "p1"})
    return list(await asyncio.gather(*(_probe_one(n, prompt) for n in readers)))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("--readers", default=",".join(PROBE_READERS), help="re-probe some readers (one retry)")
    b = sub.add_parser("build")
    b.add_argument("--readers", required=True, help="the non-Ultra arms, e.g. kimi,mistral,minimax")
    b.add_argument("--run", required=True)
    b.add_argument("--pilot", action="store_true", help="3 rows per cell")
    b.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "probe":
        os.environ.setdefault("E2E_QA_HTTP_TIMEOUT", "600")
        res = asyncio.run(probe([r for r in args.readers.split(",") if r]))
        out = ROOT / "benchmarks" / "qa" / "eval_plan" / "probe-v8.json"
        log = json.load(open(out)) if out.exists() else {}
        attempts = log.get("attempts") or ([{"ts": log["ts"], "results": log["results"]}] if log else [])
        attempts.append({"ts": time.strftime("%Y-%m-%d %H:%M %Z"), "results": res})
        json.dump({"attempts": attempts}, open(out, "w"), indent=1)
        print(json.dumps(res, indent=1))
        return
    readers = [r for r in args.readers.split(",") if r]
    res = build(readers, args.pilot)
    bad = res["missing"] + checks(res["recs"], readers)
    if bad:
        raise SystemExit("v8 queue refused:\n  " + "\n  ".join(bad[:20]))
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    manifest = {"readers": [BASE] + readers, "pilot": args.pilot, "seed": V8_SEED, "cells": {}}
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
