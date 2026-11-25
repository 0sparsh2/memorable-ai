"""
prereg v7 report (the answer prompt): bars B19-B21, the drift check, and what is reported without a bar.

    python -m benchmarks.qa.v7_report --run benchmarks/qa/runs/v7ans [--final]

A7·P0 is v6's A7 answers (same contexts, reader and judge). --final writes eval_plan/results-v7.md and
prints RESULTS: ok when every bar has a verdict.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.qa.v5_report import acc, fmt, fmtd, load, paired  # noqa: E402

RUNS = ROOT / "benchmarks" / "qa" / "runs"
V6ANS = RUNS / "v6ans"
READER = "ultra"
METRIC = {"locomo": "j_strict", "lme": "j"}
PROMPTS = ("p1", "p2")
EXPECTED = {("locomo", "A7p1"): 347, ("locomo", "A7p2"): 347, ("locomo", "A7r"): 40,
            ("lme", "A7p1"): 150, ("lme", "A7p2"): 150, ("lme", "A7r"): 20,
            ("lme", "D8p0"): 125, ("lme", "D8p1"): 125, ("lme", "D8p2"): 125}
# The refusal pattern of pilot-notes-v6 ("After the results"), over a response's first 400 characters.
REFUSAL = re.compile(r"\b(I don't (know|have)|no specific|nothing specific|not in my memor|"
                     r"there's no (specific )?(discussion|information|mention)|do(es)? not (contain|mention))", re.I)


def refuses(response: str) -> bool:
    return bool(REFUSAL.search((response or "")[:400]))


def gain_verdict(d: dict) -> str:
    """B15's form: >= +2 with CI > 0 on one benchmark, no drop > 1 on the other."""
    up = [b for b, v in d.items() if v["diff"] >= 0.02 and v["lo"] > 0]
    return "pass" if up and all(v["diff"] >= -0.01 for b, v in d.items() if b not in up) else "miss"


def report(run: Path, v6_run: Path = V6ANS) -> tuple:
    cells = {(b, a): rows for (b, a, r), rows in load(run).items() if r == READER}
    v6 = load(v6_run)
    for b in ("locomo", "lme"):
        if (b, "A7", READER) in v6:
            cells[(b, "A7p0")] = v6[(b, "A7", READER)]
    complete = all(len(cells.get(k, {})) >= n for k, n in EXPECTED.items()) and all(
        (b, "A7p0") in cells for b in ("locomo", "lme"))

    def cell(b, a):
        return cells.get((b, a), {})

    lines = [f"Reader: **{READER}**. A7·P0 rows are v6's A7 answers (same contexts, reader and judge).", "",
             "## Cells", "",
             "| Bench | Arm | J-strict | J-lenient | J (LME) | refuses | median chars | unparsed |",
             "|---|---|---|---|---|---|---|---|"]
    stop = []
    for (bench, arm), rows in sorted(cells.items()):
        rs = list(rows.values())
        cs = {k: acc(rs, k) for k in ("j_strict", "j_lenient", "j")}
        unp = sum(c["unparsed"] for c in cs.values() if c)
        nj = sum(c["n"] for c in cs.values() if c)
        if nj and unp / nj > 0.02 and arm != "A7p0":
            stop.append(f"{bench}/{arm}: {unp}/{nj} unparsed judge verdicts (> 2%)")
        ref = sum(refuses(r.get("response")) for r in rs)
        chars = statistics.median(len(r.get("response") or "") for r in rs) if rs else 0
        lines.append(f"| {bench} | {arm} | {fmt(cs['j_strict'])} | {fmt(cs['j_lenient'])} | {fmt(cs['j'])} | "
                     f"{ref} of {len(rs)} | {chars:.0f} | {unp} |")

    b19 = {p: {b: paired(cell(b, "A7p0"), cell(b, f"A7{p}"), METRIC[b]) for b in ("locomo", "lme")} for p in PROMPTS}
    b20 = {p: paired(cell("lme", "D8p0"), cell("lme", f"D8{p}"), "j") for p in PROMPTS}
    verdict = {}
    for p in PROMPTS:
        if complete and all(b19[p].values()):
            verdict[f"B19 {p}"] = gain_verdict(b19[p])
        if complete and b20[p]:
            verdict[f"B20 {p}"] = "pass" if b20[p]["diff"] >= -0.05 else "miss"
    choice, p12 = None, None
    if complete and len(verdict) == 2 * len(PROMPTS):
        ok = [p for p in PROMPTS if verdict[f"B19 {p}"] == "pass" and verdict[f"B20 {p}"] == "pass"]
        if len(ok) == 2:
            p12 = paired(cell("lme", "A7p1"), cell("lme", "A7p2"), "j")  # p2 - p1
            choice = "p1" if p12["lo"] <= 0 <= p12["hi"] else ("p2" if p12["diff"] > 0 else "p1")
        else:
            choice = ok[0] if ok else "p0"
        verdict["B21"] = choice
    drift = {}
    for b in ("locomo", "lme"):
        a, r = cell(b, "A7p0"), cell(b, "A7r")
        both = [q for q in r if q in a and METRIC[b] in r[q] and METRIC[b] in a[q]]
        if both:
            agree = sum(bool(r[q][METRIC[b]]) == bool(a[q][METRIC[b]]) for q in both)
            drift[b] = {"agree": agree / len(both), "n": len(both)}
    drift_ok = all(v["agree"] >= 0.9 for v in drift.values()) if drift else None

    lines += ["", "## Bars", "", "| Bar | Result | Verdict |", "|---|---|---|"]
    for p in PROMPTS:
        lines.append(f"| B19 A7·{p.upper()} − A7·P0 | LoCoMo-dev J-strict {fmtd(b19[p]['locomo'])}; LME-dev J "
                     f"{fmtd(b19[p]['lme'])} | {verdict.get(f'B19 {p}', 'pending')} |")
    for p in PROMPTS:
        lines.append(f"| B20 D8·{p.upper()} − D8·P0 (declined) | LME-dev {fmtd(b20[p])} | "
                     f"{verdict.get(f'B20 {p}', 'pending')} |")
    lines.append(f"| B21 choice | " + (f"P2 − P1 on LME-dev J {fmtd(p12)}" if p12 else "—")
                 + f" | {('prompt = ' + choice) if choice else 'pending'} |")
    lines.append(f"| Drift (A7r vs v6's A7) | " + "; ".join(f"{b} {100 * v['agree']:.0f}% of {v['n']}"
                                                        for b, v in drift.items())
                 + f" | {'ok' if drift_ok else ('re-answer A7·P0' if drift_ok is False else 'pending')} |")

    lines += ["", "## Reported, no bar", "", "| Bench | Category | A7·P0 | A7·P1 | A7·P2 |", "|---|---|---|---|---|"]
    for b in ("locomo", "lme"):
        by = defaultdict(lambda: defaultdict(list))
        for arm in ("A7p0", "A7p1", "A7p2"):
            for r in cell(b, arm).values():
                by[r.get("category")][arm].append(r)
        for c in sorted(by, key=str):
            lines.append(f"| {b} | {c} | " + " | ".join(fmt(acc(by[c][a], METRIC[b])) for a in ("A7p0", "A7p1", "A7p2"))
                         + " |")
    lines += ["", "| D8 category (declined) | D8·P0 | D8·P1 | D8·P2 |", "|---|---|---|---|"]
    by = defaultdict(lambda: defaultdict(list))
    for arm in ("D8p0", "D8p1", "D8p2"):
        for r in cell("lme", arm).values():
            by[r.get("category")][arm].append(r)
    for c in sorted(by, key=str):
        lines.append(f"| {c} | " + " | ".join(fmt(acc(by[c][a], "j")) for a in ("D8p0", "D8p1", "D8p2")) + " |")
    return lines, verdict, complete, stop, drift_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--v6", default=str(V6ANS), help="the v6 run whose A7 answers are A7·P0")
    args = ap.parse_args()
    lines, verdict, complete, stop, drift_ok = report(Path(args.run), Path(args.v6))
    print("\n".join(lines))
    for s in stop:
        print(f"STOP: {s}")
    if args.final:
        if not complete or "B21" not in verdict or drift_ok is None:
            raise SystemExit("RESULTS: incomplete (cells or bars pending)")
        path = ROOT / "benchmarks" / "qa" / "eval_plan" / "results-v7.md"
        path.write_text("# Results, prereg v7 (the answer prompt)\n\n<!-- takeaways: written by hand -->\n\n"
                        + "\n".join(lines) + "\n")
        print(f"RESULTS: ok ({', '.join(f'{k} {v}' for k, v in sorted(verdict.items()))}; drift "
              f"{'ok' if drift_ok else 're-answer A7·P0'}) -> {path}")


if __name__ == "__main__":
    main()
