"""
prereg v8 report (the answer model): bars B22-B24, the drift check, and what is reported without a bar.

    python -m benchmarks.qa.v8_report --run benchmarks/qa/runs/v8ans [--final]

U·P1 (Ultra, P1) is v7's A7·P1 and D8·P1 answers. --final writes eval_plan/results-v8.md and prints
RESULTS: ok when every bar has a verdict.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.qa.v5_report import acc, fmt, fmtd, load, paired  # noqa: E402
from benchmarks.qa.v7_report import gain_verdict, refuses  # noqa: E402

RUNS = ROOT / "benchmarks" / "qa" / "runs"
V7ANS = RUNS / "v7ans"
METRIC = {"locomo": "j_strict", "lme": "j"}
PREVIEW = {"step"}  # readers NIM serves as previews: they lose B24's ties
N = {"locomo": 347, "lme": 150, "d8": 125, "drift_locomo": 40, "drift_lme": 20}


def arms_in(cells: dict) -> list:
    """(a7_arm, d8_arm, reader) for every v8 arm present: Ultra step-by-step first, then other readers."""
    out = [("A7p1s", "D8p1s", "ultra")] if any(k[1] == "A7p1s" for k in cells) else []
    out += sorted({("A7p1", "D8p1", rd) for (_, a, rd) in cells if a == "A7p1" and rd != "ultra"}, key=lambda t: t[2])
    return out


def label(a7: str, rd: str) -> str:
    return "U·P1s" if a7 == "A7p1s" else f"{rd}·P1"


def answer_times(run: Path) -> dict:
    """(arm, reader) -> sorted answer seconds, from the ledger."""
    out = defaultdict(list)
    path = run / "ledger_live.jsonl"
    for x in map(json.loads, open(path)) if path.exists() else ():
        if x.get("call") == "answer":
            arm, _, rd = x["arm"].partition("@")
            out[(arm, rd)].append(x["ms"] / 1000)
    return {k: sorted(v) for k, v in out.items()}


def report(run: Path, v7_run: Path = V7ANS) -> tuple:
    cells = load(run)
    v7 = load(v7_run)
    for key in (("locomo", "A7p1", "ultra"), ("lme", "A7p1", "ultra"), ("lme", "D8p1", "ultra")):
        if key in v7:
            cells[key] = v7[key]

    def cell(b, a, rd):
        return cells.get((b, a, rd), {})

    arms = arms_in(cells)
    complete = bool(arms) and all(
        len(cell("locomo", a7, rd)) >= N["locomo"] and len(cell("lme", a7, rd)) >= N["lme"]
        and len(cell("lme", d8, rd)) >= N["d8"] for a7, d8, rd in arms) and all(
        len(cell(b, "A7p1", "ultra")) >= N[b] for b in ("locomo", "lme")) and len(cell("locomo", "A7p1r", "ultra")) >= \
        N["drift_locomo"] and len(cell("lme", "A7p1r", "ultra")) >= N["drift_lme"]
    times = answer_times(run)

    lines = ["Baseline U·P1 rows are v7's A7·P1 and D8·P1 answers (Ultra, P1). Every reader answers with thinking on.", "",
             "## Cells", "",
             "| Bench | Arm | Reader | J-strict | J-lenient | J (LME) | refuses | median chars | answer s (p50 / p95) | unparsed |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    stop = []
    for (bench, arm, rd), rows in sorted(cells.items()):
        rs = list(rows.values())
        cs = {k: acc(rs, k) for k in ("j_strict", "j_lenient", "j")}
        unp = sum(c["unparsed"] for c in cs.values() if c)
        nj = sum(c["n"] for c in cs.values() if c)
        if nj and unp / nj > 0.02 and not (arm in ("A7p1", "D8p1") and rd == "ultra"):
            stop.append(f"{bench}/{arm}@{rd}: {unp}/{nj} unparsed judge verdicts (> 2%)")
        t = times.get((arm, rd), [])
        ts = f"{statistics.median(t):.0f} / {t[max(0, int(0.95 * len(t)) - 1)]:.0f}" if t else "—"
        chars = statistics.median(len(r.get("response") or "") for r in rs) if rs else 0
        lines.append(f"| {bench} | {arm} | {rd} | {fmt(cs['j_strict'])} | {fmt(cs['j_lenient'])} | {fmt(cs['j'])} | "
                     f"{sum(refuses(r.get('response')) for r in rs)} of {len(rs)} | {chars:.0f} | {ts} | {unp} |")

    verdict, d22, d23 = {}, {}, {}
    for a7, d8, rd in arms:
        lab = label(a7, rd)
        d22[lab] = {b: paired(cell(b, "A7p1", "ultra"), cell(b, a7, rd), METRIC[b]) for b in ("locomo", "lme")}
        d23[lab] = paired(cell("lme", "D8p1", "ultra"), cell("lme", d8, rd), "j")
        if complete and all(d22[lab].values()):
            verdict[f"B22 {lab}"] = gain_verdict(d22[lab])
        if complete and d23[lab]:
            verdict[f"B23 {lab}"] = "pass" if d23[lab]["diff"] >= -0.05 else "miss"
    choice, runner = None, None
    if complete and len(verdict) == 2 * len(arms):
        ok = [(a7, rd) for a7, _, rd in arms
              if verdict[f"B22 {label(a7, rd)}"] == "pass" and verdict[f"B23 {label(a7, rd)}"] == "pass"]
        ok.sort(key=lambda t: -acc(list(cell("lme", t[0], t[1]).values()), "j")["acc"])
        choice = label(*ok[0]) if ok else "U·P1"
        if len(ok) >= 2:
            runner = paired(cell("lme", *ok[0]), cell("lme", *ok[1]), "j")  # runner-up - best
            if runner["lo"] <= 0 <= runner["hi"]:
                def tie_key(t):
                    tt = times.get(t, [])
                    return (t[1] in PREVIEW, statistics.median(tt) if tt else float("inf"))
                choice = label(*min(ok[:2], key=tie_key))
        verdict["B24"] = choice
    drift = {}
    for b in ("locomo", "lme"):
        a, r = cell(b, "A7p1", "ultra"), cell(b, "A7p1r", "ultra")
        both = [q for q in r if q in a and METRIC[b] in r[q] and METRIC[b] in a[q]]
        if both:
            drift[b] = {"agree": sum(bool(r[q][METRIC[b]]) == bool(a[q][METRIC[b]]) for q in both) / len(both),
                        "n": len(both)}
    drift_ok = all(v["agree"] >= 0.9 for v in drift.values()) if drift else None

    lines += ["", "## Bars", "", "| Bar | Result | Verdict |", "|---|---|---|"]
    for lab in d22:
        lines.append(f"| B22 {lab} − U·P1 | LoCoMo-dev J-strict {fmtd(d22[lab]['locomo'])}; LME-dev J "
                     f"{fmtd(d22[lab]['lme'])} | {verdict.get(f'B22 {lab}', 'pending')} |")
    for lab in d23:
        lines.append(f"| B23 D8 {lab} − U·P1 (declined) | LME-dev {fmtd(d23[lab])} | {verdict.get(f'B23 {lab}', 'pending')} |")
    lines.append("| B24 choice | " + (f"runner-up − best on LME-dev J {fmtd(runner)}" if runner else "—")
                 + f" | {('reader = ' + choice) if choice else 'pending'} |")
    lines.append("| Drift (U·P1r vs v7's A7·P1) | " + "; ".join(f"{b} {100 * v['agree']:.0f}% of {v['n']}"
                                                            for b, v in drift.items())
                 + f" | {'ok' if drift_ok else ('re-answer U·P1' if drift_ok is False else 'pending')} |")

    cols = [("A7p1", "ultra", "U·P1")] + [(a7, rd, label(a7, rd)) for a7, _, rd in arms]
    lines += ["", "## Reported, no bar", "", "| Bench | Category | " + " | ".join(c[2] for c in cols) + " |",
              "|---|---|" + "---|" * len(cols)]
    for b in ("locomo", "lme"):
        by = defaultdict(lambda: defaultdict(list))
        for a7, rd, lab in cols:
            for r in cell(b, a7, rd).values():
                by[r.get("category")][lab].append(r)
        for c in sorted(by, key=str):
            lines.append(f"| {b} | {c} | " + " | ".join(fmt(acc(by[c][lab], METRIC[b])) for _, _, lab in cols) + " |")
    return lines, verdict, complete, stop, drift_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--v7", default=str(V7ANS))
    args = ap.parse_args()
    lines, verdict, complete, stop, drift_ok = report(Path(args.run), Path(args.v7))
    print("\n".join(lines))
    for s in stop:
        print(f"STOP: {s}")
    if args.final:
        if not complete or "B24" not in verdict or drift_ok is None:
            raise SystemExit("RESULTS: incomplete (cells or bars pending)")
        path = ROOT / "benchmarks" / "qa" / "eval_plan" / "results-v8.md"
        path.write_text("# Results, prereg v8 (the answer model)\n\n<!-- takeaways: written by hand -->\n\n"
                        + "\n".join(lines) + "\n")
        print(f"RESULTS: ok ({', '.join(f'{k} {v}' for k, v in sorted(verdict.items()))}; drift "
              f"{'ok' if drift_ok else 're-answer U·P1'}) -> {path}")


if __name__ == "__main__":
    main()
