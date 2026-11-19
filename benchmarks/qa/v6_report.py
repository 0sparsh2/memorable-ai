"""
prereg v6 report: bars B15-B18, the drift check, and what is reported without a bar.

    python -m benchmarks.qa.v6_report --run benchmarks/qa/runs/v6ans [--final]

A3 and A0 answers are v5's (same reader, same questions). --final writes eval_plan/results-v6.md
and prints RESULTS: ok when every bar has a verdict.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.qa.v5_report import acc, fmt, fmtd, load, paired  # noqa: E402

RUNS = ROOT / "benchmarks" / "qa" / "runs"
V5, V6 = RUNS / "v5", RUNS / "v6"
METRIC = {"locomo": "j_strict", "lme": "j"}
EXPECTED = {("locomo", "A3g"): 347, ("locomo", "A7"): 347, ("locomo", "D7"): 347, ("locomo", "A3r"): 40,
            ("lme", "A3g"): 150, ("lme", "A7"): 150, ("lme", "D7"): 150, ("lme", "A3r"): 20}
LOG_TOKENS = 30_000


def observer_cost() -> dict:
    """B17 from the observer caches: calls per session, unparseable replies, thinking text."""
    from benchmarks.qa.e2e_qa import LOCOMO_DEV, _safe, lme_items, locomo_items

    sessions = {_safe(it["group"]): len(it["_sessions"]) for it in locomo_items() if it["group"] in LOCOMO_DEV}
    sessions.update({_safe(it["group"]): sum(1 for s in it["_x"]["haystack_sessions"] if any(t.get("content") for t in s))
                     for it in lme_items("dev")})
    tot = defaultdict(int)
    per = {}
    for bench in ("locomo", "lme"):
        for done in sorted((V6 / "ingest_cache" / bench).glob("*.obs.done")):
            group = done.name[: -len(".obs.done")]
            d = json.loads(done.read_text())
            llm = d.get("llm", {})
            n = sessions.get(group, 0)
            per[f"{bench}/{group}"] = {"calls": llm.get("calls", 0), "sessions": n, "ingest_s": d.get("ingest_s")}
            for k in ("calls", "json_failed", "json_repaired", "thinking", "empty", "gave_up", "retries",
                      "prompt_tokens", "completion_tokens"):
                tot[k] += int(llm.get(k, 0) or 0)
            tot["sessions"] += n
            tot["ingest_s"] += float(d.get("ingest_s") or 0)
            tot["groups"] += 1
    return {"total": dict(tot), "per": per}


def log_sizes() -> dict:
    """Unbudgeted log tokens per group, and whether the 30k budget cut lines."""
    from memorable_ai.core.context import estimate_tokens, render_observation_log

    out = {}
    for bench in ("locomo", "lme"):
        for db in sorted((V6 / "ingest_cache" / bench).glob("*.obs.db")):
            if not db.with_suffix(".done").exists():
                continue
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            obs = [{"id": i, "content": t, "metadata": json.loads(m) if m else {}} for i, t, m in c.execute(
                "select id, content, extra_metadata from memories where memory_type='observation' "
                "and invalidated_at is null")]
            c.close()
            full = estimate_tokens(render_observation_log(obs))
            out[f"{bench}/{db.name[:-len('.obs.db')]}"] = {"tokens": full, "cut": full > LOG_TOKENS}
    return out


def report(run: Path, v5_run: Path = V5) -> tuple:
    cells = load(run)
    v5 = load(v5_run)
    reader = next((r for (_, _, r) in cells), None)
    for b in ("locomo", "lme"):  # v5's answers with the same reader
        for arm in ("A3", "A0"):
            if (b, arm, reader) in v5:
                cells[(b, arm, reader)] = v5[(b, arm, reader)]
    complete = all(len(cells.get((b, a, reader), {})) >= n for (b, a), n in EXPECTED.items())
    lines = [f"Reader: **{reader}** (v5's B11 choice). A3 and A0 rows are v5's answers.", "",
             "## Cells", "", "| Bench | Context | J-strict | J-lenient | J (LME) | unparsed |", "|---|---|---|---|---|---|"]
    stop = []
    for (bench, arm, rd), rows in sorted(cells.items()):
        if rd != reader:
            continue
        rs = list(rows.values())
        cs = {k: acc(rs, k) for k in ("j_strict", "j_lenient", "j")}
        unp = sum(c["unparsed"] for c in cs.values() if c)
        nj = sum(c["n"] for c in cs.values() if c)
        if nj and unp / nj > 0.02 and arm not in ("A3", "A0"):
            stop.append(f"{bench}/{arm}: {unp}/{nj} unparsed judge verdicts (> 2%)")
        lines.append(f"| {bench} | {arm} | {fmt(cs['j_strict'])} | {fmt(cs['j_lenient'])} | {fmt(cs['j'])} | {unp} |")

    def cell(b, a):
        return cells.get((b, a, reader), {})

    d = {name: {b: paired(cell(b, x), cell(b, y), METRIC[b]) for b in ("locomo", "lme")}
         for name, x, y in (("B15", "A3g", "A7"), ("B16", "A3g", "D7"), ("B18", "A3", "A3g"))}
    verdict = {}
    if complete and all(d["B15"].values()):
        up = [b for b, v in d["B15"].items() if v["diff"] >= 0.02 and v["lo"] > 0]
        verdict["B15"] = "pass" if up and all(v["diff"] >= -0.01 for b, v in d["B15"].items() if b not in up) else "miss"
    if complete and all(d["B16"].values()):
        verdict["B16"] = "pass" if all(v["diff"] <= 0.01 for v in d["B16"].values()) else "miss"
    if complete and all(d["B18"].values()):
        verdict["B18"] = "pass" if all(v["diff"] >= -0.01 for v in d["B18"].values()) else "miss"
    cost = observer_cost()
    t = cost["total"]
    per_session = t.get("calls", 0) / t["sessions"] if t.get("sessions") else None
    unparse = t.get("json_failed", 0) / t["calls"] if t.get("calls") else None
    if complete and per_session is not None:
        verdict["B17"] = "pass" if per_session <= 2.5 and unparse <= 0.02 and not t.get("thinking") else "miss"
    drift = {}
    for b in ("locomo", "lme"):
        a, r = cell(b, "A3"), cell(b, "A3r")
        both = [q for q in r if q in a and METRIC[b] in r[q] and METRIC[b] in a[q]]
        if both:
            agree = sum(bool(r[q][METRIC[b]]) == bool(a[q][METRIC[b]]) for q in both)
            drift[b] = {"agree": agree / len(both), "n": len(both)}
    drift_ok = all(v["agree"] >= 0.9 for v in drift.values()) if drift else None

    lines += ["", "## Bars", "", "| Bar | Result | Verdict |", "|---|---|---|"]
    for name, label in (("B15", "A7 − A3g"), ("B16", "D7 − A3g"), ("B18", "A3g − A3")):
        lines.append(f"| {name} {label} | LoCoMo-dev J-strict {fmtd(d[name]['locomo'])}; LME-dev J "
                     f"{fmtd(d[name]['lme'])} | {verdict.get(name, 'pending')} |")
    lines.append(f"| B17 observer | {per_session and round(per_session, 2)} calls per session; "
                 f"{t.get('json_failed', 0)} of {t.get('calls', 0)} unparseable; {t.get('thinking', 0)} with thinking "
                 f"| {verdict.get('B17', 'pending')} |")
    lines.append(f"| Drift (A3r vs v5's A3) | " + "; ".join(f"{b} {100 * v['agree']:.0f}% of {v['n']}"
                                                        for b, v in drift.items())
                 + f" | {'ok' if drift_ok else ('re-answer A3' if drift_ok is False else 'pending')} |")

    lines += ["", "## Reported, no bar", ""]
    for b in ("locomo", "lme"):
        x = paired(cell(b, "A0"), cell(b, "A7"), METRIC[b])
        lines.append(f"- **A7 − A0, {b}:** {fmtd(x)}")
    for b in ("locomo", "lme"):
        path = run / f"contexts_{b}.jsonl"
        sizes_by = defaultdict(list)
        for r in map(json.loads, open(path)) if path.exists() else ():
            sizes_by[r["arm"].partition("@")[0]].append(len(r.get("context", "")) // 4)
        for arm in ("A3g", "A7", "D7"):
            toks = sorted(sizes_by.get(arm, ()))
            if toks:
                lines.append(f"- **Context tokens, {b} {arm}:** median {toks[len(toks) // 2]:,}, max {toks[-1]:,}")
    cats = defaultdict(dict)
    for b in ("locomo", "lme"):
        for arm in ("A3", "A3g", "A7", "A0"):
            by = defaultdict(list)
            for r in cell(b, arm).values():
                by[r.get("category")].append(r)
            for c, rs in by.items():
                a = acc(rs, METRIC[b])
                if a:
                    cats[(b, c)][arm] = a
    lines += ["", "| Bench | Category | A3 | A3g | A7 | A0 |", "|---|---|---|---|---|---|"]
    for (b, c), v in sorted(cats.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        lines.append(f"| {b} | {c} | " + " | ".join(fmt(v.get(a)) for a in ("A3", "A3g", "A7", "A0")) + " |")
    sizes = log_sizes()
    if sizes:
        toks = sorted(v["tokens"] for v in sizes.values())
        lines += ["", f"- **Log size:** median {toks[len(toks) // 2]:,} tokens, max {toks[-1]:,}, over {len(toks)} "
                      f"logs; the {LOG_TOKENS:,}-token budget cut lines in {sum(v['cut'] for v in sizes.values())}."]
    if t:
        lines.append(f"- **Observer:** {t.get('calls', 0):,} calls over {t.get('groups', 0)} logs, "
                     f"{t.get('prompt_tokens', 0):,} prompt and {t.get('completion_tokens', 0):,} output tokens, "
                     f"{t.get('retries', 0)} retries, {t.get('ingest_s', 0) / 3600:.1f} h of ingest.")
    return lines, verdict, complete, stop, drift_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--v5", default=str(V5), help="the v5 run whose A3 and A0 answers are reused")
    args = ap.parse_args()
    lines, verdict, complete, stop, drift_ok = report(Path(args.run), Path(args.v5))
    print("\n".join(lines))
    for s in stop:
        print(f"STOP: {s}")
    if args.final:
        if not complete or len(verdict) < 4:
            raise SystemExit("RESULTS: incomplete (cells or bars pending)")
        path = ROOT / "benchmarks" / "qa" / "eval_plan" / "results-v6.md"
        path.write_text("# Results, prereg v6 (the observation log)\n\n<!-- takeaways: written by hand -->\n\n"
                        + "\n".join(lines) + "\n")
        print(f"RESULTS: ok ({', '.join(f'{k} {v}' for k, v in sorted(verdict.items()))}; drift "
              f"{'ok' if drift_ok else 're-answer A3'}) -> {path}")


if __name__ == "__main__":
    main()
