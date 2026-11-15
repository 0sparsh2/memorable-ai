"""
End-to-end QA eval on LoCoMo and LongMemEval-S (preregistered: benchmarks/qa/eval_plan/prereg.md).

Two stages, so the paid part only ever sees frozen inputs:

  retrieve   local and free: ingest through Memorable's real write pipeline (LLM-free),
             then write each arm's memory context per question (contexts.jsonl)
  answer     answer model, then judge(s), with a per-call cost ledger and the stop rules
             (budget cap, first error after retries); --stub costs nothing

  python -m benchmarks.qa.e2e_qa retrieve --bench locomo --arms A0,A1,A2,A3,A3w --run runs/main
  python -m benchmarks.qa.e2e_qa retrieve --bench lme --split dev --arms A1,A3 --run runs/dry --limit 3
  python -m benchmarks.qa.e2e_qa answer   --run runs/dry --stub
  python -m benchmarks.qa.e2e_qa decoys   --run runs/dry --stub
  python -m benchmarks.qa.e2e_qa report   --run runs/main

LongMemEval split: "dev" = the 150-question tuning sample (sample_questions(150, seed=13));
"heldout" = every other question, never inspected before the final run; "all" = both.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MEMORABLE_LLM__DEFAULT_MODEL", "none")  # LLM-free ingestion (prereg)

from benchmarks.qa.prompts import ANSWER_PROMPT, MEM0_JUDGE_SYSTEM, lme_judge_prompt, mem0_judge_prompt  # noqa: E402

LOCOMO = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
LME = ROOT / "benchmarks" / "longmemeval" / "data" / "longmemeval_s_cleaned.json"
LOCOMO_CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}

# Answer and judge model (prereg v1.1): Nemotron-3.5-Lightning (30B MoE, 3B active) on NVIDIA
# NIM's free tier. Answers think (chat_template_kwargs.enable_thinking=true, up to 8,192 tokens);
# the judge does not. DeepSeek V4.1 Flash (billion-context's eval model) answers correctly but the
# free tier queues it to ~1.4 calls/min, so it only runs an optional comparability slice.
# Gemini 3 Flash is the paid fallback, used only with the owner's approval.
MODEL = os.getenv("E2E_QA_MODEL", "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b")
PRICES = {  # USD per token
    "nvidia_nim/": {"in": 0.0, "cached": 0.0, "out": 0.0},  # free tier, rate-limited
    "gemini/gemini-3-flash-preview": {"in": 0.50e-6, "cached": 0.05e-6, "out": 3.00e-6},  # ai.google.dev, 2026-11-15
}
PRICE = next(v for k, v in PRICES.items() if MODEL.startswith(k))
CAP_USD = 60.0
TURN_CHARS = 2000  # the pipeline embeds content[:2000]; longer turns are cut in the prompt too
LENIENT_ARMS = {"A0", "A3", "A4"}  # Mem0's lenient judge, only to sit beside published LoCoMo numbers
# prereg v2: thinking answers get 8,192 output tokens. With 2,000, the pilot lost 4 of 30 answers to
# thinking that hit the cap before any reply was written.
ANSWER_MAX_TOKENS = 8192

ETTIN17 = "cross-encoder/ettin-reranker-17m-v1"
ARMS: Dict[str, Dict[str, Any]] = {
    "A0": {"full_context": True},
    "A1": {"fusion": "rrf", "semantic_weight": 0.0, "bm25_weight": 1.0, "rerank": False, "k_excerpts": 10,
           "k_episodes": 0},
    "A2": {"fusion": "rrf", "semantic_weight": 0.5, "bm25_weight": 1.0, "rerank": False, "k_excerpts": 10,
           "k_episodes": 3},
    "A3": {"fusion": "score", "rerank": True, "rerank_model": ETTIN17, "rerank_depth": 20, "k_excerpts": 10,
           "k_episodes": 3},
    "A3w": {"fusion": "score", "rerank": True, "rerank_model": ETTIN17, "rerank_depth": 20, "k_excerpts": 40,
            "k_episodes": 3},
    # A4 (fine-tuned) is added after the retrieval bars are read, before the pilot (prereg).
}


# ------------------------------------------------------------------ data


def _human(dt: Optional[datetime]) -> str:
    return dt.strftime("%A, %B %d, %Y") if dt else "unknown date"


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s) if s else None
    except ValueError:
        return None


def _lme_date(s: str) -> Optional[datetime]:
    """'2023/05/20 (Sat) 02:21' -> datetime."""
    m = re.match(r"(\d{4}/\d{2}/\d{2}) \(\w+\) (\d{2}:\d{2})", s or "")
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y/%m/%d %H:%M") if m else None


def locomo_items() -> List[Dict[str, Any]]:
    from benchmarks.locomo.loader import iter_qa, iter_sessions, parse_session_datetime, sample_id

    items = []
    for ci, sample in enumerate(json.load(open(LOCOMO))):
        conv_id = sample_id(sample, ci)
        conv = sample["conversation"]
        sessions = [(key, parse_session_datetime(date), turns) for key, date, turns in iter_sessions(conv)]
        for qi, qa in enumerate(iter_qa(sample, categories=set(LOCOMO_CATS))):
            items.append({"bench": "locomo", "qid": f"{conv_id}#{qi}", "group": conv_id, "question": qa.question,
                          "answer": str(qa.answer), "category": LOCOMO_CATS[qa.category], "abstention": False,
                          "split": "all", "date_line": "", "_sample": sample, "_sessions": sessions})
    return items


def lme_items(split: str) -> List[Dict[str, Any]]:
    from benchmarks.embeddings.bench_longmemeval import sample_questions

    dev_ids = {x["question_id"] for x in sample_questions(150)}
    out = []
    for x in json.load(open(LME)):
        s = "dev" if x["question_id"] in dev_ids else "heldout"
        if split != "all" and s != split:
            continue
        qdate = _lme_date(x.get("question_date", ""))
        out.append({"bench": "lme", "qid": x["question_id"], "group": x["question_id"], "question": x["question"],
                    "answer": str(x["answer"]), "category": x["question_type"],
                    "abstention": x["question_id"].endswith("_abs"), "split": s,
                    "date_line": f"Today's date: {_human(qdate)}.\n\n" if qdate else "", "_x": x})
    return out


def a0_lme_subset(items: List[Dict[str, Any]], n: int = 100, seed: int = 7) -> set:
    """Stratified by question type over held-out ids; uses only ids and types (prereg)."""
    by = defaultdict(list)
    for it in items:
        if it["split"] == "heldout":
            by[it["category"] + ("_abs" if it["abstention"] else "")].append(it["qid"])
    total = sum(len(v) for v in by.values())
    rng = random.Random(seed)
    picked = set()
    for _, ids in sorted(by.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        picked.update(ids[: max(1, round(n * len(ids) / total))])
    return picked


# ------------------------------------------------------------------ contexts

_RERANKERS: Dict[str, Any] = {}  # one loaded reranker per spec, shared by every per-haystack engine


def _line(dt: Optional[datetime], text: str) -> str:
    return f"({_human(dt)}) {text}"


def full_context(item: Dict[str, Any]) -> str:
    lines = []
    if item["bench"] == "locomo":
        from benchmarks.locomo.loader import format_turn_text

        for _, dt, turns in item["_sessions"]:
            lines.append(f"\n--- Conversation on {_human(dt)} ---")
            lines += [f"{t.get('speaker')}: {format_turn_text(t)}" for t in turns]
    else:
        x = item["_x"]
        order = sorted(range(len(x["haystack_sessions"])), key=lambda i: _lme_date(x["haystack_dates"][i]) or datetime.min)
        for i in order:
            lines.append(f"\n--- Conversation on {_human(_lme_date(x['haystack_dates'][i]))} ---")
            lines += [f"{t['role']}: {t['content']}" for t in x["haystack_sessions"][i]]
    return "\n".join(lines).strip()


def memory_context(item: Dict[str, Any], turns: List[Dict[str, Any]], episodes: List[Dict[str, Any]]) -> tuple:
    """Oldest first, dated, no ranks or scores (as Mem0 presents memories)."""
    rows, cut = [], 0
    for e in episodes:
        dt = _parse_iso(e.get("start_time"))
        rows.append((dt or datetime.min, -1, _line(dt, f"[summary of a past conversation] {e.get('title') or ''}: "
                                                      f"{e.get('summary') or ''}".strip())))
    for t in turns:
        dt = _parse_iso(t.get("occurred_at"))
        text = t.get("content") or ""
        if len(text) > TURN_CHARS:
            text, cut = text[:TURN_CHARS] + " [...]", cut + 1
        if item["bench"] == "lme":
            text = f"{t.get('role')}: {text}"
        rows.append((dt or datetime.min, t.get("seq") or 0, _line(dt, text)))
    rows.sort(key=lambda r: (r[0], r[1]))
    return "\n".join(r[2] for r in rows) or "(no memories found)", cut


async def _ingest(item: Dict[str, Any], db: Path):
    from memorable_ai import MemoryEngine

    eng = MemoryEngine(database=f"sqlite:///{db}", write_mode="sync")
    eng._ensure_components()
    uid = item["group"]
    if item["bench"] == "locomo":
        from benchmarks.locomo.loader import format_turn_text

        conv = item["_sample"]["conversation"]
        a = conv.get("speaker_a")
        for key, dt, turns in item["_sessions"]:
            msgs = [{"role": "user" if t.get("speaker") == a else "assistant",
                     "content": f"{t.get('speaker')}: {format_turn_text(t)}"} for t in turns]
            await eng.ingest(msgs, None, user_id=uid, session_id=key, occurred_at=dt)
            await eng.end_session(user_id=uid, session_id=key)
    else:
        x = item["_x"]
        for sid, date, sess in zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]):
            msgs = [{"role": t["role"], "content": t["content"]} for t in sess if t.get("content")]
            if msgs:
                await eng.ingest(msgs, None, user_id=uid, session_id=sid, occurred_at=_lme_date(date))
                await eng.end_session(user_id=uid, session_id=sid)
    return eng


async def _contexts_for_group(group_items, arms, tmp: Path, out) -> Dict[str, Any]:
    from memorable_ai.core.scope import Scope

    stats = {"ingest_s": 0.0, "truncated_turns": 0, "empty": defaultdict(int)}
    mem_arms = [a for a in arms if not ARMS[a].get("full_context")]
    for a in arms:
        if ARMS[a].get("full_context"):
            for it in group_items:
                if it.get("_a0", True):
                    ctx = full_context(it)
                    out.write(json.dumps({"arm": a, **_public(it), "context": ctx, "retrieval_ms": 0.0}) + "\n")
    if not mem_arms:
        return stats
    db = tmp / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', group_items[0]['group'])}.db"
    t0 = time.perf_counter()
    eng = await _ingest(group_items[0], db)
    stats["ingest_s"] += time.perf_counter() - t0
    r = eng._retrieval
    base = {k: getattr(r, k) for k in ("fusion", "semantic_weight", "bm25_weight", "rerank", "rerank_model",
                                       "rerank_depth", "rerank_blend")}
    try:
        for a in mem_arms:
            cfg = ARMS[a]
            for k, v in base.items():
                setattr(r, k, cfg.get(k, v))
            if cfg.get("rerank"):
                spec = cfg["rerank_model"]
                if spec not in _RERANKERS:
                    from memorable_ai.core.rerank import load_reranker

                    _RERANKERS[spec] = load_reranker(spec)
                r._reranker = _RERANKERS[spec]
            for it in group_items:
                t = time.perf_counter()
                res = await r.recall(it["question"], Scope(user_id=it["group"]), k_facts=0,
                                     k_episodes=cfg["k_episodes"], k_excerpts=cfg["k_excerpts"])
                ms = 1000 * (time.perf_counter() - t)
                ctx, cut = memory_context(it, res["excerpts"], res["episodes"])
                stats["truncated_turns"] += cut
                if not res["excerpts"] and not res["episodes"]:
                    stats["empty"][a] += 1
                out.write(json.dumps({"arm": a, **_public(it), "context": ctx, "retrieval_ms": round(ms, 1),
                                      "n_turns": len(res["excerpts"]), "n_episodes": len(res["episodes"])}) + "\n")
    finally:
        eng.close()
        for p in tmp.glob(db.name + "*"):
            p.unlink(missing_ok=True)
    return stats


def _public(it: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in it.items() if not k.startswith("_")}


async def cmd_retrieve(args) -> None:
    import resource

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    items = locomo_items() if args.bench == "locomo" else lme_items(args.split)
    if args.bench == "lme":
        a0 = a0_lme_subset(lme_items("all"))
        for it in items:  # A0 on LME: the held-out subset only, plus dev rows when pricing a pilot
            it["_a0"] = it["qid"] in a0 or (args.a0_dev and it["split"] == "dev")
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        groups[it["group"]].append(it)
    keys = list(groups)[: args.limit] if args.limit else list(groups)
    tmp = Path(tempfile.mkdtemp(prefix="memorable_e2e_"))
    totals = {"groups": 0, "items": 0, "ingest_s": 0.0, "truncated_turns": 0, "empty": defaultdict(int)}
    t0 = time.time()
    try:
        with open(run / f"contexts_{args.bench}.jsonl", "a") as out:
            for gi, g in enumerate(keys):
                s = await _contexts_for_group(groups[g], arms, tmp, out)
                out.flush()
                totals["groups"] += 1
                totals["items"] += len(groups[g])
                totals["ingest_s"] += s["ingest_s"]
                totals["truncated_turns"] += s["truncated_turns"]
                for a, n in s["empty"].items():
                    totals["empty"][a] += n
                if gi % 10 == 0:
                    print(f"  {args.bench}: {gi + 1}/{len(keys)} groups, {time.time() - t0:.0f}s", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    totals["wall_s"] = round(time.time() - t0)
    totals["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)  # macOS: bytes
    print(json.dumps({"retrieve": args.bench, **totals}, default=dict))


# ------------------------------------------------------------------ LLM stage


class StopRun(Exception):
    pass


class Ledger:
    def __init__(self, path: Path, cap: float):
        self.path, self.cap = path, cap
        self.spent = 0.0
        if path.exists():
            for line in open(path):
                self.spent += json.loads(line).get("cost", 0.0)
        self.f = open(path, "a")

    def add(self, rec: Dict[str, Any]) -> None:
        self.spent += rec["cost"]
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        if self.spent >= self.cap:
            raise StopRun(f"budget cap reached: ${self.spent:.2f} >= ${self.cap:.2f}")


def _cost(usage: Dict[str, int]) -> float:
    cached = usage.get("cached", 0)
    return (usage.get("in", 0) - cached) * PRICE["in"] + cached * PRICE["cached"] + usage.get("out", 0) * PRICE["out"]


class RateLimiter:
    """Spaces request starts to stay under a provider's requests-per-minute limit (NIM free tier)."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.next = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        if not self.interval:
            return
        async with self.lock:
            now = time.monotonic()
            delay = self.next - now
            self.next = max(now, self.next) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


LIMITER = RateLimiter(0)


_NIM: Dict[str, Any] = {}


def _nim_client():
    """Direct OpenAI-compatible calls to NIM (exact control of the thinking switch)."""
    if not _NIM:
        import httpx
        from dotenv import dotenv_values

        env = {**dotenv_values(ROOT / ".env"), **os.environ}
        _NIM["key"] = (env.get("NVIDIA_NIM_API_KEY") or env.get("NVIDIA_API_KEY") or "").strip()
        _NIM["base"] = (env.get("NVIDIA_NIM_API_BASE") or "https://integrate.api.nvidia.com/v1").rstrip("/")
        _NIM["http"] = httpx.AsyncClient(timeout=httpx.Timeout(240, connect=15))
    return _NIM


async def _complete(msgs, *, effort: str, max_tokens: int):
    """One completion -> (text, usage). effort: "low" = think (answers), "minimal" = no thinking (judge)."""
    if MODEL.startswith("nvidia_nim/"):
        nim = _nim_client()
        think = effort != "minimal"
        r = await nim["http"].post(
            f"{nim['base']}/chat/completions", headers={"Authorization": f"Bearer {nim['key']}"},
            json={"model": MODEL.split("/", 1)[1], "messages": msgs, "temperature": 0,
                  "max_tokens": max_tokens if not think else max(max_tokens, ANSWER_MAX_TOKENS),
                  "chat_template_kwargs": {"enable_thinking": think}})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        b = r.json()
        msg, u = b["choices"][0]["message"], b.get("usage") or {}
        return (msg.get("content") or "").strip(), {"in": int(u.get("prompt_tokens") or 0), "cached": 0,
                                                    "out": int(u.get("completion_tokens") or 0), "reasoning": 0}
    import litellm

    resp = await litellm.acompletion(model=MODEL, messages=msgs, temperature=0, max_tokens=max_tokens,
                                     reasoning_effort=effort, timeout=180)
    u = resp.usage
    ptd = getattr(u, "prompt_tokens_details", None)
    ctd = getattr(u, "completion_tokens_details", None)
    return (resp.choices[0].message.content or "").strip(), {
        "in": int(u.prompt_tokens or 0), "cached": int(getattr(ptd, "cached_tokens", 0) or 0),
        "out": int(u.completion_tokens or 0), "reasoning": int(getattr(ctd, "reasoning_tokens", 0) or 0)}


async def llm_call(prompt: str, *, system: Optional[str], effort: str, max_tokens: int, stub: Optional[str]):
    if stub is not None:
        return stub, {"in": len(prompt) // 4, "cached": 0, "out": len(stub) // 4, "reasoning": 0}
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    last = None
    for attempt in range(4):
        try:
            await LIMITER.wait()
            return await _complete(msgs, effort=effort, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001 - provider errors vary
            last = e
            msg = str(e).lower()
            if attempt < 3 and any(x in msg for x in ("429", "rate", "503", "502", "500", "overloaded", "timeout",
                                                       "timed out", "readtimeout")):
                await asyncio.sleep(5 * 2**attempt)
                continue
            break
    raise StopRun(f"LLM error after retries: {type(last).__name__}: {str(last)[:300]}")


def _stub_answer(ctx: str) -> str:
    for line in ctx.splitlines():
        if line.startswith("("):
            return line.split(") ", 1)[-1][:200]
    return "I don't know."


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def _stub_verdict(gold: str, response: str) -> bool:
    """Exact-match stub scorer: the normalised gold answer must appear in the response."""
    g = _norm(gold)
    return bool(g) and f" {g} " in f" {_norm(response)} "


def parse_yes_no(text: str) -> Optional[bool]:
    t = text.strip().lower()
    t = t.split("</judge_thinking>")[-1]
    if re.search(r"\byes\b", t[:20]) or t.startswith("yes"):
        return True
    if re.search(r"\bno\b", t[:20]) or t.startswith("no"):
        return False
    return None


def parse_label(text: str) -> Optional[bool]:
    m = re.search(r'"label"\s*:\s*"(CORRECT|WRONG)"', text, re.I)
    if m:
        return m.group(1).upper() == "CORRECT"
    up = text.upper()
    if "CORRECT" in up and "WRONG" not in up:
        return True
    if "WRONG" in up and "CORRECT" not in up:
        return False
    return None


async def judge(rec: Dict[str, Any], response: str, ledger: Ledger, stub: bool, sem) -> Dict[str, Any]:
    out = {}
    jobs = []
    if rec["bench"] == "lme":
        jobs.append(("j", lme_judge_prompt(rec["category"], rec["question"], rec["answer"], response,
                                           rec["abstention"]), None, parse_yes_no))
    else:
        qtype = "temporal-reasoning" if rec["category"] == "temporal" else "single-session-user"
        jobs.append(("j_strict", lme_judge_prompt(qtype, rec["question"], rec["answer"], response), None,
                     parse_yes_no))
        if rec["arm"] in LENIENT_ARMS or rec["arm"].startswith("decoy:"):
            jobs.append(("j_lenient", mem0_judge_prompt(rec["question"], rec["answer"], response),
                         MEM0_JUDGE_SYSTEM, parse_label))
    for name, prompt, system, parse in jobs:
        stub_text = None
        if stub:
            ok = _stub_verdict(rec["answer"], response) if not rec["abstention"] else "know" in response.lower()
            stub_text = ("yes" if ok else "no") if parse is parse_yes_no else json.dumps(
                {"reasoning": "stub", "label": "CORRECT" if ok else "WRONG"})
        async with sem:
            t = time.perf_counter()
            text, usage = await llm_call(prompt, system=system, effort="minimal", max_tokens=512, stub=stub_text)
        verdict = parse(text)
        cost = _cost(usage)
        ledger.add({"arm": rec["arm"], "qid": rec["qid"], "call": name, "usage": usage, "cost": cost,
                    "ms": round(1000 * (time.perf_counter() - t)), "parsed": verdict is not None})
        out[name] = verdict
        out[f"{name}_raw"] = text[:400]
    return out


async def answer_one(rec: Dict[str, Any], ledger: Ledger, stub: bool, sem) -> Dict[str, Any]:
    prompt = ANSWER_PROMPT.format(memories=rec["context"], date_line=rec.get("date_line", ""),
                                  question=rec["question"])
    async with sem:
        t = time.perf_counter()
        text, usage = await llm_call(prompt, system=None, effort="low", max_tokens=ANSWER_MAX_TOKENS,
                                     stub=_stub_answer(rec["context"]) if stub else None)
    ledger.add({"arm": rec["arm"], "qid": rec["qid"], "call": "answer", "usage": usage, "cost": _cost(usage),
                "ms": round(1000 * (time.perf_counter() - t)), "parsed": bool(text)})
    verdicts = await judge(rec, text, ledger, stub, sem)
    return {"arm": rec["arm"], "bench": rec["bench"], "qid": rec["qid"], "split": rec.get("split"),
            "category": rec["category"], "abstention": rec["abstention"], "response": text,
            "in_tokens": usage["in"], "retrieval_ms": rec.get("retrieval_ms"), **verdicts}


async def cmd_answer(args) -> None:
    global LIMITER
    LIMITER = RateLimiter(args.rpm)
    run = Path(args.run)
    tag = "stub" if args.stub else "live"
    ledger = Ledger(run / f"ledger_{tag}.jsonl", args.cap)
    done_path = run / f"answers_{tag}.jsonl"
    done = set()
    if done_path.exists():
        done = {(r["arm"], r["qid"]) for r in map(json.loads, open(done_path))}
    recs = []
    for f in sorted(run.glob("contexts_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if (r["arm"], r["qid"]) in done:
                continue
            if args.arms and r["arm"] not in args.arms.split(","):
                continue
            if args.split and r.get("split") not in (args.split, "all"):
                continue
            recs.append(r)
    if args.per_arm:  # pilot: n items per (bench, arm)
        seen, keep = defaultdict(int), []
        for r in recs:
            key = (r["bench"], r["arm"])
            if seen[key] < args.per_arm:
                seen[key] += 1
                keep.append(r)
        recs = keep
    random.Random(0).shuffle(recs)  # judge calls for all arms interleaved (blinding)
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    with open(done_path, "a") as out:
        tasks = [asyncio.create_task(answer_one(r, ledger, args.stub, sem)) for r in recs]
        try:
            for i, fut in enumerate(asyncio.as_completed(tasks), 1):
                out.write(json.dumps(await fut) + "\n")
                out.flush()
                if i % 100 == 0:
                    print(f"  {i}/{len(recs)} answered, ${ledger.spent:.2f}, {time.time() - t0:.0f}s", flush=True)
        except StopRun as e:
            for t in tasks:
                t.cancel()
            print(f"STOP: {e}", flush=True)
    print(json.dumps({"answered": len(recs), "spent_usd": round(ledger.spent, 4), "wall_s": round(time.time() - t0)}))


# ------------------------------------------------------------------ decoys


def _perturb(gold: str, rng: random.Random) -> Optional[str]:
    """Change the answer's key fact (prereg: entity, number or year), or None if there is no clear one.

    The first pilot found two invalid decoys from an earlier version that swapped any capitalised
    word: "Likely no, ..." -> "Toronto no, ..." kept its meaning, and one item of a ten-item list
    left nine correct items. So: years and numbers first, then a proper noun that is not the first
    word, and never list answers.
    """
    if gold.count(",") >= 2:
        return None
    if re.search(r"\b(19|20)\d{2}\b", gold):  # every year moves, so no alternative stays correct
        shift = rng.choice((-2, 2))
        return re.sub(r"\b(19|20)\d{2}\b", lambda m: str(int(m.group(0)) + shift), gold)
    if re.search(r"\d+", gold):
        return re.sub(r"\d+", lambda m: str(int(m.group(0)) + 3), gold)
    if "," in gold or " and " in gold or "&" in gold:  # several items: one swap would leave others right
        return None
    words = gold.split()
    nouns = [i for i, w in enumerate(words) if i > 0 and w[:1].isupper() and w.strip(",.").isalpha()]
    if not nouns:
        return None
    i = rng.choice(nouns)
    words[i] = rng.choice(("Lisbon", "Marcus", "Toronto", "Priya", "Oslo", "Kenji"))
    return " ".join(words)


async def cmd_decoys(args) -> None:
    """Judge reliability (bar B6): gold must pass; shuffled, "don't know" and perturbed must fail."""
    global LIMITER
    LIMITER = RateLimiter(args.rpm)
    rng = random.Random(11)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    tag = "stub" if args.stub else "live"
    ledger = Ledger(run / f"ledger_decoys_{tag}.jsonl", args.cap)
    rows = []
    for bench, items in (("locomo", locomo_items()), ("lme", [i for i in lme_items("dev")
                                                             if i["category"] != "single-session-preference"])):
        order = rng.sample(items, len(items))
        pool, rest = order[:30], order[30:]
        perturbable = []
        for it in rest:  # 10 items whose answer has a clear key fact to change
            alt = _perturb(it["answer"], rng)
            if alt is not None:
                perturbable.append((it, alt))
            if len(perturbable) == 10:
                break
        alts = {it["qid"]: alt for it, alt in perturbable}
        for kind, chunk in zip(("gold", "shuffled", "dont_know", "perturbed"),
                               (pool[:10], pool[10:20], pool[20:30], [it for it, _ in perturbable])):
            for it in chunk:
                if kind == "gold":
                    resp, expect = it["answer"], True
                elif kind == "shuffled":
                    other = rng.choice([o for o in items if o["answer"] != it["answer"] and
                                        o["category"] == it["category"]])
                    resp, expect = other["answer"], False
                elif kind == "dont_know":
                    resp, expect = "I don't know; the memories don't mention it.", False
                else:
                    resp, expect = alts[it["qid"]], False
                rows.append(({**_public(it), "arm": f"decoy:{kind}"}, resp, expect, kind))
    sem = asyncio.Semaphore(args.concurrency)
    verdicts = await asyncio.gather(*(judge(r, resp, ledger, args.stub, sem) for r, resp, _, _ in rows))
    acc = defaultdict(lambda: [0, 0])
    for (r, resp, expect, kind), v in zip(rows, verdicts):
        for name in ("j", "j_strict", "j_lenient"):
            if name in v:
                key = (r["bench"], name, kind)
                acc[key][0] += v[name] is expect
                acc[key][1] += 1
    report = {f"{b}/{j}/{k}": round(h / n, 3) for (b, j, k), (h, n) in sorted(acc.items())}
    with open(run / f"decoys_{tag}_items.jsonl", "w") as f:
        for (r, resp, expect, kind), v in zip(rows, verdicts):
            f.write(json.dumps({"bench": r["bench"], "kind": kind, "question": r["question"], "gold": r["answer"],
                                "response": resp, "expect_correct": expect,
                                **{k: v[k] for k in ("j", "j_strict", "j_lenient") if k in v}}) + "\n")
    strict_ok = all(v >= 0.95 for k, v in report.items() if "/j_lenient/" not in k)
    json.dump({"decoys": report, "spent_usd": round(ledger.spent, 4)}, open(run / f"decoys_{tag}.json", "w"), indent=2)
    print(json.dumps(report, indent=1))
    print(f"CHECK: scorer-on-decoys {'ok' if strict_ok else 'FAIL'} ({tag} judge)")


# ------------------------------------------------------------------ report


def _boot(a: np.ndarray, b: Optional[np.ndarray] = None, n: int = 2000) -> tuple:
    rng = np.random.default_rng(0)
    d = a.astype(float) if b is None else b.astype(float) - a.astype(float)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return round(float(d.mean()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)


def cmd_report(args) -> None:
    run = Path(args.run)
    tag = "stub" if args.stub else "live"
    rows = [json.loads(line) for line in open(run / f"answers_{tag}.jsonl")]
    by = defaultdict(dict)
    for r in rows:
        by[(r["bench"], r["arm"])][r["qid"]] = r
    out = {}
    for (bench, arm), qs in sorted(by.items()):
        res = {"n": len(qs), "mean_in_tokens": round(float(np.mean([q["in_tokens"] for q in qs.values()])))}
        for j in ("j", "j_strict", "j_lenient"):
            vals = [q.get(j) for q in qs.values() if j in q]
            if vals:
                arr = np.array([bool(v) for v in vals])
                res[j] = _boot(arr)
                res[f"{j}_unparsed"] = sum(v is None for v in vals)
        out[f"{bench}/{arm}"] = res
    print(json.dumps(out, indent=1))
    json.dump(out, open(run / f"report_{tag}.json", "w"), indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("retrieve")
    p.add_argument("--bench", choices=("locomo", "lme"), required=True)
    p.add_argument("--split", default="all", choices=("all", "dev", "heldout"))
    p.add_argument("--arms", default="A0,A1,A2,A3,A3w")
    p.add_argument("--run", required=True)
    p.add_argument("--limit", type=int, default=0, help="first N conversations / haystacks only")
    p.add_argument("--a0-dev", action="store_true", help="also build full-context (A0) prompts for LME dev rows")
    for name in ("answer", "decoys", "report"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--stub", action="store_true", help="no API calls; deterministic fake answers and verdicts")
        p.add_argument("--cap", type=float, default=CAP_USD)
        p.add_argument("--concurrency", type=int, default=8)
        p.add_argument("--rpm", type=float, default=35.0, help="request-per-minute ceiling (0 = none)")
        if name == "answer":
            p.add_argument("--arms", default="")
            p.add_argument("--split", default="")
            p.add_argument("--per-arm", type=int, default=0, help="pilot: N items per (bench, arm)")
    args = ap.parse_args()
    if args.cmd == "retrieve":
        asyncio.run(cmd_retrieve(args))
    elif args.cmd == "answer":
        asyncio.run(cmd_answer(args))
    elif args.cmd == "decoys":
        asyncio.run(cmd_decoys(args))
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
