"""
Generate the remaining synthetic memory-QA questions on NVIDIA NIM (free tier), resumably.

Kaggle's T4s ran vLLM at ~13 output tokens/s (one 400-prompt chunk in 2.8 h), so the
rest of the prompts from training/kaggle/qgen go to NIM instead: Nemotron-3.5-Bolt
(thinking off), the same sampling (temperature 0.7, top_p 0.9) and the same parser.
NIM throttles per model, so this does not compete with the end-to-end QA eval
(Nemotron-Lightning).

  python training/qgen_nim.py --prompts training/data/shard_in_0.json:400 training/data/shard_in_1.json \\
      --out training/data/qgen_nim.jsonl --rpm 6

Each line of the output is one conversation (rebuilt from its prompt) with its questions
and the generator id. Already-done prompts are skipped on restart.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "nvidia/nemotron-3.5-bolt-e4b"

_spec = importlib.util.spec_from_file_location("qgen", ROOT / "training" / "kaggle" / "qgen" / "qgen.py")
qgen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qgen)


def conv_from_prompt(prompt: str, cid: str) -> dict:
    """Rebuild the conversation (turn ids, speakers, dates, text) from the rendered prompt."""
    wild = prompt.startswith("Below is a chat between a user and an AI assistant")
    m = re.search(r"between (.+?) and (.+?)(?: \((.+?)\))?, split into", prompt)
    people = ["User", "Assistant"] if wild else ([m.group(1), m.group(2)] if m else [])
    rel = (m.group(3) or "") if (m and not wild) else ""
    turns, session, date = [], 0, ""
    for line in prompt.splitlines():
        h = re.match(r"--- Session (\d+) \((.+)\) ---", line)
        if h:
            session, date = int(h.group(1)), h.group(2)
            continue
        t = re.match(r"\[(S\d+-T\d+)\] ([^:]+): (.*)", line)
        if t:
            text = t.group(3)
            turns.append({"tid": t.group(1), "session": session, "date": date, "speaker": t.group(2),
                          "text": text[:-6] if text.endswith(" [...]") else text})
    n_sessions = len({x["session"] for x in turns})
    source = "wildchat" if wild else ("soda" if n_sessions == 1 else ("cc" if rel else "msc"))
    return {"cid": cid, "source": source, "people": people, "relationship": rel, "turns": turns}


async def main() -> None:
    import httpx
    from dotenv import dotenv_values

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", required=True, help="shard_in files, optionally path:start")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rpm", type=float, default=6.0)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--max-hours", type=float, default=48)
    args = ap.parse_args()

    env = {**dotenv_values(ROOT / ".env"), **os.environ}
    key = (env.get("NVIDIA_NIM_API_KEY") or env.get("NVIDIA_API_KEY") or "").strip()
    base = (env.get("NVIDIA_NIM_API_BASE") or "https://integrate.api.nvidia.com/v1").rstrip("/")
    jobs = []
    for spec in args.prompts:
        path, _, start = spec.partition(":")
        items = json.load(open(path))
        for i in range(int(start or 0), len(items)):
            jobs.append((f"{Path(path).stem}#{i}", items[i]["prompt"]))
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["cid"] for line in open(out_path)}
    jobs = [j for j in jobs if j[0] not in done]
    print(f"{len(jobs)} prompts to go ({len(done)} done)", flush=True)

    interval, nxt = 60.0 / args.rpm, [0.0]
    lock = asyncio.Lock()
    deadline = time.time() + args.max_hours * 3600
    queue: asyncio.Queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)
    stats = {"ok": 0, "q": 0, "e429": 0, "err": 0}

    async def pace():
        async with lock:
            now = time.monotonic()
            wait = nxt[0] - now
            nxt[0] = max(now, nxt[0]) + interval
        if wait > 0:
            await asyncio.sleep(wait)

    async def worker(client, out):
        while time.time() < deadline:
            try:
                cid, prompt = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            backoff = 15
            while time.time() < deadline:
                await pace()
                try:
                    r = await client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                                          json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                                                "max_tokens": 1500, "temperature": 0.7, "top_p": 0.9,
                                                "chat_template_kwargs": {"enable_thinking": False}})
                except Exception:  # noqa: BLE001 - network hiccup: retry
                    stats["err"] += 1
                    await asyncio.sleep(10)
                    continue
                if r.status_code == 429:
                    stats["e429"] += 1
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 300)
                    continue
                if r.status_code != 200:
                    stats["err"] += 1
                    await asyncio.sleep(10)
                    continue
                text = r.json()["choices"][0]["message"].get("content") or ""
                conv = conv_from_prompt(prompt, cid)
                conv["questions"] = qgen.parse(text, conv)
                conv["generator"] = MODEL
                out.write(json.dumps(conv) + "\n")
                out.flush()
                stats["ok"] += 1
                stats["q"] += len(conv["questions"])
                if stats["ok"] % 50 == 0:
                    print(f"  {stats} {time.strftime('%H:%M:%S')}", flush=True)
                break

    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=15)) as client:
        with open(out_path, "a") as out:
            await asyncio.gather(*(worker(client, out) for _ in range(args.concurrency)))
    print("finished", stats, flush=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
