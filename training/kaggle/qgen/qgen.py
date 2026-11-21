"""Synthesize memory-recall training questions over public dialogue corpora (Kaggle GPU job).

Why: no public dataset pairs "question about a past conversation" with the turn that
answers it (HiNS, arXiv 2601.14857, did not release its 201k samples), so we
generate one. A local LLM (Qwen3-4B-Instruct-2507) reads a dated, multi-session
conversation and writes questions plus the ids of the evidence turns.

Sources (all public; none overlaps the evals, LoCoMo and LongMemEval):
  ConversationChronicles  multi-session, dated, two people     (cc)
  MSC (multi-session chat) multi-session persona chat            (msc)
  SODA                     single social dialogues               (soda)
  WildChat-1M              real user <-> assistant chats, English, non-toxic (wildchat)
LongMemEval's filler sessions come from ShareGPT and UltraChat, so neither is used.

Output: /kaggle/working/qgen.jsonl, one conversation per line:
  {"cid", "source", "people", "turns": [{"tid", "session", "date", "speaker", "text"}],
   "questions": [{"q", "evidence": [tid, ...], "type", "answer"}]}
plus /kaggle/working/qgen_stats.json.

Set SMOKE = False for a 10-minute check of the backend, throughput and output quality.
"""

import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

SMOKE = False
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
COUNTS = {"cc": 2600, "msc": 1100, "soda": 1600, "wildchat": 2200}
N_QUESTIONS = {"cc": 8, "msc": 8, "soda": 3, "wildchat": 4}
QUOTAS = {
    "cc": "2 fact, 2 temporal, 1 preference, 2 paraphrase, 1 multi",
    "msc": "2 fact, 2 temporal, 1 preference, 2 paraphrase, 1 multi",
    "soda": "1 fact, 1 preference, 1 paraphrase",
    "wildchat": "1 user-fact, 1 assistant-recall, 1 preference, 1 paraphrase",
}
GEN_BUDGET_S = int(2.5 * 3600)  # stop generating after this; finished chunks are kept
if SMOKE:
    COUNTS = {k: 12 for k in COUNTS}
MAX_PROMPT_TOKENS = 5000
OUT = "/kaggle/working"
SEED = 20261115

NAMES = (
    "Aaliyah Aarav Abebe Ada Adrian Aiko Alejandro Amara Amir Ana Andre Anika Arjun Astrid Aylin Beatriz "
    "Ben Bianca Bilal Camila Carlos Chen Chiara Chloe Dana Daniel Darius Dev Diego Elena Elif Eli Emeka Emma "
    "Erik Esther Fatima Felix Freya Gabriel Grace Hana Hannah Hugo Ibrahim Ines Isaac Ivan Jada Jamal Jasmine "
    "Javier Jin Jonas Joon Julia Kai Kamala Kenji Kofi Lara Leila Leo Liam Lina Luca Lucia Malik Maya Mei "
    "Mateo Mia Mila Min Nadia Naomi Nia Nikhil Noah Nora Omar Oscar Owen Priya Rafael Ravi Rosa Ruth Sami "
    "Sara Sofia Sven Tariq Tess Theo Tomas Uma Valentina Victor Wei Yara Yusuf Zara Zoe"
).split()

# ---------------------------------------------------------------- conversations


def _when(base, rng):
    t = base.replace(hour=rng.randrange(8, 22), minute=rng.randrange(60))
    return t, f"{t.strftime('%-I:%M %p').lower()} on {t.day} {t.strftime('%B')}, {t.year}"


def _gap_days(label, rng):
    s = (label or "").lower()
    if "hour" in s:
        return 0
    if "day" in s:
        return rng.randint(2, 6)
    if "week" in s:
        return rng.randint(7, 25)
    if "month" in s:
        return rng.randint(30, 150)
    if "year" in s:
        return rng.randint(365, 900)
    return rng.randint(1, 10)


def _session_dates(gaps, rng):
    day = datetime(2021, 1, 1) + timedelta(days=rng.randrange(0, 1600))
    out = []
    for g in gaps:
        day = day + timedelta(days=g)
        out.append(_when(day, rng)[1])
    return out


def _conv(cid, source, sessions, dates, people, relationship=""):
    """sessions: list of [(speaker, text)]."""
    turns = []
    for s, (sess, date) in enumerate(zip(sessions, dates), 1):
        for t, (spk, text) in enumerate(sess, 1):
            text = re.sub(r"\s+", " ", str(text)).strip()
            if text:
                turns.append({"tid": f"S{s}-T{t}", "session": s, "date": date, "speaker": spk, "text": text[:2000]})
    return {"cid": cid, "source": source, "people": people, "relationship": relationship, "turns": turns}


def _names(rng, k):
    return rng.sample(NAMES, k)


def load_cc(n, rng):
    from datasets import load_dataset

    keys = ("first", "second", "third", "fourth", "fifth")
    ds = load_dataset("jihyoung/ConversationChronicles", split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=20_000)
    out = []
    for row in ds:
        labels = sorted({s for k in keys for s in (row.get(f"{k}_session_speakers") or [])})
        if len(labels) != 2:
            continue
        people = _names(rng, 2)
        name = dict(zip(labels, people))
        sessions = [list(zip([name[s] for s in row[f"{k}_session_speakers"]], row[f"{k}_session_dialogue"]))
                    for k in keys]
        gaps = [0] + [_gap_days(x, rng) for x in (row.get("time_interval") or [])[1:5]]
        out.append(_conv(f"cc-{row['dataID']}", "cc", sessions, _session_dates(gaps, rng), people,
                         (row.get("relationship") or "").lower()))
        if len(out) >= n:
            break
    return out


def load_msc(n, rng):
    from datasets import load_dataset

    ds = load_dataset("nayohan/multi_session_chat", split="train", streaming=True)
    groups, cur, cur_id = [], [], None
    for row in ds:
        if row["dialoug_id"] != cur_id and cur:
            groups.append(cur)
            cur = []
        cur_id = row["dialoug_id"]
        cur.append(row)
    if cur:
        groups.append(cur)
    groups = [g for g in groups if len(g) >= 3]
    rng.shuffle(groups)
    out = []
    for g in groups[:n]:
        people = _names(rng, 2)
        name = {"Speaker 1": people[0], "Speaker 2": people[1]}
        sessions = [list(zip([name.get(s, s) for s in r["speaker"]], r["dialogue"])) for r in g]
        gaps = [0] + [rng.choice((0, rng.randint(1, 7), rng.randint(7, 30))) for _ in g[1:]]
        out.append(_conv(f"msc-{g[0]['dialoug_id']}", "msc", sessions, _session_dates(gaps, rng), people))
    return out


def load_soda(n, rng):
    from datasets import load_dataset

    ds = load_dataset("allenai/soda", split="train", streaming=True).shuffle(seed=SEED, buffer_size=20_000)
    out = []
    for row in ds:
        if len(row["dialogue"]) < 6 or len(set(row["speakers"])) != 2:
            continue
        people = sorted(set(row["speakers"]))
        out.append(_conv(f"soda-{row['original_index']}", "soda", [list(zip(row["speakers"], row["dialogue"]))],
                         _session_dates([0], rng), people))
        if len(out) >= n:
            break
    return out


def load_wildchat(n, rng):
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True).shuffle(seed=SEED, buffer_size=20_000)
    out = []
    for row in ds:
        conv = row.get("conversation") or []
        if (row.get("language") != "English" or str(row.get("toxic")) == "True" or str(row.get("redacted")) == "True"
                or len(conv) < 4 or len(conv) > 16):
            continue
        if any(len(m.get("content") or "") > 4000 for m in conv):
            continue
        sess = [("User" if m["role"] == "user" else "Assistant", m["content"]) for m in conv]
        out.append(_conv(f"wc-{row['conversation_hash']}", "wildchat", [sess], _session_dates([0], rng),
                         ["User", "Assistant"]))
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- prompts

TYPES_PEOPLE = """- "fact": a concrete detail someone mentioned (who, what, where, which, how many).
- "temporal": when something happened or is planned; answer with a date relative to the session date (for example "the weekend before 8 May, 2023").
- "preference": a person's likes, dislikes, opinions, goals, plans or feelings.
- "paraphrase": ask about a specific detail using different words than the turn (synonyms, a description instead of the name); do not copy its distinctive words.
- "multi": needs two turns, ideally from different sessions; give both evidence ids."""

TYPES_ASSISTANT = """- "user-fact": something the user said about themselves, their work, or their situation.
- "assistant-recall": something the assistant suggested, explained, listed or wrote.
- "preference": the user's goals, constraints, likes or dislikes.
- "paraphrase": ask about a specific detail using different words than the turn; do not copy its distinctive words."""


def render(conv):
    lines, cur = [], None
    for t in conv["turns"]:
        if t["session"] != cur:
            cur = t["session"]
            lines.append(f"\n--- Session {cur} ({t['date']}) ---")
        text = t["text"] if len(t["text"]) <= 1200 else t["text"][:1200] + " [...]"
        lines.append(f"[{t['tid']}] {t['speaker']}: {text}")
    return "\n".join(lines).strip()


def prompt_for(conv):
    n = N_QUESTIONS[conv["source"]]
    if conv["source"] == "wildchat":
        intro = "Below is a chat between a user and an AI assistant. Every turn has an id in brackets."
        who = ("Months later the user starts a new chat and asks the assistant, which kept this chat in memory, "
               "about it. Write the questions in the user's voice (\"What did I...\", \"What did you suggest...\").")
        types = TYPES_ASSISTANT
    else:
        rel = f" ({conv['relationship']})" if conv.get("relationship") else ""
        intro = (f"Below is a conversation history between {conv['people'][0]} and {conv['people'][1]}{rel}, "
                 "split into dated sessions. Every turn has an id in brackets.")
        who = ("An assistant has stored this whole history in memory. Later someone asks it questions whose "
               "answers are in specific turns. Refer to people by name.")
        types = TYPES_PEOPLE
    return f"""{intro}

{render(conv)}

{who} Write {n} such questions.

Rules:
- Each question must be answerable from the turn(s) in "evidence", and must make sense without seeing the conversation (never say "in session 2" or "in this conversation").
- Types (write exactly {QUOTAS[conv["source"]]}):
{types}
- Spread the questions over different turns (at most 2 per turn). Prefer informative turns; skip greetings and small talk.
- Keep each question under 25 words and each answer under 15 words.

Return only a JSON array: [{{"q": "...", "evidence": ["S1-T4"], "type": "fact", "answer": "..."}}]"""


META = re.compile(r",?\s*\b(in|during|from)\s+(session\s*\d+|the (first|second|third|fourth|fifth|last|earlier|previous) session|"
                  r"(this|the|our|that) (conversation|chat|discussion))\b", re.I)


def parse(text, conv):
    ids = {t["tid"] for t in conv["turns"]}
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except Exception:
        items = []
        for o in re.findall(r"\{[^{}]*\}", m.group(0)):
            try:
                items.append(json.loads(o))
            except Exception:
                pass
    out, seen = [], set()
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        q = str(it.get("q") or "").strip()
        ev = it.get("evidence") or []
        ev = [ev] if isinstance(ev, str) else ev
        ev = [e.strip("[] ") for e in ev if isinstance(e, str) and e.strip("[] ") in ids]
        q = META.sub("", q).strip()
        if len(q) < 8 or not ev or q.lower() in seen or "session" in q.lower():
            continue
        seen.add(q.lower())
        out.append({"q": q, "evidence": ev, "type": str(it.get("type") or ""), "answer": str(it.get("answer") or "")})
    return out


# ---------------------------------------------------------------- generation backends


WORKER_SRC = r"""
import json, os, sys, time


def main():
    rank, path_in, path_out, backend, MODEL, MAXLEN, SEED = sys.argv[1:8]
    rank, MAXLEN, SEED = int(rank), int(MAXLEN), int(SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    jobs = json.load(open(path_in))
    msgs = [[{"role": "user", "content": j["prompt"]}] for j in jobs]
    t0 = time.time()
    if backend == "vllm":
        from vllm import LLM, SamplingParams
        deadline = float(sys.argv[8])
        llm = LLM(model=MODEL, dtype="half", max_model_len=MAXLEN, gpu_memory_utilization=0.92, seed=SEED + rank)
        sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1000)
        texts, n_in, n_out = [None] * len(msgs), 0, 0
        for b in range(0, len(msgs), 400):
            if time.time() > deadline:
                print(f"worker {rank}: time budget reached after {b} prompts", flush=True)
                break
            res = llm.chat(msgs[b:b + 400], sp, use_tqdm=False)
            for j, r in enumerate(res):
                texts[b + j] = r.outputs[0].text
            n_out += sum(len(r.outputs[0].token_ids) for r in res)
            n_in += sum(len(r.prompt_token_ids) for r in res)
            print(f"worker {rank}: {b + len(res)}/{len(msgs)} prompts, {n_out} tokens out, "
                  f"{time.time() - t0:.0f}s", flush=True)
            json.dump({"texts": texts, "n_in": n_in, "n_out": n_out, "secs": time.time() - t0}, open(path_out, "w"))
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda:0")
        enc = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
        lens = [len(tok(e)["input_ids"]) for e in enc]
        order = sorted(range(len(enc)), key=lambda i: lens[i])
        batches, cur = [], []
        for i in order:  # token budget: prompt tokens + generated tokens per batch
            if cur and (len(cur) + 1) * (lens[i] + 1000) > 14000:
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        texts, n_in, n_out = [None] * len(enc), 0, 0
        for idx in batches:
            x = tok([enc[i] for i in idx], return_tensors="pt", padding=True).to("cuda:0")
            with torch.no_grad():
                y = model.generate(**x, max_new_tokens=1000, do_sample=True, temperature=0.7, top_p=0.9)
            for i, row in zip(idx, y[:, x["input_ids"].shape[1]:]):
                texts[i] = tok.decode(row, skip_special_tokens=True)
                n_out += int((row != tok.pad_token_id).sum())
            n_in += int(x["attention_mask"].sum())
    json.dump({"texts": texts, "n_in": n_in, "n_out": n_out, "secs": time.time() - t0}, open(path_out, "w"))
    sys.stdout.flush()
    os._exit(0)  # vLLM's engine processes can keep a normal interpreter exit hanging forever


if __name__ == "__main__":
    main()
"""


VLLM_PY = "/kaggle/tmp/vllm/bin/python"


def generate(prompts, backend):
    import torch

    n_gpu = max(1, torch.cuda.device_count())
    if backend.startswith("vllm"):
        python = VLLM_PY
        env_extra = {"VLLM_ATTENTION_BACKEND": backend.split(":", 1)[1]} if ":" in backend else {}
    else:
        python, env_extra = sys.executable, {}
    shards = [prompts[i::n_gpu] for i in range(n_gpu)]
    worker_py = f"{OUT}/shard_worker.py"
    open(worker_py, "w").write(WORKER_SRC)
    procs = []
    for r, sh in enumerate(shards):
        json.dump([{"prompt": p} for p in sh], open(f"{OUT}/shard_in_{r}.json", "w"))
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", **env_extra)
        procs.append(subprocess.Popen([python, worker_py, str(r), f"{OUT}/shard_in_{r}.json",
                                       f"{OUT}/shard_out_{r}.json", backend.split(":")[0], MODEL,
                                       str(MAX_PROMPT_TOKENS + 1100), str(SEED),
                                       str(time.time() + GEN_BUDGET_S)], env=env))
    hard_stop = time.time() + GEN_BUDGET_S + 20 * 60  # never wait forever on a hung worker
    while any(p.poll() is None for p in procs) and time.time() < hard_stop:
        time.sleep(10)
    for p in procs:
        if p.poll() is None:
            print("worker still running at the hard stop; killing it and keeping its last chunk", flush=True)
            p.kill()
    codes = [p.wait() for p in procs]
    if any(codes) and not all(os.path.exists(f"{OUT}/shard_out_{r}.json") for r in range(n_gpu)):
        raise RuntimeError(f"{backend} workers failed: {codes}")
    texts, stats = [None] * len(prompts), []
    for r in range(n_gpu):
        d = json.load(open(f"{OUT}/shard_out_{r}.json"))
        stats.append({k: d[k] for k in ("n_in", "n_out", "secs")})
        for j, t in enumerate(d["texts"]):
            texts[r + j * n_gpu] = t
    return texts, stats


VLLM_VERSION = "0.11.0"


def install_vllm():
    """vLLM in its own Python 3.12 venv (the Kaggle image is 3.13 and has its own torch)."""
    t = time.time()
    cmds = [
        [sys.executable, "-m", "pip", "install", "-q", "uv"],
        [sys.executable, "-m", "uv", "venv", "/kaggle/tmp/vllm", "--python", "3.12", "--seed"],
        [sys.executable, "-m", "uv", "pip", "install", "--python", VLLM_PY, f"vllm=={VLLM_VERSION}", "transformers>=4.56,<5"],
    ]
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True)
        if r.returncode:
            print("vllm install failed:", " ".join(c), r.stderr[-1500:], flush=True)
            return False
    print(f"vllm {VLLM_VERSION} installed in {time.time() - t:.0f}s", flush=True)
    return True


def main():
    rng = random.Random(SEED)
    t0 = time.time()
    convs = []
    for name, fn in (("cc", load_cc), ("msc", load_msc), ("soda", load_soda), ("wildchat", load_wildchat)):
        t = time.time()
        got = fn(COUNTS[name], rng)
        print(f"loaded {name}: {len(got)} conversations in {time.time() - t:.0f}s", flush=True)
        convs += got

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts, keep = [], []
    for c in convs:
        p = prompt_for(c)
        if len(tok(p)["input_ids"]) <= MAX_PROMPT_TOKENS:
            prompts.append(p)
            keep.append(c)
    print(f"{len(keep)}/{len(convs)} conversations fit in {MAX_PROMPT_TOKENS} tokens", flush=True)
    mix = list(range(len(keep)))
    rng.shuffle(mix)
    keep, prompts = [keep[i] for i in mix], [prompts[i] for i in mix]

    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[:1200], flush=True)
    backends = (["vllm", "vllm:TRITON_ATTN"] if install_vllm() else []) + (["hf"] if SMOKE else [])
    for backend in backends:
        try:
            t = time.time()
            texts, gstats = generate(prompts, backend)
            print(f"backend {backend} ok in {time.time() - t:.0f}s", flush=True)
            break
        except Exception as e:
            print(f"backend {backend} failed ({e})", flush=True)
    else:
        raise SystemExit("no generation backend worked")

    n_q, by_type, by_source, bad = 0, {}, {}, 0
    with open(f"{OUT}/qgen.jsonl", "w") as f:
        for c, txt in zip(keep, texts):
            if txt is None:
                continue
            c["questions"] = parse(txt, c)
            bad += not c["questions"]
            n_q += len(c["questions"])
            by_source[c["source"]] = by_source.get(c["source"], 0) + len(c["questions"])
            for q in c["questions"]:
                by_type[q["type"]] = by_type.get(q["type"], 0) + 1
            f.write(json.dumps(c) + "\n")
    stats = {"backend": backend, "model": MODEL, "conversations": len(keep), "questions": n_q,
             "unparsed": bad, "by_source": by_source, "by_type": by_type, "gen": gstats,
             "wall_s": round(time.time() - t0)}
    json.dump(stats, open(f"{OUT}/qgen_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    for c, txt in list(zip(keep, texts))[:: max(1, len(keep) // 6)][:6]:
        print("\n====", c["cid"], "\n", (txt or "")[:1500])
    for name in os.listdir(OUT):
        if name.startswith("shard_"):
            os.remove(f"{OUT}/{name}")


if __name__ == "__main__":
    main()
