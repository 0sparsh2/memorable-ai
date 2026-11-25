"""prereg v8's full answer run on Kaggle (benchmarks/qa/eval_plan/prereg-v8.md), CPU only.

The answer stage makes only API calls (NIM's free tier: four readers and the Lightning judge), so it
runs here instead of on a laptop that sleeps when its lid closes. The harness is the public repo at
a pinned commit. The queue (v8_build's contexts_*.jsonl, dev rows only; uploaded gzipped, unpacked by Kaggle)
comes from the private dataset memorable-v8-queue. The NIM key comes from the Kaggle secret NVIDIA_API_KEY, which
must be attached to this kernel (Add-ons > Secrets). It is never printed or written out.

A session is capped at 12 h, so the loop is stopped after BUDGET_H. Every finished answer is
already on disk, and the next session resumes from it: push again with this kernel's output mounted
(push.py v8_answers --cpu --slug memorable-v8-answers-2 --datasets memorable-v8-queue --sources memorable-v8-answers).

Output in /kaggle/working/run: answers_live.jsonl, ledger_live.jsonl, errors.jsonl, chunks.jsonl,
status.json (commit, counts, whether the queue is finished).
"""

import glob
import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import time

REPO = "https://github.com/0sparsh2/memorable-ai.git"
BRANCH = "feat/memory-architecture"
COMMIT = "4383479"  # harness as frozen for prereg v8 (prompts P1/P1s, readers, answer loop)
SRC = "/kaggle/tmp/memorable"
RUN = "/kaggle/tmp/run"
OUT = "/kaggle/working/run"
BUDGET_H = 11.0
KEEP = ("answers_live.jsonl", "ledger_live.jsonl", "errors.jsonl", "chunks.jsonl")


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def status(**kw):
    os.makedirs(OUT, exist_ok=True)
    json.dump(kw, open(os.path.join(OUT, "status.json"), "w"), indent=1)
    print(json.dumps(kw), flush=True)


def whole_lines(path):
    """Drop a trailing partial line, if a stop cut one."""
    if not os.path.exists(path):
        return 0
    good = []
    for line in open(path):
        try:
            json.loads(line)
            good.append(line if line.endswith("\n") else line + "\n")
        except json.JSONDecodeError:
            pass
    open(path, "w").writelines(good)
    return len(good)


def main():
    t0 = time.time()
    shutil.rmtree(SRC, ignore_errors=True)
    sh(["git", "clone", "-q", "-b", BRANCH, REPO, SRC])
    sh(["git", "-C", SRC, "checkout", "-q", COMMIT])
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", SRC])

    os.makedirs(RUN, exist_ok=True)
    for bench in ("locomo", "lme"):  # Kaggle unpacks uploaded .gz files, so accept either form
        src = [p for p in glob.glob(f"/kaggle/input/**/contexts_{bench}.jsonl*", recursive=True)
               if p.endswith((".jsonl", ".jsonl.gz")) and "/run/" not in p]
        if len(src) != 1:
            status(done=False, error=f"expected one contexts_{bench}.jsonl[.gz], found {len(src)}")
            return
        opener = gzip.open if src[0].endswith(".gz") else open
        with opener(src[0], "rb") as f, open(os.path.join(RUN, f"contexts_{bench}.jsonl"), "wb") as g:
            shutil.copyfileobj(f, g)
    for name in KEEP:  # a previous session's output, if mounted
        prev = [p for p in glob.glob(f"/kaggle/input/**/run/{name}", recursive=True)]
        if prev:
            shutil.copy(prev[0], os.path.join(RUN, name))
    before = whole_lines(os.path.join(RUN, "answers_live.jsonl"))
    print(f"resuming from {before} answers", flush=True)
    try:  # last, so a run without the secret still checks the clone, the install and the inputs
        from kaggle_secrets import UserSecretsClient

        key = UserSecretsClient().get_secret("NVIDIA_API_KEY")
    except Exception as e:  # noqa: BLE001
        status(done=False, ready=True, answers=before, error=f"secret NVIDIA_API_KEY not attached ({type(e).__name__})")
        return

    env = {**os.environ, "NVIDIA_API_KEY": key, "NVIDIA_NIM_API_KEY": key, "E2E_QA_HTTP_TIMEOUT": "600",
           "MEMORABLE_LLM__DEFAULT_MODEL": "none"}
    cmd = [sys.executable, "-m", "benchmarks.qa.e2e_qa", "answer-loop", "--run", RUN, "--concurrency", "16",
           "--rpm", "6", "--reader-rpm", "6", "--probe-every", "60"]
    print("+", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=SRC, env=env)
    budget = BUDGET_H * 3600 - (time.time() - t0)
    try:
        p.wait(timeout=max(60, budget))
    except subprocess.TimeoutExpired:
        print("session budget reached; stopping the loop", flush=True)
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
    os.makedirs(OUT, exist_ok=True)
    for name in KEEP:
        if os.path.exists(os.path.join(RUN, name)):
            shutil.copy(os.path.join(RUN, name), os.path.join(OUT, name))
    n = whole_lines(os.path.join(OUT, "answers_live.jsonl"))
    queued = sum(1 for b in ("locomo", "lme") for _ in open(os.path.join(RUN, f"contexts_{b}.jsonl")))
    commit = subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    status(done=n >= queued, answers=n, queued=queued, new_this_session=n - before, commit=commit,
           exit_code=p.returncode, hours=round((time.time() - t0) / 3600, 2))


if __name__ == "__main__":
    main()
