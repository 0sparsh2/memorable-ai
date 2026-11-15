"""Build the LongMemEval-S contexts for the end-to-end QA eval on Kaggle (benchmarks/qa/eval_plan).

The retrieve stage ingests each of the 500 haystacks (~50 sessions, ~500 turns) through
Memorable's real write pipeline and records every arm's memory context. That is ~2.7 h
of CPU on a laptop, so it runs here instead: the public repo at a pinned branch,
LongMemEval-S from Hugging Face, embeddings on the T4s (PyTorch backend), one worker per
GPU, each taking every n-th haystack.

Output in /kaggle/working:
  contexts_lme.jsonl   every arm's context per question (A0 only for its 99-question subset)
  manifest.json        commit, counts, timings
No answers are produced here, and no model is called; the answer stage runs separately.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

REPO = "https://github.com/0sparsh2/memorable-ai.git"
BRANCH = "feat/memory-architecture"
SRC = "/kaggle/tmp/memorable"
OUT = "/kaggle/working"
ARMS = "A0,A1,A2,A3,A3w"


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main():
    t0 = time.time()
    shutil.rmtree(SRC, ignore_errors=True)
    sh(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, SRC])
    commit = subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print("commit", commit, flush=True)
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", SRC])
    sh([sys.executable, "-m", "pip", "install", "-q", "-U", "sentence-transformers>=5.1", "onnxruntime", "model2vec"])

    from huggingface_hub import hf_hub_download

    data = hf_hub_download("xiaowu0162/longmemeval-cleaned", "longmemeval_s_cleaned.json", repo_type="dataset")
    dest = os.path.join(SRC, "benchmarks", "longmemeval", "data")
    os.makedirs(dest, exist_ok=True)
    shutil.copy(data, os.path.join(dest, "longmemeval_s_cleaned.json"))

    import torch

    n = max(1, torch.cuda.device_count())
    run = os.path.join(OUT, "run")
    os.makedirs(run, exist_ok=True)
    base_env = dict(os.environ, MEMORABLE_LLM__DEFAULT_MODEL="none", MEMORABLE_EMBED_BACKEND="torch",
                    TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    procs = []
    for i in range(n):
        env = dict(base_env, CUDA_VISIBLE_DEVICES=str(i))
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "benchmarks.qa.e2e_qa", "retrieve", "--bench", "lme", "--split", "all",
             "--arms", ARMS, "--run", run, "--shard", f"{i}/{n}"], cwd=SRC, env=env))
    codes = [p.wait() for p in procs]
    print("worker exit codes", codes, flush=True)

    rows, by_arm = 0, {}
    with open(os.path.join(OUT, "contexts_lme.jsonl"), "w") as out:
        for path in sorted(glob.glob(os.path.join(run, "contexts_lme.shard*.jsonl"))):
            for line in open(path):
                out.write(line)
                rows += 1
                arm = json.loads(line)["arm"]
                by_arm[arm] = by_arm.get(arm, 0) + 1
    manifest = {"commit": commit, "branch": BRANCH, "workers": n, "exit_codes": codes, "rows": rows,
                "rows_by_arm": by_arm, "wall_s": round(time.time() - t0)}
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    shutil.rmtree(run, ignore_errors=True)
    print(json.dumps(manifest, indent=2))
    if any(codes):
        raise SystemExit("a worker failed")


if __name__ == "__main__":
    main()
