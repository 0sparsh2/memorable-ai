"""
Evaluate the fine-tuned models against the preregistered bars on Kaggle (docs/retrieval_round2.md §0).

Mounts the memorable-embed-ft output (models/), clones the public branch, downloads
LoCoMo (official repo) and LongMemEval-S (Hugging Face), then runs on the GPUs:
  1. benchmarks/embeddings/ft_compare.py: turn, static and semantic models, and
     rerankers, base vs fine-tuned, with paired bootstrap CIs
  2. benchmarks/locomo/retrieval_eval.py end to end (real pipeline, score fusion):
     base; fine-tuned MiniLM as the semantic model (episode bar); fine-tuned
     leaf-ir as the turn model
Output: ft_compare.json, e2e_*.json, manifest.json.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

REPO = "https://github.com/0sparsh2/memorable-ai.git"
BRANCH = "feat/memory-architecture"
SRC = "/kaggle/tmp/memorable"
OUT = "/kaggle/working"
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def main():
    t0 = time.time()
    models = sorted(glob.glob("/kaggle/input/**/models/leafir", recursive=True))
    if not models:
        raise SystemExit("fine-tuned models not found under /kaggle/input")
    mdir = os.path.dirname(models[0])
    shutil.rmtree(SRC, ignore_errors=True)
    sh(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, SRC])
    commit = subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", SRC])
    sh([sys.executable, "-m", "pip", "install", "-q", "-U", "sentence-transformers>=5.1", "onnxruntime", "model2vec"])

    from huggingface_hub import hf_hub_download

    lme = hf_hub_download("xiaowu0162/longmemeval-cleaned", "longmemeval_s_cleaned.json", repo_type="dataset")
    os.makedirs(f"{SRC}/benchmarks/longmemeval/data", exist_ok=True)
    shutil.copy(lme, f"{SRC}/benchmarks/longmemeval/data/longmemeval_s_cleaned.json")
    os.makedirs(f"{SRC}/benchmarks/locomo/data", exist_ok=True)
    urllib.request.urlretrieve(LOCOMO_URL, f"{SRC}/benchmarks/locomo/data/locomo10.json")

    env = dict(os.environ, MEMORABLE_LLM__DEFAULT_MODEL="none", MEMORABLE_EMBED_BACKEND="torch",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", MEMORABLE_BENCH_CACHE="/kaggle/tmp/bench")
    sh([sys.executable, "-m", "benchmarks.embeddings.ft_compare", "--models-dir", mdir,
        "--out", f"{OUT}/ft_compare.json"], cwd=SRC, env=env)

    runs = {
        "e2e_base": {},
        "e2e_minilm_ft": {"--embedder": f"torch:{mdir}/minilm"},
        "e2e_leafir_ft_turns": {"env": {"MEMORABLE_LLM__TURN_EMBEDDING_MODEL": f"torch:{mdir}/leafir"}},
    }
    for name, opts in runs.items():
        e = dict(env, **opts.get("env", {}))
        cmd = [sys.executable, "-m", "benchmarks.locomo.retrieval_eval", "--convs", "10", "--k", "5",
               "--out", f"{OUT}/{name}.json"]
        if "--embedder" in opts:
            cmd += ["--embedder", opts["--embedder"]]
        sh(cmd, cwd=SRC, env=e, stdout=subprocess.DEVNULL)
        r = json.load(open(f"{OUT}/{name}.json"))
        print(name, {k: r["recall"][k]["overall"] for k in ("turn/pipeline", "episode/fused")}, flush=True)

    json.dump({"commit": commit, "models_dir": mdir, "wall_s": round(time.time() - t0)},
              open(f"{OUT}/manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
