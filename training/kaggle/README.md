# Kaggle training jobs

GPU jobs for [docs/retrieval_round2.md](../../docs/retrieval_round2.md). Each runs as a **private** Kaggle script kernel on 2× T4 with internet enabled.

| Job | What it does | GPU time |
|---|---|---|
| `qgen/` | Writes memory-recall questions, with evidence turn ids, over public dialogue corpora (ConversationChronicles, MSC, SODA, WildChat-1M). Generator: Qwen3-4B-Instruct-2507 under vLLM. Output: `qgen.jsonl`. | ≤ 3.5 h (generation is capped at 3 h) |
| `lme_contexts/` | Builds the end-to-end QA eval's LongMemEval-S contexts. It clones the public branch, ingests all 500 haystacks through the real pipeline, and records every arm's memory context. Embeddings run on the GPUs, one worker per T4. No model is called. | ~1 h |
| `embed_ft/` | Teacher filter and hard-negative mining (Qwen3-Embedding-0.6B), then contrastive fine-tunes of leaf-ir, MiniLM and potion-32M, plus BCE fine-tunes of three rerankers. Output: `models/`, `ft_report.json`. | about 1 h |

No LoCoMo or LongMemEval data is used. ShareGPT and UltraChat are excluded because LongMemEval builds its filler sessions from them.

## Run

```bash
kaggle config view                     # logged in?
python training/kaggle/push.py qgen    # set SMOKE = True in qgen.py first for a ~10-minute check
kaggle kernels status <user>/memorable-qgen
python training/kaggle/push.py embed_ft --sources memorable-qgen
kaggle kernels output <user>/memorable-embed-ft -p <dir>
```

`push.py` writes `kernel-metadata.json` into a temporary build directory, because it names your Kaggle account. Never commit that file.

## Notes

- The Kaggle image runs Python 3.13. vLLM is installed into its own Python 3.12 venv with `uv`, with `transformers<5` pinned, because vLLM 0.11 needs it.
- Check the training job locally on CPU before pushing it:
  ```bash
  FT_DRYRUN=1 FT_OUT=/tmp/ft FT_QGEN=<qgen.jsonl> python training/kaggle/embed_ft/embed_ft.py
  ```
  This needs `datasets` and `accelerate`.
