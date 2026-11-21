# Embedding study: solving the main bottleneck (text → vectors)

**Date:** 2026-11-15
**Machine:** Apple M-series, 10 cores, arm64, Python 3.14. CPU numbers depend on hardware; re-measure on deployment hardware with the scripts below.
**Why:** profiling showed embedding inference at about 70% of non-import runtime, with SQL and graph work negligible ([instruments_analysis.md](instruments_analysis.md)).

## TL;DR

1. **Same model, faster runtime (no quality change):**
   - ONNX Runtime gives **identical vectors** (cosine 1.00000) at 1.4× bulk and **6× single-query** speed versus PyTorch.
   - It is now used automatically when `onnxruntime` is installed and the model publishes an ONNX export. It handles query prompts, CLS/mean pooling, Dense projection layers and sidecar weight files, and refuses anything else.
2. **Better *and* faster models exist for memory retrieval** (two benchmarks: LoCoMo and LongMemEval-S):
   - **potion-base-32M**, a static model with no transformer, beats MiniLM on turn retrieval on both benchmarks at **30–175× lower cost**. It is weaker on semantic, session-level questions such as preferences.
   - **mdbr-leaf-ir** (23M transformer, MongoDB, July 2026) lifts LoCoMo dense turn recall from **0.40 to 0.62**.
3. **Tiered embeddings are the shipped design.** It has three roles and two profiles (`MEMORABLE_EMBEDDING_PROFILE`):
   - **semantic** (facts, episodes, extractive summaries): MiniLM via ONNX
   - **turn index:** leaf-ir in `quality` (the default with the `fast` extra), potion-32M in `fast`
   - **segmentation:** potion-32M (never the turn model)

   | LoCoMo, full pipeline | v0.2 | `fast` | **`quality`** |
   |---|---|---|---|
   | Fused turn recall | 0.565 | 0.579 | **0.642** |
   | Episode recall | 0.535 | 0.541 | **0.541** |
   | Multi-hop (fused) | 0.504 | 0.475 | **0.571** |
   | Ingest, 10 conversations | 46.3 s | **25.0 s** | 39.1 s |
   | Query p50 | 7.9 ms | **3.2 ms** | 10.1 ms |
4. **A safe rollout is built in.** Every layer records the vector space it was embedded in. A model change re-embeds that layer from stored text, so old and new vectors never mix.
5. **Measured dead ends:**
   - int8 isn't faster on this ARM CPU for long texts
   - Apple GPU (MPS) only helps bulk jobs
   - CoreML is 10–30× slower
   - sentence-transformers' official ONNX extra can't be installed on this stack

## 1. Where embedding time goes (call-site analysis, one LoCoMo conversation)

| Call site | Calls | Texts | Avg batch | Share of texts |
|---|---|---|---|---|
| `episodic.summarize_extractive` (sentence vectors for key-segment selection) | 31 | 1,204 | 38.8 | **70%** |
| `episodic.ingest` (turn vectors) | 214 | 419 | **2.0** | 24% |
| queries (`retrieval._encode`) | 50 | 50 | **1.0** | 3% |
| episode summary vectors | 31 | 31 | 1 | 2% |
| fact vectors | 18 | 19 | 1.1 | 1% |

Only 6.6% of texts repeat exactly, so caching document embeddings wouldn't help much. Three facts drive the fixes:
- The summarizer embeds about 3× more text than ingestion.
- Ingest calls are tiny, the worst batch shape for transformers.
- Queries are single texts, where per-call overhead dominates.

## 2. Same model (all-MiniLM-L6-v2), different runtimes

Source: [bench_backends.py](../benchmarks/embeddings/bench_backends.py) · results: [results_arm64.json](../benchmarks/embeddings/results_arm64.json)

| Runtime | Bulk ms/text | ms/text in 2-text calls (today's ingest) | Single query p50 / p95 | Recall@5 (blend) | Agreement with PyTorch |
|---|---|---|---|---|---|
| PyTorch CPU (v0.2 default) | 1.69 | 5.13 | 7.1 / 9.7 ms | 0.433 | — |
| PyTorch CPU, 8 threads | 1.70 | 3.70 | 5.7 / 6.4 | 0.433 | identical |
| PyTorch CPU, max_seq 128 | 1.59 | 3.37 | 5.3 / 6.4 | 0.433 | identical (LoCoMo turns are short) |
| PyTorch **MPS** (Apple GPU) | **0.86** | 10.1 | 7.9 / 24.2 | 0.433 | identical |
| **ONNX fp32** | 1.20 | 1.97 | **1.17** / 1.63 | 0.433 | cosine 1.00000 |
| ONNX O3 (graph-optimised export) | 1.12 | 1.70 | 1.10 / 1.28 | 0.432 | top-5 overlap 0.998 |
| ONNX int8 (ARM export) | 1.50 | 2.81 | 0.91 / 1.54 | 0.438 | cosine 0.9965, top-5 overlap 0.91 |
| ONNX int8, 4 threads | 1.12 | 1.65 | **0.66** / 0.89 | 0.438 | same |
| ONNX via CoreML (ANE/GPU) | 11.0 | 33.6 | 36.5 / 46.5 | 0.433 | identical, but **10–30× slower** (only 294/418 graph nodes supported) |

**Findings:**
- **ONNX fp32 is a free win:** identical vectors, 6× faster single queries, 2.6× faster small calls.
- **int8 is hardware- and length-dependent.** It is the fastest for single short queries. On LongMemEval's long assistant turns it was *slower* than PyTorch (357 s vs 215 s for 67k texts): ARM int8 kernels don't amortise on long sequences. Published x86/Graviton results (2.7–3.4×, [Vespa](https://blog.vespa.ai/embedding-tradeoffs-quantified/)) may not transfer, so it stays opt-in (`onnx-int8:` prefix).
- **Threads:** ONNX Runtime's default thread count contends with efficiency cores. 4 threads was best for latency here, so it is now the default (`MEMORABLE_ORT_THREADS` overrides).
- **The Apple GPU helps only bulk work** (re-indexing). Single queries are 3× slower because of launch overhead. Not used on the hot path.

## 3. Models: quality on two memory benchmarks

Sources: [bench_models.py](../benchmarks/embeddings/bench_models.py) and [bench_longmemeval.py](../benchmarks/embeddings/bench_longmemeval.py)
- **LoCoMo:** dense evidence recall@5 over each conversation's turns (1,536 questions).
- **LongMemEval-S:** 150 stratified questions, ~550 turns of haystack each, 67k unique texts.
- Query prompts are used when a model defines them.

| Model | Params | Type | LoCoMo R@5 (raw / blend) | LME turn@10 | LME sess@5 | Bulk ms/text | Single query |
|---|---|---|---|---|---|---|---|
| BM25 (reference) | — | lexical | — | 0.86 | 0.96 | — | — |
| all-MiniLM-L6-v2 (v0.2 default) | 22M | transformer | 0.401 / 0.433 | 0.84 | **0.98** | 1.69 (torch) / 1.20 (onnx) | 7.1 / 1.2 ms |
| potion-base-8M | 8M | static | 0.407 / 0.458 | 0.873 | 0.933 | 0.058 | 0.04 ms |
| **potion-base-32M** (2026-03) | 32M | static | 0.443 / **0.487** | **0.907** | 0.96 | **0.049** | **0.04 ms** |
| potion-retrieval-32M | 32M | static | 0.304 / 0.357 | — | — | 0.048 | 0.04 ms |
| static-retrieval-mrl-en-v1 | — | static | 0.280 / 0.328 | 0.913 | 0.967 | 0.081 | 0.64 ms |
| **mdbr-leaf-ir** (2026-07) | 23M (+768-d projection) | transformer | **0.621** / 0.610 | 0.887 | 0.953 | 0.82* | 6.6 ms* (torch) |

\* Measured while another benchmark competed for CPU, so timings are inflated. The ONNX path matches PyTorch exactly for leaf-ir (cosine 1.00000) and gives the usual speed-up.

The bge-small-en-v1.5 and granite-embedding-small-english-r2 runs were stopped partway. The machine ran out of swap and disk, caused partly by other workloads. LMEB (below) covers bge-small.

**Findings:**
- **Size and MTEB rank don't predict memory retrieval.** This matches [LMEB](https://arxiv.org/abs/2603.12572): across 30 models from 33M to 12B parameters, correlation with MTEB retrieval is only r = 0.27, and e5-small (33M) is within 2.7 points of Qwen3-Embedding-8B.
- **Static models behave like a smarter BM25.** They win turn-level, keyword-style lookups and lose semantic, session-level ones. potion-32M scores 0.68 vs MiniLM's 0.96 on single-session *preference* questions (round-1 per-type table).
- **"Retrieval-tuned" static models did worse on conversation** (potion-retrieval-32M, static-retrieval-mrl on LoCoMo). Domain matters more than the label.
- **leaf-ir** is the strongest transformer per parameter for conversational evidence (+22 points on LoCoMo dense). Its 768-d output doubles vector storage and search cost.

## 4. End to end (full pipeline, LoCoMo, dense + BM25 fused)

Source: [retrieval_eval.py](../benchmarks/locomo/retrieval_eval.py) · results in [benchmarks/locomo/results/](../benchmarks/locomo/results/)

| Configuration | Ingest | Query p50 | Dense turn | **Fused turn** | **Episode** | Fused multi-hop |
|---|---|---|---|---|---|---|
| MiniLM PyTorch (v0.2 / G0) | 46.3 s | 7.9 ms | 0.433 | 0.565 | 0.535 | 0.504 |
| MiniLM ONNX fp32 | 39.0 s | 3.0 ms | 0.433 | 0.565 | 0.535 | 0.504 |
| MiniLM ONNX int8 | 35.1 s | 2.4 ms | 0.439 | 0.569 | 0.546 | 0.507 |
| potion-8M everywhere | **7.1 s** | 1.5 ms | 0.458 | 0.579 | 0.509 | 0.504 |
| potion-32M everywhere | 15.4 s† | 1.8 ms | 0.487 | 0.579 | 0.505 | 0.475 |
| Tiered v1: potion-32M turns + summarizer, MiniLM facts/episodes | 27.1 s† | 4.5 ms | 0.487 | 0.579 | 0.517 | 0.475 |
| **Profile `fast`: potion-32M turns + segmentation, MiniLM facts/episodes/summarizer** (clean run) | **25.0 s** | 3.2 ms | 0.487 | **0.579** | **0.541** | 0.475 |
| leaf-ir everywhere (incl. segmentation) | 43.0 s | 8.4 ms | **0.610** | **0.642** | 0.408 (276 episodes) | **0.571** |
| leaf-ir facts/episodes/summarizer + potion-32M turns | 36.5 s | 8.6 ms | 0.487 | 0.579 | 0.456 | 0.475 |
| MiniLM facts/episodes/summarizer + leaf-ir turns *and* segmentation | 37.3 s | 10.0 ms | 0.610 | 0.642 | 0.507 (276 episodes) | 0.571 |
| **Profile `quality` (default): MiniLM semantic + leaf-ir turns + static segmentation** | 39.1 s | 10.1 ms | **0.610** | **0.642** | **0.541** | **0.571** |

† Measured while another benchmark ran, so timings are inflated; quality is unaffected.

**Ablation: which turn-tier use costs episode recall?**

| Variant | Episodes | Episode recall |
|---|---|---|
| Segmentation static, summarizer static | 471 | 0.517 |
| Segmentation **MiniLM**, summarizer static | 452 | 0.508 |
| Segmentation static, summarizer **MiniLM** | 471 | **0.541** |

The extractive summarizer picks the sentences that become each episode's searchable text, so it needs semantic vectors. Segmentation on static vectors is fine and yields slightly finer episodes. The summarizer therefore stays on the main model (with ONNX, about 1.4 s per conversation).

**Why leaf-ir must not segment.** Episode boundaries compare cosine distances with an absolute floor (`min_surprise = 0.45`), and the scale differs by model:

| Model | Median turn-to-centroid distance, same session / other session | Other-session turns above 0.45 |
|---|---|---|
| MiniLM | 0.420 / 0.458 | 54% |
| potion-32M | 0.405 / 0.400 | 38% |
| leaf-ir | 0.195 / 0.205 | **0%** |

Under leaf-ir, no topic boundary ever fires: 276 long episodes and a collapse in episode recall. Segmentation now has its own embedder (`segment_embedding_model`, which defaults to the static model when available). Any model can then index turns without disturbing episodes. Calibrating the floor per model, or Bayesian changepoint detection, would remove the absolute scale altogether (instruments_analysis.md #8).

**leaf-ir is a turn model, not a semantic-tier model.** It is excellent for question → conversational evidence (dense 0.43 → 0.61, multi-hop 0.50 → 0.57) but worse for ranking long episode summaries (0.456).

**Remaining trade-off of the fast tier:** fused multi-hop is 0.475 vs 0.504. Static turn vectors associate less across paraphrase. Fusing *both* dense turn signals would recover it, at the cost of transformer turn embeddings again.

## 5. What shipped

| Change | Where | Effect |
|---|---|---|
| `OnnxEncoder`: tokenizers + ONNX Runtime + NumPy pooling; query/document prompts; CLS or mean; Dense layers applied in NumPy; sidecar weights materialised (ORT rejects HF-cache symlinks); unknown modules refused | [embeddings.py](../memorable_ai/core/embeddings.py) | identical vectors, 2–6× faster |
| Automatic ONNX backend for plain model names (`MEMORABLE_EMBED_BACKEND=torch` opts out); explicit `onnx:` / `onnx-int8:` / `onnx-o3:` / `torch:` prefixes | `load_encoder` | — |
| `StaticEncoder` (model2vec) via `static:` | `load_encoder` | 30–175× faster |
| Roles and profiles:<br>• `turn_embedding_model` (turn index; profile default leaf-ir / potion-32M)<br>• `segment_embedding_model` (topic segmentation; default potion-32M, falling back to the main model, never the turn model)<br>• `MEMORABLE_EMBEDDING_PROFILE` = `quality` (default) / `fast` / `legacy`<br>• explicit env vars win; `none` disables a role | config, episodic, retrieval, engine | §4 profile rows |
| Vector-space safety: `space_of()`, per-layer space tags, automatic `reembed()` on change (reproduces the contextual turn blend) | storage, engine | upgrades never mix vector spaces |
| Query-embedding LRU (fixed for custom models) | `Embedder` | about 1–7 ms saved per repeated query |
| `pip install memorable-ai[fast]` = onnxruntime + model2vec + numba | setup.py, pyproject | — |

## 6. Recommendations and next experiments

1. **Decided:** the semantic tier stays on MiniLM; leaf-ir only for turns. With leaf-ir doing facts and episodes, episode recall fell to 0.456.
   - **Benchmarks disagree on the turn model:** leaf-ir wins LoCoMo by +12 pts dense, while potion-32M wins LongMemEval turn@10 (0.907 vs 0.887).
   - **Next experiment: fuse both dense turn signals** (static + leaf-ir) in RRF, to keep each one's strength. The static vectors are nearly free; this needs a second turn-vector column.
2. **Add a reranker.** LMEB's clearest gain was reranking (dialogue nDCG 50.8 → 66.4 with Qwen3-Reranker-8B). We already support `rerank=True` with a MiniLM cross-encoder. Measure it on both benchmarks, then try a small modern reranker on the top-20.
3. **Calibrated score fusion** in place of RRF. A June 2026 paper reports +8.8 to +17.2 LoCoMo Hit@1 from score-level lexical + dense fusion ([arXiv 2606.04194](https://arxiv.org/html/2606.04194)).
4. **Dual dense turn signal** (static + transformer) only if multi-hop matters more than ingest cost.
5. **Fine-tuning on Kaggle** (about 30 GPU-hours/week on 2× T4 or a P100; enough for small encoders). Never train on LoCoMo or LongMemEval, which are the evals.
   - **a. Distil a conversational static model:** model2vec / Tokenlearn with a strong teacher (e.g. leaf-ir or a larger encoder) on public dialogue corpora (SODA, MSC, UltraChat). The GPU generates teacher embeddings; the result is a potion-class model tuned for chat turns.
   - **b. Contrastive fine-tune of the semantic tier** (MiniLM / leaf-ir / bge-small) on memory question ↔ evidence pairs. Generate the pairs from dialogue corpora with the NIM LLM. Use tiered hard negatives (HiNS, [arXiv 2601.14857](https://arxiv.org/abs/2601.14857): same-session near misses, same-persona, random) and a Matryoshka loss, so 768-d models can be truncated to 256-d.
   - **c. Evaluate on held-out LoCoMo, LongMemEval and LMEB's dialogue subset.** Ship only if retrieval *and* end-to-end QA improve.

   This needs your go-ahead: it uploads training data to Kaggle and spends GPU quota.

## 7. Research landscape (November 2026)

- **LMEB, a long-horizon memory embedding benchmark:** 22 datasets, 193 tasks across episodic, dialogue, semantic and procedural memory. Dialogue is the hardest type. Size isn't predictive, and reranking helps most. [arXiv 2603.12572](https://arxiv.org/abs/2603.12572)
- **EvoEmbedding:** embeddings for agentic memory; its 4B flagship reports Recall@10 80.5 on LoCoMo / LongMemEval / REALTALK (self-reported). [arXiv 2606.21649](https://arxiv.org/html/2606.21649v1)
- **HiNS:** hierarchical negative sampling for memory-retrieval embeddings; fine-tunes BGE-small; gains on LoCoMo and PersonaMem. [arXiv 2601.14857](https://arxiv.org/abs/2601.14857)
- **Qwen3-Embedding conversational robustness:** structured conversational noise hurts coarse memory representations; query prompting mitigates it. [arXiv 2604.06176](https://arxiv.org/pdf/2604.06176)
- **mdbr-leaf-ir:** 23M, ranked #1 among ≤100M-parameter models on BEIR/RTEB, distilled from arctic-embed-m. [MongoDB blog](https://mongodb.com/company/blog/engineering/leaf-distillation-state-of-the-art-text-embedding-models)
- **Static embeddings:** [model2vec / potion](https://github.com/MinishLab/model2vec); [Tokenlearn 0.2](https://minish.ai/blog/2025-05-31-tokenlearn-release/); [Sentence Transformers static models](https://huggingface.co/blog/static-embeddings) (100–400× faster on CPU, ≥85% of the quality)
- **Granite Embedding R2** (IBM, small English 47M; multilingual 97M, 2026). [HF blog](https://huggingface.co/blog/ibm-granite/granite-embedding-multilingual-r2)
- **Quantization trade-offs:** [Vespa](https://blog.vespa.ai/embedding-tradeoffs-quantified/) (int8 2.7–3.4× on Graviton4, 94–98% quality). fastembed and sentence-transformers' ONNX paths are close on CPU ([Qdrant](https://qdrant.tech/blog/oxidizing-cross-encoders/index.md)).
- **What memory products use:** Mem0 OSS defaults to OpenAI `text-embedding-3-small`, a hosted API ([docs](https://docs.mem0.ai/components/embedders/config.md)). Running locally (static + ONNX) avoids a network round trip per write and per query.

## 8. Reproduce

```bash
pip install -e ".[fast]"
HF_HUB_OFFLINE=1 python -m benchmarks.embeddings.bench_backends            # runtimes (same model)
HF_HUB_OFFLINE=1 python -m benchmarks.embeddings.bench_models --models static:minishlab/potion-base-32M,MongoDB/mdbr-leaf-ir --out r.json
HF_HUB_OFFLINE=1 python -m benchmarks.embeddings.bench_longmemeval --questions 150
MEMORABLE_LLM__TURN_EMBEDDING_MODEL=static:minishlab/potion-base-32M \
  python -m benchmarks.locomo.retrieval_eval --convs 10 --k 5 --embedder sentence-transformers/all-MiniLM-L6-v2
```

LongMemEval-S is downloaded to `benchmarks/longmemeval/data/`, which is gitignored.
