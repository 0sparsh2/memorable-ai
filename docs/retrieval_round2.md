# Retrieval round 2: fine-tuning on Kaggle, a second turn signal, rerankers

**Date:** 2026-11-15. Follows [embedding_study.md](embedding_study.md) §6 (items 1, 2 and 5).

## 0. Preregistration (written before any result)

**Evals:** LoCoMo (1,536 questions, all 10 conversations) and LongMemEval-S (150 stratified questions).
- Neither is ever used for training.
- Measured with [turn_signals.py](../benchmarks/embeddings/turn_signals.py), which reproduces the pipeline's turn layer: contextual blend 0.5, BM25, weighted RRF k=60 over top-50 lists.
- A winner is confirmed end-to-end with [retrieval_eval.py](../benchmarks/locomo/retrieval_eval.py).
- Differences are reported with a 95% paired bootstrap CI over questions. LongMemEval's 150 questions resolve only about ±3 points.

**Ship bars.** A change ships only if it clears all of these:

| Change | Must improve | Must not regress |
|---|---|---|
| Fine-tuned turn model (vs leaf-ir) | LoCoMo fused turn@5 by ≥ 1.0 pt, CI excluding 0 | LongMemEval turn@10 and session@5 by more than 1 question (0.7 pt) |
| Fine-tuned semantic model (vs MiniLM) | LoCoMo episode recall@5 end-to-end by ≥ 1.0 pt | LongMemEval session@5; LoCoMo fused turn@5 |
| Fine-tuned static model (vs potion-32M) | fast-profile fused score on both benchmarks | segmentation (episode count within ±10%) |
| Second dense turn signal | quality-profile fused LoCoMo turn@5 by ≥ 1.0 pt, CI excluding 0 | LongMemEval turn@10 / session@5; ingest time by more than 10% |
| Reranker | fused turn@5 / turn@10 on both, CI excluding 0 | p50 query latency must stay under 50 ms on this laptop (M-series, CPU) |

**Decoys:** each comparison runs against the unmodified base model and today's profile configurations, on identical candidate sets.

## 1. Training data (Kaggle)

No public dataset pairs a question about a past conversation with the turn that answers it. HiNS did not release its data. So we generate one.

**Generator:** Qwen3-4B-Instruct-2507 under vLLM on 2× T4, given dated multi-session conversations from:
- ConversationChronicles (5 dated sessions per conversation; the closest match to LoCoMo)
- MSC
- SODA
- WildChat-1M (real user/assistant chats, English, non-toxic)

**What it writes:**
- Question types: fact, temporal, preference, paraphrase (low lexical overlap), multi-turn; plus user-fact and assistant-recall for user/assistant chats.
- Every question carries the ids of its evidence turns.

**Excluded sources:** ShareGPT and UltraChat, because LongMemEval builds its filler sessions from them.

**Filtering and negatives:**
- **Round-trip filter:** a teacher model (Qwen3-Embedding-0.6B) must rank an evidence turn in the top 10 of its own conversation. This drops vague or unanswerable generations.
- **Negatives, tiered as in HiNS:**
  - hard: the teacher's best non-evidence turn in the same conversation
  - medium: a random turn from the same conversation
  - easy: other conversations, via in-batch negatives
- **False-negative guard:** skip any candidate scoring above 95% of the evidence score, and skip turns next to an evidence turn.

Code: [training/kaggle/](../training/kaggle/).

## 2. Score fusion replaces rank fusion (shipped, default)

**Setup:** the same two signals as the quality profile (leaf-ir dense + BM25), with a different way of combining them.
- **RRF** keeps only each candidate's rank, so a near-tie in dense similarity counts as much as a decisive BM25 match.
- **Score fusion** keeps the margins: each signal's scores are z-normalised over the candidate pool, then summed with weights.

| Turn layer, offline harness | LoCoMo turn@5 | LME turn@5 | LME turn@10 | LME session@5 |
|---|---|---|---|---|
| RRF (leaf .5, BM25 1), shipped in v0.3 quality | 0.640 | 0.780 | 0.893 | 0.987 |
| z-scores over all candidates (leaf 1, BM25 .3) | 0.693 | 0.813 | 0.900 | 0.987 |
| **z-scores over the top-50 pool (leaf 1, BM25 .5)**, what the pipeline can compute | **0.690** (+5.0, CI +3.5…+6.5) | **0.813** | **0.893** | **0.987** |
| min-max over the top-50 pool (leaf 1, BM25 .3) | 0.687 | — | — | — |
| + potion as a second dense signal (z, .5) | 0.701 (+1.1 more, CI +0.1…+2.5) | 0.827 | 0.907 | 0.987 |

**Robustness:**
- The gain holds on both halves of LoCoMo: conversations 0–4 go 0.647 → 0.688, and 5–9 go 0.633 → 0.692.
- It holds in every category: multi-hop 0.560 → 0.631, open-domain 0.380 → 0.478, single-hop 0.674 → 0.722, temporal 0.695 → 0.720.
- On LongMemEval, temporal reasoning moved 0.80 → 0.72 (2 of 25 questions). That is within noise; all other types are equal or better.

**End to end** (`retrieval_eval.py`, real pipeline, SQLite FTS5 BM25), quality profile:

| | turn recall@5 (the retriever's own ranking) | episode recall@5 |
|---|---|---|
| `fusion="rrf"` | 0.642 | 0.541 |
| **`fusion="score"`** | **0.701** | **0.575** |

**Shipped:**
- `RetrievalConfig.fusion = "score"` with `fusion_weights = {semantic: 1.0, lexical: 0.5}`, for the turn and episode layers.
- Lexical-only candidates get their exact cosine via `VectorIndex.scores_for`.
- The fact layer (dense + BM25 + graph PPR) keeps RRF until it is measured.
- The hash fallback embedder keeps RRF, because its cosines carry no meaning.

**The second dense turn signal (potion):** worth +1.1 points on LoCoMo, consistent across both halves, at almost no ingest cost. It requires a second vector per turn, so it is deferred until the fine-tuned turn model is measured (§4), since that model may absorb the gain.

## 3. Rerankers (shipped, opt-in)

First stage: score fusion, top 20; each reranker reorders those 20.
- "pure" = the reranker's order alone.
- "blend" = z(reranker) + z(first stage).
- Quality is from the offline harness. Latency is per query on this M-series laptop, measured idle with ONNX Runtime at 4 threads (PyTorch where noted).

| Reranker | Params | LoCoMo turn@5 | LME turn@10 | p50, LoCoMo / LME |
|---|---|---|---|---|
| none (score fusion) | — | 0.690 | 0.893 | — |
| ms-marco-MiniLM-L6-v2, blend | 22M | 0.752 | 0.927–0.933 | 62 / 93 ms (PyTorch 48 / 50) |
| **ettin-reranker-17m-v1**, pure or blend | 17M | **0.745** | **0.920** | **32 / 50 ms** (PyTorch 36 / 40) |
| ettin-reranker-32m-v1, pure | 32M | 0.771 | 0.920 | 82 / 123 ms |
| ettin-reranker-68m-v1, pure | 68M | 0.805 | 0.933 | ~150 / ~450 ms (PyTorch, loaded) |
| ettin-reranker-150m-v1, pure | 150M | 0.820 | 0.927 | ~330 / ~920 ms (PyTorch, loaded) |
| gte-reranker-modernbert-base, pure | 149M | 0.803 | 0.940 | ~310 / ~925 ms (PyTorch, loaded) |

**Findings:**
- Reranking is the largest single gain we have measured: +5.5 to +13 points of LoCoMo turn@5 on top of score fusion. This matches LMEB, where reranking was the clearest win for dialogue.
- **Rank-fusing** the reranker with the first stage ("rrf-mix") is always worse than letting the reranker decide.
- Truncating pairs to **128 tokens costs nothing.** Depth 20 vs 256 tokens, LoCoMo / LME turn@10: MiniLM 0.748 / 0.927 both ways; Ettin-17M 0.746 / 0.920 vs 0.746 / 0.913.
- 128 tokens halves the time on LongMemEval's long assistant turns.
- **Depth 10** is cheaper (Ettin-17M 13 / 22 ms) but keeps only about +4.5 points on LoCoMo.
- **int8** saves at most 20% (MiniLM) and nothing for Ettin, while changing 2–10% of top-5 choices. fp32 stays the default.
- **ONNX Runtime** is not faster than PyTorch for rerankers on this machine (unlike single-query embedding), and more ORT threads are slower: 4 threads is best.
- The ONNX path matches PyTorch exactly (max |Δlogit| ≤ 1e-5). Modular cross-encoders such as Ettin export only the transformer, so the pooling → Dense(GELU) → LayerNorm → Dense head runs in NumPy.

**Shipped:**
- `memorable_ai/core/rerank.py`: `load_reranker`, ONNX first, PyTorch fallback.
- Turn reranking in `recall()`.
- Settings `rerank_depth` (20) and `rerank_blend` (0).
- The reranker is opt-in for now (`rerank=True`). The default model is chosen after the conversation-tuned rerankers (§4).

