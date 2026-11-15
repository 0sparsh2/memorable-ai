# Instruments: maths, models, languages and hardware that can make Memorable better

**Date:** 2026-11-15, written after step one (G0) of the v0.3 roadmap.
**Priorities:** quality and speed first, set by the user. Novel methods are welcome where they are measurable.
**Rule:** optimise what has been measured. Every proposal below states the number it should move and how to check it.

---

## Status after the 2026-11-15 study

Details: [g0_deep_evaluation.md](g0_deep_evaluation.md) and [embedding_study.md](embedding_study.md).

| # | Instrument | Outcome |
|---|---|---|
| 1 | Micro-batched embeddings | **Superseded** by tiering: turns use a static model (0.05 ms/text), so small-batch overhead now applies only to the few fact and episode vectors |
| 2 | Device routing (MPS) | **Measured:** 1.75× on bulk, 3× slower per query, so only for re-index/backfill jobs and kept off the hot path |
| 3 | Reuse turn vectors in summaries | **Rejected by ablation:** a static summarizer cost 2.4 pts of episode recall; the summarizer stays on the main model (via ONNX) |
| 6 | Quantized ONNX | **Shipped:** fp32 ONNX is used automatically (identical vectors, 6× faster queries). int8 is opt-in because it is hardware- and length-dependent (slower on ARM for long texts) |
| 14 | Numba | **Shipped as optional:** push PPR 25–50× faster than Python; org fallback 2.3 ms vs exact 64 ms |
| 15 | Hypothesis | **Shipped (dev):** found a real defect (dead-node seeds); with the new concurrency test and a repro, four G0 defects found and fixed in total |
| new | Tiered embeddings + profiles | **Shipped:** `quality` profile (leaf-ir turns) takes LoCoMo fused turn recall 0.565 → 0.642 with ingest 46 → 39 s; `fast` profile (static turns) gives 0.579 with ingest 25 s |
| new | Vector-space tags + auto re-embed | **Shipped:** model changes can no longer mix vector spaces |
| new | Fine-tuning (Kaggle) | **Planned** (embedding_study.md §6), awaiting go-ahead |

## TL;DR

1. **The bottleneck is embedding inference, not SQL or the graph.**
   - About 70% of runtime, excluding the one-off library import.
   - On CPU, batching gives about 3×. The Apple GPU gives another 1.75× for bulk work but is 3× *slower* for single queries.
   - The levers are batching, caching and device routing first, then quantized ONNX.
2. **Sparse linear algebra was the right graph instrument.** G0 is 8–20× faster than v0.2's graph ranking and far more accurate at scale. A rank-aware convergence tolerance gave another ~40% with no change in ranking quality.
3. **The biggest *quality* instruments are statistical**, and none of them are in the product yet:
   - conformal risk control for abstention (a guaranteed false-suggestion rate)
   - calibrated score fusion in place of plain RRF
   - Bayesian online changepoint detection for episode boundaries
   - submodular context packing
   - ACT-R activation for memory strength
4. **Systems models:**
   - Treat L0 as an **event-sourced log**, so every derived layer can be replayed (re-extract or re-embed after a model change).
   - Use an **LSM-style base + delta** for graph writes at org scale.
5. **Languages:**
   - Not a rewrite yet. **Numba** for the few pure-Python hot loops comes first.
   - Python 3.14 free-threading is a cheap later win, since we already use threads.
   - Rust only if a standalone server needs predictable latency.

## 1. Measurements behind this analysis

| What | Setup | Result |
|---|---|---|
| End-to-end profile | 1 LoCoMo conversation (~600 turns, 150 queries) through the real pipeline, MiniLM, CPU (Apple M-series, 10 cores, NumPy on Accelerate) | Total 13.9 s:<br>• library import 4.7 s (one-off)<br>• embedding inference **6.4 s, ~70% of the rest**<br>• extractive episode summaries 1.8 s (mostly their embeddings)<br>• SQL and graph below the profiler's 0.4 s cut |
| Embedding device and batching | MiniLM, 419 LoCoMo turns | **CPU:** 1.67 ms/turn batched; **4.8 ms/turn** in today's 1–2-text calls; 6.0 ms per single query<br>**MPS (Apple GPU):** 0.96 ms/turn batched; 17.2 ms per single query |
| Graph ranking (G0) | [ppr_engines.py](../benchmarks/graph/ppr_engines.py), synthetic Zipf graphs | per-user 1.7 ms (v0.2: 34 ms) · team 12.4 ms (186 ms) · org 86 ms (751 ms; p95 199 ms vs 3.0 s)<br>Overlap@20 with exact PPR: 0.91–0.93 (v0.2: 0.89 → 0.52 → 0.52) |
| Convergence tolerance | team graph | tol 1e-6 → 1e-4: 10.7 → 6.6 ms; top-20 overlap unchanged (0.91 → 0.92, within noise) |
| LoCoMo retrieval regression | 10 conversations, 1,536 questions, no LLM | Identical recall to v0.2 (turn fused 0.565, episode 0.535); ingest 46 s (v0.2: 57 s) |

## 2. Applied in this commit

| Instrument | Field | Effect |
|---|---|---|
| CSR sparse matrices + SpMV power iteration | linear algebra | 8–20× faster graph ranking; no hop/node caps, so it's correct at scale |
| Exact induced-subgraph masking (degrees recomputed as `A @ mask`) | graph maths | The "self" hub is excluded without distorting the other nodes; tier isolation is exact |
| Block matrix views (shared org matrix reused, only the user's small matrix remapped) | linear algebra | User + org queries cost no copy of the org graph |
| Rank-aware tolerance (1e-4) | numerical analysis | About 40% fewer iterations; ranking unchanged |
| Change cursor: `MAX(id)`, `MAX(invalidated_at)` as index lookups | systems | Per-query sync dropped from about 8 ms to a few lookups |
| Query-embedding LRU | caching | About 6 ms saved per repeated query (agent tool loops repeat the user message on every call) |
| Vectorised top-k (kind codes + `argpartition`) | numerics | Removed an O(nodes) Python loop (team: 48.7 → 18.6 ms before the cursor fix) |

## 3. Ranked opportunities

Impact: ●●● large, ●● medium, ● small. "Deps" means a new package is needed, which I'll ask about before adding.

| # | Instrument | Moves | Expected | Cost | Deps |
|---|---|---|---|---|---|
| 1 | **Micro-batched embeddings** in the background worker: queue texts, encode in batches of ≥32 | ingest time | **~3×** on embedding time (4.8 → 1.67 ms/text, measured) | S | none |
| 2 | **Device routing:** queries on CPU, batches ≥32 on MPS/CUDA when present | ingest, re-index | another ~1.75× on batches (measured); avoids the 3× MPS penalty on single queries | S | none |
| 3 | **Reuse turn vectors** in extractive summaries instead of re-embedding sentences | ingest | most of the 1.8 s summary cost | S | none |
| 4 | **Conformal risk control** for abstention (procedure recall verdict, later memory injection) | quality: wrong-suggestion rate | a *guaranteed* bound P(wrong) ≤ α from logged runs, replacing the hand-set θ/margin | M | none |
| 5 | **Calibrated fusion:** per-signal score → probability (isotonic), then probability fusion or learning-to-rank in place of plain RRF | retrieval recall@k | RRF throws away confidence; calibrated fusion typically beats it once there are labels (LoCoMo has evidence labels) | M | none |
| 6 | **Quantized ONNX embeddings** (int8, ONNX Runtime or OpenVINO backend in sentence-transformers) | all embedding time | ~2× on CPU expected, to be measured; small quality loss to be measured | S | onnxruntime |
| 7 | **Event-sourced projections:** each derived row records `(projection, version, model)` and can be rebuilt from L0 | quality over time, migrations | re-extract or re-embed the history when a better model ships; makes dreaming safe | M | none |
| 8 | **BOCPD** episode segmentation (Adams & MacKay 2007) in place of the z-score heuristic | episode quality → multi-hop recall | principled run-length posterior; handles gradual drift | M | none |
| 9 | **Submodular context packing:** budgeted facility location + query-aspect coverage (Lin & Bilmes 2011) in place of MMR + truncation | answer quality per token | approximation guarantees under the token budget | M | none |
| 10 | **ACT-R base-level activation** `B = ln Σ t_j^(-d)` as the memory-strength prior | ranking ties, forgetting, consolidation | principled recency × frequency, replacing ad-hoc importance and decay; also fixes the PPR score ties seen in benchmarks | S | none |
| 11 | **Contextual bandit / Thompson sampling** over weight profiles (RRF weights, typed-PPR weights per query type) | retrieval quality online | learns from which memories were actually used; no offline tuning on the test set | M | none |
| 12 | **MDL / BPE over tool-call sequences** for sub-procedure discovery | procedure library quality | keep a sub-procedure only if it compresses the trace corpus; replaces the "prefix shared by ≥m" rule | M | none |
| 13 | **LSM graph:** base CSR + delta COO for appends + tombstones, merged in the background | graph latency after writes | avoids O(nnz) rebuilds (org first load 1.6 s) when the org tier is written often | M | none |
| 14 | **Numba** for pure-Python loops: local-push PPR (72 ms per-user today), BM25 scoring, value-flow matching | fallback latency, CPU | typically 20–100× on such loops | S | numba |
| 15 | **Property-based tests** (Hypothesis) for invariants: idempotent ingest, no cross-user leakage, lifecycle/promotion state machines | correctness | finds edge cases that example tests miss | S | hypothesis (dev) |
| 16 | **Embedding quantization for storage** (int8/binary + rescoring, Matryoshka truncation) | memory and scale | large memory savings with small recall loss (published); matters at ≥1M vectors | M | none |
| 17 | **Free-threaded Python 3.14** for parallel CPU work (extraction regexes, BM25, push) | throughput | scales the threads we already run across cores | S | ecosystem wheels |

## 4. Details by field

### 4.1 Hardware and inference (the measured bottleneck)
- **Batching beats device choice.** The model's fixed cost per call dominates small batches: 4.8 ms per text at batch size 2 vs 1.67 ms at 64. Ingestion encodes per call today, so a micro-batching queue on the background worker is the cheapest big win. Writes are already asynchronous, so latency is unaffected.
- **Routing by batch size.** MPS is 1.75× faster on bulk batches and 3× slower on single queries, because of kernel-launch overhead. Policy:
  - queries → CPU
  - ingest batches ≥32, backfills and re-indexing → GPU when available
  - on CUDA the crossover will differ, so measure on first use
- **Quantized ONNX** (ONNX Runtime / OpenVINO via sentence-transformers' `backend=`) is the next step for CPU-only deployments. Needs a new dependency.
- **Static embeddings** (model2vec-style) are orders of magnitude faster on CPU at lower quality. They could serve as a *fast tier* for L0 turn indexing, with MiniLM-class models kept for facts and episodes. Only worth it if measured recall holds.
- **Cold start:** the 4.7 s import. Load the embedder on a background thread at engine construction. Matters for CLIs and serverless.
- **GPU for the graph or vector search isn't warranted** below about 10M edges or vectors. G0 already runs a team graph in about 12 ms on CPU.

### 4.2 Linear algebra and graph maths
- **Done:** CSR PPR, masking, block views, rank-aware tolerance.
- **Next:**
  - *Hub dampening:* symmetric normalisation `D^-1/2 A D^-1/2` or a degree penalty, as a quality knob for hub entities (popular people, the user's employer).
  - *Leiden communities* for dreaming. Unlike Louvain, it guarantees connected communities.
  - *Query-adaptive typed weights* (G2) are just a different `Σ w_k A_k`, so per-query cost stays one sparse sum (or a cached profile).
- **Not needed now:** Chebyshev/Anderson acceleration. The tolerance change captured most of the iteration savings.

### 4.3 Statistics and decision models (quality)
- **Conformal risk control** (Angelopoulos et al., 2022):
  - Choose the abstention threshold from logged (score, outcome) pairs so the expected wrong-suggestion rate is ≤ α, with a finite-sample guarantee and no distribution assumptions.
  - The observation runs designed for procedural P3 produce exactly this data.
  - As far as I know this hasn't been applied to agent-memory abstention. That makes it a candidate novel contribution.
- **Calibrated fusion:** RRF is robust but rank-only. Isotonic calibration per signal on LoCoMo's evidence labels, then summing calibrated log-odds (or a small LambdaMART), is the standard next step. It can be measured with the existing `retrieval_eval.py`.
- **BOCPD:** keeps a posterior over the "run length" since the last topic change, using turn embeddings as observations. It replaces the hand-tuned `surprise_z` and `min_surprise` thresholds with a hazard rate and a predictive model.
- **Submodular packing:** context selection is budgeted coverage. A monotone submodular objective (relevance + diversity + coverage of the query's aspects) under a token knapsack gets approximation guarantees with greedy cost-scaled selection.
- **ACT-R activation:** `B_i = ln Σ_j (t_now − t_j)^(-d)` over a memory's access and mention times (d ≈ 0.5). It is the best-validated cognitive model of recall probability (Anderson & Schooler, 1991). Use it as the node prior in PPR tie-breaking and as the decay signal in consolidation.
- **Bandits:** Thompson sampling over a handful of weight profiles, rewarded when a retrieved memory is used: cited in the answer, or followed in a procedure run. This makes weights adaptive per deployment without test-set tuning.

### 4.4 Systems models
- **Event sourcing:** L0 turns, and later tool events, are the immutable log. Facts, edges, episodes, summaries and procedures are *projections* with `(projection, version, model)` stamps, so:
  - a model upgrade becomes a replay job
  - dreaming writes new projection versions instead of mutating rows
  - provenance (`derived_from`) is the projection's lineage
- **LSM graph:** appends go to a small delta matrix added in the SpMV. Invalidations become tombstones. A background merge folds both into the base when the delta exceeds about 5%. This bounds post-write latency for a busy org tier.
- **LLM cascade with validators:** try a small NIM model first and escalate to a larger one only when the deterministic validator rejects the output or confidence is low. Quality is kept, because validators gate the output. Speed improves, because most calls stay small. Value-of-information gating skips extraction on low-information turns; we already compute information density for key segments.
- **SLOs and continuous benchmarks:** per-stage OTel spans, plus `ppr_engines.py` and `retrieval_eval.py` in CI with regression thresholds.

### 4.5 Languages and runtimes
| Option | Verdict |
|---|---|
| **Numba** | First choice for the few pure-Python hot loops (push PPR, BM25, value-flow). No build toolchain needed. |
| **Python 3.14 free-threading** (PEP 779, supported in 3.14) | Our background worker and threads could use multiple cores once NumPy, SciPy, torch and SQLAlchemy wheels are confirmed. Test later; little code change. |
| **Rust (PyO3/maturin)** | Only if numba falls short, or for a standalone memory server with strict p99s. Higher maintenance and wheel builds. Not yet. |
| **DuckDB** | Good for analytical dreaming jobs (statistics over runs, community summaries) without loading Postgres. Optional. |
| **Postgres** | Production multi-process store (already supported); pgvector arrives with the backends track. |

### 4.6 Correctness instruments
- **Property-based tests** (Hypothesis), checking these invariants on random operation sequences:
  - ingest is idempotent under resent histories
  - no user's rows ever appear in another user's view
  - lifecycle and promotion state machines only take legal transitions
  - cascade deletes leave no unsupported derived rows
- **Differential tests:** GraphEngine vs networkx on random graphs; push vs exact (two are already in `test_graph_engine.py`).

## 5. What I would not do now
- **A Rust rewrite:** the bottleneck is model inference, which Rust doesn't change.
- **A GPU graph engine** (cuGraph) or **GPU vector search** (FAISS-GPU): there's no need below about 10M edges or vectors.
- **HNSW or approximate vector indexes:** brute-force NumPy on Accelerate BLAS is exact and fast at current scale. Switch at ≥1M vectors (backends track).
- **Neo4j as the default:** the in-process CSR engine is faster at our scale, and it keeps self-hosting dependency-free.

## 6. Proposed order (interleaved with the roadmap)
1. **Now, no new dependencies:** #1 micro-batching, #2 device routing, #3 vector reuse, #10 ACT-R prior. Ship these with P1, since P1 touches ingestion anyway.
2. **With P3** (observation runs give labelled data): #4 conformal abstention, #5 calibrated fusion, #11 bandits.
3. **With dreaming:** #7 event-sourced projections, #12 MDL sub-procedures, Leiden communities.
4. **When evaluations resume:** #8 BOCPD and #9 submodular packing, measured on LoCoMo.
5. **On request (new dependencies):** #6 onnxruntime, #14 numba, #15 hypothesis.
