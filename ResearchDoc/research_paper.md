# Memorable: Interceptor-Based Long-Term Memory with Hybrid Extraction and Multi-Signal Retrieval

**Working title · draft · May 2026**

| | |
|---|---|
| **Status** | Living document — updated as benchmarks complete |
| **Artifacts** | `ResearchDoc/figures/output/`, `benchmarks/locomo/data/measured_subset.json` |
| **Competitive research** | [`agent_memory_landscape_2026/report.md`](../agent_memory_landscape_2026/report.md) |

---

## Abstract

Long-horizon conversational agents require memory systems that (1) integrate without refactoring application code, (2) maintain factual consistency over hundreds of sessions, and (3) retrieve the right evidence under tight latency and token budgets. We present **Memorable**, an open memory layer that combines **transparent LLM interception** (in the style of Memori), **research-backed extraction and update operations** (Mem0, arXiv:2504.19413), and **optional graph-augmented retrieval** (Supermemory/Zep-inspired). Memorable stores memories in **SQL-first** storage with **bi-temporal validity** fields, **SQLite FTS5** keyword search, and **hybrid retrieval** fusing semantic embeddings, BM25, entity overlap, and graph hops. We implement a **LoCoMo-compatible benchmark harness** aligned with snap-research/LOCOMO and Mem0-style LLM-as-judge evaluation. On a measured subset (1 conversation, 10 questions, NVIDIA NIM `deepseek-v4-flash`), Memorable achieves **83% evidence recall** on temporal questions but **10% end-to-end judge accuracy**, indicating that **retrieval is ahead of answer generation** in our current stack. We document gaps versus published Mem0 baselines and outline a roadmap toward full-leaderboard measurement and production hardening.

**Keywords:** agent memory, LoCoMo, long-term dialogue, retrieval-augmented generation, LLM interception, temporal knowledge graphs

---

## 1. Introduction

### 1.1 Motivation

Production agents must remember user preferences, prior decisions, and narrative context across sessions. Naïve approaches—stuffing full chat logs into the context window—scale poorly in **latency**, **cost**, and **accuracy** as conversations grow [Mem0 paper; LoCoMo benchmark]. Dedicated memory products (Mem0, Zep, Supermemory, Memori) trade off **integration friction**, **storage model**, and **research validation** differently; no single open system has combined **zero-code interception**, **SQL portability**, and **published benchmark methodology**.

### 1.2 Contributions

1. **Memorable architecture** — Interceptor → hybrid extraction → ADD/UPDATE/DELETE/NOOP update → SQL + optional Neo4j → multi-signal retrieval with identity-aware scoping.
2. **Temporal and validity model** — `valid_from`, `valid_to`, `invalidated_at`, `superseded_by_id` on memory rows (Zep-inspired).
3. **LOCOMO harness** — Dataset loader, per-conversation isolation, evidence `dia_id` recall, token F1, and LLM judge pipeline.
4. **Competitive deep research** — 14 validated JSON artifacts comparing Mem0, Zep, Supermemory, Letta, LangMem, and benchmarks (May 2026).
5. **Initial measured results** — Honest subset scores with analysis of retrieval–generation gap.

### 1.3 Paper organization

§2 Related work · §3 System design · §4 Implementation · §5 Experiments · §6 Discussion · §7 Conclusion · References

---

## 2. Related Work

### 2.1 Mem0 (arXiv:2504.19413)

Mem0 proposes LLM-based memory extraction, consolidation with **ADD/UPDATE/DELETE** operations, and hybrid retrieval. The paper reports **~26% relative improvement** over OpenAI’s memory on LoCoMo with an LLM judge; 2026 product materials cite **~91.6%** judge accuracy on full leaderboards. Memorable **ports** the extraction/update/retrieval patterns while preserving **interceptor** integration and **SQL-first** deployment.

### 2.2 LoCoMo (snap-research/LOCOMO)

LoCoMo is a benchmark of **10 long conversations** and **~1,540 scored QA pairs** across single-hop, multi-hop, temporal, and open-domain categories. Official scripts use token **F1**; community leaderboards (e.g., Mem0) often report **LLM-as-judge** accuracy. Our harness supports both.

### 2.3 Zep / Graphiti (temporal KG)

Zep emphasizes **bi-temporal** validity and temporal knowledge graphs for enterprise agents. Memorable adopts **validity columns** and invalidation semantics in SQL without requiring a graph database.

### 2.4 Memori and Supermemory

**Memori** demonstrates interceptor-based integration with SQL storage. **Supermemory** emphasizes graph-centric multi-hop retrieval and connectors. Memorable unifies interceptor ergonomics with optional graph retrieval and research-backed extraction.

### 2.5 RAG and full-context baselines

Full-context GPT-style baselines set an upper bound on accuracy at unsustainable cost. Memorable targets **high recall with bounded context** injection at query time.

---

## 3. System Design

### 3.1 Design principles

| Principle | Rationale |
|-----------|-----------|
| Zero friction | Interceptor patches LiteLLM/OpenAI calls — no `add_memory()` in app code |
| Research first | Hybrid extraction, update ops, fusion weights from literature |
| SQL first | SQLite/Postgres/MySQL; optional Neo4j for graph |
| Honest measurement | Separate retrieval recall from end-to-end judge |

### 3.2 Pipeline

See **Figure 6** (`figures/output/fig6_pipeline_schematic.png`).

```
Application → Interceptor (inject context, capture turns)
           → Extract (pattern | LLM | hybrid)
           → Update (ADD / UPDATE / DELETE / NOOP)
           → Store (SQL + embeddings + FTS5)
           → Retrieve (semantic + BM25 + entity + graph)
           → Answer LLM (benchmark / user query)
```

### 3.3 Memory schema (conceptual)

Each memory row carries: `content`, `memory_type`, `embedding`, `namespace`, `session_id`, `extra_metadata` (e.g., `dia_id`, `canonical_user_id`), and temporal fields for invalidation and supersession.

### 3.4 Retrieval fusion

HybridRetriever combines signals with configurable weights (defaults: semantic 0.35, BM25 0.25, entity 0.15, graph 0.25). IdentityResolver scopes queries to a canonical user across session aliases.

---

## 4. Implementation

| Module | Role |
|--------|------|
| `memorable_ai/core/interceptor.py` | LLM call interception |
| `memorable_ai/core/extraction.py` | Hybrid memory extraction |
| `memorable_ai/core/update.py` | Mem0-style update operations |
| `memorable_ai/core/retrieval.py` | Multi-signal search |
| `memorable_ai/core/storage.py` | SQL + FTS5 + temporal columns |
| `memorable_ai/core/temporal.py` | Timestamped ingest |
| `memorable_ai/core/identity.py` | Cross-session user resolution |
| `benchmarks/locomo/` | LOCOMO loader, judge, metrics |

**LLM providers:** NVIDIA NIM (OpenAI-compatible), Google Gemini, via LiteLLM (`memorable_ai/core/llm_client.py`).

---

## 5. Experiments

### 5.1 Benchmark protocol

- **Dataset:** snap-research/LOCOMO `locomo10.json` (10 conversations).
- **Ingest:** Per-conversation SQLite DB; turns stored with `dia_id` and session timestamps.
- **Retrieval:** `search_memories(query, limit=12)`.
- **Answer:** Short phrase from retrieved context (truncated prompt caps).
- **Judge:** Binary CORRECT/WRONG vs gold (Mem0-style).
- **Secondary:** Token F1 (Snap); evidence `dia_id` recall.

### 5.2 Measured subset (2026-05-17)

| Setting | Value |
|---------|--------|
| Conversations | 1 (`conv-26`) |
| Questions | 10 (stratified: 6 temporal, 3 multi-hop, 1 open-domain) |
| Answer model | `deepseek-ai/deepseek-v4-flash` (NVIDIA NIM) |
| QA timeout | 300s answer / 180s judge |

**Initial run (`baseline_raw` ingest, minimal context):**

| Metric | Overall | Multi-hop | Temporal | Open-domain |
|--------|---------|-----------|----------|-------------|
| **LLM-judge accuracy** | **10%** | 33% | 0% | 0% |
| **Mean token F1** | 12.9% | 33.3% | 4.8% | 0% |
| **Evidence recall** | — | 33% | **83%** | 0% |

### 5.3 Ablation ladder (same subset, 2026-05-17)

| Experiment | Ingest | Context | Judge acc | F1 | Temp recall |
|------------|--------|---------|-----------|-----|-------------|
| `retrieval_only_dated` | raw | dated | **80%** | 80% | 83% |
| `oracle` | gold `dia_id` | dated | **60%** | 42.9% | 100% |
| `dated_context` | raw | dated | **40%** | 30.8% | 83% |
| `baseline_raw` (initial) | raw | minimal | **10%** | 12.9% | 83% |

*Figure 7* (`fig7_ablation_ladder.png`) visualizes this ladder. Artifacts: `ResearchDoc/experiments/*.json`, `experiment_log.md`.

**Interpretation:** Session-date anchors in the answer prompt (`dated_context`) raised overall judge accuracy **10% → 40%** with unchanged retrieval (83% temporal evidence recall). Oracle ingest (perfect evidence) reaches **60%**, so roughly **half** of remaining error is retrieval and **half** is answer extraction / judge strictness—not a broken store.

Early `extracted` runs (0% judge) were invalidated: facts were dropped by the Mem0 update LLM (`NOOP`) and lacked `dia_id` metadata. Fixed by direct storage, session `dia_ids`, raw-turn fallback, and rate-limit backoff (2026-05-17).

**Comparison to Mem0 published category targets (LLM-judge, not same split):**

| Category | Memorable (`dated_context`) | Mem0 (target) |
|----------|----------------------------|---------------|
| Multi-hop | 33% | 72% |
| Temporal | 50% | 68% |
| Open-domain | 0% | 65% |
| Overall | 40% | 72.5% |

*Figures:* `figures/output/fig1`–`fig7`.

### 5.4 Key finding: retrieval–generation gap

Temporal questions show **~83% evidence recall** but only **50% judge accuracy** after the dated-context fix (was 0% before). Retrieved turns often contain the fact (e.g., support-group attendance) while the answer LLM fails to extract the **calendar date** gold answer requires.

**Multi-angle diagnosis** (mathematical, biological, architectural, competitive): see [`diagnosis_retrieval_generation_gap.md`](diagnosis_retrieval_generation_gap.md). Headline causes:

1. **Benchmark ingest stores raw dialogue**, not Mem0-style extracted facts (unfair vs published Mem0 numbers).
2. **`session_date` is in metadata but omitted from the answer prompt** — gold uses absolute dates; dialogue uses “yesterday” / “last week”.
3. **Subset skew:** 6/10 questions are temporal (category 2).
4. **NIM latency** produced empty answers on some runs before timeout tuning.

Mitigations: Mem0-fair extraction ingest; inject session timestamps into QA context; temporal normalization at write time; stratified benchmark + ablations.

### 5.5 Planned experiments

- [ ] Full LoCoMo (1540 questions, 10 conversations)
- [ ] LongMemEval subset
- [ ] Ablations: w/o BM25, w/o graph, pattern-only extraction
- [ ] Latency and token savings vs full-context baseline

---

## 6. Discussion

### 6.1 Threats to validity

- **Subset size** — 10 questions is insufficient for leaderboard claims.
- **Judge variance** — LLM judges are noisy; F1 and recall should be reported together.
- **Provider latency** — NIM timeouts required 180–300s client limits; scores may improve with Gemini or local models.

### 6.2 Positioning

Memorable’s differentiator is **deployment shape** (interceptor + SQL + optional graph), not yet **SOTA LoCoMo scores**. Competitive research (May 2026) shows Mem0 leads published accuracy; Zep leads temporal modeling; Memori leads integration ergonomics.

---

## 7. Conclusion

We described Memorable, an interceptor-based long-term memory system combining Mem0-style extraction/update, Zep-inspired temporal validity, and multi-signal retrieval with a LoCoMo measurement harness. Initial results show strong **retrieval recall** on temporal items but weak **end-to-end judge accuracy**, guiding next work on answer prompting, full-benchmark scale, and ablations. Artifacts and figures are maintained in **`ResearchDoc/`** and updated as benchmarks complete.

---

## 8. Competitive Landscape (Detailed)

Our May 2026 deep-research pass (14 validated JSON artifacts under `agent_memory_landscape_2026/results/`) surfaces three distinct integration philosophies. **Mem0** is API-first: explicit `add`/`search`, 21 framework adapters, and optional **Mem0^g** graph memory on Neo4j—no universal LLM interceptor. **Zep / Graphiti** targets enterprise temporal knowledge graphs with bi-temporal `valid_from` / `valid_to` / `invalid_at` semantics and multi-backend graph stores (Neo4j, FalkorDB, Kuzu, Neptune). **Memorable** occupies the **Memori-style interceptor** niche: `memory.enable()` patches LiteLLM/OpenAI/Anthropic with zero app refactors, SQL-first persistence (SQLite/Postgres/MySQL), and optional Neo4j—combining Mem0-inspired extraction/update (ADD/UPDATE/DELETE/NOOP, hybrid retrieval) with Zep-inspired validity columns on a single `memories` table rather than a mandatory graph DB.

On published benchmarks, the gap to leaders is large and well-quantified. Mem0’s 2026 product materials claim **91.6%** LoCoMo LLM-judge accuracy, **93.4%** on LongMemEval, and **64.1% / 48.6%** on BEAM at 1M / 10M tokens; the 2025 paper reports a **~26% relative** judge gain over OpenAI memory and **~91% lower p95 latency** versus full-context baselines with **~90%+ token savings**. Zep’s temporal-KG line cites **71.2%** LongMemEval and **94.8%** on the DMR benchmark, with sub-200ms retrieval in marketing materials—positioning it as the temporal-fact validity leader rather than the zero-code integration leader. Supermemory and Letta fill adjacent lanes (managed graph API with connectors/multimodal ingest vs. agent-controlled core/archival/recall tools) but lack comparable open LoCoMo leaderboard numbers in the artifacts we validated.

Memorable’s JSON self-assessment (`Memorable.json`) records feature parity on extraction modes, multi-signal fusion (semantic + BM25 + entity + optional graph), and temporal invalidation—but explicitly flags missing **published** LoCoMo/LongMemEval/BEAM scores, user profiles, bundled connectors, and cross-encoder reranking. Priority tags in the corpus align with our roadmap: **P0** close the Mem0 paper loop (full LoCoMo harness, reproduction targets); **P1** harden Zep-style temporal validity in SQL; **P2** optional graph UX and connector plugins. Until full-leaderboard runs complete, Memorable should be evaluated on **deployment shape** (interceptor + portable SQL) and **diagnostic metrics** (evidence recall vs. judge accuracy) rather than headline accuracy claims—a positioning Mem0 and Zep already own on public leaderboards.

---

## References

1. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413, 2025.
2. LoCoMo: Long-term Conversational Memory benchmark. snap-research/LOCOMO, GitHub.
3. Zep / Graphiti temporal knowledge graphs. arXiv:2501.13956 (Zep technical report series).
4. Memori. GibsonAI/Memori — interceptor memory for LLMs.
5. Supermemory. supermemoryai/supermemory — graph-augmented memory.
6. LongMemEval. Long-horizon memory evaluation benchmark, 2024–2025.

---

## Appendix A — Figure index

| Figure | File | Caption |
|--------|------|---------|
| 1 | `figures/output/fig1_locomo_judge_by_category.png` | LoCoMo judge accuracy by category |
| 2 | `figures/output/fig2_retrieval_vs_judge.png` | Evidence recall vs judge accuracy |
| 3 | `figures/output/fig3_overall_accuracy.png` | Overall accuracy comparison |
| 4 | `figures/output/fig4_f1_by_category.png` | Token F1 by category |
| 5 | `figures/output/fig5_capability_heatmap.png` | Competitive capability matrix |
| 6 | `figures/output/fig6_pipeline_schematic.png` | Memorable pipeline |

## Appendix B — Reproducibility

```bash
# Smoke (CI)
make locomo-smoke

# Measured subset
make locomo-subset

# Figures
python ResearchDoc/figures/generate_figures.py
```

---

*Last updated: 2026-05-17 · Auto-sync with `ResearchDoc/STATUS.md`*
