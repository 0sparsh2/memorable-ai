# Diagnosis: What Went Wrong on LOCOMO (Multi-Angle Analysis)

**Date:** 2026-05-17  
**Evidence:** `benchmarks/locomo/data/measured_subset.json` — 10% judge, 83% temporal evidence recall  
**Question:** Why is retrieval “working” while end-to-end accuracy collapses?

---

## 1. Executive summary (one paragraph)

The benchmark is **not failing primarily at retrieval**. It is failing at **(a) representation**, **(b) temporal reasoning**, and **(c) the answer/judge stack**. Memorable ingests **raw dialogue turns** into memory, while Mem0’s published LoCoMo numbers assume **LLM-extracted atomic facts** plus a **conversation summary** and tuned QA. Our first 10 questions are **60% temporal** (absolute dates in gold, relative language in dialogue). Retrieved chunks contain the *event* but often not the *calendar date* the judge expects—hence high **evidence recall** and near-zero **judge accuracy** on temporal items.

---

## 2. Mathematical decomposition

Treat end-to-end correctness as a product of stages:

\[
P(\text{correct}) \approx P(E \mid Q) \cdot P(A \mid E, Q) \cdot P(J \mid A, G)
\]

| Symbol | Meaning | Our subset (temporal) |
|--------|---------|------------------------|
| \(P(E \mid Q)\) | Evidence recall — gold `dia_id` in top-k | **~0.83** |
| \(P(A \mid E, Q)\) | Answer model outputs gold-equivalent text | **~0** (F1 ~5%) |
| \(P(J \mid A, G)\) | LLM judge accepts vs gold | **~0** |

**Implication:** Optimizing retrieval alone cannot lift judge accuracy from 10% to 70%+; you must raise \(P(A \mid E, Q)\) and/or change the metric (Snap **F1** on short answers is still low at 12.9% overall → generation is weak even before judge noise).

**Information-theoretic view:** Gold answer \(G\) is a **compressed sufficient statistic** (e.g. `"7 May 2023"`). Retrieved context \(E\) is **high-entropy dialogue** (`"yesterday"`, `"last Sunday"`). Mutual information \(I(G; E)\) is positive (event present) but \(H(G \mid E)\) remains large unless **session timestamps** or **normalized facts** are in \(E\).

**Bayesian sketch:** Let gold date \(d^\*\) and dialogue relative phrase \(r\). With session anchor \(t_{\text{session}}\), \(d^\* = f(r, t_{\text{session}})\). If \(t_{\text{session}} \notin\) context, the answer model must integrate over missing \(t\) → high error rate on temporal category.

---

## 3. Root causes (ranked by impact)

### R1 — Benchmark ingest bypasses Mem0-style extraction (architectural mismatch)

**What we do:** `LOCOMOBenchmark._ingest_full_conversation` calls `add_memory("Speaker: text")` or `temporal.add_temporal_memory` per turn.

**What Mem0 does (paper):** LLM extracts **atomic facts** from \((m{=}10\) recent msgs, summary \(S\), new pair\); **update** phase merges into KB.

**Effect:** Retrieval returns **episodic** snippets, not **semantic** facts. LoCoMo gold is overwhelmingly **semantic** (dates, entities, relations).

**Severity:** **Critical** for fair comparison to Mem0 91.6%.

### R2 — Session date not shown to the answer model (representation bug)

`session_date` is stored in `metadata` (`session_date`, `session_key`) but `format_memory_context()` only prints `content`. Temporal gold requires resolving:

| Gold | Typical dialogue |
|------|------------------|
| `7 May 2023` | “yesterday” (session 1 = 8 May 2023) |
| `The week before 9 June 2023` | “last week” |
| `June 2023` | “next month” |

**Severity:** **Critical** for category 2 (6/10 questions in subset).

### R3 — Relative vs absolute time (task–store mismatch)

LoCoMo temporal QA is **calendar grounding**. Dialogue is **deictic** (“yesterday”, “Sunday before 25 May”). Without explicit normalization at ingest, **no retrieval method** fixes this at query time unless the answer LLM performs **coreference + date arithmetic**—hard in a 128-token answer call.

**Severity:** **Critical** for temporal; aligns with Zep’s bi-temporal fact layer.

### R4 — Empty or weak answers under NIM latency (computational)

Logs show **APITimeoutError** on answer/judge (120–300s client vs 270s+ server). Failed answers → `prediction=""` → judge=False. This **deflates** scores independent of memory quality.

**Severity:** **High** (operational); partially mitigated by context caps + longer timeouts.

### R5 — Truncation drops evidence (information bottleneck)

Caps: ~4500 chars, 10 memories, 380 chars/line. Long turns truncated with `...` may remove the only date mention. Retry schedule shrinks context further on timeout.

**Severity:** **Medium** (amplifies R2–R3).

### R6 — Judge variance and strict binary rubric (measurement)

Mem0 leaderboard uses **LLM-as-judge**; Snap official uses **token F1**. Binary judge penalizes correct-but-differently-phrased answers (“May 7, 2023” vs “7 May 2023”). Our F1 is also low (12.9%) → not **only** judge noise.

**Severity:** **Medium** for iteration; use F1 + recall for debugging.

### R7 — No cross-encoder reranker (retrieval polish)

Mem0 2026 fuses semantic + BM25 + entity; leaders often add **reranking**. We fuse but don’t rerank top-50 → 12. Evidence recall 83% suggests top-12 often enough for *dialogue id* but not for *answerable text*.

**Severity:** **Low–medium** given high recall.

### R8 — Subset composition (statistical)

First 10 QA: **6 temporal, 3 multi-hop, 1 open-domain, 0 single-hop**. Overall 10% is dominated by hardest category with known R2–R3 issues. Not representative of full 1540-Q distribution.

**Severity:** **High** for interpreting headline number.

---

## 4. Biological / cognitive analogy

| Brain system | Memorable today | LoCoMo needs | Mem0 / Zep |
|--------------|-----------------|--------------|------------|
| **Hippocampus** (episodic trace) | Raw turns in SQL | Dialogue evidence | Episodes |
| **Neocortex** (semantic schema) | Weak / skipped in benchmark | “Caroline went to group on 7 May” | LLM-extracted facts |
| **Time cells / context** | `valid_from` on ingest, not in QA prompt | Absolute dating | Zep bi-temporal facts |
| **Consolidation sleep** | Consolidator exists, not in LOCOMO path | Roll episodic → semantic | Summary + update ops |

**Consolidation failure:** We **store episodes** but **test semantic recall**—like examining someone on factual quizzes using only raw diary entries without rereading dates.

---

## 5. Architectural comparison (methods)

```
                    INGEST                    STORE                    QUERY
─────────────────────────────────────────────────────────────────────────────
Full-context GPT    [all turns]  ──────────►  context window  ──►  LLM(Q, all)
RAG (LoCoMo official) chunks/embed ───────►  vector index   ──►  retrieve + LLM
Mem0 paper          LLM extract + update ──►  vector KB      ──►  multi-signal + LLM
Memorable (bench)   raw turn add      ──►  SQL + embed    ──►  hybrid + LLM  ◄── mismatch
Memorable (prod)    interceptor extract ►  SQL + optional graph (not measured)
Zep / Graphiti      episode → facts  ──►  temporal KG    ──►  valid facts @ t
```

**Fair Memorable benchmark** should either:

- **Path A:** Ingest via `MemoryExtractor` + `MemoryUpdater` per session batch (Mem0-fair), or  
- **Path B:** Keep raw turns but evaluate with **RAG-style** metrics only (recall + reading comprehension on full retrieved turns), or  
- **Path C:** Inject `session_date` + normalize times in context (minimal fix for temporal).

---

## 6. Computational / IR perspective

**Retrieval objective mismatch:**

- We optimize **similarity** to question embedding.
- LoCoMo evidence metric optimizes **dia_id overlap** (dialogue retrieval).
- Judge optimizes **semantic equivalence** to short gold string.

These are **three different loss functions**. High recall on dia_ids does not imply high NDCG on answerable spans.

**Recommended IR stack (industry pattern):**

1. Retrieve \(k{=}50\) (wide)  
2. **Cross-encoder rerank** to \(k{=}10\)  
3. **Compress** with extractive summary biased to question  
4. **Answer** with date-aware system prompt  

Mem0 effectively folds (1–3) into extraction at write time.

---

## 7. What is working for others (evidence-based)

| System | Mechanism | Why temporal LoCoMo works better |
|--------|-----------|--------------------------------|
| **Mem0 2026** | ADD-only hierarchical extract; +29.6 LoCoMo pts claimed; multi-signal | Facts stored with resolved entities; summary carries discourse |
| **Mem0 paper** | \(m{=}10\), \(s{=}10\) update; graph variant | Explicit extraction prompt; tool-call UPDATE |
| **Zep** | Bi-temporal facts; invalidate stale | Query **valid facts at time t**; dates are first-class |
| **LoCoMo RAG baselines** | Chunk by turn; often full session date in chunk header | Official scripts pass `session_*_date_time` into context |
| **Full-context** | No retrieval error | Upper bound; unusable at scale |

---

## 8. Recursive hypothesis tree

```
Low judge accuracy (10%)
├── Retrieval failure? ──NO (temporal recall 83%)
├── Answer failure?
│   ├── Empty (timeout) ──YES (some runs)
│   ├── Wrong content retrieved ──PARTIAL
│   └── Right content, wrong reasoning ──YES (temporal)
│       ├── No session_date in prompt ──YES
│       ├── Relative language in text ──YES
│       └── No extracted fact "X on DATE" ──YES
├── Representation failure?
│   └── Episodic store vs semantic QA ──YES
├── Metric failure?
│   ├── Judge too strict ──PARTIAL
│   └── Subset skewed to temporal ──YES
└── Comparison unfair?
    └── Mem0 uses extract; we use raw turns ──YES
```

---

## 9. What to do differently (prioritized)

| Priority | Change | Expected effect | Effort |
|----------|--------|-----------------|--------|
| **P0** | **Mem0-fair ingest:** batch sessions through `extract` + `update` before QA | Large judge lift | High (LLM cost) |
| **P0** | **Context format:** `[D1:3] [session: 8 May 2023] Speaker: ...` in `format_memory_context` | Fix temporal without full re-ingest | Low |
| **P0** | **Temporal normalization pass** at ingest: LLM or rule `relative → absolute` using `session_date` | Zep-like facts in SQL | Medium |
| **P1** | **Answer prompt:** “Resolve relative dates using session timestamps” | Medium temporal gain | Low |
| **P1** | **Metric dashboard:** report recall, F1, judge separately; save `rows` always | Faster debugging | Low |
| **P1** | **Cross-encoder rerank** (e.g. ms-marco-MiniLM) on top-50 | +multi-hop | Medium |
| **P2** | **Full-context upper bound** on same 10 Q | Separates retrieval vs reasoning | Medium |
| **P2** | **Rebalanced subset** (stratified by category) | Honest headlines | Low |
| **P2** | **Gemini/local model** for QA to avoid NIM 270s tails | Fewer empty answers | Low |

---

## 10. Proposed experiments (validate hypotheses)

1. **Ablation A:** Raw turns vs extracted facts (same retrieval) → isolates R1.  
2. **Ablation B:** Add `session_date` to context only → isolates R2.  
3. **Ablation C:** Oracle context = gold evidence turns only → upper bound on \(P(A \mid E)\).  
4. **Ablation D:** `--no-llm` evidence recall on full 1540 Q → retrieval-only leaderboard.  
5. **Ablation E:** Snap F1 without judge → reduces \(P(J \mid \cdot)\) noise.

**Success criteria (subset):** Temporal judge > 40% with session dates in context; > 60% with Mem0-fair extraction.

---

## 11. Comparison to our stated architecture

Production Memorable **has** extraction, update, summary, FTS5, temporal columns—but the **LOCOMO harness does not exercise them**. The gap is **evaluation design**, not necessarily product incapability.

**Action:** Add `ingest_mode: raw | extracted | hybrid` to `LOCOMOBenchmark` and report both in `ResearchDoc/STATUS.md`.

---

## 12. Synthesis (one sentence)

We built a **semantic exam** on an **episodic notebook**, graded with a **strict LLM teacher**, under **API time pressure**—and measured against competitors who **rewrite the notebook into facts first**.

---

## 13. Experiment results (2026-05-17)

| Experiment ID | Ingest | Context | Judge acc | Notes |
|---------------|--------|---------|-----------|-------|
| `retrieval_only_dated` | raw | dated | **80%** | No answer LLM — metric = evidence recall / hit |
| `dated_context` | raw | dated | (running) | Session dates in QA prompt |
| `extracted` | extracted | dated | (running) | Per-session LLM facts |
| `oracle` | oracle | dated | (running) | Gold evidence turns only |

**Insight:** 80% retrieval-only proves the **store + retriever** are viable; gains must come from **answer generation** and **semantic ingest**.

*Log: `ResearchDoc/experiments/experiment_log.md` · JSON: `ResearchDoc/experiments/*.json`*
