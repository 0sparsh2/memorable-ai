# Agent Memory Landscape 2026 — Research Report

**Topic:** Competitive analysis for Memorable (memorable-ai)  
**Methodology:** Deep-research skill (RhinoInsight-inspired), 14 validated JSON artifacts, web + paper sources (2025–2026)  
**Date:** May 2026

## Table of Contents

- [Executive Summary](#executive-summary)
- [Gap Matrix](#gap-matrix)
- [Per-System Findings](#per-system-findings)
- [Industry Open Problems](#industry-open-problems)
- [Paper-Backed Technique Catalog](#paper-backed-technique-catalog)
- [Memorable Thesis and Priorities](#memorable-thesis-and-priorities)
- [Implementation Roadmap](#implementation-roadmap)

## Executive Summary

Memorable’s **unique position** is zero-code interceptor integration plus SQL-first storage with optional graph — a combination no single competitor ships today. However, the **alpha codebase lags marketing**: pattern-only extraction, no Mem0-style ADD/UPDATE/DELETE pipeline, no BM25/entity retrieval fusion, no published benchmarks, and no bi-temporal validity schema.

**Primary recommendation:** Extend Memorable in-place with Mem0-paper extraction/update (P0), Zep-inspired temporal validity (P1), and measured LOCOMO/LongMemEval gates (P0) — not a pivot to hosted Mem0/Supermemory.

## Gap Matrix

| Capability | Memorable (actual) | Mem0 | Supermemory | Zep | Memori |
|------------|-------------------|------|-------------|-----|--------|
| Zero-code interceptor | Yes | No | No | No | Yes |
| SQL-first storage | Yes | Partial | No | Partial | Yes |
| LLM extraction + update ops | No (TODO) | Yes | Yes | Yes | Partial |
| Multi-signal retrieval | Partial | Yes (2026) | Yes | Yes | Partial |
| Temporal validity (valid_to) | No | Partial | Partial | Yes | No |
| Production graph DB | No (NetworkX) | Neo4j (Mem0^g) | Proprietary | Yes | No |
| Published LoCoMo scores | No | 91.6 (2026) | [uncertain] | Strong temporal | No |
| Multimodal ingest | No | Limited | Yes | Limited | No |
| Connectors (Notion/Slack) | No | Platform | Yes | Enterprise | No |
| Framework integrations | Stubs | 21 | Many | Several | Few |

## Per-System Findings

### Memorable (self-audit)

See [results/Memorable.json](results/Memorable.json). Implementation truth: interceptor works; research pipeline is aspirational.

### Mem0 — P0 port target

See [results/Mem0.json](results/Mem0.json). Port extraction+update, multi-signal retrieval, conversation summary, benchmark harness.

### Supermemory — P2 parity

See [results/Supermemory.json](results/Supermemory.json). Port graph merge/contradiction patterns and connector plugin SDK; skip proprietary graph engine.

### Zep — P1 differentiator

See [results/Zep_Graphiti.json](results/Zep_Graphiti.json). Port bi-temporal validity columns and invalidation semantics into SQL.

### Memori, Letta, LangMem

Parity/stubs: Memori interceptor hardening (P2); Letta adapter optional (P3); LangChain callback (P2).

### Benchmarks and papers

- LoCoMo / LongMemEval / BEAM: measurement gates — [results/LoCoMo_benchmark.json](results/LoCoMo_benchmark.json)
- Mem0 paper: [results/Mem0_paper.json](results/Mem0_paper.json)
- Zep paper: [results/Zep_temporal_KG_paper.json](results/Zep_temporal_KG_paper.json)

## Industry Open Problems

| Problem | Who states it | Memorable design response |
|---------|---------------|---------------------------|
| Cross-session identity | Mem0 2026 blog | Namespace + session alias table + confidence-linked identities |
| Memory staleness | Mem0 2026 blog | valid_to, invalidated_at, superseded_by_id on memories |
| Temporal abstraction at scale | Mem0, BEAM | Episodic→semantic rollup in consolidation job |

## Paper-Backed Technique Catalog

| Technique | Source | Port? | Memorable module |
|-----------|--------|-------|------------------|
| LLM extract + summary window | Mem0 arXiv:2504.19413 | Yes | extraction.py, summary.py |
| ADD/UPDATE/DELETE/NOOP update | Mem0 paper | Yes | update.py |
| Multi-signal retrieval fusion | Mem0 2026 | Yes | retrieval.py, bm25.py |
| Graph entity+relation extract | Mem0^g | Yes | graph/builder.py |
| Bi-temporal fact validity | Zep arXiv:2501.13956 | Yes | storage.py, temporal.py |
| OS memory tiers | Letta/MemGPT | Optional adapter | integrations/letta_adapter (future) |

**Skip:** Full managed SaaS clone; SM token billing; forcing Neo4j without SQL fallback.

## Memorable Thesis and Priorities

**Thesis:** *Interceptor + SQL + research pipeline + optional temporal KG*

| Priority | Work |
|----------|------|
| P0 | LLM pipeline, LOCOMO harness, tests, honest benchmarks docs |
| P1 | Temporal validity, multi-signal retrieval, Neo4j graph path |
| P2 | LangChain, LongMemEval, connector plugins |
| P3 | Letta adapter, BEAM 1M, multimodal extras |

## Implementation Roadmap

This report drives updates to [ROADMAP.md](../ROADMAP.md) and implementation in memorable_ai/ per plan phases C0–C5. Validated research JSON: `agent_memory_landscape_2026/results/` (14/14 PASS).
