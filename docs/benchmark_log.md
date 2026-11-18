# Benchmark log

One entry per milestone (`python -m benchmarks.milestone --name ...`). Retrieval numbers are LLM-free LoCoMo recall@5 with paired CIs against the previous entry; QA numbers come from the end-to-end eval (interim until it completes). Published systems report their own protocols; see `benchmarks/qa/published_results.json`.

## 2026-11-18: P1 tool events and tasks (commit f4c7719)

Retrieval, LoCoMo (1,536 questions, 10 conversations, no LLM):

| Signal | recall@5 | vs previous milestone (paired) |
|---|---|---|
| turn/pipeline | 0.701 | first entry |
| episode/fused | 0.575 | first entry |
| fact/graph | 0.014 | first entry |
| fact/pipeline | 0.057 | first entry |

Ingest 43.6 s, query p50 10.4 ms; graph: 940 entities, 113 relations.

## 2026-11-18: G1 ontology and entity resolution (commit 3e869cf)

| Signal | recall@5 | vs previous milestone (paired) |
|---|---|---|
| turn/pipeline | 0.701 | unchanged |
| episode/fused | 0.575 | unchanged |
| fact/graph | 0.018 | +0.4 pts (CI +0.1…+0.7) |
| fact/pipeline | 0.059 | +0.2 pts (CI 0.0…+0.5) |

Ingest 44.6 s, query p50 11.3 ms; graph: 907 entities, 113 relations (previous: 43.6 s, 10.4 ms).

## 2026-11-18: Extraction fixes, user name, A4 and v4.2 arms launched (commit 71627d3)

Retrieval, LoCoMo (1536 questions, 10 conversations, no LLM):

| Signal | recall@5 | vs previous milestone (paired) |
|---|---|---|
| turn/pipeline | 0.701 | unchanged |
| episode/fused | 0.575 | unchanged |
| fact/graph | 0.018 | unchanged |
| fact/pipeline | 0.059 | unchanged |

Ingest 48.6 s, query p50 11.2 ms; graph: 907 entities, 113 relations (previous: 44.6 s, 11.3 ms).

| QA arm | LoCoMo J-strict | LoCoMo J-lenient | LongMemEval J | answered / queued |
|---|---|---|---|---|
| A0 full context | 67.2 [62–72] | 91.0 [88–94] | 55.0 [35–75] | 377 / 1639 |
| A1 BM25 only | 52.0 [47–57] | — | 68.4 [60–76] | 459 / 2040 |
| A2 v0.3 (RRF) | 62.4 [57–68] | — | 70.3 [61–79] | 407 / 2040 |
| A3 current (score fusion + Ettin-17M) | 68.6 [64–74] | 86.5 [82–90] | 74.8 [67–83] | 436 / 2040 |
| A3w current, 40 turns | 70.9 [66–76] | — | 72.9 [64–81] | 468 / 2040 |

Published (self-reported, other answer models): best LoCoMo 93.6 (Agent Zero Memory), best LongMemEval 95.6 (Agent Zero Memory); Zep 81.5 / 71.2. Full table: `python -m benchmarks.qa.compare`.
