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

## 2026-11-19: G2 part 1: graph walk ranks turns (commit 8161d7d)

Retrieval, LoCoMo (1536 questions, 10 conversations, no LLM):

| Signal | recall@5 | vs previous milestone (paired) |
|---|---|---|
| turn/pipeline | 0.701 | unchanged |
| turn/recall | 0.676 | first entry |
| episode/fused | 0.575 | unchanged |
| fact/graph | 0.018 | unchanged |
| fact/pipeline | 0.059 | unchanged |

Ingest 3487.6 s, query p50 32.2 ms; graph: 907 entities, 113 relations (previous: 48.6 s, 11.2 ms).

Note: the 3,487 s ingest is the machine, not G2. During this run swap was 21.8 of 22.5 GB and the load average was about 10. An identical ingest measured 14.8 s CPU without the G2 mention linking and 13.4 s with it (LoCoMo conversation 1, 19 sessions, 1,080 turn-mention edges), at 925 s and 1,026 s of wall time. The query p50 includes the new `turn/recall` signal (a second retrieval per question). From the next entry, the log also records ingest CPU time. Without an LLM there are only 113 relations, so `turn/recall` measures the walk on a near-empty graph. G2's measured effect is on the LLM-built graphs: held-out all-evidence@10 +1.3 pts (CI +0.4…+2.2), `docs/g2_results.md`.

| QA arm | LoCoMo J-strict | LoCoMo J-lenient | LongMemEval J | answered / queued |
|---|---|---|---|---|
| A0 full context | 70.5 [67–74] | 92.3 [90–95] | 50.0 [32–68] | 560 / 1639 |
| A1 BM25 only | 53.7 [50–58] | — | 70.8 [64–78] | 727 / 2040 |
| A2 v0.3 (RRF) | 60.2 [56–64] | — | 72.8 [66–80] | 661 / 2040 |
| A3 current (score fusion + Ettin-17M) | 67.7 [64–72] | 84.7 [82–88] | 74.6 [68–81] | 688 / 2040 |
| A3w current, 40 turns | 68.7 [65–72] | — | 74.3 [67–81] | 723 / 2040 |

Published (self-reported, other answer models): best LoCoMo 93.6 (Agent Zero Memory), best LongMemEval 95.6 (Agent Zero Memory); Zep 81.5 / 71.2. Full table: `python -m benchmarks.qa.compare`.
