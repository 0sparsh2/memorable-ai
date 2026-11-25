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

## 2026-11-19: Observer pilot and batched observer (commit 232557f)

Retrieval, LoCoMo (1536 questions, 10 conversations, no LLM):

| Signal | recall@5 | vs previous milestone (paired) |
|---|---|---|
| turn/pipeline | 0.701 | unchanged |
| turn/recall | 0.68 | +0.4 pts (CI -0.4…+1.2) |
| episode/fused | 0.575 | unchanged |
| fact/graph | 0.018 | unchanged |
| fact/pipeline | 0.059 | unchanged |

Ingest 50.1 s wall / 172.8 s CPU, query p50 34.4 ms; graph: 756 entities, 113 relations (previous: 3487.6 s, 32.2 ms).

| QA arm | LoCoMo J-strict | LoCoMo J-lenient | LongMemEval J | answered / queued |
|---|---|---|---|---|
| A0 full context | 70.9 [67–75] | 92.4 [90–95] | 53.3 [37–70] | 580 / 1639 |
| A1 BM25 only | 53.0 [49–57] | — | 70.7 [64–77] | 761 / 2040 |
| A2 v0.3 (RRF) | 61.0 [57–65] | — | 72.8 [66–80] | 694 / 2040 |
| A3 current (score fusion + Ettin-17M) | 66.9 [63–71] | 84.1 [81–87] | 74.7 [68–81] | 717 / 2040 |
| A3w current, 40 turns | 69.1 [66–73] | — | 75.3 [69–81] | 758 / 2040 |
| A4 current + fine-tuned MiniLM | 63.9 [47–81] | 80.6 [67–92] | 93.3 [80–100] | 51 / 2040 |
| A5 current + 20 LLM facts | 75.0 [50–100] | 75.0 [50–100] | — | 8 / 500 |
| A6 current, 80 turns | 77.8 [44–100] | — | — | 9 / 500 |
| D5 decoy: another conversation's facts | 0.0 [0–0] | — | — | 1 / 100 |

Published (self-reported, other answer models): best LoCoMo 93.6 (Agent Zero Memory), best LongMemEval 95.6 (Agent Zero Memory); Zep 81.5 / 71.2. Full table: `python -m benchmarks.qa.compare`.

Notes:
- `turn/recall` +0.4 pts (CI −0.4…+1.2): no measurable change from the chat-opener filter and the backfill for episode-named entities (a4e6107, 33ecb73) on LLM-free graphs. The filter removed 151 junk entities (907 → 756). Ingest 50.1 s wall, 172.8 s CPU (embedding is multithreaded); the previous entry's 3,487 s was a swap stall.
- QA rows are interim (the v3 loop is paused while prereg v5 runs). A4, A5, A6 and D5 have fewer than 60 answers each, so their intervals are wide. Reporting only.

**Observation log (prereg v6 observer pilot; dev rows; no answers yet, so these are proxies):**

| Context for LoCoMo conv-48 (191 questions with evidence) | All gold evidence in context / cited | Short answer in one line (111) | Tokens |
|---|---|---|---|
| A3g: top 10 turns + 3 episodes (LLM-free ingest) | 63.4% (all@10, `docs/g2_results.md`) | — | ~0.8k |
| **A7 log, one Ultra call per episode (46 calls)** | **94.8%** | **65.8%** | 10.8k |
| Batched observer, 12k tokens per call (2 calls) | 74.9% | 48.6% | 4.2k |
| Batched observer, 8k / 20k (3 / 2 calls) | 57.6% / 74.9% | 36.0% / 45.9% | 3.6k / 4.1k |

- On 5 LongMemEval-dev haystacks (all single-session-user), the per-episode log is 25–26.5k tokens, the size Mastra reports for its LongMemEval contexts (~30k; Mastra 94.9 with gpt-5-mini). All 5 logs cite the evidence turn and hold the answer.
- **The batched observer loses detail.** Given 8–20k tokens of messages at once, Ultra wrote about a quarter as many observations per message as it does one episode at a time (94–129 against 393), well inside its output budget. Prereg v6.1's rule (within 3 points on evidence cited, 5 on answers contained) picks no batch size, so v6.1 is not run.

## 2026-11-20: v5 answer model chosen (Ultra); v4.2 LME build published (commit 12c974e)

Retrieval, LoCoMo (1536 questions, 10 conversations, no LLM):

| Signal | recall@5 | vs previous milestone (paired) |
|---|---|---|
| turn/pipeline | 0.701 | unchanged |
| turn/recall | 0.68 | unchanged |
| episode/fused | 0.575 | unchanged |
| fact/graph | 0.018 | unchanged |
| fact/pipeline | 0.059 | unchanged |

Ingest 46.9 s wall / 160.0 s CPU, query p50 31.7 ms; graph: 756 entities, 113 relations (previous: 50.1 s, 34.4 ms).

| QA arm | LoCoMo J-strict | LoCoMo J-lenient | LongMemEval J | answered / queued |
|---|---|---|---|---|
| A0 full context | 70.9 [67–75] | 92.4 [90–95] | 53.3 [37–70] | 580 / 1639 |
| A1 BM25 only | 53.0 [49–57] | — | 70.7 [64–77] | 761 / 2040 |
| A2 v0.3 (RRF) | 61.0 [57–65] | — | 72.8 [66–80] | 694 / 2040 |
| A3 current (score fusion + Ettin-17M) | 66.9 [63–71] | 84.1 [81–87] | 74.7 [68–81] | 717 / 2040 |
| A3w current, 40 turns | 69.1 [66–73] | — | 75.3 [69–81] | 758 / 2040 |
| A4 current + fine-tuned MiniLM | 63.9 [47–81] | 80.6 [67–92] | 93.3 [80–100] | 51 / 2040 |
| A5 current + 20 LLM facts | 75.0 [50–100] | 75.0 [50–100] | — | 8 / 599 |
| A6 current, 80 turns | 77.8 [44–100] | — | — | 9 / 599 |
| D5 decoy: another conversation's facts | 0.0 [0–0] | — | — | 1 / 100 |

Published (self-reported, other answer models): best LoCoMo 93.6 (Agent Zero Memory), best LongMemEval 95.6 (Agent Zero Memory); Zep 81.5 / 71.2. Full table: `python -m benchmarks.qa.compare`.

Notes:
- Retrieval is unchanged, as expected: nothing in retrieval changed since the last entry. `MEMORABLE_TORCH_DEVICE` (f9f7689) only pins where PyTorch runs.
- The QA rows above are the paused v3/v4 loop, answered by Lightning. They are unchanged since v5 took the judge quota. The v4.2 LME contexts (A5, A6: 99 held-out haystacks each) are now queued for that loop, behind v6's answers.

**Answer model (prereg v5, final; dev rows: LoCoMo conv-48/49, 347 questions; LongMemEval-dev, 150; Lightning judge):**

| Context | Reader | LoCoMo J-strict | LoCoMo J-lenient | LongMemEval J |
|---|---|---|---|---|
| A3 current memory | Lightning | 62.5 [57–68] | 83.9 [80–88] | 68.0 [60–75] |
| A3 current memory | **Ultra (chosen)** | 71.8 [67–76] | **89.6** [86–93] | 74.7 [67–81] |
| A3 current memory | Mistral Large 4 | 72.3 [67–77] | 86.5 [83–90] | **78.0** [71–85] |
| A0 full context | Ultra | 86.2 [82–90] | **98.3** [97–99] | 56.2 [42–71] (n=48) |
| A0 full context | Mistral Large 4 | 87.9 [84–91] | 95.7 [93–98] | **83.3** [73–94] (n=48) |
| Published best (self-reported) | — | — | 93.6 (Agent Zero) | 95.6 (Agent Zero) |

- Ultra over Lightning on A3: +9.2 LoCoMo J-strict (CI +5.5…+13.0) and +6.7 LME J (CI +1.3…+12.7). Mistral's lead over Ultra is inside the noise on both, so B11 picks the faster reader (`results-v5.md`).
- Against the published best, our memory arm with Ultra is 4.0 points short on LoCoMo J-lenient and 20.9 short on LongMemEval. With the whole conversation, Ultra scores 98.3 on LoCoMo, so the reader no longer limits LoCoMo; memory does. These are dev rows with a Lightning judge, while published systems use GPT-4o-class judges on the full sets, so the comparison is indicative only.
- Ultra handles LongMemEval's full ~115k-token history badly (56.2 against Mistral's 83.3).
- Corrected 2026-11-23: four cells in this table (Ultra A3 LME, Mistral A3 on all three, Ultra A0 J-strict) were 0.1–0.5 off the stored v5 answers. They now match `results-v5.md`'s cell table; no verdict changes.

## 2026-11-23: v6 observation log passes (A7 +11.5 LoCoMo, +6.7 LME) (commit 962b801)

Retrieval, LoCoMo (1536 questions, 10 conversations, no LLM):

| Signal | recall@5 | vs previous milestone (paired) |
|---|---|---|
| turn/pipeline | 0.701 | unchanged |
| turn/recall | 0.68 | unchanged |
| episode/fused | 0.575 | unchanged |
| fact/graph | 0.018 | unchanged |
| fact/pipeline | 0.059 | unchanged |

Ingest 44.4 s wall / 154.0 s CPU, query p50 31.0 ms; graph: 756 entities, 113 relations (previous: 46.9 s, 31.7 ms).

| QA arm | LoCoMo J-strict | LoCoMo J-lenient | LongMemEval J | answered / queued |
|---|---|---|---|---|
| A0 full context | 70.9 [67–75] | 92.4 [90–95] | 53.3 [37–70] | 580 / 1639 |
| A1 BM25 only | 53.0 [49–57] | — | 70.7 [64–77] | 761 / 2040 |
| A2 v0.3 (RRF) | 61.0 [57–65] | — | 72.8 [66–80] | 694 / 2040 |
| A3 current (score fusion + Ettin-17M) | 66.9 [63–71] | 84.1 [81–87] | 74.7 [68–81] | 717 / 2040 |
| A3w current, 40 turns | 69.1 [66–73] | — | 75.3 [69–81] | 758 / 2040 |
| A4 current + fine-tuned MiniLM | 63.9 [47–81] | 80.6 [67–92] | 93.3 [80–100] | 51 / 2040 |
| A5 current + 20 LLM facts | 75.0 [50–100] | 75.0 [50–100] | — | 8 / 599 |
| A6 current, 80 turns | 77.8 [44–100] | — | — | 9 / 599 |
| D5 decoy: another conversation's facts | 0.0 [0–0] | — | — | 1 / 100 |

Published (self-reported, other answer models): best LoCoMo 93.6 (Agent Zero Memory), best LongMemEval 95.6 (Agent Zero Memory); Zep 81.5 / 71.2. Full table: `python -m benchmarks.qa.compare`.

Notes:
- Retrieval is unchanged, as expected: nothing in retrieval changed since the last entry. The v6 results commit is d46de84; the milestone started at 962b801, just before it.
- The QA rows above are the paused v3/v4 loop (answered by Lightning), unchanged since the last entry. That loop resumed at 18:20 EST today, after v6's answers.

**Observation log (prereg v6, final; reader Ultra, Lightning judge; dev rows: LoCoMo conv-48/49, 347 questions; LongMemEval-dev, 150):**

| Context | LoCoMo J-strict | LoCoMo J-lenient | LongMemEval J | Reader prompt, median |
|---|---|---|---|---|
| A3 current memory (v5) | 71.8 [67–76] | 89.6 [86–93] | 74.7 [67–81] | — |
| A3g + graph walk | 71.2 [67–76] | 91.1 [88–94] | 75.3 [68–82] | 1.0k / 3.0k |
| **A7 A3g + observation log (≤30k)** | **82.7** [78–87] | **96.0** [94–98] | **82.0** [76–88] | 12.3k / 26.3k |
| D7 decoy: another conversation's log | 71.2 [66–76] | — | 72.0 [65–79] | 10.9k / 26.3k |
| A0 full context (v5) | 86.2 [82–90] | 98.3 [97–99] | 56.2 [42–71] (n=48) | 22.3k / 108.5k |
| Published best (self-reported) | — | 93.6 (Agent Zero) | 95.6 (Agent Zero) | |

- A7 − A3g: +11.5 LoCoMo J-strict (CI +6.9…+16.1) and +6.7 LME J (CI +0.7…+12.7). B15 passes on both benchmarks. The decoy log gains nothing (B16), and the observer costs 1.18 calls per session (B17). The graph walk alone makes no measurable change (B18). See `results-v6.md`.
- Against the published best: LoCoMo J-lenient is **+2.4 above** (96.0 against 93.6), but only on the two dev conversations, with a Lightning judge, while published systems use GPT-4o-class judges on the full sets. It is a lead, not a claim, until the eight held-out conversations are run with the frozen configuration. LongMemEval is 13.6 short, and preference questions are 15 of A7's 27 misses.

## 2026-11-24: v7 answer prompt P1 (LME-dev 88.7, LoCoMo-dev 97.7 lenient) (commit a7d0256)

Retrieval, LoCoMo (1536 questions, 10 conversations, no LLM):

| Signal | recall@5 | vs previous milestone (paired) |
|---|---|---|
| turn/pipeline | 0.701 | unchanged |
| turn/recall | 0.68 | unchanged |
| episode/fused | 0.575 | unchanged |
| fact/graph | 0.018 | unchanged |
| fact/pipeline | 0.059 | unchanged |

Ingest 43.0 s wall / 151.4 s CPU, query p50 30.8 ms; graph: 756 entities, 113 relations (previous: 44.4 s, 31.0 ms).

| QA arm | LoCoMo J-strict | LoCoMo J-lenient | LongMemEval J | answered / queued |
|---|---|---|---|---|
| A0 full context | 71.1 [67–75] | 92.4 [90–94] | 54.8 [39–71] | 612 / 1639 |
| A1 BM25 only | 52.7 [49–57] | — | 70.9 [65–77] | 816 / 2040 |
| A2 v0.3 (RRF) | 62.0 [58–66] | — | 73.0 [66–79] | 749 / 2040 |
| A3 current (score fusion + Ettin-17M) | 67.0 [63–71] | 84.0 [81–87] | 75.1 [69–81] | 769 / 2040 |
| A3w current, 40 turns | 68.9 [65–73] | — | 74.3 [68–80] | 798 / 2040 |
| A4 current + fine-tuned MiniLM | 62.5 [52–73] | 85.2 [77–92] | 78.1 [62–91] | 120 / 2040 |
| A5 current + 20 LLM facts | 68.6 [51–83] | 82.9 [69–94] | 100.0 [100–100] | 40 / 599 |
| A6 current, 80 turns | 90.9 [77–100] | — | 50.0 [0–100] | 26 / 599 |
| D5 decoy: another conversation's facts | 50.0 [0–100] | — | — | 2 / 100 |
| A3r A3 answered again (drift check) | 100.0 [100–100] | — | 100.0 [100–100] | 3 / 60 |

Published (self-reported, other answer models): best LoCoMo 93.6 (Agent Zero Memory), best LongMemEval 95.6 (Agent Zero Memory); Zep 81.5 / 71.2. Full table: `python -m benchmarks.qa.compare`.

Notes:
- Retrieval is unchanged, as expected: v7 changed only the reader's prompt.
- The QA rows above are the v3/v4 loop (answered by Lightning). It was paused for v7 at 3,932 answers and resumed at 19:23 EST today.

**Answer prompt (prereg v7, final; A7 contexts, reader Ultra with thinking on, Lightning judge; dev rows: LoCoMo conv-48/49, 347 questions; LongMemEval-dev, 150):**

| Prompt | LoCoMo J-strict | LoCoMo J-lenient | LongMemEval J | Declines D8 (guard) |
|---|---|---|---|---|
| P0, previous ("short phrase … say you don't know") | 82.7 [78–87] | 96.0 [94–98] | 82.0 [76–88] | 97.6 [94–100] |
| **P1, targeted (chosen)** | **88.2** [85–91] | **97.7** [96–99] | **88.7** [83–93] | 96.8 [94–99] |
| P2, rules removed (LongMemEval's reader style) | 89.0 [86–92] | 98.9 [97–100] | 86.7 [81–92] | 95.2 [91–98] |
| Published best (self-reported) | — | 93.6 (Agent Zero) | 95.6 (Agent Zero) | |

- P1 − P0: +5.5 LoCoMo J-strict (CI +2.6…+8.9) and +6.7 LME J (CI +1.3…+12.7). LME preference questions go from 40 to 80. B21 picks P1 over P2 (P2 − P1 on LME is −2.0, CI −6.7…+2.7). See `results-v7.md`.
- Against the published best: LoCoMo J-lenient is +4.1 above on dev; LongMemEval is 6.9 short (17 misses of 150: preference 5, temporal 4, multi-session 3, knowledge-update 3, assistant 2). These are dev rows with a Lightning judge, so the comparison is indicative only.
- Correction: earlier entries and preregs called the reader "thinking off". Readers have always answered with thinking on (Ultra: a median of 465 output tokens for an 83-token answer); only the observer runs with thinking off.
