# Pilot notes, prereg v6 (the observation log)

## Observer pilot (started 2026-11-19)

Prereg v6 was frozen and committed before this pilot wrote anything. The observer half of the pilot runs now, alongside v5: v5 asks Ultra for about one answer a minute, so Ultra's quota is mostly idle. The 3 answers per cell follow once v5 has named the reader.

- **Groups:** conv-48 (LoCoMo-dev) and the first 5 LME-dev haystacks in file order.
- **Command:** `e2e_qa retrieve --arms A7` into `runs/v6`, LLM-free base ingest as for A3g (`MEMORABLE_EMBED_BACKEND=torch` on LME, as in the A3g build), observer on Ultra with thinking off, `--llm-rpm 6`.
- **Kept for the full run:** the observer ingests (`runs/v6/ingest_cache/*/*.obs.db`) and these groups' A7 contexts.

### conv-48 (LoCoMo-dev): passed

| Check | Result | Bar |
|---|---|---|
| Observer calls | 46 (one per episode; 30 sessions = 1.53 per session), 0 failed after retries | ≤ 2.5 per session; ≤ 5% failing |
| Replies | 0 empty, 0 with thinking text, 0 unparseable | none with thinking; ≤ 2% unparseable |
| Episodes with an observation | 46 of 46 (100%) | ≥ 80% |
| Hand check, 20 random observations (seed 20261119) | 0 misstate their source turns; 1 date more than a day off | ≤ 2 and ≤ 2 |

- **Cost:** 44.5k prompt and 28.5k output tokens; 521 s of ingest at 6 requests/min.
- **The log:** 393 observations (125 high, 189 medium, 79 low; 322 with an event date), about 10.8k tokens, so A7's 30k budget cuts nothing here.
- **The date that was off:** on Tue 2023-09-12 Jolene says "last weekend my partner and I traveled to a meditation retreat for a few weeks"; the observer dated the retreat 2023-08-29 instead of about 09-09.
- **Detail, without answers** (`benchmarks/qa/observer_batch.py`, 191 questions with evidence): every gold evidence turn is cited by some observation for 94.8% of questions (A3g's top 10 turns hold all of it for about 63%), and one observation holds every content word of the short answer for 65.8% of the 111 non-temporal questions with a 1–5 word answer. Reported only; v6's bars are on answers.

### 5 LME-dev haystacks: passed

| Haystack | Sessions | Calls | Retries (429) | Observations | Log tokens | Ingest |
|---|---|---|---|---|---|---|
| 1e043500 | 50 | 55 | 0 | 519 | 26,345 | 1,211 s |
| 7527f7e2 | 45 | 54 | 0 | 506 | 25,437 | 1,121 s |
| c960da58 | 46 | 56 | 14 | 553 | 26,451 | 1,455 s |
| 3b6f954b | 50 | 57 | 21 | 553 | 25,210 | 2,155 s |
| 726462e0 | 48 | 56 | 7 | 510 | 25,334 | 1,178 s |

- **Bars:** 278 calls, 1.16 per session (≤ 2.5); 0 failed after retries, 0 empty, 0 unparseable, 0 with thinking text. Every episode yielded observations.
- **The log fits, barely:** 25–26.5k tokens per haystack against A7's 30k budget, so the budget cuts nothing on these five. Other haystacks may run over; the report counts cuts.
- **Detail, without answers:** all 5 logs cite their evidence turn and hold the answer in one observation. All five are single-session-user questions (the first five in file order), so this says nothing yet about multi-session or temporal questions.
- **Rate:** 278 calls in 3,827 s, 4.4 calls/min with 2 haystacks in parallel and v5 asking Ultra for about one answer a minute. The 42 retries were 429s.

### Projection

`PILOT_PROJECTION: per_item=55.6 calls/haystack (sd 1.1, n=5) total_items=145 LME haystacks + conv-49 (48 calls) projected=31 h (8,110 calls at 4.4/min) band95=22.5..38.6 h (6.0..3.5 calls/min) cap=72 h verdict=within`

- Timed: the five pilot haystacks above (wall clock, `retrieve --arms A7 --parallel 2 --llm-rpm 6`). The limit is NIM's per-model request rate on Ultra; the local ingest is about 4 s of CPU per session.
- Answers (after v5 names the reader): A3g, A7 and D7 on 497 questions each plus the 60-question drift check, 1,551 answers and about 2,250 judge calls on Lightning. At the judge's 4–6 calls/min, 7–10 h.
- D7 needs every log, so it is built in a second pass once the A7 pass has finished.

## Approval and full run

On 2026-11-19 the owner approved the full run as preregistered, and chose the judge order: v6's answers before the paused v3/v4 loop resumes. The observer ingest started at 15:20 EST (`runs/v6/full_build.log`, pid in `full_build.pid`): an A7 pass over LoCoMo-dev and the 150 LME-dev haystacks (the pilot's six logs are reused from the cache), then a D7 pass, which needs every log. Same settings as the pilot: Ultra, thinking off, 6 requests/min offered, 2 haystacks in parallel.

Prereg v6.1 (the batched observer) is not run: its rule picked no batch size (`prereg-v6.1.md`).

## Stops and resume, 2026-11-20

**First stop, 02:13 EST.** The build stopped on its first non-429 error, as the stop rules require: `sqlite3.OperationalError: unable to open database file` while it wrote haystack `afdc33df` (LME A7, 43rd of 150). The disk was full: macOS swap had grown to 29 GB on the same volume. Nothing written before the stop was damaged: the v5 answers and both v6 context files parse line by line.

**Cause: a memory leak in the build processes, not the observer.** A resume at 02:27 grew to 9.4 GB in five minutes. The v4.2 LME build, running beside it, had reached 18 GB. Both embed with PyTorch, and sentence-transformers places PyTorch on Apple's MPS by default. Measured on LME-dev haystacks with the harness's own ingest:

| Device | After 1 haystack | Each further haystack | Time per haystack |
|---|---|---|---|
| MPS | 4.5 GB | about +0.4 GB | 13–20 s |
| MPS with `gc.collect()` and `torch.mps.empty_cache()` each haystack | 4.2 GB | still grows (4.8 GB after 4) | 13–19 s |
| CPU | 0.6 GB | flat (0.59 GB after 4) | 17–19 s |

Both builds were stopped by hand at 02:35 (v6) and 02:45 (v4.2). Neither lost finished work: every finished log has its `.done` marker, the output files parse, and haystacks in flight are rebuilt from scratch.

**Fix: `MEMORABLE_TORCH_DEVICE=cpu`.** It pins the embedder and the reranker to CPU. Before resuming, A3g was rebuilt on CPU for both LoCoMo-dev conversations and 20 LME-dev haystacks. All 367 contexts are byte-identical to the MPS-built ones in `contexts_*.jsonl`. No prompt, model, budget or context changed.

**Correction, 17:40 EST: identical almost always, not always.** The v4.2 build, also resumed on CPU, failed its own identity check on 1 of 26 held-out haystacks built on CPU. One extractive episode-summary line differed. Rebuilt on MPS, that haystack reproduces the stored A3 exactly; on CPU it does not. So the device can flip a near-tie in sentence choice. CPU and MPS agree on 53 of 53 dev haystacks so far (the 20 above and 33 A7 haystacks built on CPU, each equal to A3g plus its log). `v6_build` now refuses to queue an A7 or D7 row whose excerpts differ from A3g's context for the same question. Any such row is rebuilt on MPS from the cached log before answering. When repairing D7, do not pass `--groups`: the harness then deranges logs within those groups only, and one group would get its own log.

**Rate so far:** 37 new LME haystacks in 10.5 h, about 3.2 observer calls/min. That is below the pilot's 4.4/min because v5's Ultra reader draws on the same per-model limit. At that rate the remaining 108 take about 31 h, so the build totals about 43 h. That is above the pilot's band (22.5–38.6 h) and within the 72 h cap.

## Answer pilot (reader Ultra), 2026-11-20

The reader was filled in from v5 at 18:50 EST (`RESULTS: ok (reader = ultra)`), before any v6 question was answered. Then 3 answers per built cell, 18 in all (`runs/v6ans_pilot`): A3g, A7 and A3r on LoCoMo-dev and LME-dev. D7 is not built yet; it runs the same path as A7.
- **Answers:** all 18 answered by Ultra, none fell back to another model, and every judge verdict parsed (0 unparsed). One stop on a 429 was retried; no other errors.
- **Context sizes:** A7 contexts are 12.4k tokens on LoCoMo and 28–32k on LME (log budget 30k plus the excerpts). A3g contexts are 0.9–3.4k.
- **Rate:** 18 answers in 17 min, while Ultra also carries the observer build.

`PILOT_PROJECTION (answers): 557 now (A3g 497 + A3r 60) and 994 after the build (A7, D7), at 1–2 answers/min with Ultra shared by the observer: 5–9 h now, 8–16 h after the build; cap 4 days; verdict=within.`

The full answer run is part of the owner's approval of 2026-11-19. Stage 1 (A3g, A3r) started at 19:12 EST in `runs/v6ans`. A7 and D7 are appended once the build ends and `v6_build` finds every A7 and D7 row equal to A3g plus its log.

**Stage 1 done, 2026-11-21 at 02:00 EST.** All 557 rows answered (A3g 497, A3r 60) in 6.8 h, about 1.4 answers/min with Ultra shared by the observer build. All were answered by Ultra; none fell back, and 0 verdicts were unparsed. The drift check passes: A3r agrees with v5's A3 verdicts on 98% of 40 LoCoMo and 95% of 20 LME questions (bar 90%), so v5's A3 and A0 answers stand in for this run. B18 and the other bars are scored only by `v6_report --final`, after A7 and D7. When stage 1 ended, the build had finished 104 of 150 LME A7 logs; D7 is still to come.
