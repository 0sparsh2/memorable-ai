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
