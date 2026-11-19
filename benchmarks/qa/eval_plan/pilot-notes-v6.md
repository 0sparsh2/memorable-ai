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
