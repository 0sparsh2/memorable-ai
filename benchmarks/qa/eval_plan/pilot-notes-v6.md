# Pilot notes, prereg v6 (the observation log)

## Observer pilot (started 2026-11-19)

Prereg v6 was frozen and committed before this pilot wrote anything. The observer half of the pilot runs now, alongside v5: v5 asks Ultra for about one answer a minute, so Ultra's quota is mostly idle. The 3 answers per cell follow once v5 has named the reader.

- **Groups:** conv-48 (LoCoMo-dev) and the first 5 LME-dev haystacks in file order.
- **Command:** `e2e_qa retrieve --arms A7` into `runs/v6`, LLM-free base ingest as for A3g (`MEMORABLE_EMBED_BACKEND=torch` on LME, as in the A3g build), observer on Ultra with thinking off, `--llm-rpm 6`.
- **Kept for the full run:** the observer ingests (`runs/v6/ingest_cache/*/*.obs.db`) and these groups' A7 contexts.
