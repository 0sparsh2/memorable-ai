# Preregistration v4.1: LLM facts and a wider window, end-to-end QA

**Version:** v4.1, 2026-11-18. Supersedes [prereg-v4.md](prereg-v4.md), which froze when pilot attempt 1 started. It freezes when pilot attempt 2 starts. Everything in v4 holds (bars B7–B10, arms A5 / A6 / D5, questions, baselines, decoys, held-out exposure, blinding, diagnostics) except the changes below.

**Changes from v4**, from pilot attempt 1 ([pilot-notes-v4.md](pilot-notes-v4.md)):
1. **Extraction keeps facts from long sessions** (a product fix, commit 115022d). Attempt 1 lost every fact of 9 of 38 sessions: the extraction reply was capped at 1,200 tokens, was cut off mid-item, and the JSON fallback then failed on the whole reply. The budget now grows with the session (800 + 150 tokens per message, at most 4,096), and a cut-off array keeps its complete items. A5 is defined as Memorable's pipeline with the LLM on, so it uses the fixed pipeline.
2. **The harness counts JSON outcomes** per conversation: replies that parse, replies repaired by keeping complete items, and replies that fail.
3. **Stop rule, sharpened.** v4 stopped on "more than 5% of extraction calls failing after retries", meaning transport failures. JSON that fails after repair now counts toward the same 5%. Repaired replies are reported; if more than 20% of extraction replies need repair, the budget is too small and the run stops.

`PREREG: ok` (frozen at pilot attempt 2)
