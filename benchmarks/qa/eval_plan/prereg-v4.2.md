# Preregistration v4.2: LLM facts and a wider window, end-to-end QA

**Version:** v4.2, 2026-11-18. Supersedes [prereg-v4.1.md](prereg-v4.1.md), which froze when pilot attempt 2 started. It freezes when pilot attempt 3 starts. Everything in v4 and v4.1 holds except the change below.

**Change from v4.1**, from pilot attempt 2 ([pilot-notes-v4.md](pilot-notes-v4.md)):
- **LoCoMo names the user.** The harness feeds LoCoMo's speaker A in as the user. The extraction prompt said "refer to the user as User", so 7 of 20 hand-checked facts said "User" where the questions ask about Caroline or Jon: the reranker cannot match them by name, and the answer model cannot tell who "User" is. A5's LoCoMo ingest now calls `engine.set_user_name(speaker A)` (commit 4b04562), as an app that knows its user's name would (Zep keeps a user's first name on the user node). Facts are written with the name, and mentions of it link to the user's graph node. LongMemEval is second person ("I", "my"), so its user stays unnamed and its facts keep "User".
- No other arm changes: the name is set only for A5's LLM ingest, so A3, A4 and A6 contexts are untouched, and A5 without its facts must still equal A3 byte for byte.

`PREREG: ok` (frozen at pilot attempt 3)
