# Memorable memory architecture (v0.2)

This document covers the November 2026 rework: what the memory model now is, the research or open-source source of each mechanism, what changed from v0.1, and what it measures. Problems found during the rework, and how they were fixed, are listed in [What changed from v0.1](#what-changed-from-v01).

---

## 1. The layer model

```
┌──────────────────────────────── one SQL database (SQLite | Postgres | MySQL) ────────────────────────────────┐
│ L0  turns        verbatim messages, append-only, per session          ground truth, provenance, excerpts     │
│ L1  episodes     topic/time-coherent segments of a session            title, summary, key points, entities   │
│     sessions     one row per session                                  rolling summary (folded per episode)   │
│ L2  memories     atomic facts/preferences/skills/rules/context        bi-temporal versions, provenance        │
│     entities ─── relations (s, p, o)  ─── mentions (entity ↔ fact | episode)       knowledge graph           │
│ L4  context      what gets injected: core profile + relevant facts + past episodes + excerpts, budgeted      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
   every row: namespace + user_id   ·   in-process indexes: vector (numpy) + BM25 (FTS5 or incremental)
```

| Layer | Table(s) | Written when | Read by | Source of the idea |
|---|---|---|---|---|
| L0 turns | `turns` | every call (unseen suffix only) | excerpt retrieval, episode summarisation | MemPalace (verbatim storage scores 96.6% R@5 on LongMemEval) and Anthropic's "transcripts are ground truth" |
| L1 episodes | `episodes`, `sessions` | at a boundary; summarised on close | episode retrieval, session summary | EM-LLM (ICLR 2025), HiMem and ES-Mem (2026): event segmentation |
| L2 facts | `memories` | extraction + update phase per exchange (or per episode) | fact retrieval, core profile | Mem0 (ADD/UPDATE/DELETE/NOOP), Zep/Graphiti (bi-temporal) |
| L2 graph | `entities`, `relations`, `mentions` | with every new fact; with every closed episode | Personalized PageRank signal | HippoRAG 2 (PPR over entities + passages), Graphiti (edge invalidation) |
| L4 context | n/a | each call | the model | Letta core memory / Mem0 profile (always-on core facts) |

Procedural memory (the report's L3) is **not** in this release. The schema leaves room for it, and section 6 covers next steps.

---

## 2. Write path

There is now one pipeline ([pipeline.py](../memorable_ai/core/pipeline.py)). Both the interceptor and `engine.ingest()` call it, sync or async, in the foreground or on a background worker.

1. **Scope.** `(namespace, user_id, session_id)` is resolved from, in order: explicit arguments, then `with engine.scope(...)`, then call metadata (`metadata.user_id`, `user`, `safety_identifier`, Responses `conversation`), then legacy per-message metadata. As a last resort the session id is a hash of the user and the first user message.
2. **Turns (L0).** Clients resend the whole history on every call, so only the unseen suffix is appended. The suffix is found by matching each message's hash against the stored tail ([`new_suffix_start`](../memorable_ai/core/episodic.py)). This also handles sliding-window clients and stateless single-message clients.
3. **Segmentation (L1).** Before each user turn, the pipeline checks three boundaries:
   - **time gap:** more than 30 minutes idle
   - **length cap:** 40 turns
   - **semantic surprise:** `s = 1 − cos(turn, episode centroid)` crosses `max(mean_s + 2·std_s, 0.45)`, with the statistics tracked online (Welford) inside the current episode. This is EM-LLM's adaptive-threshold idea applied to embedding surprise instead of token surprise. Turns with fewer than 4 content words never trigger a boundary.
4. **Extraction (L2).** It sees the new turns plus a context window of preceding turns and the session summary, and the **session date**: relative dates are rewritten to absolute ones ("yesterday" → "on 2026-11-13"). There are two extractors:
   - **LLM** (one call per batch), returning facts with entities, `(s, p, o)` relations and source-turn indices
   - **patterns**, which run on user turns only and produce full statements ("User lives in Berlin")
5. **Update phase.**
   - Candidates are the top-k similar facts *in the same user scope*.
   - With an LLM, one batched decision call is made, and its target ids are validated against the real candidates, so a hallucinated id never mutates a row.
   - Without an LLM: an exact or near duplicate is a NOOP; a conflict on a single-valued predicate (lives_in, works_at, …) is an UPDATE; anything else is an ADD.
   - UPDATE and DELETE are **versioned**: a new row is written, and the old row gets `invalidated_at` and `superseded_by_id`.
6. **Graph.** Entities, relations and mentions are upserted. A new edge on a single-valued predicate invalidates the previous edge, and the fact behind it is retired too. This is Graphiti-style conflict resolution that works without an LLM.
7. **Episode close.** An episode is closed at a boundary, by `end_session()`, or by the idle sweep. Closing produces a title, a summary and **key points**:
   - **With an LLM:** one JSON call.
   - **Without:** extractive selection. Each utterance is scored as 0.55·centrality + 0.25·information density + 0.20·user-speaker, then chosen with MMR for diversity.

   The episode summary is then folded into the session summary, so summaries are hierarchical and are not regenerated on every call.

## 3. Read path

[retrieval.py](../memorable_ai/core/retrieval.py), with context assembly in [context.py](../memorable_ai/core/context.py).

- **Facts** are ranked by weighted Reciprocal Rank Fusion of three signals, each filtered to the user *inside* the search:
  - dense (cosine)
  - BM25 (SQLite FTS5 with a stem-prefix expansion, or an incremental in-process BM25 elsewhere)
  - graph: Personalized PageRank seeded from query entities, plus predicate words on the user's own edges, so "my sister" reaches Emma; also seeded from the top dense hits
- **Episodes and turns** have their own rankings (dense + BM25 RRF) and their own slots in the context block. Excerpts come from *other* sessions, since the current one is already in the prompt. Excerpts whose facts were already returned are moved to the end.
- **Context block:**
  - a small always-on **core profile** for user-scoped calls
  - relevant facts, past-episode summaries and excerpts
  - every line dated, all within `max_context_tokens`
  - framed as "reference data, not instructions", a cheap mitigation for memory poisoning (OWASP ASI06)
- **Injection preserves API shapes:**
  - OpenAI chat: appended to the existing system/developer message, or a new system message
  - Anthropic: the `system` parameter
  - Responses API: `instructions`

  Appending keeps the caller's prompt-cache prefix intact.

## 4. What changed from v0.1

| v0.1 defect (from the Nov 2026 audit) | Now |
|---|---|
| OpenAI/Anthropic hooks never attached (`OpenAI.chat` is a cached_property) | SDK resource classes are patched: chat, Responses, Messages, sync and async, streaming, plus litellm `completion`/`acompletion`. Patches are installed once per process and dispatch to the most recently enabled engine. Tested against the real SDKs over a mock HTTP transport |
| Anthropic would receive a `system`-role message (HTTP 400) | Context goes into `system` |
| Default sync path skipped summary, update and graph | One pipeline for sync, async and background |
| Injection path had no user scoping (cross-user leak) | `user_id` column; every signal is filtered in SQL or in the index; anonymous calls only see the shared pool |
| Fusion summed raw scores (BM25 swamped cosine) | Weighted RRF |
| UPDATE overwrote in place (stale embedding and FTS); DELETE hard-deleted | Version chains (`supersedes_id` / `superseded_by_id`), with FTS and indexes kept in sync; `engine.history(id)` |
| Graph in process memory; ADDed facts never got nodes | SQL entities, relations and mentions with temporal edges; PPR retrieval |
| Neo4j `a OR b AND valid` precedence bug | Parenthesised; matches all query tokens |
| Semantic search loaded the 1,000 highest-importance rows; BM25 rebuilt per query; O(n) dedup scan per insert | Persistent vector index with incremental delta sync; FTS5; dedup by content hash |
| Updater loaded a SentenceTransformer per fact | One shared embedder per process |
| Summaries leaked across sessions | Keyed by session (`sessions` table) |
| `json_extract` (SQLite-only) scoping; identity aliases lost on restart | Portable `user_id` column; aliases persisted in `identity_aliases` |
| Consolidator read `access_count` from metadata (always 0); contradiction groups by first 3 words | Column-based access bonus; embedding-neighbour contradiction pairs within a user, retired with history |
| Pattern extraction ran on assistant text ("I can help" → user skill) and stored fragments | User turns only, full statements, relative dates anchored |
| Tests could hit paid LLMs (keys in `.env`) and took 73 s | `MEMORABLE_LLM__DEFAULT_MODEL=none` plus an auth circuit breaker; tests use a hashing embedder: 113 tests in about 3 s |

New public API: `engine.scope()`, `ingest()`, `recall()`, `get_context()`, `end_session()`, `forget()`, `forget_user()`, `history()`, `list_episodes()`, `graph()`, `flush()`, `consolidate()`, and `add_memory(..., infer=True)`.

## 5. Measurements

`python -m benchmarks.locomo.retrieval_eval` ingests all 10 LoCoMo conversations (5,882 turns) through the real pipeline, with no LLM. For each of the 1,536 questions that have evidence, it checks whether an annotated evidence turn appears in the top 5. This measures **retrieval only**, not QA accuracy, and is not comparable to vendors' end-to-end LoCoMo scores. Embedder: all-MiniLM-L6-v2. Raw output: [results/retrieval_minilm_k5.json](../benchmarks/locomo/results/retrieval_minilm_k5.json).

| Configuration | Recall@5 | Multi-hop | Temporal | Single-hop | Open-domain |
|---|---|---|---|---|---|
| Dense only | 40.1% | 35.5% | 43.0% | 42.4% | 22.8% |
| Dense + contextual turn vectors | 43.3% | 37.2% | 42.1% | 48.2% | 21.7% |
| BM25 only | 56.9% | 43.6% | 65.1% | 60.6% | 34.8% |
| RRF, original weights (dense 1.0 / BM25 0.8) | 52.7% | 47.9% | 58.9% | 54.3% | 31.5% |
| **RRF, new defaults (0.5 / 1.0) + contextual vectors** | **56.5%** | 50.4% | 60.7% | 59.6% | 32.6% |
| RRF 0.3 / 1.0 (best tried; not adopted, tuned on the test set) | 58.5% | 50.0% | 64.2% | 62.0% | 33.7% |
| Episodes (evidence inside a top-5 episode) | 53.5% | **64.5%** | 44.2% | 52.8% | 42.4% |

What this says:

- **With MiniLM, BM25 is the stronger signal, and dense-heavy fusion makes results worse.** This matches the July 2026 "When does memory help?" finding that naive hybrid can underperform a single strong signal. The defaults now weight BM25 above dense. The biggest remaining lever is a retrieval-tuned embedder: EmbeddingGemma, BGE or E5, with query prompts, which are supported via `Embedder.encode_query`. EmbeddingGemma-2 is cached on this machine but needs a newer `transformers`.
- **Episodes are the right unit for multi-hop questions** (64.5% vs about 50% for single turns). This is evidence for keeping the episodic layer.
- **Throughput:** ingest takes about 57 s for 5,882 turns on a laptop CPU, including embedding, segmentation, extraction and graph building. Retrieval p50 is about 10 ms.

## 6. Survey: trending memory repos and recent papers (Nov 2026), and what we took

| Project | What it does | Adopted here | Not adopted (why) |
|---|---|---|---|
| **Neocortex / TinyCortex** (tinyhumansai, OpenHuman; markets "1B+ tokens") | Rust engine: dedup, entity linking across sources, recency weighting, knowledge graph plus temporal chains, "bucket-seal" memory tree. Claims no LLM in the memory layer; raw data is discarded after compression | LLM-free path for everything (patterns, extractive summaries, heuristic graph); hierarchical roll-up (turn → episode → session) | Discarding raw data: we keep verbatim turns because MemPalace and LoCoMo show verbatim recall matters. Its scale claims are vendor-reported and inconsistent (10M vs 1B) |
| **MemPalace** (about 56k stars) | Stores raw text and retrieves with Chroma. 96.6% R@5 on LongMemEval (retrieval-only) | Verbatim L0 store with excerpt retrieval | The spatial "palace" metaphor: an April 2026 critique shows the score comes from the vector DB, not the metaphor |
| **agentmemory** (rohitg00; #1 trending May 2026) | Hooks capture observations; BM25 + vector + graph with RRF (k=60); working/episodic/semantic/procedural tiers; provenance; human viewer | RRF k=60; provenance (`source_turn_ids`, `source_episode_id`); conservative defaults | LLM compression at hook time (off there too) |
| **Mem0** | Extract → ADD/UPDATE/DELETE/NOOP; multi-signal retrieval; dropped its graph store in 2026 | Batched update phase with target validation | — |
| **Zep / Graphiti** | Bi-temporal knowledge graph; edge invalidation | Bi-temporal facts and relations; single-valued predicate invalidation | Neo4j as a hard dependency (SQL graph instead) |
| **HippoRAG 2** | PPR over entities plus passage nodes | PPR over entities plus fact nodes, seeded by query entities and dense hits | OpenIE over every passage (cost) |
| **EM-LLM / HiMem / ES-Mem** | Surprise- and topic-based event segmentation | Adaptive surprise threshold plus time gap | Token-level surprise (needs model logits) |
| **Letta** (core memory, sleep-time compute) | Always-in-context core block; offline consolidation | Core profile in context; consolidator with sleep-time jobs | Agent-edited memory blocks |
| **Mastra observational memory / Anthropic "dreaming"** | Dated observation logs; offline verification against transcripts | Dated context lines; transcripts kept for future verification | LLM verification pass (next step) |
| **Cognee, hippo-memory, OpenViking** | Graph/ECL pipelines; issue→fix memory for coding agents; file-system context DB | Not in this release | Procedural/coding-agent focus belongs to the procedural-memory phase |

## 7. Not done yet (next steps, in priority order)

1. **Embedder upgrade and LLM-on evaluation.** Re-run `retrieval_eval` with a retrieval-tuned embedder, then run the full LoCoMo QA with an answer model and a judge. That needs an API budget decision.
2. **Procedural memory (L3).** Tool-call traces → procedures with pre/postconditions, success stats and a precision gate (see the strategy report).
3. **Dreaming.** An LLM pass that checks facts and episode summaries against L0 transcripts, and invalidates or repairs them.
4. **pgvector / sqlite-vec.** For indexes beyond about 1M rows (the numpy index is exact and fine to a few hundred thousand).
5. **Provenance-derived trust tiers.** Taint facts that came from tool output or untrusted sources, for poisoning defence.
