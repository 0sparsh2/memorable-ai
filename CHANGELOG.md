# Changelog

All notable changes to Memorable will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased

Memory graph foundation (G0). See `docs/memory_graph_design.md` and `docs/instruments_analysis.md`.

### Added
- One typed, bi-temporal `edges` table links every layer:
  - `relates`, `about`, `mentions`
  - `derived_from` (fact → turn/episode provenance)
  - `part_of` (turn → episode → session)
  - `next` (episode chain)
  - `supersedes`
- `GraphEngine`: in-process scipy CSR partitions with delta sync (change cursor + `graph_epoch`), exact float32 PPR with exact node masking, and a local-push fallback.
  - Graph ranking is 8–20× faster than v0.2 and stays accurate at scale (`benchmarks/graph/ppr_engines.py`).
- Tiers: `user` (private) plus the shared `org` tier (`user_id` NULL). Queries walk the user's partition with the org partition. Other users' edges are never loaded.
- `agent_id` on `Scope`, `engine.scope(agent_id=...)`, provider `metadata.agent_id`, and on memories, turns, episodes, sessions, entities and edges.
- `Storage.unsupported_dependents()` (cascade primitive), `add_edges`, `invalidate_edges`, `delete_node_edges`, `graph_state`.
- Query-embedding LRU cache.

### Changed
- `relations` and `mentions` are replaced by `edges`. Existing databases are backfilled once; the legacy tables are left untouched.
- Invalidating a fact retires its walkable edges; provenance edges stay as history.

## [0.2.0] - Unreleased

Foundation rework. See `docs/memory_architecture.md`.

### Fixed
- Interceptor now patches SDK resource classes: OpenAI Chat Completions + Responses, Anthropic Messages (sync and async), LiteLLM `completion`/`acompletion`, including streaming. The OpenAI/Anthropic hooks previously never attached.
- Anthropic calls get memory in the `system` parameter (the API rejects a system-role message).
- Single write pipeline for sync/async/background paths (the sync path skipped update, summary and graph).
- Per-user scoping on every retrieval signal; anonymous calls only see the shared pool (cross-user injection was possible).
- Reciprocal Rank Fusion replaces raw-score summation (BM25 dominated).
- Updates/deletes are versioned instead of destructive; FTS and indexes stay in sync.
- Summaries keyed by session; identity aliases persisted; portable user filter (no `json_extract`).
- Consolidator uses the real `access_count`, finds contradiction candidates by embedding within a user, and keeps history.
- Neo4j query operator-precedence bug.
- Pattern extraction no longer reads the assistant's text as user facts, and produces full statements.

### Added
- Episodic memory: verbatim turns, surprise/time-gap episode segmentation, episode summaries with extractive key-segment selection (centrality + MMR) or LLM.
- SQL knowledge graph (entities, relations, mentions) with temporal edge invalidation and Personalized PageRank retrieval.
- Relative-date anchoring of facts to the session date; dated, token-budgeted context with a core user profile.
- APIs: `scope()`, `ingest()`, `recall()`, `get_context()`, `end_session()`, `forget()`, `forget_user()`, `history()`, `list_episodes()`, `graph()`, `flush()`, `consolidate()`, `add_memory(infer=True)`.
- Background write mode, LLM circuit breaker, `MEMORABLE_LLM__DEFAULT_MODEL=none`, hashing embedder for tests.
- `benchmarks/locomo/retrieval_eval.py` (LLM-free evidence recall@k on LoCoMo).

### Changed
- Default retrieval weights: semantic 0.5, BM25 1.0, graph 0.6 (RRF). `entity_weight` is deprecated (folded into the graph signal).
- `delete_memory()` is hard by default (right to be forgotten); the update phase invalidates instead.

## [0.1.0] - 2025-01-XX

### Added
- Initial release of Memorable
- Interceptor-based architecture for transparent LLM call interception
- SQL-first storage layer (PostgreSQL, SQLite, MySQL support)
- Memory extraction from conversations (facts, preferences, skills, rules, context)
- Hybrid retrieval system (semantic + keyword + graph search)
- Graph-based memory for multi-hop reasoning (optional)
- Three memory modes: Conscious, Auto, and Hybrid
- Memory consolidation background agent
- Temporal memory tracking and coherence
- Framework integrations: OpenAI, Anthropic, LiteLLM, LangChain, AutoGen
- Comprehensive test suite (unit and integration tests)
- Benchmark implementations (LOCOMO, multi-hop, temporal)
- Complete documentation (README, architecture, API, benchmarks)

### References
- Memori: https://github.com/GibsonAI/Memori (interceptor architecture)
- Mem0: https://github.com/mem0ai/mem0 (research-backed techniques)
- Supermemory: https://github.com/supermemoryai/supermemory (graph architecture)

