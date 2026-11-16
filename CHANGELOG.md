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
- Embeddings (see `docs/embedding_study.md`):
  - ONNX Runtime backend, used automatically for models that publish an ONNX export: identical vectors, about 6× faster queries. It supports prompts, CLS/mean pooling, Dense layers and sidecar weights.
  - `onnx-int8:`, `onnx-o3:`, `static:` (model2vec) and `torch:` backend prefixes.
  - Embedding roles: semantic (main model), turn index (`turn_embedding_model`) and segmentation (`segment_embedding_model`).
  - Profiles via `MEMORABLE_EMBEDDING_PROFILE`:
    - `quality` (default with the `fast` extra): leaf-ir turns. LoCoMo fused turn recall 0.565 → 0.642, multi-hop 0.504 → 0.571, ingest 46 → 39 s.
    - `fast`: potion-32M turns. Fused 0.579, ingest 25 s, query 3.2 ms.
    - `legacy`: one model for everything (v0.2 behaviour).
  - Per-layer vector-space tags with automatic re-embedding when the model changes.
- `memorable-ai[fast]` extra (onnxruntime, model2vec, numba). Hypothesis property tests (dev extra), a concurrency test, and LongMemEval / model / backend benchmarks.
- Optional numba kernel for local-push PPR.
- Retrieval round 2 (`docs/retrieval_round2.md`):
  - **Score fusion** for the turn and episode layers (`RetrievalConfig.fusion = "score"`, the new default). Each signal's scores are z-normalised over the candidate pool. LoCoMo end to end: turn recall@5 0.642 → 0.701, episode recall@5 0.541 → 0.575. `fusion="rrf"` restores rank fusion. Non-semantic (hash) embedders keep RRF.
  - **Cross-encoder reranking of turn excerpts** (opt-in: `rerank=True`), with `rerank_depth` (20) and `rerank_blend` (0). `memorable_ai/core/rerank.py` runs rerankers on ONNX Runtime with identical scores, including modular cross-encoders such as the Ettin family. Ettin-17M adds +5.5 pts LoCoMo turn@5 at about 32–50 ms per query.
  - `VectorIndex.scores_for` / `Storage.vector_scores`: exact cosine for given ids.
- `MemoryEngine.expand("episode" | "fact", id)`: restores the source turns behind an episode summary or a fact, scoped to the caller.
  - Benchmarks: `benchmarks/embeddings/turn_signals.py` (fusion rules, rerankers, bootstrap CIs on LoCoMo and LongMemEval) and `bench_rerank_latency.py`. `retrieval_eval.py` gains `turn/pipeline`, `--fusion` and `--rerank`.
  - `training/kaggle/`: synthetic memory-QA generation and contrastive / reranker fine-tuning jobs (Kaggle GPUs).

### Changed
- **Memory injection is prompt-cache friendly** (`memory.injection="tail"`, the new default). Retrieved memory goes in front of the latest user message instead of into the system prompt. Everything before that message now stays byte-identical across turns, so provider prompt caches (OpenAI, Anthropic, Gemini) keep the whole conversation prefix; before, a changed memory block forced the whole history to be re-billed every turn. The memory block is reused while the latest user message is unchanged (`context_reuse_seconds`, 600), so agent tool loops send identical bytes and skip repeated retrieval. `injection="system"` restores the v0.2 placement. Adopted from billion-context; see `docs/procedural_memory_design.md` §17.
- Python 3.10+ is required. 3.8 and 3.9 are past end of life, and current numpy, sentence-transformers and onnxruntime need 3.10+. CI now tests 3.10–3.14 with current GitHub Actions.

### Fixed (found by the G0 deep evaluation, `docs/g0_deep_evaluation.md`)
- The graph engine synced invalidations by timestamp and could permanently miss one committed out of order. Sync now follows a commit-ordered write sequence.
- Nodes whose edges were all invalidated kept teleport mass as seeds.
- Graph counters used a Python read-modify-write, which could give duplicate values under concurrent writers. They now use an atomic SQL increment.
- `as_embedder()` wrappers crashed in `encode_query`.
- CI ignored test failures (`|| true`).
- Property tests leaked temporary databases.
- Graph engine memory went from 342 to 101 bytes per edge.
- `scipy` (imported by the graph engine) is now a declared dependency. Before, it was only installed transitively.
- `requirements.txt` no longer installs the obsolete PyPI `asyncio` backport (asyncio is in the standard library).
- The NVIDIA NIM default model (`deepseek-ai/deepseek-v4-flash`) was retired by NIM on 2026-08-07, so every default NIM call failed with 410 Gone. The new default is `nvidia/nemotron-3.5-lightning-30b-a3b`, and the model registry now lists only models checked live on 2026-11-16.

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

