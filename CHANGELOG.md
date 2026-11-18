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
- **Tool events in L0** (procedural memory P1a, `docs/procedural_memory_design.md` §4.1, §5):
  - Tool calls and tool results are stored as turns (`kind = tool_call | tool_result`) in the session's order, for OpenAI chat, Anthropic content blocks, the OpenAI Responses API and LiteLLM.
  - The model's own tool calls are recorded too, not just its text, including streamed calls: OpenAI `delta.tool_calls` fragments, Anthropic `input_json_delta` and Responses `output_item.done`.
  - Payloads are redacted: secrets always, PII per `capture.pii_redaction` (`off` by default, or `emails_phones` / `strict`). They are capped at `capture.max_payload_bytes` (256 KB; head + tail + sha256 + length above it).
  - `salient_values` are extracted before any cut, and the error flag is set conservatively.
  - Tool events are not embedded, are not excerpt candidates, and do not drive episode segmentation or fact extraction.
  - Resent agent histories stay idempotent; message hashes are unchanged, so existing databases need no rewrite.
  - `capture.tool_events=False` turns capture off.
- **Tasks and outcomes** (procedural memory P1b, design §6):
  - A `TaskTracker` segments each session's event stream into tasks. A task opens at the user message that leads to a tool call (that message is the goal). It closes at a tool-free answer, at the next user message, or after the idle gap. Follow-ups such as "now do the same for auth" open a child task (`parent_task_id`).
  - Outcomes come from the strongest available source:
    - explicit: confidence 1.0, via `with engine.task("...") as t:` + `t.success()` / `t.failure()`, or `engine.mark_outcome()`; an exception in the block marks the task failed;
    - tool signals: 0.8 (tool errors, test summaries, HTTP status, exit codes);
    - the user's next message: 0.6 (patterns for now).
  - Offered tools (`tools=` / `functions=`) and used tools are recorded per task, with LLM-call, tool-call and error counts.
  - `engine.list_tasks()` and `engine.get_task()` (scoped to the caller) read tasks back. Turns get `task_id` plus `part_of` edges, and tasks get `part_of` edges to their episode.
- `engine.record_trace(goal, steps, outcome=None, ...)`: records an agent run from any framework (no intercepted SDK needed) as L0 events plus one explicit task. Steps take tool, args, result, error flag and rationale; tool signals label the task when no outcome is given.
- `engine.ingest_otel(spans)`: ingests OpenTelemetry GenAI spans (OTLP-JSON or plain dicts). Chat spans contribute `gen_ai.input.messages` / `gen_ai.output.messages` (text, `tool_call` and `tool_call_response` parts) and `gen_ai.tool.definitions`; `execute_tool` spans contribute tool name, call id, arguments, result and error status. Each conversation (`gen_ai.conversation.id`, else the trace) becomes one canonical event stream with no duplicates, and re-sent exports are idempotent.
  - Benchmarks: `benchmarks/embeddings/turn_signals.py` (fusion rules, rerankers, bootstrap CIs on LoCoMo and LongMemEval) and `bench_rerank_latency.py`. `retrieval_eval.py` gains `turn/pipeline`, `--fusion` and `--rerank`.
  - `training/kaggle/`: synthetic memory-QA generation and contrastive / reranker fine-tuning jobs (Kaggle GPUs).
- Knowledge-graph ontology and entity resolution (G1, `docs/memory_graph_design.md` §4.1):
  - `graph/ontology.py`: entity types and a predicate registry. Each entry records functional, symmetric, inverse, domain, range and aliases, plus the role words a question uses for each end ("mother" vs "daughter"). Extracted predicates map onto canonical ones: `moved_to` → `lives_in`, `child_of` → `parent_of` (ends swapped), `has_wife` → `married_to`. Symmetric edges are stored once, with the user first. Unknown predicates are kept and flagged `custom`. Apps add their own with `register_type` / `register_predicate`.
  - The LLM extraction prompt carries the registry (types, predicates, which are single-valued or symmetric).
  - The extractor's own word survives on the edge (`extra.raw`, `extra.role`), so "my sister" still beats "my brother" after both become `sibling_of`.
  - `graph/resolution.py`: mentions resolve to existing entities by exact name, alias, nickname ("Bob" → "Robert Chen") or partial name ("Ana" → "Ana Silva"; the fuller name becomes the display name). Two different full names never merge. A near-miss surname ("Sliva") is never linked without an LLM.
  - Same name, different person: kinship roles are exclusive, so the user's daughter Ana is not the user's sister Ana.
  - Ambiguous names are settled by graph context: the same relation to the same entity, any relation to the fact's other entities, or facts mentioning both. When an LLM is configured, it adjudicates only what stays ambiguous, in one small JSON call per fact (`memory.entity_resolution_llm`, on by default). Otherwise the rules fall back conservatively.
  - New `entity_aliases` table. Entities get a fact-sheet `summary` built from their valid relations (no LLM), refreshed on write and used to adjudicate.
  - `KnowledgeGraph.merge_entities` / `engine.merge_entities(duplicate, into, user_id=...)`: names become aliases, edges move (marked `extra.merged_from`) and an `alias_of` edge records the merge.
  - `retrieval_eval.py` gains fact-level signals (`fact/graph`, `fact/pipeline`), graph size counts and per-question hit vectors for paired tests. On LoCoMo without an LLM, G1 leaves turn and episode recall unchanged and lifts graph-only fact recall@5 from 0.014 to 0.018 (+0.4 pts, 95% CI +0.1 to +0.7). Entities fall from 940 to 907.

### Changed
- **Memory injection is prompt-cache friendly** (`memory.injection="tail"`, the new default). Retrieved memory goes in front of the latest user message instead of into the system prompt. Everything before that message now stays byte-identical across turns, so provider prompt caches (OpenAI, Anthropic, Gemini) keep the whole conversation prefix; before, a changed memory block forced the whole history to be re-billed every turn. The memory block is reused while the latest user message is unchanged (`context_reuse_seconds`, 600), so agent tool loops send identical bytes and skip repeated retrieval. `injection="system"` restores the v0.2 placement. Adopted from billion-context; see `docs/procedural_memory_design.md` §17.
- Graph relations are stored under canonical predicates, and single-valued predicates come from the ontology instead of `FUNCTIONAL_PREDICATES` (kept as a derived constant). A move written as `relocated_to` now supersedes `lives_in`, and a symmetric single-valued relation (`married_to`) is superseded from either end.
- Python 3.10+ is required. 3.8 and 3.9 are past end of life, and current numpy, sentence-transformers and onnxruntime need 3.10+. CI now tests 3.10–3.14 with current GitHub Actions.

### Fixed
- `forget_user()` left the user's procedural `tasks` rows behind (found while building G1). It now deletes them, and the new entity aliases too.

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

