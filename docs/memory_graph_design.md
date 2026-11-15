# Memory graph (v0.3): design proposal

**Status:** draft for review; nothing here is implemented.
**Builds on:** [memory_architecture.md](memory_architecture.md) (v0.2 layers).
**Companion documents:** [procedural_memory_design.md](procedural_memory_design.md) and [industry_norms_2026-11.md](industry_norms_2026-11.md).
**Requested** on 2026-11-15: "heavily involve knowledge graph in memory".

---

## TL;DR

In v0.2 the knowledge graph is a **side index on L2**. Entities and relations are extracted from facts, and Personalized PageRank (PPR) over them is one input to RRF.

This proposal makes the graph **the backbone that connects every layer**: one typed, bi-temporal property graph. Every kind of memory is a node, and provenance is an edge. Node kinds:
- **What happened:** turns, episodes, tasks
- **What is true:** entities, facts
- **How to do things:** procedures, steps, tools
- **What was learned:** insights, communities

The graph is useful for three things:

1. **Lifecycle as graph operations.**
   - Deleting raw data finds and removes the derived memories that no longer have support.
   - Trust taint flows along `derived_from` edges.
   - Dreaming re-verifies a fact against its sources.
2. **Procedures live next to the entities they act on.**
   - "Deploy payments" reaches the deploy procedure through the `payments` entity, even with no textual overlap.
   - A procedure's arguments bind to *this user's* entity and fact nodes.
3. **A three-tier walk:** insights ↔ tasks/episodes ↔ raw interactions (G-Memory). One query returns both abstract guidance and concrete precedent.

Facts stay text rows. They remain the system of record for *content*; the graph holds *structure, time and provenance*.

**Speed:** SQL stays the durable store, and an in-process sparse-matrix engine runs the walks. Measured at about 2 ms per user and about 70 ms on an 800k-edge org graph, against 48 ms to 862 ms (p95 3.1 s) for the current code, which is also less accurate at scale (§3.3).

**Scope:** private `user` tier plus a shared `org` tier, with governed promotion (§3.4).

## 1. Evidence: for and against

| For a graph-centred design | Against / caution |
|---|---|
| **Zep / Graphiti**: episodic, semantic-entity and community subgraphs. Bi-temporal edges. Episodes stored losslessly, and every edge cites its episodes ([Zep](https://blog.getzep.com/how-zep-tracks-provenance-in-agent-memory/)) | **Mem0g**, the graph variant, scores only about 2% above plain Mem0 on LoCoMo ([paper](https://arxiv.org/html/2504.19413v1)) |
| **HippoRAG 2**: PPR over a passage + entity graph improves multi-hop and associative recall (we already use this) | **Mem0 removed graph memory from OSS v3** (PR #4805, April 2026) and replaced it with entity linking ([docs](https://docs.mem0.ai/open-source/features/graph-memory)) |
| **G-Memory** (NeurIPS 2025): a three-tier insight / query / interaction graph, up to +20.89% across five benchmarks ([paper](https://arxiv.org/abs/2506.07398)) | Our own LoCoMo measurement: BM25 is the strongest single signal (56.9% R@5). The graph's share of the gain hasn't been isolated yet |
| **AriGraph**: semantic and episodic memory in one graph for agents | — |
| **Procedural Graphs** (2609.09153): (procedure, relation, procedure) triplets; **Goal-Oriented Graphs** (2610.25996): goals plus prerequisite dependencies ([arXiv](https://arxiv.org/abs/2610.25996)) | — |
| **Cognee**: graph-centric with ontology support. **The MCP reference memory server is a knowledge graph** of entities, relations and observations ([README](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)) | — |

**What this means for the design:**
- A graph used as a *second copy of facts* for QA retrieval doesn't earn its keep. That is the Mem0 lesson.
- The graph does pay off when it carries what text rows can't:
  - (a) provenance and lifecycle
  - (b) multi-hop and entity-centred retrieval
  - (c) procedural structure (steps, tools, branches, bindings)
  - (d) temporal validity ("as of" queries)
- The design targets exactly these four.

## 2. The graph at a glance

```
                       insight ◄──supports── episode / task
                          ▲                      │
                   member_of│                   part_of
   community ◄──member_of── entity ◄──mentions── turn ──part_of──► episode ──part_of──► session
                          ▲   ▲ │                  ▲                  │next
                    about │   │ └─relates(pred, valid_from..valid_to)─► entity
                          │   │                    │ derived_from
                        fact ─┘◄────derived_from───┘
                          │ supersedes
                          ▼
                        fact(v1)

   task ──instance_of──► procedure ──part_of◄── step ──next/branch/alt──► step
                              │                  │ uses_tool ──► entity(type=tool)
                              └─operates_on──► entity      binds ──► fact / entity predicate
   pitfall ──pitfall_of──► step
```

## 3. Schema

### 3.1 Nodes are rows in their own tables
No duplicate node table. A node is identified by `(kind, id)`:

| Kind | Table | Layer | New in v0.3 |
|---|---|---|---|
| `turn` | `turns` (incl. `tool_call` / `tool_result` kinds) | L0 | tool kinds (procedural P1) |
| `session`, `episode` | `sessions`, `episodes` | L1 | — |
| `task` | `tasks` | L1 | yes (procedural P1) |
| `entity` | `entities` (typed; `tool` is an entity type whose schema is in metadata) | L2 | `summary`, `summary_embedding`, `agent_id` |
| `fact` | `memories` | L2 | — |
| `procedure`, `step`, `pitfall` | `procedures`, `procedure_steps`, `procedure_pitfalls` | L3 | yes |
| `insight`, `community` | `insights`, `communities` | dreaming | yes |

### 3.2 One edge table

```sql
edges(
  id, namespace, user_id, agent_id,
  src_kind, src_id, kind, dst_kind, dst_id,
  predicate,                 -- canonical predicate when kind = 'relates'
  role,                      -- e.g. subject|object for 'about', arg name for 'binds'
  weight FLOAT, confidence FLOAT,
  valid_from, valid_to,      -- event time (when it was true)
  created_at, invalidated_at, superseded_by_id,  -- transaction time (when we learned/retired it)
  evidence_kind, evidence_id, -- the turn/episode/task that asserted this edge
  source, trust_tier,        -- provenance root and trust (trust track)
  extra JSON
)
INDEX (src_kind, src_id, kind), (dst_kind, dst_id, kind), (namespace, user_id, kind, invalidated_at)
```

| Family | Edge kinds |
|---|---|
| Provenance | `derived_from` (fact/summary/procedure/insight → turn/episode/task), `supersedes` |
| Episodic | `part_of`, `next`, `mentions` |
| Semantic | `relates` (+ predicate, bi-temporal), `about` (fact → entity, with role), `alias_of`, `member_of` |
| Procedural | `instance_of` (task → procedure), `next` / `branch` / `alt` (step → step), `uses_tool`, `operates_on`, `binds`, `subprocedure`, `pitfall_of` |
| Evidence | `supports`, `contradicts` (weighted) |

**Migration:**
- v0.2's `relations` table becomes `edges(kind='relates')`, and `mentions` becomes `edges(kind='mentions')`.
- `memories.source_turn_ids` and `source_episode_id` become `derived_from` edges; the columns stay as a cache.
- Both tables were created in the uncommitted v0.2 work, and the published v0.1.1 schema doesn't have them, so unifying now breaks no user databases.

### 3.3 Graph engine: SQL for durability, an in-process sparse-matrix engine for speed (decided)

**Decision (2026-11-15, priorities: quality and speed):** a third option, neither "SQL + networkx" nor "graph server".

- **SQL** (SQLite / Postgres) stays the system of record: transactions, durability, portability, the same database as everything else.
- **`GraphEngine`** keeps each scope partition in memory as **scipy CSR sparse matrices**, one per edge kind. It stays current through the `edges` id watermark plus an invalidation log, the same pattern `VectorIndex` already uses.
- Ranking runs as sparse matrix–vector products, the same model FalkorDB / GraphBLAS use server-side, but with no server and no network hop.

**Measured** with [benchmarks/graph/ppr_engines.py](../benchmarks/graph/ppr_engines.py): synthetic Zipf-distributed graphs; facts and entities are nodes; mention and relation edges; 2 seed entities per query. The reference is exact full-graph PPR.

| Partition size | Current: SQL BFS + networkx (2-hop, 4,000-node cap) | CSR exact (float64) | CSR exact (float32, tol 1e-6) | CSR local push (ε 1e-5) | CSR build from SQL (one-off) |
|---|---|---|---|---|---|
| per-user: 5k facts, ~13k edges | 48 ms p50 / 64 ms p95; overlap@20 **0.85** | **1.4 ms** | — | 16.5 ms; 0.77 | 7 ms |
| team: 50k facts, ~134k edges | 271 ms / 514 ms; **0.62** | **16 ms** | — | 10–12 ms; 0.60–0.81 | 49 ms |
| org: 300k facts, ~810k edges | 862 ms / **3.1 s**; **0.26** | 146 ms | **71 ms / 142 ms; 0.88** | 5–10 ms; 0.72–0.81 | 509 ms |

**Reading the numbers:**
- The current approach is slow, and it is also **wrong at scale**: the hop and node caps truncate the walk, so it agrees with exact PPR only 26% of the time on the org graph.
- Overlap@20 understates agreement for every engine, because many facts attached to the same entity tie on score. float32 vs float64 exact still lands at only 0.88.
- The fix for ties is quality work in its own right: break them with node priors (§5).
- The pure-Python local push is size-independent, but noisy and slower at small graphs.

**Policy:**
- **Exact float32 power iteration by default.** About 2 ms per user, about 16 ms per team, about 70 ms at 800k edges.
- Above `graph.exact_max_edges` (default 2M), or under a hard latency budget, use local push. A compiled push kernel is a later optimisation if it's ever needed.
- Community detection and the other dreaming jobs use the same matrices.

**Optional server backends** are for multi-node deployments, where one process can't hold the graph:

| Backend | Why |
|---|---|
| **FalkorDB** | The same sparse-matrix model (GraphBLAS), so behaviour stays equivalent. Graphiti and Cognee support it |
| **Neo4j** | For teams that already run it. Unifies today's [neo4j_backend.py](../memorable_ai/graph/neo4j_backend.py) |

Kùzu is ruled out: the repository was archived on 2025-10-10 ([The Register](https://www.theregister.com/2025/10/14/kuzudb_abandoned/)).

The **`GraphStore`** protocol has two parts:
- **SQL side:** `add_edges`, `invalidate_edge`, `delete_node_cascade`
- **`GraphEngine` side:** `ppr(seeds, weights, as_of)`, `neighbors(nodes, kinds, hops)`, `communities()`

This is where the storage-backends track attaches for graphs.

### 3.4 Scope tiers in the graph (D2, decided)

Every node and edge carries a tier: **`user`** (private; `user_id` set) or **`org`** (shared across the app; `user_id` NULL within the namespace). An optional **`team`** tier (`group_id`) can come later. `agent_id` is an *attribute*, used for filtering and ranking, **not a wall**.

- **Reads** take the union of the caller's `user` subgraph and the `org` subgraph, with every signal scoped at the data layer, as v0.2 already does for the shared pool.
- A user entity ("payments" in Alice's graph) links to the org entity of the same identity through an `alias_of` edge, so walks cross tiers without copying data.
- **Writes** derived from a user's session land in `user`. Moving anything to `org` is a **governed promotion**:
  - PII-safe generalisation
  - evidence from at least 2 distinct users, or admin approval
  - not quarantined

  This matches the enterprise pattern of restricting writes to broader scopes (see the norms doc §3).
- **Unused org memory expires** through the lifecycle (stale → deprecated), the way GitHub Copilot Memory drops unused memories after 28 days.

## 4. Write path

| Step | Today (v0.2) | v0.3 |
|---|---|---|
| Extraction | Regex patterns, plus LLM extraction when one is configured. Predicates are free-form | LLM-first (NIM). The prompt carries the **ontology**: entity types and a predicate registry. A validator maps or flags predicates; regexes stay as the no-LLM fallback |
| Entity resolution | Exact `norm_name` match | Graphiti-style:<br>• candidates from exact name, aliases, character n-grams and an embedding of (name + summary), within scope<br>• one exact candidate → link<br>• ambiguous → an LLM decides with the candidate summaries in hand<br>• result: an `alias_of` edge or a new entity |
| Entity summaries | none | LLM-maintained and updated incrementally from new facts. Embedded, used for resolution, entity cards and the profile |
| Fact ↔ graph | Relation rows with `memory_id` | `about` edges (fact → subject/object), `relates` edges with `evidence` → fact, and `derived_from` edges fact → turns/episode |
| Single-valued predicates | Hard-coded `FUNCTIONAL_PREDICATES` | A registry entry: `{functional, inverse, symmetric, domain, range, aliases}`. Invalidation as now (Graphiti) |
| Episodes / tasks | Mentions only | `part_of` (turn → episode → session, task → episode), `next` (episode chain), `mentions` |
| Procedures | — | A subgraph written by the procedural compiler (see the procedural doc) |
| Dreaming | — | Communities, insights, predicate canonicalisation, fact re-verification |

**Ontology.** Entity types to start: `person, organization, place, project, service, tool, file, product, event, concept`. The predicate registry is seeded from today's `FUNCTIONAL_PREDICATES` and `_REL_PATTERNS`. Unknown predicates are allowed but flagged `custom`; dreaming clusters them and folds them into canonical ones. Applications can register their own types and predicates, as in Graphiti custom types and Cognee ontologies.

## 5. Read path

1. **Seeds:**
   - query entities: n-gram + embedding match, plus optional LLM entity linking
   - the top dense / BM25 hits in every layer (facts, episodes, tasks, procedures)
2. **Query-adaptive typed PPR** over the full partition (`GraphEngine`, §3.3); no hop or node caps any more. The novel parts, each of which can be measured on its own once evals resume:
   - **Per-query edge weights.** The transition matrix is `A(q) = Σ_k w_k(q) · A_k` over the edge-kind matrices. Combining is a cheap sparse sum. A router picks a weight profile from the query type:

     | Profile | Edge kinds weighted up |
     |---|---|
     | *factual* | `relates`, `about` |
     | *temporal* | `next`, `part_of`, with the temporal kernel on |
     | *procedural* | `operates_on`, `uses_tool`, `instance_of` |
     | *multi-hop* | flat weights |

     Heuristics come first, with an optional LLM router. Later the weights are **learned from logs**: which edge kinds led to memories that were actually used.
   - **Temporal kernel.** Edge weights are multiplied by validity at `as_of` (0 outside `[valid_from, valid_to)`) and an optional recency decay. That makes the walk time-aware, not merely filtered afterwards.
   - **Fused teleport vector.** The seeds mix entity matches with the top dense and BM25 hits mapped to their nodes, weighted by calibrated retriever scores. HippoRAG 2 seeds with passages plus entities; we extend that to every layer.
   - **Tie-breaking by node priors** (importance, recency, access count, verification status), which addresses the score ties seen in the benchmark.
   - Output is **one ranked list per node kind**. Each list goes into RRF as that layer's graph signal (today only facts get one).
3. **Three-tier walk for task-like queries** (G-Memory):
   - query → similar past `task` nodes
   - **up** to `procedure` (instance_of) and `insight` (supports)
   - **down** to the concise trace turns (`part_of` task)
4. **Entity cards:** for the top 1–3 entities, the summary plus currently valid relations go into L4 (Zep-style). This replaces some of today's loose profile facts.
5. **Temporal:** `as_of=t` keeps only edges with `valid_from ≤ t < valid_to` that were not invalidated before *t*. That answers "where did I live in 2024".

## 6. Lifecycle through the graph

- **Forget / retention:** deleting a turn or episode walks the inbound `derived_from` edges. A derived node with no remaining support is invalidated, or deleted when the deletion is hard. This is Zep's rule, and it gives right-to-be-forgotten for derived memory, not only raw data. `forget_user()` stays a scoped bulk delete.
- **Trust taint:** a node's trust tier is the *minimum* over its `derived_from` ancestors. Tool-sourced text flowing into a fact or procedure quarantines it.
- **Verification (dreaming):** re-read a fact's source turns and confirm, correct or invalidate it, through a versioned edit with evidence.
- **Contradictions:** `contradicts` edges collect evidence across episodes before a fact or insight is retired, instead of acting on a single message.

## 7. LLM use

The user decided LLM use is unconstrained, using NVIDIA NIM for now. Default model: `DEFAULT_MEMORY_MODEL` in [nvidia_nim.py](../memorable_ai/integrations/nvidia_nim.py) (`deepseek-ai/deepseek-v4-flash`). A stronger model can be set per job, for example dreaming.

| Job | When | Notes |
|---|---|---|
| Extraction (entities, relations with registry predicates) | Each write batch | Validated against the registry; the regex fallback stays |
| Entity resolution | Ambiguous candidates only | Batched |
| Entity summaries | Incremental after new facts | — |
| Communities + summaries | Dreaming | Louvain via networkx / Neo4j |
| Insights | Dreaming | Supported by episode/task evidence edges |
| Predicate canonicalisation | Dreaming | — |

The circuit breaker and `model="none"` keep working, so tests stay LLM-free with mocked replies.

## 8. Phases

| Phase | Deliverable | Testable by |
|---|---|---|
| **G0 Edge table, engine & provenance** | `edges` + `GraphStore`; the CSR `GraphEngine` (watermark sync, exact float32 PPR, push fallback); migration of relations/mentions; `derived_from` / `part_of` / `next` written by the existing pipeline; `user`/`org` tiers; `agent_id` on scope | existing 113 tests stay green; provenance and cascade tests; `ppr_engines.py` shows the speed-up |
| **G1 Ontology & resolution** | Type and predicate registry, registry-guided LLM extraction, entity resolution, entity summaries, `alias_of` | mocked-LLM tests; resolution fixtures (same name, different person; nicknames) |
| **G2 Multi-layer graph retrieval** | Typed PPR producing per-kind lists, entity cards in L4, `as_of` queries | retrieval unit tests; LoCoMo retrieval eval when evals resume |
| **G3 Procedural subgraph** | Procedural P2–P4 write and read through `edges` | procedural golden traces |
| **G4 Dreaming on the graph** | Communities, insights, predicate canonicalisation, fact re-verification | mocked-LLM tests |
| **G5 Lifecycle** | Cascade forget/retention, trust-taint propagation | cascade tests |

## 9. Decisions (made 2026-11-15; priorities: quality and speed)

| # | Decision | Made | Why |
|---|---|---|---|
| G-a | One `edges` table vs typed tables | **One table**, indexed `(src_kind, src_id, kind)` and `(dst_kind, dst_id, kind)` | **Quality:** uniform provenance, typed weights and cascades. **Speed:** one watermark stream feeds the CSR engine instead of N tables. Free now: both old tables exist only in uncommitted v0.2 code |
| G-b | Graph backend | **SQL record + in-process CSR `GraphEngine`; FalkorDB/Neo4j optional** | 10–80× faster than the current code, and correct at scale (§3.3 benchmark). No server needed |
| D2 | Scope | **`user` (private) + `org` (shared) tiers, governed promotion, `agent_id` as an attribute** | Enterprise pattern (§3.4; norms doc §3) |
