# Procedural memory (L3): design proposal, v2

**Status:** draft v2 for review. Nothing here is implemented yet.
**Builds on:** [memory_architecture.md](memory_architecture.md) (L0 turns, L1 episodes, L2 facts, L4 context).
**Companion documents:**
- [memory_graph_design.md](memory_graph_design.md): the unified knowledge graph that procedures live in
- [industry_norms_2026-11.md](industry_norms_2026-11.md): the research behind D3 and D4

**v2 changes** reflect the user's answers of 2026-11-15:
- injection is deferred
- the knowledge graph is the backbone
- LLM-first compilation on NVIDIA NIM
- payloads and integrations follow industry norms
- the name stays

## Decisions recorded

| # | Question | Decision | Source |
|---|---|---|---|
| D1 | Default injection | **Deferred.** Memory architecture comes first. Procedures are learned, matched and exposed through an API, and matches are logged as *observation runs*. Injecting into prompts is a later switch | user |
| D2 | Who sees a procedure | **Two tiers: `user` (private) + `org` (shared across the app and all its agents)**, with governed promotion from private to shared. `agent_id` is a ranking/filter attribute, not a wall (§15; [norms §3](industry_norms_2026-11.md#3-d2-how-enterprises-scope-shared-memory)) | user, enterprise pattern |
| G-a, G-b | Graph storage | One `edges` table; SQL as the record plus an in-process CSR `GraphEngine` (benchmarked 10–80× faster than the current code, and correct at scale) ([graph doc §3.3](memory_graph_design.md#33-graph-engine-sql-for-durability-an-in-process-sparse-matrix-engine-for-speed-decided)) | delegated, decided on quality and speed |
| D3 | Payload retention | **Full payloads in the user's own database.** Secrets always redacted; PII configurable; a 256 KB cap for database health; TTL configurable, default keep. Matches Zep / Letta / LangGraph / AgentCore ([norms §1](industry_norms_2026-11.md#1-d3-what-other-systems-keep)) | user: "industry norm" |
| D4 | Integration surfaces | **Standards first.** Interceptor (have), OpenTelemetry GenAI ingest and emit, MCP server, then framework-native adapters ([norms §2](industry_norms_2026-11.md#2-d4-integration-surfaces)) | user: "industry standards" |
| D5 | LLM use | **Unconstrained, LLM-first compilation** on NVIDIA NIM (default `nvidia/nemotron-3.5-lightning-30b-a3b` since 2026-11-16; `deepseek-v4-flash` was retired by NIM), with deterministic scaffolding and validators | user |
| D6 | Name | **Keep "Memorable."** | user |

---

## TL;DR

Agents re-derive the same workflow every time they meet a similar task. Procedural memory records **how a task was done** and compiles it into a parameterised procedure. Procedures are stored **as subgraphs of the memory graph**, connected to the tools, entities and user facts they involve. They are learned from evidence over time.

1. **Capture from where we already sit.** Traces come from three places, all landing in L0 as turn events:
   - The interceptor sees tool calls in responses and tool results in the next request.
   - OpenTelemetry GenAI agent traces (`invoke_agent` / `execute_tool`) cover any instrumented framework.
   - `record_trace()` covers everything else.
2. **Compilation is LLM-first, scaffolded and checked.**
   - A deterministic pre-pass builds the value-flow graph and candidate bindings (TraceCompiler).
   - The LLM synthesises the procedure (intent, slots, conditions, pitfalls) on top of that pre-pass.
   - Deterministic validators reject hallucinated steps, unresolvable bindings and leaked secrets or PII.
3. **Procedures are graph nodes.** Steps link with `next` / `branch` / `alt`. Tools are entities (`uses_tool`), and so are the objects acted on (`operates_on`). Arguments bind to fact patterns that resolve in *the caller's own* subgraph. Tasks link with `instance_of`, and pitfalls hang off steps.
4. **Retrieval is an API now; injection comes later.**
   - `procedures.recall(goal, tools=…)` returns ranked candidates, a calibrated verdict (`confident | ambiguous | none`) and a rendered pointer.
   - Every new task is matched in **observation mode**, which logs whether the agent's own trace followed the procedure and how it ended. That gives learning signal without changing agent behaviour.
5. **Learning keeps a version history with evidence:** Beta statistics, lifecycle states, and **validated edits** that are accepted only if evidence doesn't get worse. Rejected edits are kept.

## 1. Terms

| Term | Meaning |
|---|---|
| **Task** | One goal an agent pursues: from the user message that starts it, through the tool loop, to the final answer. It has an outcome. A `task` node. |
| **Trace** | The task's ordered events: rationale, tool calls (name, args), tool results (output, error), final answer. These are `turn` nodes `part_of` the task. |
| **Procedure** | A generalised workflow for an *intent*. A `procedure` node plus `step` nodes, edges, pitfalls and statistics. |
| **Run** | One task matched to a procedure. `mode = observe` (now), or later `injected` / `holdout`. Records match score, adherence and outcome. |

The granularity follows Mem^p and Anything2Skill: a procedure is a *script* with step-level guidance, not a free-text skill file. A `SKILL.md` export exists for review, but it isn't the system of record.

## 2. Research and industry basis

| Work | What we take | What we don't |
|---|---|---|
| **TraceCompiler** (2608.02680) | Value-flow dependency edges; the binding taxonomy; refusing ambiguous side effects | Fully deterministic replay |
| **ReTrace** (2611.14492) | Preconditions from the agent's rationale; merging traces | Planner-free execution |
| **Procedural Graphs** (2609.09153) | (procedure, relation, procedure) graph; active-node localisation; **edits accepted only if validation doesn't drop; rejected edits remembered** | — |
| **G-Memory** (NeurIPS 2025, [2506.07398](https://arxiv.org/abs/2506.07398)) | Three tiers: insight ↔ query/task ↔ interaction; walk up for guidance and down for precedent | Multi-agent-only framing |
| **Goal-Oriented Graphs** ([2610.25996](https://arxiv.org/abs/2610.25996)) | Goals and prerequisites as inspectable structure, giving `depends_on` edges | — |
| **Demystifying Agent Skills** (2608.14036) | Precision is the bottleneck, so gate and abstain; short step outlines carry most of the value | — |
| **Managing Procedural Memory** (2606.23127) | Test-gated promotion; traces from diverse models make better skills; core + adapters | `SKILL.md` as storage |
| **Anything2Skill** (2606.09316) | Lifecycle states; Create / Merge / Revise / Discard; contraindications | — |
| **Workflow-to-Skill** (2606.06893) | *Observed* vs *inferred* provenance on every step | — |
| **ReasoningBank**, **ExpeL** | Failures become pitfalls attached to steps | Standalone failure rules (they hurt) |
| **AWM**, **Mem^p**, **ACE**, **LEGOMem** | Workflow induction; delta edits rather than rewrites; modular sub-procedures | — |
| **Mem0** procedural memory ([prompts.py](https://github.com/mem0ai/mem0/blob/main/mem0/configs/prompts.py)) | Scoped by `agent_id`; keeps action results verbatim | A free-text summary as the procedure |
| **LangMem** | Procedural memory as prompt optimisation, offered as an optional export target | Prompt rewriting as the only form |
| **memorable.sh**; **Hyperspell** `/trace/add`; **Mubit** | Industry shape: trace in → generalised procedure with placeholders out; exact → lexical → semantic recall | Server-side embeddings; per-harness hooks only |
| **MemoryGraft** (2512.16962) | Poisoned experiences get replayed, so provenance and quarantine are mandatory | — |

## 3. Architecture overview

```
 capture sources                                   memory graph (edges table, see memory_graph_design.md)
 ─────────────────                                 ───────────────────────────────────────────────────
 Interceptor (openai/anthropic/litellm) ─┐
 OTel GenAI agent traces (ingest)  ──────┼─► L0 TraceCapture ─► turns (+tool events) ──part_of──► task
 engine.record_trace(...)          ──────┘        │ redaction, payload cap, provenance               │
                                                  ▼                                                  │
                                      TaskTracker ─► OutcomeLabeler (explicit > signal > LLM feedback > judge)
                                                  │ on task open: ProcedureMatcher (observe run)     │
                                                  ▼ on task close (background worker)               │
                         Compiler: scaffold ─► LLM synth (NIM) ─► validators ─► Assembler ─► graph write
                                                                                     ▲   (versioned, edit log)
                                         Learner: runs, adherence, Beta stats, lifecycle, validated edits
                                         Dreaming: batch re-compile, sub-procedures, staleness, insights
 read side:  procedures.recall(goal, tools) ─► candidates + verdict + pointer  (injection: later switch)
```

New package `memorable_ai/procedural/`:

```
capture.py     provider/OTel/import → TurnEvent (tool_call / tool_result / rationale); redaction; payload cap
tasks.py       TaskTracker: task segmentation over L0, late outcome updates
outcomes.py    OutcomeLabeler: explicit API, tool signals, LLM feedback classifier, LLM judge
scaffold.py    deterministic pre-pass: pairing, values, value-flow graph, sinks, binding candidates, side effects
compiler.py    scaffold → LLM synthesis → validators → ProcedureCandidate | Refusal
assembly.py    candidate ↔ existing procedures: alignment merge, branches, sub-procedures → EditProposal
store.py       versioned CRUD over procedure tables + edges; lifecycle; markdown/SKILL.md export
matcher.py     candidate generation, tool filter, calibrated verdict, active-node localisation
render.py      pointer rendering (slots filled from goal, memory bindings from the caller's subgraph)
learner.py     runs, adherence, Beta stats, staleness, edit validation
otel.py        OTel GenAI span ingest (execute_tool/invoke_agent) → TurnEvents
```

## 4. Data model

Every row carries `namespace`, `user_id` and **`agent_id`** (new on `Scope`; see the norms doc §2.2.4). Relationships between nodes live in the unified `edges` table ([memory_graph_design.md §3](memory_graph_design.md#3-schema)), not in per-feature edge tables.

### 4.1 L0 extension (`turns`)

| New column | Type | Notes |
|---|---|---|
| `kind` | str | `message` (default) \| `tool_call` \| `tool_result` \| `rationale` |
| `tool_name`, `tool_call_id` | str | pair a result with its call |
| `payload` | JSON/text | full args (call) or full result, redacted, up to `max_payload_bytes` (256 KB). Beyond that: head + tail + `sha256` + `length`, `truncated=1` |
| `values` | JSON | salient values (ids, numbers, quoted strings, JSON leaves) extracted **before** any cut, for value-flow attribution |
| `is_error` | bool | error flag, tracebacks, non-zero exit code |
| `task_id` | int | owning task (also an edge: `turn —part_of→ task`) |
| `source`, `trust_tier` | str | `user` \| `assistant` \| `tool` \| `import` \| `otel` (trust track) |

The dedupe hash includes `kind` and `tool_call_id`, so resent agent histories stay idempotent.

### 4.2 Node tables

```sql
tasks(id, namespace, user_id, agent_id, session_id, episode_id, parent_task_id,
  goal_text, intent_text, intent_hash, intent_embedding, keywords JSON,
  status, outcome, outcome_confidence, outcome_source,
  offered_tools JSON, used_tools JSON,
  started_at, ended_at, llm_calls, tool_calls, tokens_in, tokens_out, latency_ms)

procedures(id, namespace, user_id, agent_id, tier,          -- user | org (D2); user_id NULL for org
  promoted_from_id, promoted_at, promotion_evidence JSON,  -- governed promotion user → org
  title, intent_text, intent_hash, intent_embedding, keywords JSON, slots JSON,
  required_tools JSON, contraindications JSON,
  version, status,          -- candidate | provisional | active | stale | quarantined | deprecated
  trust_tier, source,       -- trace | otel | import | human
  compiled_by,              -- 'scaffold' | 'llm:<model>'
  success_count, failure_count, unknown_count, avg_tool_calls, avg_tokens, avg_latency_ms,
  last_used_at, last_verified_at, created_from_task_id, superseded_by_id, created_at, updated_at)

procedure_steps(id, procedure_id, version, seq, node_key,
  action_kind,              -- tool | verify | ask_user | llm
  tool_name, arg_template JSON, bindings JSON,
  precondition, postcondition, verify JSON,
  side_effect, requires_confirmation, optional, observed, observed_count)

procedure_pitfalls(id, procedure_id, node_key, text, observed_count, last_seen_at)
procedure_runs(id, procedure_id, version, task_id, mode,  -- observe | injected | holdout
  matched_by, match_score, margin, verdict, adherence, deviation_node, outcome,
  tool_calls, tokens, latency_ms, created_at)
procedure_edits(id, procedure_id, from_version, to_version, kind, diff JSON,
  status, evidence JSON, reason, created_at)
```

### 4.3 The procedure subgraph (in `edges`)

| Edge | From → to | Purpose |
|---|---|---|
| `part_of` | step → procedure; turn → task | structure |
| `next` / `branch` / `alt` (+ condition, counts) | step → step | control flow, with per-edge success statistics |
| `depends_on` | step → step | value flow or prerequisites (TraceCompiler, Goal-Oriented Graphs) |
| `subprocedure` | step → procedure | modular reuse (LEGOMem) |
| `uses_tool` | step → entity(type=tool) | tool-set filtering, "what uses this tool", staleness when a tool disappears |
| `operates_on` | procedure/step → entity | entity-centred recall ("payments" → the deploy procedure) via PPR |
| `instance_of` | task → procedure | evidence: every run is an edge with an outcome |
| `derived_from` | procedure (version) → task(s) | provenance; cascade on forget; trust taint |
| `pitfall_of` | pitfall → step | failures stay attached to their step |
| `supports` / `contradicts` | task → procedure | the validated-edit evidence base |

Procedures are **versioned like facts**. An accepted edit writes `version+1` steps and edges (old ones are invalidated, not deleted), so `history(procedure_id)` comes for free.

## 5. Capture

| Source | Tool call | Tool result | Rationale |
|---|---|---|---|
| OpenAI chat | `assistant.tool_calls[]` | `role:"tool"` + `tool_call_id` | assistant `content` next to tool_calls |
| Anthropic | `tool_use {id,name,input}` | `tool_result {tool_use_id, content, is_error}` | `text` blocks before `tool_use` |
| OpenAI Responses | `function_call` + built-in `*_call` items | `function_call_output` | `message` items / reasoning summaries |
| LiteLLM | OpenAI shape | OpenAI shape | OpenAI shape |
| **OTel GenAI** | `execute_tool` span: `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments` | `gen_ai.tool.call.result`, error status | `chat` spans' output messages under the same `invoke_agent` |
| `record_trace()` | `steps[].tool/args` | `steps[].result/is_error` | `steps[].rationale` |

- **Offered tools** (`tools=` in the request, or `gen_ai.tool.definitions` in OTel) are captured per call into `tasks.offered_tools`.
- **Retention follows D3:**
  - Secrets are always redacted.
  - PII redaction can be `off`, `emails_phones` or `strict`.
  - Payloads are capped at `max_payload_bytes`.
  - `retention.raw_days` sets a TTL; the default is keep.
  - Deletion cascades through `derived_from` edges.
- **Code to change:**
  - `normalize_messages` currently drops tool messages.
  - The Responses adapter drops `function_call` items.
  - Adapters need `get_tools(kwargs)`.

## 6. Tasks and outcomes

**Segmentation** (`TaskTracker`):
- A task opens at a user message that is followed by at least one tool call.
- It closes at a tool-free assistant answer, at the next user message, or after the idle gap.
- An anaphoric follow-up ("now do the same for auth") opens a child task with `parent_task_id`.
- Explicit override: `with engine.task("…") as t:`.

**Outcomes**, from most to least trusted:

| Source | Signal | Confidence |
|---|---|---|
| explicit | `t.success()`, `t.failure(reason)`, `engine.mark_outcome(...)` | 1.0 |
| signal | exit codes, test summaries, HTTP status, `is_error`, tracebacks, a verify step's postcondition | 0.8 |
| feedback | the next user turn, classified by the **LLM** (on by default now that D5 removed the cost constraint), with patterns as fallback | 0.6 |
| judge | LLM self-judgement over the whole trace; never enough alone to make a procedure `verified` | 0.4 |

Outcomes can arrive late; statistics are replayed when they do. A task with no signal stays `unknown`. It can seed a `candidate` procedure but can never verify one.

## 7. Compilation: scaffold → LLM → validators

`ProcedureCompiler.compile(task) -> ProcedureCandidate | Refusal`, run on the background worker after a task closes.

### 7.1 Scaffold (deterministic, always runs)
1. **Pair** each tool call with its result. A step is (tool, args, result values, is_error, preceding rationale).
2. **Value-flow graph** (TraceCompiler): an edge `i → j` when a value from step *i*'s result appears verbatim, or after a known transform, in step *j*'s args or in the final answer. Values must be at least 3 characters, not stopwords, and not in the goal text.
3. **Sinks:** the final answer, successful side-effecting steps, and verify steps. **Keep** the steps that reach a sink; mark the others `pruned`, so the LLM sees them but they never become steps.
4. **Failed-then-fixed:** when a failing call to X is later followed by a successful call to X, the failure becomes a pitfall candidate plus the argument diff.
5. **Binding candidates** for each kept argument:

| Kind | Rule | Template |
|---|---|---|
| `copied_output` | equals a value from an earlier kept step's result | `{steps.<node>.<path>}` |
| `user_input` | appears in the goal text (a slot) | `{goal.<slot>}` |
| `memory` | equals the value of an active L2 fact for this user | `{memory:<subject>.<predicate>}`, a **pattern** resolved in the caller's own subgraph at recall time (makes D2 sharing safe) |
| `transform` | a simple function of the above | `{transform(...)}` |
| `constant` | a literal; never a secret or PII | literal |
| `llm_decision` | none of the above | `<decide>` |

6. **Side effects:** from tool names (create, update, delete, send, deploy, write, merge, pay…) and from the tool schema descriptions in `offered_tools`.
7. **Scan** for secrets and PII in every literal.

### 7.2 LLM synthesis (NIM; one call per closed successful task, batched in dreaming when backlogged)

**Input:**
- the goal
- the offered tool schemas
- the trace with kept/pruned marks and values
- the scaffold's binding candidates and side-effect flags
- the referenced user facts
- the top 3 similar existing procedures, as merge hints

**Output (JSON):**
- `title`, `intent_template`, `slots`
- `steps[]`: each step has either `from_call` (an index into the trace) or `inferred:true`, plus `tool`, `arg_template`, `precondition`, `postcondition`, `verify`, `optional`
- `edges[]`: branch / alt with conditions
- `pitfalls[]`, `contraindications[]`, `required_tools`
- `merge_with`: a procedure id or null

### 7.3 Validators (deterministic; the LLM never writes to the graph directly)
- **Steps:**
  - Every `from_call` step must reference a *kept* call with the same tool name.
  - `inferred` steps are stored with `observed=False`.
- **Templates and tools:**
  - Templates may only reference declared slots, earlier steps' outputs, memory patterns that resolve for this user, or `<decide>`.
  - `required_tools` must be a subset of the tools actually used.
- **Safety:**
  - Literals must pass the secret/PII scan. A leak becomes a slot, or the candidate is rejected.
  - A side-effecting step with a `<decide>` argument gets `requires_confirmation=True`. In strict mode the compile is refused.
  - If tool-sourced content flowed into side-effect arguments (provenance taint), the procedure is `quarantined`.
- **On validation failure:** one retry that includes the validator's errors. If that also fails, fall back to the **scaffold-only candidate** (`compiled_by='scaffold'`), so a procedure is always produced deterministically.
- **Failure traces** never create procedures. They only add pitfalls and `contradicts` evidence to the procedure they matched.

## 8. Matching and recall (an API now; injection later)

**Candidate generation** (cheap and precise first):
1. Exact `intent_hash`.
2. BM25 over intent, keywords and tool names.
3. Dense over `intent_embedding`.
4. **Graph:** typed PPR from the goal's entities through `operates_on` / `uses_tool`, plus the three-tier walk: similar past tasks → `instance_of` procedures ([graph doc §5](memory_graph_design.md#5-read-path)).

Fused with RRF.

**Filters:**
- `required_tools ⊆ tools`, when the caller passes the tools (the interceptor always knows them)
- contraindications don't match
- status is `active` or `provisional`
- trust is not `untrusted`
- the tier is visible to the caller (D2): the caller's own `user` procedures plus `org` procedures. A private procedure ranks above its org counterpart when both match. `agent_id` affinity is a ranking feature

**Verdict** (calibrated, not raw RRF):

| Verdict | Condition (defaults) |
|---|---|
| `confident` | exact match, or cos ≥ θ (0.75, per embedder) with slot-stripped keyword overlap ≥ 0.5; margin over the runner-up ≥ 0.05; Beta(1+s, 1+f) 10th percentile ≥ 0.5 |
| `ambiguous` | a candidate passes the filters but fails the margin or the posterior check |
| `none` | nothing passes |

**Observation runs.** At task open, the matcher runs automatically and logs `procedure_runs(mode='observe')` with the candidates and verdict. At task close it computes:
- **adherence:** the LCS of the executed tool calls against procedure nodes, checking binding consistency
- the deviation node
- the outcome

When a natural trace follows a procedure and succeeds, that is evidence at zero risk. It also produces exactly the logged data needed to calibrate thresholds when evals resume.

**Rendered pointer** (50–300 tokens), returned by `recall()` for callers who want it. It is injection-safe: no raw tool output, values escaped, framed as advisory.

```
<procedure id="17" v="3" runs="12/13 ok" verified="2026-11-10" note="advisory; adapt if the situation differs">
Goal: deploy {service=payments} to {env=staging}
1. run_tests(service=payments)              expect: "0 failed"
2. build_image(service=payments)            → tag
3. deploy(service=payments, env=staging, tag={tag}, region=eu-west-1 [your setting])
Verify: get_status(service=payments, env=staging) == "healthy"
Pitfall: deploy fails if the image tag was not pushed; push before deploy.
</procedure>
```

**Injection later:** `procedural.inject = off` (now) | `advise` | `shadow`, with an optional holdout. It plugs into `MemoryEngine._aprepare_call` and the L4 budget. It ships only after evals resume, at which point the observation-run log already holds the data to tune it.

## 9. Assembly: one graph, not a pile of near-duplicates

A new candidate is compared with existing procedures that share `intent_hash`, have `required_tools` Jaccard ≥ 0.6 and intent cosine ≥ τ, or that the LLM flagged in `merge_with`.

- **Merge:**
  - Align step sequences (Needleman–Wunsch over `tool_name + arg-shape`).
  - Matched steps get `observed_count++`.
  - Gaps become `optional` steps.
  - Divergences become `branch` / `alt` edges with conditions.
  - A constant that differs between traces becomes a slot.
- **Sub-procedures:** a prefix shared by at least m procedures is extracted into its own procedure and linked with `subprocedure` edges.
- **Every structural change is an `EditProposal`.** Nothing changes the active version without validation (§10).

## 10. Learning loop

1. **Runs and statistics:**
   - Beta posteriors per procedure and per `branch` edge, from observation runs and explicit outcomes.
   - Cost per run: tool calls, tokens, latency.
2. **Lifecycle:**
   - candidate → provisional: at least 1 success with a signal or explicit outcome
   - provisional → active: at least k verified successes and the posterior lower bound ≥ p_min
   - active → stale: the lower bound drops, N consecutive failures, or a `uses_tool` target disappears from offers
   - stale → active again (re-verified) or → deprecated
   - any state → quarantined (trust)
3. **Validated edits** (Procedural Graphs, test-gated promotion):
   - An edit is accepted only if replaying historical `instance_of` evidence against the edited graph shows success no lower than the current version, with a minimum sample size.
   - Rejected edits are kept with a reason, so they aren't proposed again.
4. **Pitfalls** come from failed-then-fixed steps and from failed matched runs. They are rendered only with their step.
5. **Fact dependency:** when an L2 fact a memory binding resolves to is superseded, recall re-resolves it automatically. Procedures whose `operates_on` entity was invalidated are flagged for review.

## 11. Integration with existing layers

| Layer | Change |
|---|---|
| Scope | `agent_id` added (`Scope(namespace, user_id, agent_id, session_id)`); the contextvar and derivation follow the user_id pattern |
| L0 `turns` | tool events, full redacted payloads, `task_id`, `source`/`trust_tier`; suffix-dedup covers tool events |
| L1 | `tasks` table; `task —part_of→ episode`; episode summaries mention task outcomes |
| L2 facts + graph | memory-binding patterns resolve against `relates` edges; procedures are reachable via PPR |
| L4 | none for now; procedure rendering is reached through `recall()` |
| Pipeline | `WritePipeline.ingest` → `TaskTracker.on_turns` → on close: label → compile → assemble → learn (background) |
| Interceptor | `Adapter.get_tools(kwargs)`; the observation run opens in `_aprepare_call` and closes in `_record_call` |
| Integrations | OTel ingest (`otel.py`), `record_trace`, and the MCP tools `recall_procedures` / `get_procedure` |
| Consolidation | dreaming: batch re-compile, sub-procedure extraction, edit validation, staleness |

## 12. Public API (proposed)

```python
engine = MemoryEngine(..., procedural=True)

with engine.scope(user_id="alice", agent_id="ops-bot", session_id="ops-7"):
    run_my_agent("Deploy payments to staging")         # captured, tasks segmented, matched in observe mode

with engine.task("Deploy payments to staging") as t:   # optional explicit boundaries and outcome
    t.success() if run_my_agent(...).ok else t.failure("health check failed")

engine.mark_outcome(success=True)

await engine.record_trace(goal="...", steps=[{"tool": "run_tests", "args": {...}, "result": "...", "is_error": False}],
                          outcome="success", user_id="alice", agent_id="ops-bot")
await engine.ingest_otel(spans)                         # OTel GenAI spans → traces

hits = engine.procedures.recall("deploy auth to prod", tools=["deploy", "run_tests"])
# → {"verdict": "confident", "candidates": [...], "pointer": "<procedure ...>"}

engine.procedures.list(status="active"); engine.procedures.get(17); engine.procedures.history(17)
engine.procedures.runs(17); engine.procedures.quarantine(17)
engine.procedures.promote(17)  # user → org (admin approval path); auto-promotion uses the evidence rule
engine.procedures.demote(42)   # org → back to private copies; recorded as an edit
engine.procedures.export_markdown(17)                   # SKILL.md-style
```

Config (`memory.procedural`): `enabled`, `pii_redaction`, `max_payload_bytes`, `promote_min_users`, `auto_promote`, `compile_model`, `strict_side_effects`, `min_verified_runs`, `p_min`, `theta_dense`, `margin`, `inject` (default `off`), `holdout_rate`. The root `retention` block has `raw_days`.

## 13. The other tracks

| Track | Where | Order |
|---|---|---|
| **Knowledge graph** | [memory_graph_design.md](memory_graph_design.md): unified `edges`, ontology, entity resolution, typed PPR, cascade lifecycle | **G0 first:** procedures are stored in it |
| **Trust tiers** | `source` / `trust_tier` on turns, facts, episodes, procedures and edges; taint is the minimum over `derived_from` ancestors | Columns in G0/P1; propagation in G5 |
| **Backends** | `GraphStore` (SQL / Neo4j), `VectorIndex` (numpy / sqlite-vec / pgvector), `Embedder` (`encode_query` / `encode_document`; rows record `embedding_model` for re-indexing) | In parallel |
| **Dreaming** | Graph jobs: communities, insights, fact re-verification, predicate canonicalisation, procedure re-compile and sub-procedures, staleness | After P4 |
| **Integrations** | OTel emit/ingest, MCP server, native adapters ([norms §2.3](industry_norms_2026-11.md#23-recommended-order)) | OTel ingest with P1; the rest in parallel |

One rule runs through all of them: **every derived memory (fact, summary, insight, procedure) is a versioned node with `derived_from` edges to its evidence. A background job can only change it through a validated, logged edit.**

## 14. Combined roadmap

| Step | Deliverable | Testable by |
|---|---|---|
| 1. **G0** | `edges` + `GraphStore`(SQL), migration of relations/mentions, provenance edges, `agent_id` on `Scope` | existing 113 tests green; provenance and cascade tests |
| 2. **P1 Capture & tasks** | tool events in L0 (all providers + OTel ingest + `record_trace`), redaction and payload cap, offered tools, `TaskTracker`, `OutcomeLabeler`, trust columns. **P1a done (2026-11-16):** tool events in L0 for OpenAI chat / Anthropic / Responses / LiteLLM, redaction, payload cap, values, `source` column. **P1b done (2026-11-16):** `tasks` table, `TaskTracker` segmentation, explicit `engine.task()`, outcomes (explicit / signal / feedback patterns), offered and used tools. **P1c in progress:** `record_trace` and OTel GenAI ingest done; next LLM feedback classification, streaming tool calls | mock-transport multi-call agent loops; OTel span fixtures |
| 3. **G1** | ontology, LLM extraction with the registry, entity resolution, entity summaries | mocked-LLM tests |
| 4. **P2 Compiler & store** | scaffold, LLM synthesis, validators, fallback, procedure subgraph, versioning, export | golden traces → expected procedures; validator rejection cases |
| 5. **P3 Matching & recall** | candidates (lexical / dense / graph), filters, verdict, observation runs, adherence | verdict unit tests; end-to-end loops |
| 6. **G2** | typed multi-layer PPR (facts / episodes / procedures), entity cards, `as_of` | retrieval tests |
| 7. **P4 Assembly & learning** | merge, branches, sub-procedures, Beta statistics, lifecycle, validated edits | multi-trace fixtures |
| 8. **G4 / dreaming**, **G5 lifecycle** | communities, insights, re-verification; cascade forget, taint | mocked-LLM and cascade tests |
| Later | injection + holdout; governance UI | when evals resume |

Evaluations stay paused. Observation runs and the edit log record everything an evaluation will need.

## 15. D2: who sees a procedure (decided)

**Decision:** private per user **and** shared across the whole app, as two tiers with governed promotion between them. This is the enterprise pattern ([norms §3](industry_norms_2026-11.md#3-d2-how-enterprises-scope-shared-memory)):
- GitHub Copilot Memory is shared across all of Copilot's agents for a repository, only contributors with write access can create it, and it is validated against citations before use.
- Enterprise guidance converges on private → team → org scopes, with restricted writes to the broader ones.

| Tier | Contents | Who reads it |
|---|---|---|
| `user` | Everything compiled from a user's own sessions | That user, on any agent |
| `org` | Generalised procedures promoted from `user` | Every user and every agent in the app (namespace) |

**Example:** `support-bot` learns "refund: `lookup_order` → `check_policy` → `issue_refund` → `send_email`" in Alice's session.
1. It is stored in Alice's `user` tier at once, so it helps her next time.
2. When Bob's session independently succeeds with an aligned trace (2 distinct users), or an admin approves, the merged version is **promoted** to `org`. From then on Bob, Carol, and a different `billing-bot` that offers the same tools can all recall it.

**Promotion rules:**
- Values must be generalised:
  - Alice's order id and email are **slots** (`{goal.order_id}`).
  - Her personal settings are **memory patterns** (`{memory:self.preferred_contact}`) that resolve in each caller's own subgraph.
  - Validators reject literals that look like secrets or PII.
- **Evidence** from at least `promote_min_users` (2) distinct users, or `approve()` by an admin.
- The procedure must not be quarantined, and its trust is the minimum over all source traces.
- `procedure_edits` records each promotion with evidence, so it can be undone with `demote()`.

**Why "all agents" doesn't hurt:** the concern was that skills don't transfer between roles (−7.5 in Managing Procedural Memory). That is handled at retrieval, not by walling procedures off:
- `required_tools ⊆ offered_tools` means an agent never sees a procedure it can't run.
- `agent_id` affinity is a ranking feature.
- Per-agent statistics are kept on `instance_of` edges, so a procedure that works for `ops-bot` but fails for `support-bot` is down-ranked for the latter only.

An application can pin a procedure to `user` (`tier_lock`), or add a `team` tier later via `group_id`.

## 16. Risks

| Risk | Mitigation |
|---|---|
| **Bad labels → bad procedures** | the outcome hierarchy; `unknown` never verifies; observation runs before anything is trusted |
| **LLM compile hallucination** | `from_call` grounding; validators; inferred steps marked; scaffold fallback; validated edits |
| **Precision at scale** | tool filter, calibrated verdict with margin, partition by tool set, sub-procedures, deprecation; logged decisions make thresholds tunable |
| **Stale procedures** | verify postconditions; auto-stale on failures or a missing tool; memory patterns re-resolve; `operates_on` invalidation flags |
| **Poisoning** | provenance taint along `derived_from`, quarantine, injection-safe rendering, advisory framing |
| **Storage growth from full payloads** | 256 KB cap, `retention.raw_days`, cascade deletion; payload columns can move to a blob store later (backends track) |
| **Privacy across users** | private by default; promotion to `org` only after generalisation and evidence from at least 2 users or approval; slots, memory patterns, secret/PII scan; `tier_lock`; reversible `demote()` |
| **Hot-path latency** | matching is SQL + index lookups (target < 20 ms); compilation and the LLM run on the background worker |

## 17. Lessons adopted from billion-context (November 2026)

[billion-context](https://github.com/ranxianglei/billion-context) (MIT) manages a coding agent's context **within** one session: model-driven, incremental, tiered summaries that can be restored on demand. It solves a different problem from cross-session memory, so it is neither a dependency nor a competitor. Its benchmark claims are self-reported. Five of its ideas carry over:

| Lesson | What we do with it | Status |
|---|---|---|
| **Never rewrite the prefix.** Its fold keeps the provider's prompt cache valid (94% cache-read share) by changing only the tail. | The interceptor now puts memory in front of the latest user message (`memory.injection="tail"`, the default). The memory block is reused while that message is unchanged (`context_reuse_seconds`), so an agent's tool-loop calls send identical bytes. Tests check the conversation prefix stays byte-identical across turns. | Shipped |
| **Every summary is reversible.** Each compressed block can be decompressed to its source text. | `MemoryEngine.expand("episode" \| "fact", id)` returns the source turns, within the caller's scope. Task summaries and procedures (§7) must carry source refs, so this works for them too. | Shipped (episodes, facts); P2 for procedures |
| **Compress each increment once, never re-summarize history.** Re-summarizing everything each round makes early content decay geometrically (ρ^k). Folding each piece once keeps retention constant. | Episodes already close once. Future roll-ups (episode → session → topic, G4) fold *summaries* and never re-read raw history. A turn is summarized exactly once in its life. | Design rule |
| **Compress between task steps, not mid-step** (its growth-gated nudge). | `TaskTracker` (§6) never closes or summarizes a task inside a tool burst. It closes at a tool-free answer, the next user message, or the idle gap, as specified. | P1 |
| **Models rarely call recovery tools unprompted** (its own pilot finding). | Keep automatic injection as the default. If an `expand` tool is exposed to agents, the injected block names it explicitly. | Design rule |

**Task-summary spec for P1/P2.** Adapted from billion-context's compression doctrine (`acp-kernel/DESIGN-prompts.md`).
- **Keep verbatim:**
  - file paths with line numbers
  - signatures and load-bearing code lines
  - exact error strings
  - decisions *with their rationale*
  - exact values
  - how the goal evolved
  - source turn ids (for `expand`)
- **Drop:** verbose logs once their result is extracted, duplicate reads, and the step-by-step path to a conclusion.
- **Dead ends:** drop the exploration but keep the lesson; it becomes a `pitfall` (§7.1, failed-then-fixed).
- **When the budget is tight, keep in this order:** goal, then decisions with rationale, then artifacts, then conclusions, then lessons.

