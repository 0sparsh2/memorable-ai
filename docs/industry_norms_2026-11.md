# Industry norms, November 2026: raw-data retention and integration surfaces

**Purpose:** answers three decisions from [procedural_memory_design.md](procedural_memory_design.md):
- **D3:** how much raw tool and trace data to keep.
- **D4:** which integration surfaces to support.
- **D2:** how shared memory is scoped (§3).

The answers come from what other memory systems and standards actually do, plus an audit of this repo.
**Date of research:** 2026-11-15. Vendor behaviour changes quickly, so each row names its source.

---

## 1. D3: what other systems keep

| System | Raw source kept? | Tool payloads | Retention | Derived layer |
|---|---|---|---|---|
| **Zep / Graphiti** | Yes. Each episode node stores the raw content as ingested, and derived edges cite the episodes behind them ([Zep provenance](https://blog.getzep.com/how-zep-tracks-provenance-in-agent-memory/), [graph overview](https://help.getzep.com/graph-overview)) | Whatever is ingested as an episode | No default TTL documented. For retention / right to be forgotten: delete a fact only when no remaining episode supports it ([talk](https://ai.engineer/talks/citation-needed-provenance-for-llm-built-knowledge-graphs)) | Entity/edge graph, communities |
| **Supermemory** | Yes. Documents and chunks are kept next to the extracted memories: "derive memories and keep chunks so you can personalize *and* ground" ([how it works](https://supermemory.ai/docs/concepts/how-it-works)) | n/a (document model) | Not documented | Memories, made in a "dreaming" phase |
| **Mem0 (OSS)** | Facts only by default. Raw is kept only with `infer=False`; a history table records ADD/UPDATE/DELETE ([graph memory docs](https://docs.mem0.ai/open-source/features/graph-memory)) | The procedural-memory prompt requires each agent action's output **verbatim**: "Action Result (Mandatory, Unmodified)" ([prompts.py](https://github.com/mem0ai/mem0/blob/main/mem0/configs/prompts.py)) | — | Facts, procedural summaries (scoped by `agent_id`) |
| **Letta** | Yes. The full message history, including tool-return messages, stays searchable after eviction from context ([messages API](https://docs.letta.com/api/resources/agents/subresources/messages/index.md), [forum](https://forum.letta.com/t/how-can-i-recover-the-complete-message/78)) | Stored as messages | — | Core blocks, archival memory |
| **LangGraph / LangMem** | Yes. The checkpointer keeps the full thread state, tool messages included. `BaseStore` holds long-term items ([LangMem launch](https://blog.langchain.com/langmem-sdk-launch)) | Full | Backend-defined | Store items. Procedural memory means **rewriting the system prompt** |
| **OpenAI Agents SDK** | Yes. A `Session` stores conversation items, tool calls and outputs included (SQLite, Redis, SQLAlchemy, Conversations API) ([sessions](https://openai.github.io/openai-agents-python/sessions/)) | Full | Backend-defined | None built in |
| **AWS Bedrock AgentCore Memory** | Yes. Short-term memory is raw events; long-term records are extracted asynchronously by strategies (semantic, summary, user preference, episodic) ([how it works](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/how-it-works.md)) | Stored as events | **Configurable, up to 365 days** ([create memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-create-a-memory-store.html)) | Strategy records |
| **Google Memory Bank (ADK)** | Yes. Session events are the source; memories are generated in the background, scoped by user ([overview](https://docs.cloud.google.com/agent-builder/agent-engine/memory-bank/overview)) | Stored as events | — | Memories |
| **Anthropic memory tool** | The developer hosts the files (`/memories`), and Claude decides what to write. Context editing (`clear_tool_uses_20250919`) drops old tool results from the **context**, not from storage ([memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool), [context management](https://www.anthropic.com/news/context-management)) | Not stored by the tool | Developer-defined | Files |
| **memorable.sh (YC)** | The exception, because it is a hosted service. "The transcript is never sent"; only the prompt and allow-listed tool arguments leave the machine ([memorable.sh](https://memorable.sh)) | Not sent | — | Procedures in your own store |
| **OpenTelemetry GenAI semconv** (observability standard) | n/a | `gen_ai.tool.call.arguments` / `.result` are **Opt-In** and marked "may contain sensitive information". So are message content, `gen_ai.memory.query.text` and `gen_ai.memory.records` ([execute-tool spec](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/execute-tool.md)) | — | — |

### The pattern
1. **Memory systems keep a lossless raw layer and derive memory from it:** Zep, Supermemory, Letta, LangGraph, the Agents SDK, AgentCore and ADK all do this. Even Mem0's procedural summaries keep tool outputs verbatim.
2. **Retention is handled with a TTL, not by truncating at write time.** AgentCore exposes this as a setting.
3. **Content is opt-in only when it leaves the store,** for example telemetry (OTel) or a hosted service (memorable.sh).
4. **The agent's context gets pruned; storage doesn't** (Anthropic context editing).

### Decision for Memorable (replaces the earlier "2 KB head/tail" recommendation)
- **Payloads:** L0 keeps **full tool payloads** in the user's own database.
- **Redaction:**
  - Secrets (API keys, tokens, passwords, private keys) are **always** redacted before storage.
  - PII redaction is configurable (`off | emails_phones | strict`).
- **Size cap:** a high per-payload cap (`max_payload_bytes`, default 256 KB) protects the database, not privacy. Above the cap, store head + tail + sha256 + length and mark the row `truncated`. Salient `values` are extracted before cutting.
- **Retention:** `retention.raw_days`, default `None` (keep, the Zep/Letta/LangGraph norm). Set it to get AgentCore-style expiry.
- **Deletion cascades through provenance edges** (Zep rule): a derived node with no remaining support is deleted or invalidated. See [memory_graph_design.md](memory_graph_design.md) §6.
- **Telemetry:** OTel export of content is opt-in, following the spec.

---

## 2. D4: integration surfaces

### 2.1 What this repo has today

| Surface | File | State | Verdict |
|---|---|---|---|
| SDK interceptor: OpenAI chat + Responses, Anthropic Messages, LiteLLM; sync, async and streaming | [interceptor.py](../memorable_ai/core/interceptor.py), [providers.py](../memorable_ai/core/providers.py) | Rebuilt in v0.2, 12 mock-transport tests. Also catches LangChain, CrewAI and others, because they call these SDKs internally | **Keep; this is the differentiator** |
| Shims `openai.py`, `anthropic.py`, `litellm.py` | [integrations/](../memorable_ai/integrations/) | Thin pointers to the interceptor | Keep |
| LangChain callback | [langchain.py](../memorable_ai/integrations/langchain.py) | Problems:<br>• `on_chat_model_start` cannot change the prompt, so the "injected" context is discarded<br>• fire-and-forget `create_task`<br>• tool messages flattened away<br>• **double-captures** when the interceptor is also on | Replace with a LangGraph `BaseStore` adapter |
| AutoGen bridge | [autogen.py](../memorable_ai/integrations/autogen.py) | Calls compatibility helpers. AutoGen has been in maintenance mode since Q1 2026; Microsoft Agent Framework 1.0 replaced it ([report](https://newclawtimes.com/articles/langgraph-crewai-autogen-2026-framework-platform-comparison/)) | Replace with a Microsoft Agent Framework adapter |
| CrewAI helper | [crewai.py](../memorable_ai/integrations/crewai.py) | Wraps `search_memories` into a string; works around event loops with a thread pool | Replace with an adapter for CrewAI's native memory |
| Mem0 adapter | [mem0_adapter.py](../memorable_ai/integrations/mem0_adapter.py) | Wraps the Mem0 cloud client; not wired into anything | Remove, or turn into a one-off importer |
| NVIDIA NIM | [nvidia_nim.py](../memorable_ai/integrations/nvidia_nim.py) | Model registry; `DEFAULT_MEMORY_MODEL = nvidia/nemotron-3.5-lightning-30b-a3b` (deepseek-v4-flash retired 2026-08-07) | **Default LLM for the memory pipeline** (D5) |
| Connectors | [connectors/base.py](../memorable_ai/connectors/base.py) | Stub | Later |
| **Missing** | — | • `agent_id` scope<br>• MCP server<br>• OTel spans (emit or ingest)<br>• framework-native memory interfaces<br>• trace import | See §2.2 |

### 2.2 The standards in practice

1. **OpenTelemetry GenAI semantic conventions.** Status is Development; they moved to their own repo with v1.42 ([repo](https://github.com/open-telemetry/semantic-conventions-genai)).
   - **Agent spans:** `invoke_agent`, with `chat` and `execute_tool` children. Tool spans carry `gen_ai.tool.name`, `gen_ai.tool.call.id`, and opt-in arguments and results.
   - **Memory spans:** `create_memory`, `search_memory`, `update_memory`, `upsert_memory`, `delete_memory`, `create_memory_store`, `delete_memory_store`, with attributes `gen_ai.memory.store.id`, `gen_ai.memory.record.id` and `gen_ai.memory.record.count`. AWS AgentCore and Google ADK instrumentation already emit them ([memory span report](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/reference/reports/memory-span.md)).
   - **For us:**
     - (a) Emit memory spans for every engine operation.
     - (b) **Accept OTel agent traces as a capture source** for procedural memory. Any OTel-instrumented framework then becomes a trace source with no per-framework code.
2. **MCP.** For agent clients, memory is usually reached through an MCP server.
   - The official reference memory server is itself a **knowledge graph** of entities, relations and observations ([README](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)).
   - memorable.sh ships a read-only MCP server with five tools.
   - **For us:** a `memorable mcp` server with graph-aware read and write tools.
3. **Framework-native memory interfaces.** Each is a thin adapter that writes through the same pipeline:
   - LangGraph `BaseStore` (LangMem builds on it)
   - OpenAI Agents SDK `Session` protocol
   - Google ADK `MemoryService`
   - Microsoft Agent Framework
   - Claude memory tool (client-side backend)
4. **Scoping dimensions.** Every major API scopes below "user":
   - Mem0: `user_id`, `agent_id`, `run_id`; procedural memory *requires* `agent_id`
   - AgentCore: actor + session
   - ADK: app + user + session
   - OTel: `gen_ai.agent.id`

   **For us:** add `agent_id` to `Scope`.
5. **Background extraction from raw events** (AgentCore strategies, Memory Bank generation, Supermemory dreaming) is the norm. Our `WritePipeline` and background worker already follow it.
6. **Coding-agent harness hooks** (memorable.sh: Claude Code, Codex and Cursor hooks; `POST /v1/extract`). **For us:** `record_trace()` import, then a Claude Code transcript importer.

**Adoption data is thin.** The only independent figure found is a Stack Overflow pulse survey from April 2026 (about 1,100 developers): LangChain 22%, OpenClaw 17%, LangGraph 14% ([summary](https://the-agent-report.com/2026/07/ai-agent-frameworks-comparison-2026-langgraph-crewai-autogen/)). CrewAI's "60% of the Fortune 500" figure is self-reported.

### 2.3 Recommended order
1. Interceptor (done).
2. `agent_id` scope plus OTel memory spans (emit) and OTel agent-trace ingest.
3. MCP server.
4. LangGraph `BaseStore` and OpenAI Agents `Session` adapters.
5. Claude memory-tool backend, ADK `MemoryService`, Microsoft Agent Framework.
6. Harness importers (Claude Code transcripts).

Retire the legacy LangChain, AutoGen, CrewAI and Mem0 shims once their replacements exist.

---

## 3. D2: how enterprises scope shared memory

| System / source | Scope model | Who may write to the shared scope | Safeguards |
|---|---|---|---|
| **GitHub Copilot Memory** ([docs](https://docs.github.com/en/copilot/concepts/agents/copilot-memory), [changelog](https://github.blog/changelog/2026-01-15-agentic-memory-for-github-copilot-is-in-public-preview/)) | **Repository-scoped and shared across Copilot features and agents.** What the coding agent learns, code review uses too | Contributors with write permission | Every memory carries **citations, validated against the current code before use**; unused memories are deleted after 28 days |
| **Claude Code memory** | Hierarchy: enterprise managed policy → project (shared through the repo) → user → local | Admins (policy), repo contributors (project) | Precedence order; plain files under review |
| **Mem0** | `org` / `project` → `user_id` / `agent_id` / `run_id` | API key holders | Scoped filters |
| **memorable.sh** | One store shared by many agents; a procedure one agent records, others recall | Agents with `memorable enable` | Opt-in writes |
| **Enterprise guidance** ([private / team / org / customer scopes](https://www.mintmcp.com/blog/scopes-agent-memory), [team sharing without exposing private data](https://www.mindstudio.ai/blog/share-ai-agent-memory-across-team), [governed shared memory, arXiv 2606.24535](https://arxiv.org/html/2606.24535v1)) | Private → team → org (→ customer) | **Restricted**: admin-only, or a review / confidence threshold before committing | Isolation enforced at the data layer; private entries never cross users without an explicit policy; offboarding follows data retention |

**The pattern:** private per user *plus* shared across the whole app or org, with **governed promotion** into the shared tier and **validation at use time**. Agents aren't walled off from each other; Copilot shares memory across its agents on purpose.

**Decision for Memorable:**
- `user` + `org` tiers (an optional `team` tier later).
- Session-derived memory lands in `user`.
- Promotion to `org` requires PII-safe generalisation and either evidence from at least 2 distinct users or admin approval.
- At use time: verify postconditions, re-check provenance, and expire unused memory.
- `agent_id` is a ranking and filter attribute; the tool-availability filter stops cross-role misuse.

Details: [memory_graph_design.md §3.4](memory_graph_design.md#34-scope-tiers-in-the-graph-d2-decided) and [procedural_memory_design.md §15](procedural_memory_design.md#15-d2-who-sees-a-procedure-decided).
