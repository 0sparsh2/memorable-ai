# Memorable

<div align="center">

**The First Unified Memory System Combining Interceptor-Based Integration, Research-Backed Techniques, and Graph Architecture**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Zero-code integration • Research-validated • Production-ready*

[Quick Start](#quick-start) • [Documentation](docs/) • [Examples](examples/) • [Contributing](CONTRIBUTING.md)

</div>

---

## What is Memorable?

Memorable is the first memory system that combines the best of three leading approaches:

- **Memori's** zero-code interceptor architecture for seamless integration
- **Mem0's** research-validated memory techniques from academic papers
- **Supermemory's** graph-based architecture for multi-hop reasoning

**Result**: A production-ready memory system that requires no code changes, works with any SQL database, supports 100+ LLM models, and includes optional graph capabilities.


## Quick Start

### Installation

**From PyPI (Recommended):**
```bash
pip install memorable-ai
```

**From GitHub:**
```bash
pip install git+https://github.com/0sparsh2/memorable-ai.git
```

**From Source:**
```bash
git clone https://github.com/0sparsh2/memorable-ai.git
cd memorable-ai
pip install -e .
```

**With Optional Dependencies:**
```bash
# With graph support
pip install memorable-ai[graph]

# With development dependencies
pip install memorable-ai[dev]
```

### 30-Second Example

```python
from memorable_ai import MemoryEngine
from openai import OpenAI

# Initialize and enable (that's it!)
memory = MemoryEngine(database="sqlite:///memory.db", mode="auto")
memory.enable()

# Your existing code works unchanged!
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "I'm building a FastAPI project"}]
)

# Later conversation - memories automatically injected!
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Help me add authentication"}]
# ✨ LLM automatically knows about your FastAPI project!
```

**That's it!** Memorable automatically:
- ✅ Injects relevant memories as context before each LLM call
- ✅ Extracts and stores new memories after each conversation
- ✅ Maintains conversation history
- ✅ Consolidates important memories in the background

## Key Features

### 🚀 Zero-Code Integration
Transparently intercepts LLM calls - no code changes required. Works with OpenAI, Anthropic, LiteLLM, and more.

### 🧠 Research-Backed
Uses validated techniques from Mem0's research paper (arXiv:2504.19413) for memory extraction, retrieval, and consolidation.

### 🕸️ Optional Graph Support
Enable graph-based memory for multi-hop reasoning. Works with NetworkX (default) or Neo4j.

### 🔍 Hybrid Retrieval
Combines semantic search (embeddings), keyword search (full-text), and graph traversal for optimal results.

### 🎯 Multiple Memory Modes
- **Auto Mode**: Dynamic per-query retrieval (default, best accuracy)
- **Conscious Mode**: One-shot working memory (fastest)
- **Hybrid Mode**: Combines both approaches (best of both worlds)

### 💾 SQL-First Storage
Works with PostgreSQL, SQLite, MySQL, Neon, Supabase - any SQL database you already use.

### 🌐 Multi-Model Support
Works with 100+ models via LiteLLM: OpenAI, Anthropic, Gemini, Llama, Mistral, and more.

## Installation

### From PyPI (Recommended)

```bash
pip install memorable-ai
```

### From Source

```bash
git clone https://github.com/0sparsh2/memorable-ai.git
cd memorable-ai
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Usage Examples

### Basic Usage (OpenAI)

```python
from memorable_ai import MemoryEngine
from openai import OpenAI

memory = MemoryEngine(database="sqlite:///memory.db", mode="auto")
memory.enable()

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What do I like?"}]
)
```

### With Anthropic Claude

```python
from memorable_ai import MemoryEngine
from anthropic import Anthropic

memory = MemoryEngine(database="sqlite:///memory.db")
memory.enable()

client = Anthropic()
response = client.messages.create(
    model="claude-3-opus",
    messages=[{"role": "user", "content": "Tell me about myself"}]
)
```

### With Gemini via LiteLLM

```python
from memorable_ai import MemoryEngine
import litellm

memory = MemoryEngine(database="sqlite:///memory.db")
memory.enable()

response = litellm.completion(
    model="gemini/gemini-2.5-flash",
    messages=[{"role": "user", "content": "describe me"}]
)
```

### With Graph Support

```python
from memorable_ai import MemoryEngine

memory = MemoryEngine(
    database="postgresql://user:pass@localhost/memorable",
    graph_enabled=True,  # Enable graph for multi-hop reasoning
    mode="hybrid"
)
memory.enable()

# Now supports relationship traversal and multi-hop queries!
```

### Users, sessions and explicit APIs

```python
import asyncio
from openai import OpenAI
from memorable_ai import MemoryEngine

memory = MemoryEngine(database="sqlite:///memory.db")
memory.enable()                                   # patches OpenAI / Anthropic / LiteLLM (sync, async, streaming)
client = OpenAI()

# Attribute calls to a user + session. Memories never cross users.
with memory.scope(user_id="alice", session_id="chat-42"):
    client.chat.completions.create(model="gpt-4o-mini", messages=[...])
# ...or pass it per call: metadata={"user_id": "alice", "session_id": "chat-42"}

async def main():
    # Explicit write path (for frameworks you wire yourself); waits for completion
    await memory.ingest(messages, response, user_id="alice", session_id="chat-42")
    await memory.end_session(user_id="alice", session_id="chat-42")   # summarise open episodes

    ctx = await memory.get_context("Any allergies I should mention?", user_id="alice")
    hits = await memory.recall("trip to Kyoto", user_id="alice")      # facts + episodes + excerpts
    mid = await memory.add_memory("User works at Stripe", user_id="alice", infer=True)
    print(memory.history(mid))                    # version chain (updates never overwrite)
    print(memory.graph(user_id="alice"))          # knowledge-graph edges
    await memory.forget_user("alice")             # right to be forgotten

asyncio.run(main())
memory.flush()   # wait for background writes (default write_mode="background")
```

## Architecture

Memorable uses an interceptor-based architecture that transparently:

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                      │
│  (OpenAI, Anthropic, LiteLLM calls - unchanged code)     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Memorable Interceptor                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Pre-call:   │  │  LLM Call    │  │  Post-call:  │  │
│  │  Inject      │→ │  (original)  │→ │  Extract &   │  │
│  │  Context     │  │              │  │  Store       │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Memory Engine                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Storage  │  │Retrieval │  │Extraction│  │  Graph   │ │
│  │  (SQL)   │  │ (Hybrid) │  │ (Pattern)│  │(Optional)│ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Memory layers (v0.2)

| Layer | What | Used for |
|---|---|---|
| L0 turns | Verbatim messages per session (only the unseen suffix of resent histories) | Ground truth, provenance, excerpt retrieval |
| L1 episodes | Topic/time-coherent segments: event-segmentation boundaries; title, summary, key points | Past-conversation recall, multi-hop questions |
| L2 facts | Atomic statements with bi-temporal versions (`supersedes_id` / `superseded_by_id`) | Precise recall, core user profile |
| L2 graph | Entities, relations, mentions in SQL; single-valued edges invalidate old ones | Personalized PageRank retrieval ("where does my sister live?") |
| L4 context | Core profile + relevant facts + episodes + excerpts, dated, token-budgeted | What gets injected |

Retrieval fuses dense, BM25 and graph signals with Reciprocal Rank Fusion, scoped per user.
See [docs/memory_architecture.md](docs/memory_architecture.md) for the design, the research it draws on, and LoCoMo retrieval measurements.

## Configuration

### Environment Variables

```bash
# Database
export MEMORABLE_DATABASE__CONNECTION_STRING="postgresql://user:pass@localhost/memorable"

# Memory Settings
export MEMORABLE_MEMORY__MODE="auto"  # conscious, auto, or hybrid
export MEMORABLE_MEMORY__NAMESPACE="production"
export MEMORABLE_MEMORY__MAX_CONTEXT_TOKENS="2000"

# Graph (Optional)
export MEMORABLE_GRAPH__ENABLED="true"
export MEMORABLE_GRAPH__CONNECTION_STRING="neo4j://localhost:7687"

# LLM API Keys
export OPENAI_API_KEY="sk-your-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
export GEMINI_API_KEY="your-key-here"
```

### Programmatic Configuration

```python
from memorable_ai import MemoryEngine, MemorableConfig

config = MemorableConfig.from_env()
memory = MemoryEngine(config=config)
memory.enable()
```

See [docs/api.md](docs/api.md) for complete configuration options.

## Database Support

Memorable works with any SQL database:

| Database | Connection String Example |
|----------|--------------------------|
| **SQLite** | `sqlite:///memory.db` |
| **PostgreSQL** | `postgresql://user:pass@localhost/memorable` |
| **MySQL** | `mysql://user:pass@localhost/memorable` |
| **Neon** | `postgresql://user:pass@ep-*.neon.tech/memorable` |
| **Supabase** | `postgresql://postgres:pass@db.*.supabase.co/postgres` |

## Memory Modes

### Auto Mode (Default)
Dynamic per-query retrieval - best for most use cases.

```python
memory = MemoryEngine(mode="auto")
```

### Conscious Mode
One-shot working memory injection - faster, good for simple conversations.

```python
memory = MemoryEngine(mode="conscious")
```

### Hybrid Mode
Combines both approaches - best accuracy, slightly slower.

```python
memory = MemoryEngine(mode="hybrid")
```

## Benchmarks

**Implemented today:** LOCOMO smoke harness (`benchmarks/locomo/test_data.py`), measured retrieval latency via `benchmarks/comparison/compare_systems.py`, and unit tests (`pytest tests/unit`).

**Targets** (not yet validated on the full snap-research/LOCOMO leaderboard):

- 30%+ improvement over Mem0 on LOCOMO
- 95%+ accuracy on multi-hop reasoning
- Under 100ms retrieval latency (p95)
- 50%+ token savings vs full-context

See [docs/benchmarks.md](docs/benchmarks.md) for measured vs target metrics and [agent_memory_landscape_2026/report.md](agent_memory_landscape_2026/report.md) for competitive gap analysis.

## Research & Citations

### Research Papers

1. **Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory**
   - arXiv:2504.19413 (April 2025)
   - https://arxiv.org/abs/2504.19413
   - *Reference for memory extraction, retrieval, and consolidation techniques*

2. **Highly engaging events reveal semantic and temporal compression in online community discourse**
   - PNAS Nexus (March 2025)
   - *Reference for temporal memory and discourse analysis*

3. **XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model**
   - arXiv:2207.07115 (July 2022)
   - https://arxiv.org/abs/2207.07115
   - *Reference for memory hierarchy models*

4. **Abstractive Summarization of Reddit Posts with Multi-level Memory Networks**
   - ACL (2019)
   - *Reference for multi-level memory architectures*

### Open Source Repositories

1. **Memori (GibsonAI/Memori)**
   - https://github.com/GibsonAI/Memori
   - License: Apache 2.0
   - *Reference for interceptor-based architecture and SQL storage*

2. **Mem0 (mem0ai/mem0)**
   - https://github.com/mem0ai/mem0
   - *Reference for research-backed memory techniques and LOCOMO benchmark*

3. **Supermemory (supermemoryai/supermemory)**
   - https://github.com/supermemoryai/supermemory
   - License: MIT
   - *Reference for knowledge graph architecture*

See [docs/research.md](docs/research.md) for detailed research approach and methodology.

## Documentation

- **[Architecture](docs/architecture.md)** - System design and components
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Benchmarks](docs/benchmarks.md)** - Performance metrics and methodology
- **[Research](docs/research.md)** - Research approach and unique features
- **[Examples](examples/)** - Real-world usage examples

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/0sparsh2/memorable-ai.git
cd memorable-ai

# Install in development mode
pip install -e ".[dev]"

# Run tests
make test

# Run with coverage
make test-cov
```

## Roadmap

- [x] Mem0-style extraction/update pipeline (pattern / LLM / hybrid)
- [x] Multi-signal retrieval (semantic + BM25 + entity + graph)
- [x] Temporal validity (valid_from, invalidated_at, superseded_by)
- [x] LangChain callback, AutoGen bridge, CrewAI helpers, optional Mem0 adapter
- [x] Neo4j graph backend (optional `[graph]` extra)
- [x] Competitive research (`agent_memory_landscape_2026/`)
- [ ] Full LOCOMO / LongMemEval / BEAM leaderboard runs
- [ ] Publish benchmark results vs Mem0
- [ ] Bundled connectors (Notion, Slack) and full multimodal ingest
- [ ] Enterprise features (admin dashboard, audit logs)
- [ ] Research paper publication

See [ROADMAP.md](ROADMAP.md) for detailed plans.

## License

Apache 2.0 - see [LICENSE](LICENSE) file.

## Acknowledgments

Memorable is built on the excellent work of:

- **Memori** team for the interceptor architecture
- **Mem0** team for research-backed techniques
- **Supermemory** team for graph architecture inspiration

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: https://github.com/0sparsh2/memorable-ai/issues
- **Discussions**: https://github.com/0sparsh2/memorable-ai/discussions

---

<div align="center">

**Memorable** - The easiest, most efficient memory system for AI applications.

[⭐ Star us on GitHub](https://github.com/0sparsh2/memorable-ai) • [📖 Read the Docs](docs/) • [💬 Join Discussions](https://github.com/0sparsh2/memorable-ai/discussions)

</div>
