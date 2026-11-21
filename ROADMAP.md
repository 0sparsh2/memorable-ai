# Memorable Roadmap

## Short-term (0-3 months)

> Priorities from [agent_memory_landscape_2026/report.md](agent_memory_landscape_2026/report.md): P0 accuracy+benchmarks, P1 temporal validity, P2 ecosystem.

### Core Improvements
- [x] LLM extraction + ADD/UPDATE/DELETE/NOOP update phase (`memorable_ai/core/update.py`)
- [x] Multi-signal retrieval (semantic + BM25 + entity fusion)
- [x] Temporal validity schema (`valid_from`, `invalidated_at`, `superseded_by_id`)
- [x] LOCOMO smoke harness + unit tests
- [ ] Complete async/await support throughout codebase
- [ ] Improved graph entity extraction (spaCy/LLM NER)
- [ ] Better error handling and recovery
- [ ] Performance optimizations (caching, query optimization)
- [x] Full snap-research/LOCOMO dataset integration (`benchmarks/locomo/loader.py`)
- [ ] Full LOCOMO leaderboard run (1540 questions, publish scores)
- [x] SQLite FTS5 for keyword search (`memorable_ai/core/storage.py`)
- [x] LOCOMO smoke in Makefile + CI

### Framework Integrations
- [x] LangChain callback implementation (`integrations/langchain.py`)
- [x] Optional Mem0 adapter (`integrations/mem0_adapter.py`)
- [x] AutoGen integration (`AutoGenMemoryBridge`)
- [x] CrewAI integration (`setup_crewai_integration`)
- [x] Cross-session identity resolver (`memorable_ai/core/identity.py`)
- [ ] CamelAI integration
- [ ] Swarms integration

### Testing & Validation
- [x] Initial unit tests (extraction, retrieval, storage, LOCOMO smoke)
- [ ] Expand unit test coverage to >90%
- [ ] Add more integration test scenarios
- [x] LOCOMO benchmark with minimal test_data (`benchmarks/locomo/test_data.py`)
- [ ] Full LOCOMO dataset from snap-research/LOCOMO
- [ ] Add performance regression tests
- [ ] Add load testing

## Medium-term (3-6 months)

### Features
- [x] Multimodal plugin SDK stub (`memorable_ai/multimodal/`)
- [x] Connector plugin SDK stub (`memorable_ai/connectors/`)
- [ ] Multi-modal memory support (images, video, audio) — full ingest
- [x] Neo4j graph backend (`memorable_ai/graph/neo4j_backend.py`)
- [ ] Memory versioning and history
- [ ] Memory sharing between users/agents
- [ ] Advanced consolidation strategies

### Enterprise Features
- [ ] Admin dashboard/web UI
- [ ] Audit logging
- [ ] Role-based access control
- [ ] Data export/import tools
- [ ] Compliance documentation (GDPR, SOC2)

### Research & Validation
- [ ] Publish research paper
- [ ] Conduct comprehensive benchmarks
- [ ] Compare with all major systems
- [ ] Longitudinal studies
- [ ] Real-world case studies

## Long-term (6-12 months)

### Advanced Features
- [ ] Memory compression and summarization
- [ ] Predictive memory (anticipate user needs)
- [ ] Memory federation (distributed memory)
- [ ] Advanced temporal reasoning
- [ ] Causal relationship modeling

### Ecosystem
- [ ] Plugin system
- [ ] Third-party integrations marketplace
- [ ] Community contributions
- [ ] Standardization efforts
- [ ] Industry partnerships

### Platform
- [ ] Cloud-hosted version
- [ ] Managed service offering
- [ ] Enterprise support
- [ ] Training and certification

## Research Goals

- [ ] 30%+ improvement over Mem0 on LOCOMO
- [ ] 95%+ accuracy on multi-hop reasoning
- [ ] <100ms retrieval latency
- [ ] 50%+ token savings vs full-context
- [ ] Published research paper
- [ ] Industry recognition

## Community Goals

- [ ] 10,000+ GitHub stars
- [ ] Active community forums
- [ ] Regular contributions
- [ ] Conference presentations
- [ ] Blog posts and tutorials

