"""
Configuration management for Memorable.

Supports environment variables, config files, and programmatic configuration.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseConfig(BaseModel):
    """Database configuration."""

    connection_string: Optional[str] = Field(
        default=None, description="Database connection string"
    )
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Max overflow connections")


class GraphConfig(BaseModel):
    """Graph database configuration (optional)."""

    enabled: bool = Field(default=False, description="Enable graph storage")
    connection_string: Optional[str] = Field(
        default=None, description="Graph database connection string (Neo4j, etc.)"
    )


class ExtractionConfig(BaseModel):
    """Memory extraction and update configuration."""

    mode: str = Field(
        default="hybrid",
        description="Extraction mode: 'pattern', 'llm', or 'hybrid'",
    )
    add_only: bool = Field(
        default=False,
        description="Mem0 2026 style: ADD-only updates (no UPDATE/DELETE)",
    )
    recent_window: int = Field(default=10, description="Recent messages for extraction context")
    similar_top_k: int = Field(default=10, description="Similar memories for update phase")
    timing: str = Field(
        default="turn",
        description="When to extract facts: 'turn' (every exchange) or 'episode' (when an episode closes)",
    )


class RetrievalConfig(BaseModel):
    """
    Retrieval fusion. Weights are relative weights in weighted Reciprocal Rank
    Fusion (score = sum w / (rrf_k + rank)), so their scale does not matter.
    """

    # Defaults from benchmarks/locomo/retrieval_eval.py with MiniLM: BM25 beat MiniLM
    # dense retrieval on LoCoMo, and dense-heavy fusion lost to BM25 alone. Re-tune per
    # embedder; stronger retrieval embedders warrant a higher semantic weight.
    semantic_weight: float = Field(default=0.5)
    bm25_weight: float = Field(default=1.0)
    entity_weight: float = Field(default=0.0, description="Deprecated: entity matching now seeds the graph signal")
    graph_weight: float = Field(default=0.6)
    rrf_k: int = Field(default=60)
    max_facts: int = Field(default=10, description="Facts injected per call")
    max_episodes: int = Field(default=3, description="Past-episode summaries injected per call")
    max_excerpts: int = Field(default=3, description="Verbatim past-turn excerpts injected per call")
    profile_facts: int = Field(
        default=3, description="Always-on core facts (highest importance) injected for user-scoped calls"
    )
    min_semantic_score: float = Field(
        default=0.2, description="Dense hits below this cosine are dropped before fusion"
    )
    rerank: bool = Field(default=False, description="Cross-encoder rerank of fused candidates")
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    include_shared: bool = Field(
        default=False, description="Let user-scoped calls also see memories stored without a user"
    )


class EpisodicConfig(BaseModel):
    """Episode segmentation (event-segmentation style boundaries)."""

    enabled: bool = Field(default=True)
    gap_minutes: float = Field(default=30.0, description="Idle gap that always starts a new episode")
    min_turns: int = Field(default=4, description="Minimum turns before a topic boundary may close an episode")
    max_turns: int = Field(default=40, description="Hard cap on turns per episode")
    surprise_z: float = Field(default=2.0, description="Boundary when surprise > mean + z * std")
    min_surprise: float = Field(default=0.45, description="Absolute floor for a topic-shift boundary")
    index_turns: bool = Field(default=True, description="Embed and index verbatim turns for excerpt retrieval")
    turn_context_weight: float = Field(
        default=0.5, description="Blend this fraction of the previous turn's vector into each turn's index vector"
    )


class MemoryConfig(BaseModel):
    """Memory engine configuration."""

    mode: str = Field(
        default="auto", description="Memory mode: 'conscious', 'auto', or 'hybrid'"
    )
    namespace: Optional[str] = Field(
        default=None, description="Memory namespace for multi-tenant support"
    )
    max_context_tokens: int = Field(
        default=2000, description="Maximum tokens to inject as context"
    )
    consolidation_interval: int = Field(
        default=21600, description="Memory consolidation interval in seconds (6 hours)"
    )
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    predictive_prefetch: bool = Field(
        default=False, description="Enable predictive memory prefetch"
    )
    episodic: EpisodicConfig = Field(default_factory=EpisodicConfig)
    write_mode: str = Field(
        default="background",
        description="'background' (writes run on a worker thread, off the request path) or 'sync'",
    )


def _resolve_default_model() -> str:
    """Pick LLM model from env (NVIDIA NIM catalog-aware). 'none' disables LLM features."""
    from memorable_ai.integrations.nvidia_nim import resolve_memory_model

    gemini = os.getenv("GEMINI_MODEL")
    explicit = os.getenv("MEMORABLE_LLM__DEFAULT_MODEL")
    if explicit and explicit.strip().lower() in ("none", "off", "disabled", "false", "0"):
        return "none"
    if gemini and not explicit and not os.getenv("NVIDIA_MODEL"):
        return gemini if gemini.startswith("gemini/") else f"gemini/{gemini}"
    if explicit and explicit.startswith("gemini/"):
        return explicit
    if os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_MODEL"):
        return resolve_memory_model()
    if explicit:
        return explicit
    if gemini:
        return gemini if gemini.startswith("gemini/") else f"gemini/{gemini}"
    return "gpt-4o-mini"


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key"
    )
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    nvidia_api_key: Optional[str] = Field(
        default=None, description="NVIDIA NIM API key (nvapi-...)"
    )
    default_model: str = Field(
        default="gpt-4o-mini", description="Default LLM model (LiteLLM id)"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for semantic search",
    )


class MemorableConfig(BaseModel):
    """Main configuration for Memorable."""

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @classmethod
    def from_env(cls) -> "MemorableConfig":
        """Load configuration from environment variables."""
        return cls(
            database=DatabaseConfig(
                connection_string=os.getenv("MEMORABLE_DATABASE__CONNECTION_STRING"),
                pool_size=int(os.getenv("MEMORABLE_DATABASE__POOL_SIZE", "5")),
                max_overflow=int(os.getenv("MEMORABLE_DATABASE__MAX_OVERFLOW", "10")),
            ),
            graph=GraphConfig(
                enabled=os.getenv("MEMORABLE_GRAPH__ENABLED", "false").lower() == "true",
                connection_string=os.getenv("MEMORABLE_GRAPH__CONNECTION_STRING"),
            ),
            memory=MemoryConfig(
                mode=os.getenv("MEMORABLE_MEMORY__MODE", "auto"),
                namespace=os.getenv("MEMORABLE_MEMORY__NAMESPACE"),
                max_context_tokens=int(
                    os.getenv("MEMORABLE_MEMORY__MAX_CONTEXT_TOKENS", "2000")
                ),
                consolidation_interval=int(
                    os.getenv("MEMORABLE_MEMORY__CONSOLIDATION_INTERVAL", "21600")
                ),
                extraction=ExtractionConfig(
                    mode=os.getenv("MEMORABLE_EXTRACTION__MODE", "hybrid"),
                    add_only=os.getenv("MEMORABLE_EXTRACTION__ADD_ONLY", "false").lower()
                    == "true",
                    timing=os.getenv("MEMORABLE_EXTRACTION__TIMING", "turn"),
                ),
                write_mode=os.getenv("MEMORABLE_MEMORY__WRITE_MODE", "background"),
            ),
            llm=LLMConfig(
                openai_api_key=os.getenv("OPENAI_API_KEY")
                or os.getenv("MEMORABLE_LLM__OPENAI_API_KEY"),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
                or os.getenv("MEMORABLE_LLM__ANTHROPIC_API_KEY"),
                gemini_api_key=os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY"),
                nvidia_api_key=os.getenv("NVIDIA_NIM_API_KEY")
                or os.getenv("NVIDIA_API_KEY"),
                default_model=_resolve_default_model(),
                embedding_model=os.getenv(
                    "MEMORABLE_LLM__EMBEDDING_MODEL",
                    "sentence-transformers/all-MiniLM-L6-v2",
                ),
            ),
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MemorableConfig":
        """Load configuration from dictionary."""
        return cls(**config_dict)

