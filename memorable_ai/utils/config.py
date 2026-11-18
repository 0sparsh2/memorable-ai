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
    Retrieval fusion.

    Turn and episode layers default to score fusion: each signal's scores are
    z-normalised over the candidate pool and summed with ``fusion_weights``. The fact
    layer (dense + BM25 + graph) uses weighted Reciprocal Rank Fusion
    (score = sum w / (rrf_k + rank)) with ``semantic_weight`` / ``bm25_weight`` /
    ``graph_weight``, as do all layers when ``fusion="rrf"``.
    """

    # Defaults from benchmarks/locomo/retrieval_eval.py with MiniLM: BM25 beat MiniLM
    # dense retrieval on LoCoMo, and dense-heavy fusion lost to BM25 alone. Re-tune per
    # embedder; stronger retrieval embedders warrant a higher semantic weight.
    semantic_weight: float = Field(default=0.5)
    bm25_weight: float = Field(default=1.0)
    entity_weight: float = Field(default=0.0, description="Deprecated: entity matching now seeds the graph signal")
    graph_weight: float = Field(default=0.6)
    rrf_k: int = Field(default=60)
    fusion: str = Field(
        default="score",
        description="Turn/episode fusion: 'score' (z-normalised scores, +5 pts LoCoMo recall@5) or 'rrf'",
    )
    fusion_weights: Dict[str, float] = Field(
        default_factory=lambda: {"semantic": 1.0, "lexical": 0.5},
        description="Score-fusion weights per signal (docs/retrieval_round2.md)",
    )
    max_facts: int = Field(default=10, description="Facts injected per call")
    max_episodes: int = Field(default=3, description="Past-episode summaries injected per call")
    max_excerpts: int = Field(default=3, description="Verbatim past-turn excerpts injected per call")
    profile_facts: int = Field(
        default=3, description="Always-on core facts (highest importance) injected for user-scoped calls"
    )
    min_semantic_score: float = Field(
        default=0.2, description="Dense hits below this cosine are dropped before fusion"
    )
    rerank: bool = Field(default=False, description="Cross-encoder rerank of fused fact and turn candidates")
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        description="Reranker spec (ONNX Runtime when available; 'torch:' or 'onnx-int8:' prefixes as for embedders)",
    )
    rerank_depth: int = Field(default=20, description="Turn candidates the reranker reads per query")
    rerank_blend: float = Field(
        default=0.0, description="Weight of the first-stage score blended into the reranker's (0 = reranker only)"
    )
    include_shared: bool = Field(
        default=False, description="Let user-scoped calls also see memories stored without a user"
    )


class CaptureConfig(BaseModel):
    """What L0 keeps from agent traffic (docs/procedural_memory_design.md §5, decision D3)."""

    tool_events: bool = Field(default=True, description="Store tool calls and tool results as L0 events")
    pii_redaction: str = Field(
        default="off", description="PII redaction for stored messages and payloads: off | emails_phones | strict "
        "(secrets are always redacted)",
    )
    max_payload_bytes: int = Field(
        default=256 * 1024, description="Per-payload cap; above it, keep head + tail + sha256 + length",
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
    injection: str = Field(
        default="tail",
        description=(
            "Where retrieved memory goes: 'tail' (in front of the latest user message; the rest of the "
            "request stays byte-identical, so the provider's prompt cache keeps the conversation prefix) "
            "or 'system' (system prompt / instructions, the v0.2 behaviour)"
        ),
    )
    context_reuse_seconds: int = Field(
        default=600,
        description="Reuse the memory block while the latest user message is unchanged (tool loops); 0 = off",
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
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    entity_resolution_llm: bool = Field(
        default=True,
        description="Ask the LLM (when one is configured) which entity an ambiguous name means; "
        "otherwise only the deterministic rules run",
    )
    write_mode: str = Field(
        default="background",
        description="'background' (writes run on a worker thread, off the request path) or 'sync'",
    )


# Embedding profiles (docs/embedding_study.md §4, LoCoMo full pipeline vs v0.2):
#   quality: leaf-ir turns   -> fused turn recall 0.565 -> 0.642, multi-hop 0.504 -> 0.571, ingest 46 -> 39 s
#   fast:    static turns    -> fused 0.579, ingest 46 -> 25 s, query 7.9 -> 3.2 ms
#   legacy:  one model for everything (v0.2 behaviour)
# Facts, episodes and extractive summaries keep the main model in every profile;
# segmentation uses the static model whenever it is available.
DEFAULT_TURN_EMBEDDING_MODEL = "static:minishlab/potion-base-32M"  # fast profile / segmentation
QUALITY_TURN_EMBEDDING_MODEL = "MongoDB/mdbr-leaf-ir"


def _installed(module: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def _is_hash(model: str) -> bool:
    return (model or "").lower() in ("hash", "hashing", "hash-384")


def _resolve_turn_embedding_model(main_model: str) -> Optional[str]:
    """
    Turn-tier embedder. Explicit env wins ('none'/'off' disables tiering). Otherwise
    ``MEMORABLE_EMBEDDING_PROFILE`` (default ``quality``) picks it, provided the
    ``fast`` extra's runtimes are installed; without them, the main model is used.
    """
    env = os.getenv("MEMORABLE_LLM__TURN_EMBEDDING_MODEL")
    if env is not None:
        return None if env.strip().lower() in ("", "none", "off", "same") else env
    if _is_hash(main_model):
        return None
    profile = os.getenv("MEMORABLE_EMBEDDING_PROFILE", "quality").strip().lower()
    if profile == "legacy":
        return None
    if profile == "quality" and _installed("onnxruntime"):
        return QUALITY_TURN_EMBEDDING_MODEL
    if _installed("model2vec"):
        return DEFAULT_TURN_EMBEDDING_MODEL
    return None


def _resolve_segment_embedding_model(main_model: str) -> Optional[str]:
    """
    Segmentation embedder: explicit env, else the static model if available, else
    None, which means the *main* model (never the turn model; see EpisodicMemory).
    """
    env = os.getenv("MEMORABLE_LLM__SEGMENT_EMBEDDING_MODEL")
    if env is not None:
        return None if env.strip().lower() in ("", "none", "off", "same") else env
    if _is_hash(main_model) or os.getenv("MEMORABLE_EMBEDDING_PROFILE", "").strip().lower() == "legacy":
        return None
    return DEFAULT_TURN_EMBEDDING_MODEL if _installed("model2vec") else None


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
    segment_embedding_model: Optional[str] = Field(
        default=None,
        description=(
            "Embedder for episode segmentation (topic-shift surprise). Thresholds are cosine "
            "distances, so this is kept independent of the turn index model; defaults to the "
            "static model when model2vec is installed, else the turn embedder."
        ),
    )
    turn_embedding_model: Optional[str] = Field(
        default=None,
        description=(
            "Optional fast embedder for the high-volume turn tier (L0 index, segmentation, "
            "extractive summaries), e.g. 'static:minishlab/potion-base-8M'. Facts and episodes "
            "keep embedding_model. See docs/embedding_study.md."
        ),
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
                turn_embedding_model=_resolve_turn_embedding_model(
                    os.getenv("MEMORABLE_LLM__EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                ),
                segment_embedding_model=_resolve_segment_embedding_model(
                    os.getenv("MEMORABLE_LLM__EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                ),
            ),
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MemorableConfig":
        """Load configuration from dictionary."""
        return cls(**config_dict)

