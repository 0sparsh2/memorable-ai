"""
Main Memory Engine.

Wires the memory layers together:

    L0 turns  ->  L1 episodes / sessions  ->  L2 facts + knowledge graph  ->  L4 injected context

and exposes three ways to use them:

1. **Zero-code**: ``engine.enable()`` patches the OpenAI / Anthropic / LiteLLM SDKs.
   Every call gets relevant memory injected and is recorded afterwards.
2. **Explicit**: ``await engine.ingest(messages, response, user_id=...)`` and
   ``await engine.get_context(query, user_id=...)`` for frameworks you wire yourself.
3. **Manual**: ``add_memory`` / ``search_memories`` / ``forget`` / ``history``.

Scope: set the user and session once with ``with engine.scope(user_id=..., session_id=...)``,
or pass ``metadata={"user_id": ...}`` in the LLM call. Without a user, memories
go to a shared pool that user-scoped calls never see.
"""

import hashlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from memorable_ai.core.consolidation import MemoryConsolidator
from memorable_ai.core.extraction import MemoryExtractor, response_text
from memorable_ai.core.identity import IdentityResolver
from memorable_ai.core.interceptor import LLMInterceptor
from memorable_ai.core.prefetch import PredictivePrefetch
from memorable_ai.core.providers import Adapter, OpenAIChatAdapter
from memorable_ai.core.retrieval import HybridRetriever
from memorable_ai.core.scope import (
    BackgroundWorker,
    Scope,
    active_scope,
    current_session,
    current_user,
    run_sync,
)
from memorable_ai.core.storage import Storage
from memorable_ai.core.summary import ConversationSummaryStore
from memorable_ai.core.temporal import TemporalMemory
from memorable_ai.core.update import MemoryUpdater
from memorable_ai.utils.config import MemorableConfig

from memorable_ai.core.models import utcnow  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class PreparedCall:
    """State carried from the pre-call (retrieval) to the post-call (recording) hook."""

    adapter: Adapter
    scope: Scope
    messages: List[Any]
    kwargs: Dict[str, Any]
    context: str = ""
    started_at: datetime = field(default_factory=utcnow)


class MemoryEngine:
    """Memory for LLMs, AI agents and multi-agent systems."""

    def __init__(
        self,
        database: Optional[str] = None,
        graph_enabled: bool = False,
        graph_database: Optional[str] = None,
        mode: str = "auto",
        config: Optional[MemorableConfig] = None,
        namespace: Optional[str] = None,
        write_mode: Optional[str] = None,
        **kwargs: Any,
    ):
        if config is None:
            try:
                self.config = MemorableConfig.from_env()
            except Exception:
                self.config = MemorableConfig()
            if database:
                self.config.database.connection_string = database
            if graph_enabled:
                self.config.graph.enabled = True
            if graph_database:
                self.config.graph.connection_string = graph_database
            if mode:
                self.config.memory.mode = mode
        else:
            self.config = config
        if namespace is not None:
            self.config.memory.namespace = namespace
        if write_mode is not None:
            self.config.memory.write_mode = write_mode
        for key, value in kwargs.items():
            if hasattr(self.config.memory, key):
                setattr(self.config.memory, key, value)

        self._storage: Optional[Storage] = None
        self._embedder = None
        self._retrieval: Optional[HybridRetriever] = None
        self._extraction: Optional[MemoryExtractor] = None
        self._updater: Optional[MemoryUpdater] = None
        self._episodic = None
        self._knowledge_graph = None
        self._pipeline = None
        self._graph = None  # optional legacy/Neo4j mirror
        self._consolidator: Optional[MemoryConsolidator] = None
        self._temporal: Optional[TemporalMemory] = None
        self._summary_store: Optional[ConversationSummaryStore] = None
        self._prefetch: Optional[PredictivePrefetch] = None
        self._identity: Optional[IdentityResolver] = None
        self._mode_handler = None
        self._worker: Optional[BackgroundWorker] = None
        self._interceptor = LLMInterceptor(self)
        self._enabled = False

    # ------------------------------------------------------------------ lifecycle

    def enable(self) -> None:
        """Start intercepting LLM calls (zero-code integration)."""
        if self._enabled:
            logger.warning("Memory engine already enabled")
            return
        self._ensure_components()
        self._interceptor.enable()
        if self._worker is not None and self._consolidator is not None and self.config.memory.consolidation_interval > 0:
            self._worker.spawn(self._consolidator.start())
        self._enabled = True
        logger.info("Memory engine enabled (mode=%s, write=%s)", self.config.memory.mode, self.config.memory.write_mode)

    def disable(self) -> None:
        """Stop intercepting; waits for pending background writes."""
        if not self._enabled:
            return
        self._interceptor.disable()
        if self._worker is not None:
            if self._consolidator is not None:
                self._consolidator._running = False
            self._worker.stop()
        self._enabled = False
        logger.info("Memory engine disabled")

    def close(self) -> None:
        self.disable()
        if self._worker is not None:
            self._worker.stop()
        if self._storage is not None:
            self._storage.engine.dispose()

    def flush(self, timeout: Optional[float] = None) -> None:
        """Block until background memory writes have finished."""
        if self._worker is not None:
            self._worker.flush(timeout=timeout)

    def _initialize_components(self) -> None:
        self._ensure_components()

    def _ensure_components(self) -> None:
        if self._storage is not None:
            return
        from memorable_ai.core.embeddings import get_embedder
        from memorable_ai.core.episodic import EpisodicMemory
        from memorable_ai.core.pipeline import WritePipeline
        from memorable_ai.graph.knowledge_graph import KnowledgeGraph

        cfg = self.config
        if cfg.llm.nvidia_api_key:
            try:
                from memorable_ai.integrations.nvidia_nim import configure_litellm_env

                configure_litellm_env(api_key=cfg.llm.nvidia_api_key)
            except Exception as e:
                logger.debug("NIM env setup skipped: %s", e)

        self._storage = Storage(
            connection_string=cfg.database.connection_string or "sqlite:///memorable.db",
            namespace=cfg.memory.namespace,
        )
        self._embedder = get_embedder(cfg.llm.embedding_model)
        llm_model = cfg.llm.default_model
        ext_cfg = cfg.memory.extraction
        rcfg = cfg.memory.retrieval

        self._knowledge_graph = KnowledgeGraph(self._storage)
        self._summary_store = ConversationSummaryStore(self._storage, llm_model=llm_model)
        self._extraction = MemoryExtractor(
            embedding_model=self._embedder,
            llm_model=llm_model,
            extraction_mode=ext_cfg.mode,
            summary_store=self._summary_store,
        )
        self._updater = MemoryUpdater(
            storage=self._storage,
            embedding_model=self._embedder,
            llm_model=llm_model,
            similar_top_k=ext_cfg.similar_top_k,
            add_only=ext_cfg.add_only,
            knowledge_graph=self._knowledge_graph,
        )
        if cfg.memory.episodic.enabled:
            self._episodic = EpisodicMemory(
                self._storage,
                embedder=self._embedder,
                llm_model=llm_model,
                config=cfg.memory.episodic,
                knowledge_graph=self._knowledge_graph,
            )
        self._pipeline = WritePipeline(
            storage=self._storage,
            extractor=self._extraction,
            updater=self._updater,
            episodic=self._episodic,
            knowledge_graph=self._knowledge_graph,
            summary_store=self._summary_store,
            timing=ext_cfg.timing,
            context_window=ext_cfg.recent_window,
        )
        if cfg.graph.enabled and cfg.graph.connection_string:
            from memorable_ai.graph.builder import GraphBuilder

            self._graph = GraphBuilder(connection_string=cfg.graph.connection_string)
        self._retrieval = HybridRetriever(
            storage=self._storage,
            embedding_model=self._embedder,
            knowledge_graph=self._knowledge_graph,
            semantic_weight=rcfg.semantic_weight,
            bm25_weight=rcfg.bm25_weight,
            entity_weight=rcfg.entity_weight,
            graph_weight=rcfg.graph_weight,
            rrf_k=rcfg.rrf_k,
            min_semantic_score=rcfg.min_semantic_score,
            rerank=rcfg.rerank,
            rerank_model=rcfg.rerank_model,
            max_context_tokens=cfg.memory.max_context_tokens,
            max_facts=rcfg.max_facts,
            max_episodes=rcfg.max_episodes,
            max_excerpts=rcfg.max_excerpts,
            include_shared=rcfg.include_shared,
            profile_facts=rcfg.profile_facts,
        )
        if cfg.memory.predictive_prefetch:
            self._prefetch = PredictivePrefetch(self._storage, self._retrieval)
        self._initialize_mode_handler()
        self._temporal = TemporalMemory(self._storage)
        self._identity = IdentityResolver(self._storage, namespace=cfg.memory.namespace)
        self._consolidator = MemoryConsolidator(
            storage=self._storage,
            interval=cfg.memory.consolidation_interval,
            embedder=self._embedder,
            knowledge_graph=self._knowledge_graph,
            jobs=[self._pipeline.close_idle],
        )
        if cfg.memory.write_mode == "background" and not self._storage.in_memory:
            self._worker = BackgroundWorker()
        logger.info("Components initialized")

    def _initialize_mode_handler(self) -> None:
        from memorable_ai.modes.auto import AutoMode
        from memorable_ai.modes.conscious import ConsciousMode
        from memorable_ai.modes.hybrid import HybridMode

        mode = self.config.memory.mode
        if mode == "conscious":
            self._mode_handler = ConsciousMode(self._retrieval)
        elif mode == "hybrid":
            self._mode_handler = HybridMode(self._retrieval)
        else:
            if mode != "auto":
                logger.warning("Unknown mode %s, defaulting to auto", mode)
            self._mode_handler = AutoMode(self._retrieval)

    # ------------------------------------------------------------------ scope

    @contextmanager
    def scope(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Iterator["MemoryEngine"]:
        """Attribute intercepted calls in this block to ``user_id`` / ``session_id``."""
        with active_scope(user_id=user_id, session_id=session_id):
            yield self

    def register_identity(self, alias: str, canonical_user_id: str, confidence: float = 1.0) -> None:
        """Link a session/device alias to a canonical user id (persisted)."""
        self._ensure_components()
        if self._identity:
            self._identity.register_alias(alias, canonical_user_id, confidence)

    def _canonical(self, user_id: Optional[str]) -> Optional[str]:
        if user_id is None:
            return None
        return self._identity.resolve(user_id) if self._identity else user_id

    def resolve_scope(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        messages: Optional[Sequence[Any]] = None,
        hints: Optional[Dict[str, Optional[str]]] = None,
    ) -> Scope:
        """
        Precedence: explicit args > ``engine.scope()`` > call hints (metadata.user_id,
        user, ...) > legacy per-message metadata > derived session id.
        """
        self._ensure_components()
        hints = hints or {}
        uid = user_id or current_user() or hints.get("user_id")
        sid = session_id or current_session() or hints.get("session_id")
        if (uid is None or sid is None) and messages:
            for msg in messages:
                md = msg.get("metadata") if isinstance(msg, dict) else None
                if isinstance(md, dict):
                    sid = sid or md.get("session_id")
                    uid = uid or md.get("user_id")
        uid = self._canonical(uid)
        if sid is None:
            sid = self._derive_session_id(uid, messages or [])
        return Scope(namespace=self.config.memory.namespace, user_id=uid, session_id=sid)

    @staticmethod
    def _derive_session_id(user_id: Optional[str], messages: Sequence[Any]) -> str:
        """Stable id for clients that resend history: hash of user + first user message."""
        from memorable_ai.core.episodic import message_text

        first = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                first = message_text(msg.get("content"))
                break
            if isinstance(msg, str):
                first = msg
                break
        digest = hashlib.sha1(f"{user_id or ''}\x00{first}".encode("utf-8")).hexdigest()[:16]
        return f"auto-{digest}"

    def _get_session_id(self, messages: List[Dict[str, Any]]) -> str:
        """Compatibility: session id for ``messages`` under the current scope."""
        return self.resolve_scope(messages=messages).session_id or "default"

    # ------------------------------------------------------------------ interception hooks

    async def _context_for(self, messages: Sequence[Any], scope: Scope) -> str:
        if self._mode_handler is None:
            return ""
        mode = self.config.memory.mode
        if mode in ("conscious", "hybrid"):
            return await self._mode_handler.get_context(scope.session_id, list(messages), scope=scope)
        return await self._mode_handler.get_context(list(messages), scope=scope)

    async def _aprepare_call(self, adapter: Adapter, kwargs: Dict[str, Any]) -> PreparedCall:
        messages = adapter.get_messages(kwargs)
        scope = self.resolve_scope(messages=messages, hints=adapter.get_hints(kwargs))
        context = await self._context_for(messages, scope) if messages else ""
        new_kwargs = adapter.inject(kwargs, context) if context else kwargs
        return PreparedCall(adapter, scope, messages, new_kwargs, context)

    def _prepare_call(self, adapter: Adapter, kwargs: Dict[str, Any]) -> PreparedCall:
        return run_sync(lambda: self._aprepare_call(adapter, kwargs), timeout=30)

    def _record_call(self, call: PreparedCall, text: str) -> None:
        """Record an exchange: on the background worker, or inline in sync mode."""
        if not call.messages and not text:
            return
        job = lambda: self._pipeline.ingest(call.scope, call.messages, text, call.started_at)  # noqa: E731
        if self._worker is not None:
            self._worker.submit(job)
        else:
            run_sync(job)

    async def _arecord_call(self, call: PreparedCall, text: str) -> None:
        if self._worker is not None:
            self._worker.submit(lambda: self._pipeline.ingest(call.scope, call.messages, text, call.started_at))
        else:
            await self._pipeline.ingest(call.scope, call.messages, text, call.started_at)

    # ------------------------------------------------------------------ compatibility hooks

    async def _inject_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return ``messages`` with memory context in a system message."""
        try:
            self._ensure_components()
            call = await self._aprepare_call(OpenAIChatAdapter(), {"messages": messages})
            return call.kwargs.get("messages", messages)
        except Exception as e:
            logger.error("Failed to inject context: %s", e)
            return messages

    def _inject_context_sync(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            return run_sync(lambda: self._inject_context(messages), timeout=30)
        except Exception as e:
            logger.error("Failed to inject context (sync): %s", e)
            return messages

    async def _store_conversation(self, messages: List[Dict[str, Any]], response: Any) -> None:
        """Run the write pipeline for one exchange (awaits completion)."""
        try:
            await self.ingest(messages, response)
        except Exception as e:
            logger.error("Failed to store conversation: %s", e)

    _store_conversation_async = _store_conversation

    def _store_conversation_sync(self, messages: List[Dict[str, Any]], response: Any) -> None:
        try:
            run_sync(lambda: self.ingest(messages, response))
        except Exception as e:
            logger.error("Failed to store conversation (sync): %s", e)

    def _store_in_new_loop(self, messages: List[Dict[str, Any]], response: Any) -> None:
        self._store_conversation_sync(messages, response)

    @staticmethod
    def _response_to_dict(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            try:
                return response.model_dump()
            except Exception:
                pass
        return response if isinstance(response, (dict, str, type(None))) else {"text": response_text(response)}

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""
        return "\n".join(["Relevant memories:"] + [f"- [{m.get('type', 'fact')}] {m.get('content', '')}" for m in memories])

    # ------------------------------------------------------------------ public API: write

    async def ingest(
        self,
        messages: Sequence[Any],
        response: Any = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        occurred_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Record a conversation exchange through the full pipeline and wait for it.
        ``messages`` may be the whole history; only unseen turns are added.
        """
        self._ensure_components()
        scope = self.resolve_scope(user_id, session_id, messages)
        text = response if isinstance(response, str) else response_text(response)
        report = await self._pipeline.ingest(scope, list(messages), text, occurred_at)
        return report.as_dict()

    async def end_session(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Close the session's open episodes (summaries, key points, graph links)."""
        self._ensure_components()
        scope = self.resolve_scope(user_id, session_id)
        report = await self._pipeline.end_session(scope)
        if self._mode_handler is not None and hasattr(self._mode_handler, "clear_session"):
            self._mode_handler.clear_session(scope.session_id, scope)
        return report.as_dict()

    async def add_memory(
        self,
        content: str,
        memory_type: str = "fact",
        user_id: Optional[str] = None,
        infer: bool = False,
        **metadata: Any,
    ) -> Optional[int]:
        """
        Store a memory directly. ``infer=True`` runs it through the update phase
        (dedup/contradiction handling); otherwise it is stored as given, except that
        exact duplicates are merged. Returns the memory id.
        """
        self._ensure_components()
        try:
            uid = self._canonical(user_id or metadata.pop("session_id", None) or current_user())
            if uid is not None:
                metadata.setdefault("canonical_user_id", uid)
            memory: Dict[str, Any] = {
                "content": content,
                "type": memory_type,
                "metadata": metadata,
                "user_id": uid,
                "session_id": user_id or uid,
                "source": "manual",
                "embedding": self._embedder.encode_list(content) if self._embedder else None,
                "importance_score": metadata.pop("importance_score", 0.5),
            }
            scope = Scope(namespace=self.config.memory.namespace, user_id=uid)
            if infer:
                applied = await self._updater.apply_facts([memory], scope)
            else:
                mid, op = self._storage.insert_memory(memory)
                applied = [{**memory, "id": mid, "op": op}]
            for fact in applied:
                if fact.get("op") in ("add", "update", "delete") and self._knowledge_graph is not None:
                    for old in self._knowledge_graph.ingest_memory(fact, scope):
                        if old != fact["id"]:
                            self._storage._invalidate(old, superseded_by_id=fact["id"], reason="graph_conflict")
            if self._graph is not None:
                await self._graph.update_graph(applied)
            return applied[0]["id"] if applied else None
        except Exception as e:
            logger.error("Failed to add memory: %s", e)
            return None

    async def forget(self, memory_id: int, hard: bool = True) -> bool:
        """Delete one memory (hard by default) and its graph links."""
        self._ensure_components()
        if self._knowledge_graph is not None:
            self._knowledge_graph.invalidate_memory(memory_id)
        return await self._storage.delete_memory(memory_id, hard=hard)

    async def forget_user(self, user_id: str) -> Dict[str, int]:
        """Hard-delete everything stored for a user (right to be forgotten)."""
        self._ensure_components()
        self.flush()
        return self._storage.forget_user(self._canonical(user_id))

    # ------------------------------------------------------------------ public API: read

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search facts for a user (or the shared pool when no user is given)."""
        self._ensure_components()
        try:
            scope = Scope(
                namespace=self.config.memory.namespace,
                user_id=self._canonical(user_id or current_user()),
            )
            return await self._retrieval.search(query, limit=limit, memory_type=memory_type, scope=scope)
        except Exception as e:
            logger.error("Failed to search memories: %s", e)
            return []

    async def recall(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        k_facts: Optional[int] = None,
        k_episodes: Optional[int] = None,
        k_excerpts: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Facts, past episodes and verbatim excerpts relevant to ``query``."""
        self._ensure_components()
        scope = Scope(
            namespace=self.config.memory.namespace,
            user_id=self._canonical(user_id or current_user()),
            session_id=session_id or current_session(),
        )
        res = await self._retrieval.recall(query, scope, k_facts, k_episodes, k_excerpts)
        for f in res["facts"]:
            f.pop("embedding", None)
        return res

    async def get_context(
        self,
        query: Union[str, Sequence[Any]],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """The memory block that would be injected for ``query`` (string or messages)."""
        self._ensure_components()
        messages = [{"role": "user", "content": query}] if isinstance(query, str) else list(query)
        scope = self.resolve_scope(user_id, session_id, messages)
        return await self._context_for(messages, scope)

    def history(self, memory_id: int) -> List[Dict[str, Any]]:
        """Version chain of a memory, oldest first."""
        self._ensure_components()
        return self._storage.memory_history(memory_id)

    def list_episodes(
        self, user_id: Optional[str] = None, session_id: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        self._ensure_components()
        if self._episodic is None:
            return []
        scope = Scope(self.config.memory.namespace, self._canonical(user_id or current_user()), session_id)
        return self._episodic.list_episodes(scope, limit=limit)

    def graph(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Valid knowledge-graph edges for a user."""
        self._ensure_components()
        scope = Scope(self.config.memory.namespace, self._canonical(user_id or current_user()))
        return self._knowledge_graph.relations(scope)

    async def consolidate(self) -> Dict[str, Any]:
        """Run one consolidation pass now (importance, contradictions, idle episodes)."""
        self._ensure_components()
        return await self._consolidator.consolidate()

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "enabled": self._enabled,
            "mode": self.config.memory.mode,
            "graph_enabled": self.config.graph.enabled,
            "write_mode": self.config.memory.write_mode,
            "hooked": self._interceptor.hooked,
        }
        if self._storage is not None:
            try:
                stats.update(self._storage.get_stats())
            except Exception:
                pass
        if self._knowledge_graph is not None:
            try:
                stats["graph"] = self._knowledge_graph.stats()
                stats["graph"]["nodes"] = stats["graph"]["entities"]
                stats["graph"]["edges"] = stats["graph"]["relations"]
            except Exception:
                pass
        return stats
