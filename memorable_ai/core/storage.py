"""
Storage layer.

SQL-first: every memory layer lives in one SQL database (SQLite, Postgres, MySQL),
defined in :mod:`memorable_ai.core.models`. This module owns:

- engine/session setup (WAL + busy timeout on SQLite, safe pooling across threads)
- additive schema migration for databases created by older versions
- scoped CRUD for semantic memories, with version chains instead of in-place edits
- search primitives used by retrieval: vector search, lexical search (SQLite FTS5
  or an in-process BM25 fallback), and order-preserving hydration with scope checks
- the graph ``edges`` table (writes, invalidation, cascades, and the ``graph_epoch``
  signal that tells in-process graph engines to rebuild after hard deletes)
"""

import hashlib
import json
import logging
import re
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import create_engine, event, inspect, or_, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from memorable_ai.core.errors import StorageError
from memorable_ai.core.models import (  # noqa: F401  (re-exported for compatibility)
    Base,
    Conversation,
    Counter,
    Edge,
    Entity,
    Episode,
    IdentityAlias,
    MemorableMeta,
    Memory,
    SessionRecord,
    Turn,
)
from memorable_ai.core.vector_index import STOPWORDS, LexicalIndex, VectorIndex, light_stem

from memorable_ai.core.models import utcnow  # noqa: E402

logger = logging.getLogger(__name__)


class _AnyUser:
    """Sentinel: do not filter by user (admin/listing APIs only)."""

    def __repr__(self) -> str:
        return "ANY_USER"


ANY_USER: Any = _AnyUser()
_UNSET: Any = object()

KIND_MODELS = {"fact": Memory, "episode": Episode, "turn": Turn}
# kind -> (fts table, id column)
FTS_TABLES = {
    "fact": ("memories_fts", "memory_id"),
    "episode": ("episodes_fts", "ref_id"),
    "turn": ("turns_fts", "ref_id"),
}

_WS_RE = re.compile(r"\s+")
_PUNCT_TAIL_RE = re.compile(r"[\s\.\!\?,;:]+$")


def normalize_text(text_value: str) -> str:
    t = _WS_RE.sub(" ", (text_value or "").strip().lower())
    return _PUNCT_TAIL_RE.sub("", t)


def content_hash(text_value: str) -> str:
    return hashlib.sha256(normalize_text(text_value).encode("utf-8")).hexdigest()


# Edge kinds that put a fact into the walkable graph (retired when the fact is invalidated).
WALK_FACT_KINDS = ("about", "mentions")


def _chunks(seq: Sequence[int], n: int):
    for i in range(0, len(seq), n):
        yield list(seq[i : i + n])


def episode_search_text(ep: Episode) -> str:
    parts = [ep.title or "", ep.summary or ""]
    if ep.key_points:
        parts.extend(str(k) for k in ep.key_points)
    return "\n".join(p for p in parts if p)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


class Storage:
    """SQL storage for all memory layers."""

    def __init__(self, connection_string: str, namespace: Optional[str] = None):
        self.connection_string = connection_string
        self.namespace = namespace
        self.is_sqlite = connection_string.startswith("sqlite")
        path_part = connection_string.split("///", 1)[1] if "///" in connection_string else ""
        self._in_memory = self.is_sqlite and (path_part in ("", ":memory:") or "mode=memory" in path_part)

        if self.is_sqlite:
            kwargs: Dict[str, Any] = {"connect_args": {"check_same_thread": False, "timeout": 30}}
            if self._in_memory:
                kwargs["poolclass"] = StaticPool
            self.engine = create_engine(connection_string, **kwargs)
            in_memory = self._in_memory

            @event.listens_for(self.engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
                cur = dbapi_conn.cursor()
                try:
                    if not in_memory:
                        cur.execute("PRAGMA journal_mode=WAL")
                    cur.execute("PRAGMA busy_timeout=30000")
                    cur.execute("PRAGMA synchronous=NORMAL")
                finally:
                    cur.close()
        else:
            self.engine = create_engine(connection_string, pool_pre_ping=True)

        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._fts_enabled = False
        self._fts_kinds: Dict[str, bool] = {}
        self._sync_lock = threading.RLock()
        self._reset_indexes()

        Base.metadata.create_all(self.engine)
        self._migrate_schema()
        logger.info("Storage initialized: %s", self._redacted_url())

    def _reset_indexes(self, kind: Optional[str] = None) -> None:
        kinds = [kind] if kind else list(KIND_MODELS)
        if not hasattr(self, "_vec"):
            self._vec: Dict[str, VectorIndex] = {}
            self._lex: Dict[str, LexicalIndex] = {}
            self._vec_watermark: Dict[str, int] = {}
            self._lex_watermark: Dict[str, int] = {}
        for k in kinds:
            self._vec[k] = VectorIndex()
            self._lex[k] = LexicalIndex()
            self._vec_watermark[k] = 0
            self._lex_watermark[k] = 0

    def _redacted_url(self) -> str:
        return re.sub(r"//([^:/@]+):([^@]+)@", r"//\1:***@", self.connection_string)

    @property
    def in_memory(self) -> bool:
        return self._in_memory

    # ------------------------------------------------------------------ schema

    def _migrate_schema(self) -> None:
        """Additive migration for databases created by earlier versions."""
        insp = inspect(self.engine)
        tables = set(insp.get_table_names())
        added_user_col = False
        json_type = "TEXT" if self.is_sqlite else "JSON"
        with self.engine.connect() as conn:
            if "memories" in tables:
                existing = {c["name"] for c in insp.get_columns("memories")}
                for col, col_type in [
                    ("valid_from", "TIMESTAMP"),
                    ("valid_to", "TIMESTAMP"),
                    ("invalidated_at", "TIMESTAMP"),
                    ("superseded_by_id", "INTEGER"),
                    ("supersedes_id", "INTEGER"),
                    ("confidence", "FLOAT DEFAULT 1.0"),
                    ("session_id", "VARCHAR(255)"),
                    ("user_id", "VARCHAR(255)"),
                    ("content_hash", "VARCHAR(64)"),
                    ("last_accessed_at", "TIMESTAMP"),
                    ("source", "VARCHAR(32)"),
                    ("source_episode_id", "INTEGER"),
                    ("source_turn_ids", json_type),
                    ("agent_id", "VARCHAR(255)"),
                ]:
                    if col not in existing:
                        try:
                            conn.execute(text(f"ALTER TABLE memories ADD COLUMN {col} {col_type}"))
                            conn.commit()
                            if col == "user_id":
                                added_user_col = True
                        except Exception as e:
                            conn.rollback()
                            logger.debug("Migration skip memories.%s: %s", col, e)
            if "conversations" in tables:
                conv_cols = {c["name"] for c in insp.get_columns("conversations")}
                for col, col_type in [
                    ("summary", "TEXT"),
                    ("session_id", "VARCHAR(255)"),
                    ("user_id", "VARCHAR(255)"),
                ]:
                    if col not in conv_cols:
                        try:
                            conn.execute(text(f"ALTER TABLE conversations ADD COLUMN {col} {col_type}"))
                            conn.commit()
                        except Exception as e:
                            conn.rollback()
                            logger.debug("Migration skip conversations.%s: %s", col, e)
            if "edges" in tables and "seq" not in {c["name"] for c in insp.get_columns("edges")}:
                try:
                    conn.execute(text("ALTER TABLE edges ADD COLUMN seq INTEGER"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_edges_sync ON edges (namespace, user_id, seq)"))
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logger.debug("Migration skip edges.seq: %s", e)
            for table in ("turns", "episodes", "sessions", "entities"):
                if table not in tables:
                    continue
                cols = {c["name"] for c in insp.get_columns(table)}
                if "agent_id" not in cols:
                    try:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN agent_id VARCHAR(255)"))
                        conn.commit()
                    except Exception as e:
                        conn.rollback()
                        logger.debug("Migration skip %s.agent_id: %s", table, e)
            for stmt in (
                "CREATE INDEX IF NOT EXISTS idx_memory_scope ON memories (namespace, user_id, invalidated_at)",
                "CREATE INDEX IF NOT EXISTS ix_memories_content_hash ON memories (content_hash)",
            ):
                try:
                    conn.execute(text(stmt))
                    conn.commit()
                except Exception as e:  # MySQL has no IF NOT EXISTS for indexes
                    conn.rollback()
                    logger.debug("Index skip: %s", e)
        self._backfill(added_user_col)
        self._ensure_counters()
        self._backfill_edges(tables)
        self._ensure_fts()

    def _backfill(self, added_user_col: bool) -> None:
        """Fill user_id/content_hash on legacy rows (portable Python loop, runs once)."""
        session = self.get_session()
        try:
            rows = session.query(Memory).filter(Memory.content_hash.is_(None)).limit(200000).all()
            for m in rows:
                m.content_hash = content_hash(m.content)
                if added_user_col and m.user_id is None and isinstance(m.extra_metadata, dict):
                    m.user_id = m.extra_metadata.get("canonical_user_id")
            if rows:
                session.commit()
        except Exception as e:
            session.rollback()
            logger.debug("Backfill skipped: %s", e)
        finally:
            session.close()

    def _ensure_counters(self) -> None:
        """Create the counter rows up front so writers only ever UPDATE them."""
        for key in ("graph_seq", "graph_epoch"):
            session = self.get_session()
            try:
                if session.get(Counter, key) is None:
                    session.add(Counter(key=key, value=0))
                    session.commit()
            except Exception:  # another process created it first
                session.rollback()
            finally:
                session.close()

    def _bump(self, session: Session, key: str) -> int:
        """Atomically increment a counter inside the caller's transaction; return the new value."""
        n = (
            session.query(Counter)
            .filter(Counter.key == key)
            .update({Counter.value: Counter.value + 1}, synchronize_session=False)
        )
        if not n:  # counters table created by a newer process after our init
            session.add(Counter(key=key, value=1))
            session.flush()
            return 1
        # This transaction now holds the write lock on the row, so the read is our own value.
        return int(session.query(Counter.value).filter(Counter.key == key).scalar())

    def _backfill_edges(self, tables: set) -> None:
        """
        One-time copy of pre-v0.3 graph tables (``relations``, ``mentions``) and fact
        provenance columns into ``edges``. Legacy tables are left in place, untouched.
        """
        session = self.get_session()
        try:
            if session.query(Edge.id).first() is not None:
                return
            copied = 0
            if "relations" in tables:
                copied += session.execute(text(
                    "INSERT INTO edges (namespace, user_id, src_kind, src_id, kind, dst_kind, dst_id, predicate,"
                    " weight, confidence, valid_from, valid_to, created_at, invalidated_at, evidence_kind, evidence_id)"
                    " SELECT namespace, user_id, 'entity', subject_id, 'relates', 'entity', object_id, predicate,"
                    " 1.0, confidence, valid_from, valid_to, created_at, invalidated_at,"
                    " CASE WHEN memory_id IS NULL THEN NULL ELSE 'fact' END, memory_id FROM relations"
                )).rowcount or 0
            if "mentions" in tables:
                for col, src_kind, kind in (("memory_id", "fact", "about"), ("episode_id", "episode", "mentions"),
                                            ("turn_id", "turn", "mentions")):
                    copied += session.execute(text(
                        "INSERT INTO edges (namespace, user_id, src_kind, src_id, kind, dst_kind, dst_id, weight,"
                        f" confidence, created_at) SELECT e.namespace, e.user_id, '{src_kind}', m.{col}, '{kind}',"
                        f" 'entity', m.entity_id, 1.0, 1.0, m.created_at FROM mentions m"
                        f" JOIN entities e ON e.id = m.entity_id WHERE m.{col} IS NOT NULL"
                    )).rowcount or 0
            prov = (
                session.query(Memory)
                .filter(or_(Memory.source_episode_id.isnot(None), Memory.source_turn_ids.isnot(None)))
                .limit(200000)
                .all()
            )
            for m in prov:
                edges = self._provenance_edges(m.id, m.namespace, m.user_id, m.agent_id,
                                               m.source_episode_id, m.source_turn_ids)
                if edges:
                    self.add_edges(edges, session=session)
                    copied += len(edges)
            if copied:
                session.commit()
                logger.info("Backfilled %d graph edges from legacy tables", copied)
        except Exception as e:
            session.rollback()
            logger.debug("Edge backfill skipped: %s", e)
        finally:
            session.close()

    def _ensure_fts(self) -> None:
        """SQLite FTS5 tables per kind (BM25 ranking). Other databases use LexicalIndex."""
        if not self.is_sqlite:
            return
        try:
            with self.engine.connect() as conn:
                for kind, (fts, idcol) in FTS_TABLES.items():
                    conn.execute(
                        text(
                            f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts} USING fts5("
                            f"content, {idcol} UNINDEXED, tokenize='porter unicode61')"
                        )
                    )
                    conn.commit()
                    self._fts_kinds[kind] = True
                count = conn.execute(text("SELECT COUNT(*) FROM memories_fts")).scalar()
                if count == 0:
                    conn.execute(
                        text(
                            "INSERT INTO memories_fts(memory_id, content) "
                            "SELECT id, content FROM memories WHERE invalidated_at IS NULL"
                        )
                    )
                    conn.commit()
            self._fts_enabled = True
        except Exception as e:
            logger.debug("FTS5 not available, using in-process BM25: %s", e)
            self._fts_enabled = False
            self._fts_kinds = {}

    @staticmethod
    def _fts_match_query(query: str) -> str:
        """Build a safe FTS5 MATCH string (quoted OR of content words)."""
        words = re.findall(r"\w+", query)
        terms = [t for t in words if len(t) > 1 and t.lower() not in STOPWORDS]
        if not terms:
            terms = [t for t in words if len(t) > 2]
        if not terms:
            safe = query.replace('"', '""').strip()
            return f'"{safe}"' if safe else '""'
        clauses = []
        for t in terms[:16]:
            clauses.append('"' + t.replace('"', '""') + '"')
            # Prefix on the light stem catches morphology Porter splits (allergies/allergic).
            stem = light_stem(t)
            if len(stem) >= 4 and stem.isalnum() and stem != t.lower():
                clauses.append(f"{stem}*")
        return " OR ".join(clauses)

    def get_session(self) -> Session:
        return self.SessionLocal()

    # ------------------------------------------------------------------ scope

    def _ns_filter(self, query, model):
        if self.namespace:
            return query.filter(model.namespace == self.namespace)
        return query

    @staticmethod
    def _user_filter(query, model, user_id: Any, include_shared: bool = False):
        if user_id is ANY_USER:
            return query
        if user_id is None:
            return query.filter(model.user_id.is_(None))
        if include_shared:
            return query.filter(or_(model.user_id == user_id, model.user_id.is_(None)))
        return query.filter(model.user_id == user_id)

    def _user_scope_clause(self, canonical_user_id: Optional[str], session_id: Optional[str] = None):
        """Portable user filter (kept for API compatibility)."""
        if not canonical_user_id:
            return None
        clauses = [Memory.user_id == canonical_user_id]
        if session_id and session_id != canonical_user_id:
            clauses.append(Memory.user_id == session_id)
        return or_(*clauses)

    @staticmethod
    def _resolve_user_arg(user_id: Optional[str], canonical_user_id: Optional[str]) -> Any:
        """Legacy listing APIs: no user given -> no user filter."""
        uid = canonical_user_id or user_id
        return uid if uid else ANY_USER

    # ------------------------------------------------------------------ FTS / index sync

    def fts_insert(self, session: Session, kind: str, ref_id: int, content: str) -> None:
        if self._fts_kinds.get(kind) and content:
            fts, idcol = FTS_TABLES[kind]
            session.execute(
                text(f"INSERT INTO {fts}({idcol}, content) VALUES (:rid, :c)"),
                {"rid": ref_id, "c": content},
            )

    def fts_delete(self, session: Session, kind: str, ref_id: int) -> None:
        if self._fts_kinds.get(kind):
            fts, idcol = FTS_TABLES[kind]
            session.execute(text(f"DELETE FROM {fts} WHERE {idcol} = :rid"), {"rid": ref_id})

    def index_add(
        self,
        kind: str,
        ref_id: int,
        user_id: Optional[str],
        content: str,
        embedding: Optional[Sequence[float]] = None,
    ) -> None:
        """Add a committed row to the in-process indexes (if they are loaded)."""
        if embedding is not None and self._vec[kind].loaded:
            self._vec[kind].add(ref_id, user_id, embedding)
        if not self._fts_kinds.get(kind) and self._lex[kind].loaded:
            self._lex[kind].add(ref_id, user_id, content)

    def index_remove(self, kind: str, ref_id: int) -> None:
        self._vec[kind].remove(ref_id)
        self._lex[kind].remove(ref_id)

    @staticmethod
    def _valid_filter(query, model):
        if model is Memory:
            return query.filter(Memory.invalidated_at.is_(None))
        if model is Episode:
            return query.filter(Episode.status == "closed")
        return query

    def _sync_vector(self, kind: str) -> VectorIndex:
        """Load rows the vector index has not seen yet (``id > watermark``)."""
        model = KIND_MODELS[kind]
        with self._sync_lock:
            index = self._vec[kind]
            session = self.get_session()
            try:
                q = session.query(model.id, model.user_id, model.embedding).filter(
                    model.id > self._vec_watermark[kind], model.embedding.isnot(None)
                )
                q = self._valid_filter(self._ns_filter(q, model), model)
                max_seen = self._vec_watermark[kind]
                for rid, uid, emb in q.order_by(model.id).all():
                    if isinstance(emb, str):
                        try:
                            emb = json.loads(emb)
                        except Exception:
                            emb = None
                    if emb:
                        index.add(rid, uid, emb)
                    max_seen = max(max_seen, rid)
                self._vec_watermark[kind] = max_seen
                index.loaded = True
            finally:
                session.close()
            return index

    def _sync_lexical(self, kind: str) -> LexicalIndex:
        model = KIND_MODELS[kind]
        with self._sync_lock:
            index = self._lex[kind]
            session = self.get_session()
            try:
                q = session.query(model).filter(model.id > self._lex_watermark[kind])
                q = self._valid_filter(self._ns_filter(q, model), model)
                max_seen = self._lex_watermark[kind]
                for row in q.order_by(model.id).all():
                    body = episode_search_text(row) if kind == "episode" else row.content
                    index.add(row.id, row.user_id, body)
                    max_seen = max(max_seen, row.id)
                self._lex_watermark[kind] = max_seen
                index.loaded = True
            finally:
                session.close()
            return index

    # ------------------------------------------------------------------ search primitives

    def vector_search(
        self,
        kind: str,
        query_vec: Sequence[float],
        k: int = 20,
        user_id: Any = None,
        include_shared: bool = False,
    ) -> List[Tuple[int, float]]:
        index = self._sync_vector(kind)
        if user_id is ANY_USER:
            return index.search(query_vec, k)
        return index.search(query_vec, k, user_id=user_id, include_shared=include_shared)

    def lexical_search(
        self,
        kind: str,
        query: str,
        k: int = 20,
        user_id: Any = None,
        include_shared: bool = False,
        memory_type: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """BM25 search. Higher score = better."""
        if not query or not query.strip():
            return []
        if self._fts_kinds.get(kind):
            fts, idcol = FTS_TABLES[kind]
            table = KIND_MODELS[kind].__tablename__
            sql = (
                f"SELECT f.{idcol} AS rid, bm25({fts}) AS s FROM {fts} f "
                f"JOIN {table} t ON t.id = f.{idcol} WHERE {fts} MATCH :q"
            )
            params: Dict[str, Any] = {"q": self._fts_match_query(query), "lim": k}
            if kind == "fact":
                sql += " AND t.invalidated_at IS NULL"
                if memory_type:
                    sql += " AND t.memory_type = :mt"
                    params["mt"] = memory_type
            elif kind == "episode":
                sql += " AND t.status = 'closed'"
            if self.namespace:
                sql += " AND t.namespace = :ns"
                params["ns"] = self.namespace
            if user_id is not ANY_USER:
                if user_id is None:
                    sql += " AND t.user_id IS NULL"
                elif include_shared:
                    sql += " AND (t.user_id = :uid OR t.user_id IS NULL)"
                    params["uid"] = user_id
                else:
                    sql += " AND t.user_id = :uid"
                    params["uid"] = user_id
            sql += " ORDER BY s LIMIT :lim"
            session = self.get_session()
            try:
                rows = session.execute(text(sql), params).fetchall()
                return [(int(r[0]), float(-r[1])) for r in rows]
            except Exception as e:
                logger.debug("FTS search failed (%s); using in-process BM25", e)
            finally:
                session.close()
        index = self._sync_lexical(kind)
        if user_id is ANY_USER:
            hits = index.search(query, k * 2)
        else:
            hits = index.search(query, k * 2, user_id=user_id, include_shared=include_shared)
        if kind == "fact" and hits:
            valid = {
                m["id"]: m["type"] for m in self.hydrate_memories([h[0] for h in hits], user_id=ANY_USER)
            }
            hits = [h for h in hits if h[0] in valid and (not memory_type or valid[h[0]] == memory_type)]
        return hits[:k]

    def hydrate_memories(
        self,
        ids: Iterable[int],
        user_id: Any = ANY_USER,
        include_shared: bool = False,
        include_invalid: bool = False,
    ) -> List[Dict[str, Any]]:
        """Load memories by id, preserving order and enforcing scope/validity."""
        ids = [int(i) for i in ids]
        if not ids:
            return []
        session = self.get_session()
        try:
            q = session.query(Memory).filter(Memory.id.in_(ids))
            if not include_invalid:
                q = q.filter(Memory.invalidated_at.is_(None))
            q = self._user_filter(self._ns_filter(q, Memory), Memory, user_id, include_shared)
            by_id = {m.id: m for m in q.all()}
            return [self._memory_to_dict(by_id[i]) for i in ids if i in by_id]
        finally:
            session.close()

    def hydrate_episodes(
        self, ids: Iterable[int], user_id: Any = ANY_USER, include_shared: bool = False
    ) -> List[Dict[str, Any]]:
        ids = [int(i) for i in ids]
        if not ids:
            return []
        session = self.get_session()
        try:
            q = session.query(Episode).filter(Episode.id.in_(ids))
            q = self._user_filter(self._ns_filter(q, Episode), Episode, user_id, include_shared)
            by_id = {e.id: e for e in q.all()}
            return [self.episode_to_dict(by_id[i]) for i in ids if i in by_id]
        finally:
            session.close()

    def hydrate_turns(
        self, ids: Iterable[int], user_id: Any = ANY_USER, include_shared: bool = False
    ) -> List[Dict[str, Any]]:
        ids = [int(i) for i in ids]
        if not ids:
            return []
        session = self.get_session()
        try:
            q = session.query(Turn).filter(Turn.id.in_(ids))
            q = self._user_filter(self._ns_filter(q, Turn), Turn, user_id, include_shared)
            by_id = {t.id: t for t in q.all()}
            return [self.turn_to_dict(by_id[i]) for i in ids if i in by_id]
        finally:
            session.close()

    # ------------------------------------------------------------------ memories (write)

    def insert_memory(self, data: Dict[str, Any], dedup: bool = True) -> Tuple[int, str]:
        """
        Insert one semantic memory. Returns ``(id, op)`` with op ``"add"`` or
        ``"duplicate"`` (same normalised content, type and scope already active).
        """
        content = (data.get("content") or "").strip()
        if not content:
            raise StorageError("Memory content is empty")
        memory_type = data.get("type") or data.get("memory_type") or "fact"
        namespace = self.namespace or data.get("namespace")
        meta = dict(data.get("metadata") or {})
        user_id = data.get("user_id")
        if user_id is None:
            user_id = meta.get("canonical_user_id")
        session_id = data.get("session_id") or meta.get("session_id")
        agent_id = data.get("agent_id") or meta.get("agent_id")
        chash = content_hash(content)
        now = utcnow()

        session = self.get_session()
        try:
            if dedup:
                q = session.query(Memory).filter(
                    Memory.content_hash == chash,
                    Memory.memory_type == memory_type,
                    Memory.invalidated_at.is_(None),
                )
                q = q.filter(Memory.namespace == namespace if namespace else Memory.namespace.is_(None))
                q = self._user_filter(q, Memory, user_id)
                existing = q.first()
                if existing:
                    existing.updated_at = now
                    existing.access_count = (existing.access_count or 0) + 1
                    session.commit()
                    return existing.id, "duplicate"

            embedding = data.get("embedding")
            if embedding is not None and hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            memory = Memory(
                content=content,
                memory_type=memory_type,
                namespace=namespace,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id or user_id,
                extra_metadata=meta,
                embedding=embedding,
                content_hash=chash,
                importance_score=float(data.get("importance_score") or 0.0),
                valid_from=data.get("valid_from") or now,
                valid_to=data.get("valid_to"),
                confidence=float(data.get("confidence", 1.0)),
                source=data.get("source") or meta.get("source"),
                source_episode_id=data.get("source_episode_id"),
                source_turn_ids=data.get("source_turn_ids"),
                supersedes_id=data.get("supersedes_id"),
                created_at=now,
                updated_at=now,
            )
            session.add(memory)
            session.flush()
            self.fts_insert(session, "fact", memory.id, content)
            edges = self._provenance_edges(
                memory.id, namespace, user_id, agent_id,
                data.get("source_episode_id"), data.get("source_turn_ids"),
            )
            if data.get("supersedes_id"):
                edges.append(self._edge_dict(
                    "fact", memory.id, "supersedes", "fact", int(data["supersedes_id"]),
                    namespace=namespace, user_id=user_id, agent_id=agent_id,
                ))
            if edges:
                self.add_edges(edges, session=session)
            session.commit()
            self.index_add("fact", memory.id, user_id, content, embedding)
            return memory.id, "add"
        except Exception as e:
            session.rollback()
            raise StorageError(f"Failed to store memory: {e}") from e
        finally:
            session.close()

    async def store_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Store memories (exact duplicates are merged). Returns them with ``id``/``op``."""
        stored: List[Dict[str, Any]] = []
        for memory_data in memories or []:
            if not (memory_data.get("content") or "").strip():
                continue
            mid, op = self.insert_memory(memory_data)
            stored.append({**memory_data, "id": mid, "op": op})
        return stored

    def supersede_memory(self, old_id: int, new_data: Dict[str, Any]) -> Optional[int]:
        """
        Versioned update: insert the new statement and invalidate the old one.
        The old row is kept (``superseded_by_id`` -> new row) so history survives.
        """
        session = self.get_session()
        try:
            old = session.query(Memory).filter(Memory.id == old_id).first()
            if old is None:
                return None
            data = dict(new_data)
            data.setdefault("type", old.memory_type)
            data.setdefault("user_id", old.user_id)
            data.setdefault("session_id", old.session_id)
            meta = dict(old.extra_metadata or {})
            meta.pop("invalidation_reason", None)
            meta.update(data.get("metadata") or {})
            data["metadata"] = meta
            data["supersedes_id"] = old_id
            data.setdefault("importance_score", old.importance_score or 0.0)
        finally:
            session.close()
        new_id, _ = self.insert_memory(data, dedup=False)
        self._invalidate(old_id, superseded_by_id=new_id, reason="superseded")
        return new_id

    def _invalidate(
        self, memory_id: int, superseded_by_id: Optional[int] = None, reason: Optional[str] = None
    ) -> bool:
        session = self.get_session()
        try:
            memory = session.query(Memory).filter(Memory.id == memory_id).first()
            if memory is None or memory.invalidated_at is not None:
                return False
            now = utcnow()
            memory.invalidated_at = now
            memory.valid_to = memory.valid_to or now
            if superseded_by_id:
                memory.superseded_by_id = superseded_by_id
            if reason:
                meta = dict(memory.extra_metadata or {})
                meta["invalidation_reason"] = reason
                memory.extra_metadata = meta
            self.fts_delete(session, "fact", memory_id)
            # The fact leaves the walkable graph; provenance edges (derived_from,
            # supersedes) stay valid as history.
            self.invalidate_edges(session, node=("fact", memory_id), kinds=WALK_FACT_KINDS, at=now)
            self.invalidate_edges(session, evidence=("fact", memory_id), at=now)
            session.commit()
            self.index_remove("fact", memory_id)
            return True
        except Exception as e:
            session.rollback()
            logger.error("Failed to invalidate memory %s: %s", memory_id, e)
            return False
        finally:
            session.close()

    async def invalidate_memory(
        self, memory_id: int, superseded_by_id: Optional[int] = None, reason: Optional[str] = None
    ) -> bool:
        """Soft delete: the memory stops being retrieved but stays in history."""
        return self._invalidate(memory_id, superseded_by_id=superseded_by_id, reason=reason)

    async def update_memory_content(
        self,
        memory_id: int,
        content: str,
        invalidate_superseded: bool = True,
        embedding: Optional[Sequence[float]] = None,
    ) -> Optional[int]:
        """
        Change a memory's content. By default this creates a new version and
        invalidates the old row; returns the id of the row now holding ``content``.
        With ``invalidate_superseded=False`` the row is edited in place, and FTS
        and the embedding are kept in sync.
        """
        if invalidate_superseded:
            data: Dict[str, Any] = {"content": content}
            if embedding is not None:
                data["embedding"] = list(embedding)
            return self.supersede_memory(memory_id, data)
        session = self.get_session()
        try:
            memory = session.query(Memory).filter(Memory.id == memory_id).first()
            if memory is None:
                return None
            memory.content = content
            memory.content_hash = content_hash(content)
            memory.updated_at = utcnow()
            memory.embedding = list(embedding) if embedding is not None else None
            self.fts_delete(session, "fact", memory_id)
            self.fts_insert(session, "fact", memory_id, content)
            session.commit()
            # Indexes skip ids they already hold, so rebuild the fact indexes lazily.
            with self._sync_lock:
                self._reset_indexes("fact")
            return memory_id
        except Exception as e:
            session.rollback()
            logger.error("Failed to update memory: %s", e)
            return None
        finally:
            session.close()

    async def delete_memory(self, memory_id: int, hard: bool = True) -> bool:
        """
        Delete a memory. ``hard=True`` (default) physically removes the row and its
        graph links (right to be forgotten); ``hard=False`` only invalidates it.
        """
        if not hard:
            return self._invalidate(memory_id, reason="deleted")
        session = self.get_session()
        try:
            memory = session.query(Memory).filter(Memory.id == memory_id).first()
            if memory is None:
                return False
            self.fts_delete(session, "fact", memory_id)
            self.delete_node_edges(session, "fact", memory_id)
            session.delete(memory)
            session.commit()
            self.index_remove("fact", memory_id)
            return True
        except Exception as e:
            session.rollback()
            logger.error("Failed to delete memory: %s", e)
            return False
        finally:
            session.close()

    async def update_memory_importance(self, memory_id: int, importance_score: float) -> None:
        session = self.get_session()
        try:
            memory = session.query(Memory).filter(Memory.id == memory_id).first()
            if memory:
                memory.importance_score = importance_score
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Failed to update memory importance: %s", e)
        finally:
            session.close()

    def touch_memories(self, ids: Iterable[int]) -> None:
        """Record retrieval: bump access_count and last_accessed_at."""
        ids = [int(i) for i in ids]
        if not ids:
            return
        session = self.get_session()
        try:
            now = utcnow()
            for m in session.query(Memory).filter(Memory.id.in_(ids)).all():
                m.access_count = (m.access_count or 0) + 1
                m.last_accessed_at = now
            session.commit()
        except Exception as e:
            session.rollback()
            logger.debug("touch_memories failed: %s", e)
        finally:
            session.close()

    def memory_history(self, memory_id: int) -> List[Dict[str, Any]]:
        """Full version chain containing ``memory_id``, oldest first."""
        session = self.get_session()
        try:
            node = session.query(Memory).filter(Memory.id == memory_id).first()
            if node is None:
                return []
            seen = {node.id}
            while node.supersedes_id and node.supersedes_id not in seen:
                prev = session.query(Memory).filter(Memory.id == node.supersedes_id).first()
                if prev is None:
                    break
                seen.add(prev.id)
                node = prev
            chain = [node]
            seen = {node.id}
            while node.superseded_by_id and node.superseded_by_id not in seen:
                nxt = session.query(Memory).filter(Memory.id == node.superseded_by_id).first()
                if nxt is None:
                    break
                chain.append(nxt)
                seen.add(nxt.id)
                node = nxt
            return [self._memory_to_dict(m) for m in chain]
        finally:
            session.close()

    def forget_user(self, user_id: str) -> Dict[str, int]:
        """Hard-delete every row belonging to ``user_id`` in this namespace."""
        if not user_id:
            raise ValueError("user_id is required")
        counts: Dict[str, int] = {}
        session = self.get_session()
        try:

            def _ids(model) -> List[int]:
                q = self._ns_filter(session.query(model.id), model).filter(model.user_id == user_id)
                return [r[0] for r in q.all()]

            mem_ids, ep_ids, turn_ids, ent_ids = _ids(Memory), _ids(Episode), _ids(Turn), _ids(Entity)
            for kind, ids in (("fact", mem_ids), ("episode", ep_ids), ("turn", turn_ids)):
                for rid in ids:
                    self.fts_delete(session, kind, rid)
            counts["edges"] = (
                self._ns_filter(session.query(Edge), Edge)
                .filter(Edge.user_id == user_id)
                .delete(synchronize_session=False)
            )
            # Edges stored in the org tier that point at this user's rows
            for kind, ids in (("fact", mem_ids), ("episode", ep_ids), ("turn", turn_ids), ("entity", ent_ids)):
                for chunk in _chunks(ids, 500):
                    session.query(Edge).filter(
                        or_(
                            (Edge.src_kind == kind) & Edge.src_id.in_(chunk),
                            (Edge.dst_kind == kind) & Edge.dst_id.in_(chunk),
                        )
                    ).delete(synchronize_session=False)
            self.bump_graph_epoch(session)
            for name, model, ids in (
                ("entities", Entity, ent_ids),
                ("memories", Memory, mem_ids),
                ("episodes", Episode, ep_ids),
                ("turns", Turn, turn_ids),
            ):
                counts[name] = len(ids)
                if ids:
                    session.query(model).filter(model.id.in_(ids)).delete(synchronize_session=False)
            counts["sessions"] = (
                self._ns_filter(session.query(SessionRecord), SessionRecord)
                .filter(SessionRecord.user_id == user_id)
                .delete(synchronize_session=False)
            )
            session.query(Conversation).filter(Conversation.user_id == user_id).delete(
                synchronize_session=False
            )
            session.commit()
            for kind, ids in (("fact", mem_ids), ("episode", ep_ids), ("turn", turn_ids)):
                for rid in ids:
                    self.index_remove(kind, rid)
            return counts
        except Exception as e:
            session.rollback()
            raise StorageError(f"forget_user failed: {e}") from e
        finally:
            session.close()

    # ------------------------------------------------------------------ graph edges

    def _edge_dict(
        self, src_kind: str, src_id: int, kind: str, dst_kind: str, dst_id: int, **attrs: Any
    ) -> Dict[str, Any]:
        return {"src_kind": src_kind, "src_id": int(src_id), "kind": kind,
                "dst_kind": dst_kind, "dst_id": int(dst_id), **attrs}

    def _provenance_edges(
        self,
        fact_id: int,
        namespace: Optional[str],
        user_id: Optional[str],
        agent_id: Optional[str],
        episode_id: Optional[int],
        turn_ids: Optional[Sequence[int]],
    ) -> List[Dict[str, Any]]:
        """``fact -derived_from-> turn/episode`` edges for a fact's recorded sources."""
        out: List[Dict[str, Any]] = []
        attrs = {"namespace": namespace, "user_id": user_id, "agent_id": agent_id}
        for tid in dict.fromkeys(int(t) for t in (turn_ids or []) if t is not None):
            out.append(self._edge_dict("fact", fact_id, "derived_from", "turn", tid, **attrs))
        if episode_id:
            out.append(self._edge_dict("fact", fact_id, "derived_from", "episode", int(episode_id), **attrs))
        return out

    def add_edges(self, edges: Sequence[Dict[str, Any]], session: Optional[Session] = None) -> List[int]:
        """Insert edges (dicts with src_kind/src_id/kind/dst_kind/dst_id + attributes)."""
        if not edges:
            return []
        own = session is None
        session = session or self.get_session()
        try:
            now = utcnow()
            seq = self.next_graph_seq(session)
            rows = [
                Edge(
                    seq=seq,
                    namespace=self.namespace or e.get("namespace"),
                    user_id=e.get("user_id"),
                    agent_id=e.get("agent_id"),
                    src_kind=e["src_kind"],
                    src_id=int(e["src_id"]),
                    kind=e["kind"],
                    dst_kind=e["dst_kind"],
                    dst_id=int(e["dst_id"]),
                    predicate=e.get("predicate"),
                    role=e.get("role"),
                    weight=float(e.get("weight", 1.0)),
                    confidence=float(e.get("confidence", 1.0)),
                    valid_from=e.get("valid_from"),
                    valid_to=e.get("valid_to"),
                    created_at=now,
                    evidence_kind=e.get("evidence_kind"),
                    evidence_id=e.get("evidence_id"),
                    source=e.get("source"),
                    trust_tier=e.get("trust_tier"),
                    extra=e.get("extra"),
                )
                for e in edges
            ]
            session.add_all(rows)
            session.flush()
            ids = [r.id for r in rows]
            if own:
                session.commit()
            return ids
        except Exception:
            if own:
                session.rollback()
            raise
        finally:
            if own:
                session.close()

    @staticmethod
    def _node_clause(kind: str, node_id: int):
        return or_(
            (Edge.src_kind == kind) & (Edge.src_id == node_id),
            (Edge.dst_kind == kind) & (Edge.dst_id == node_id),
        )

    def invalidate_edges(
        self,
        session: Optional[Session] = None,
        *,
        ids: Optional[Sequence[int]] = None,
        node: Optional[Tuple[str, int]] = None,
        evidence: Optional[Tuple[str, int]] = None,
        kinds: Optional[Sequence[str]] = None,
        at: Optional[datetime] = None,
        superseded_by: Optional[int] = None,
    ) -> List[int]:
        """Retire matching valid edges (transaction time); returns their ids."""
        if ids is None and node is None and evidence is None:
            raise ValueError("invalidate_edges needs ids, node or evidence")
        own = session is None
        session = session or self.get_session()
        try:
            q = session.query(Edge).filter(Edge.invalidated_at.is_(None))
            if ids is not None:
                q = q.filter(Edge.id.in_(list(ids)))
            if node is not None:
                q = q.filter(self._node_clause(node[0], int(node[1])))
            if evidence is not None:
                q = q.filter(Edge.evidence_kind == evidence[0], Edge.evidence_id == int(evidence[1]))
            if kinds:
                q = q.filter(Edge.kind.in_(list(kinds)))
            now = at or utcnow()
            out = []
            rows = q.all()
            seq = self.next_graph_seq(session) if rows else None
            for e in rows:
                e.seq = seq
                e.invalidated_at = now
                e.valid_to = e.valid_to or now
                if superseded_by:
                    e.superseded_by_id = superseded_by
                out.append(e.id)
            if own:
                session.commit()
            return out
        except Exception:
            if own:
                session.rollback()
            raise
        finally:
            if own:
                session.close()

    def delete_node_edges(self, session: Session, kind: str, node_id: int) -> int:
        """Hard-delete every edge touching a node or asserted by it; bumps ``graph_epoch``."""
        n = (
            session.query(Edge)
            .filter(or_(self._node_clause(kind, node_id),
                        (Edge.evidence_kind == kind) & (Edge.evidence_id == node_id)))
            .delete(synchronize_session=False)
        )
        if n:
            self.bump_graph_epoch(session)
        return n

    def unsupported_dependents(
        self, kind: str, node_id: int, session: Optional[Session] = None
    ) -> List[Tuple[str, int]]:
        """
        Rows derived from ``(kind, node_id)`` that have no *other* valid source.
        Deleting the node should cascade to these (Zep rule: a derived memory survives
        only while some evidence supports it).
        """
        own = session is None
        session = session or self.get_session()
        try:
            deps = (
                session.query(Edge.src_kind, Edge.src_id)
                .filter(Edge.kind == "derived_from", Edge.dst_kind == kind, Edge.dst_id == node_id,
                        Edge.invalidated_at.is_(None))
                .distinct()
                .all()
            )
            out: List[Tuple[str, int]] = []
            for sk, sid in deps:
                other = (
                    session.query(Edge.id)
                    .filter(Edge.kind == "derived_from", Edge.src_kind == sk, Edge.src_id == sid,
                            Edge.invalidated_at.is_(None),
                            ~((Edge.dst_kind == kind) & (Edge.dst_id == node_id)))
                    .first()
                )
                if other is None:
                    out.append((sk, int(sid)))
            return out
        finally:
            if own:
                session.close()

    def graph_epoch(self) -> int:
        return self.graph_state()[0]

    def bump_graph_epoch(self, session: Session) -> None:
        self._bump(session, "graph_epoch")

    def next_graph_seq(self, session: Session) -> int:
        """
        Allocate the next graph write sequence inside the caller's transaction.

        The increment is an atomic SQL UPDATE in the same transaction as the edges.
        It takes the write lock, so sequence order is commit order and no two
        transactions share a value. Timestamps don't have that property: a
        concurrent writer can commit an older ``invalidated_at`` after a newer one,
        which an engine syncing by time would miss forever.
        """
        return self._bump(session, "graph_seq")

    def graph_state(self) -> Tuple[int, int]:
        """``(graph_epoch, graph_seq)``: one indexed read tells engines whether to sync."""
        session = self.get_session()
        try:
            vals = dict(session.query(Counter.key, Counter.value)
                        .filter(Counter.key.in_(("graph_epoch", "graph_seq"))).all())
            return int(vals.get("graph_epoch") or 0), int(vals.get("graph_seq") or 0)
        finally:
            session.close()

    @staticmethod
    def _partition_filter(q, namespace: Optional[str], user_id: Optional[str]):
        q = q.filter(Edge.namespace == namespace) if namespace else q.filter(Edge.namespace.is_(None))
        return q.filter(Edge.user_id == user_id) if user_id is not None else q.filter(Edge.user_id.is_(None))

    def partition_edges(
        self, namespace: Optional[str], user_id: Optional[str]
    ) -> List[Tuple[int, str, int, str, str, int, float]]:
        """All valid edges of one tier partition (graph-engine initial load)."""
        session = self.get_session()
        try:
            q = session.query(Edge.id, Edge.src_kind, Edge.src_id, Edge.kind, Edge.dst_kind, Edge.dst_id,
                              Edge.weight).filter(Edge.invalidated_at.is_(None))
            return [tuple(r) for r in self._partition_filter(q, namespace, user_id).order_by(Edge.id).all()]
        finally:
            session.close()

    def partition_changes(
        self, namespace: Optional[str], user_id: Optional[str], after_seq: int
    ) -> List[Tuple[int, str, int, str, str, int, float, bool]]:
        """Edges of a partition inserted or invalidated after ``after_seq`` (+ invalid flag)."""
        session = self.get_session()
        try:
            q = session.query(Edge.id, Edge.src_kind, Edge.src_id, Edge.kind, Edge.dst_kind, Edge.dst_id,
                              Edge.weight, Edge.invalidated_at).filter(Edge.seq > after_seq)
            rows = self._partition_filter(q, namespace, user_id).order_by(Edge.seq, Edge.id).all()
            return [(*r[:7], r[7] is not None) for r in rows]
        finally:
            session.close()

    # ------------------------------------------------------------------ embedding spaces

    def _space_key(self, kind: str) -> str:
        return f"embedding_space:{self.namespace or ''}:{kind}"

    def embedding_space(self, kind: str) -> Optional[str]:
        session = self.get_session()
        try:
            row = session.get(MemorableMeta, self._space_key(kind))
            return row.value if row is not None else None
        finally:
            session.close()

    def set_embedding_space(self, kind: str, space: str) -> None:
        session = self.get_session()
        try:
            row = session.get(MemorableMeta, self._space_key(kind))
            if row is None:
                session.add(MemorableMeta(key=self._space_key(kind), value=space[:255]))
            else:
                row.value = space[:255]
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()

    def has_embeddings(self, kind: str) -> bool:
        model = KIND_MODELS[kind]
        session = self.get_session()
        try:
            q = self._ns_filter(session.query(model.id).filter(model.embedding.isnot(None)), model)
            return q.first() is not None
        finally:
            session.close()

    def reembed(self, kind: str, encode, turn_context_weight: float = 0.5, batch_size: int = 256) -> int:
        """
        Recompute every stored vector of ``kind`` with ``encode(list[str]) -> ndarray``
        after an embedding-model change, so old and new vectors never mix in one index.
        Turns reproduce the ingest-time contextual blend (previous turn, same session).
        """
        model = KIND_MODELS[kind]
        session = self.get_session()
        n = 0
        try:
            q = self._ns_filter(session.query(model), model)
            if kind == "turn":
                rows = q.order_by(Turn.namespace, Turn.session_id, Turn.seq).all()
                texts = [(t.content or "")[:2000] for t in rows]
            elif kind == "episode":
                rows = q.filter(Episode.status == "closed").all()
                texts = [f"{e.title or ''}\n{e.summary or ''}\n" + "\n".join(e.key_points or []) for e in rows]
            else:
                rows = q.filter(Memory.embedding.isnot(None)).all()
                texts = [m.content for m in rows]
            prev = None
            prev_session = None
            for start in range(0, len(rows), batch_size):
                vecs = np.asarray(encode(texts[start : start + batch_size]), dtype=np.float32)
                for row, vec in zip(rows[start : start + batch_size], vecs):
                    out = vec
                    if kind == "turn":
                        sess = (row.namespace, row.session_id)
                        if turn_context_weight > 0 and prev is not None and prev_session == sess:
                            b = vec + turn_context_weight * prev
                            out = b / (np.linalg.norm(b) or 1.0)
                        prev, prev_session = vec, sess
                    row.embedding = out.tolist()
                    n += 1
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        with self._sync_lock:
            self._reset_indexes(kind)
        return n

    # ------------------------------------------------------------------ memories (read)

    async def get_memories(
        self,
        memory_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        user_id: Optional[str] = None,
        canonical_user_id: Optional[str] = None,
        include_invalid: bool = False,
        scope_user: Any = _UNSET,
        include_shared: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List memories ordered by importance.

        ``user_id``/``canonical_user_id`` filter to that user; with neither, all users
        are listed (admin view). Pass ``scope_user`` (``None`` = shared pool) for
        strict retrieval scoping instead.
        """
        session = self.get_session()
        try:
            q = session.query(Memory)
            if not include_invalid:
                q = q.filter(Memory.invalidated_at.is_(None))
            q = self._ns_filter(q, Memory)
            if memory_type:
                q = q.filter(Memory.memory_type == memory_type)
            uid = scope_user if scope_user is not _UNSET else self._resolve_user_arg(user_id, canonical_user_id)
            q = self._user_filter(q, Memory, uid, include_shared)
            rows = (
                q.order_by(Memory.importance_score.desc(), Memory.id.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )
            return [self._memory_to_dict(m) for m in rows]
        except Exception as e:
            logger.error("Failed to get memories: %s", e)
            return []
        finally:
            session.close()

    async def search_memories_text(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        user_id: Optional[str] = None,
        canonical_user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Keyword search (FTS5/BM25, then LIKE as a last resort)."""
        uid = self._resolve_user_arg(user_id, canonical_user_id)
        try:
            hits = self.lexical_search("fact", query, k=limit, user_id=uid, memory_type=memory_type)
            if hits:
                return self.hydrate_memories([h[0] for h in hits], user_id=uid)[:limit]
            session = self.get_session()
            try:
                q = session.query(Memory).filter(
                    Memory.content.like(f"%{query}%"), Memory.invalidated_at.is_(None)
                )
                q = self._user_filter(self._ns_filter(q, Memory), Memory, uid)
                if memory_type:
                    q = q.filter(Memory.memory_type == memory_type)
                rows = q.order_by(Memory.importance_score.desc()).limit(limit).all()
                return [self._memory_to_dict(m) for m in rows]
            finally:
                session.close()
        except Exception as e:
            logger.error("Failed to search memories: %s", e)
            return []

    # ------------------------------------------------------------------ sessions & summaries

    def upsert_session(self, session_id: str, user_id: Optional[str], turns_added: int = 0) -> None:
        session = self.get_session()
        try:
            rec = (
                self._ns_filter(session.query(SessionRecord), SessionRecord)
                .filter(SessionRecord.session_id == session_id)
                .first()
            )
            now = utcnow()
            if rec is None:
                session.add(
                    SessionRecord(
                        namespace=self.namespace,
                        session_id=session_id,
                        user_id=user_id,
                        started_at=now,
                        last_active_at=now,
                        turn_count=turns_added,
                    )
                )
            else:
                rec.last_active_at = now
                rec.turn_count = (rec.turn_count or 0) + turns_added
                if user_id and not rec.user_id:
                    rec.user_id = user_id
            session.commit()
        except Exception as e:
            session.rollback()
            logger.debug("upsert_session failed: %s", e)
        finally:
            session.close()

    def get_session_record(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.get_session()
        try:
            rec = (
                self._ns_filter(session.query(SessionRecord), SessionRecord)
                .filter(SessionRecord.session_id == session_id)
                .first()
            )
            if rec is None:
                return None
            return {
                "session_id": rec.session_id,
                "user_id": rec.user_id,
                "summary": rec.summary,
                "turn_count": rec.turn_count,
                "started_at": _iso(rec.started_at),
                "last_active_at": _iso(rec.last_active_at),
                "ended_at": _iso(rec.ended_at),
            }
        finally:
            session.close()

    async def store_summary(self, key: str, summary: str, session_id: Optional[str] = None) -> None:
        """Persist the rolling summary for one session (``key`` is ``namespace:session``)."""
        sid = session_id or (key.split(":", 1)[1] if ":" in key else key)
        session = self.get_session()
        try:
            rec = (
                self._ns_filter(session.query(SessionRecord), SessionRecord)
                .filter(SessionRecord.session_id == sid)
                .first()
            )
            if rec is None:
                rec = SessionRecord(namespace=self.namespace, session_id=sid)
                session.add(rec)
            rec.summary = summary
            rec.last_active_at = utcnow()
            session.commit()
        except Exception as e:
            session.rollback()
            logger.debug("store_summary failed: %s", e)
        finally:
            session.close()

    async def get_session_summary(self, session_id: str) -> Optional[str]:
        rec = self.get_session_record(session_id)
        return rec.get("summary") if rec else None

    async def store_conversation(
        self,
        messages: List[Dict[str, Any]],
        response: Optional[Any] = None,
        extracted_memories: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Legacy per-call log (L0 lives in ``turns``)."""

        def _jsonable(obj: Any) -> Any:
            try:
                return json.loads(json.dumps(obj, default=str))
            except Exception:
                return str(obj)

        session = self.get_session()
        try:
            session.add(
                Conversation(
                    namespace=self.namespace,
                    messages=_jsonable(messages),
                    response=_jsonable(response),
                    extracted_memories=_jsonable(
                        [
                            {k: v for k, v in m.items() if k != "embedding"}
                            for m in extracted_memories or []
                            if isinstance(m, dict)
                        ]
                    ),
                    session_id=session_id,
                    user_id=user_id,
                )
            )
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Failed to store conversation: %s", e)
        finally:
            session.close()

    async def get_recent_conversations(self, limit: int = 5) -> List[Dict[str, Any]]:
        session = self.get_session()
        try:
            q = self._ns_filter(session.query(Conversation), Conversation)
            rows = q.order_by(Conversation.created_at.desc()).limit(limit).all()
            return [
                {
                    "id": c.id,
                    "summary": c.summary,
                    "session_id": c.session_id,
                    "created_at": _iso(c.created_at),
                }
                for c in rows
            ]
        finally:
            session.close()

    # ------------------------------------------------------------------ serialisation

    def _memory_to_dict(self, m: Memory) -> Dict[str, Any]:
        return {
            "kind": "fact",
            "id": m.id,
            "content": m.content,
            "type": m.memory_type,
            "metadata": m.extra_metadata or {},
            "embedding": m.embedding,
            "importance_score": m.importance_score,
            "access_count": m.access_count or 0,
            "created_at": _iso(m.created_at),
            "updated_at": _iso(m.updated_at),
            "valid_from": _iso(m.valid_from),
            "valid_to": _iso(m.valid_to),
            "invalidated_at": _iso(m.invalidated_at),
            "superseded_by_id": m.superseded_by_id,
            "supersedes_id": m.supersedes_id,
            "confidence": m.confidence,
            "session_id": m.session_id,
            "user_id": m.user_id,
            "agent_id": getattr(m, "agent_id", None),
            "source": m.source,
            "source_episode_id": m.source_episode_id,
            "source_turn_ids": m.source_turn_ids,
        }

    @staticmethod
    def episode_to_dict(e: Episode) -> Dict[str, Any]:
        return {
            "kind": "episode",
            "id": e.id,
            "session_id": e.session_id,
            "user_id": e.user_id,
            "agent_id": getattr(e, "agent_id", None),
            "status": e.status,
            "title": e.title,
            "summary": e.summary,
            "key_points": e.key_points or [],
            "entities": e.entity_names or [],
            "start_time": _iso(e.start_time),
            "end_time": _iso(e.end_time),
            "start_seq": e.start_seq,
            "end_seq": e.end_seq,
            "turn_count": e.turn_count,
            "boundary_reason": e.boundary_reason,
        }

    @staticmethod
    def turn_to_dict(t: Turn) -> Dict[str, Any]:
        return {
            "kind": "turn",
            "id": t.id,
            "session_id": t.session_id,
            "user_id": t.user_id,
            "agent_id": getattr(t, "agent_id", None),
            "seq": t.seq,
            "role": t.role,
            "content": t.content,
            "episode_id": t.episode_id,
            "occurred_at": _iso(t.occurred_at),
        }

    def get_stats(self) -> Dict[str, Any]:
        session = self.get_session()
        try:
            total = session.query(Memory).count()
            ns_q = self._ns_filter(session.query(Memory), Memory)
            return {
                "total_memories": total,
                "namespace_memories": ns_q.count() if self.namespace else total,
                "active_memories": ns_q.filter(Memory.invalidated_at.is_(None)).count(),
                "episodes": self._ns_filter(session.query(Episode), Episode).count(),
                "turns": self._ns_filter(session.query(Turn), Turn).count(),
                "sessions": self._ns_filter(session.query(SessionRecord), SessionRecord).count(),
                "entities": self._ns_filter(session.query(Entity), Entity).count(),
                "relations": self._ns_filter(session.query(Edge), Edge)
                .filter(Edge.kind == "relates", Edge.invalidated_at.is_(None))
                .count(),
                "edges": self._ns_filter(session.query(Edge), Edge).filter(Edge.invalidated_at.is_(None)).count(),
                "namespace": self.namespace,
                "fts": self._fts_enabled,
            }
        except Exception as e:
            logger.error("Failed to get stats: %s", e)
            return {}
        finally:
            session.close()
