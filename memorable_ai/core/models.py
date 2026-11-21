"""
SQL schema for the Memorable memory model.

Layers (all in one SQL database):

    L0  turns            verbatim messages, append-only; ground truth and provenance
    L1  episodes         contiguous segments of a session, split at topic/time boundaries,
                         with a title, summary and extracted key points
        sessions         one row per (namespace, session_id): rolling summary, activity
    L2  memories         semantic facts, versioned bi-temporally (valid_* = when true in
                         the world, created_at/invalidated_at = when we learned/retracted it)
        entities         knowledge-graph nodes
        identity_aliases session/device aliases -> canonical user id
    graph edges          ONE typed, bi-temporal edge table linking rows of every layer:
                         relates (entity->entity, with predicate), about (fact->entity),
                         mentions (episode/turn->entity), derived_from (fact->turn/episode),
                         part_of (turn->episode->session), next (episode->episode),
                         supersedes (fact->fact). Nodes are the rows of the tables above,
                         addressed as (kind, id). See docs/memory_graph_design.md.
        memorable_meta   small key/value table (graph epoch for cache invalidation)

Every row carries ``namespace`` and ``user_id`` so all queries can be scoped in SQL.
``user_id`` NULL is the shared ``org`` tier of the namespace. ``agent_id`` records which
agent produced a row; it is an attribute for ranking/filtering, not an isolation wall.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    JSON,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def utcnow() -> datetime:
    """Naive UTC now (datetime.utcnow is deprecated in Python 3.12+)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Memory(Base):
    """L2 semantic memory: one versioned fact/preference/skill/rule/context statement."""

    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    memory_type = Column(String(50), nullable=False, index=True)
    namespace = Column(String(255), index=True)
    user_id = Column(String(255), nullable=True, index=True)
    agent_id = Column(String(255), nullable=True)
    session_id = Column(String(255), nullable=True, index=True)
    extra_metadata = Column(JSON)
    embedding = Column(JSON)
    content_hash = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, default=utcnow, index=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    last_accessed_at = Column(DateTime, nullable=True)
    access_count = Column(Integer, default=0)
    importance_score = Column(Float, default=0.0, index=True)
    confidence = Column(Float, default=1.0)
    # Bi-temporal validity (Zep/Graphiti style)
    valid_from = Column(DateTime, nullable=True)
    valid_to = Column(DateTime, nullable=True)
    invalidated_at = Column(DateTime, nullable=True, index=True)
    superseded_by_id = Column(Integer, nullable=True)
    supersedes_id = Column(Integer, nullable=True)
    # Provenance
    source = Column(String(32), nullable=True)  # conversation | manual | import | consolidation
    source_episode_id = Column(Integer, nullable=True, index=True)
    source_turn_ids = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_memory_type_namespace", "memory_type", "namespace"),
        Index("idx_created_at", "created_at"),
        Index("idx_memory_scope", "namespace", "user_id", "invalidated_at"),
    )


class Conversation(Base):
    """Legacy per-call log (kept for backward compatibility; ``turns`` is the L0 store)."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(255), index=True)
    messages = Column(JSON, nullable=False)
    response = Column(JSON)
    extracted_memories = Column(JSON)
    summary = Column(Text, nullable=True)
    session_id = Column(String(255), nullable=True, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    created_at = Column(DateTime, default=utcnow, index=True)

    __table_args__ = (Index("idx_namespace_created", "namespace", "created_at"),)


class SessionRecord(Base):
    """One conversation session. Holds the rolling summary keyed by session (not namespace)."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(255), index=True)
    session_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    agent_id = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)
    turn_count = Column(Integer, default=0)
    started_at = Column(DateTime, default=utcnow)
    last_active_at = Column(DateTime, default=utcnow, index=True)
    ended_at = Column(DateTime, nullable=True)
    extra_metadata = Column(JSON, nullable=True)

    __table_args__ = (UniqueConstraint("namespace", "session_id", name="uq_session_ns"),)


class Turn(Base):
    """L0: one verbatim message in a session."""

    __tablename__ = "turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(255), index=True)
    user_id = Column(String(255), nullable=True, index=True)
    agent_id = Column(String(255), nullable=True)
    session_id = Column(String(255), nullable=False, index=True)
    seq = Column(Integer, nullable=False)
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    episode_id = Column(Integer, nullable=True, index=True)
    occurred_at = Column(DateTime, default=utcnow, index=True)
    created_at = Column(DateTime, default=utcnow)
    embedding = Column(JSON, nullable=True)
    extra_metadata = Column(JSON, nullable=True)
    # Procedural P1 (docs/procedural_memory_design.md §4.1): tool events live in L0 too.
    kind = Column(String(16), nullable=True, default="message")  # message | tool_call | tool_result; NULL = message
    tool_name = Column(String(255), nullable=True)
    tool_call_id = Column(String(255), nullable=True)
    payload = Column(JSON, nullable=True)  # redacted args / result; head+tail+sha256 above max_payload_bytes
    salient_values = Column(JSON, nullable=True)  # values extracted before any size cut (value flow)
    is_error = Column(Boolean, nullable=True, default=False)
    task_id = Column(Integer, nullable=True, index=True)
    source = Column(String(32), nullable=True)  # user | assistant | tool | import | otel
    trust_tier = Column(String(32), nullable=True)

    __table_args__ = (Index("idx_turn_session_seq", "namespace", "session_id", "seq"),)


class Task(Base):
    """
    L1 (procedural P1): one goal pursued with tools, segmented from the event stream
    or declared with ``engine.task()`` (docs/procedural_memory_design.md §4.2, §6).
    """

    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(255), index=True)
    user_id = Column(String(255), nullable=True, index=True)
    agent_id = Column(String(255), nullable=True)
    session_id = Column(String(255), nullable=False, index=True)
    episode_id = Column(Integer, nullable=True)
    parent_task_id = Column(Integer, nullable=True)
    goal_text = Column(Text, nullable=True)
    explicit = Column(Boolean, default=False)
    status = Column(String(16), default="open", index=True)  # open | closed
    close_reason = Column(String(32), nullable=True)  # final_answer | next_user_message | idle | explicit
    outcome = Column(String(16), default="unknown")  # success | failure | unknown
    outcome_confidence = Column(Float, default=0.0)
    outcome_source = Column(String(16), nullable=True)  # explicit | signal | feedback | judge
    outcome_reason = Column(Text, nullable=True)
    feedback_seen = Column(Boolean, default=False)
    offered_tools = Column(JSON, nullable=True)
    used_tools = Column(JSON, nullable=True)
    llm_calls = Column(Integer, default=0)
    tool_calls = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    started_at = Column(DateTime, nullable=True, index=True)
    last_event_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    __table_args__ = (Index("idx_task_session", "namespace", "session_id", "status"),)


class Episode(Base):
    """L1: a contiguous, topically coherent segment of a session."""

    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(255), index=True)
    user_id = Column(String(255), nullable=True, index=True)
    agent_id = Column(String(255), nullable=True)
    session_id = Column(String(255), nullable=False, index=True)
    status = Column(String(16), default="open", index=True)  # open | closed
    title = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)
    key_points = Column(JSON, nullable=True)
    entity_names = Column(JSON, nullable=True)
    start_seq = Column(Integer, nullable=True)
    end_seq = Column(Integer, nullable=True)
    start_time = Column(DateTime, nullable=True, index=True)
    end_time = Column(DateTime, nullable=True)
    turn_count = Column(Integer, default=0)
    boundary_reason = Column(String(32), nullable=True)  # why the *previous* episode ended
    # Running statistics for surprise-based segmentation while the episode is open
    centroid = Column(JSON, nullable=True)
    centroid_n = Column(Integer, default=0)
    surprise_mean = Column(Float, default=0.0)
    surprise_m2 = Column(Float, default=0.0)
    surprise_n = Column(Integer, default=0)
    embedding = Column(JSON, nullable=True)
    importance_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=utcnow)
    closed_at = Column(DateTime, nullable=True)
    extra_metadata = Column(JSON, nullable=True)

    __table_args__ = (Index("idx_episode_scope", "namespace", "user_id", "status"),)


class Entity(Base):
    """Knowledge-graph node (person, place, organisation, concept, ...)."""

    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(255), index=True)
    user_id = Column(String(255), nullable=True, index=True)
    agent_id = Column(String(255), nullable=True)
    name = Column(String(255), nullable=False)
    norm_name = Column(String(255), nullable=False, index=True)
    entity_type = Column(String(50), default="concept")
    is_self = Column(Integer, default=0)  # 1 for the implicit "User" node
    mention_count = Column(Integer, default=0)
    first_seen = Column(DateTime, default=utcnow)
    last_seen = Column(DateTime, default=utcnow)
    extra_metadata = Column(JSON, nullable=True)
    # G1: a fact sheet rendered from the entity's valid relations (no LLM), refreshed on write
    summary = Column(Text, nullable=True)
    summary_updated_at = Column(DateTime, nullable=True)
    # Set when this entity was merged into another (an ``alias_of`` edge records the merge)
    merged_into_id = Column(Integer, nullable=True)

    __table_args__ = (Index("idx_entity_scope_name", "namespace", "user_id", "norm_name"),)


class EntityAlias(Base):
    """Another surface form of an entity ("Bob" for "Robert Chen"), used by entity resolution."""

    __tablename__ = "entity_aliases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(255), nullable=True)
    user_id = Column(String(255), nullable=True)
    entity_id = Column(Integer, nullable=False, index=True)
    alias = Column(String(255), nullable=False)  # normalised, as Entity.norm_name
    source = Column(String(16), nullable=True)  # surface | nickname | partial | llm | merge | user
    created_at = Column(DateTime, default=utcnow)

    __table_args__ = (Index("idx_entity_alias_scope", "namespace", "user_id", "alias"),)


# Node kinds addressed by edges. Each maps to a table above (or a later layer).
NODE_KINDS = ("turn", "session", "episode", "task", "entity", "fact", "procedure", "step", "insight", "community")


class Edge(Base):
    """
    One typed edge of the memory graph.

    ``valid_from``/``valid_to`` are event time (when it was true); ``created_at``/
    ``invalidated_at`` are transaction time (when we learned / retired it).
    ``evidence_*`` points at the row that asserted the edge (e.g. the fact behind a
    ``relates`` edge), so deleting evidence can cascade.
    """

    __tablename__ = "edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(255), index=True)
    user_id = Column(String(255), nullable=True, index=True)
    agent_id = Column(String(255), nullable=True)
    src_kind = Column(String(16), nullable=False)
    src_id = Column(Integer, nullable=False)
    kind = Column(String(24), nullable=False)
    dst_kind = Column(String(16), nullable=False)
    dst_id = Column(Integer, nullable=False)
    predicate = Column(String(100), nullable=True)
    role = Column(String(32), nullable=True)
    weight = Column(Float, default=1.0)
    confidence = Column(Float, default=1.0)
    valid_from = Column(DateTime, nullable=True)
    valid_to = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    invalidated_at = Column(DateTime, nullable=True, index=True)
    superseded_by_id = Column(Integer, nullable=True)
    # Graph write sequence of the transaction that last inserted/invalidated this edge.
    # Allocated under the DB write lock, so it follows commit order (unlike timestamps).
    seq = Column(Integer, nullable=True)
    evidence_kind = Column(String(16), nullable=True)
    evidence_id = Column(Integer, nullable=True)
    source = Column(String(16), nullable=True)
    trust_tier = Column(String(16), nullable=True)
    extra = Column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_edges_src", "src_kind", "src_id", "kind"),
        Index("ix_edges_dst", "dst_kind", "dst_id", "kind"),
        Index("ix_edges_scope", "namespace", "user_id", "kind", "invalidated_at"),
        Index("ix_edges_evidence", "evidence_kind", "evidence_id"),
        Index("ix_edges_predicate", "kind", "predicate"),
        Index("ix_edges_sync", "namespace", "user_id", "seq"),
    )


class MemorableMeta(Base):
    """Tiny key/value table for settings and markers."""

    __tablename__ = "memorable_meta"

    key = Column(String(64), primary_key=True)
    value = Column(String(255), nullable=True)


class Counter(Base):
    """
    Integer counters bumped with an atomic ``UPDATE ... SET value = value + 1``.

    ``graph_seq`` orders graph writes and ``graph_epoch`` signals hard deletes. The
    increment runs in SQL, not as a Python read-modify-write: two concurrent writers
    must never end up with the same value (found by the concurrency test).
    """

    __tablename__ = "memorable_counters"

    key = Column(String(64), primary_key=True)
    value = Column(Integer, nullable=False, default=0)


class IdentityAlias(Base):
    """Persisted session/device alias -> canonical user id."""

    __tablename__ = "identity_aliases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(255), nullable=True, index=True)
    alias = Column(String(255), nullable=False, index=True)
    canonical_user_id = Column(String(255), nullable=False)
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=utcnow)

    __table_args__ = (UniqueConstraint("namespace", "alias", name="uq_alias_ns"),)
