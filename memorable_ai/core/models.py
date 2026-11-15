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
        relations        knowledge-graph edges (subject, predicate, object), versioned like facts
        mentions         links entities to the facts / episodes / turns that mention them
        identity_aliases session/device aliases -> canonical user id

Every row carries ``namespace`` and ``user_id`` so all queries can be scoped in SQL.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
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

    __table_args__ = (Index("idx_turn_session_seq", "namespace", "session_id", "seq"),)


class Episode(Base):
    """L1: a contiguous, topically coherent segment of a session."""

    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(255), index=True)
    user_id = Column(String(255), nullable=True, index=True)
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
    name = Column(String(255), nullable=False)
    norm_name = Column(String(255), nullable=False, index=True)
    entity_type = Column(String(50), default="concept")
    is_self = Column(Integer, default=0)  # 1 for the implicit "User" node
    mention_count = Column(Integer, default=0)
    first_seen = Column(DateTime, default=utcnow)
    last_seen = Column(DateTime, default=utcnow)
    extra_metadata = Column(JSON, nullable=True)

    __table_args__ = (Index("idx_entity_scope_name", "namespace", "user_id", "norm_name"),)


class Relation(Base):
    """Knowledge-graph edge with bi-temporal validity and provenance."""

    __tablename__ = "relations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(255), index=True)
    user_id = Column(String(255), nullable=True, index=True)
    subject_id = Column(Integer, ForeignKey("entities.id"), nullable=False, index=True)
    predicate = Column(String(100), nullable=False, index=True)
    object_id = Column(Integer, ForeignKey("entities.id"), nullable=False, index=True)
    memory_id = Column(Integer, nullable=True, index=True)
    episode_id = Column(Integer, nullable=True)
    confidence = Column(Float, default=1.0)
    valid_from = Column(DateTime, default=utcnow)
    valid_to = Column(DateTime, nullable=True)
    invalidated_at = Column(DateTime, nullable=True, index=True)
    superseded_by_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=utcnow)


class Mention(Base):
    """Entity <-> evidence link. Exactly one of memory_id / episode_id / turn_id is set."""

    __tablename__ = "mentions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(Integer, ForeignKey("entities.id"), nullable=False, index=True)
    memory_id = Column(Integer, nullable=True, index=True)
    episode_id = Column(Integer, nullable=True, index=True)
    turn_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime, default=utcnow)


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
