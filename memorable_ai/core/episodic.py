"""
Episodic memory: verbatim turns (L0) grouped into episodes (L1).

Each session is an append-only sequence of turns. Turns are grouped into
**episodes**, contiguous and topically coherent segments, using event-segmentation
boundaries in the spirit of EM-LLM (Fountas et al., ICLR 2025), HiMem and ES-Mem
(2026):

* **time gap**: an idle gap longer than ``gap_minutes`` always starts a new episode
* **topic shift (semantic surprise)**: for each substantive user turn,
  ``s = 1 - cos(turn, episode_centroid)``. A boundary is placed when
  ``s > max(mean_s + z * std_s, min_surprise)``, with the mean and std of surprise
  tracked online (Welford) inside the current episode. So the threshold adapts
  to how varied the conversation already is
* **length cap**: ``max_turns``

A boundary is only ever placed *before* a user turn, so a reply never gets
separated from its question.

When an episode closes it gets a title, a short summary and **key points**: the
few utterances that carry the episode. With an LLM these come from one JSON call.
Without one they are chosen extractively: utterances are scored by centrality
(cosine to the episode centroid), information density (names, numbers, dates)
and speaker, then picked with Maximal Marginal Relevance (Carbonell & Goldstein,
1998) for coverage without redundancy.

Clients usually resend the whole conversation on every call. Only the
genuinely new suffix is appended, found by matching a hash of each message against
the stored tail, which also tolerates clients that truncate old history.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from memorable_ai.core.models import Episode, SessionRecord, Turn
from memorable_ai.core.scope import Scope
from memorable_ai.core.storage import episode_search_text
from memorable_ai.core.vector_index import tokenize

from memorable_ai.core.models import utcnow  # noqa: E402

logger = logging.getLogger(__name__)

CONVERSATION_ROLES = {"user", "assistant"}
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_INFO_RE = re.compile(r"\b(?:[A-Z][a-z]+|\d[\d,./:-]*|\d{4}-\d{2}-\d{2})\b")

EPISODE_PROMPT = """Summarize this conversation segment for long-term memory.
Date: {date}

{turns}

Return JSON only:
{{"title": "5-10 word title", "summary": "2-4 sentences: what happened, decisions, outcomes; include names, numbers and dates",
  "key_points": ["up to 5 standalone statements worth recalling later"],
  "entities": ["people, places, organizations, products mentioned"]}}"""


@dataclass
class EpisodicConfigView:
    enabled: bool = True
    gap_minutes: float = 30.0
    min_turns: int = 4
    max_turns: int = 40
    surprise_z: float = 2.0
    min_surprise: float = 0.45
    index_turns: bool = True
    turn_context_weight: float = 0.5


@dataclass
class IngestResult:
    new_turns: List[Dict[str, Any]] = field(default_factory=list)
    closed_episode_ids: List[int] = field(default_factory=list)
    open_episode_id: Optional[int] = None


def turn_hash(role: str, content: str) -> str:
    return hashlib.sha1(f"{role}\x00{content.strip()}".encode("utf-8")).hexdigest()


def message_text(content: Any) -> str:
    """Flatten OpenAI/Anthropic content (string or list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") in ("text", "input_text", "output_text") or "text" in p:
                    parts.append(str(p.get("text", "")))
            else:
                text = getattr(p, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(x for x in parts if x)
    return str(content)


def normalize_messages(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    """Keep user/assistant turns with text, as ``{role, content, occurred_at?}``."""
    out: List[Dict[str, Any]] = []
    for m in messages or []:
        if isinstance(m, str):
            out.append({"role": "user", "content": m})
            continue
        if not isinstance(m, dict):
            role = getattr(m, "role", None)
            content = getattr(m, "content", None)
            m = {"role": role, "content": content}
        role = m.get("role")
        if role not in CONVERSATION_ROLES:
            continue
        text = message_text(m.get("content")).strip()
        if not text:
            continue
        item: Dict[str, Any] = {"role": role, "content": text}
        ts = m.get("timestamp") or m.get("occurred_at") or (m.get("metadata") or {}).get("timestamp")
        if ts:
            item["occurred_at"] = _parse_time(ts)
        out.append(item)
    return out


def new_suffix_start(stored: Sequence[str], incoming: Sequence[str]) -> int:
    """
    Index in ``incoming`` where unseen turns begin, given the stored hash sequence.
    Handles full-history resends, truncated windows and single-message calls.
    """
    if not stored or not incoming:
        return 0
    n = len(stored)
    if len(incoming) >= n and list(incoming[:n]) == list(stored):
        return n
    for j in range(min(n, len(incoming)), 0, -1):
        if list(incoming[:j]) == list(stored[-j:]):
            return j
    last = stored[-1]
    for i in range(len(incoming) - 1, -1, -1):
        if incoming[i] == last:
            return i + 1
    return 0


def _parse_time(ts: Any) -> Optional[datetime]:
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None)
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, timezone.utc).replace(tzinfo=None)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
    return None


class EpisodicMemory:
    """Turn store + online episode segmentation + episode summarisation."""

    def __init__(
        self,
        storage: Any,
        embedder: Any = None,
        llm_model: Optional[str] = None,
        config: Any = None,
        knowledge_graph: Any = None,
    ):
        from memorable_ai.core.embeddings import as_embedder

        self.storage = storage
        self.embedder = as_embedder(embedder)
        self.llm_model = llm_model
        cfg = config or EpisodicConfigView()
        self.cfg = EpisodicConfigView(
            enabled=getattr(cfg, "enabled", True),
            gap_minutes=getattr(cfg, "gap_minutes", 30.0),
            min_turns=getattr(cfg, "min_turns", 4),
            max_turns=getattr(cfg, "max_turns", 40),
            surprise_z=getattr(cfg, "surprise_z", 2.0),
            min_surprise=getattr(cfg, "min_surprise", 0.45),
            index_turns=getattr(cfg, "index_turns", True),
            turn_context_weight=getattr(cfg, "turn_context_weight", 0.5),
        )
        self.knowledge_graph = knowledge_graph

    # ------------------------------------------------------------------ ingest

    def _session_query(self, session, model, scope: Scope):
        q = session.query(model).filter(model.session_id == scope.session_id)
        if self.storage.namespace:
            q = q.filter(model.namespace == self.storage.namespace)
        return q

    def stored_hashes(self, scope: Scope) -> List[str]:
        session = self.storage.get_session()
        try:
            rows = (
                self._session_query(session, Turn, scope)
                .with_entities(Turn.content_hash)
                .order_by(Turn.seq)
                .all()
            )
            return [r[0] for r in rows]
        finally:
            session.close()

    async def ingest(
        self,
        scope: Scope,
        messages: Sequence[Any],
        response_text: Optional[str] = None,
        occurred_at: Optional[datetime] = None,
    ) -> IngestResult:
        """Append unseen turns for ``scope.session_id`` and update segmentation."""
        if not scope.session_id:
            raise ValueError("EpisodicMemory.ingest requires scope.session_id")
        incoming = normalize_messages(messages)
        if response_text and response_text.strip():
            incoming.append({"role": "assistant", "content": response_text.strip()})
        if not incoming:
            return IngestResult()
        hashes = [turn_hash(t["role"], t["content"]) for t in incoming]
        start = new_suffix_start(self.stored_hashes(scope), hashes)
        new = incoming[start:]
        if not new:
            return IngestResult()
        now = occurred_at or utcnow()
        for t in new:
            t.setdefault("occurred_at", now)
            if t["occurred_at"] is None:
                t["occurred_at"] = now

        vecs: Optional[np.ndarray] = None
        if self.embedder is not None:
            try:
                vecs = self.embedder.encode([t["content"][:2000] for t in new])
            except Exception as e:
                logger.debug("Turn embedding failed: %s", e)

        result = IngestResult()
        session = self.storage.get_session()
        added: List[Tuple[Turn, Optional[np.ndarray]]] = []
        try:
            last = self._session_query(session, Turn, scope).order_by(Turn.seq.desc()).first()
            seq = (last.seq if last else -1) + 1
            ep = (
                self._session_query(session, Episode, scope)
                .filter(Episode.status == "open")
                .order_by(Episode.id.desc())
                .first()
            )
            prev_vec: Optional[np.ndarray] = None
            if last is not None and last.embedding and self.cfg.turn_context_weight > 0:
                prev_vec = np.asarray(last.embedding, dtype=np.float32)
            for i, t in enumerate(new):
                vec = vecs[i] if vecs is not None else None
                index_vec = vec
                w = self.cfg.turn_context_weight
                if vec is not None and w > 0 and prev_vec is not None and prev_vec.size == vec.size:
                    # Contextual turn embedding: blend in the preceding turn so replies like
                    # "Yes, last Tuesday!" stay findable by what they answer.
                    blended = vec + w * prev_vec
                    index_vec = blended / (np.linalg.norm(blended) or 1.0)
                prev_vec = vec
                reason = self._boundary(ep, t, vec) if ep is not None else None
                if ep is None or reason:
                    if ep is not None:
                        self._mark_closed(ep, t["occurred_at"])
                        result.closed_episode_ids.append(ep.id)
                    ep = Episode(
                        namespace=self.storage.namespace,
                        user_id=scope.user_id,
                        session_id=scope.session_id,
                        status="open",
                        start_seq=seq,
                        start_time=t["occurred_at"],
                        turn_count=0,
                        boundary_reason=reason or "session_start",
                        created_at=utcnow(),
                    )
                    session.add(ep)
                    session.flush()
                row = Turn(
                    namespace=self.storage.namespace,
                    user_id=scope.user_id,
                    session_id=scope.session_id,
                    seq=seq,
                    role=t["role"],
                    content=t["content"],
                    content_hash=turn_hash(t["role"], t["content"]),
                    episode_id=ep.id,
                    occurred_at=t["occurred_at"],
                    created_at=utcnow(),
                    embedding=index_vec.tolist() if (index_vec is not None and self.cfg.index_turns) else None,
                )
                session.add(row)
                session.flush()
                if self.cfg.index_turns:
                    self.storage.fts_insert(session, "turn", row.id, row.content)
                self._update_stats(ep, t, vec)
                ep.end_seq = seq
                ep.end_time = t["occurred_at"]
                ep.turn_count = (ep.turn_count or 0) + 1
                added.append((row, index_vec))
                seq += 1
            result.open_episode_id = ep.id if ep is not None else None
            rec = (
                self._session_query(session, SessionRecord, scope).first()
            )
            if rec is None:
                session.add(
                    SessionRecord(
                        namespace=self.storage.namespace,
                        session_id=scope.session_id,
                        user_id=scope.user_id,
                        started_at=new[0]["occurred_at"],
                        last_active_at=new[-1]["occurred_at"],
                        turn_count=len(new),
                    )
                )
            else:
                rec.last_active_at = new[-1]["occurred_at"]
                rec.turn_count = (rec.turn_count or 0) + len(new)
                rec.ended_at = None
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        for row, vec in added:
            if self.cfg.index_turns:
                self.storage.index_add("turn", row.id, row.user_id, row.content, vec)
            result.new_turns.append(
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "occurred_at": row.occurred_at,
                    "episode_id": row.episode_id,
                    "seq": row.seq,
                }
            )
        return result

    def _boundary(self, ep: Episode, turn: Dict[str, Any], vec: Optional[np.ndarray]) -> Optional[str]:
        if not ep.turn_count:
            return None
        if ep.end_time and turn["occurred_at"] - ep.end_time > timedelta(minutes=self.cfg.gap_minutes):
            return "time_gap"
        if turn["role"] != "user":
            return None
        if ep.turn_count >= self.cfg.max_turns:
            return "max_length"
        if vec is None or not ep.centroid or ep.turn_count < self.cfg.min_turns:
            return None
        if len(tokenize(turn["content"])) < 4:
            return None  # "ok thanks" is not a topic change
        s = self._surprise(ep, vec)
        if s is None or (ep.surprise_n or 0) < 2:
            return None
        n = ep.surprise_n or 0
        var = (ep.surprise_m2 or 0.0) / (n - 1) if n > 1 else 0.0
        threshold = max((ep.surprise_mean or 0.0) + self.cfg.surprise_z * math.sqrt(max(var, 0.0)), self.cfg.min_surprise)
        return "topic_shift" if s > threshold else None

    @staticmethod
    def _surprise(ep: Episode, vec: np.ndarray) -> Optional[float]:
        c = np.asarray(ep.centroid or [], dtype=np.float32)
        if c.size != vec.size:
            return None
        cn = float(np.linalg.norm(c))
        if cn == 0:
            return None
        return float(1.0 - np.dot(vec, c) / cn)

    def _update_stats(self, ep: Episode, turn: Dict[str, Any], vec: Optional[np.ndarray]) -> None:
        if vec is None:
            return
        if turn["role"] == "user" and ep.centroid and len(tokenize(turn["content"])) >= 4:
            s = self._surprise(ep, vec)
            if s is not None:
                n = (ep.surprise_n or 0) + 1
                delta = s - (ep.surprise_mean or 0.0)
                mean = (ep.surprise_mean or 0.0) + delta / n
                ep.surprise_m2 = (ep.surprise_m2 or 0.0) + delta * (s - mean)
                ep.surprise_mean = mean
                ep.surprise_n = n
        n = ep.centroid_n or 0
        c = np.asarray(ep.centroid, dtype=np.float32) if ep.centroid else None
        if c is None or c.size != vec.size:
            ep.centroid = vec.tolist()
            ep.centroid_n = 1
        else:
            ep.centroid = ((c * n + vec) / (n + 1)).tolist()
            ep.centroid_n = n + 1

    @staticmethod
    def _mark_closed(ep: Episode, at: Optional[datetime] = None) -> None:
        ep.status = "closing"
        ep.closed_at = at or utcnow()

    # ------------------------------------------------------------------ closing

    async def close_episode(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """Summarise and index an episode. Safe to call more than once."""
        session = self.storage.get_session()
        try:
            ep = session.query(Episode).filter(Episode.id == episode_id).first()
            if ep is None or ep.status == "closed":
                return None
            turns = session.query(Turn).filter(Turn.episode_id == episode_id).order_by(Turn.seq).all()
            turn_dicts = [
                {"role": t.role, "content": t.content, "occurred_at": t.occurred_at, "id": t.id}
                for t in turns
            ]
        finally:
            session.close()
        if not turn_dicts:
            self._finalize(episode_id, None, [], [], None)
            return None

        summary = await self._summarize_llm(turn_dicts)
        if summary is None:
            summary = self.summarize_extractive(turn_dicts)
        title, text, key_points, entities = (
            summary.get("title"),
            summary.get("summary"),
            summary.get("key_points") or [],
            summary.get("entities") or [],
        )
        if not entities and self.knowledge_graph is not None:
            names: List[str] = []
            for kp in key_points:
                ents, _ = self.knowledge_graph.extract(kp)
                names.extend(e["name"] for e in ents)
            entities = list(dict.fromkeys(names))[:12]
        embedding = None
        if self.embedder is not None:
            try:
                embedding = self.embedder.encode_list(f"{title or ''}\n{text or ''}\n" + "\n".join(key_points))
            except Exception:
                embedding = None
        ep_dict = self._finalize(episode_id, title, key_points, entities, embedding, text)
        if ep_dict and self.knowledge_graph is not None and entities:
            self.knowledge_graph.ingest_episode(
                episode_id,
                entities,
                Scope(namespace=self.storage.namespace, user_id=ep_dict["user_id"], session_id=ep_dict["session_id"]),
            )
        return ep_dict

    def _finalize(
        self,
        episode_id: int,
        title: Optional[str],
        key_points: List[str],
        entities: List[str],
        embedding: Optional[List[float]],
        summary: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        session = self.storage.get_session()
        try:
            ep = session.query(Episode).filter(Episode.id == episode_id).first()
            if ep is None or ep.status == "closed":
                return None
            ep.title = (title or "")[:255] or None
            ep.summary = summary
            ep.key_points = key_points
            ep.entity_names = entities
            ep.embedding = embedding
            ep.status = "closed"
            ep.closed_at = ep.closed_at or utcnow()
            ep.centroid = None  # only needed while open
            ep.importance_score = min(1.0, 0.3 + 0.05 * (ep.turn_count or 0) + 0.05 * len(entities))
            self.storage.fts_insert(session, "episode", ep.id, episode_search_text(ep))
            session.commit()
            self.storage.index_add("episode", ep.id, ep.user_id, episode_search_text(ep), embedding)
            return self.storage.episode_to_dict(ep)
        except Exception as e:
            session.rollback()
            logger.warning("Failed to finalize episode %s: %s", episode_id, e)
            return None
        finally:
            session.close()

    async def close_pending(self, scope: Optional[Scope] = None) -> List[int]:
        """Summarise episodes that were cut by a boundary but not yet summarised."""
        session = self.storage.get_session()
        try:
            q = session.query(Episode.id).filter(Episode.status == "closing")
            if self.storage.namespace:
                q = q.filter(Episode.namespace == self.storage.namespace)
            if scope is not None and scope.session_id:
                q = q.filter(Episode.session_id == scope.session_id)
            ids = [r[0] for r in q.all()]
        finally:
            session.close()
        for eid in ids:
            await self.close_episode(eid)
        return ids

    async def end_session(self, scope: Scope) -> List[int]:
        """Close every open episode of a session (call at conversation end)."""
        session = self.storage.get_session()
        try:
            eps = (
                self._session_query(session, Episode, scope)
                .filter(Episode.status.in_(("open", "closing")))
                .all()
            )
            ids = [e.id for e in eps]
            rec = self._session_query(session, SessionRecord, scope).first()
            if rec is not None:
                rec.ended_at = utcnow()
            session.commit()
        finally:
            session.close()
        for eid in ids:
            await self.close_episode(eid)
        return ids

    async def close_idle(self, idle_minutes: Optional[float] = None) -> List[int]:
        """Close open episodes with no activity for ``idle_minutes`` (sleep-time job)."""
        cutoff = utcnow() - timedelta(minutes=idle_minutes or self.cfg.gap_minutes)
        session = self.storage.get_session()
        try:
            q = session.query(Episode.id).filter(
                Episode.status.in_(("open", "closing")), Episode.end_time < cutoff
            )
            if self.storage.namespace:
                q = q.filter(Episode.namespace == self.storage.namespace)
            ids = [r[0] for r in q.all()]
        finally:
            session.close()
        for eid in ids:
            await self.close_episode(eid)
        return ids

    # ------------------------------------------------------------------ reads

    def recent_turns(self, scope: Scope, limit: int = 8, before_seq: Optional[int] = None) -> List[Dict[str, Any]]:
        session = self.storage.get_session()
        try:
            q = self._session_query(session, Turn, scope)
            if before_seq is not None:
                q = q.filter(Turn.seq < before_seq)
            rows = q.order_by(Turn.seq.desc()).limit(limit).all()
            return [self.storage.turn_to_dict(t) for t in reversed(rows)]
        finally:
            session.close()

    def episode_turns(self, episode_id: int) -> List[Dict[str, Any]]:
        session = self.storage.get_session()
        try:
            rows = session.query(Turn).filter(Turn.episode_id == episode_id).order_by(Turn.seq).all()
            return [self.storage.turn_to_dict(t) for t in rows]
        finally:
            session.close()

    def list_episodes(self, scope: Scope, limit: int = 20, include_open: bool = True) -> List[Dict[str, Any]]:
        session = self.storage.get_session()
        try:
            q = session.query(Episode)
            if self.storage.namespace:
                q = q.filter(Episode.namespace == self.storage.namespace)
            q = q.filter(Episode.user_id == scope.user_id if scope.user_id else Episode.user_id.is_(None))
            if scope.session_id:
                q = q.filter(Episode.session_id == scope.session_id)
            if not include_open:
                q = q.filter(Episode.status == "closed")
            rows = q.order_by(Episode.start_time.desc()).limit(limit).all()
            return [self.storage.episode_to_dict(e) for e in rows]
        finally:
            session.close()

    # ------------------------------------------------------------------ summarisation

    async def _summarize_llm(self, turns: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        from memorable_ai.core.llm_client import llm_enabled

        if not llm_enabled(self.llm_model):
            return None
        budget, lines = 12000, []
        for t in turns:
            line = f"{t['role']}: {t['content'][:800]}"
            if budget - len(line) < 0:
                lines.append("...")
                break
            budget -= len(line)
            lines.append(line)
        when = turns[0].get("occurred_at")
        prompt = EPISODE_PROMPT.format(
            date=when.strftime("%Y-%m-%d") if isinstance(when, datetime) else "unknown",
            turns="\n".join(lines),
        )
        try:
            from memorable_ai.core import llm_client

            resp = await llm_client.acompletion_with_fallback(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0,
            )
            data = llm_client.parse_json(llm_client.message_content(resp))
            if not isinstance(data, dict) or not data.get("summary"):
                return None
            return {
                "title": str(data.get("title") or "")[:255],
                "summary": str(data.get("summary")),
                "key_points": [str(k) for k in (data.get("key_points") or [])][:8],
                "entities": [str(e) for e in (data.get("entities") or [])][:20],
            }
        except Exception as e:
            logger.debug("Episode LLM summary failed, using extractive: %s", e)
            return None

    def summarize_extractive(self, turns: List[Dict[str, Any]], max_points: int = 5) -> Dict[str, Any]:
        """Key-segment extraction: centrality + information density + MMR diversity."""
        units: List[Tuple[str, str]] = []
        for t in turns:
            for sent in _SENT_RE.split(t["content"]):
                sent = sent.strip()
                if len(tokenize(sent)) >= 3 and len(sent) <= 600:
                    units.append((t["role"], sent))
        if not units:
            units = [(t["role"], t["content"][:300]) for t in turns if t["content"].strip()]
        if not units:
            return {"title": None, "summary": None, "key_points": [], "entities": []}

        k = max(1, min(max_points, max(2, len(units) // 3)))
        if self.embedder is not None and len(units) > 1:
            vecs = self.embedder.encode([u[1] for u in units])
            centroid = vecs.mean(axis=0)
            norm = float(np.linalg.norm(centroid)) or 1.0
            centrality = vecs @ (centroid / norm)
        else:
            vecs = None
            centrality = np.ones(len(units), dtype=np.float32)
        info = np.array([len(_INFO_RE.findall(u[1][1:])) / (1 + len(u[1]) / 80) for u in units], dtype=np.float32)
        if info.max() > 0:
            info = info / info.max()
        speaker = np.array([1.0 if u[0] == "user" else 0.0 for u in units], dtype=np.float32)
        score = 0.55 * centrality + 0.25 * info + 0.20 * speaker

        selected: List[int] = []
        lam = 0.7
        candidates = list(range(len(units)))
        while candidates and len(selected) < k:
            if vecs is not None and selected:
                sims = vecs[candidates] @ vecs[selected].T
                redundancy = sims.max(axis=1)
            else:
                redundancy = np.zeros(len(candidates))
            mmr = lam * score[candidates] - (1 - lam) * redundancy
            best = candidates[int(np.argmax(mmr))]
            selected.append(best)
            candidates.remove(best)
        selected.sort()
        key_points = [f"{'User' if units[i][0] == 'user' else 'Assistant'}: {units[i][1]}" for i in selected]

        title = None
        for t in turns:
            if t["role"] != "user":
                continue
            for sent in _SENT_RE.split(t["content"].strip()):
                if len(tokenize(sent)) >= 3:
                    title = sent if len(sent) <= 80 else sent[:77].rsplit(" ", 1)[0] + "..."
                    break
            if title:
                break
        title = title or units[selected[0]][1][:80]
        summary = " ".join(units[i][1] for i in selected)
        if len(summary) > 700:
            summary = summary[:697].rsplit(" ", 1)[0] + "..."
        return {"title": title, "summary": summary, "key_points": key_points, "entities": []}
