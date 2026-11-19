"""
The write pipeline: the single path every conversation takes into memory.

    messages + response
      -> L0 turns (only the unseen suffix)            episodic.ingest
      -> L1 segmentation (boundary detection)         episodic.ingest
      -> L2 extraction (new turns + context window)   extractor.extract_from_turns
      -> update phase (ADD/UPDATE/DELETE/NOOP)        updater.apply_facts
      -> knowledge graph (entities, relations,        knowledge_graph.ingest_memory
         single-valued-edge conflicts)
      -> closed episodes: summary + key points,       episodic.close_episode
         folded into the session summary              summary_store.fold_episode

The old engine had three different write paths. The default synchronous one
skipped the summary, the update phase and the graph entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from memorable_ai.core.scope import Scope

from memorable_ai.core.models import utcnow  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    new_turns: int = 0
    facts: List[Dict[str, Any]] = field(default_factory=list)
    closed_episodes: List[int] = field(default_factory=list)
    retired_facts: List[int] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "new_turns": self.new_turns,
            "facts": [
                {"id": f.get("id"), "op": f.get("op"), "content": f.get("content"), "type": f.get("type")}
                for f in self.facts
            ],
            "closed_episodes": self.closed_episodes,
            "retired_facts": self.retired_facts,
        }


class WritePipeline:
    def __init__(
        self,
        storage: Any,
        extractor: Any,
        updater: Any,
        episodic: Optional[Any] = None,
        knowledge_graph: Optional[Any] = None,
        summary_store: Optional[Any] = None,
        timing: str = "turn",
        context_window: int = 6,
        legacy_log: bool = True,
        tasks: Optional[Any] = None,
        facts: bool = True,
        observations: bool = False,
        observe_batch_tokens: int = 0,
    ):
        self.storage = storage
        self.extractor = extractor
        self.updater = updater
        self.episodic = episodic
        self.knowledge_graph = knowledge_graph
        self.summary_store = summary_store
        self.timing = timing if timing in ("turn", "episode") else "turn"
        self.context_window = context_window
        self.legacy_log = legacy_log
        self.tasks = tasks  # TaskTracker (procedural P1b)
        self.facts = facts  # fact extraction on (off: only the observer writes memories)
        self.observations = observations  # the observer at episode close (docs/observation_memory.md)
        # 0: one observer call per closed episode. N > 0: closed episodes queue until their messages
        # reach N tokens, then one call observes them all (Mastra batches ~30k tokens)
        self.observe_batch_tokens = max(0, int(observe_batch_tokens or 0))

    async def ingest(
        self,
        scope: Scope,
        messages: Sequence[Any],
        response_text: Optional[str] = None,
        occurred_at: Optional[datetime] = None,
        response_events: Optional[Sequence[Any]] = None,
        task_hint: Optional[Dict[str, Any]] = None,
    ) -> IngestReport:
        report = IngestReport(session_id=scope.session_id, user_id=scope.user_id)

        if self.episodic is None:
            # Episodic layer disabled: extract from the latest exchange only.
            facts = await self.extractor.extract(
                list(messages), response_text, session_id=scope.session_id, session_date=occurred_at
            )
            report.facts = await self._apply(facts, scope, episode_id=None, turn_ids=None)
            return report

        res = await self.episodic.ingest(scope, messages, response_text, occurred_at, response_events=response_events)
        report.new_turns = len(res.new_turns)
        if not res.new_turns:
            return report
        if self.tasks is not None:
            hint = task_hint or {}
            try:
                self.tasks.on_events(scope, res.new_turns, offered_tools=hint.get("offered_tools"),
                                     explicit_task_id=hint.get("task_id"))
            except Exception as e:  # task segmentation must never break recording
                logger.warning("Task segmentation failed: %s", e)
        # Facts come from conversation; tool calls/results feed tasks and procedures instead.
        msg_turns = [t for t in res.new_turns if t.get("kind", "message") == "message"]

        if self.timing == "turn" and msg_turns and self.facts:
            context = self.episodic.recent_turns(
                scope, limit=self.context_window, before_seq=msg_turns[0]["seq"]
            )
            summary = await self._summary(scope)
            facts = await self.extractor.extract_from_turns(
                msg_turns,
                context_turns=context,
                session_summary=summary,
                session_date=msg_turns[0]["occurred_at"],
                user_name=self._user_name(scope),
            )
            user_turn_ids = [t["id"] for t in msg_turns if t["role"] == "user"]
            report.facts.extend(
                await self._apply(facts, scope, episode_id=res.open_episode_id, turn_ids=user_turn_ids)
            )

        for eid in res.closed_episode_ids:
            await self._close_episode(eid, scope, report)

        for f in report.facts:
            report.retired_facts.extend(f.get("retired", []))
            if f.get("superseded_id"):
                report.retired_facts.append(f["superseded_id"])
        report.retired_facts = sorted(set(report.retired_facts))
        if self.legacy_log:
            try:
                await self.storage.store_conversation(
                    list(messages)[-2:] if messages else [],
                    {"text": response_text} if response_text else None,
                    report.facts,
                    session_id=scope.session_id,
                    user_id=scope.user_id,
                )
            except Exception as e:
                logger.debug("Legacy conversation log failed: %s", e)
        return report

    async def end_session(self, scope: Scope) -> IngestReport:
        """Close the session's open episodes (summaries, key points, episode-timed facts)."""
        report = IngestReport(session_id=scope.session_id, user_id=scope.user_id)
        if self.episodic is None or not scope.session_id:
            return report
        ids = [
            e["id"]
            for e in self.episodic.list_episodes(scope, limit=1000)
            if e["status"] in ("open", "closing")
        ]
        for eid in ids:
            await self._close_episode(eid, scope, report)
        await self.episodic.end_session(scope)
        return report

    async def close_idle(self, idle_minutes: Optional[float] = None) -> List[int]:
        """Sleep-time job: close episodes idle longer than the gap threshold."""
        if self.episodic is None:
            return []
        closed: List[int] = []
        from datetime import timedelta

        from memorable_ai.core.models import Episode

        cutoff = utcnow() - timedelta(minutes=idle_minutes or self.episodic.cfg.gap_minutes)
        session = self.storage.get_session()
        try:
            q = session.query(Episode).filter(
                Episode.status.in_(("open", "closing")), Episode.end_time < cutoff
            )
            if self.storage.namespace:
                q = q.filter(Episode.namespace == self.storage.namespace)
            rows = [(e.id, e.user_id, e.session_id) for e in q.all()]
        finally:
            session.close()
        for eid, uid, sid in rows:
            report = IngestReport()
            await self._close_episode(
                eid, Scope(namespace=self.storage.namespace, user_id=uid, session_id=sid), report
            )
            closed.append(eid)
        return closed

    # ------------------------------------------------------------------ internals

    def _user_name(self, scope: Scope) -> Optional[str]:
        kg = self.knowledge_graph
        try:
            return kg.user_name(scope) if kg is not None and hasattr(kg, "user_name") else None
        except Exception:  # noqa: BLE001 - naming is a nicety; extraction must not fail on it
            return None

    async def _summary(self, scope: Scope) -> str:
        if self.summary_store is None or not scope.session_id:
            return ""
        try:
            return await self.summary_store.get_summary(scope.session_id)
        except Exception:
            return ""

    async def _close_episode(self, episode_id: int, scope: Scope, report: IngestReport) -> None:
        ep = await self.episodic.close_episode(episode_id)
        if ep is None:
            return
        report.closed_episodes.append(episode_id)
        if self.summary_store is not None and scope.session_id:
            try:
                await self.summary_store.fold_episode(scope.session_id, ep)
            except Exception as e:
                logger.debug("Session summary update failed: %s", e)
        if self.observations:
            if self.observe_batch_tokens:
                self._flag_episodes([episode_id], observe_pending=True)
                await self.observe_pending(scope, report)
            else:
                await self._observe(episode_id, scope, report)
        if self.timing == "episode" and self.facts:
            turns = self.episodic.episode_turns(episode_id)
            for t in turns:
                if isinstance(t.get("occurred_at"), str):
                    t["occurred_at"] = datetime.fromisoformat(t["occurred_at"])
            facts = await self.extractor.extract_from_turns(
                turns,
                session_summary=await self._summary(scope),
                session_date=turns[0]["occurred_at"] if turns else None,
                user_name=self._user_name(scope),
            )
            report.facts.extend(
                await self._apply(
                    facts, scope, episode_id=episode_id,
                    turn_ids=[t["id"] for t in turns if t["role"] == "user"],
                )
            )

    async def _observe(self, episode_id: int, scope: Scope, report: IngestReport) -> None:
        """Observer: dated observations of the closed episode, appended to the log (no update phase:
        a log keeps what was true when, and later observations say what changed)."""
        observe = getattr(self.extractor, "observe", None)
        if observe is None:
            return
        # stored turn dicts carry the layer in "kind" and the message kind in "event_kind"
        turns = [t for t in self.episodic.episode_turns(episode_id) if t.get("event_kind", "message") == "message"]
        for t in turns:
            if isinstance(t.get("occurred_at"), str):
                t["occurred_at"] = datetime.fromisoformat(t["occurred_at"])
        try:
            obs = await observe(turns, session_summary=await self._summary(scope),
                                session_date=turns[0]["occurred_at"] if turns else None,
                                user_name=self._user_name(scope))
        except Exception as e:  # noqa: BLE001 - the log is additive; recording must not fail on it
            logger.warning("Observer failed for episode %s: %s", episode_id, e)
            return
        for o in obs:
            o.update(user_id=scope.user_id, session_id=scope.session_id, source="conversation",
                     source_episode_id=episode_id)
            await self._store_observation(o, scope, report)
        self._flag_episodes([episode_id], observed=True)

    async def _store_observation(self, o: Dict[str, Any], scope: Scope, report: IngestReport) -> None:
        if scope.agent_id:
            o.setdefault("agent_id", scope.agent_id)
        stored = self.updater._add(o)
        if stored.get("op") != "add":
            return
        report.facts.append(stored)
        if self.knowledge_graph is not None:
            ingest = getattr(self.knowledge_graph, "aingest_memory", None)
            await ingest(stored, scope) if ingest else self.knowledge_graph.ingest_memory(stored, scope)

    def _flag_episodes(self, ids: Sequence[int], **flags: Any) -> None:
        """Set (or, with None, clear) keys in the episodes' metadata."""
        from memorable_ai.core.models import Episode

        session = self.storage.get_session()
        try:
            for ep in session.query(Episode).filter(Episode.id.in_(list(ids))).all():
                meta = dict(ep.extra_metadata or {})
                for k, v in flags.items():
                    if v is None:
                        meta.pop(k, None)
                    else:
                        meta[k] = v
                ep.extra_metadata = meta
            session.commit()
        finally:
            session.close()

    def _pending_episodes(self, scope: Scope) -> List[int]:
        """Closed episodes of this user queued for the batched observer, oldest first."""
        from memorable_ai.core.models import Episode

        session = self.storage.get_session()
        try:
            q = session.query(Episode.id, Episode.extra_metadata).filter(Episode.status == "closed")
            q = q.filter(Episode.user_id == scope.user_id if scope.user_id is not None else Episode.user_id.is_(None))
            if self.storage.namespace:
                q = q.filter(Episode.namespace == self.storage.namespace)
            rows = q.order_by(Episode.start_time, Episode.id).all()
        finally:
            session.close()
        return [i for i, meta in rows if (meta or {}).get("observe_pending")]

    async def observe_pending(self, scope: Scope, report: Optional[IngestReport] = None,
                              force: bool = False) -> int:
        """
        Batched observer: observe the queued episodes in calls of at least ``observe_batch_tokens``
        message tokens (``force``: the remainder too). Returns the number of observer calls.
        """
        from memorable_ai.core.context import estimate_tokens, render_observation_log

        observe = getattr(self.extractor, "observe_batch", None)
        if observe is None or self.episodic is None:
            return 0
        report = report or IngestReport(session_id=scope.session_id, user_id=scope.user_id)
        batches: List[List[Any]] = []
        cur: List[Any] = []
        size = 0
        for eid in self._pending_episodes(scope):
            turns = [t for t in self.episodic.episode_turns(eid) if t.get("event_kind", "message") == "message"]
            cur.append((eid, turns))
            size += sum(estimate_tokens(t.get("content") or "") for t in turns)
            if size >= max(1, self.observe_batch_tokens):
                batches.append(cur)
                cur, size = [], 0
        if cur and force:
            batches.append(cur)
        for batch in batches:
            turns, origin = [], {}
            for eid, ts in batch:
                for t in ts:
                    if isinstance(t.get("occurred_at"), str):
                        t["occurred_at"] = datetime.fromisoformat(t["occurred_at"])
                    origin[t.get("id")] = (eid, t.get("session_id"))
                    turns.append(t)
            prior = await self.storage.get_memories(memory_type="observation", limit=100_000,
                                                    scope_user=scope.user_id)
            prior = sorted(prior, key=lambda m: m.get("id") or 0)[-120:]
            try:
                obs = await observe(turns, prior_log=render_observation_log(prior, max_tokens=1500),
                                    user_name=self._user_name(scope))
            except Exception as e:  # noqa: BLE001 - the log is additive; the batch stays queued
                logger.warning("Observer failed for %d queued episodes: %s", len(batch), e)
                return batches.index(batch)
            for o in obs:
                eid, sid = origin.get((o.get("source_turn_ids") or [None])[0], (batch[0][0], scope.session_id))
                o.update(user_id=scope.user_id, session_id=sid or scope.session_id, source="conversation",
                         source_episode_id=eid)
                await self._store_observation(o, scope, report)
            self._flag_episodes([eid for eid, _ in batch], observe_pending=None, observed=True)
        return len(batches)

    async def _apply(
        self,
        facts: List[Dict[str, Any]],
        scope: Scope,
        episode_id: Optional[int],
        turn_ids: Optional[List[int]],
    ) -> List[Dict[str, Any]]:
        if not facts:
            return []
        for f in facts:
            f["user_id"] = scope.user_id
            f["session_id"] = scope.session_id
            if scope.agent_id:
                f.setdefault("agent_id", scope.agent_id)
            f.setdefault("source", "conversation")
            if episode_id is not None:
                f.setdefault("source_episode_id", episode_id)
            if turn_ids and not f.get("source_turn_ids"):
                f["source_turn_ids"] = turn_ids
        applied = await self.updater.apply_facts(facts, scope)
        if self.knowledge_graph is None:
            return applied
        for fact in applied:
            if fact.get("op") not in ("add", "update", "delete"):
                continue
            ingest = getattr(self.knowledge_graph, "aingest_memory", None)
            retired = (await ingest(fact, scope)) if ingest else self.knowledge_graph.ingest_memory(fact, scope)
            for old_id in retired:
                if old_id == fact["id"]:
                    continue
                # A single-valued edge changed (e.g. lives_in): retire the stale fact too.
                if self.storage._invalidate(old_id, superseded_by_id=fact["id"], reason="graph_conflict"):
                    fact.setdefault("retired", []).append(old_id)
        return applied
