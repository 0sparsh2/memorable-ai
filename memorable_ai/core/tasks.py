"""
Tasks and outcomes (procedural P1b, docs/procedural_memory_design.md §6).

``TaskTracker.on_events`` segments a session's L0 event stream into tasks:
  * a task opens at the user message that is followed by a tool call; that message is
    the goal. A follow-up like "now do the same for auth" opens a child task
    (``parent_task_id``);
  * it closes at a tool-free assistant answer, at the next user message, or after an
    idle gap;
  * ``engine.task("...")`` declares a task explicitly: every event inside it attaches
    to it, and it closes when the block exits.

Outcomes, from most to least trusted (the higher confidence wins):
  explicit  t.success() / t.failure() / engine.mark_outcome()            1.0
  signal    tool errors, test summaries, exit codes, HTTP status          0.8
  feedback  the user's next message ("thanks, that worked" / "still broken")  0.6
A task with no signal stays ``unknown``.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy import or_

from memorable_ai.core.models import Task, Turn, utcnow
from memorable_ai.core.scope import Scope

logger = logging.getLogger(__name__)

CONFIDENCE = {"explicit": 1.0, "signal": 0.8, "feedback": 0.6, "judge": 0.4}

_ANAPHORA = re.compile(
    r"^\s*(?:ok(?:ay)?[,.]?\s+)?(?:now|then|also|next)\b|\b(?:the same|same thing|do that|that again|as well|"
    r"likewise|repeat (?:it|that)|again for)\b", re.I)
_NEGATIVE = re.compile(
    r"\b(?:didn'?t work|doesn'?t work|not working|still (?:fail\w*|broken|wrong|error\w*|crash\w*)|that'?s wrong|"
    r"(?:is|was) wrong|incorrect|it broke|broke it|nope|undo|revert|roll ?back|failed again)\b", re.I)
_POSITIVE = re.compile(
    r"\b(?:thanks|thank you|thx|great|perfect|awesome|works now|it works|that worked|worked|that did it|nice|"
    r"exactly|lgtm|looks good|solved|fixed|all good)\b", re.I)
_TESTS_FAILED = re.compile(r"\b([1-9]\d*) (?:failed|failures?|errors?)\b", re.I)
_TESTS_PASSED = re.compile(r"\b\d+ passed\b|\ball tests passed\b|\bOK \(\d+ tests?\)", re.I)
_HTTP = re.compile(r"\b(?:status(?:_code)?|HTTP/\d(?:\.\d)?)[\"':\s=]+([1-5]\d\d)\b", re.I)


def classify_feedback(text: str) -> Optional[str]:
    """"success" / "failure" from the user's next message, or None. Negative wins over positive."""
    if not text:
        return None
    if _NEGATIVE.search(text):
        return "failure"
    if _POSITIVE.search(text):
        return "success"
    return None


def signal_outcome(results: Sequence[Dict[str, Any]]) -> Optional[tuple]:
    """(outcome, reason) from a task's tool results, judged by the last one, or None."""
    if not results:
        return None
    last = results[-1]
    text = f"{last.get('content') or ''} {last.get('payload_text') or ''}"
    if last.get("is_error"):
        return "failure", f"last tool result ({last.get('tool_name') or 'tool'}) failed"
    m = _TESTS_FAILED.search(text)
    if m:
        return "failure", f"tests: {m.group(0)}"
    if _TESTS_PASSED.search(text):
        return "success", "tests passed"
    h = _HTTP.search(text)
    if h:
        code = int(h.group(1))
        return ("success", f"HTTP {code}") if code < 400 else ("failure", f"HTTP {code}")
    payload = last.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("exit_code"), int):
        return ("success", "exit code 0") if payload["exit_code"] == 0 else ("failure", f"exit code {payload['exit_code']}")
    return None


def task_to_dict(t: Task) -> Dict[str, Any]:
    return {
        "id": t.id, "session_id": t.session_id, "user_id": t.user_id, "agent_id": t.agent_id,
        "parent_task_id": t.parent_task_id, "episode_id": t.episode_id, "goal": t.goal_text,
        "explicit": bool(t.explicit), "status": t.status, "close_reason": t.close_reason,
        "outcome": t.outcome or "unknown", "outcome_confidence": t.outcome_confidence or 0.0,
        "outcome_source": t.outcome_source, "outcome_reason": t.outcome_reason,
        "offered_tools": t.offered_tools or [], "used_tools": t.used_tools or [],
        "llm_calls": t.llm_calls or 0, "tool_calls": t.tool_calls or 0, "error_count": t.error_count or 0,
        "started_at": t.started_at.isoformat() if t.started_at else None,
        "ended_at": t.ended_at.isoformat() if t.ended_at else None,
    }


class TaskTracker:
    def __init__(self, storage: Any, gap_minutes: float = 30.0):
        self.storage = storage
        self.gap = timedelta(minutes=gap_minutes)

    # ------------------------------------------------------------------ helpers

    def _scoped(self, session, scope: Scope):
        q = session.query(Task).filter(Task.session_id == scope.session_id)
        if self.storage.namespace:
            q = q.filter(Task.namespace == self.storage.namespace)
        return q.filter(Task.user_id == scope.user_id) if scope.user_id else q.filter(Task.user_id.is_(None))

    @staticmethod
    def _label(task: Task, outcome: str, source: str, reason: Optional[str]) -> bool:
        conf = CONFIDENCE[source]
        if (task.outcome_confidence or 0.0) > conf:
            return False
        task.outcome, task.outcome_confidence, task.outcome_source, task.outcome_reason = outcome, conf, source, reason
        return True

    def _attach(self, session, task: Task, turn_ids: Iterable[int], edges: List[Dict[str, Any]], scope: Scope) -> None:
        ids = [i for i in turn_ids if i]
        if not ids:
            return
        session.query(Turn).filter(Turn.id.in_(ids)).update({Turn.task_id: task.id}, synchronize_session=False)
        for tid in ids:
            edges.append(self.storage._edge_dict("turn", tid, "part_of", "task", task.id,
                                                 namespace=self.storage.namespace, user_id=scope.user_id,
                                                 agent_id=scope.agent_id))

    def _close(self, session, task: Task, reason: str, at: Optional[datetime]) -> None:
        if task.status == "closed":
            return
        task.status, task.close_reason = "closed", reason
        task.ended_at = at or task.last_event_at or utcnow()
        if (task.outcome_confidence or 0.0) < CONFIDENCE["signal"]:
            rows = (session.query(Turn).filter(Turn.task_id == task.id, Turn.kind == "tool_result")
                    .order_by(Turn.seq).all())
            results = [{"content": r.content, "is_error": r.is_error, "tool_name": r.tool_name, "payload": r.payload,
                        "payload_text": str(r.payload)[:4000] if r.payload is not None else ""} for r in rows]
            sig = signal_outcome(results)
            if sig:
                self._label(task, sig[0], "signal", sig[1])

    # ------------------------------------------------------------------ segmentation

    def on_events(
        self,
        scope: Scope,
        events: Sequence[Dict[str, Any]],
        offered_tools: Optional[Sequence[str]] = None,
        explicit_task_id: Optional[int] = None,
    ) -> List[int]:
        """Assign new L0 events (``EpisodicMemory.ingest`` output) to tasks; returns touched task ids."""
        if not events or not scope.session_id:
            return []
        touched: Dict[int, Task] = {}
        edges: List[Dict[str, Any]] = []
        session = self.storage.get_session()
        try:
            explicit = session.query(Task).filter(Task.id == explicit_task_id).first() if explicit_task_id else None
            open_task = explicit or (self._scoped(session, scope).filter(Task.status == "open", Task.explicit.isnot(True))
                                     .order_by(Task.id.desc()).first())
            pending_goal: Optional[Dict[str, Any]] = None

            def touch(t: Task, ev: Dict[str, Any]) -> None:
                touched[t.id] = t
                t.last_event_at = ev.get("occurred_at") or t.last_event_at

            for i, ev in enumerate(events):
                kind = ev.get("kind") or "message"
                at = ev.get("occurred_at")
                if (open_task is not None and not open_task.explicit and at and open_task.last_event_at
                        and at - open_task.last_event_at > self.gap):
                    self._close(session, open_task, "idle", open_task.last_event_at)
                    touched[open_task.id] = open_task
                    open_task = None
                if explicit is not None:  # everything in an explicit task belongs to it
                    self._attach(session, explicit, [ev["id"]], edges, scope)
                    if kind == "tool_call":
                        explicit.tool_calls = (explicit.tool_calls or 0) + 1
                        explicit.used_tools = sorted(set(explicit.used_tools or []) | {ev.get("tool_name") or "tool"})
                    elif kind == "tool_result" and ev.get("is_error"):
                        explicit.error_count = (explicit.error_count or 0) + 1
                    touch(explicit, ev)
                    continue
                if kind == "message" and ev["role"] == "user":
                    if open_task is not None:
                        fb = classify_feedback(ev["content"])
                        if fb:
                            self._label(open_task, fb, "feedback", "user's next message")
                        open_task.feedback_seen = True
                        self._close(session, open_task, "next_user_message", open_task.last_event_at)
                        touched[open_task.id] = open_task
                        open_task = None
                    else:
                        self._feedback_for_last_closed(session, scope, ev, touched)
                    pending_goal = ev
                    continue
                if kind == "tool_call":
                    if open_task is None:
                        open_task, goal_id = self._open(session, scope, ev, pending_goal)
                        if goal_id:
                            self._attach(session, open_task, [goal_id], edges, scope)
                    self._attach(session, open_task, [ev["id"]], edges, scope)
                    open_task.tool_calls = (open_task.tool_calls or 0) + 1
                    open_task.used_tools = sorted(set(open_task.used_tools or []) | {ev.get("tool_name") or "tool"})
                    touch(open_task, ev)
                elif kind == "tool_result" and open_task is not None:
                    self._attach(session, open_task, [ev["id"]], edges, scope)
                    if ev.get("is_error"):
                        open_task.error_count = (open_task.error_count or 0) + 1
                    touch(open_task, ev)
                elif kind == "message" and open_task is not None:  # assistant text
                    self._attach(session, open_task, [ev["id"]], edges, scope)
                    touch(open_task, ev)
                    if not any((e.get("kind") or "message") == "tool_call" for e in events[i + 1:]):
                        self._close(session, open_task, "final_answer", at)
                        open_task = None
            for t in touched.values():
                if offered_tools:
                    t.offered_tools = sorted(set(t.offered_tools or []) | set(offered_tools))
                t.llm_calls = (t.llm_calls or 0) + 1
            if edges:
                self.storage.add_edges(edges, session=session)
            session.commit()
            return list(touched)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _open(self, session, scope: Scope, ev: Dict[str, Any], pending_goal: Optional[Dict[str, Any]]) -> tuple:
        """Open an auto task at a tool call; returns (task, goal turn id or None)."""
        goal = pending_goal
        if goal is None:  # the goal message arrived in an earlier batch
            row = (session.query(Turn).filter(Turn.session_id == scope.session_id, Turn.role == "user",
                                              or_(Turn.kind.is_(None), Turn.kind == "message"),
                                              Turn.seq < ev["seq"]).order_by(Turn.seq.desc()).first())
            if row is not None:
                goal = {"id": row.id, "content": row.content, "occurred_at": row.occurred_at}
        parent = None
        if goal is not None and _ANAPHORA.search(goal["content"] or ""):
            parent = self._scoped(session, scope).order_by(Task.id.desc()).first()
        task = Task(
            namespace=self.storage.namespace, user_id=scope.user_id, agent_id=scope.agent_id,
            session_id=scope.session_id, episode_id=ev.get("episode_id"),
            parent_task_id=parent.id if parent is not None else None,
            goal_text=(goal or {}).get("content"), explicit=False, status="open",
            started_at=(goal or {}).get("occurred_at") or ev.get("occurred_at"),
            last_event_at=ev.get("occurred_at"), used_tools=[], offered_tools=[],
        )
        session.add(task)
        session.flush()
        if ev.get("episode_id"):
            self.storage.add_edges([self.storage._edge_dict(
                "task", task.id, "part_of", "episode", ev["episode_id"], namespace=self.storage.namespace,
                user_id=scope.user_id, agent_id=scope.agent_id)], session=session)
        return task, (goal or {}).get("id")

    def _feedback_for_last_closed(self, session, scope: Scope, ev: Dict[str, Any], touched: Dict[int, Task]) -> None:
        """The first user message after a task's final answer is feedback on it."""
        last = self._scoped(session, scope).order_by(Task.id.desc()).first()
        if last is None or last.status != "closed" or last.feedback_seen or last.close_reason != "final_answer":
            return
        at, ended = ev.get("occurred_at"), last.ended_at
        last.feedback_seen = True
        if at and ended and at - ended > self.gap:
            return
        fb = classify_feedback(ev["content"])
        if fb:
            self._label(last, fb, "feedback", "user's next message")
        touched[last.id] = last

    # ------------------------------------------------------------------ explicit API

    def open_explicit(self, scope: Scope, goal: str, parent_task_id: Optional[int] = None) -> int:
        session = self.storage.get_session()
        try:
            t = Task(namespace=self.storage.namespace, user_id=scope.user_id, agent_id=scope.agent_id,
                     session_id=scope.session_id, goal_text=goal, explicit=True, status="open",
                     parent_task_id=parent_task_id, started_at=utcnow(), last_event_at=utcnow(),
                     used_tools=[], offered_tools=[])
            session.add(t)
            session.commit()
            return t.id
        finally:
            session.close()

    def set_outcome(self, task_id: int, success: bool, reason: Optional[str] = None,
                    source: str = "explicit") -> Optional[Dict[str, Any]]:
        session = self.storage.get_session()
        try:
            t = session.query(Task).filter(Task.id == task_id).first()
            if t is None:
                return None
            self._label(t, "success" if success else "failure", source, reason)
            session.commit()
            return task_to_dict(t)
        finally:
            session.close()

    def close(self, task_id: int, reason: str = "explicit") -> Optional[Dict[str, Any]]:
        session = self.storage.get_session()
        try:
            t = session.query(Task).filter(Task.id == task_id).first()
            if t is None:
                return None
            self._close(session, t, reason, utcnow())
            session.commit()
            return task_to_dict(t)
        finally:
            session.close()

    def latest(self, scope: Scope) -> Optional[int]:
        session = self.storage.get_session()
        try:
            t = self._scoped(session, scope).order_by(Task.id.desc()).first()
            return t.id if t is not None else None
        finally:
            session.close()

    def list(self, scope: Scope, limit: int = 20, any_session: bool = False) -> List[Dict[str, Any]]:
        session = self.storage.get_session()
        try:
            q = session.query(Task)
            if self.storage.namespace:
                q = q.filter(Task.namespace == self.storage.namespace)
            q = q.filter(Task.user_id == scope.user_id) if scope.user_id else q.filter(Task.user_id.is_(None))
            if not any_session and scope.session_id:
                q = q.filter(Task.session_id == scope.session_id)
            return [task_to_dict(t) for t in q.order_by(Task.id.desc()).limit(limit).all()]
        finally:
            session.close()

    def get(self, task_id: int, user_id: Optional[str]) -> Dict[str, Any]:
        session = self.storage.get_session()
        try:
            t = session.query(Task).filter(Task.id == task_id).first()
            if t is None or t.user_id != user_id:
                return {}
            turns = session.query(Turn).filter(Turn.task_id == task_id).order_by(Turn.seq).all()
            return {"task": task_to_dict(t), "events": [self.storage.turn_to_dict(r) for r in turns]}
        finally:
            session.close()
