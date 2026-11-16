"""
Runtime scoping and concurrency primitives.

- ``Scope``: (namespace, user_id, session_id, agent_id) used by every read and write
  so one user's memories can never be injected into another user's prompt.
  ``user_id=None`` is the namespace's shared ``org`` tier; ``agent_id`` is an
  attribute (which agent produced/asks), never an isolation boundary.
- Context variables let callers set the active user/session once
  (``with engine.scope(user_id=...)``) instead of threading ids through every call.
- ``suppressed()`` marks code paths where LLM calls must not be intercepted
  (Memorable's own extraction calls, and provider SDK calls nested inside an
  already-intercepted call, e.g. litellm -> openai).
- ``run_sync`` runs a coroutine to completion from synchronous code whether or
  not an event loop is already running in the current thread.
- ``BackgroundWorker`` owns one thread with a persistent event loop so writes can
  happen off the hot path and in order.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextvars
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Iterator, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_current_user: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "memorable_user_id", default=None
)
_current_session: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "memorable_session_id", default=None
)
_current_agent: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "memorable_agent_id", default=None
)
_current_task: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "memorable_current_task", default=None
)
_suppress: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "memorable_suppress_interception", default=False
)


@dataclass(frozen=True)
class Scope:
    """
    Who a memory belongs to. ``user_id=None`` means the shared ``org`` tier.

    ``agent_id`` is last so positional ``Scope(ns, user, session)`` calls keep working.
    """

    namespace: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None

    def with_session(self, session_id: Optional[str]) -> "Scope":
        return replace(self, session_id=session_id)

    @property
    def tier(self) -> str:
        return "org" if self.user_id is None else "user"


def current_user() -> Optional[str]:
    return _current_user.get()


def current_session() -> Optional[str]:
    return _current_session.get()


def current_agent() -> Optional[str]:
    return _current_agent.get()


@contextmanager
def active_scope(
    user_id: Optional[str] = None, session_id: Optional[str] = None, agent_id: Optional[str] = None
) -> Iterator[None]:
    """Set the active user/session/agent for intercepted calls in this context."""
    tok_u = _current_user.set(user_id) if user_id is not None else None
    tok_s = _current_session.set(session_id) if session_id is not None else None
    tok_a = _current_agent.set(agent_id) if agent_id is not None else None
    try:
        yield
    finally:
        if tok_a is not None:
            _current_agent.reset(tok_a)
        if tok_s is not None:
            _current_session.reset(tok_s)
        if tok_u is not None:
            _current_user.reset(tok_u)


def is_suppressed() -> bool:
    return _suppress.get()


@contextmanager
def suppressed() -> Iterator[None]:
    """Disable interception for LLM calls made inside this block."""
    tok = _suppress.set(True)
    try:
        yield
    finally:
        _suppress.reset(tok)


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def run_sync(coro_factory: Callable[[], Awaitable[T]], timeout: Optional[float] = None) -> T:
    """
    Run ``coro_factory()`` to completion from sync code.

    If no loop is running in this thread, uses ``asyncio.run``. If one is running
    (e.g. a sync SDK call made from inside async code), runs the coroutine in a
    worker thread with a fresh loop; context variables are copied so scope and
    suppression carry over.
    """
    if _running_loop() is None:
        return asyncio.run(coro_factory())  # type: ignore[arg-type]

    ctx = contextvars.copy_context()

    def _target() -> T:
        return asyncio.run(coro_factory())  # type: ignore[arg-type]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(ctx.run, _target)
        return future.result(timeout=timeout)


class BackgroundWorker:
    """A daemon thread running its own event loop; jobs execute in submission order."""

    def __init__(self, name: str = "memorable-writer"):
        self._name = name
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._pending: List[concurrent.futures.Future] = []
        self._lock = threading.Lock()
        self._started = threading.Event()
        self._serial: Optional[asyncio.Lock] = None
        atexit.register(self._atexit)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._started.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        self._started.wait(timeout=5)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._serial = asyncio.Lock()
        self._started.set()
        try:
            loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            finally:
                loop.close()

    def submit(self, coro_factory: Callable[[], Awaitable[Any]]) -> concurrent.futures.Future:
        """Schedule a job. Jobs run one at a time, in order."""
        self.start()
        assert self._loop is not None and self._serial is not None
        serial = self._serial

        async def _job() -> Any:
            async with serial:
                return await coro_factory()

        ctx = contextvars.copy_context()

        def _schedule() -> concurrent.futures.Future:
            return asyncio.run_coroutine_threadsafe(_job(), self._loop)  # type: ignore[arg-type]

        future = ctx.run(_schedule)
        with self._lock:
            self._pending = [f for f in self._pending if not f.done()]
            self._pending.append(future)

        def _log_error(f: concurrent.futures.Future) -> None:
            exc = f.exception()
            if exc is not None:
                logger.error("Background memory job failed: %s", exc)

        future.add_done_callback(_log_error)
        return future

    def spawn(self, coro: Awaitable[Any]) -> Optional[concurrent.futures.Future]:
        """Run a long-lived coroutine (e.g. a periodic loop) on the worker loop."""
        self.start()
        if self._loop is None:
            return None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]

    def flush(self, timeout: Optional[float] = None) -> None:
        """Block until every submitted job has finished."""
        with self._lock:
            pending = list(self._pending)
        if pending:
            concurrent.futures.wait(pending, timeout=timeout)

    def stop(self, timeout: float = 10.0) -> None:
        if not self.running:
            return
        self.flush(timeout=timeout)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None
        self._loop = None

    def _atexit(self) -> None:
        try:
            self.stop(timeout=30.0)
        except Exception:  # pragma: no cover - interpreter shutdown
            pass


def current_task() -> Optional[int]:
    """The explicit task (``engine.task(...)``) the current code runs in, if any."""
    return _current_task.get()

