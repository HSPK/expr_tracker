"""Spans: how long each part of a step took, and how the parts nest.

A closed span adds its duration to the open row, so it becomes an ordinary metric
that ``history()``, alert rules and plots already understand. The full tree, with
timestamps and attributes, goes to ``spans.jsonl`` for trace viewers.

Durations never commit a step of their own; they ride along with whatever
``log()`` commits.
"""

from __future__ import annotations

import functools
import threading
import time
from contextvars import ContextVar
from typing import Any

from loguru import logger

# Per-thread and per-task by construction: a new thread starts from the default,
# and each asyncio task gets its own copy, which is exactly span nesting.
_STACK: ContextVar[tuple[Span, ...]] = ContextVar("et_span_stack", default=())

DURATION_SUFFIX = "duration_ms"
COUNT_SUFFIX = "count"


def current_span() -> Span | None:
    stack = _STACK.get()
    return stack[-1] if stack else None


def active_path() -> str:
    """The ``/`` joined names of the spans currently open on this thread."""
    return "/".join(span.name for span in _STACK.get())


class Span:
    """One timed region. Prefer :func:`span`; construct it directly only when
    you need to end it somewhere else."""

    __slots__ = (
        "_ended",
        "_store",
        "_token",
        "attributes",
        "duration_ms",
        "error",
        "name",
        "path",
        "start",
        "started_at",
        "track",
    )

    def __init__(self, name: str, store, **attributes: Any):
        self.name = str(name).strip("/") or "span"
        self.attributes = dict(attributes)
        self._store = store
        self.path = self.name
        self.track = 0
        self.start = 0.0
        self.started_at = 0.0
        self.duration_ms = 0.0
        self.error: str | None = None
        self._token = None
        self._ended = False

    # ------------------------------------------------------------------ lifecycle

    def begin(self) -> Span:
        parent = current_span()
        self.path = f"{parent.path}/{self.name}" if parent else self.name
        # Children share their parent's track, so a tree never straddles lanes
        self.track = parent.track if parent else threading.get_ident()
        self.started_at = time.time()
        self.start = time.perf_counter()
        self._token = _STACK.set((*_STACK.get(), self))
        return self

    def end(self, error: BaseException | None = None) -> float:
        """Close the span and record it. Returns the duration in milliseconds."""
        if self._ended:
            return self.duration_ms
        self._ended = True
        self.duration_ms = (time.perf_counter() - self.start) * 1000.0
        if error is not None:
            self.error = type(error).__name__
        if self._token is not None:
            try:
                _STACK.reset(self._token)
            except ValueError:  # pragma: no cover - ended on another thread/task
                _STACK.set(tuple(s for s in _STACK.get() if s is not self))
        self._record()
        return self.duration_ms

    def set(self, **attributes: Any) -> Span:
        """Attach attributes; they reach ``spans.jsonl``, not the metrics."""
        self.attributes.update(attributes)
        return self

    # ------------------------------------------------------------------ context

    def __enter__(self) -> Span:
        return self

    def __exit__(self, kind, value, traceback) -> bool:
        self.end(value)
        return False

    async def __aenter__(self) -> Span:
        return self

    async def __aexit__(self, kind, value, traceback) -> bool:
        self.end(value)
        return False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = f"{self.duration_ms:.3f}ms" if self._ended else "open"
        return f"Span({self.path}, {state})"

    # ------------------------------------------------------------------ output

    def _record(self):
        if self._store is None:
            return
        metrics = {
            f"{self.path}/{DURATION_SUFFIX}": self.duration_ms,
            f"{self.path}/{COUNT_SUFFIX}": 1,
        }
        record = {
            "name": self.path,
            "depth": self.path.count("/"),
            "track": self.track,
            "start": self.started_at,
            "dur_ms": self.duration_ms,
        }
        if self.attributes:
            record["args"] = self.attributes
        if self.error:
            record["error"] = self.error
        try:
            self._store.record_span(metrics, record)
        except Exception as e:  # a measurement must never break the thing measured
            logger.warning(f"Failed to record span {self.path!r}: {e}")


class _NullSpan(Span):
    """What you get without a run: still times, records nothing."""

    def __init__(self, name: str, **attributes: Any):
        super().__init__(name, None, **attributes)


def start_span(name: str, **attributes: Any) -> Span:
    """Begin a span that you will ``end()`` yourself, possibly in another scope."""
    from .run import current_run

    run = current_run()
    store = run.history if run is not None else None
    span = Span(name, store, **attributes) if store is not None else _NullSpan(name)
    return span.begin()


class span:
    """Time a region, as a context manager, an async context manager or a decorator.

    ```python
    with et.span("forward"):
        with et.span("attention"):
            ...

    @et.span("preprocess")
    def preprocess(batch): ...

    async with et.span("fetch"):
        await loader.next()
    ```

    Nested names join with ``/``, so the duration lands on the metric
    ``forward/attention/duration_ms`` and window functions work on it directly.
    """

    def __init__(self, name: str, **attributes: Any):
        self._name = name
        self._attributes = attributes
        self._span: Span | None = None

    def __enter__(self) -> Span:
        self._span = start_span(self._name, **self._attributes)
        return self._span

    def __exit__(self, kind, value, traceback) -> bool:
        if self._span is not None:
            self._span.end(value)
        return False

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, kind, value, traceback) -> bool:
        return self.__exit__(kind, value, traceback)

    def __call__(self, fn):
        import asyncio

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                current = start_span(self._name, **self._attributes)
                error = None
                try:
                    return await fn(*args, **kwargs)
                except BaseException as e:
                    error = e
                    raise
                finally:
                    current.end(error)

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            current = start_span(self._name, **self._attributes)
            error = None
            try:
                return fn(*args, **kwargs)
            except BaseException as e:
                error = e
                raise
            finally:
                current.end(error)

        return wrapper
