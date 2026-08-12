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
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from typing import Any

from loguru import logger

# Per-thread and per-task by construction: a new thread starts from the default,
# and each asyncio task gets its own copy, which is exactly span nesting.
_STACK: ContextVar[tuple[Span, ...]] = ContextVar("et_span_stack", default=())

DURATION_SUFFIX = "duration_ms"
COUNT_SUFFIX = "count"


_WARNED: set[tuple[str, str, str]] = set()


def _warn_once(*key: str) -> None:
    """A plugin that fails once fails every span; say so only the first time."""
    if key in _WARNED:
        return
    if len(_WARNED) > 256:  # a pathological plugin must not leak memory
        _WARNED.clear()
    _WARNED.add(key)
    logger.warning(key[-1])


def _safely(plugin: Any, hook: str, span: Span):
    """Call a plugin hook; a broken plugin must not break the measured code."""
    method = getattr(plugin, hook, None)
    if method is None and hook == "end" and callable(plugin):
        method = plugin  # a plain callable is an end-only plugin
    if method is None:
        return None
    try:
        return method(span)
    except Exception as e:
        name = type(plugin).__name__
        _warn_once(name, hook, f"Span plugin {name} failed in {hook}(): {e}")
        return None


def _pretty(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


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
        "depth",
        "duration_ms",
        "error",
        "metrics",
        "name",
        "path",
        "plugins",
        "print_depth",
        "print_fn",
        "start",
        "started_at",
        "track",
    )

    def __init__(
        self,
        name: str,
        store,
        *,
        print_fn: Callable[[str], None] | None = None,
        plugins: Sequence[Any] = (),
        **attributes: Any,
    ):
        self.name = str(name).strip("/") or "span"
        self.attributes = dict(attributes)
        self._store = store
        self.print_fn = print_fn
        self.plugins = tuple(plugins)
        self.metrics: dict[str, Any] = {}
        self.depth = 0
        self.print_depth = 0
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
        self.depth = parent.depth + 1 if parent else 0
        # Children share their parent's track, so a tree never straddles lanes
        self.track = parent.track if parent else threading.get_ident()
        if self.print_fn is None and parent is not None:
            self.print_fn = parent.print_fn  # inherited, so a tree prints as one
        if parent is not None and parent.print_fn is self.print_fn:
            self.print_depth = parent.print_depth + 1
        # else this span roots its own output: indent from 0, not from a parent
        # that printed nothing
        if not self.plugins and parent is not None:
            self.plugins = parent.plugins
        for plugin in self.plugins:
            _safely(plugin, "start", self)
        self.started_at = time.time()
        self.start = time.perf_counter()
        self._token = _STACK.set((*_STACK.get(), self))
        self._announce("->", None)
        return self

    def end(self, error: BaseException | None = None) -> float:
        """Close the span and record it. Returns the duration in milliseconds."""
        if self._ended:
            return self.duration_ms
        self._ended = True
        self.duration_ms = (time.perf_counter() - self.start) * 1000.0
        if error is not None:
            self.error = type(error).__name__
        for plugin in self.plugins:
            measured = _safely(plugin, "end", self)
            if isinstance(measured, dict):
                self.metrics.update(measured)
        if self._token is not None:
            # Drop ourselves rather than resetting the token. A token restores
            # the whole stack as it was, so a span ended out of order would
            # resurrect spans that already closed underneath it.
            self._token = None
            _STACK.set(tuple(s for s in _STACK.get() if s is not self))
        self._record()
        self._announce("<-", self.duration_ms)
        return self.duration_ms

    def _announce(self, marker: str, duration_ms: float | None):
        if self.print_fn is None:
            return
        indent = "\t" * self.print_depth
        stamp = time.strftime("%H:%M:%S", time.localtime())
        if duration_ms is None:
            line = f"{indent}{marker} {self.name}  {stamp}"
        else:
            extra = "".join(
                f"  {key}={_pretty(value)}" for key, value in self.metrics.items()
            )
            failed = f"  !{self.error}" if self.error else ""
            line = (
                f"{indent}{marker} {self.name}  {stamp}  "
                f"{duration_ms:.3f}ms{extra}{failed}"
            )
        try:
            self.print_fn(line)
        except Exception as e:  # printing must never break the thing measured
            logger.warning(f"Span print handler failed: {e}")

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
            **{f"{self.path}/{key}": value for key, value in self.metrics.items()},
        }
        record = {
            "name": self.path,
            "depth": self.depth,
            "track": self.track,
            "start": self.started_at,
            "dur_ms": self.duration_ms,
        }
        if self.attributes:
            record["args"] = self.attributes
        if self.metrics:
            record["metrics"] = self.metrics
        if self.error:
            record["error"] = self.error
        try:
            self._store.record_span(metrics, record)
        except Exception as e:  # a measurement must never break the thing measured
            logger.warning(f"Failed to record span {self.path!r}: {e}")


class _NullSpan(Span):
    """What you get without a run: still times and prints, records nothing."""

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, None, **kwargs)


def start_span(
    name: str,
    *,
    print_fn: Callable[[str], None] | None = None,
    plugins: Sequence[Any] = (),
    **attributes: Any,
) -> Span:
    """Begin a span that you will ``end()`` yourself, possibly in another scope."""
    from .run import current_run

    run = current_run()
    store = run.history if run is not None else None
    if store is not None:
        # The span's own arguments win over the run-wide defaults
        print_fn = print_fn if print_fn is not None else store.options.span_print_fn
        plugins = plugins or store.options.span_plugins
    options = {"print_fn": print_fn, "plugins": plugins, **attributes}
    span = (
        Span(name, store, **options)
        if store is not None
        else _NullSpan(name, **options)
    )
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

    def __init__(
        self,
        name: str,
        *,
        print_fn: Callable[[str], None] | None = None,
        plugins: Sequence[Any] = (),
        **attributes: Any,
    ):
        self._name = name
        self._attributes = {"print_fn": print_fn, "plugins": plugins, **attributes}
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
