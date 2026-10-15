"""History options and transaction/notification boundary types."""

from __future__ import annotations

import functools
import json
import operator
from collections.abc import Callable, Sequence
from contextvars import Context
from dataclasses import dataclass, fields
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Literal,
    ParamSpec,
    TypeAlias,
    TypeVar,
)

from loguru import logger

from .._native import (
    NativeRecord,
)
from .._process import check_process
from .naming import (
    validate_stream,
)
from .rotation import DEFAULT_MAX_FILE_BYTES, validate_file_limit
from .series import DEFAULT_WINDOW
from .writer import (
    DEFAULT_BUFFER_INTERVAL,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_MAX_BUFFER_SECONDS,
    DEFAULT_MAX_PENDING_BYTES,
    DEFAULT_MAX_PENDING_RECORDS,
    DEFAULT_MAX_RECORD_BYTES,
    validate_byte_limits,
)

if TYPE_CHECKING:
    from .store import HistoryStore

DEFAULT_CACHE_BYTES = 1 << 30  # 1 GiB
DEFAULT_CACHE_ROWS = 2_000_000
BIG_READ_WARN_ROWS = 2_000_000
_LOCAL_OVERRIDES = frozenset(
    {"ingest", "_ingest", "_emit", "_record_from_json", "_notify", "_store_row"}
)
_WRITER_OVERRIDES = frozenset(
    {"enqueue_with_status", "enqueue", "append_with_status", "append"}
)


STEP_POLICIES = ("monotonic", "allow")
Reduction = Literal["max", "mean"]
_P = ParamSpec("_P")
_T = TypeVar("_T")


def _serialized_write(
    method: Callable[Concatenate[HistoryStore, _P], _T],
) -> Callable[Concatenate[HistoryStore, _P], _T]:
    @functools.wraps(method)
    def wrapped(self: HistoryStore, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        check_process(self._owner_pid)
        pending: list[_Notification] = []
        target = None
        try:
            with self._write_lock:
                parent = getattr(self._callback_state, "pending", None)
                if parent is not None:
                    return method(self, *args, **kwargs)
                self._callback_state.pending = pending
                try:
                    return method(self, *args, **kwargs)
                finally:
                    del self._callback_state.pending
                    if pending:
                        target = self._notify_queue.enqueue(pending)
        finally:
            if target is not None:
                self._deliver_notifications(target)

    return wrapped


def resolve_commit(step: int | None, commit: bool | None) -> bool:
    """Without an explicit ``commit``: a step-less log commits, a stepped one waits."""
    return (step is None) if commit is None else commit


def _normalize_step(step) -> int | None:
    if step is None or type(step) is int:
        return step
    try:
        return operator.index(step)
    except TypeError as error:
        raise TypeError("step must be an integer or None") from error


@dataclass(frozen=True)
class HistoryOptions:
    """Tunables for a run's local history, validated once at ``init()``."""

    # disk buffering
    buffer_size: int = DEFAULT_BUFFER_SIZE
    buffer_interval: float | None = DEFAULT_BUFFER_INTERVAL
    max_buffer_seconds: float | None = DEFAULT_MAX_BUFFER_SECONDS
    max_pending_records: int = DEFAULT_MAX_PENDING_RECORDS
    max_pending_bytes: int | None = DEFAULT_MAX_PENDING_BYTES
    max_record_bytes: int | None = DEFAULT_MAX_RECORD_BYTES
    max_file_bytes: int | None = DEFAULT_MAX_FILE_BYTES
    # in-memory cache
    cache_bytes: int = DEFAULT_CACHE_BYTES
    cache_rows: int = DEFAULT_CACHE_ROWS
    # step assembly
    step_policy: str = "monotonic"
    rank_aware: bool = True
    # an independent producer writing its own file with its own step cursor
    stream: str | None = None
    # write the full span tree to spans.jsonl beside the metrics
    spans: bool = True
    span_count: bool = False
    span_metrics: bool = True
    span_async_write: bool = True
    # run-wide defaults for et.span(); a span's own arguments win
    span_print_fn: Callable[[str], None] | None = None
    span_plugins: Sequence[Any] = ()
    # alerts and output
    alert_window: int = DEFAULT_WINDOW
    print_to_screen: bool = False
    print_handle: Callable[[str], None] = print

    def __post_init__(self):
        validate_byte_limits(self.max_pending_bytes, self.max_record_bytes)
        validate_file_limit(self.max_file_bytes)
        if self.step_policy not in STEP_POLICIES:
            raise ValueError(
                f"Unknown step_policy {self.step_policy!r}; "
                f"expected one of {', '.join(STEP_POLICIES)}."
            )
        if self.stream is not None:
            object.__setattr__(self, "stream", validate_stream(self.stream))
        clamped = {
            "cache_bytes": max(0, int(self.cache_bytes)),
            "cache_rows": max(1, int(self.cache_rows)),
        }
        for key, value in clamped.items():
            object.__setattr__(self, key, value)

    @classmethod
    def from_kwargs(cls, **kwargs) -> HistoryOptions:
        """Build options, rejecting unknown names instead of ignoring them."""
        unknown = set(kwargs) - {f.name for f in fields(cls)}
        if unknown:
            raise TypeError(
                f"Unknown history options: {sorted(unknown)}. "
                f"Valid options: {', '.join(sorted(f.name for f in fields(cls)))}."
            )
        return cls(**kwargs)


@dataclass(frozen=True)
class _CacheView:
    """A consistent look at the cache, taken under one lock.

    ``complete`` means the snapshot already answers the query, so the disk does not
    have to be touched at all.
    """

    rows: list[tuple[int, int, bytes]]
    complete: bool
    start_row: int
    disk_end: int | None

    def records(self) -> list[dict]:
        return _parse_cached(self.rows)


@dataclass
class _Notification:
    record: dict
    options: HistoryOptions
    on_commit: Callable[[dict], None] | None
    context: Context


_PreparedCommit: TypeAlias = tuple[
    str | None,
    list[NativeRecord],
    bool,
    bool,
    list[tuple[str, str]],
    list[tuple[str, str]],
]


def _parse_cached(entries: Sequence[tuple[int, int, bytes]]) -> list[dict]:
    records = []
    for _, _, line in entries:
        try:
            records.append(json.loads(line))
        except Exception as e:  # pragma: no cover - cache is produced here
            logger.warning(f"Skipping corrupted cached record: {e}")
    return records
