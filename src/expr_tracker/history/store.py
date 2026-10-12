"""Step-oriented metric history: open-row assembly, in-memory cache, queries.

Write path::

    log() --merge--> open row --commit--> cache + writer --notify--> series + callbacks

Reads are served from the cache; disk is touched only once rows have been
evicted (or after a resume), addressed by physical row ordinal.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import json
import operator
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextvars import Context, copy_context
from dataclasses import dataclass, fields
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Concatenate, Literal, ParamSpec, TypeVar

from loguru import logger

from .._process import check_process, process_id
from .codec import RESERVED_KEYS, RecordCodec, encode_line
from .frame import project, to_output
from .naming import (
    metrics_filename,
    resolve_log_dir,
    sidecar_filename,
    spans_filename,
    validate_stream,
)
from .reader import JsonlReader, merge_steps, recent_steps
from .rotation import DEFAULT_MAX_FILE_BYTES, validate_file_limit
from .series import DEFAULT_WINDOW, MetricSeries
from .writer import (
    DEFAULT_BUFFER_INTERVAL,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_MAX_BUFFER_SECONDS,
    DEFAULT_MAX_PENDING_BYTES,
    DEFAULT_MAX_PENDING_RECORDS,
    DEFAULT_MAX_RECORD_BYTES,
    JsonlWriter,
    in_diagnostic_callback,
    validate_byte_limits,
)

DEFAULT_CACHE_BYTES = 1 << 30  # 1 GiB
DEFAULT_CACHE_ROWS = 2_000_000
BIG_READ_WARN_ROWS = 2_000_000


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
                        with self._notification_condition:
                            self._notifications.extend(pending)
                            self._notification_condition.notify_all()
        finally:
            if pending:
                self._deliver_notifications(pending[-1])

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
    done: bool = False


class HistoryStore:
    """Local jsonl history: the ``jsonl`` backend and the source of ``et.history()``."""

    def __init__(self):
        self._owner_pid = process_id()
        self._lock = threading.RLock()
        self._write_lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._lifecycle_owner: int | None = None
        self._notification_condition = threading.Condition()
        self._notifications: deque[_Notification] = deque()
        self._notification_owner: int | None = None
        self._notification_depth = 0
        self._callback_state = threading.local()
        self._closing = threading.Event()
        self._codec = RecordCodec()
        self._atexit_registered = False
        self.writer: JsonlWriter | None = None
        self.log_dir: Path | None = None
        self.log_fp: Path | None = None
        self.config_fp: Path | None = None
        self.spans_fp: Path | None = None
        self.span_writer: JsonlWriter | None = None
        self.stream: str | None = None
        self.project: str = ""
        self.name: str = ""
        self._reset(HistoryOptions(), on_commit=None)

    def _reset(self, options: HistoryOptions, on_commit: Callable | None):
        """The one place run state is (re)initialised, so nothing can be forgotten."""
        with self._lock:
            self.options = options
            self.on_commit = on_commit
            self.started_at = time.time()
            self._series = MetricSeries(options.alert_window)
            self._cache: deque[tuple[int, int, bytes]] = deque()  # (row, step, line)
            self._cache_bytes = 0
            self._has_disk_prefix = False
            self._needs_merge = False
            self._queries = 0
            self._disk_queries = 0
            self._evicted_rows = 0
            self._closed = False
            self._closing.clear()
            self._codec.reset()
            self._open_step: int | None = None
            self._open_row: dict = {}
            self._open_weights: dict[str, float] = {}
            self._open_time = 0.0
            self._next_step = 0
            self._next_row = 0
            self._next_span_row = 0
            self._last_committed_step: int | None = None
            self._last_emitted_step: int | None = None
            self._last_commit_time: float | None = None

    # ------------------------------------------------------------------ setup

    def init(
        self,
        project: str,
        name: str | None = None,
        config: dict | None = None,
        dir: str | None = None,
        run_dir: str | None = None,
        on_commit: Callable[[dict], None] | None = None,
        **options,
    ) -> HistoryStore:
        """Open (or resume) a run. Tunables are named in :class:`HistoryOptions`."""
        self.require_outside_callback()
        opts = HistoryOptions.from_kwargs(**options)
        if name is None:
            name = time.strftime("run-%Y%m%d-%H%M%S")
            logger.warning(f"No run name provided, using generated name: {name}")
        log_dir = resolve_log_dir(project, name, dir, run_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        with self._lifecycle_operation():
            self._closing.set()
            self._close_previous()
            with self._write_lock:
                self._init_run(project, name, log_dir, config, opts, on_commit)
        return self

    def _init_run(
        self,
        project: str,
        name: str,
        log_dir: Path,
        config: dict | None,
        opts: HistoryOptions,
        on_commit: Callable[[dict], None] | None,
    ):
        self.project, self.name = project, name
        self.log_dir = log_dir
        self.stream = opts.stream
        self.config_fp = self.log_dir / sidecar_filename("config", opts.stream, "json")
        self.spans_fp = (
            self.log_dir / spans_filename(opts.stream, opts.rank_aware)
            if opts.spans
            else None
        )
        self.log_fp = self.log_dir / metrics_filename(opts.stream, opts.rank_aware)

        self._reset(opts, on_commit)
        self.writer = self._open_writer(opts)
        self.span_writer = (
            JsonlWriter(
                self.spans_fp,
                buffer_size=opts.buffer_size,
                buffer_interval=opts.buffer_interval,
                max_buffer_seconds=opts.max_buffer_seconds,
                max_pending_records=opts.max_pending_records,
                max_pending_bytes=opts.max_pending_bytes,
                max_record_bytes=opts.max_record_bytes,
                max_file_bytes=opts.max_file_bytes,
                async_write=opts.span_async_write,
            )
            if self.spans_fp is not None
            else None
        )
        self._next_span_row = self.span_writer.lines if self.span_writer else 0
        self._resume(opts.alert_window)
        self._write_config(config)
        self._register_atexit()
        # Say where the files went: dir is a root, so the run is two levels below
        # what the caller passed, and guessing that is nobody's idea of fun
        resumed = f", resuming at step {self._next_step}" if self._next_step else ""
        logger.info(f"Run {self.name!r} -> {self.log_dir}{resumed}")

    def _close_previous(self):
        """Commit using the old run's metadata and callback before replacing it."""
        self.flush(commit_open=True)
        with self._write_lock:
            for writer in (self.writer, self.span_writer):
                if writer is not None:
                    writer.close()

    def _open_writer(self, options: HistoryOptions) -> JsonlWriter:
        assert self.log_fp is not None
        return JsonlWriter(
            self.log_fp,
            buffer_size=options.buffer_size,
            buffer_interval=options.buffer_interval,
            max_buffer_seconds=options.max_buffer_seconds,
            max_pending_records=options.max_pending_records,
            max_pending_bytes=options.max_pending_bytes,
            max_record_bytes=options.max_record_bytes,
            max_file_bytes=options.max_file_bytes,
            meta_extra={
                "project": self.project,
                "name": self.name,
                "started_at": self.started_at,
            },
        )

    def _register_atexit(self):
        if not self._atexit_registered:
            atexit.register(self._atexit)
            self._atexit_registered = True

    def _resume(self, alert_window: int):
        """Pick up where an existing file left off: cursors, ordinals, windows."""
        writer = self.writer
        assert writer is not None
        self._needs_merge = writer.has_duplicate_steps or not writer.sorted
        # Existing rows live on disk only, so queries have to fall back to it
        self._has_disk_prefix = bool(writer.lines)
        self._next_row = writer.lines
        if writer.max_step is None:
            return
        # The highest step, not the last written one: a patched or out-of-order
        # file ends on an earlier step, and resuming there would reuse it
        self._last_committed_step = writer.max_step
        self._last_emitted_step = writer.max_step
        self._next_step = writer.max_step + 1
        try:
            self._series.backfill(writer.reader().tail(alert_window))
        except Exception as e:
            logger.warning(f"Could not backfill metric series from {self.log_fp}: {e}")

    def _write_config(self, config: dict | None):
        if config is None or self.config_fp is None:
            return
        if self.config_fp.exists():
            logger.warning(
                f"Config file {self.config_fp} already exists. It will be overwritten."
            )
        temporary = None
        try:
            encoded = self._codec.encode(config, kind="config")
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                errors="backslashreplace",
                dir=self.config_fp.parent,
                prefix=f".{self.config_fp.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                temporary = Path(f.name)
                json.dump(
                    encoded,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            temporary.replace(self.config_fp)
        except Exception as e:
            logger.error(f"Failed to write config to {self.config_fp}: {e}")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    logger.warning(
                        f"Could not remove temporary config {temporary}: {error}"
                    )

    def _require_init(self):
        check_process(self._owner_pid)
        if self.writer is None:
            raise RuntimeError(
                "HistoryStore.init() must be called before logging or querying."
            )

    # ------------------------------------------------------------------ writing

    def _encode_span_data(self, data: dict, *, kind: str) -> dict:
        self._require_init()
        return self._codec.encode(data, kind=kind)

    def log(
        self, metrics: dict, step: int | None = None, commit: bool | None = None
    ) -> int | None:
        """Merge metrics into the open row; ``commit`` decides when it is written.

        - ``log(d)``: ``commit`` defaults to ``True``, the step advances afterwards.
        - ``log(d, step=N)``: ``commit`` defaults to ``False``; the row is committed
          when the step advances or on ``finish()``.

        Returns the ``_step`` the metrics were merged into, or ``None`` if the call
        was rejected (closed store, or a backward step under
        ``step_policy="monotonic"``). A rejected call must not reach other sinks
        either, so callers should propagate the result -- and test it with
        ``is None``, since step 0 is falsy.
        """
        self._require_init()
        if self._closed or self._closing.is_set():
            logger.warning("Tracker is already finished; dropping metrics.")
            return None
        step = _normalize_step(step)
        return self.ingest(self._codec.encode(metrics), step=step, commit=commit)

    @_serialized_write
    def ingest(
        self,
        encoded: dict,
        *,
        step: int | None = None,
        commit: bool | None = None,
        accumulate: bool = False,
        reductions: Mapping[str, Reduction] | None = None,
        weight: float = 1.0,
    ) -> int | None:
        """The single path metrics take into the open row.

        ``log()`` and span durations both come through here, so the step policy,
        the merge and the commit rules only exist once. ``accumulate`` adds to any
        existing value instead of replacing it. ``reductions`` overrides that for
        peak and weighted-mean measurements; weights belong only to this open row.
        """
        self._require_init()
        if self._closed or self._closing.is_set():
            logger.warning("Tracker is already finished; dropping metrics.")
            return None
        step = _normalize_step(step)
        return self._ingest(encoded, step, commit, accumulate, reductions, weight)

    def _ingest(
        self,
        encoded: dict,
        step: int | None,
        commit: bool | None,
        accumulate: bool,
        reductions: Mapping[str, Reduction] | None,
        weight: float,
    ) -> int | None:
        now = time.time()
        pending: list[dict] = []
        with self._lock:
            if step is not None and not self._accept_step(step):
                return None
            pending.append(self._switch_open_row(step, now))
            resolved = self._open_step
            self._update_open_row(encoded, accumulate, reductions, weight)
            if resolve_commit(step, commit):
                pending.append(self._close_open_row())
        accepted = True
        for record in filter(None, pending):
            if self._emit(record) is False and record["_step"] == resolved:
                accepted = False
        return resolved if accepted else None

    @_serialized_write
    def record_span(
        self,
        metrics: dict,
        record: dict | None = None,
        *,
        reductions: Mapping[str, Reduction] | None = None,
        weight: float = 1.0,
    ) -> int | None:
        """A finished span: durations join the open row, the tree goes to disk.

        Durations never commit a step of their own -- they ride along with
        whatever ``log()`` commits, so a span costs no extra row. The metrics are
        built from floats and ints here, so they skip the encoder: spans are
        recorded per sub-step and cannot afford it.
        """
        self._require_init()
        if self._closed or self._closing.is_set():
            logger.warning("Tracker is already finished; dropping span.")
            return None
        step = (
            self._ingest(metrics, None, False, True, reductions, weight)
            if self.options.span_metrics
            else self.current_step
        )
        if record is None or self.span_writer is None or step is None:
            return step
        try:
            line = encode_line({"_step": step, **record})
        except Exception as e:
            logger.warning(f"Dropping span that cannot be serialized: {e}")
            return step
        if not self.span_writer.append_with_status(step, self._next_span_row, line):
            return None
        self._next_span_row += 1
        return step

    def _switch_open_row(self, step: int | None, now: float) -> dict | None:
        """Point the open row at ``step``, returning the row it displaced, if any.

        Without an explicit step the current open row is reused, which is what makes
        ``log(a, commit=False)`` then ``log(b)`` land on one row.
        """
        target = self.current_step if step is None else step
        if self._open_step == target:
            return None
        closed = self._close_open_row()
        self._start_open_row(target, now)
        return closed

    def _accept_step(self, step: int) -> bool:
        reference = (
            self._open_step
            if self._open_step is not None
            else self._last_committed_step
        )
        if reference is None or step >= reference:
            return True
        if self.options.step_policy == "monotonic":
            logger.warning(
                f"Dropping metrics for step {step}: step must be >= {reference} "
                '(set step_policy="allow" to keep out-of-order writes).'
            )
            return False
        return True

    def _start_open_row(self, step: int, now: float):
        self._open_step, self._open_row, self._open_time = step, {}, now
        self._open_weights.clear()

    def _update_open_row(
        self,
        encoded: dict,
        accumulate: bool = False,
        reductions: Mapping[str, Reduction] | None = None,
        weight: float = 1.0,
    ):
        for key in RESERVED_KEYS & set(encoded):
            self._codec.warn_once(
                key, f"Metric name {key!r} is reserved and will be ignored."
            )
            encoded.pop(key)
        if accumulate:
            for key, value in encoded.items():
                reduction = reductions.get(key) if reductions else None
                self._open_row[key] = self._accumulate_metric(
                    key, value, reduction, weight
                )
            return
        for key in encoded:
            self._open_weights.pop(key, None)
            if key in self._open_row:
                self._codec.warn_once(
                    key,
                    f"Metric {key!r} logged twice for step {self._open_step}; "
                    "keeping the latest value.",
                )
        self._open_row.update(encoded)

    def _accumulate_metric(
        self, key: str, value: Any, reduction: Reduction | None, weight: float
    ):
        current = self._open_row.get(key)
        if not _addable(value):
            self._open_weights.pop(key, None)
            return value
        if reduction == "mean":
            previous_weight = (
                self._open_weights.get(key, 0.0) if _addable(current) else 0.0
            )
            self._open_weights[key] = previous_weight + weight
            if previous_weight == 0:
                return value
            if weight == 0:
                return current
            return (current * previous_weight + value * weight) / (
                previous_weight + weight
            )
        self._open_weights.pop(key, None)
        if not _addable(current):
            return value
        if reduction == "max":
            # A new NaN must not disappear behind a previous finite peak.
            return value if value != value else max(current, value)
        return value + current

    def _close_open_row(self) -> dict | None:
        if self._open_step is None:
            return None
        # Metadata first, so every output format orders columns the same way;
        # _update_open_row already stripped any user metric using those names
        record = {"_step": self._open_step, "_time": self._open_time, **self._open_row}
        self._last_committed_step = self._open_step
        self._next_step = self._open_step + 1
        self._open_step, self._open_row = None, {}
        self._open_weights.clear()
        return record

    def _emit(self, record: dict) -> bool:
        """Admit a row to the cache and writer, then stage its notification."""
        try:
            line = encode_line(record)
        except Exception as e:
            logger.error(f"Dropping record that cannot be serialized: {e}")
            return False
        should_flush = self._store_row(record, line)
        if should_flush is None:
            return False
        if should_flush:
            self.writer.flush()
        else:
            self.writer.schedule_timer()
        self._evict()
        self._callback_state.pending.append(
            _Notification(record, self.options, self.on_commit, copy_context())
        )
        return True

    def _store_row(self, record: dict, line: bytes) -> bool | None:
        """Cache admitted records; return flush-due, or None for rejection.

        The write lock serializes admission and ordinal allocation. Do not
        hold the state lock while waiting for queue capacity: reads must stay live.
        """
        step = record["_step"]
        row = self._next_row
        accepted, should_flush = self.writer.enqueue_with_status(step, row, line)
        if not accepted:
            return None
        with self._lock:
            if self._last_emitted_step is not None and step <= self._last_emitted_step:
                self._needs_merge = True
            self._last_emitted_step = step
            self._next_row += 1
            self._cache.append((row, step, line))
            self._cache_bytes += len(line)
            self._last_commit_time = time.time()
        return should_flush

    def _deliver_notifications(self, target: _Notification | None = None):
        """Deliver in commit order, without holding write or state locks.

        Producers wait for their own ticket, bounding queued rows by in-flight
        calls rather than accumulating asynchronous work. A callback may recurse:
        its thread delivers earlier queued tickets before its nested commit, and
        completion is tracked separately from delivery order.

        As with synchronous recursive logging, nested delivery advances the live
        series seen by a resumed outer callback; it is not a per-callback snapshot.
        Undelivered tickets must never contribute points to that series.
        """
        check_process(self._owner_pid)
        owner = threading.get_ident()
        with self._notification_condition:
            if target is None and self._notification_owner == owner:
                # A callback's flush must not wait for its own active ancestors.
                if not self._notifications:
                    return
                target = self._notifications[-1]
        while True:
            with self._notification_condition:
                if target is not None and target.done:
                    return
                if (
                    target is None
                    and not self._notifications
                    and self._notification_owner is None
                ):
                    return
                if self._notifications and self._notification_owner in (None, owner):
                    notification = self._notifications.popleft()
                    self._notification_owner = owner
                    self._notification_depth += 1
                else:
                    self._notification_condition.wait()
                    continue
            try:
                self._notify(notification)
            finally:
                with self._notification_condition:
                    notification.done = True
                    self._notification_depth -= 1
                    if self._notification_depth == 0:
                        self._notification_owner = None
                    self._notification_condition.notify_all()

    def _notify(self, notification: _Notification):
        """Advance alert series and call hooks in the committing caller's context."""
        previous = getattr(self._callback_state, "active", False)
        self._callback_state.active = True
        try:
            notification.context.run(self._notify_callbacks, notification)
        finally:
            self._callback_state.active = previous

    def require_outside_callback(self):
        check_process(self._owner_pid)
        if getattr(self._callback_state, "active", False):
            raise RuntimeError(
                "Cannot finish or reinitialize a run inside a history commit callback."
            )
        if getattr(self._callback_state, "pending", None) is not None:
            raise RuntimeError(
                "Cannot finish or reinitialize a run inside a history "
                "write transaction."
            )
        current = threading.current_thread()
        if any(
            writer is not None and writer._worker is current
            for writer in (self.writer, self.span_writer)
        ):
            raise RuntimeError(
                "Cannot finish or reinitialize a run from a history writer worker."
            )
        if in_diagnostic_callback():
            raise RuntimeError(
                "Cannot finish or reinitialize a run from a writer diagnostic callback."
            )
        if self._lifecycle_owner == threading.get_ident():
            raise RuntimeError("Cannot reenter history lifecycle operations.")

    @contextlib.contextmanager
    def _lifecycle_operation(self):
        self.require_outside_callback()
        with self._lifecycle_lock:
            self._lifecycle_owner = threading.get_ident()
            try:
                yield
            finally:
                self._lifecycle_owner = None

    def _notify_callbacks(self, notification: _Notification):
        record, options = notification.record, notification.options
        try:
            self._series.add(record["_step"], record.get("_time") or 0.0, record)
        except Exception as e:  # alert windows must never break logging
            logger.warning(f"Could not update metric series: {e}")
        if options.print_to_screen:
            try:
                options.print_handle(f"{record}")
            except Exception as e:
                logger.warning(f"Failed to print metrics to screen: {e}")
        if notification.on_commit is not None:
            try:
                notification.on_commit(record)
            except Exception as e:
                logger.warning(f"on_commit callback failed: {e}")

    def _evict(self):
        """Drop the oldest cached rows once over budget, but never undurable ones."""
        if self.writer is None or self._evict_durable():
            return
        # Only rows that are not on disk yet are left: persist them, then retry
        self.writer.flush()
        self._evict_durable()

    def _evict_durable(self) -> bool:
        """Drop durable rows while over budget; returns whether the budget is met.

        The watermark is a row ordinal, not a step: patch lines repeat a step, so a
        step watermark would wrongly consider them already written.
        """
        with self._lock:
            flushed_row = self.writer.flushed_row
            while (
                self._over_budget() and self._cache and self._cache[0][0] <= flushed_row
            ):
                self._cache_bytes -= len(self._cache.popleft()[2])
                self._has_disk_prefix = True
                self._evicted_rows += 1
            return not self._over_budget()

    def _over_budget(self) -> bool:
        return (
            self._cache_bytes > self.options.cache_bytes
            or len(self._cache) > self.options.cache_rows
        )

    # ------------------------------------------------------------------ lifecycle

    def flush(self, commit_open: bool = False):
        """Flush rows and notifications, optionally committing the open row.

        Inside a callback, only pending notifications are drained: active callback
        ancestors cannot finish until this call returns.
        """
        check_process(self._owner_pid)
        self._flush(commit_open)
        self._deliver_notifications()

    @_serialized_write
    def _flush(self, commit_open: bool):
        record = None
        if commit_open:
            with self._lock:
                record = self._close_open_row()
        if record is not None:
            self._emit(record)
        for writer in (self.writer, self.span_writer):
            if writer is not None:
                writer.flush()

    def finish(self):
        """Drain writes and close, logging I/O failures rather than masking errors."""
        self.require_outside_callback()
        self._closing.set()
        with self._lifecycle_operation():
            # An init ahead of us may have cleared the event while we waited.
            self._closing.set()
            self._finish()

    def _finish(self):
        with self._write_lock, self._lock:
            self._closed = True
        try:
            self.flush(commit_open=True)
        except Exception as e:
            logger.warning(f"Failed to flush history on finish: {e}")
        self._deliver_notifications()
        with self._write_lock:
            for writer in (self.writer, self.span_writer):
                if writer is not None:
                    try:
                        writer.close()
                    except Exception as e:
                        logger.warning(f"Failed to close {writer.path}: {e}")
        if self._atexit_registered:
            atexit.unregister(self._atexit)
            self._atexit_registered = False

    def _atexit(self):
        if self._owner_pid != process_id():
            return
        with contextlib.suppress(Exception):  # pragma: no cover - interpreter shutdown
            self.finish()

    # ------------------------------------------------------------------ queries

    @property
    def current_step(self) -> int:
        check_process(self._owner_pid)
        with self._lock:
            return self._open_step if self._open_step is not None else self._next_step

    @property
    def series(self) -> MetricSeries:
        """Numeric history through the notification currently being delivered."""
        check_process(self._owner_pid)
        return self._series

    @property
    def last_commit_time(self) -> float | None:
        """When the latest metric row was accepted, independently of its sample time."""
        check_process(self._owner_pid)
        with self._lock:
            return self._last_commit_time

    def open_record(self) -> dict | None:
        check_process(self._owner_pid)
        with self._lock:
            if self._open_step is None:
                return None
            return {
                "_step": self._open_step,
                "_time": self._open_time,
                **self._open_row,
            }

    def stats(self) -> dict:
        check_process(self._owner_pid)
        writer, span_writer = self.writer, self.span_writer
        pending_bytes = writer.pending_bytes if writer else 0
        span_pending_bytes = span_writer.pending_bytes if span_writer else 0
        with self._lock:
            return {
                "log_dir": self.log_dir.as_posix() if self.log_dir else None,
                "metrics_file": self.log_fp.as_posix() if self.log_fp else None,
                "rows_on_disk": self.writer.lines if self.writer else 0,
                "last_step": self._last_committed_step,
                "rows_logged": self._next_row,
                "pending_bytes": pending_bytes,
                "span_pending_bytes": span_pending_bytes,
                "dropped_records": self.writer.dropped if self.writer else 0,
                "dropped_spans": self.span_writer.dropped if self.span_writer else 0,
                "cached_rows": len(self._cache),
                "cached_bytes": self._cache_bytes,
                "cache_limit_bytes": self.options.cache_bytes,
                "disk_prefix": self._has_disk_prefix,
                "evicted_rows": self._evicted_rows,
                "queries": self._queries,
                # Queries the cache could not answer alone; the rest were served
                # entirely from memory
                "disk_queries": self._disk_queries,
            }

    def get(
        self,
        n: int | None = 50,
        *,
        output_type: str = "dict",
        metrics: Sequence[str] | None = None,
        step_range: tuple[int | None, int | None] | None = None,
        include_meta: bool = True,
        include_open: bool = True,
        fill_missing: bool = False,
        dropna: bool = False,
    ):
        self._require_init()
        with self._lock:
            self._queries += 1
        records = self._collect(n, step_range, include_open)
        rows = project(
            records,
            metrics=metrics,
            include_meta=include_meta,
            fill_missing=fill_missing,
            dropna=dropna,
        )
        return to_output(rows, output_type)

    def _collect(
        self,
        n: int | None,
        step_range: tuple[int | None, int | None] | None,
        include_open: bool,
    ) -> list[dict]:
        limit = None if n is None or n < 0 else max(0, n)
        if limit == 0:
            return []
        writer = self.writer
        assert writer is not None
        # Error handlers can inspect history while a writer lock is held.
        # Keep snapshot lock ordering consistent with that path.
        with writer._lock, self._lock:
            opened = self._open_rows(include_open, step_range)
            merge = (
                self._needs_merge
                or bool(opened and self._open_repeats_a_step())
                or bool(self.writer and self.writer.dropped)
            )
            view = (
                self._view_tail(limit, merge=merge)
                if step_range is None
                else self._view_range(step_range, merge=merge)
            )
        if step_range is None:
            records = self._collect_tail(limit, view, opened, merge=merge) + opened
            return self._take_steps(records, limit, merge=merge)
        records = self._collect_range(step_range, view) + opened
        # Ranges are selected in step order, independently of write recency.
        if merge:
            records = merge_steps(records)
        return records if limit is None else records[-limit:]

    def _open_repeats_a_step(self) -> bool:
        """Whether the open row carries a step that was already committed.

        That happens when a row is explicitly committed and more metrics arrive
        for the same step, so results have to be merged even though
        no second row has been emitted yet.
        """
        with self._lock:
            return (
                self._open_step is not None
                and self._last_emitted_step is not None
                and self._open_step <= self._last_emitted_step
            )

    def _open_rows(
        self, include_open: bool, step_range: tuple[int | None, int | None] | None
    ) -> list[dict]:
        """The still-uncommitted row, when it is wanted and falls in range."""
        record = self.open_record() if include_open else None
        if record is None:
            return []
        if step_range is not None and not _in_range(record["_step"], step_range):
            return []
        return [record]

    def _view_tail(self, steps: int | None, *, merge: bool = False) -> _CacheView:
        """Snapshot the cache rows covering the newest ``steps`` steps."""
        with self._lock:
            if steps is None or merge:
                return self._make_view(list(self._cache), complete=False)
            rows, complete = _scan_tail(self._cache, steps)
            return self._make_view(rows, complete)

    def _view_range(
        self, step_range: tuple[int | None, int | None], *, merge: bool = False
    ) -> _CacheView:
        """Snapshot the cache rows inside ``step_range``."""
        start = step_range[0]
        with self._lock:
            if not self._cache:
                # Nothing cached: _view decides whether the disk still has to answer
                return self._make_view([], complete=False)
            oldest = self._cache[0][1]
            if merge:  # steps may be unordered: nothing can be skipped
                rows = [e for e in self._cache if _in_range(e[1], step_range)]
                return self._make_view(rows, complete=False)
            forward = _nearer_front(step_range, oldest, self._cache[-1][1])
            rows = _scan_range(self._cache, step_range, forward)
            # Everything asked for starts at or after the oldest cached step
            return self._make_view(rows, complete=start is not None and start >= oldest)

    def _make_view(self, rows: list, complete: bool) -> _CacheView:
        """Wrap a cache slice; query snapshots hold writer then store-state locks."""
        first = self._cache[0] if self._cache else None
        disk_end = None
        if self.writer is not None and self.writer.dropped:
            # Dropped ordinals cannot locate a physical boundary. Snapshot the
            # durable file and merge its overlap with the newer cached rows.
            disk_end = self.writer.size
        return _CacheView(
            rows=rows,
            complete=complete or not self._has_disk_prefix or self.writer is None,
            start_row=first[0] if first else self._next_row,
            disk_end=disk_end,
        )

    def _take_steps(
        self, records: list[dict], limit: int | None, merge: bool = False
    ) -> list[dict]:
        """Merge rows by step and keep the newest ``limit`` steps.

        Several rows can share a step (patch lines), so trimming counts steps rather
        than rows; otherwise the oldest step comes back half-merged.
        """
        if limit is not None:
            wanted = _newest_steps(records, limit)
            records = [r for r in records if r.get("_step") in wanted]
        return merge_steps(records) if self._needs_merge or merge else records

    def _disk_reader(self, view: _CacheView) -> tuple[JsonlReader, int]:
        """A reader plus the byte offset where the cached rows begin.

        The boundary is a physical row ordinal, not a step: patch lines repeat a
        step, and a step lookup would also exclude that step's earlier row. Row
        numbers shift once records are dropped, so that case merges an overlapping
        durable snapshot instead. Eviction only removes durable rows: a read never
        needs to flush pending metrics, spans, or commit notifications.
        """
        with self._lock:
            self._disk_queries += 1
        assert self.writer is not None
        reader = self.writer.reader()
        if view.disk_end is not None:
            return reader, view.disk_end
        return reader, reader.offset_of_line(view.start_row)

    def _collect_tail(
        self,
        steps: int | None,
        view: _CacheView,
        opened: list[dict],
        *,
        merge: bool,
    ) -> list[dict]:
        """Rows covering the newest ``steps`` steps (``None`` for the whole run)."""
        if steps == 0:
            return []
        records = view.records()
        if view.complete:
            return records
        return (
            self._older(
                lambda: self._older_tail(view, steps, records + opened, merge=merge)
            )
            + records
        )

    def _older(self, fetch: Callable[[], list[dict]]) -> list[dict]:
        """Rows from before the cache; an unreadable file degrades to cache-only.

        A query must not take the training loop down, so a partial answer with a
        warning beats an exception.
        """
        try:
            return fetch()
        except Exception as e:
            logger.warning(
                f"Could not read older history from {self.log_fp}: {e}; "
                "returning cached rows only."
            )
            return []

    def _older_tail(
        self,
        view: _CacheView,
        steps: int | None,
        newer: list[dict],
        *,
        merge: bool,
    ) -> list[dict]:
        """The rows before the cache that are still needed to reach ``steps`` steps."""
        reader, end = self._disk_reader(view)
        if steps is None:
            if view.start_row > BIG_READ_WARN_ROWS:
                logger.warning(
                    "Reading the full history from disk; this may take a while."
                )
            return reader.parse_rows(reader.iter_raw(0, end))
        if not merge and not reader.merge:
            return reader.tail(max(steps - len(newer), 0), end)
        selected = reader.select_recent_steps(
            steps, end, newer=(record["_step"] for record in reversed(newer))
        )
        return reader.read_selected_steps(selected, end)

    def _collect_range(
        self, step_range: tuple[int | None, int | None], view: _CacheView
    ) -> list[dict]:
        """Rows whose step falls inside ``step_range``."""
        records = view.records()
        if view.complete:
            return records
        return self._older(lambda: self._older_range(view, step_range)) + records

    def _older_range(
        self, view: _CacheView, step_range: tuple[int | None, int | None]
    ) -> list[dict]:
        """The rows before the cache that fall inside ``step_range``."""
        reader, end = self._disk_reader(view)
        # Seeks to the first matching step and stops past the last one, so the scan
        # costs O(range) rather than O(everything before the cache)
        return [
            record
            for record in reader.read_steps(step_range[0], step_range[1], end)
            if _in_range(record.get("_step"), step_range)
        ]


# ---------------------------------------------------------------------- helpers


def _addable(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_cached(entries: Sequence[tuple[int, int, bytes]]) -> list[dict]:
    records = []
    for _, _, line in entries:
        try:
            records.append(json.loads(line))
        except Exception as e:  # pragma: no cover - cache is produced here
            logger.warning(f"Skipping corrupted cached record: {e}")
    return records


def _scan_tail(
    cache: Sequence[tuple[int, int, bytes]], steps: int
) -> tuple[list[tuple[int, int, bytes]], bool]:
    """Cache entries covering the newest ``steps`` steps, newest last.

    Requires strictly ascending steps. The walk stops when an older step shows up:
    a query costs O(rows returned) rather than O(cache). Seeing that older step is
    also the proof that the newest ``steps`` steps are wholly cached, which the
    second return value reports.
    """
    seen: set = set()
    rows: list = []
    for entry in reversed(cache):
        if entry[1] not in seen:
            if len(seen) == steps:
                rows.reverse()
                return rows, True
            seen.add(entry[1])
        rows.append(entry)
    rows.reverse()
    return rows, False


def _nearer_front(
    step_range: tuple[int | None, int | None], oldest: int, newest: int
) -> bool:
    """Whether ``step_range`` sits nearer the oldest cached step than the newest."""
    start, end = step_range
    from_front = (start if start is not None else oldest) - oldest
    from_back = newest - (end - 1 if end is not None else newest)
    return from_front <= from_back


def _scan_range(
    cache: Sequence[tuple[int, int, bytes]],
    step_range: tuple[int | None, int | None],
    forward: bool,
) -> list[tuple[int, int, bytes]]:
    """Cache entries inside ``step_range``, scanned from the nearer end.

    Steps ascend here (callers exclude the unordered case), so the scan can stop at
    the far edge of the range and a narrow range costs O(range), not O(cache).
    """
    stop = step_range[1] if forward else step_range[0]
    beyond = operator.ge if forward else operator.lt  # the far edge of the range
    rows: list = []
    for entry in cache if forward else reversed(cache):
        step = entry[1]
        if stop is not None and beyond(step, stop):
            break
        if _in_range(step, step_range):
            rows.append(entry)
    return rows if forward else rows[::-1]


def _newest_steps(records: Sequence[dict], limit: int) -> set:
    """The newest ``limit`` step values present in ``records``."""
    return recent_steps((record["_step"] for record in reversed(records)), limit)


def _in_range(step, step_range: tuple[int | None, int | None]) -> bool:
    start, end = step_range
    if not isinstance(step, int):
        return False
    return (start is None or step >= start) and (end is None or step < end)
