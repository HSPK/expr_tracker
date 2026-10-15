"""Step-oriented metric history: open-row assembly, in-memory cache, queries.

Write path::

    log() --merge--> open row --commit--> cache + writer --notify--> series + callbacks

Reads are served from the cache; disk is touched only once rows have been
evicted (or after a resume), addressed by physical row ordinal.
"""

from __future__ import annotations

import atexit
import contextlib
import io
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from loguru import logger

from .._native import (
    NativeCommitQueue,
    NativeHistoryState,
    NativeRecord,
    NativeSummary,
    history_atomic_write,
    history_nearer_front,
    history_scan_range,
    history_scan_tail,
)
from .._process import check_process, process_id
from ._store_types import (
    BIG_READ_WARN_ROWS as BIG_READ_WARN_ROWS,
)
from ._store_types import (
    DEFAULT_CACHE_BYTES as DEFAULT_CACHE_BYTES,
)
from ._store_types import (
    DEFAULT_CACHE_ROWS as DEFAULT_CACHE_ROWS,
)
from ._store_types import (
    STEP_POLICIES as STEP_POLICIES,
)
from ._store_types import (
    HistoryOptions as HistoryOptions,
)
from ._store_types import (
    Reduction as Reduction,
)
from ._store_types import (
    _CacheView as _CacheView,
)
from ._store_types import (
    _Notification as _Notification,
)
from ._store_types import (
    _parse_cached as _parse_cached,
)
from ._store_types import (
    _serialized_write as _serialized_write,
)
from ._store_types import (
    resolve_commit as resolve_commit,
)
from ._store_writes import HistoryWrites
from .codec import RecordCodec, encode_line
from .frame import project_owned as project
from .frame import to_output
from .naming import (
    metrics_filename,
    resolve_log_dir,
    sidecar_filename,
    spans_filename,
)
from .reader import _decode
from .series import MetricSeries
from .writer import (
    JsonlWriter,
    in_diagnostic_callback,
)


class HistoryStore(HistoryWrites):
    """Local jsonl history: the ``jsonl`` backend and the source of ``et.history()``."""

    def __init__(self):
        self._owner_pid = process_id()
        self._lock = threading.RLock()
        self._write_lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._lifecycle_owner: int | None = None
        self._notify_queue = NativeCommitQueue()
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
            self._inline_callback: Callable[[dict], None] | None = None
            self._inline_summary: Callable[[], NativeSummary | None] | None = None
            self.started_at = time.time()
            self._series = MetricSeries(options.alert_window)
            self._native = NativeHistoryState(
                options.cache_bytes,
                options.cache_rows,
                options.step_policy == "allow",
            )
            self._closed = False
            self._closing.clear()
            self._codec.reset()
            self._next_span_row = 0

    def __getattr__(self, name):
        if name == "_notifications":
            return tuple(range(self._notify_queue.count))
        if name == "_notification_owner":
            return self._notify_queue.owner
        if name == "_notification_depth":
            return self._notify_queue.depth
        if name in {
            "_cache",
            "_cache_bytes",
            "_has_disk_prefix",
            "_needs_merge",
            "_queries",
            "_disk_queries",
            "_evicted_rows",
            "_open_step",
            "_open_row",
            "_open_time",
            "_next_step",
            "_next_row",
            "_last_committed_step",
            "_last_emitted_step",
            "_last_commit_time",
            "_open_weights",
        }:
            check_process(self._owner_pid)
            value = self._native.get(name)
            if name in {
                "_open_step",
                "_next_step",
                "_last_committed_step",
                "_last_emitted_step",
            }:
                return None if value is None else int(value)
            if name == "_open_row":
                return json.loads(value)
            if name == "_cache":
                return [(row, int(step), line.encode()) for row, step, line in value]
            return value
        raise AttributeError(name)

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
        self._native.resume(
            writer.lines,
            None if writer.max_step is None else str(writer.max_step),
            writer.has_duplicate_steps or not writer.sorted,
        )
        if writer.max_step is None:
            return
        # The highest step, not the last written one: a patched or out-of-order
        # file ends on an earlier step, and resuming there would reuse it
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
        try:
            encoded = self._codec.encode(config, kind="config")
            boundary = io.StringIO()
            json.dump(encoded, boundary, indent=4, ensure_ascii=False)
            history_atomic_write(
                self.config_fp,
                boundary.getvalue().encode("utf-8", "backslashreplace"),
            )
        except Exception as e:
            logger.error(f"Failed to write config to {self.config_fp}: {e}")

    def _require_init(self):
        check_process(self._owner_pid)
        if self.writer is None:
            raise RuntimeError(
                "HistoryStore.init() must be called before logging or querying."
            )

    # ------------------------------------------------------------------ writing

    def _ingest(
        self,
        encoded: dict,
        step: int | None,
        commit: bool | None,
        accumulate: bool,
        reductions: Mapping[str, Reduction] | None,
        weight: float,
    ) -> int | None:
        arguments = (
            None if step is None else str(step),
            bool(resolve_commit(step, commit)),
            accumulate,
            dict(reductions or {}),
            weight,
            time.time(),
        )
        try:
            result = self._native.ingest_values(encoded, *arguments)
            if result is None:
                payload = encode_line(encoded).decode("utf-8")
                result = self._native.ingest(payload, *arguments)
        except (ValueError, TypeError, OverflowError, RecursionError) as error:
            logger.error(f"Dropping record that cannot be serialized: {error}")
            return None
        resolved, pending, warnings = result
        for key, message in warnings:
            if key:
                self._codec.warn_once(key, message)
            else:
                logger.warning(message)
        if resolved is None:
            return None
        resolved = int(resolved)
        accepted = True
        for text in pending:
            record = self._record_from_json(text)
            # Preserve native serialization while SDK hooks receive a decoded row.
            previous = getattr(self._callback_state, "encoded_row", None)
            self._callback_state.encoded_row = (
                record,
                text if isinstance(text, NativeRecord) else text + "\n",
            )
            try:
                emitted = self._emit(record)
            finally:
                if previous is None:
                    del self._callback_state.encoded_row
                else:
                    self._callback_state.encoded_row = previous
            if emitted is False and record["_step"] == resolved:
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

    def _emit(self, record: dict) -> bool:
        """Admit a row to the cache and writer, then stage its notification."""
        try:
            encoded = getattr(self._callback_state, "encoded_row", None)
            line = (
                encoded[1]
                if encoded is not None
                and encoded[0] is record
                and isinstance(encoded[1], NativeRecord)
                else encoded[1].encode("utf-8")
                if encoded is not None and encoded[0] is record
                else encode_line(record)
            )
        except Exception as e:
            logger.error(f"Dropping record that cannot be serialized: {e}")
            return False
        should_flush = self._store_row(record, line)
        if should_flush is None:
            return False
        self._schedule_commit(should_flush)
        self._stage_notification(record)
        return True

    def _deliver_notifications(self, target=None):
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
        self._notify_queue.drain(threading.get_ident(), self._notify, target)

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
        return self._native.evict_writer(self.writer._native)

    def _over_budget(self) -> bool:
        return self._native.get("_over_budget")

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
        return int(self._native.get("current_step"))

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
        record = self._native.open_record()
        return json.loads(record) if record is not None else None

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
                "cached_rows": self._native.get("cached_rows"),
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
        writer = self.writer
        assert writer is not None
        start, end = step_range if step_range is not None else (None, None)
        records, warning = self._native.collect(
            writer._native,
            limit,
            None if start is None else str(start),
            None if end is None else str(end),
            include_open,
            step_range is not None,
        )
        if warning is not None:
            logger.warning(warning)
        return _decode(records)

    def _view_tail(self, steps: int | None, *, merge: bool = False) -> _CacheView:
        """Snapshot the cache rows covering the newest ``steps`` steps."""
        rows, complete, start = self._native.view_tail(steps, merge)
        return _CacheView(
            [(row, int(step), line.encode()) for row, step, line in rows],
            complete,
            start,
            None,
        )


def _scan_tail(
    cache: Sequence[tuple[int, int, bytes]], steps: int
) -> tuple[list[tuple[int, int, bytes]], bool]:
    """Cache entries covering the newest ``steps`` steps, newest last.

    Requires strictly ascending steps. The walk stops when an older step shows up:
    a query costs O(rows returned) rather than O(cache). Seeing that older step is
    also the proof that the newest ``steps`` steps are wholly cached, which the
    second return value reports.
    """
    indices, complete = history_scan_tail([str(e[1]) for e in cache], steps)
    return [cache[index] for index in indices], complete


def _nearer_front(
    step_range: tuple[int | None, int | None], oldest: int, newest: int
) -> bool:
    """Whether ``step_range`` sits nearer the oldest cached step than the newest."""
    start, end = step_range
    return history_nearer_front(
        None if start is None else str(start),
        None if end is None else str(end),
        str(oldest),
        str(newest),
    )


def _scan_range(
    cache: Sequence[tuple[int, int, bytes]],
    step_range: tuple[int | None, int | None],
    forward: bool,
) -> list[tuple[int, int, bytes]]:
    """Cache entries inside ``step_range``, scanned from the nearer end.

    Steps ascend here (callers exclude the unordered case), so the scan can stop at
    the far edge of the range and a narrow range costs O(range), not O(cache).
    """
    start, end = step_range
    indices = history_scan_range(
        [str(entry[1]) for entry in cache],
        None if start is None else str(start),
        None if end is None else str(end),
        forward,
    )
    return [cache[index] for index in indices]
