"""Step-oriented metric history: open-row assembly, in-memory cache, queries.

Write path::

    log() --merge--> open row (current step) --commit--> cache + series + writer

Reads are served from the cache; disk is touched only once rows have been
evicted (or after a resume), addressed by physical row ordinal.
"""

from __future__ import annotations

import atexit
import json
import operator
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from loguru import logger

from .codec import RESERVED_KEYS, RecordCodec, encode_line
from .frame import project, to_output
from .naming import (
    metrics_filename,
    sidecar_filename,
    spans_filename,
    validate_stream,
)
from .reader import JsonlReader, merge_steps
from .series import DEFAULT_WINDOW, MetricSeries
from .writer import (
    DEFAULT_BUFFER_INTERVAL,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_MAX_BUFFER_SECONDS,
    DEFAULT_MAX_PENDING_RECORDS,
    JsonlWriter,
)

DEFAULT_CACHE_BYTES = 1 << 30  # 1 GiB
DEFAULT_CACHE_ROWS = 2_000_000
DEFAULT_MAX_OPEN_SECONDS = 60.0
BIG_READ_WARN_ROWS = 2_000_000


STEP_POLICIES = ("monotonic", "allow")


def resolve_commit(step: int | None, commit: bool | None) -> bool:
    """Without an explicit ``commit``: a step-less log commits, a stepped one waits."""
    return (step is None) if commit is None else commit


@dataclass(frozen=True)
class HistoryOptions:
    """Tunables for a run's local history, validated once at ``init()``."""

    # disk buffering
    buffer_size: int = DEFAULT_BUFFER_SIZE
    buffer_interval: float | None = DEFAULT_BUFFER_INTERVAL
    max_buffer_seconds: float | None = DEFAULT_MAX_BUFFER_SECONDS
    max_pending_records: int = DEFAULT_MAX_PENDING_RECORDS
    # in-memory cache
    cache_bytes: int = DEFAULT_CACHE_BYTES
    cache_rows: int = DEFAULT_CACHE_ROWS
    # step assembly
    max_open_seconds: float | None = DEFAULT_MAX_OPEN_SECONDS
    step_policy: str = "monotonic"
    rank_aware: bool = True
    # an independent producer writing its own file with its own step cursor
    stream: str | None = None
    # write the full span tree to spans.jsonl beside the metrics
    spans: bool = True
    # run-wide defaults for et.span(); a span's own arguments win
    span_print_fn: Callable[[str], None] | None = None
    span_plugins: Sequence[Any] = ()
    # alerts and output
    alert_window: int = DEFAULT_WINDOW
    print_to_screen: bool = False
    print_handle: Callable[[str], None] = print

    def __post_init__(self):
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
            "max_open_seconds": (
                None if self.max_open_seconds is None else float(self.max_open_seconds)
            ),
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
    start_step: int | None

    def records(self) -> list[dict]:
        return _parse_cached(self.rows)


class HistoryStore:
    """Local jsonl history: the ``jsonl`` backend and the source of ``et.history()``."""

    def __init__(self):
        self._lock = threading.RLock()
        self._codec = RecordCodec()
        self._open_timer: threading.Timer | None = None
        self._open_generation = 0
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
            self._codec.reset()
            self._open_step: int | None = None
            self._open_row: dict = {}
            self._open_time = 0.0
            self._open_generation += 1
            self._next_step = 0
            self._next_row = 0
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
        on_commit: Callable[[dict], None] | None = None,
        **options,
    ) -> HistoryStore:
        """Open (or resume) a run. Tunables are named in :class:`HistoryOptions`."""
        opts = HistoryOptions.from_kwargs(**options)
        if name is None:
            name = time.strftime("run-%Y%m%d-%H%M%S")
            logger.warning(f"No run name provided, using generated name: {name}")
        self.project, self.name = project, name
        self.log_dir = Path(dir or "./tracker/jsonl") / project / name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stream = opts.stream
        self.config_fp = self.log_dir / sidecar_filename("config", opts.stream, "json")
        self.spans_fp = (
            self.log_dir / spans_filename(opts.stream, opts.rank_aware)
            if opts.spans
            else None
        )
        self.log_fp = self.log_dir / metrics_filename(opts.stream, opts.rank_aware)

        self._close_previous()
        self._reset(opts, on_commit)
        self.writer = self._open_writer(opts)
        self.span_writer = (
            JsonlWriter(
                self.spans_fp,
                buffer_size=opts.buffer_size,
                buffer_interval=opts.buffer_interval,
                max_buffer_seconds=opts.max_buffer_seconds,
                max_pending_records=opts.max_pending_records,
            )
            if self.spans_fp is not None
            else None
        )
        self._resume(opts.alert_window)
        self._write_config(config)
        self._register_atexit()
        return self

    def _close_previous(self):
        """Re-initialising must not strand the previous writer's buffer or timer."""
        self._cancel_open_timer()
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
        try:
            with open(self.config_fp, "w", encoding="utf-8") as f:
                json.dump(
                    self._codec.encode(config, kind="config"),
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.error(f"Failed to write config to {self.config_fp}: {e}")

    def _require_init(self):
        if self.writer is None:
            raise RuntimeError(
                "HistoryStore.init() must be called before logging or querying."
            )

    # ------------------------------------------------------------------ writing

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
        return self.ingest(self._codec.encode(metrics), step=step, commit=commit)

    def ingest(
        self,
        encoded: dict,
        *,
        step: int | None = None,
        commit: bool | None = None,
        accumulate: bool = False,
    ) -> int | None:
        """The single path metrics take into the open row.

        ``log()`` and span durations both come through here, so the step policy,
        the merge and the commit rules only exist once. ``accumulate`` adds to any
        existing value instead of replacing it, which is what repeated spans in
        one step need.
        """
        self._require_init()
        now = time.time()
        pending: list[dict] = []
        with self._lock:
            if self._closed:
                logger.warning("Tracker is already finished; dropping metrics.")
                return None
            if step is not None and not self._accept_step(step):
                return None
            pending.append(self._switch_open_row(step, now))
            resolved = self._open_step
            self._update_open_row(encoded, accumulate)
            if resolve_commit(step, commit):
                pending.append(self._close_open_row())
        for record in filter(None, pending):
            self._emit(record)
        self._schedule_open_timer()
        return resolved

    def record_span(self, metrics: dict, record: dict | None = None) -> int | None:
        """A finished span: durations join the open row, the tree goes to disk.

        Durations never commit a step of their own -- they ride along with
        whatever ``log()`` commits, so a span costs no extra row. The metrics are
        built from floats and ints here, so they skip the encoder: spans are
        recorded per sub-step and cannot afford it.
        """
        step = self.ingest(metrics, commit=False, accumulate=True)
        if record is None or self.span_writer is None or step is None:
            return step
        try:
            line = encode_line({"_step": step, **record})
        except Exception as e:
            logger.warning(f"Dropping span that cannot be serialized: {e}")
            return step
        # enqueue, not append: the batch goes out when the step commits, so a
        # span does not pay for a flush decision of its own
        if self.span_writer.enqueue(step, self.span_writer.lines, line):
            self.span_writer.flush()
        return step

    def _switch_open_row(self, step: int | None, now: float) -> dict | None:
        """Point the open row at ``step``, returning the row it displaced, if any.

        Without an explicit step the current open row is reused, which is what makes
        ``log(a, commit=False)`` then ``log(b)`` land on one row.
        """
        target = self._next_step if step is None else step
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
        self._open_generation += 1
        timer, self._open_timer = self._open_timer, None
        if timer is not None:
            timer.cancel()

    def _update_open_row(self, encoded: dict, accumulate: bool = False):
        for key in RESERVED_KEYS & set(encoded):
            self._codec.warn_once(
                key, f"Metric name {key!r} is reserved and will be ignored."
            )
            encoded.pop(key)
        if accumulate:
            # Repeated spans in one step sum rather than overwrite
            for key, value in encoded.items():
                current = self._open_row.get(key)
                self._open_row[key] = value + current if _addable(current) else value
            return
        for key in encoded:
            if key in self._open_row:
                self._codec.warn_once(
                    key,
                    f"Metric {key!r} logged twice for step {self._open_step}; "
                    "keeping the latest value.",
                )
        self._open_row.update(encoded)

    def _close_open_row(self) -> dict | None:
        if self._open_step is None:
            return None
        # Metadata first, so every output format orders columns the same way;
        # _update_open_row already stripped any user metric using those names
        record = {"_step": self._open_step, "_time": self._open_time, **self._open_row}
        self._last_committed_step = self._open_step
        self._next_step = self._open_step + 1
        self._open_step, self._open_row = None, {}
        return record

    def _emit(self, record: dict):
        """Push a committed row to the cache, series and writer, then notify."""
        try:
            line = encode_line(record)
        except Exception as e:
            logger.error(f"Dropping record that cannot be serialized: {e}")
            return
        if self._store_row(record, line):
            self.writer.flush()
        else:
            self.writer.schedule_timer()
        self._evict()
        self._notify(record)

    def _store_row(self, record: dict, line: bytes) -> bool:
        """Record the row everywhere in-memory; returns whether a flush is due.

        Allocating the ordinal and buffering the line must be atomic, or concurrent
        logging would break the ``row == physical line`` invariant.
        """
        step = record["_step"]
        with self._lock:
            if self._last_emitted_step is not None and step <= self._last_emitted_step:
                self._needs_merge = True
            self._last_emitted_step = step
            row = self._next_row
            self._next_row += 1
            self._cache.append((row, step, line))
            self._cache_bytes += len(line)
            self._last_commit_time = record.get("_time") or time.time()
            # Allocation and enqueue must stay adjacent and unfailable; anything that
            # can raise in between would consume an ordinal without writing a line
            should_flush = self.writer.enqueue(step, row, line)
        try:
            self._series.add(step, record.get("_time") or 0.0, record)
        except Exception as e:  # alert windows must never break logging
            logger.warning(f"Could not update metric series: {e}")
        return should_flush

    def _notify(self, record: dict):
        """Hand the committed row to the screen and to the alert engine."""
        if self.options.print_to_screen:
            try:
                self.options.print_handle(f"{record}")
            except Exception as e:
                logger.warning(f"Failed to print metrics to screen: {e}")
        if self.on_commit is not None:
            try:
                self.on_commit(record)
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

    def _schedule_open_timer(self):
        if self.options.max_open_seconds is None:
            return
        with self._lock:
            if self._open_step is None or self._open_timer is not None:
                return
            generation = self._open_generation
            timer = threading.Timer(
                self.options.max_open_seconds, self._on_open_timeout, args=(generation,)
            )
            timer.daemon = True
            self._open_timer = timer
        timer.start()

    def _cancel_open_timer(self):
        with self._lock:
            timer, self._open_timer = self._open_timer, None
        if timer is not None:
            timer.cancel()

    def _on_open_timeout(self, generation: int):
        """Commit the open row on timeout; ``generation`` pins the row it timed."""
        with self._lock:
            self._open_timer = None
            if self._closed or generation != self._open_generation:
                return
            record = self._close_open_row()
        if record is not None:
            self._emit(record)

    # ------------------------------------------------------------------ lifecycle

    def flush(self, commit_open: bool = False):
        """Write buffered rows to disk, also committing the open row if asked."""
        record = None
        if commit_open:
            self._cancel_open_timer()
            with self._lock:
                record = self._close_open_row()
        if record is not None:
            self._emit(record)
        for writer in (self.writer, self.span_writer):
            if writer is not None:
                writer.flush()

    def finish(self):
        """Close the store. Never raises: a bad disk must not mask the real error."""
        with self._lock:
            self._closed = True
        self._cancel_open_timer()
        try:
            self.flush(commit_open=True)
        except Exception as e:
            logger.warning(f"Failed to flush history on finish: {e}")
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
        try:
            self.flush(commit_open=True)
            if self.writer is not None:
                self.writer.close()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass

    # ------------------------------------------------------------------ queries

    @property
    def current_step(self) -> int:
        with self._lock:
            return self._open_step if self._open_step is not None else self._next_step

    @property
    def series(self) -> MetricSeries:
        return self._series

    @property
    def last_commit_time(self) -> float | None:
        """Timestamp of the last committed step, used by ``no_data()``/``age()``."""
        with self._lock:
            return self._last_commit_time

    def open_record(self) -> dict | None:
        with self._lock:
            if self._open_step is None:
                return None
            return {
                "_step": self._open_step,
                "_time": self._open_time,
                **self._open_row,
            }

    def stats(self) -> dict:
        with self._lock:
            return {
                "log_dir": self.log_dir.as_posix() if self.log_dir else None,
                "metrics_file": self.log_fp.as_posix() if self.log_fp else None,
                "rows_on_disk": self.writer.lines if self.writer else 0,
                "last_step": self._last_committed_step,
                "rows_logged": self._next_row,
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
        if step_range is None:
            records = self._collect_tail(limit)
        else:
            records = self._collect_range(step_range)
        records += self._open_rows(include_open, step_range)
        return self._take_steps(records, limit, merge=self._open_repeats_a_step())

    def _open_repeats_a_step(self) -> bool:
        """Whether the open row carries a step that was already committed.

        That happens when a ``max_open_seconds`` timeout commits a row and more
        metrics arrive for the same step, so results have to be merged even though
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

    def _view_tail(self, steps: int | None) -> _CacheView:
        """Snapshot the cache rows covering the newest ``steps`` steps."""
        with self._lock:
            if steps is None:
                return self._make_view(list(self._cache), complete=False)
            rows, complete = _scan_tail(self._cache, steps)
            return self._make_view(rows, complete)

    def _view_range(self, step_range: tuple[int | None, int | None]) -> _CacheView:
        """Snapshot the cache rows inside ``step_range``."""
        start = step_range[0]
        with self._lock:
            if not self._cache:
                # Nothing cached: _view decides whether the disk still has to answer
                return self._make_view([], complete=False)
            oldest = self._cache[0][1]
            if self._needs_merge:  # steps may be unordered: nothing can be skipped
                rows = [e for e in self._cache if _in_range(e[1], step_range)]
                return self._make_view(rows, complete=False)
            forward = _nearer_front(step_range, oldest, self._cache[-1][1])
            rows = _scan_range(self._cache, step_range, forward)
            # Everything asked for starts at or after the oldest cached step
            return self._make_view(rows, complete=start is not None and start >= oldest)

    def _make_view(self, rows: list, complete: bool) -> _CacheView:
        """Wrap a cache slice with its boundary. Must be called holding the lock."""
        first = self._cache[0] if self._cache else None
        return _CacheView(
            rows=rows,
            complete=complete or not self._has_disk_prefix or self.writer is None,
            start_row=first[0] if first else self._next_row,
            start_step=first[1] if first else None,
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
        numbers shift once records are dropped, so that case falls back to steps.
        """
        with self._lock:
            self._disk_queries += 1
        self.flush()
        assert self.writer is not None
        reader = self.writer.reader()
        if self.writer.dropped:
            if view.start_step is None:
                return reader, reader.size
            return reader, reader.offset_of_step(view.start_step)
        return reader, reader.offset_of_line(view.start_row)

    def _collect_tail(self, steps: int | None) -> list[dict]:
        """Rows covering the newest ``steps`` steps (``None`` for the whole run)."""
        if steps == 0:
            return []
        view = self._view_tail(steps)
        records = view.records()
        if view.complete:
            return records
        return self._older(lambda: self._older_tail(view, steps, records)) + records

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
        self, view: _CacheView, steps: int | None, newer: list[dict]
    ) -> list[dict]:
        """The rows before the cache that are still needed to reach ``steps`` steps."""
        reader, end = self._disk_reader(view)
        if steps is None:
            if view.start_row > BIG_READ_WARN_ROWS:
                logger.warning(
                    "Reading the full history from disk; this may take a while."
                )
            return reader.parse_rows(reader.iter_raw(0, end))
        # Widen until one whole step past the limit is in hand, so the oldest
        # returned step cannot be cut in half by the read window
        want = max(steps - len(newer), 1)
        older: list[dict] = []
        for _ in range(12):
            older = reader.tail_rows(want, end)
            if len(older) < want or _count_steps(older + newer) > steps:
                break
            want *= 2
        return older

    def _collect_range(self, step_range: tuple[int | None, int | None]) -> list[dict]:
        """Rows whose step falls inside ``step_range``."""
        view = self._view_range(step_range)
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


def _count_steps(records: Sequence[dict]) -> int:
    return len({record.get("_step") for record in records})


def _scan_tail(
    cache: Sequence[tuple[int, int, bytes]], steps: int
) -> tuple[list[tuple[int, int, bytes]], bool]:
    """Cache entries covering the newest ``steps`` steps, newest last.

    Entries carry their step, so the walk stops as soon as one older step shows up:
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
    seen: list = []
    unique: set = set()
    for record in reversed(records):
        step = record.get("_step")
        if step not in unique:
            unique.add(step)
            seen.append(step)
            if len(seen) > limit:
                break
    return set(seen[:limit])


def _in_range(step, step_range: tuple[int | None, int | None]) -> bool:
    start, end = step_range
    if not isinstance(step, int):
        return False
    return (start is None or step >= start) and (end is None or step < end)
