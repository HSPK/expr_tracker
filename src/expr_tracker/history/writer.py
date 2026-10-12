"""Append-only JSONL writer: adaptive buffering, sparse index, meta sidecar.

The writer only appends already-encoded lines durably; step assembly lives in
:mod:`.store`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

from loguru import logger

from .._process import check_process, process_id, register_fork_reset
from .reader import JsonlReader
from .rotation import (
    DEFAULT_MAX_FILE_BYTES,
    segment_path,
    segments,
    validate_file_limit,
)

DEFAULT_BUFFER_SIZE = 50
DEFAULT_BUFFER_INTERVAL = 1.0
DEFAULT_MAX_BUFFER_SECONDS = 5.0
# Cap for slow or failing writes; overflow is dropped only after write failures
DEFAULT_MAX_PENDING_RECORDS = 100_000
DEFAULT_MAX_PENDING_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_RECORD_BYTES = 8 * 1024 * 1024
DEFAULT_INDEX_EVERY = 1000
# Index resolution halves past this, so memory and the sidecar stay bounded
MAX_INDEX_ANCHORS = 4096
TAIL_CHUNK_SIZE = 64 * 1024
META_MIN_INTERVAL = 2.0
MIN_RETRY_INTERVAL = 0.1
SCHEMA_VERSION = 2


_diagnostic_emission = threading.local()


def in_diagnostic_callback() -> bool:
    return getattr(_diagnostic_emission, "active", False)


def _reset_diagnostic_context():
    global _diagnostic_emission
    _diagnostic_emission = threading.local()


register_fork_reset(_reset_diagnostic_context)


def _emit_diagnostic(level: str, message: str):
    """Keep sink failures nonfatal without recursively entering Loguru's sink lock."""
    operation_error = sys.exc_info()[1]
    emitting = in_diagnostic_callback()
    try:
        if emitting:
            os.write(
                2, f"{level.upper()}: {message}\n".encode("utf-8", "backslashreplace")
            )
            return
        _diagnostic_emission.active = True
        try:
            getattr(logger, level)(message)
        except Exception as error:
            os.write(
                2,
                (
                    f"{level.upper()}: {message}\n"
                    f"Writer diagnostic sink failed: {type(error).__name__}: {error}\n"
                ).encode("utf-8", "backslashreplace"),
            )
    except OSError:
        # Even an unavailable stderr must not replace an operation's own exception.
        if operation_error is not None:
            raise operation_error from None
        raise
    finally:
        _diagnostic_emission.active = emitting


class _Diagnostics(threading.local):
    """One lazy, bounded-by-operation batch per thread, shared by nested methods."""

    def __init__(self):
        self.depth = 0
        self.messages: list[tuple[str, str]] | None = None

    def __enter__(self):
        self.depth += 1

    def __exit__(self, exc_type, exc_value, traceback):
        self.depth -= 1
        if self.depth == 0 and self.messages is not None:
            messages, self.messages = self.messages, None
            for level, message in messages:
                _emit_diagnostic(level, message)

    def add(self, level: str, message: str):
        if self.depth:
            if self.messages is None:
                self.messages = []
            self.messages.append((level, message))
        else:
            _emit_diagnostic(level, message)


def validate_byte_limits(
    max_pending_bytes: int | None, max_record_bytes: int | None
) -> None:
    for name, value in (
        ("max_pending_bytes", max_pending_bytes),
        ("max_record_bytes", max_record_bytes),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer or None")


class JsonlWriter:
    """Buffered appender whose flush policy adapts to the append frequency.

    Writers belong to their creating process and cannot be reused after fork.

    - ``buffer_size``: flush as soon as this many records are buffered.
    - ``buffer_interval``: a gap of at least this long means low-frequency writing,
      so the record is written straight through; ``None`` disables the check.
    - ``max_buffer_seconds``: longest a record may sit in memory before a background
      worker flushes it; ``None`` disables the deadline.
    - ``max_pending_records``: cap on queued and in-flight records. Healthy writers
      wait for capacity; after write failures, overflow drops the oldest queued
      records instead (in-flight writes cannot be evicted).
    - ``max_pending_bytes``: cap on queued plus in-flight encoded line bytes.
      ``None`` disables it. Batch assembly can temporarily use up to this much
      additional memory; caller-owned lines waiting for admission are not counted.
    - ``max_record_bytes``: reject larger encoded lines, logging and counting them
      as dropped. ``None`` disables this limit, but a line must still fit within
      ``max_pending_bytes``. Both byte limits must be positive integers or ``None``.
    - ``async_write``: enqueue without producer disk I/O; ``flush`` and ``close``
      still wait for accepted writes. The default preserves synchronous flushing.
    - ``max_file_bytes``: rotate before a record would exceed this file size.
      All segments are retained; a larger record occupies a segment by itself.
      ``None`` disables rotation.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        buffer_interval: float | None = DEFAULT_BUFFER_INTERVAL,
        max_buffer_seconds: float | None = DEFAULT_MAX_BUFFER_SECONDS,
        max_pending_records: int = DEFAULT_MAX_PENDING_RECORDS,
        max_pending_bytes: int | None = DEFAULT_MAX_PENDING_BYTES,
        max_record_bytes: int | None = DEFAULT_MAX_RECORD_BYTES,
        max_file_bytes: int | None = DEFAULT_MAX_FILE_BYTES,
        index_every: int = DEFAULT_INDEX_EVERY,
        meta_extra: dict | None = None,
        async_write: bool = False,
    ):
        self._owner_pid = process_id()
        validate_byte_limits(max_pending_bytes, max_record_bytes)
        validate_file_limit(max_file_bytes)
        self.path = Path(path)
        self.max_file_bytes = max_file_bytes
        active = segments(self.path)[-1]
        self._active_number = 0 if active == self.path else int(active.stem)
        self._active_size = 0
        self.meta_path = self.path.with_name(self.path.stem + ".meta.json")
        self.buffer_size = max(1, int(buffer_size))
        self.buffer_interval = (
            None if buffer_interval is None else float(buffer_interval)
        )
        self.max_buffer_seconds = (
            None if max_buffer_seconds is None else float(max_buffer_seconds)
        )
        self.max_pending_records = max(self.buffer_size, int(max_pending_records or 0))
        self.max_pending_bytes = max_pending_bytes
        self.max_record_bytes = max_record_bytes
        self.index_every = max(1, int(index_every))
        self.meta_extra = dict(meta_extra or {})
        self.async_write = async_write

        self._buffer: deque[tuple[int, int, bytes]] = deque()  # (step, row, line)
        self._queued_bytes = 0
        self._inflight_bytes = 0
        self.index: list[tuple[int, int, int]] = []  # (step, line, byte offset)
        self.lines = 0
        self.size = 0
        self.last_step: int | None = None
        # Highest step anywhere in the file. With patch lines or step_policy="allow"
        # the last line is not the highest one, and resume must not reuse a step.
        self.max_step: int | None = None
        # Highest row ordinal known to be durable (-1 when nothing is written yet)
        self.flushed_row = -1
        self.sorted = True
        self.has_duplicate_steps = False
        self.dropped = 0
        self.dropped_bytes = 0

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._write_lock = threading.Lock()
        self._diagnostics = _Diagnostics()
        self._last_append_time: float | None = None
        self._first_buffered_time: float | None = None
        self._worker: threading.Thread | None = None
        self._flush_deadline: float | None = None
        self._flush_requested = False
        self._retry_deadline: float | None = None
        self._inflight_records = 0
        self._closing = False
        self._closed = False
        self._next_index_line = 0
        self._last_meta_time = 0.0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._diagnostics:
            self._load()

    # ------------------------------------------------------------------ open/resume

    @property
    def _active_path(self) -> Path:
        return (
            self.path
            if self._active_number == 0
            else segment_path(self.path, self._active_number)
        )

    def _load(self):
        """Restore state from an existing file, repairing a torn tail and stale meta."""
        if not self.path.exists():
            return
        self._repair_torn_tail()
        self._active_size = self._active_path.stat().st_size
        size = JsonlReader(self.path).size
        if not self._restore_from_meta(size):
            self._rebuild_from_disk(size)
        self.size = size
        self._next_index_line = self._next_anchor_line()
        # Rows already on disk are durable; ordinals continue from the line count
        self.flushed_row = self.lines - 1

    def _repair_torn_tail(self):
        """Drop a partially written final line so appends cannot glue onto it."""
        size = self._active_path.stat().st_size
        if size == 0:
            return
        with open(self._active_path, "rb") as f:
            f.seek(size - 1)
            if f.read(1) == b"\n":
                return
            cut, pos = 0, size
            while pos > 0:
                read = min(TAIL_CHUNK_SIZE, pos)
                pos -= read
                f.seek(pos)
                index = f.read(read).rfind(b"\n")
                if index >= 0:
                    cut = pos + index + 1
                    break
        try:
            os.truncate(self._active_path, cut)
        except Exception as e:
            self._diagnostics.add(
                "error", f"Could not truncate torn tail of {self.path}: {e}"
            )
            return
        self._diagnostics.add(
            "warning",
            f"Dropped {size - cut} bytes of a torn final line in {self.path}.",
        )

    def _restore_from_meta(self, size: int) -> bool:
        """Load the sidecar. Returns False when it is missing or out of date."""
        meta = self._read_meta()
        if not meta:
            return False
        self.index = [tuple(e) for e in meta.get("index", [])]
        self.sorted = bool(meta.get("sorted", True))
        self.has_duplicate_steps = bool(meta.get("has_duplicate_steps", False))
        self.lines = int(meta.get("lines", 0))
        self.last_step = meta.get("last_step")
        self.max_step = meta.get("max_step", self.last_step)
        return int(meta.get("size", -1)) == size

    def _rebuild_from_disk(self, size: int):
        """Rescan from the last anchor that still lies inside the file."""
        # Unsorted prefixes can contain a maximum between anchors. Rescan them
        # fully; after truncation even the old metadata's maximum may be gone.
        usable = (
            [entry for entry in self.index if entry[2] < size] if self.sorted else []
        )
        anchor_step, start_line, start_offset = usable[-1] if usable else (None, 0, 0)
        self.index = usable[:-1]  # the last anchor is re-added while rescanning
        # No predecessor for the first rescanned line: the anchor line is read again,
        # and seeding last_step with it would look like a repeated step
        self.last_step = None
        # Anchors cover the region before the rescan, so their steps bound the max
        self.max_step = max((entry[0] for entry in self.index), default=None)
        self._rescan(start_offset, start_line)
        if self.last_step is None:  # nothing readable past the anchor
            self.last_step = anchor_step
        if anchor_step is not None:
            known = self.max_step
            self.max_step = anchor_step if known is None else max(known, anchor_step)

    def _rescan(self, offset: int, line_no: int):
        """Fold every line from ``offset`` onwards back into the counters and index."""
        self.lines = self._next_index_line = line_no
        for position, raw in JsonlReader(self.path)._snapshot()._scan(offset):
            if raw.strip():
                self._track_line(_peek_step(raw), position)

    def _track_line(self, step: int | None, offset: int):
        """Account for one line that is already on disk."""
        if step is not None:
            self._track_step_order(step)
            self._track_anchor(step, offset)
            self.last_step = step
            self.max_step = step if self.max_step is None else max(self.max_step, step)
        self.lines += 1

    def _track_step_order(self, step: int):
        """Track whether the file is still ascending and free of repeated steps."""
        if self.last_step is None:
            return
        if step < self.last_step:
            self.sorted = False
        elif step == self.last_step:
            self.has_duplicate_steps = True

    def _track_anchor(self, step: int, offset: int):
        """Sample an index anchor when one is due, halving resolution if too many."""
        if self.lines < self._next_index_line:
            return
        self.index.append((step, self.lines, offset))
        self._next_index_line = self.lines + self.index_every
        if len(self.index) > MAX_INDEX_ANCHORS:
            self._halve_index()

    def _next_anchor_line(self) -> int:
        if not self.index:
            return 0
        return self.index[-1][1] + self.index_every

    def _read_meta(self) -> dict | None:
        try:
            with open(self.meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            return meta if meta.get("schema") == SCHEMA_VERSION else None
        except Exception:
            return None

    # ------------------------------------------------------------------ writing

    @property
    def buffer(self) -> list[tuple[int, int, bytes]]:
        """Snapshot of queued records, in order; mutating it does not edit the queue."""
        check_process(self._owner_pid)
        with self._lock:
            return list(self._buffer)

    @property
    def queued_bytes(self) -> int:
        """Encoded bytes waiting for a write, including requeued failures."""
        check_process(self._owner_pid)
        with self._lock:
            return self._queued_bytes

    @property
    def inflight_bytes(self) -> int:
        """Encoded bytes in the current write, still charged against capacity."""
        check_process(self._owner_pid)
        with self._lock:
            return self._inflight_bytes

    @property
    def pending_bytes(self) -> int:
        """Atomic sum of queued and in-flight encoded bytes."""
        check_process(self._owner_pid)
        with self._lock:
            return self._queued_bytes + self._inflight_bytes

    def enqueue(self, step: int, row: int, line: bytes) -> bool:
        """Buffer an encoded line under the lock; returns whether a flush is due.

        Callers that allocate ``row`` themselves must call this while holding their
        own lock, so ordinal assignment and buffer order can never diverge.
        Async writers also wake their worker here. False means no flush is due,
        not an admission acknowledgement; closed writers log and return False.
        At capacity, healthy writers wait without doing producer disk I/O. Closing
        rejects admissions still waiting for capacity. Use ``enqueue_with_status``
        when the caller also needs to know whether the record was accepted.
        """
        check_process(self._owner_pid)
        return self.enqueue_with_status(step, row, line)[1]

    def enqueue_with_status(
        self, step: int, row: int, line: bytes
    ) -> tuple[bool, bool]:
        """Return ``(accepted, flush_due)``; acceptance does not imply durability.

        Closed, oversized, or failure-overflowed new records return ``(False,
        False)``. Accepted records can still be evicted after a later write failure.
        Row allocation must be serialized by the caller, as with ``enqueue``.
        """
        check_process(self._owner_pid)
        line_bytes = len(line)
        diagnostic = None
        try:
            with self._condition:
                if self._closing:
                    diagnostic = (
                        "warning",
                        f"Ignoring write to closing or closed writer {self.path}.",
                    )
                    return False, False
                if (
                    self.max_record_bytes is not None
                    and line_bytes > self.max_record_bytes
                ) or (
                    self.max_pending_bytes is not None
                    and line_bytes > self.max_pending_bytes
                ):
                    self.dropped += 1
                    self.dropped_bytes += line_bytes
                    diagnostic = (
                        "error",
                        f"Rejected oversized record for {self.path}: "
                        f"{line_bytes} bytes "
                        f"(max_record_bytes={self.max_record_bytes}, "
                        f"max_pending_bytes={self.max_pending_bytes}); "
                        "dropped 1 record.",
                    )
                    return False, False
                while (
                    not self._closing
                    and self._retry_deadline is None
                    and not self._has_capacity(line_bytes)
                ):
                    # Capacity pressure must progress even without deadlines or an
                    # explicit flush from an enqueue-only caller.
                    if not self._flush_requested:
                        self._schedule_flush(immediate=True)
                    self._condition.wait()
                if self._closing:
                    diagnostic = (
                        "warning",
                        f"Ignoring write to closing or closed writer {self.path}.",
                    )
                    return False, False
                if not self._has_capacity(line_bytes):
                    accepted, message = self._cap_buffer(line_bytes)
                    diagnostic = ("error", message)
                    if not accepted:
                        return False, False
                now = time.monotonic()
                self._buffer.append((step, row, line))
                self._queued_bytes += line_bytes
                if self._first_buffered_time is None:
                    self._first_buffered_time = now
                interval = (
                    None
                    if self._last_append_time is None
                    else now - self._last_append_time
                )
                self._last_append_time = now
                should_flush = self._should_flush(now, interval)
                if self.async_write:
                    self._schedule_flush(immediate=should_flush)
                return True, should_flush
        finally:
            if diagnostic is not None:
                self._diagnostics.add(*diagnostic)

    def append(self, step: int, row: int, line: bytes):
        """Buffer an encoded line, flushing if needed. ``row`` is the append ordinal."""
        self.append_with_status(step, row, line)

    def append_with_status(self, step: int, row: int, line: bytes) -> bool:
        """Append with the usual flush policy, returning whether admission succeeded."""
        accepted, should_flush = self.enqueue_with_status(step, row, line)
        if accepted and not self.async_write:
            if should_flush:
                self.flush()
            else:
                self.schedule_timer()
        return accepted

    def _should_flush(self, now: float, interval: float | None) -> bool:
        if len(self._buffer) >= self.buffer_size:
            return True
        # First append: create file content as early as possible
        if interval is None:
            return True
        if self.buffer_interval is not None and interval >= self.buffer_interval:
            return True
        return (
            self.max_buffer_seconds is not None
            and self._first_buffered_time is not None
            and now - self._first_buffered_time >= self.max_buffer_seconds
        )

    def schedule_timer(self):
        """Arm the reusable worker's deadline without creating a timer per batch."""
        check_process(self._owner_pid)
        with self._lock:
            self._schedule_flush()

    def _schedule_flush(self, *, immediate: bool = False):
        """Schedule work while holding the buffer lock."""
        if self._closing or not self._buffer:
            return
        self._flush_requested |= immediate
        if self.max_buffer_seconds is not None and self._flush_deadline is None:
            first = self._first_buffered_time
            if first is None:
                first = time.monotonic()
            self._flush_deadline = first + max(0.0, self.max_buffer_seconds)
        if not self._flush_requested and self._flush_deadline is None:
            return
        if self._worker is None:
            self._worker = threading.Thread(
                target=self._run_worker,
                name=f"expr-tracker-writer-{self.path.name}",
                daemon=True,
            )
            self._worker.start()
        self._condition.notify_all()

    def _run_worker(self):
        while True:
            with self._condition:
                while True:
                    if self._closing:
                        return
                    now = time.monotonic()
                    deadline = now if self._flush_requested else self._flush_deadline
                    if deadline is None:
                        self._condition.wait()
                        continue
                    if self._retry_deadline is not None:
                        deadline = max(deadline, self._retry_deadline)
                    delay = deadline - now
                    if delay <= 0:
                        break
                    self._condition.wait(delay)
            self.flush()

    def flush(self):
        """Wait for earlier writes and drain the buffer; log and requeue failures."""
        check_process(self._owner_pid)
        # Swap and write under one lock: two concurrent flushes would otherwise be
        # able to invert whole batches on disk and break row-ordinal addressing.
        with self._diagnostics, self._write_lock:
            with self._lock:
                if self._closed:
                    return
                self._flush_requested = False
                self._flush_deadline = None
                self._first_buffered_time = None
                if not self._buffer:
                    return
                records, self._buffer = self._buffer, deque()
                self._inflight_records = len(records)
                self._inflight_bytes, self._queued_bytes = self._queued_bytes, 0

            payload = b"".join(line for _, _, line in records)
            # Use the tracked offset instead of stat(): stat costs ms on network mounts
            size_before = self.size
            failed = False
            try:
                self._write(payload)
            except Exception as e:
                failed = True
                self._diagnostics.add(
                    "error", f"Failed to flush metrics to {self.path}: {e}"
                )
                if not self._truncate_partial_write(size_before):
                    # Rollback failed, so a retry may duplicate lines; force the
                    # reader into merge mode rather than dropping data
                    with self._lock:
                        self.has_duplicate_steps = True
            # Release the joined copy (and any exception traceback) before capacity
            # is made available to producers, including during slow metadata writes.
            del payload
            if failed:
                with self._lock:
                    self._requeue(records)
                    retry_interval = (
                        DEFAULT_BUFFER_INTERVAL
                        if self.max_buffer_seconds is None
                        else self.max_buffer_seconds
                    )
                    self._retry_deadline = time.monotonic() + max(
                        MIN_RETRY_INTERVAL, retry_interval
                    )
                    self._schedule_flush(immediate=self.async_write)
                    self._condition.notify_all()
                return
            self._track_flushed(records, size_before)
            # Serialize sidecar replacement as well as data writes: both use one
            # temporary path, and an older snapshot must never replace a newer one.
            self._save_meta()

    def _write(self, payload: bytes):
        """Split a batch at record boundaries, without renaming pinned segments."""
        if (
            self.max_file_bytes is None
            or self._active_size + len(payload) <= self.max_file_bytes
        ):
            self._append(payload)
            self._active_size += len(payload)
            return
        start = 0
        while start < len(payload):
            end = start
            while end < len(payload):
                next_end = payload.find(b"\n", end) + 1
                if next_end == 0:
                    next_end = len(payload)
                if self._active_size + next_end - start > self.max_file_bytes and (
                    self._active_size or end > start
                ):
                    break
                end = next_end
            if end == start:
                self._active_number += 1
                self._active_size = 0
                continue
            self._append(payload[start:end])
            self._active_size += end - start
            start = end

    def _append(self, payload: bytes):
        """Append, recreating the parent directory once if it disappeared."""
        try:
            with open(self._active_path, "ab") as f:
                f.write(payload)
        except FileNotFoundError:
            self._active_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._active_path, "ab") as f:
                f.write(payload)

    def _track_flushed(self, records: deque[tuple[int, int, bytes]], offset: int):
        """Account for a batch that just reached the file."""
        with self._lock:
            for step, row, line in records:
                self._track_line(step, offset)
                self.flushed_row = max(self.flushed_row, row)
                offset += len(line)
            self.size = offset
            self.flushed_step = self.last_step
            records.clear()
            self._inflight_records = 0
            self._inflight_bytes = 0
            self._retry_deadline = None
            self._condition.notify_all()

    def _halve_index(self):
        """Halve the index resolution instead of letting it grow unbounded."""
        self.index = self.index[::2]
        self.index_every *= 2
        self._next_index_line = self.index[-1][1] + self.index_every

    def _truncate_partial_write(self, size_before: int) -> bool:
        """Roll back a partial write so retries cannot duplicate lines."""
        try:
            remaining = size_before
            for path in segments(self.path):
                if not path.exists():
                    continue
                length = path.stat().st_size
                if remaining >= length:
                    remaining -= length
                elif remaining or path == self.path:
                    os.truncate(path, remaining)
                    remaining = 0
                else:
                    path.unlink()
            active = segments(self.path)[-1]
            self._active_number = 0 if active == self.path else int(active.stem)
            self._active_size = (
                self._active_path.stat().st_size if self._active_path.exists() else 0
            )
            return True
        except Exception as e:
            self._diagnostics.add(
                "warning", f"Could not roll back partial write on {self.path}: {e}"
            )
            return False

    def _requeue(self, records: deque[tuple[int, int, bytes]]):
        with self._lock:
            records.extend(self._buffer)
            self._buffer = records
            self._queued_bytes += self._inflight_bytes
            self._inflight_records = 0
            self._inflight_bytes = 0
            self._first_buffered_time = time.monotonic()

    def _has_capacity(self, incoming_bytes: int) -> bool:
        return len(
            self._buffer
        ) + self._inflight_records < self.max_pending_records and (
            self.max_pending_bytes is None
            or self._queued_bytes + self._inflight_bytes + incoming_bytes
            <= self.max_pending_bytes
        )

    def _cap_buffer(self, incoming_bytes: int) -> tuple[bool, str]:
        """After write failure, evict oldest queued records to fit one new line."""
        dropped = dropped_bytes = 0
        while self._buffer and not self._has_capacity(incoming_bytes):
            _, _, line = self._buffer.popleft()
            size = len(line)
            self._queued_bytes -= size
            dropped += 1
            dropped_bytes += size
        accepted = self._has_capacity(incoming_bytes)
        if not accepted:  # In-flight records cannot be evicted; drop the new line.
            dropped += 1
            dropped_bytes += incoming_bytes
        self.dropped += dropped
        self.dropped_bytes += dropped_bytes
        message = (
            f"Metrics buffer exceeded {self.max_pending_records} records or "
            f"max_pending_bytes={self.max_pending_bytes}, "
            f"dropped {dropped} oldest queued records ({dropped_bytes} bytes)."
        )
        return accepted, message

    # ------------------------------------------------------------------ meta

    def _save_meta(self, force: bool = False):
        now = time.monotonic()
        if not force and now - self._last_meta_time < META_MIN_INTERVAL:
            return
        self._last_meta_time = now
        with self._lock:
            meta = {
                "schema": SCHEMA_VERSION,
                "size": self.size,
                "lines": self.lines,
                "last_step": self.last_step,
                "max_step": self.max_step,
                "sorted": self.sorted,
                "has_duplicate_steps": self.has_duplicate_steps,
                "index": [list(e) for e in self.index],
                "run": self.meta_extra,
            }
        tmp = self.meta_path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8", errors="backslashreplace") as f:
                json.dump(meta, f, ensure_ascii=False)
            os.replace(tmp, self.meta_path)
        except Exception as e:
            self._diagnostics.add("warning", f"Failed to write {self.meta_path}: {e}")

    def reader(self) -> JsonlReader:
        """Return a reader sharing this writer's in-memory index."""
        check_process(self._owner_pid)
        return JsonlReader(
            self.path,
            index=self.index,
            merge=self.has_duplicate_steps or not self.sorted,
        )

    def close(self):
        """Reject new writes, join the worker, and attempt one final drain."""
        check_process(self._owner_pid)
        if self._worker is threading.current_thread():
            raise RuntimeError("Cannot close a writer from its own background worker.")
        with self._diagnostics:
            with self._condition:
                if (
                    in_diagnostic_callback()
                    and self._worker is not None
                    and self._worker.is_alive()
                ):
                    raise RuntimeError(
                        "Cannot close a writer from a diagnostic sink while its "
                        "background worker is active."
                    )
                if self._closing:
                    self._condition.wait_for(lambda: self._closed)
                    return
                self._closing = True
                self._flush_requested = False
                self._flush_deadline = None
                worker = self._worker
                self._condition.notify_all()
            try:
                # No writer locks may be held while waiting for an in-flight flush.
                if worker is not None:
                    worker.join()
                self.flush()
            finally:
                with self._write_lock:
                    try:
                        self._save_meta(force=True)
                        with self._lock:
                            if self._buffer:
                                self._diagnostics.add(
                                    "error",
                                    f"Closing {self.path} with {len(self._buffer)} "
                                    "unwritten records after a failed final flush "
                                    f"({self._queued_bytes} bytes).",
                                )
                    finally:
                        with self._condition:
                            self._closed = True
                            self._condition.notify_all()


def _peek_step(raw: bytes) -> int | None:
    try:
        value = json.loads(raw)
    except Exception:
        return None
    step = value.get("_step") if isinstance(value, dict) else None
    return step if isinstance(step, int) else None
