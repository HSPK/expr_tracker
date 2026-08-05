"""Append-only JSONL writer: adaptive buffering, sparse index, meta sidecar.

The writer only appends already-encoded lines durably; step assembly lives in
:mod:`.store`.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from loguru import logger

from .reader import JsonlReader

DEFAULT_BUFFER_SIZE = 50
DEFAULT_BUFFER_INTERVAL = 1.0
DEFAULT_MAX_BUFFER_SECONDS = 5.0
# Buffer cap while writes keep failing; oldest records are dropped beyond it
DEFAULT_MAX_PENDING_RECORDS = 100_000
DEFAULT_INDEX_EVERY = 1000
# Index resolution halves past this, so memory and the sidecar stay bounded
MAX_INDEX_ANCHORS = 4096
TAIL_CHUNK_SIZE = 64 * 1024
META_MIN_INTERVAL = 2.0
SCHEMA_VERSION = 2


class JsonlWriter:
    """Buffered appender whose flush policy adapts to the append frequency.

    - ``buffer_size``: flush as soon as this many records are buffered.
    - ``buffer_interval``: a gap of at least this long means low-frequency writing,
      so the record is written straight through; ``None`` disables the check.
    - ``max_buffer_seconds``: longest a record may sit in memory before a background
      timer flushes it; ``None`` disables the timer.
    - ``max_pending_records``: buffer cap while writes keep failing.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        buffer_interval: float | None = DEFAULT_BUFFER_INTERVAL,
        max_buffer_seconds: float | None = DEFAULT_MAX_BUFFER_SECONDS,
        max_pending_records: int = DEFAULT_MAX_PENDING_RECORDS,
        index_every: int = DEFAULT_INDEX_EVERY,
        meta_extra: dict | None = None,
    ):
        self.path = Path(path)
        self.meta_path = self.path.with_name(self.path.stem + ".meta.json")
        self.buffer_size = max(1, int(buffer_size))
        self.buffer_interval = (
            None if buffer_interval is None else float(buffer_interval)
        )
        self.max_buffer_seconds = (
            None if max_buffer_seconds is None else float(max_buffer_seconds)
        )
        self.max_pending_records = max(self.buffer_size, int(max_pending_records or 0))
        self.index_every = max(1, int(index_every))
        self.meta_extra = dict(meta_extra or {})

        self.buffer: list[tuple[int, int, bytes]] = []  # (step, row, line)
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

        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._last_append_time: float | None = None
        self._first_buffered_time: float | None = None
        self._flush_timer: threading.Timer | None = None
        self._next_index_line = 0
        self._last_meta_time = 0.0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------ open/resume

    def _load(self):
        """Restore state from an existing file, repairing a torn tail and stale meta."""
        if not self.path.exists():
            return
        self._repair_torn_tail()
        size = self.path.stat().st_size
        if not self._restore_from_meta(size):
            self._rebuild_from_disk(size)
        self.size = size
        self._next_index_line = self._next_anchor_line()
        # Rows already on disk are durable; ordinals continue from the line count
        self.flushed_row = self.lines - 1

    def _repair_torn_tail(self):
        """Drop a partially written final line so appends cannot glue onto it."""
        size = self.path.stat().st_size
        if size == 0:
            return
        with open(self.path, "rb") as f:
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
            os.truncate(self.path, cut)
        except Exception as e:
            logger.error(f"Could not truncate torn tail of {self.path}: {e}")
            return
        logger.warning(
            f"Dropped {size - cut} bytes of a torn final line in {self.path}."
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
        usable = [entry for entry in self.index if entry[2] < size]
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
        with open(self.path, "rb") as f:
            f.seek(offset)
            for raw in f:
                if raw.strip():
                    self._track_line(_peek_step(raw), offset)
                offset += len(raw)

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

    def enqueue(self, step: int, row: int, line: bytes) -> bool:
        """Buffer an encoded line under the lock; returns whether a flush is due.

        Callers that allocate ``row`` themselves must call this while holding their
        own lock, so ordinal assignment and buffer order can never diverge.
        """
        now = time.monotonic()
        with self._lock:
            self.buffer.append((step, row, line))
            if self._first_buffered_time is None:
                self._first_buffered_time = now
            interval = (
                None if self._last_append_time is None else now - self._last_append_time
            )
            self._last_append_time = now
            return self._should_flush(now, interval)

    def append(self, step: int, row: int, line: bytes):
        """Buffer an encoded line, flushing if needed. ``row`` is the append ordinal."""
        if self.enqueue(step, row, line):
            self.flush()
        else:
            self.schedule_timer()

    def _should_flush(self, now: float, interval: float | None) -> bool:
        if len(self.buffer) >= self.buffer_size:
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
        if self.max_buffer_seconds is None:
            return
        with self._lock:
            if self._flush_timer is not None or not self.buffer:
                return
            now = time.monotonic()
            elapsed = now - (self._first_buffered_time or now)
            timer = threading.Timer(
                max(0.0, self.max_buffer_seconds - elapsed), self._on_timer
            )
            timer.daemon = True
            self._flush_timer = timer
        timer.start()

    def _on_timer(self):
        with self._lock:
            self._flush_timer = None
        self.flush()

    def _cancel_timer(self):
        with self._lock:
            timer, self._flush_timer = self._flush_timer, None
        if timer is not None:
            timer.cancel()

    def flush(self):
        """Write the buffer to disk; a failed batch is rolled back and requeued."""
        self._cancel_timer()
        # Swap and write under one lock: two concurrent flushes would otherwise be
        # able to invert whole batches on disk and break row-ordinal addressing.
        with self._write_lock:
            with self._lock:
                self._first_buffered_time = None
                if not self.buffer:
                    return
                records, self.buffer = self.buffer, []

            payload = b"".join(line for _, _, line in records)
            # Use the tracked offset instead of stat(): stat costs ms on network mounts
            size_before = self.size
            try:
                self._write(payload)
            except Exception as e:
                logger.error(f"Failed to flush metrics to {self.path}: {e}")
                if not self._truncate_partial_write(size_before):
                    # Rollback failed, so a retry may duplicate lines; force the
                    # reader into merge mode rather than dropping data
                    self.has_duplicate_steps = True
                self._requeue(records)
                self.schedule_timer()
                return
            self._track_flushed(records, size_before)
        self._save_meta()

    def _write(self, payload: bytes):
        """Append, recreating the parent directory once if it disappeared."""
        try:
            with open(self.path, "ab") as f:
                f.write(payload)
        except FileNotFoundError:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "ab") as f:
                f.write(payload)

    def _track_flushed(self, records: list[tuple[int, int, bytes]], offset: int):
        """Account for a batch that just reached the file."""
        with self._lock:
            for step, row, line in records:
                self._track_line(step, offset)
                self.flushed_row = max(self.flushed_row, row)
                offset += len(line)
            self.size = offset
            self.flushed_step = self.last_step

    def _halve_index(self):
        """Halve the index resolution instead of letting it grow unbounded."""
        self.index = self.index[::2]
        self.index_every *= 2
        self._next_index_line = self.index[-1][1] + self.index_every

    def _truncate_partial_write(self, size_before: int) -> bool:
        """Roll back a partial write so retries cannot duplicate lines."""
        try:
            if not self.path.is_file() or self.path.stat().st_size <= size_before:
                return True  # nothing reached the file
            os.truncate(self.path, size_before)
            return True
        except Exception as e:
            logger.warning(f"Could not roll back partial write on {self.path}: {e}")
            return False

    def _requeue(self, records: list[tuple[int, int, bytes]]):
        with self._lock:
            self.buffer[:0] = records
            overflow = len(self.buffer) - self.max_pending_records
            if overflow > 0:
                del self.buffer[:overflow]
                self.dropped += overflow
                logger.error(
                    f"Metrics buffer exceeded {self.max_pending_records} records, "
                    f"dropped {overflow} oldest records."
                )
            self._first_buffered_time = time.monotonic()

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
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
            os.replace(tmp, self.meta_path)
        except Exception as e:
            logger.warning(f"Failed to write {self.meta_path}: {e}")

    def reader(self) -> JsonlReader:
        """Return a reader sharing this writer's in-memory index."""
        return JsonlReader(
            self.path,
            index=self.index,
            merge=self.has_duplicate_steps or not self.sorted,
        )

    def close(self):
        self._cancel_timer()
        self.flush()
        self._save_meta(force=True)


def _peek_step(raw: bytes) -> int | None:
    try:
        value = json.loads(raw)
    except Exception:
        return None
    step = value.get("_step") if isinstance(value, dict) else None
    return step if isinstance(step, int) else None
