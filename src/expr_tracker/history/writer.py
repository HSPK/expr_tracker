"""SDK diagnostics/thread trampolines for the Rust JSONL writer."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from loguru import logger

from .._native import NativeWriter
from .._process import check_process, process_id, register_fork_reset
from .codec import encode_line
from .reader import JsonlReader
from .rotation import DEFAULT_MAX_FILE_BYTES, validate_file_limit

DEFAULT_BUFFER_SIZE = 50
DEFAULT_BUFFER_INTERVAL = 1.0
DEFAULT_MAX_BUFFER_SECONDS = 5.0
DEFAULT_MAX_PENDING_RECORDS = 100_000
DEFAULT_MAX_PENDING_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_RECORD_BYTES = 8 * 1024 * 1024
DEFAULT_INDEX_EVERY = 1000
MAX_INDEX_ANCHORS = 4096
TAIL_CHUNK_SIZE = 64 * 1024
META_MIN_INTERVAL = 2.0
MIN_RETRY_INTERVAL = 0.1
SCHEMA_VERSION = 2

_diagnostic_emission = threading.local()


def in_diagnostic_callback():
    return getattr(_diagnostic_emission, "active", False)


def _reset_diagnostic_context():
    global _diagnostic_emission
    _diagnostic_emission = threading.local()


register_fork_reset(_reset_diagnostic_context)


def _emit_diagnostic(level, message):
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
        if operation_error is not None:
            raise operation_error from None
        raise
    finally:
        _diagnostic_emission.active = emitting


class _Diagnostics(threading.local):
    def __init__(self):
        self.depth = 0
        self.messages = None

    def __enter__(self):
        self.depth += 1

    def __exit__(self, exc_type, exc_value, traceback):
        self.depth -= 1
        if self.depth == 0 and self.messages is not None:
            messages, self.messages = self.messages, None
            for level, message in messages:
                _emit_diagnostic(level, message)

    def add(self, level, message):
        if self.depth:
            if self.messages is None:
                self.messages = []
            self.messages.append((level, message))
        else:
            _emit_diagnostic(level, message)


def validate_byte_limits(max_pending_bytes, max_record_bytes):
    for name, value in (
        ("max_pending_bytes", max_pending_bytes),
        ("max_record_bytes", max_record_bytes),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer or None")


_ATTRIBUTES = frozenset(
    {
        "path",
        "meta_path",
        "buffer_size",
        "buffer_interval",
        "max_buffer_seconds",
        "max_pending_records",
        "max_pending_bytes",
        "max_record_bytes",
        "max_file_bytes",
        "index_every",
        "async_write",
        "index",
        "lines",
        "size",
        "last_step",
        "max_step",
        "flushed_row",
        "sorted",
        "has_duplicate_steps",
        "dropped",
        "dropped_bytes",
        "queued_bytes",
        "inflight_bytes",
        "pending_bytes",
        "buffer",
        "_closed",
        "_closing",
        "_inflight_records",
        "_active_number",
        "_active_size",
        "_active_path",
        "_flush_requested",
        "_retry_deadline",
        "_last_meta_time",
    }
)


class JsonlWriter:
    """Rust-owned admission, buffering, retries, rotation, counters and metadata.

    The reusable Python worker is only a callback trampoline: native condition
    waits determine its deadlines, retry delays and shutdown. Diagnostic sinks
    run after all native and SDK locks have been released.
    """

    def __init__(
        self,
        path,
        *,
        buffer_size=DEFAULT_BUFFER_SIZE,
        buffer_interval=DEFAULT_BUFFER_INTERVAL,
        max_buffer_seconds=DEFAULT_MAX_BUFFER_SECONDS,
        max_pending_records=DEFAULT_MAX_PENDING_RECORDS,
        max_pending_bytes=DEFAULT_MAX_PENDING_BYTES,
        max_record_bytes=DEFAULT_MAX_RECORD_BYTES,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        index_every=DEFAULT_INDEX_EVERY,
        meta_extra=None,
        async_write=False,
    ):
        self._owner_pid = process_id()
        validate_byte_limits(max_pending_bytes, max_record_bytes)
        validate_file_limit(max_file_bytes)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._write_lock = threading.Lock()
        self._diagnostics = _Diagnostics()
        self._worker = None
        self.meta_extra = dict(meta_extra or {})
        run = encode_line(self.meta_extra).decode("utf-8").strip()
        self._native = NativeWriter(
            Path(path),
            max(1, int(buffer_size)),
            None if buffer_interval is None else float(buffer_interval),
            None if max_buffer_seconds is None else float(max_buffer_seconds),
            max(max(1, int(buffer_size)), int(max_pending_records or 0)),
            max_pending_bytes,
            max_record_bytes,
            max_file_bytes,
            max(1, int(index_every)),
            run,
            bool(async_write),
        )
        self._messages(self._native.startup_diagnostics())

    def __getattr__(self, name):
        if name == "flushed_step":
            return self.last_step
        if name not in _ATTRIBUTES:
            raise AttributeError(name)
        check_process(self._owner_pid)
        value = self._native.get(name)
        if name in ("last_step", "max_step"):
            return None if value is None else int(value)
        if name == "index":
            return [(int(s), line, o) for s, line, o in value]
        if name == "buffer":
            return [(int(s), r, b) for s, r, b in value]
        if name == "_retry_deadline":
            return None if value is None else time.monotonic() + value
        return value

    def __setattr__(self, name, value):
        if name in _ATTRIBUTES and "_native" in self.__dict__:
            check_process(self._owner_pid)
            self._native.set(name, value)
        else:
            object.__setattr__(self, name, value)

    def _messages(self, messages):
        for level, message in messages:
            self._diagnostics.add(level, message)

    def _ensure_worker(self):
        with self._lock:
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._run_worker,
                    name=f"expr-tracker-writer-{self.path.name}",
                    daemon=True,
                )
                self._worker.start()

    def _run_worker(self):
        while self._native.wait_work():
            with self._lock:
                pass
            self.flush()

    def _capacity_wait(self):
        # SDK instrumentation can observe a native wait without owning its state.
        with self._condition:
            self._condition.wait(0)

    def enqueue_with_status(self, step, row, line):
        check_process(self._owner_pid)
        accepted, due, messages = self._native.enqueue(
            str(step), row, line, self._ensure_worker, self._capacity_wait
        )
        self._messages(messages)
        return accepted, due

    def enqueue(self, step, row, line):
        return self.enqueue_with_status(step, row, line)[1]

    def append_with_status(self, step, row, line):
        accepted, due = self.enqueue_with_status(step, row, line)
        if accepted and not self.async_write:
            if due:
                self.flush()
            else:
                self.schedule_timer()
        return accepted

    def append(self, step, row, line):
        self.append_with_status(step, row, line)

    def schedule_timer(self):
        check_process(self._owner_pid)
        self._native.schedule(self._ensure_worker)

    def _write(self, payload):
        self._native.write(payload, self.__dict__.get("_append"))

    def _append(self, payload):
        self._native.append(payload)

    def _truncate_partial_write(self, size_before):
        return self._native.rollback(size_before)

    def flush(self):
        check_process(self._owner_pid)
        with self._diagnostics, self._write_lock:
            self._messages(
                self._native.flush(
                    self.__dict__.get("_write")
                    or (self._write if "_append" in self.__dict__ else None),
                    self.__dict__.get("_truncate_partial_write"),
                    self.__dict__.get("_save_meta"),
                )
            )
            self._native.schedule(self._ensure_worker)
        with self._condition:
            self._condition.notify_all()

    def _save_meta(self, force=False):
        self._messages(self._native.save_meta(force))

    def reader(self):
        check_process(self._owner_pid)
        return JsonlReader(
            self.path,
            index=self.index,
            merge=self.has_duplicate_steps or not self.sorted,
        )

    def close(self):
        check_process(self._owner_pid)
        if self._worker is threading.current_thread():
            raise RuntimeError("Cannot close a writer from its own background worker.")
        if (
            in_diagnostic_callback()
            and self._worker is not None
            and self._worker.is_alive()
        ):
            raise RuntimeError(
                "Cannot close a writer from a diagnostic sink while its "
                "background worker is active."
            )
        with self._diagnostics:
            if not self._native.begin_close():
                return
            with self._condition:
                self._condition.notify_all()
            try:
                if self._worker is not None:
                    self._worker.join()
                self.flush()
            finally:
                with self._write_lock:
                    try:
                        self._save_meta(force=True)
                    finally:
                        self._messages(self._native.end_close())
                        with self._condition:
                            self._condition.notify_all()
