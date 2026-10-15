"""Metric/span admission and commit publication on the owning HistoryStore."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from contextvars import copy_context

from loguru import logger

from .._native import (
    NativeRecord,
    NativeSummary,
)
from ._store_types import (
    _LOCAL_OVERRIDES,
    _WRITER_OVERRIDES,
    Reduction,
    _normalize_step,
    _Notification,
    _PreparedCommit,
    _serialized_write,
    resolve_commit,
)
from .series import MetricSeries


class HistoryWrites:
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
        if type(metrics) is dict and self._can_admit_native():
            result = self._log_local(metrics, step, commit)
            if result is not None:
                return result[0]
        return self.ingest(self._codec.encode(metrics), step=step, commit=commit)

    def _can_admit_native(self):
        return self.__dict__.keys().isdisjoint(
            _LOCAL_OVERRIDES
        ) and self.writer.__dict__.keys().isdisjoint(_WRITER_OVERRIDES)

    def _set_inline_publication(
        self,
        callback: Callable[[dict], None],
        summary_provider: Callable[[], NativeSummary | None],
    ):
        self._inline_callback = callback
        self._inline_summary = summary_provider

    def _publish_local(self, records: list[NativeRecord]) -> bool:
        if self.options.print_to_screen or not self._can_admit_native():
            return False
        if (
            "add" in self._series.__dict__
            or MetricSeries.add is not MetricSeries._native_add
        ):
            return False
        summary = None
        if self.on_commit is not None:
            if (
                self.on_commit is not self._inline_callback
                or self._inline_summary is None
            ):
                return False
            summary = self._inline_summary()
            if summary is None:
                return False
        previous = getattr(self._callback_state, "active", False)
        self._callback_state.active = True
        try:
            messages = self._native.publish_local(
                records, self._series._native, summary
            )
        finally:
            self._callback_state.active = previous
        if messages is None:
            return False
        for level, message in messages:
            getattr(logger, level)(message)
        return True

    @_serialized_write
    def _log_local(self, metrics, step, commit):
        self._require_init()
        if self._closed or self._closing.is_set():
            logger.warning("Tracker is already finished; dropping metrics.")
            return (None,)
        try:
            result: _PreparedCommit | None = self._native.prepare_local(
                metrics,
                None if step is None else str(step),
                resolve_commit(step, commit),
                self.writer._native,
                self._notify_queue,
                self.writer._ensure_worker,
                self.writer._capacity_wait,
                time.time(),
            )
        except (ValueError, TypeError, OverflowError, RecursionError) as error:
            logger.error(f"Dropping record that cannot be serialized: {error}")
            return (None,)
        if result is None:
            return None
        step, records, flush_due, cache_within_budget, diagnostics, warnings = result
        for key, message in warnings:
            if key:
                self._codec.warn_once(key, message)
            else:
                logger.warning(message)
        self.writer._messages(diagnostics)
        if records:
            self._schedule_commit(flush_due, cache_within_budget)
            if not self._publish_local(records):
                for record in records:
                    self._stage_notification(record.decode())
        return (None if step is None else int(step),)

    def _schedule_commit(self, flush_due: bool, cache_within_budget: bool = False):
        if flush_due:
            self.writer.flush()
        else:
            self.writer.schedule_timer()
        if not cache_within_budget:
            self._evict()

    def _stage_notification(self, record: dict):
        self._callback_state.pending.append(
            _Notification(record, self.options, self.on_commit, copy_context())
        )

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

    @_serialized_write
    def _record_span_native(self, span):
        self._require_init()
        if self._closed or self._closing.is_set():
            logger.warning("Tracker is already finished; dropping span.")
            return True
        writer = self.span_writer
        if writer is not None and not _WRITER_OVERRIDES.isdisjoint(writer.__dict__):
            return False
        result = self._native.record_span_native(
            span._native,
            span.path,
            span.depth,
            span.track,
            span.error,
            self.options.span_metrics,
            self.options.span_count,
            writer._native if writer is not None else None,
            self._next_span_row,
            writer._ensure_worker if writer is not None else None,
            writer._capacity_wait if writer is not None else None,
            time.time(),
        )
        if result is None:
            return False
        _, accepted, flush, diagnostics, warnings = result
        for key, message in warnings:
            if key:
                self._codec.warn_once(key, message)
            else:
                logger.warning(message)
        if writer is not None:
            writer._messages(diagnostics)
            if accepted:
                self._next_span_row += 1
                if flush:
                    writer.flush()
                elif not writer.async_write:
                    writer.schedule_timer()
        return True

    def _close_open_row(self) -> dict | None:
        record = self._native.close_open()
        return self._record_from_json(record) if record is not None else None

    def _record_from_json(self, record):
        if isinstance(record, NativeRecord):
            return record.decode()
        return json.loads(record)

    def _store_row(self, record: dict, line: bytes) -> bool | None:
        """Cache admitted records; return flush-due, or None for rejection.

        The write lock serializes admission and ordinal allocation. Do not
        hold the state lock while waiting for queue capacity: reads must stay live.
        """
        if isinstance(line, NativeRecord) and _WRITER_OVERRIDES.isdisjoint(
            self.writer.__dict__
        ):
            accepted, should_flush, messages = self._native.enqueue_record(
                self.writer._native,
                line,
                self.writer._ensure_worker,
                self.writer._capacity_wait,
                time.time(),
            )
            self.writer._messages(messages)
            return should_flush if accepted else None
        if isinstance(line, NativeRecord):
            line = line.bytes()
        step = record["_step"]
        row = self._next_row
        accepted, should_flush = self.writer.enqueue_with_status(step, row, line)
        if not accepted:
            return None
        encoded = getattr(self._callback_state, "encoded_row", None)
        text = (
            encoded[1].text()
            if encoded is not None
            and encoded[0] is record
            and isinstance(encoded[1], NativeRecord)
            else encoded[1]
            if encoded is not None and encoded[0] is record
            else line.decode("utf-8")
        )
        self._native.admitted(str(step), text, time.time())
        return should_flush
