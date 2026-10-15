"""Bounded subprocesses isolate deadlocks and Loguru's process-global sinks."""

import json
import subprocess
import sys
import textwrap

import pytest

from expr_tracker.history.writer import JsonlWriter


def run_script(tmp_path, body, **variables):
    prelude = """
        import json
        import os
        import sys
        import threading
        import time
        import traceback
        from pathlib import Path
        from loguru import logger
        from expr_tracker.history import HistoryStore
        from expr_tracker.history.writer import JsonlWriter

        logger.remove()
        root = Path(sys.argv[1])
        errors = []

        def line(step, size=32):
            prefix = json.dumps({"_step": step}).encode()
            return prefix + b" " * (size - len(prefix) - 1) + b"\\n"

        def start(target):
            def checked():
                try:
                    target()
                except BaseException:
                    errors.append(traceback.format_exc())
            thread = threading.Thread(target=checked, daemon=True)
            thread.start()
            return thread

        def join(thread):
            thread.join(3)
            assert not thread.is_alive(), "writer operation deadlocked"

        def add_sink(sink, level="ERROR"):
            def checked(message):
                try:
                    sink(message)
                except BaseException:
                    errors.append(traceback.format_exc())
                    raise
            return logger.add(checked, level=level, catch=False)

        def assert_unlocked(writer):
            acquired = []
            def probe():
                for lock in (writer._lock, writer._write_lock):
                    success = lock.acquire(timeout=0.3)
                    acquired.append(success)
                    if success:
                        lock.release()
            join(start(probe))
            assert acquired == [True, True], acquired
    """
    script = (
        textwrap.dedent(prelude)
        + "\n"
        + "\n".join(f"{name} = {value!r}" for name, value in variables.items())
        + "\n"
        + textwrap.dedent(body)
        + "\nassert not errors, errors\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_two_writer_error_sink_stats_reentry(tmp_path):
    run_script(
        tmp_path,
        """
        history = HistoryStore().init(
            project="p", name="r", dir=str(root), max_record_bytes=64
        )
        entered, release, second_logging = (threading.Event() for _ in range(3))
        seen = []
        original_error = logger.error

        def error(message):
            if "spans.jsonl" in message:
                second_logging.set()
            original_error(message)

        logger.error = error

        def sink(message):
            if not seen:
                entered.set()
                assert release.wait(3)
            seen.append(history.stats())

        add_sink(sink)
        first = start(lambda: history.writer.enqueue(0, 0, line(0, 128)))
        assert entered.wait(3)
        second = start(lambda: history.span_writer.enqueue(0, 0, line(0, 128)))
        assert second_logging.wait(3)
        release.set()
        join(first)
        join(second)
        assert len(seen) == 2
        assert history.writer.dropped == history.span_writer.dropped == 1
        assert history.writer.dropped_bytes == history.span_writer.dropped_bytes == 128
        assert all(item["pending_bytes"] == item["span_pending_bytes"] == 0 for item in seen)
        history.finish()
    """,
    )


@pytest.mark.parametrize("rejection", ["oversized", "closed"])
def test_rejection_sink_can_read_and_flush_same_writer(tmp_path, rejection):
    run_script(
        tmp_path,
        """
        writer = JsonlWriter(root / "metrics.jsonl", max_record_bytes=64)
        writer.enqueue(0, 0, line(0))
        if rejection == "closed":
            writer.close()
        seen = []

        def sink(message):
            assert_unlocked(writer)
            assert writer.pending_bytes == (32 if rejection == "oversized" else 0)
            assert len(writer.buffer) == (1 if rejection == "oversized" else 0)
            writer.reader().read_all()
            writer.flush()
            assert writer.pending_bytes == 0
            assert writer.reader().read_all() == [{"_step": 0}]
            seen.append(message.record["level"].name)

        add_sink(sink, level="WARNING")
        assert writer.enqueue_with_status(1, 1, line(1, 128)) == (False, False)
        assert seen == ["ERROR" if rejection == "oversized" else "WARNING"]
        writer.close()
    """,
        rejection=rejection,
    )


def test_failed_write_sink_sees_rollback_and_requeued_bytes_before_retry(tmp_path):
    run_script(
        tmp_path,
        """
        writer = JsonlWriter(root / "metrics.jsonl", max_buffer_seconds=None)
        original_write = writer._write
        observed = []

        def fail(payload):
            with open(writer.path, "ab") as stream:
                stream.write(payload[:11])
            writer.enqueue(1, 1, line(1, 48))
            raise OSError("partial write")

        def sink(message):
            assert_unlocked(writer)
            assert writer.path.read_bytes() == b""
            assert writer.queued_bytes == writer.pending_bytes == 80
            assert writer.inflight_bytes == writer._inflight_records == 0
            assert [row for _, row, _ in writer.buffer] == [0, 1]
            assert writer.lines == writer.size == 0
            assert writer.flushed_row == -1
            assert writer._retry_deadline > time.monotonic()
            writer._write = original_write
            writer.flush()
            assert writer.pending_bytes == 0
            observed.append(message.record["message"])

        writer._write = fail
        add_sink(sink)
        writer.enqueue(0, 0, line(0))
        writer.flush()
        assert len(observed) == 1 and "partial write" in observed[0]
        writer.close()
        assert writer.reader().read_all() == [{"_step": 0}, {"_step": 1}]
    """,
    )


def test_rollback_warning_follows_consistent_failure_accounting(tmp_path):
    run_script(
        tmp_path,
        """
        writer = JsonlWriter(root / "metrics.jsonl", max_buffer_seconds=None)
        original_write, original_truncate = writer._write, writer._truncate_partial_write
        observed = []

        def fail(payload):
            with open(writer.path, "ab") as stream:
                stream.write(payload)
            raise OSError("flush failed")

        def cannot_truncate(*args):
            raise OSError("rollback failed")

        def sink(message):
            assert_unlocked(writer)
            assert writer.has_duplicate_steps
            assert writer.pending_bytes == writer.queued_bytes == 32
            assert writer.inflight_bytes == writer._inflight_records == 0
            assert writer._retry_deadline > time.monotonic()
            assert writer.reader().merge
            observed.append((message.record["level"].name, message.record["message"]))

        writer._write, writer._truncate_partial_write = fail, cannot_truncate
        add_sink(sink, level="WARNING")
        writer.enqueue(0, 0, line(0))
        writer.flush()
        assert [level for level, _ in observed] == ["ERROR", "WARNING"]
        assert "flush failed" in observed[0][1]
        assert "rollback failed" in observed[1][1]
        writer._write, writer._truncate_partial_write = original_write, original_truncate
        writer.close()
        assert writer.reader().read_all() == [{"_step": 0}]
    """,
    )


def test_failure_overflow_sink_runs_after_admission_is_complete(tmp_path):
    run_script(
        tmp_path,
        """
        writer = JsonlWriter(
            root / "metrics.jsonl", max_pending_bytes=64, max_buffer_seconds=None
        )
        original_write = writer._write

        def fail(payload):
            raise OSError("offline")

        writer._write = fail
        writer.enqueue(0, 0, line(0))
        writer.enqueue(1, 1, line(1))
        writer.flush()
        seen = []

        def sink(message):
            assert_unlocked(writer)
            assert writer.pending_bytes == writer.queued_bytes == 64
            assert writer.inflight_bytes == 0
            assert writer.dropped == 1 and writer.dropped_bytes == 32
            assert [row for _, row, _ in writer.buffer] == [1, 2]
            writer._write = original_write
            writer.flush()
            seen.append(message.record["message"])

        add_sink(sink)
        assert writer.enqueue_with_status(2, 2, line(2))[0]
        assert len(seen) == 1 and "dropped 1 oldest queued records (32 bytes)" in seen[0]
        writer.close()
        assert writer.reader().read_all() == [{"_step": 1}, {"_step": 2}]
    """,
    )


def test_inflight_overflow_sink_can_wait_for_flush_without_holding_buffer_lock(
    tmp_path,
):
    run_script(
        tmp_path,
        """
        writer = JsonlWriter(
            root / "metrics.jsonl", max_pending_bytes=64, max_buffer_seconds=None
        )
        original_write = writer._write
        writing, release = threading.Event(), threading.Event()
        seen = []

        def fail(payload):
            raise OSError("offline")

        def retry(payload):
            writing.set()
            assert release.wait(3)
            original_write(payload)

        def sink(message):
            assert writer.inflight_bytes == writer.pending_bytes == 64
            assert writer.queued_bytes == 0 and writer.buffer == []
            assert writer.dropped == 1 and writer.dropped_bytes == 32
            release.set()
            writer.flush()
            assert writer.pending_bytes == 0 and writer.lines == 1
            seen.append(message.record["message"])

        writer._write = fail
        writer.enqueue(0, 0, line(0, 64))
        writer.flush()
        writer._write = retry
        flusher = start(writer.flush)
        assert writing.wait(3)
        add_sink(sink)
        assert writer.enqueue_with_status(1, 1, line(1)) == (False, False)
        join(flusher)
        assert len(seen) == 1 and "(32 bytes)" in seen[0]
        writer.close()
        assert writer.reader().read_all() == [{"_step": 0}]
    """,
    )


@pytest.mark.parametrize("operation", ["flush", "close", "save"])
def test_metadata_warning_sink_runs_outside_nested_locks(tmp_path, operation):
    run_script(
        tmp_path,
        """
        writer = JsonlWriter(root / "metrics.jsonl", max_buffer_seconds=None)
        writer.meta_path = root / "missing" / "metrics.meta.json"
        if operation != "save":
            writer.enqueue(0, 0, line(0))
        seen = []

        def sink(message):
            assert_unlocked(writer)
            assert writer.pending_bytes == 0
            assert writer._closed == (operation == "close")
            writer.reader().read_all()
            writer.flush()
            if operation == "close":
                writer.close()
            seen.append(message.record["message"])

        handle = add_sink(sink, level="WARNING")
        if operation == "save":
            writer._save_meta(force=True)
        else:
            getattr(writer, operation)()
        assert len(seen) == (2 if operation == "close" else 1)
        assert all("Failed to write" in message for message in seen)
        logger.remove(handle)
        writer.meta_path.parent.mkdir()
        writer.close()
    """,
        operation=operation,
    )


def test_final_close_diagnostics_observe_closed_state_and_allow_reentry(tmp_path):
    run_script(
        tmp_path,
        """
        writer = JsonlWriter(root / "metrics.jsonl", max_buffer_seconds=None)
        writer.meta_path = root / "missing" / "metrics.meta.json"
        seen = []

        def fail(payload):
            raise OSError("final flush failed")

        def sink(message):
            assert_unlocked(writer)
            assert writer._closed and writer._closing
            assert writer.pending_bytes == writer.queued_bytes == 32
            assert writer.inflight_bytes == writer._inflight_records == 0
            writer.flush()
            writer.close()
            seen.append((message.record["level"].name, message.record["message"]))

        writer._write = fail
        writer.enqueue(0, 0, line(0))
        add_sink(sink, level="WARNING")
        writer.close()
        assert [level for level, _ in seen] == ["ERROR", "WARNING", "ERROR"]
        assert "final flush failed" in seen[0][1]
        assert "Failed to write" in seen[1][1]
        assert "1 unwritten records" in seen[2][1] and "(32 bytes)" in seen[2][1]
    """,
    )


@pytest.mark.parametrize("already_closing", [False, True])
def test_worker_sink_cannot_close_itself_or_wait_for_its_closer(
    tmp_path, already_closing
):
    run_script(
        tmp_path,
        """
        writer = JsonlWriter(
            root / "metrics.jsonl", async_write=True, max_buffer_seconds=30
        )
        original_write = writer._write
        entered, release, seen = (threading.Event() for _ in range(3))
        attempts = []

        def fail(payload):
            attempts.append(time.monotonic())
            raise OSError("worker failure")

        def sink(message):
            assert threading.current_thread() is writer._worker
            entered.set()
            assert release.wait(3)
            assert_unlocked(writer)
            assert writer._closing == already_closing and not writer._closed
            try:
                writer.close()
            except RuntimeError as error:
                assert str(error) == "Cannot close a writer from its own background worker."
            else:
                raise AssertionError("self-close was not rejected")
            assert writer._closing == already_closing and not writer._closed
            writer._write = original_write
            seen.set()

        writer._write = fail
        add_sink(sink)
        writer.enqueue(0, 0, line(0))
        assert entered.wait(3)
        if already_closing:
            closer = start(writer.close)
            with writer._condition:
                assert writer._condition.wait_for(lambda: writer._closing, timeout=3)
        release.set()
        assert seen.wait(3)
        if already_closing:
            join(closer)
        else:
            assert writer._worker.is_alive()
            assert writer._retry_deadline > time.monotonic()
            writer.close()
        assert len(attempts) == 1
        assert writer._closed and not writer._worker.is_alive()
        assert writer.pending_bytes == 0 and writer.lines == 1
    """,
        already_closing=already_closing,
    )


def test_sink_failure_is_nonfatal_and_does_not_replace_close_exception(tmp_path):
    result = run_script(
        tmp_path,
        """
        writer = JsonlWriter(root / "metrics.jsonl", max_record_bytes=64)
        writer.meta_path = root / "missing" / "metrics.meta.json"

        def broken_sink(message):
            raise ValueError("sink exploded")

        logger.add(broken_sink, level="WARNING", catch=False)
        assert writer.enqueue_with_status(0, 0, line(0, 128)) == (False, False)
        assert writer.dropped_bytes == 128 and writer.pending_bytes == 0
        original = RuntimeError("original operation error")

        def broken_flush():
            raise original

        writer.flush = broken_flush
        try:
            writer.close()
        except RuntimeError as error:
            assert error is original
        else:
            raise AssertionError("operation exception was swallowed")
        assert writer._closed
    """,
    )
    assert "Rejected oversized record" in result.stderr
    assert "Failed to write" in result.stderr
    assert (
        result.stderr.count("Writer diagnostic sink failed: ValueError: sink exploded")
        == 2
    )


def test_sink_cannot_join_another_worker_waiting_for_loguru(tmp_path):
    run_script(
        tmp_path,
        """
        writer = JsonlWriter(root / "metrics.jsonl", max_record_bytes=64)
        other = JsonlWriter(
            root / "spans.jsonl", async_write=True, max_buffer_seconds=30
        )
        entered, release, worker_logging = (threading.Event() for _ in range(3))
        original_write, original_error = other._write, logger.error
        seen = []

        def fail(payload):
            raise OSError("other worker failed")

        def error(message):
            if "other worker failed" in message:
                worker_logging.set()
            original_error(message)

        def sink(message):
            if "Rejected oversized" not in str(message):
                return
            entered.set()
            assert release.wait(3)
            try:
                other.close()
            except RuntimeError as error:
                seen.append(str(error))
            else:
                raise AssertionError("unsafe worker join was not rejected")
            assert not other._closing and not other._closed

        other._write, logger.error = fail, error
        add_sink(sink)
        producer = start(lambda: writer.enqueue(0, 0, line(0, 128)))
        assert entered.wait(3)
        other.enqueue(0, 0, line(0))
        assert worker_logging.wait(3)
        release.set()
        join(producer)
        assert seen == [
            "Cannot close a writer from a diagnostic sink while its background worker is active."
        ]
        other._write = original_write
        writer.close()
        other.close()
        assert other.lines == 1 and not other._worker.is_alive()
    """,
    )


def test_raising_sink_does_not_kill_worker_or_lose_retry_bytes(tmp_path):
    result = run_script(
        tmp_path,
        """
        writer = JsonlWriter(
            root / "metrics.jsonl", async_write=True, max_buffer_seconds=30
        )
        original_write = writer._write
        received = threading.Event()

        def fail(payload):
            raise OSError("retryable write failure")

        def broken_sink(message):
            received.set()
            raise ValueError("worker sink failed")

        writer._write = fail
        logger.add(broken_sink, level="ERROR", catch=False)
        writer.enqueue(0, 0, line(0))
        assert received.wait(3)
        assert writer._worker.is_alive() and not writer._closing
        assert writer.pending_bytes == writer.queued_bytes == 32
        assert writer.inflight_bytes == writer._inflight_records == 0
        assert writer._retry_deadline > time.monotonic()
        writer._write = original_write
        writer.close()
        assert writer.lines == 1 and writer.pending_bytes == 0
        assert not writer._worker.is_alive()
    """,
    )
    assert "retryable write failure" in result.stderr
    assert (
        "Writer diagnostic sink failed: ValueError: worker sink failed" in result.stderr
    )


def test_reentrant_diagnostics_use_stderr_without_recursive_sink_calls(tmp_path):
    result = run_script(
        tmp_path,
        """
        writer = JsonlWriter(root / "metrics.jsonl", max_record_bytes=64)
        other = JsonlWriter(root / "spans.jsonl", max_record_bytes=64)
        seen = []

        def sink(message):
            assert_unlocked(writer)
            assert writer.enqueue_with_status(1, 1, line(1, 128)) == (False, False)
            assert other.enqueue_with_status(0, 0, line(0, 128)) == (False, False)
            seen.append(message.record["message"])

        add_sink(sink)
        writer.enqueue(0, 0, line(0, 128))
        assert len(seen) == 1
        assert writer.dropped == 2 and other.dropped == 1
        writer.close()
        other.close()
    """,
    )
    assert result.stderr.count("ERROR: Rejected oversized record") == 2


def test_metadata_surrogates_round_trip_without_escaping_normal_unicode(tmp_path):
    extra = {"path": "surrogate-\udcff", "normal": "\u4e2d\u6587", "\udc80": ["\ud800"]}
    writer = JsonlWriter(tmp_path / "metrics.jsonl", meta_extra=extra)
    writer.append(0, 0, b'{"_step":0}\n')
    writer.close()
    raw = writer.meta_path.read_bytes()
    meta = json.loads(raw)
    assert meta["run"] == extra
    assert "\u4e2d\u6587".encode() in raw
    assert b"\\udcff" in raw and b"\\ud800" in raw
    assert not writer.meta_path.with_suffix(".json.tmp").exists()
    reopened = JsonlWriter(writer.path)
    assert reopened.lines == 1 and reopened.last_step == 0
    reopened.close()
