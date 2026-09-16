"""Accepted writes drain before shutdown; trace-only workers keep bounded state."""

import json
import subprocess
import sys
import textwrap
import threading

import pytest
from loguru import logger

import expr_tracker as et


def records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("async_write", [False, True])
def test_finish_waits_for_an_accepted_span_append(run, monkeypatch, async_write):
    instance = run(span_async_write=async_write, max_buffer_seconds=60)
    with et.span("first"):
        pass
    instance.history.flush()
    writer = instance.history.span_writer
    accepted, release, finishing, finished = (
        threading.Event() for _ in range(4)
    )
    errors = []
    original = writer.append_with_status

    def delayed_append(*args, **kwargs):
        accepted.set()
        if not release.wait(5):
            raise RuntimeError("span append was not released")
        return original(*args, **kwargs)

    def produce():
        with et.span("last"):
            pass

    def close():
        finishing.set()
        try:
            instance.finish()
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    monkeypatch.setattr(writer, "append_with_status", delayed_append)
    producer = threading.Thread(target=produce)
    closer = threading.Thread(target=close)
    producer.start()
    try:
        assert accepted.wait(5)
        closer.start()
        assert finishing.wait(5)
        assert not finished.wait(0.05)
    finally:
        release.set()
        producer.join(5)
        if closer.ident is not None:
            closer.join(5)
    assert not producer.is_alive() and not closer.is_alive()
    assert finished.is_set() and not errors
    assert writer.buffer == []
    assert [row["name"] for row in records(writer.path)] == ["first", "last"]
    row = records(instance.history.log_fp)[0]
    assert "time_ms/last" in row
    assert instance.summary["time_ms/last"] == row["time_ms/last"]
    assert instance.history.record_span({"time_ms/rejected": 1}, {"name": "rejected"}) is None
    assert len(records(writer.path)) == 2


def test_finish_waits_for_an_accepted_metric_commit(run, monkeypatch):
    instance = run()
    accepted, release, finished = (threading.Event() for _ in range(3))
    original = instance.history._emit
    errors = []

    def delayed_emit(row):
        accepted.set()
        if not release.wait(5):
            raise RuntimeError("metric commit was not released")
        original(row)

    def produce():
        try:
            instance.log({"loss": 0.25})
        except Exception as error:
            errors.append(error)

    def close():
        try:
            instance.finish()
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    monkeypatch.setattr(instance.history, "_emit", delayed_emit)
    producer = threading.Thread(target=produce)
    closer = threading.Thread(target=close)
    producer.start()
    try:
        assert accepted.wait(5)
        closer.start()
        assert not finished.wait(0.05)
    finally:
        release.set()
        producer.join(5)
        if closer.ident is not None:
            closer.join(5)
    assert not producer.is_alive() and not closer.is_alive()
    assert finished.is_set() and not errors
    assert records(instance.history.log_fp)[0]["loss"] == 0.25
    assert json.loads(instance.summary.path.read_text())["loss"] == 0.25


@pytest.mark.parametrize("span_count", [False, True])
def test_trace_only_spans_do_not_grow_open_metric_state(run, span_count):
    instance = run(span_metrics=False, span_count=span_count, cache_bytes=1)
    for index in range(1000):
        with et.span(
            f"request/{index}", plugins=[lambda span: {"gpu_percent": 60.0}]
        ):
            pass
    assert instance.step == 0
    assert instance.history._open_row == {}
    assert instance.history._open_weights == {}
    assert instance.history_query(-1) == []
    instance.history.flush()
    spans = records(instance.history.spans_fp)
    assert len(spans) == 1000
    assert all(span["_step"] == 0 for span in spans)
    assert all(span["metrics"] == {"gpu_percent": 60.0} for span in spans)
    assert records(instance.history.log_fp) == []

    et.log({"loss": 1.0})
    with et.span("after-log"):
        pass
    et.finish()
    rows = records(instance.history.log_fp)
    assert len(rows) == 1 and rows[0]["loss"] == 1.0
    assert not any(key.startswith(("time_ms/", "count/")) for key in rows[0])
    assert records(instance.history.spans_fp)[-1]["_step"] == 1


def test_async_span_ordinals_do_not_depend_on_flush_progress(run, monkeypatch):
    instance = run(span_metrics=False)
    writer = instance.history.span_writer
    writing, release = threading.Event(), threading.Event()
    original = writer._write

    def delayed_write(payload):
        writing.set()
        if not release.wait(5):
            raise RuntimeError("writer was not released")
        original(payload)

    monkeypatch.setattr(writer, "_write", delayed_write)
    with et.span("first"):
        pass
    try:
        assert writing.wait(5)
        for _ in range(9):
            with et.span("queued"):
                pass
        assert instance.history._next_span_row == 10
    finally:
        release.set()
    instance.history.flush()
    assert writer.lines == 10
    assert writer.flushed_row == 9


@pytest.mark.parametrize("operation", ["finish", "init"])
def test_lifecycle_changes_in_commit_callbacks_fail_without_deadlocking(run, operation):
    instance = run()
    errors = []

    def callback(row):
        with pytest.raises(RuntimeError, match="inside a history commit callback") as error:
            if operation == "finish":
                et.finish()
            else:
                et.init(project="another", backends=[])
        errors.append(str(error.value))

    instance.history.on_commit = callback
    instance.log({"loss": 1.0})
    assert len(errors) == 1
    assert not instance.history._closed
    assert et.get_run() is instance
    instance.history.on_commit = None
    et.finish()


def test_callback_cannot_deadlock_a_concurrent_module_finish(run):
    instance = run()
    entered, release, finished = (threading.Event() for _ in range(3))
    errors = []

    def callback(row):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("callback was not released")
        with pytest.raises(RuntimeError, match="inside a history commit callback"):
            et.finish()

    def produce():
        instance.log({"loss": 1.0})

    def close():
        try:
            et.finish()
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    instance.history.on_commit = callback
    producer = threading.Thread(target=produce)
    closer = threading.Thread(target=close)
    producer.start()
    try:
        assert entered.wait(5)
        closer.start()
        assert instance.history._closing.wait(5)
        assert not finished.is_set()
    finally:
        release.set()
        producer.join(5)
        if closer.ident is not None:
            closer.join(5)
    assert finished.is_set() and not errors
    assert not producer.is_alive() and not closer.is_alive()
    assert et.get_run() is None


def test_all_concurrent_run_finish_callers_wait_for_completion(run, monkeypatch):
    instance = run()
    entered, release = threading.Event(), threading.Event()
    done = [threading.Event(), threading.Event()]
    errors = []
    original = instance.history._emit

    def delayed_emit(row):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("metric commit was not released")
        original(row)

    def close(index):
        try:
            instance.finish()
        except Exception as error:
            errors.append(error)
        finally:
            done[index].set()

    monkeypatch.setattr(instance.history, "_emit", delayed_emit)
    producer = threading.Thread(target=lambda: instance.log({"loss": 1.0}))
    closers = [threading.Thread(target=close, args=(index,)) for index in range(2)]
    producer.start()
    try:
        assert entered.wait(5)
        closers[0].start()
        assert instance.history._closing.wait(5)
        closers[1].start()
        assert not done[1].wait(0.05)
        assert not done[0].is_set()
        assert not instance._finished
    finally:
        release.set()
        producer.join(5)
        for closer in closers:
            if closer.ident is not None:
                closer.join(5)
    assert not errors
    assert all(event.is_set() for event in done)
    assert instance._finished
    assert records(instance.history.log_fp)[0]["loss"] == 1.0


def test_alert_delivery_cannot_finish_its_own_run(run):
    received = threading.Event()
    errors = []

    def handler(message):
        try:
            et.finish()
        except RuntimeError as error:
            errors.append(str(error))
        finally:
            received.set()

    instance = run(
        alert={
            "channels": [{
                "type": "callable",
                "name": "handler",
                "options": {"handler": handler},
                "policy": {"async_send": True},
            }]
        }
    )
    et.alert("shutdown from a callback")
    assert received.wait(5)
    assert errors == ["Cannot finish or reinitialize from an alert delivery callback."]
    assert not instance._finished
    et.finish()
    assert instance._finished


def test_shutdown_failure_wakes_other_finish_callers(tmp_path, monkeypatch):
    from expr_tracker.run import Run

    instance = Run(project="p", name="failed-close", dir=str(tmp_path), backends=[])
    entered, release = threading.Event(), threading.Event()
    errors = []

    def fail(exit_code):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("shutdown was not released")
        raise OSError("simulated shutdown failure")

    def close():
        try:
            instance.finish()
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=close) for _ in range(2)]
    try:
        with monkeypatch.context() as context:
            context.setattr(instance, "_finish", fail)
            threads[0].start()
            assert entered.wait(5)
            threads[1].start()
            release.set()
            for thread in threads:
                thread.join(5)
            assert all(not thread.is_alive() for thread in threads)
            assert len(errors) == 2
            assert any(isinstance(error, OSError) for error in errors)
            assert any(isinstance(error, RuntimeError) for error in errors)
            assert instance._finish_done.is_set() and not instance._finished
            with pytest.raises(RuntimeError, match="Run shutdown failed"):
                instance.finish()
    finally:
        release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(5)
        instance._finish(None)


def test_writer_error_sink_cannot_close_its_own_run(run, monkeypatch):
    instance = run()
    writer = instance.history.span_writer
    original = writer._write
    failed = False
    received = threading.Event()
    errors = []

    def write(payload):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated writer failure")
        original(payload)

    def sink(message):
        if "simulated writer failure" not in str(message):
            return
        try:
            et.finish()
        except RuntimeError as error:
            errors.append(str(error))
        finally:
            received.set()

    monkeypatch.setattr(writer, "_write", write)
    handle = logger.add(sink, level="ERROR")
    try:
        with et.span("work"):
            pass
        assert received.wait(5)
        assert errors == [
            "Cannot finish or reinitialize a run from a history writer worker."
        ]
        assert et.get_run() is instance and not instance._finished
        instance.history.flush()
    finally:
        logger.remove(handle)
    et.finish()
    assert writer.lines == 1


def test_write_error_sink_cannot_finish_an_unfinished_transaction(run):
    instance = run(max_record_bytes=128)
    errors = []

    def sink(message):
        if "Rejected oversized record" not in str(message):
            return
        try:
            et.finish()
        except RuntimeError as error:
            errors.append(str(error))

    handle = logger.add(sink, level="ERROR")
    try:
        assert instance.history.log({"large": "x" * 512}) is None
    finally:
        logger.remove(handle)
    assert errors == [
        "Cannot finish or reinitialize a run inside a history write transaction."
    ]
    assert et.get_run() is instance and not instance._finished
    instance.log({"loss": 1.0})
    assert instance.history_query(1)[0]["loss"] == 1.0


def test_config_error_sink_cannot_reenter_initialization(tmp_path):
    script = textwrap.dedent("""
        import json
        import sys
        from unittest.mock import patch
        from loguru import logger
        from expr_tracker.history import HistoryStore

        logger.remove()
        history = HistoryStore()
        errors = []
        def sink(message):
            try:
                history.finish()
            except RuntimeError as error:
                errors.append(str(error))
        logger.add(sink, level="ERROR")
        with patch("expr_tracker.history.store.json.dump", side_effect=OSError("config failure")):
            history.init(project="p", name="r", dir=sys.argv[1], config={"value": 1})
        assert errors == ["Cannot reenter history lifecycle operations."]
        history.log({"still_running": 1})
        history.finish()
        print(json.dumps({"complete": True}))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert json.loads(result.stdout) == {"complete": True}


def test_nonworker_writer_diagnostic_cannot_partially_close_a_run(run):
    instance = run(max_record_bytes=512)
    with et.span("warm"):
        pass
    instance.history.flush()
    assert instance.history.span_writer._worker.is_alive()
    errors = []

    def sink(message):
        if "Rejected oversized record" not in str(message):
            return
        try:
            et.finish()
        except RuntimeError as error:
            errors.append(str(error))

    handle = logger.add(sink, level="ERROR")
    try:
        assert instance.history.writer.enqueue_with_status(0, 0, b"x" * 1024) == (
            False,
            False,
        )
    finally:
        logger.remove(handle)
    assert errors == [
        "Cannot finish or reinitialize a run from a writer diagnostic callback."
    ]
    assert et.get_run() is instance
    assert not instance._finished and not instance.history._closing.is_set()
    assert not instance.history.span_writer._closing
    et.finish()
