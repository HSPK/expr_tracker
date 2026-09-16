"""Rejected writer records must not become immortal cache entries or reach sinks."""

import json
import subprocess
import sys
import textwrap
import threading

import pytest

import expr_tracker as et


def test_oversized_metric_is_not_cached_or_forwarded(run):
    instance = run(max_pending_bytes=4096, max_record_bytes=512, cache_bytes=1)
    history = instance.history
    observed = []
    original = history.on_commit

    def notify(row):
        observed.append(row)
        original(row)

    history.on_commit = notify
    assert history.log({"payload": "x" * 4096}) is None
    assert history.get(-1) == []
    assert history.stats()["rows_logged"] == 0
    assert history.stats()["cached_bytes"] == 0
    assert history.writer.dropped == 1
    assert not observed
    assert "payload" not in instance.summary
    assert history.series.points("payload") == ()

    history.log({"value": 1})
    history.flush()
    assert len(observed) == 1 and observed[0]["value"] == 1
    assert instance.summary["value"] == 1
    assert history.get(-1)[0]["value"] == 1
    assert history.stats()["rows_logged"] == 1


def test_oversized_span_keeps_valid_timing_but_not_its_large_event(run):
    instance = run(max_pending_bytes=4096, max_record_bytes=512)
    with et.span("large", payload="x" * 4096):
        pass
    assert instance.history.span_writer.dropped == 1
    with et.span("small"):
        pass
    et.finish()
    spans = [
        json.loads(line)
        for line in instance.history.spans_fp.read_text().splitlines()
    ]
    assert [span["name"] for span in spans] == ["small"]
    row = instance.history_query(1)[0]
    assert "time_ms/large" in row and "time_ms/small" in row


@pytest.mark.parametrize("name", ["max_pending_bytes", "max_record_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1024"])
def test_bad_byte_limits_do_not_close_an_existing_store(store, tmp_path, name, value):
    history = store()
    original_path = history.log_fp
    with pytest.raises((TypeError, ValueError), match=name):
        history.init(
            project="other",
            name="invalid",
            dir=str(tmp_path),
            **{name: value},
        )
    assert history.log_fp == original_path
    assert not history._closed
    history.log({"still_running": 1})
    assert history.get(1)[0]["still_running"] == 1


def test_history_reads_remain_available_while_admission_is_backpressured(run, monkeypatch):
    instance = run(
        buffer_size=2,
        buffer_interval=None,
        max_buffer_seconds=0.02,
        max_pending_records=2,
    )
    history = instance.history
    instance.log({"warm": 0})
    writing, release, produced, read = (threading.Event() for _ in range(4))
    original = history.writer._write
    errors = []

    def delayed_write(payload):
        writing.set()
        if not release.wait(5):
            raise RuntimeError("writer was not released")
        original(payload)

    def produce():
        try:
            instance.log({"three": 3})
        except Exception as error:
            errors.append(error)
        finally:
            produced.set()

    def query():
        try:
            history.stats()
            history.get(1)
        except Exception as error:
            errors.append(error)
        finally:
            read.set()

    monkeypatch.setattr(history.writer, "_write", delayed_write)
    instance.log({"one": 1})
    producer = threading.Thread(target=produce)
    reader = threading.Thread(target=query)
    try:
        assert writing.wait(5)
        instance.log({"two": 2})
        producer.start()
        assert not produced.wait(0.05)
        reader.start()
        assert read.wait(1), "queue backpressure blocked history reads"
    finally:
        release.set()
        for thread in (producer, reader):
            if thread.ident is not None:
                thread.join(5)
    assert not errors
    assert produced.is_set()
    assert history.writer.dropped == 0


def test_dropped_writer_error_handler_and_reader_do_not_invert_locks(tmp_path):
    script = textwrap.dedent("""
        import json
        import os
        import sys
        import threading
        from loguru import logger
        from expr_tracker.history import HistoryStore

        logger.remove()
        history = HistoryStore().init(project="p", name="r", dir=sys.argv[1], max_record_bytes=64)
        entered, release, reached_view = (threading.Event() for _ in range(3))
        written, read = threading.Event(), threading.Event()

        def sink(message):
            entered.set()
            release.wait(3)
            history.open_record()

        logger.add(sink, level="ERROR")
        original = history._make_view

        def view(*args, **kwargs):
            reached_view.set()
            return original(*args, **kwargs)

        history._make_view = view

        def produce():
            history.writer.enqueue(0, 0, json.dumps({"_step": 0, "payload": "x" * 100}).encode())
            written.set()

        def query():
            history.get(1)
            read.set()

        threading.Thread(target=produce, daemon=True).start()
        assert entered.wait(3)
        threading.Thread(target=query, daemon=True).start()
        reached_view.wait(0.1)
        release.set()
        complete = written.wait(2) and read.wait(2)
        os.write(1, json.dumps({"complete": complete}).encode())
        os._exit(0)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert json.loads(result.stdout) == {"complete": True}
