"""Callback shutdown, reinitialization, retention and fork behavior."""

import ast
import gc
import json
import threading
import time
import weakref

import pytest
from _callback_helpers import (
    Backend as Backend,
)
from _callback_helpers import (
    assert_drained as assert_drained,
)
from _callback_helpers import (
    threads as threads,
)
from _callback_helpers import (
    wait_pending as wait_pending,
)
from loguru import logger

import expr_tracker as et


@pytest.mark.parametrize("finish_run", [False, True])
def test_finish_drains_callbacks_and_open_row_before_closing_sinks(
    run, threads, monkeypatch, finish_run
):
    entered, release = threading.Event(), threading.Event()
    committed = []

    def block():
        entered.set()
        assert release.wait(5)

    backend = Backend(block)
    instance = run(backends=[backend])
    history = instance.history
    original = history.on_commit

    def callback(row):
        original(row)
        committed.append(row["_step"])

    history.on_commit = callback
    closed = []
    for component, method in (
        (history.writer, "close"),
        (history.span_writer, "close"),
        (instance.summary, "save"),
        (instance.alerts, "close"),
    ):
        original_close = getattr(component, method)

        def close(original_close=original_close):
            closed.append((original_close, list(committed)))
            return original_close()

        monkeypatch.setattr(component, method, close)

    producer = threads(lambda: instance.log({"first": 0}))
    try:
        assert entered.wait(5)
        staged = threads(lambda: history.log({"open": 1}, commit=False))
        assert staged.wait(1), "uncommitted metrics waited for a callback"
        closer = threads(instance.finish if finish_run else history.finish)
        assert history._closing.wait(5)
        wait_pending(history, 1)
        rejected = []

        def write_after_closing():
            rejected.append(history.log({"rejected": 1}))
            rejected.append(history.record_span({}, {"name": "rejected"}))

        assert threads(write_after_closing).wait(1)
        assert rejected == [None, None]
        assert not closer.is_set() and not producer.is_set()
        assert not closed and not backend.finished.is_set()
    finally:
        release.set()
    assert producer.wait(5) and closer.wait(5)
    assert committed == [0, 1]
    assert closed and all(steps == [0, 1] for _, steps in closed)
    assert backend.finished.is_set() == finish_run
    assert instance.summary["open"] == 1
    assert history.writer.buffer == []
    assert history.span_writer.buffer == []
    assert_drained(history)


def test_reinit_preserves_old_metadata_hooks_and_callback_flush(
    store, threads, tmp_path
):
    entered, release = threading.Event(), threading.Event()
    old_rows, new_rows, old_prints, new_prints, rejected = [], [], [], [], []

    def old_callback(row):
        if row["_step"] == 0:
            entered.set()
            assert release.wait(5)
        history.flush()
        old_rows.append((history.project, history.name, history.log_fp, row["_step"]))
        rejected.append(history.log({"too_late": 1}))
        assert history.get(-1)

    history = store(
        name="old",
        on_commit=old_callback,
        print_to_screen=True,
        print_handle=old_prints.append,
    )
    old_path = history.log_fp
    old_writer = history.writer
    first = threads(lambda: history.log({"first": 0}))
    try:
        assert entered.wait(5)
        second = threads(lambda: history.log({"second": 1}))
        wait_pending(history, 1)
        assert threads(lambda: history.log({"open": 2}, commit=False)).wait(1)
        reinit = threads(
            lambda: history.init(
                project="new-project",
                name="new",
                dir=str(tmp_path),
                on_commit=new_rows.append,
                print_to_screen=True,
                print_handle=new_prints.append,
            )
        )
        assert history._closing.wait(5)
        wait_pending(history, 2)
        assert history.log_fp == old_path and history.name == "old"
        assert not reinit.is_set()
        assert not new_rows and not new_prints
    finally:
        release.set()
    assert first.wait(5) and second.wait(5) and reinit.wait(5)
    # Recursive flushes may finish nested callbacks before the outer callback.
    assert sorted(old_rows) == [("p", "old", old_path, step) for step in range(3)]
    assert rejected == [None, None, None]
    assert len(old_prints) == 3
    assert old_writer.buffer == []
    assert len(old_path.read_text().splitlines()) == 3
    assert history.project == "new-project" and history.name == "new"
    assert history.log({"new": 3}) == 0
    assert len(new_rows) == len(new_prints) == 1
    assert new_rows[0]["new"] == 3
    assert len(old_prints) == 3
    assert_drained(history)


@pytest.mark.parametrize("sink", ["print", "on_commit"])
def test_callback_can_log_read_and_flush_but_not_change_lifecycle(run, threads, sink):
    instance = run(cache_rows=1)
    history = instance.history
    observed, nested_returns, rejections = [], [], []

    def callback(row):
        observed.append(row["_step"])
        if row["_step"] != 0:
            return
        for operation in (
            history.finish,
            lambda: history.init(project="forbidden"),
            et.finish,
            lambda: et.init(project="forbidden", backends=[]),
        ):
            try:
                operation()
            except RuntimeError as error:
                rejections.append(str(error))
        et.log({"nested": 1})
        nested_returns.append(history.log({"open": 2}, commit=False))
        history.flush(commit_open=True)
        history.flush()
        nested_returns.append([r["_step"] for r in history.get(-1)])

    if sink == "print":
        # Reinit also exercises changing immutable print options between runs.
        history.init(
            project=history.project,
            name=history.name,
            run_dir=str(history.log_dir),
            cache_rows=1,
            print_to_screen=True,
            print_handle=lambda text: callback(ast.literal_eval(text)),
        )
    else:
        history.on_commit = callback
    assert threads(lambda: et.log({"outer": 0})).wait(5)
    assert observed == [0, 1, 2]
    assert nested_returns == [2, [0, 1, 2]]
    assert len(rejections) == 4
    assert all("inside a history commit callback" in error for error in rejections)
    assert not history._closed
    assert_drained(history)


def test_two_row_commit_keeps_recursive_notifications_in_admission_order(
    store, threads
):
    observed = []

    def callback(row):
        observed.append(row["_step"])
        if row["_step"] == 0:
            history.log({"recursive": 2})

    history = store(on_commit=callback)
    history.log({"open": 0}, commit=False)
    assert threads(lambda: history.log({"next": 1}, step=1, commit=True)).wait(5)
    history.flush()
    assert observed == [0, 1, 2]
    assert [
        json.loads(line)["_step"] for line in history.log_fp.read_text().splitlines()
    ] == observed
    assert_drained(history)


@pytest.mark.parametrize("sink", ["print", "on_commit"])
def test_callback_failure_releases_waiting_producers_and_finish(
    store, threads, monkeypatch, sink
):
    entered, release = threading.Event(), threading.Event()
    observed, warnings = [], []

    def callback(row):
        observed.append(row["_step"])
        if row["_step"] == 0:
            entered.set()
            assert release.wait(5)
        raise ValueError("broken callback")

    history = store(
        on_commit=callback if sink == "on_commit" else None,
        print_to_screen=sink == "print",
        print_handle=lambda text: callback(ast.literal_eval(text)),
    )
    monkeypatch.setattr(logger, "warning", warnings.append)
    first = threads(lambda: history.log({"first": 0}))
    try:
        assert entered.wait(5)
        second = threads(lambda: history.log({"second": 1}))
        wait_pending(history, 1)
        closer = threads(history.finish)
        assert history._closing.wait(5)
        assert not closer.is_set()
    finally:
        release.set()
    assert first.wait(5) and second.wait(5) and closer.wait(5)
    assert observed == [0, 1]
    assert len(warnings) == 2
    assert all("broken callback" in message for message in warnings)
    assert history.stats()["rows_logged"] == 2
    assert_drained(history)


def test_interrupted_callback_releases_notification_ownership(store, threads):
    entered, release = threading.Event(), threading.Event()
    observed = []

    def callback(row):
        observed.append(row["_step"])
        if row["_step"] == 0:
            entered.set()
            assert release.wait(5)
            raise KeyboardInterrupt

    history = store(on_commit=callback)

    def interrupted_log():
        with pytest.raises(KeyboardInterrupt):
            history.log({"first": 0})

    first = threads(interrupted_log)
    try:
        assert entered.wait(5)
        second = threads(lambda: history.log({"second": 1}))
        wait_pending(history, 1)
        closer = threads(history.finish)
        assert history._closing.wait(5)
    finally:
        release.set()
    assert first.wait(5) and second.wait(5) and closer.wait(5)
    assert observed == [0, 1]
    assert_drained(history)


def test_finish_still_waits_for_callbacks_when_writer_flush_fails(
    store, threads, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    closed, warnings = [], []

    def callback(row):
        entered.set()
        assert release.wait(5)

    history = store(on_commit=callback)
    producer = threads(lambda: history.log({"first": 0}))
    try:
        assert entered.wait(5)
        original_close = history.writer.close

        def fail_flush():
            raise OSError("flush failed")

        def close():
            closed.append(True)
            original_close()

        with monkeypatch.context() as patch:
            patch.setattr(history.writer, "flush", fail_flush)
            patch.setattr(history.writer, "close", close)
            patch.setattr(logger, "warning", warnings.append)
            closer = threads(history.finish)
            assert history._closing.wait(5)
            assert not closer.wait(0.05)
            assert not closed
            release.set()
            assert producer.wait(5) and closer.wait(5)
        assert any(
            "Failed to flush history on finish: flush failed" in w for w in warnings
        )
        assert closed == [True]
    finally:
        release.set()
    assert_drained(history)


def test_completed_notifications_and_old_hooks_are_not_retained(
    store, monkeypatch, tmp_path
):
    history = store()
    tickets, hooks = [], []
    original = history._notify

    def notify(ticket):
        tickets.append(weakref.ref(ticket))
        original(ticket)

    class Callback:
        def __call__(self, row):
            if row["_step"] == 0:
                history.log({"recursive": 1})
            else:
                raise ValueError("broken callback")

    monkeypatch.setattr(history, "_notify", notify)
    for index in range(10):
        hook = Callback()
        hooks.append(weakref.ref(hook))
        history.init(
            project="p", name=f"repeat-{index}", dir=str(tmp_path), on_commit=hook
        )
        history.log({"outer": 0})
        history.finish()
        assert_drained(history)
        assert all(ticket() is None for ticket in tickets)
    del hook
    history.init(project="p", name="last", dir=str(tmp_path))
    gc.collect()
    assert all(hook() is None for hook in hooks)
    assert_drained(history)


@pytest.mark.parametrize(
    "operation", ["log", "span", "flush", "finish", "init", "notifications"]
)
def test_inherited_store_rejects_before_any_notification_or_lifecycle_lock(
    store, monkeypatch, operation
):
    history = store()

    class InheritedLock:
        def __enter__(self):
            pytest.fail("acquired an inherited lock")

        def __exit__(self, *args):
            pass

    class InheritedQueue:
        def drain(self, *args):
            pytest.fail("accessed an inherited native queue")

        def enqueue(self, *args):
            pytest.fail("accessed an inherited native queue")

    calls = {
        "log": lambda: history.log({"value": 1}),
        "span": lambda: history.record_span({}),
        "flush": history.flush,
        "finish": history.finish,
        "init": lambda: history.init(project="child"),
        "notifications": history._deliver_notifications,
    }
    with monkeypatch.context() as patch:
        patch.setattr(history, "_owner_pid", -1)
        patch.setattr(history, "_notify_queue", InheritedQueue())
        for name in (
            "_lock",
            "_write_lock",
            "_lifecycle_lock",
        ):
            patch.setattr(history, name, InheritedLock())
        with pytest.raises(RuntimeError, match="after fork"):
            calls[operation]()


@pytest.mark.benchmark
@pytest.mark.parametrize("span_metrics", [False, True])
def test_slow_callback_does_not_scale_span_latency(
    run, threads, span_metrics, record_property
):
    entered, release = threading.Event(), threading.Event()
    timings = []
    count = 500

    def block(row):
        entered.set()
        assert release.wait(5)

    instance = run(span_metrics=span_metrics)
    instance.history.on_commit = block

    def spans():
        start = time.perf_counter()
        for _ in range(count):
            with et.span("fast"):
                pass
        timings.append(time.perf_counter() - start)

    producer = threads(lambda: et.log({"value": 0}))
    try:
        assert entered.wait(5)
        started = time.perf_counter()
        completed = threads(spans)
        assert completed.wait(1), "span batch stalled behind the callback"
        time.sleep(0.05)
        assert not producer.is_set()
        assert instance.history._next_span_row == count
        assert timings[0] / count < 0.001
        record_property("span_us", timings[0] * 1e6 / count)
        record_property("blocked_callback_ms", (time.perf_counter() - started) * 1e3)
    finally:
        release.set()
    assert producer.wait(5)
