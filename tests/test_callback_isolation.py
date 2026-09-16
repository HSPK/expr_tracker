"""Ordered synchronous callbacks must not hold up unrelated span transactions."""

import ast
import gc
import json
import threading
import time
import weakref
from contextvars import ContextVar

import pytest
from loguru import logger

import expr_tracker as et
from expr_tracker.spans import active_path


@pytest.fixture
def threads():
    started, errors = [], []

    def start(call):
        done = threading.Event()

        def invoke():
            try:
                call()
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=invoke, daemon=True)
        started.append(thread)
        thread.start()
        return done

    yield start
    for thread in started:
        thread.join(5)
    assert not any(thread.is_alive() for thread in started), "thread deadlocked"
    assert not errors


def wait_pending(history, count):
    with history._notification_condition:
        assert history._notification_condition.wait_for(
            lambda: len(history._notifications) == count, timeout=5
        ), "commits did not reach notification delivery"


def assert_drained(history):
    assert not history._notifications
    assert history._notification_owner is None
    assert history._notification_depth == 0
    assert not getattr(history._callback_state, "active", False)
    assert not hasattr(history._callback_state, "pending")


class Backend:
    def __init__(self, callback):
        self.callback = callback
        self.finished = threading.Event()

    def init(self, **kwargs):
        pass

    def log(self, data, step=None, commit=None):
        self.callback()

    def finish(self):
        self.finished.set()


@pytest.mark.parametrize("sink", ["on_commit", "print", "backend"])
@pytest.mark.parametrize("span_metrics", [False, True])
@pytest.mark.parametrize("span_async_write", [False, True])
def test_blocked_callback_does_not_block_parallel_span(
    run, threads, sink, span_metrics, span_async_write
):
    entered, release = threading.Event(), threading.Event()

    def block(*args):
        entered.set()
        assert release.wait(5), "callback was not released"

    backend = Backend(block)
    instance = run(
        backends=[backend] if sink == "backend" else [],
        print_to_screen=sink == "print",
        print_handle=block,
        span_metrics=span_metrics,
        span_async_write=span_async_write,
    )
    history = instance.history
    if sink == "on_commit":
        history.on_commit = block

    def span():
        with et.span("parallel"):
            pass

    producer = threads(lambda: et.log({"loss": 1}))
    try:
        assert entered.wait(5)
        finished_span = threads(span)
        assert finished_span.wait(1), "span waited for an unrelated callback"
        assert not producer.is_set(), "committing caller must wait for its callback"
        assert history._next_span_row == 1
        assert history.current_step == 1
        if span_metrics:
            assert "time_ms/parallel" in history.open_record()
        else:
            assert history.open_record() is None
    finally:
        release.set()
    assert producer.wait(5)
    history.flush()
    assert json.loads(history.spans_fp.read_text())["name"] == "parallel"
    assert_drained(history)


def test_concurrent_commits_notify_in_physical_commit_order(store, threads):
    entered, release = threading.Event(), threading.Event()
    observed, completed = [], []

    def callback(row):
        observed.append((row["_step"], row["value"]))
        if row["value"] == 0:
            entered.set()
            assert release.wait(5)
        completed.append(row["value"])

    history = store(step_policy="allow", on_commit=callback)
    producers = [threads(lambda: history.log({"value": 0}, step=9, commit=True))]
    try:
        assert entered.wait(5)
        for value, step in enumerate([3, 3, 7], 1):
            producers.append(
                threads(
                    lambda value=value, step=step: history.log(
                        {"value": value}, step=step, commit=True
                    )
                )
            )
            wait_pending(history, value)
        assert observed == [(9, 0)]
        assert completed == []
        assert not any(done.is_set() for done in producers)
        assert history.stats()["rows_logged"] == 4
    finally:
        release.set()
    assert all(done.wait(5) for done in producers)
    history.flush()
    rows = [json.loads(line) for line in history.log_fp.read_text().splitlines()]
    assert observed == [(row["_step"], row["value"]) for row in rows]
    assert completed == [0, 1, 2, 3]
    assert_drained(history)


@pytest.mark.parametrize(
    ("condition", "warmup", "queued", "expected_steps", "window", "resume"),
    [
        ("loss > 50", [], [100], [1], 2, False),
        ("loss > 50", [], [100, 0, 100], [1, 3], 2, False),
        ("mean(loss[2]) > 50", [0], [200, 0], [2], 2, False),
        ("mean(loss[4]) > 50", [0, 0, 0], [400, 0, 0, 0], [4], 2, False),
        ("mean(loss[4]) > 50", [0, 0, 0], [400, 0, 0, 0], [4], 4, True),
    ],
    ids=["scalar", "scalar-episodes", "window", "expanded-window", "resumed-window"],
)
def test_run_alerts_only_see_delivered_series(
    run, threads, collector, condition, warmup, queued, expected_steps, window, resume
):
    entered, release = threading.Event(), threading.Event()
    channel, messages = collector
    backend = Backend(lambda: None)
    options = {
        "backends": [backend],
        "alert": {"channels": [channel()]},
        "alert_rules": [f"{condition} => warning: high at {{step}}"],
        "alert_window": window,
    }
    instance = run(**options)
    for loss in warmup:
        instance.log({"loss": loss})
    if resume:
        et.finish()
        backend = Backend(lambda: None)
        instance = run(**{**options, "backends": [backend]})
    history, series = instance.history, instance.history.series
    assert [point[2] for point in series.points("loss")] == warmup
    assert not messages

    def block():
        entered.set()
        assert release.wait(5)

    backend.callback = block
    producers = [threads(lambda: instance.log({"loss": 0}))]
    try:
        assert entered.wait(5)
        for index, loss in enumerate(queued, 1):
            producers.append(threads(lambda loss=loss: instance.log({"loss": loss})))
            wait_pending(history, index)
        assert history.stats()["rows_logged"] == len(warmup) + 1 + len(queued)
        assert [point[2] for point in series.points("loss")] == [*warmup, 0]
        assert not messages
        assert not any(producer.is_set() for producer in producers)
    finally:
        release.set()
    assert all(producer.wait(5) for producer in producers)
    assert [message.text for message in messages] == [
        f"high at {step}" for step in expected_steps
    ]
    assert [message.fields["step"] for message in messages] == expected_steps
    assert history.series is series
    assert series.window == (4 if "loss[4]" in condition else 2)
    assert [point[2] for point in series.points("loss")] == [*warmup, 0, *queued][
        -series.window :
    ]
    instance.log({"loss": 0})
    assert series.latest("loss")[2] == 0
    assert_drained(history)


def test_rejected_rows_never_reach_lagging_alert_series(run, threads, collector):
    entered, release = threading.Event(), threading.Event()
    channel, messages = collector

    def block():
        entered.set()
        assert release.wait(5)

    instance = run(
        backends=[Backend(block)],
        alert={"channels": [channel()]},
        alert_rules=["loss > 50 => warning: high at {step}"],
        max_pending_bytes=4096,
        max_record_bytes=512,
    )
    history = instance.history
    first = threads(lambda: instance.log({"loss": 0}))
    rejected = []
    try:
        assert entered.wait(5)
        oversized = threads(
            lambda: rejected.append(
                history.log({"loss": 999, "rejected": 999, "payload": "x" * 4096})
            )
        )
        assert oversized.wait(1)
        assert rejected == [None]
        assert history.writer.dropped == 1
        assert history.series.names() == {"loss"}
        assert history.series.latest("loss")[2] == 0
        wait_pending(history, 0)
        second = threads(lambda: instance.log({"loss": 100}))
        wait_pending(history, 1)
        assert history.series.latest("loss")[2] == 0
    finally:
        release.set()
    assert first.wait(5) and second.wait(5)
    assert [message.text for message in messages] == ["high at 2"]
    assert [(point[0], point[2]) for point in history.series.points("loss")] == [
        (0, 0),
        (2, 100),
    ]
    assert history.series.names() == {"loss"}
    assert history.stats()["rows_logged"] == 2
    assert_drained(history)


def test_recursive_commit_drains_predecessors_and_releases_completed_waiters(
    store, threads
):
    entered, recurse, nested, release = (threading.Event() for _ in range(4))
    observed, returned, series_at_entry, series_after_recursion = [], [], [], []

    def callback(row):
        observed.append(row["_step"])
        series_at_entry.append([point[0] for point in history.series.points("value")])
        if row["_step"] == 0:
            entered.set()
            assert recurse.wait(5)
            returned.append(history.log({"value": 2}))
            series_after_recursion.extend(history.series.points("value"))
        elif row["_step"] == 2:
            nested.set()
            assert release.wait(5)

    history = store(on_commit=callback)
    first = threads(lambda: history.log({"value": 0}))
    try:
        assert entered.wait(5)
        second = threads(lambda: history.log({"value": 1}))
        wait_pending(history, 1)
        recurse.set()
        assert nested.wait(5)
        assert observed == [0, 1, 2]
        assert second.wait(1), "completed ticket waited for a later callback"
        assert not first.is_set()
        closer = threads(history.finish)
        assert history._closing.wait(5)
        assert not closer.wait(0.05), "finish missed active recursive callbacks"
    finally:
        recurse.set()
        release.set()
    assert first.wait(5) and closer.wait(5)
    assert returned == [2]
    assert series_at_entry == [[0], [0, 1], [0, 1, 2]]
    assert [point[0] for point in series_after_recursion] == [0, 1, 2]
    assert_drained(history)


@pytest.mark.parametrize("sink", ["print", "on_commit"])
def test_migrated_recursive_callbacks_preserve_and_release_producer_context(
    run, threads, monkeypatch, sink
):
    entered, recurse = threading.Event(), threading.Event()
    request = ContextVar("request")
    marker = ContextVar("marker")
    observed, restored, producers, guards = [], [], [], []
    references, producer_threads = [], {}

    class Request:
        def __init__(self, name):
            self.name = name

    class Row(dict):
        pass

    def snapshot():
        return request.get().name, active_path(), marker.get()

    def callback(row):
        step = row["_step"]
        observed.append((step, snapshot(), threading.get_ident()))
        for operation in (history.finish, lambda: history.init(project="forbidden")):
            with pytest.raises(RuntimeError, match="inside a history commit callback"):
                operation()
            guards.append(step)
        marker.set(f"callback:{step}")
        if step == 0:
            entered.set()
            assert recurse.wait(5)
        if step in (0, 1):
            with et.span("nested"):
                et.log({"nested": step})
            history.flush()
        restored.append((step, snapshot()))

    instance = run(
        span_metrics=False,
        print_to_screen=sink == "print",
        print_handle=lambda text: callback(ast.literal_eval(text)),
    )
    history = instance.history
    if sink == "on_commit":
        history.on_commit = callback
    original_close, original_notify = history._close_open_row, history._notify

    def close_row():
        row = original_close()
        return Row(row) if row is not None else None

    def notify(ticket):
        references.extend(
            weakref.ref(value) for value in (ticket, ticket.context, ticket.record)
        )
        original_notify(ticket)

    monkeypatch.setattr(history, "_close_open_row", close_row)
    monkeypatch.setattr(history, "_notify", notify)

    def produce(name):
        value = Request(name)
        references.append(weakref.ref(value))
        request_token = request.set(value)
        marker_token = marker.set(f"producer:{name}")
        producer_threads[name] = threading.get_ident()
        try:
            with et.span(name):
                et.log({"producer": name})
                producers.append(snapshot())
        finally:
            request.reset(request_token)
            marker.reset(marker_token)

    first = threads(lambda: produce("first"))
    try:
        assert entered.wait(5)
        second = threads(lambda: produce("second"))
        wait_pending(history, 1)
    finally:
        recurse.set()
    assert first.wait(5) and second.wait(5)
    history.flush()
    assert producer_threads["first"] != producer_threads["second"]
    assert observed == [
        (0, ("first", "first", "producer:first"), producer_threads["first"]),
        (1, ("second", "second", "producer:second"), producer_threads["first"]),
        (2, ("first", "first/nested", "callback:0"), producer_threads["first"]),
        (3, ("second", "second/nested", "callback:1"), producer_threads["first"]),
    ]
    assert sorted(restored) == [
        (0, ("first", "first", "callback:0")),
        (1, ("second", "second", "callback:1")),
        (2, ("first", "first/nested", "callback:2")),
        (3, ("second", "second/nested", "callback:3")),
    ]
    assert sorted(producers) == [
        ("first", "first", "producer:first"),
        ("second", "second", "producer:second"),
    ]
    assert guards == [0, 0, 1, 1, 2, 2, 3, 3]
    assert len(references) == 14
    gc.collect()
    assert all(reference() is None for reference in references)
    assert_drained(history)


def test_flush_waits_for_existing_callback_without_blocking_its_reentrant_flush(
    store, threads
):
    entered, release = threading.Event(), threading.Event()
    reentered = []

    def callback(row):
        entered.set()
        assert release.wait(5)
        history.flush()
        reentered.append(True)

    history = store(on_commit=callback)
    producer = threads(lambda: history.log({"value": 0}))
    try:
        assert entered.wait(5)
        flushed = threads(history.flush)
        assert not flushed.wait(0.05)
    finally:
        release.set()
    assert producer.wait(5) and flushed.wait(5)
    assert reentered == [True]
    assert_drained(history)


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
        for name in (
            "_lock",
            "_write_lock",
            "_lifecycle_lock",
            "_notification_condition",
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
