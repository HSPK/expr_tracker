"""Concurrent and recursive callback delivery."""

import ast
import gc
import json
import threading
import weakref
from contextvars import ContextVar

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

import expr_tracker as et
from expr_tracker.spans import active_path


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
    original_decode, original_notify = history._record_from_json, history._notify

    def close_row(text):
        row = original_decode(text)
        return Row(row) if row is not None else None

    def notify(ticket):
        references.extend(
            weakref.ref(value) for value in (ticket, ticket.context, ticket.record)
        )
        original_notify(ticket)

    monkeypatch.setattr(history, "_record_from_json", close_row)
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
