"""Span concurrency, lifecycle and metric naming."""

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
from _span_helpers import (
    counted as counted,
)
from _span_helpers import (
    group_of as group_of,
)
from _span_helpers import (
    metrics_of as metrics_of,
)
from _span_helpers import (
    run as run,
)
from _span_helpers import (
    spans_of as spans_of,
)
from _span_helpers import (
    timed as timed,
)

import expr_tracker as et
from expr_tracker.history import HistoryStore
from expr_tracker.history.naming import spans_filename
from expr_tracker.spans import (
    COUNT_METRIC,
    TIME_METRIC,
    active_path,
    current_span,
    span_metric,
)


def test_each_thread_keeps_its_own_stack(run):
    instance = run()
    seen: list[str] = []
    ready = threading.Event()

    def worker():
        with et.span("worker"):
            ready.wait(timeout=5)
            seen.append(active_path())

    thread = threading.Thread(target=worker)
    with et.span("main"):
        thread.start()
        time.sleep(0.05)
        seen.append(active_path())
        ready.set()
        thread.join(timeout=5)

    assert sorted(seen) == ["main", "worker"]  # neither nested inside the other
    values = metrics_of(instance)
    assert timed("main") in values and timed("worker") in values


def test_concurrent_tasks_do_not_nest(run):
    instance = run()

    async def work(name):
        async with et.span(name):
            await asyncio.sleep(0.01)
            return active_path()

    async def main():
        return await asyncio.gather(work("a"), work("b"))

    assert sorted(asyncio.run(main())) == ["a", "b"]
    values = metrics_of(instance)
    assert timed("a") in values and timed("b") in values


def test_many_threads_record_every_span(run):
    instance = run(span_count=True)

    def worker(index):
        for _ in range(20):
            with et.span("parallel"):
                pass

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert metrics_of(instance)[counted("parallel")] == 160


@pytest.mark.parametrize(
    ("given", "expected"),
    [("forward", "forward"), ("/forward/", "forward"), ("a/b", "a/b"), ("", "span")],
)
def test_span_names_are_normalised(run, given, expected):
    run()
    with et.span(given) as span:
        pass
    assert span.name == expected


def test_current_span_and_path(run):
    run()
    assert current_span() is None and active_path() == ""
    with et.span("outer") as outer:
        assert current_span() is outer and active_path() == "outer"
        with et.span("inner"):
            assert active_path() == "outer/inner"
        assert current_span() is outer
    assert current_span() is None


def test_a_span_metric_is_usable_in_an_alert_rule(tmp_path):
    """The point of emitting durations as metrics: the DSL already handles them."""
    received: list = []
    et.init(
        project="sp",
        name="alerting",
        dir=str(tmp_path),
        backends=[],
        alert={
            "channels": [
                {
                    "type": "callable",
                    "name": "c",
                    "options": {"handler": received.append},
                    "policy": {"async_send": False, "dedup_window": 0},
                }
            ]
        },
        alert_rules=[timed("forward") + " > 5 => warning: forward is slow"],
    )
    try:
        with et.span("forward"):
            pass
        et.log({"loss": 1.0})
        assert received == []
        with et.span("forward"):
            time.sleep(0.01)
        et.log({"loss": 1.0})
        assert [m.text for m in received] == ["forward is slow"]
    finally:
        et.finish()


def test_record_span_accumulates_without_the_encoder(tmp_path):
    """Span metrics are built from floats here, so they bypass the codec."""
    store = HistoryStore()
    store.init(project="sp", name="direct", dir=str(tmp_path))
    try:
        store.record_span({"a/duration_ms": 1.5, "a/count": 1}, None)
        store.record_span({"a/duration_ms": 2.5, "a/count": 1}, None)
        store.flush(commit_open=True)
        row = store.get(-1)[0]
        assert row["a/duration_ms"] == 4.0 and row["a/count"] == 2
    finally:
        store.finish()


def test_accumulate_replaces_a_non_numeric_value(tmp_path):
    store = HistoryStore()
    store.init(project="sp", name="mixed", dir=str(tmp_path))
    try:
        store.log({"a": "text"}, commit=False)
        store.record_span({"a": 1.0}, None)
        store.flush(commit_open=True)
        assert store.get(-1)[0]["a"] == 1.0  # cannot add to a string, so replace
    finally:
        store.finish()


def test_a_span_record_that_cannot_be_serialised_is_dropped(tmp_path):
    store = HistoryStore()
    store.init(project="sp", name="bad", dir=str(tmp_path))
    try:
        store.record_span({"a/duration_ms": 1.0}, {"name": "a", "args": {1: object()}})
        store.log({"loss": 1.0})
        store.flush(commit_open=True)
        assert store.get(-1)[0]["a/duration_ms"] == 1.0  # the metric still landed
    finally:
        store.finish()


def test_the_spans_file_is_named_beside_the_metrics():
    assert spans_filename(None, rank_aware=False) == "spans.jsonl"
    assert spans_filename("data", rank_aware=False) == "spans.data.jsonl"


def test_reinitialising_closes_the_span_writer(tmp_path):
    store = HistoryStore()
    store.init(project="sp", name="a", dir=str(tmp_path))
    store.record_span({"x/duration_ms": 1.0}, {"name": "x"})
    first = store.span_writer
    store.init(project="sp", name="b", dir=str(tmp_path))
    try:
        assert store.span_writer is not first
        assert (Path(tmp_path) / "sp" / "a" / "spans.jsonl").exists()
    finally:
        store.finish()


@pytest.mark.parametrize("same_run", [False, True])
def test_reinitialising_saves_the_open_row_with_its_original_run(tmp_path, same_run):
    store = HistoryStore()
    seen = []
    store.init(
        project="sp",
        name="a",
        dir=str(tmp_path),
        stream="worker",
        on_commit=lambda row: seen.append(
            (store.name, store.stream, store.log_dir, dict(row))
        ),
        buffer_interval=None,
        max_buffer_seconds=None,
    )
    old_dir, old_metrics, old_spans = store.log_dir, store.log_fp, store.spans_fp
    store.log({"loss": 1.0}, step=10)
    store.record_span({timed("work"): 2.0}, {"name": "work", "dur_ms": 2.0})
    store.record_span({timed("work"): 3.0}, {"name": "work", "dur_ms": 3.0})
    assert seen == []
    store.init(
        project="sp",
        name="a" if same_run else "b",
        dir=str(tmp_path),
        stream="worker" if same_run else None,
    )
    try:
        rows = [json.loads(line) for line in old_metrics.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["_step"] == 10 and rows[0][timed("work")] == 5.0
        assert rows[0]["loss"] == 1.0
        assert seen == [("a", "worker", old_dir, rows[0])]
        assert len(old_spans.read_text().splitlines()) == 2
        assert store.current_step == (11 if same_run else 0)
        if not same_run:
            assert store.get(-1) == []
    finally:
        store.finish()


def test_a_started_span_works_as_a_context_manager(run):
    """`with et.start_span(...)` closes it on exit, like any other guard."""
    instance = run()
    with et.start_span("manual") as span:
        time.sleep(0.003)
    assert span.duration_ms >= 2.0
    assert metrics_of(instance)[timed("manual")] >= 2.0


def test_a_started_span_records_an_exception(run, tmp_path):
    run()
    with pytest.raises(ValueError), et.start_span("manual"):
        raise ValueError("boom")
    et.log({"loss": 1.0})
    et.finish()
    assert spans_of(tmp_path)[0]["error"] == "ValueError"


def test_a_started_span_works_as_an_async_context_manager(run):
    instance = run()

    async def work():
        async with et.start_span("amanual"):
            await asyncio.sleep(0.003)

    asyncio.run(work())
    assert metrics_of(instance)[timed("amanual")] >= 2.0


def test_a_started_span_records_an_async_exception(run, tmp_path):
    run()

    async def work():
        async with et.start_span("afail"):
            raise ValueError("boom")

    with pytest.raises(ValueError):
        asyncio.run(work())
    et.log({"loss": 1.0})
    et.finish()
    assert spans_of(tmp_path)[0]["error"] == "ValueError"


def test_the_async_decorator_records_an_exception(run, tmp_path):
    run()

    @et.span("adecor")
    async def failing():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        asyncio.run(failing())
    et.log({"loss": 1.0})
    et.finish()
    record = spans_of(tmp_path)[0]
    assert record["name"] == "adecor" and record["error"] == "RuntimeError"


def test_the_repr_shows_the_state(run):
    run()
    span = et.start_span("shown")
    assert "shown" in repr(span) and "open" in repr(span)
    span.end()
    assert "ms" in repr(span)


def test_the_track_is_recorded(run, tmp_path):
    run()
    with et.span("outer"), et.span("inner"):
        pass
    et.log({"loss": 1.0})
    et.finish()
    records = spans_of(tmp_path)
    assert len({r["track"] for r in records}) == 1  # a tree shares one track
    assert records[0]["track"] == threading.get_ident()


def test_a_timing_is_named_measurement_first(run):
    instance = run(span_count=True)
    with et.span("forward"), et.span("attention"):
        pass
    values = metrics_of(instance)
    assert "time_ms/forward" in values
    assert "time_ms/forward/attention" in values
    assert "count/forward/attention" in values


def test_timings_never_land_in_a_metric_group_of_yours(run):
    """A span called `train` must not file its duration beside train/loss."""
    instance = run(span_count=True)
    with et.span("train"):
        pass
    instance.log({"train/loss": 1.0, "train/lr": 3e-4})
    instance.history.flush(commit_open=True)
    row = instance.history_query(-1)[-1]
    groups = {}
    for key in row:
        if not key.startswith("_"):
            groups.setdefault(group_of(key), []).append(key)
    assert groups["train"] == ["train/loss", "train/lr"]
    assert groups["time_ms"] == ["time_ms/train"]
    assert groups["count"] == ["count/train"]


def test_every_timing_in_a_run_shares_one_group(run):
    instance = run()
    with et.span("step"):
        with et.span("forward"):
            pass
        with et.span("backward"):
            pass
    values = metrics_of(instance)
    timings = [key for key in values if group_of(key) == "time_ms"]
    assert sorted(timings) == [
        "time_ms/step",
        "time_ms/step/backward",
        "time_ms/step/forward",
    ]


def test_a_plugin_metric_gets_its_own_group_too(run):
    instance = run()
    with et.span("step", plugins=[lambda span: {"gpu_mem_mb": 512.0}]):
        pass
    assert metrics_of(instance)["gpu_mem_mb/step"] == 512.0


def test_the_name_helper_is_what_the_store_records(run):
    """The helper is the contract; nothing should build these names by hand."""
    instance = run(span_count=True)
    with et.span("a/b"):  # a slash in a name is part of the path, not a level
        pass
    values = metrics_of(instance)
    assert span_metric(TIME_METRIC, "a/b") in values
    assert values[span_metric(COUNT_METRIC, "a/b")] == 1
