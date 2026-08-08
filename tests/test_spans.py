"""Spans: nesting, aggregation, the metric surface and the recorded tree."""

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

import expr_tracker as et
from expr_tracker.history import HistoryStore
from expr_tracker.history.naming import spans_filename
from expr_tracker.spans import COUNT_SUFFIX, DURATION_SUFFIX, active_path, current_span


@pytest.fixture
def run(tmp_path):
    created = []

    def factory(**options):
        options.setdefault("max_open_seconds", None)
        created.append(
            et.init(
                project="sp",
                name=options.pop("name", "r"),
                dir=str(tmp_path),
                backends=[],
                **options,
            )
        )
        return created[-1]

    yield factory
    if et.get_run() is not None:
        et.finish()


def metrics_of(instance):
    instance.history.flush(commit_open=True)
    row = instance.history_query(-1)[-1]
    return {k: v for k, v in row.items() if not k.startswith("_")}


def spans_of(tmp_path, name="r", stream=None):
    path = tmp_path / "sp" / name / spans_filename(stream, rank_aware=False)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


# ------------------------------------------------------------------ basics


def test_a_span_records_its_duration_as_a_metric(run):
    instance = run()
    with et.span("forward"):
        time.sleep(0.01)
    values = metrics_of(instance)
    assert values[f"forward/{DURATION_SUFFIX}"] >= 9.0
    assert values[f"forward/{COUNT_SUFFIX}"] == 1


def test_a_span_returns_its_duration(run):
    run()
    with et.span("forward") as span:
        time.sleep(0.005)
    assert span.duration_ms >= 4.0
    assert span.name == "forward" and span.path == "forward"


def test_nested_spans_join_their_names(run):
    instance = run()
    with et.span("forward"):
        with et.span("attention"):
            time.sleep(0.004)
        with et.span("mlp"):
            time.sleep(0.002)
    values = metrics_of(instance)
    assert set(values) == {
        f"forward/{DURATION_SUFFIX}",
        f"forward/{COUNT_SUFFIX}",
        f"forward/attention/{DURATION_SUFFIX}",
        f"forward/attention/{COUNT_SUFFIX}",
        f"forward/mlp/{DURATION_SUFFIX}",
        f"forward/mlp/{COUNT_SUFFIX}",
    }


def test_a_parent_covers_its_children(run):
    instance = run()
    with et.span("outer"), et.span("inner"):
        time.sleep(0.01)
    values = metrics_of(instance)
    assert (
        values[f"outer/{DURATION_SUFFIX}"] >= values[f"outer/inner/{DURATION_SUFFIX}"]
    )


def test_three_levels_of_nesting(run):
    instance = run()
    with et.span("a"), et.span("b"), et.span("c"):
        time.sleep(0.002)
    assert f"a/b/c/{DURATION_SUFFIX}" in metrics_of(instance)


def test_the_same_name_under_different_parents_stays_distinct(run):
    instance = run()
    with et.span("forward"), et.span("norm"):
        pass
    with et.span("backward"), et.span("norm"):
        pass
    values = metrics_of(instance)
    assert f"forward/norm/{DURATION_SUFFIX}" in values
    assert f"backward/norm/{DURATION_SUFFIX}" in values


# ------------------------------------------------------------------ aggregation


def test_repeated_spans_sum_and_count(run):
    instance = run()
    for _ in range(4):
        with et.span("layer"):
            time.sleep(0.002)
    values = metrics_of(instance)
    assert values[f"layer/{COUNT_SUFFIX}"] == 4
    assert values[f"layer/{DURATION_SUFFIX}"] >= 7.0  # summed, not overwritten


def test_aggregation_resets_between_steps(run):
    instance = run()
    for _ in range(3):
        with et.span("layer"):
            pass
    et.log({"loss": 1.0})
    with et.span("layer"):
        pass
    et.log({"loss": 2.0})
    rows = instance.history_query(-1)
    assert rows[0][f"layer/{COUNT_SUFFIX}"] == 3
    assert rows[1][f"layer/{COUNT_SUFFIX}"] == 1


def test_a_span_does_not_commit_a_step(run):
    """Durations ride along with log(); they never create a row of their own."""
    instance = run()
    with et.span("forward"):
        pass
    assert instance.info()["history"]["rows_on_disk"] == 0
    assert instance.step == 0
    et.log({"loss": 1.0})
    assert instance.info()["history"]["rows_on_disk"] == 1


def test_spans_and_metrics_share_one_row(run):
    instance = run()
    with et.span("forward"):
        pass
    et.log({"loss": 0.5})
    rows = instance.history_query(-1)
    assert len(rows) == 1
    assert rows[0]["loss"] == 0.5 and f"forward/{DURATION_SUFFIX}" in rows[0]


# ------------------------------------------------------------------ forms


def test_the_decorator_form(run):
    instance = run()

    @et.span("preprocess")
    def preprocess(x):
        time.sleep(0.003)
        return x * 2

    assert preprocess(21) == 42
    assert metrics_of(instance)[f"preprocess/{DURATION_SUFFIX}"] >= 2.0


def test_the_decorator_preserves_the_function(run):
    run()

    @et.span("named")
    def documented(a, b=2):
        """Doc."""
        return a + b

    assert documented.__name__ == "documented" and documented.__doc__ == "Doc."
    assert documented(1, b=3) == 4


def test_the_async_context_manager(run):
    instance = run()

    async def work():
        async with et.span("fetch"):
            await asyncio.sleep(0.005)

    asyncio.run(work())
    assert metrics_of(instance)[f"fetch/{DURATION_SUFFIX}"] >= 4.0


def test_the_async_decorator(run):
    instance = run()

    @et.span("load")
    async def load():
        await asyncio.sleep(0.004)
        return "done"

    assert asyncio.run(load()) == "done"
    assert metrics_of(instance)[f"load/{DURATION_SUFFIX}"] >= 3.0


def test_the_manual_form_spans_scopes(run):
    instance = run()
    span = et.start_span("epoch")
    time.sleep(0.004)
    duration = span.end()
    assert duration >= 3.0
    assert metrics_of(instance)[f"epoch/{DURATION_SUFFIX}"] >= 3.0


def test_ending_twice_is_harmless(run):
    instance = run()
    span = et.start_span("once")
    first = span.end()
    assert span.end() == first
    assert metrics_of(instance)[f"once/{COUNT_SUFFIX}"] == 1


# ------------------------------------------------------------------ attributes


def test_attributes_reach_the_span_file_not_the_metrics(run, tmp_path):
    instance = run()
    with et.span("load", batch_size=32) as span:
        span.set(rows=128)
    et.log({"loss": 1.0})
    et.finish()

    assert "batch_size" not in metrics_of(instance)
    record = spans_of(tmp_path)[0]
    assert record["args"] == {"batch_size": 32, "rows": 128}


def test_a_span_without_attributes_has_no_args_key(run, tmp_path):
    run()
    with et.span("plain"):
        pass
    et.log({"loss": 1.0})
    et.finish()
    assert "args" not in spans_of(tmp_path)[0]


# ------------------------------------------------------------------ span file


def test_the_span_file_records_the_tree(run, tmp_path):
    run()
    with et.span("forward"), et.span("attention"):
        time.sleep(0.002)
    et.log({"loss": 1.0})
    et.finish()

    records = spans_of(tmp_path)
    assert [r["name"] for r in records] == ["forward/attention", "forward"]
    assert [r["depth"] for r in records] == [1, 0]
    assert all(r["_step"] == 0 for r in records)
    assert all(r["dur_ms"] > 0 and r["start"] > 0 for r in records)


def test_children_close_before_their_parent(run, tmp_path):
    run()
    with et.span("outer"), et.span("inner"):
        pass
    et.log({"loss": 1.0})
    et.finish()
    records = spans_of(tmp_path)
    assert records[0]["name"] == "outer/inner"  # the child is written first


def test_the_span_file_can_be_switched_off(run, tmp_path):
    instance = run(spans=False)
    with et.span("forward"):
        pass
    et.log({"loss": 1.0})
    et.finish()
    assert spans_of(tmp_path) == []
    assert f"forward/{DURATION_SUFFIX}" in instance.history_query(-1)[0]


def test_spans_follow_the_stream(run, tmp_path):
    run(stream="data", name="r")
    with et.span("produce"):
        pass
    et.log({"rows": 32})
    et.finish()
    assert spans_of(tmp_path, stream="data")[0]["name"] == "produce"
    assert spans_of(tmp_path) == []


# ------------------------------------------------------------------ errors


def test_an_exception_is_recorded_and_reraised(run, tmp_path):
    instance = run()
    with pytest.raises(ValueError, match="boom"), et.span("risky"):
        raise ValueError("boom")
    et.log({"loss": 1.0})
    et.finish()

    assert spans_of(tmp_path)[0]["error"] == "ValueError"
    assert f"risky/{DURATION_SUFFIX}" in metrics_of(instance)


def test_the_decorator_re_raises(run, tmp_path):
    run()

    @et.span("failing")
    def failing():
        raise KeyError("nope")

    with pytest.raises(KeyError):
        failing()
    et.log({"loss": 1.0})
    et.finish()
    assert spans_of(tmp_path)[0]["error"] == "KeyError"


def test_an_exception_unwinds_the_stack(run):
    run()
    with pytest.raises(ValueError), et.span("outer"), et.span("inner"):
        raise ValueError
    assert active_path() == ""  # both spans left the stack


def test_a_recording_failure_does_not_break_the_caller(run, monkeypatch):
    instance = run()
    monkeypatch.setattr(
        instance.history,
        "record_span",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sink down")),
    )
    with et.span("forward"):
        pass
    et.log({"loss": 1.0})
    assert len(instance.history_query(-1)) == 1


def test_spans_without_a_run_are_harmless():
    assert et.get_run() is None
    with et.span("orphan") as span:
        time.sleep(0.002)
    assert span.duration_ms >= 1.0


def test_a_span_open_at_finish_still_records(run, tmp_path):
    run()
    span = et.start_span("unfinished")
    et.finish()
    span.end()  # after the store closed: must not raise
    assert isinstance(span.duration_ms, float)


# ------------------------------------------------------------------ concurrency


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
    assert f"main/{DURATION_SUFFIX}" in values and f"worker/{DURATION_SUFFIX}" in values


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
    assert f"a/{DURATION_SUFFIX}" in values and f"b/{DURATION_SUFFIX}" in values


def test_many_threads_record_every_span(run):
    instance = run()

    def worker(index):
        for _ in range(20):
            with et.span("parallel"):
                pass

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert metrics_of(instance)[f"parallel/{COUNT_SUFFIX}"] == 160


# ------------------------------------------------------------------ names


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
        max_open_seconds=None,
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
        alert_rules=[f"forward/{DURATION_SUFFIX} > 5 => warning: forward is slow"],
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


# ------------------------------------------------------------------ store level


def test_record_span_accumulates_without_the_encoder(tmp_path):
    """Span metrics are built from floats here, so they bypass the codec."""
    store = HistoryStore()
    store.init(project="sp", name="direct", dir=str(tmp_path), max_open_seconds=None)
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
    store.init(project="sp", name="mixed", dir=str(tmp_path), max_open_seconds=None)
    try:
        store.log({"a": "text"}, commit=False)
        store.record_span({"a": 1.0}, None)
        store.flush(commit_open=True)
        assert store.get(-1)[0]["a"] == 1.0  # cannot add to a string, so replace
    finally:
        store.finish()


def test_a_span_record_that_cannot_be_serialised_is_dropped(tmp_path):
    store = HistoryStore()
    store.init(project="sp", name="bad", dir=str(tmp_path), max_open_seconds=None)
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
    store.init(project="sp", name="a", dir=str(tmp_path), max_open_seconds=None)
    store.record_span({"x/duration_ms": 1.0}, {"name": "x"})
    first = store.span_writer
    store.init(project="sp", name="b", dir=str(tmp_path), max_open_seconds=None)
    try:
        assert store.span_writer is not first
        assert (Path(tmp_path) / "sp" / "a" / "spans.jsonl").exists()
    finally:
        store.finish()
