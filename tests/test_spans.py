"""Span nesting, aggregation, errors and recorded files."""

import asyncio
import time

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
from expr_tracker.spans import (
    active_path,
)


def test_a_span_records_its_duration_as_a_metric(run):
    instance = run()
    with et.span("forward"):
        time.sleep(0.01)
    values = metrics_of(instance)
    assert values[timed("forward")] >= 9.0
    assert counted("forward") not in values


def test_a_span_returns_its_duration(run):
    run()
    with et.span("forward") as span:
        time.sleep(0.005)
    assert span.duration_ms >= 4.0
    assert span.name == "forward" and span.path == "forward"


@pytest.mark.parametrize("span_count", [False, True])
def test_nested_spans_join_their_names(run, span_count):
    instance = run(span_count=span_count)
    with et.span("forward"):
        with et.span("attention"):
            time.sleep(0.004)
        with et.span("mlp"):
            time.sleep(0.002)
    values = metrics_of(instance)
    paths = ["forward", "forward/attention", "forward/mlp"]
    expected = {timed(path) for path in paths}
    if span_count:
        expected.update(counted(path) for path in paths)
    assert set(values) == expected


def test_a_parent_covers_its_children(run):
    instance = run()
    with et.span("outer"), et.span("inner"):
        time.sleep(0.01)
    values = metrics_of(instance)
    assert values[timed("outer")] >= values[timed("outer/inner")]


def test_three_levels_of_nesting(run):
    instance = run()
    with et.span("a"), et.span("b"), et.span("c"):
        time.sleep(0.002)
    assert timed("a/b/c") in metrics_of(instance)


def test_the_same_name_under_different_parents_stays_distinct(run):
    instance = run()
    with et.span("forward"), et.span("norm"):
        pass
    with et.span("backward"), et.span("norm"):
        pass
    values = metrics_of(instance)
    assert timed("forward/norm") in values
    assert timed("backward/norm") in values


@pytest.mark.parametrize("span_count", [False, True])
@pytest.mark.parametrize("spans", [False, True])
def test_repeated_spans_sum_and_optionally_count(run, span_count, spans):
    instance = run(span_count=span_count, spans=spans)
    for _ in range(4):
        with et.span("layer"):
            time.sleep(0.002)
    values = metrics_of(instance)
    if span_count:
        assert values[counted("layer")] == 4
    else:
        assert counted("layer") not in values
    assert values[timed("layer")] >= 7.0  # summed, not overwritten


def test_aggregation_resets_between_steps(run):
    instance = run(span_count=True)
    for _ in range(3):
        with et.span("layer"):
            pass
    et.log({"loss": 1.0})
    with et.span("layer"):
        pass
    et.log({"loss": 2.0})
    rows = instance.history_query(-1)
    assert rows[0][counted("layer")] == 3
    assert rows[1][counted("layer")] == 1


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
    assert rows[0]["loss"] == 0.5 and timed("forward") in rows[0]


def test_spans_reuse_an_explicit_open_step(run, tmp_path):
    instance = run()
    et.log({"loss": 1.0}, step=10)
    with et.span("outer"), et.span("inner"):
        pass
    assert instance.step == 10
    assert instance.history_query(-1, include_open=False) == []
    et.log({"accuracy": 0.5})
    assert instance.step == 11
    et.finish()

    rows = instance.history_query(-1)
    assert len(rows) == 1 and rows[0]["_step"] == 10
    assert rows[0]["loss"] == 1.0 and rows[0]["accuracy"] == 0.5
    assert timed("outer") in rows[0] and timed("outer/inner") in rows[0]
    assert [record["_step"] for record in spans_of(tmp_path)] == [10, 10]


def test_a_span_uses_the_step_at_end_not_at_start(run, tmp_path):
    instance = run()
    with et.span("outer"):
        with et.span("inner"):
            pass
        et.log({})
    et.finish()

    records = spans_of(tmp_path)
    assert [(record["name"], record["_step"]) for record in records] == [
        ("outer/inner", 0),
        ("outer", 1),
    ]
    rows = instance.history_query(-1)
    assert timed("outer/inner") in rows[0] and timed("outer") not in rows[0]
    assert timed("outer") in rows[1] and timed("outer/inner") not in rows[1]


def test_step_is_an_attribute_not_a_span_step_override(run, tmp_path):
    instance = run()
    with et.span("work", step=99):
        pass
    assert instance.step == 0
    et.finish()
    record = spans_of(tmp_path)[0]
    assert record["_step"] == 0
    assert record["args"] == {"step": 99}


def test_the_decorator_form(run):
    instance = run()

    @et.span("preprocess")
    def preprocess(x):
        time.sleep(0.003)
        return x * 2

    assert preprocess(21) == 42
    assert metrics_of(instance)[timed("preprocess")] >= 2.0


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
    assert metrics_of(instance)[timed("fetch")] >= 4.0


def test_the_async_decorator(run):
    instance = run()

    @et.span("load")
    async def load():
        await asyncio.sleep(0.004)
        return "done"

    assert asyncio.run(load()) == "done"
    assert metrics_of(instance)[timed("load")] >= 3.0


def test_the_manual_form_spans_scopes(run):
    instance = run()
    span = et.start_span("epoch")
    time.sleep(0.004)
    duration = span.end()
    assert duration >= 3.0
    assert metrics_of(instance)[timed("epoch")] >= 3.0


def test_ending_twice_is_harmless(run):
    instance = run(span_count=True)
    span = et.start_span("once")
    first = span.end()
    assert span.end() == first
    assert metrics_of(instance)[counted("once")] == 1


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


@pytest.mark.parametrize("stream", [None, "worker"])
def test_span_timer_flushes_without_committing_metrics(run, tmp_path, stream):
    instance = run(
        stream=stream, buffer_size=1000, buffer_interval=None, max_buffer_seconds=0.02
    )
    with et.span("first"):
        pass
    instance.history.flush()
    writer = instance.history.span_writer
    with et.span("last"):
        pass
    deadline = time.monotonic() + 2
    while writer.lines < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert writer.lines == 2, "the last span stayed buffered while the process was idle"
    assert [record["name"] for record in spans_of(tmp_path, stream=stream)] == [
        "first",
        "last",
    ]
    assert writer.buffer == []
    assert instance.step == 0
    assert instance.history_query(-1, include_open=False) == []
    assert instance.info()["history"]["rows_on_disk"] == 0


def test_span_flush_timer_can_be_disabled(run, tmp_path):
    instance = run(
        buffer_size=1000,
        buffer_interval=None,
        max_buffer_seconds=None,
        span_async_write=False,
    )
    with et.span("first"):
        pass
    with et.span("last"):
        pass
    writer = instance.history.span_writer
    assert len(spans_of(tmp_path)) == 1
    assert len(writer.buffer) == 1
    instance.history.flush()
    assert len(spans_of(tmp_path)) == 2
    assert instance.step == 0


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
    assert timed("forward") in instance.history_query(-1)[0]


def test_spans_follow_the_stream(run, tmp_path):
    run(stream="data", name="r")
    with et.span("produce"):
        pass
    et.log({"rows": 32})
    et.finish()
    assert spans_of(tmp_path, stream="data")[0]["name"] == "produce"
    assert spans_of(tmp_path) == []


def test_an_exception_is_recorded_and_reraised(run, tmp_path):
    instance = run()
    with pytest.raises(ValueError, match="boom"), et.span("risky"):
        raise ValueError("boom")
    et.log({"loss": 1.0})
    et.finish()

    assert spans_of(tmp_path)[0]["error"] == "ValueError"
    assert timed("risky") in metrics_of(instance)


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
