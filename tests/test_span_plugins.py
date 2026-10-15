"""Span plugin protocol and metric reductions."""

import math
import time

import pytest
from _span_plugin_helpers import (
    MB as MB,
)
from _span_plugin_helpers import (
    FakeCuda as FakeCuda,
)
from _span_plugin_helpers import (
    FakeDevice as FakeDevice,
)
from _span_plugin_helpers import (
    FakeNvml as FakeNvml,
)
from _span_plugin_helpers import (
    FastFakeCuda as FastFakeCuda,
)
from _span_plugin_helpers import (
    Recorder as Recorder,
)
from _span_plugin_helpers import (
    _visual_indent as _visual_indent,
)
from _span_plugin_helpers import (
    depth_of as depth_of,
)
from _span_plugin_helpers import (
    fake_nvml as fake_nvml,
)
from _span_plugin_helpers import (
    fake_torch as fake_torch,
)
from _span_plugin_helpers import (
    fast_torch as fast_torch,
)
from _span_plugin_helpers import (
    has_cuda as has_cuda,
)
from _span_plugin_helpers import (
    has_nvml as has_nvml,
)
from _span_plugin_helpers import (
    lines as lines,
)
from _span_plugin_helpers import (
    measured_span as measured_span,
)
from _span_plugin_helpers import (
    needs_cuda as needs_cuda,
)
from _span_plugin_helpers import (
    needs_nvml as needs_nvml,
)
from _span_plugin_helpers import (
    resource_span as resource_span,
)
from _span_plugin_helpers import (
    run as run,
)
from _span_plugin_helpers import (
    span_records as span_records,
)

import expr_tracker as et
from expr_tracker.spans import COUNT_METRIC, TIME_METRIC, span_metric


def test_a_plugin_sees_start_then_end(run):
    run()
    plugin = Recorder()
    with et.span("a", plugins=[plugin]):
        pass
    assert plugin.calls == [("start", "a"), ("end", "a")]


def test_plugin_metrics_land_under_the_span_path(run):
    run()
    with et.span("outer", plugins=[Recorder(2.5)]), et.span("inner"):
        pass
    et.log({"loss": 1.0})
    row = et.history(1)[0]
    assert row[span_metric("thing", "outer")] == 2.5
    assert row[span_metric("thing", "outer/inner")] == 2.5


def test_plugin_metrics_reach_the_span_file(run):
    r = run()
    with et.span("a", plugins=[Recorder(4.0)]):
        pass
    et.log({"loss": 1.0})
    et.finish()
    assert span_records(r)[0]["metrics"] == {"thing": 4.0}


def test_a_plain_callable_is_an_end_only_plugin(run):
    seen = []

    def measure(span):
        seen.append(span.duration_ms)
        return {"n": 1}

    run()
    with et.span("a", plugins=[measure]):
        pass
    et.log({"loss": 1.0})
    assert len(seen) == 1 and seen[0] > 0
    assert et.history(1)[0][span_metric("n", "a")] == 1


def test_children_inherit_plugins(run):
    plugin = Recorder()
    run()
    with et.span("outer", plugins=[plugin]), et.span("inner"):
        pass
    assert [path for _, path in plugin.calls] == [
        "outer",
        "outer/inner",
        "outer/inner",
        "outer",
    ]


def test_a_child_can_override_the_plugins(run):
    outer, inner = Recorder(key="o"), Recorder(key="i")
    run()
    with et.span("outer", plugins=[outer]), et.span("inner", plugins=[inner]):
        pass
    assert [path for _, path in outer.calls] == ["outer", "outer"]
    assert [path for _, path in inner.calls] == ["outer/inner", "outer/inner"]


def test_the_run_supplies_default_plugins(run):
    plugin = Recorder()
    run(span_plugins=[plugin])
    with et.span("a"):
        pass
    assert len(plugin.calls) == 2


def test_span_plugins_beat_the_run_default(run):
    default, mine = Recorder(key="d"), Recorder(key="m")
    run(span_plugins=[default])
    with et.span("a", plugins=[mine]):
        pass
    assert not default.calls and len(mine.calls) == 2


def test_several_plugins_all_contribute(run):
    run()
    with et.span("a", plugins=[Recorder(1, "x"), Recorder(2, "y")]):
        pass
    et.log({"loss": 1.0})
    row = et.history(1)[0]
    assert row[span_metric("x", "a")] == 1 and row[span_metric("y", "a")] == 2


def test_a_later_plugin_wins_a_key_collision(run):
    run()
    with et.span("a", plugins=[Recorder(1, "x"), Recorder(2, "x")]):
        pass
    et.log({"loss": 1.0})
    assert et.history(1)[0][span_metric("x", "a")] == 2


def test_a_plugin_failing_in_start_does_not_break_the_span(run):
    class Broken:
        def start(self, span):
            raise RuntimeError("nope")

        def end(self, span):
            return {"ok": 1}

    run()
    with et.span("a", plugins=[Broken()]) as sp:
        pass
    et.log({"loss": 1.0})
    assert sp.duration_ms > 0
    assert et.history(1)[0][span_metric("ok", "a")] == 1


def test_a_plugin_failing_in_end_does_not_break_the_span(run):
    run()

    def broken(span):
        raise RuntimeError("nope")

    with et.span("a", plugins=[broken, Recorder(5, "good")]) as sp:
        pass
    et.log({"loss": 1.0})
    assert sp.duration_ms > 0
    assert et.history(1)[0][span_metric("good", "a")] == 5


def test_a_repeatedly_failing_plugin_warns_once(run, monkeypatch):
    """A plugin that fails once fails every span; it must not flood the log."""
    from expr_tracker import spans as spans_mod

    monkeypatch.setattr(spans_mod, "_WARNED", set())
    warnings = []
    handle = spans_mod.logger.add(lambda m: warnings.append(m), level="WARNING")
    try:
        run()

        def broken(span):
            raise RuntimeError("always")

        for _ in range(50):
            with et.span("a", plugins=[broken]):
                pass
    finally:
        spans_mod.logger.remove(handle)
    assert len(warnings) == 1


def test_the_warning_cache_cannot_grow_without_bound(monkeypatch):
    from expr_tracker import spans as spans_mod

    monkeypatch.setattr(spans_mod, "_WARNED", set())
    for i in range(600):
        spans_mod._warn_once("P", "end", f"failure {i}")
    assert len(spans_mod._WARNED) <= 257


@pytest.mark.parametrize("span_count", [False, True])
def test_a_plugin_cannot_overwrite_the_span_timing(run, span_count):
    """time_ms and count are the span's contract, not a plugin's to set."""
    run(span_count=span_count)
    with et.span(
        "a", plugins=[lambda s: {TIME_METRIC: 999.0, COUNT_METRIC: 42, "ok": 7}]
    ) as sp:
        pass
    et.log({"loss": 1.0})
    row = et.history(1)[0]
    assert row[span_metric(TIME_METRIC, "a")] == sp.duration_ms != 999.0
    if span_count:
        assert row[span_metric(COUNT_METRIC, "a")] == 1
    else:
        assert span_metric(COUNT_METRIC, "a") not in row
    assert row[span_metric("ok", "a")] == 7  # the plugin's own metrics still land
    assert TIME_METRIC not in sp.metrics
    assert COUNT_METRIC not in sp.metrics


def test_a_span_that_never_began_records_nothing(run):
    """Constructing a Span without begin() must not invent an epoch duration."""
    from expr_tracker.spans import Span

    r = run()
    assert Span("orphan", r.history).end() == 0.0
    et.log({"loss": 1.0})
    assert not any(key.startswith("orphan") for key in et.history(1)[0])
    et.finish()
    assert span_records(r) == []


def test_a_broken_print_handler_warns_once(run, monkeypatch):
    from expr_tracker import spans as spans_mod

    monkeypatch.setattr(spans_mod, "_WARNED", set())
    warnings = []
    handle = spans_mod.logger.add(lambda m: warnings.append(m), level="WARNING")
    try:
        run()

        def broken(line):
            raise RuntimeError("printer is down")

        for _ in range(20):
            with et.span("a", print_fn=broken):
                pass
    finally:
        spans_mod.logger.remove(handle)
    assert len(warnings) == 1


def test_ending_twice_does_not_run_the_hooks_again(run, lines):
    plugin = Recorder()
    run()
    sp = et.start_span("a", print_fn=lines.append, plugins=[plugin])
    first = sp.end()
    assert sp.end() == first
    assert len(plugin.calls) == 2
    assert len(lines) == 2


def test_a_plugin_returning_a_non_dict_is_ignored(run):
    run()
    with et.span("a", plugins=[lambda span: 42]) as sp:
        pass
    assert sp.metrics == {}


def test_a_plugin_with_neither_hook_is_ignored(run):
    run()
    with et.span("a", plugins=[object()]) as sp:
        pass
    assert sp.metrics == {}


def test_plugins_run_without_a_run():
    plugin = Recorder()
    with et.span("orphan", plugins=[plugin]) as sp:
        pass
    assert len(plugin.calls) == 2
    assert sp.metrics == {"thing": 1.0}


def test_a_plugin_sees_the_duration_at_end(run):
    seen = {}

    def measure(span):
        seen["d"] = span.duration_ms
        return {}

    run()
    with et.span("a", plugins=[measure]):
        time.sleep(0.01)
    assert seen["d"] > 5


@pytest.mark.parametrize("span_count", [False, True])
def test_repeated_spans_sum_their_plugin_metrics(run, span_count):
    """Plugin metrics ride the same accumulate path as duration_ms."""
    run(span_count=span_count)
    for _ in range(3):
        with et.span("a", plugins=[Recorder(2.0)]):
            pass
    et.log({"loss": 1.0})
    row = et.history(1)[0]
    assert row[span_metric("thing", "a")] == 6.0
    if span_count:
        assert row[span_metric(COUNT_METRIC, "a")] == 3
    else:
        assert span_metric(COUNT_METRIC, "a") not in row


@pytest.mark.parametrize("spans", [False, True])
@pytest.mark.parametrize("span_count", [False, True])
def test_plugin_gauges_keep_their_mean_and_peak_semantics(
    run, measured_span, spans, span_count
):
    instance = run(spans=spans, span_count=span_count)
    first = {
        "cpu_percent": 25.0,
        "gpu_percent": 60.0,
        "gpu_mem_peak_mb": 512.0,
        "gpu_mem_used_mb": 4096.0,
        "cpu_time_ms": 250.0,
        "gpu_mem_delta_mb": 64.0,
        "items": 2,
    }
    second = {
        "cpu_percent": 75.0,
        "gpu_percent": 20.0,
        "gpu_mem_peak_mb": 256.0,
        "gpu_mem_used_mb": 2048.0,
        "cpu_time_ms": 2250.0,
        "gpu_mem_delta_mb": -32.0,
        "items": 7,
    }
    a = measured_span("train/work", 1, first)
    b = measured_span("train/work", 3, second)
    measured_span("eval/work", 2, {"gpu_percent": 90.0, "gpu_mem_peak_mb": 128.0})
    et.log({})
    row = et.history(1)[0]
    assert row["time_ms/train/work"] == 4000
    assert row["cpu_percent/train/work"] == 62.5
    assert row["gpu_percent/train/work"] == 30.0
    assert row["gpu_mem_peak_mb/train/work"] == 512.0
    assert row["gpu_mem_used_mb/train/work"] == 4096.0
    assert row["cpu_time_ms/train/work"] == 2500.0
    assert row["gpu_mem_delta_mb/train/work"] == 32.0
    assert row["items/train/work"] == 9
    assert row["gpu_percent/eval/work"] == 90.0
    assert row["gpu_mem_peak_mb/eval/work"] == 128.0
    if span_count:
        assert row["count/train/work"] == 2
    else:
        assert "count/train/work" not in row
    assert a.metrics == first and b.metrics == second
    et.finish()
    if spans:
        assert [record["metrics"] for record in span_records(instance)][:2] == [
            first,
            second,
        ]


def test_weighted_means_ignore_spans_without_the_measurement(run, measured_span):
    run()
    measured_span("work", 1, {"gpu_percent": 20})
    measured_span("work", 10, {})
    measured_span("work", 3, {"gpu_percent": 80})
    row = et.history(1)[0]
    assert row["time_ms/work"] == 14000
    assert row["gpu_percent/work"] == 65


def test_zero_duration_spans_do_not_bias_a_weighted_mean(run, measured_span):
    run()
    measured_span("work", 0, {"gpu_percent": 10})
    measured_span("work", 0, {"gpu_percent": 20})
    assert et.history(1)[0]["gpu_percent/work"] == 20
    measured_span("work", 2, {"gpu_percent": 40})
    measured_span("work", 0, {"gpu_percent": 100})
    assert et.history(1)[0]["gpu_percent/work"] == 40


def test_weighted_means_reset_on_commit_and_explicit_overwrite(run, measured_span):
    instance = run()
    measured_span("work", 1, {"gpu_percent": 10})
    et.log({})
    assert instance.history._open_weights == {}
    measured_span("work", 3, {"gpu_percent": 90})
    assert et.history(1)[0]["gpu_percent/work"] == 90
    et.log({"gpu_percent/work": 15}, commit=False)
    measured_span("work", 1, {"gpu_percent": 30})
    assert et.history(1)[0]["gpu_percent/work"] == 30
    et.log({}, step=5)
    measured_span("work", 1, {"gpu_percent": 70})
    assert et.history(1)[0]["gpu_percent/work"] == 70
    et.finish()
    assert instance.history._open_weights == {}


@pytest.mark.parametrize("values", [(float("nan"), 512.0), (512.0, float("nan"))])
def test_peak_aggregation_preserves_nan(run, measured_span, values):
    run()
    for value in values:
        measured_span("work", 1, {"gpu_mem_peak_mb": value})
    assert math.isnan(et.history(1)[0]["gpu_mem_peak_mb/work"])


def test_plugin_metrics_reach_the_trace(run, tmp_path):
    from expr_tracker.trace import build_trace

    r = run()
    with et.span("a", plugins=[Recorder(3.0)]):
        pass
    et.log({"loss": 1.0})
    et.finish()
    events = build_trace(r.history.log_dir)["traceEvents"]
    complete = [e for e in events if e.get("ph") == "X"]
    assert complete[0]["args"]["thing"] == 3.0
