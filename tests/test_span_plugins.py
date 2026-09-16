"""Span print_fn output and the plugin protocol, including the built-ins."""

import asyncio
import contextlib
import json
import math
import sys
import threading
import time
import timeit
import types
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import pytest

import expr_tracker as et
from expr_tracker import plugins as pl
from expr_tracker.history.naming import spans_filename
from expr_tracker.plugins import CpuTime, GpuStats, TorchMemory, stop_samplers
from expr_tracker.spans import COUNT_METRIC, INDENT, TIME_METRIC, span_metric


@pytest.fixture
def run(tmp_path):
    created = []

    def factory(**options):
        created.append(
            et.init(
                project="pg",
                name="r",
                dir=str(tmp_path),
                backends=[],
                resume="never",
                **options,
            )
        )
        return created[-1]

    yield factory
    for _ in created:
        with contextlib.suppress(Exception):
            et.finish()


@pytest.fixture
def lines():
    return []


def depth_of(line):
    """Nesting depth of a printed line, measured from its leading indent."""
    return (len(line) - len(line.lstrip(" "))) // len(INDENT)


def span_records(run_obj):
    run_obj.history.flush()
    path = run_obj.history.log_dir / spans_filename(None, True)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def measured_span(monkeypatch):
    now = 0.0
    monkeypatch.setattr(time, "perf_counter", lambda: now)

    def measure(name, seconds, metrics):
        nonlocal now
        with et.span(name, plugins=[lambda span: metrics]) as span:
            now += seconds
        return span

    return measure


@pytest.fixture
def resource_span():
    def make(started_at=1.0):
        return types.SimpleNamespace(started_at=started_at, duration_ms=1000.0)

    return make


# --------------------------------------------------------------------------- #
# print_fn
# --------------------------------------------------------------------------- #


def test_prints_a_start_and_an_end_line(run, lines):
    run()
    with et.span("forward", print_fn=lines.append):
        pass
    assert len(lines) == 2
    assert lines[0].startswith("-> forward  ")
    assert lines[1].startswith("<- forward  ")
    assert lines[1].endswith("ms")


def test_indents_one_tab_per_level(run, lines):
    run()
    with et.span("a", print_fn=lines.append), et.span("b"), et.span("c"):
        pass
    assert [depth_of(line) for line in lines] == [0, 1, 2, 2, 1, 0]
    assert lines[2].startswith(INDENT * 2 + "-> c")


def test_indentation_is_relative_to_where_printing_started(run, lines):
    """A silent parent must not leave a stray leading tab on its child."""
    run()
    with (
        et.span("outer"),
        et.span("mid"),
        et.span("inner", print_fn=lines.append),
        et.span("leaf"),
    ):
        pass
    assert [depth_of(line) for line in lines] == [0, 1, 1, 0]
    assert lines[0] == lines[0].lstrip(" ")


def test_an_overriding_child_roots_its_own_indentation(run, lines):
    other = []
    run()
    with (
        et.span("a", print_fn=lines.append),
        et.span("b"),
        et.span("c", print_fn=other.append),
        et.span("d"),
    ):
        pass
    assert [depth_of(line) for line in other] == [0, 1, 1, 0]


def test_the_run_default_still_indents_the_whole_tree(run, lines):
    """The default is one shared handler, so nesting must survive it."""
    run(span_print_fn=lines.append)
    with et.span("a"), et.span("b"), et.span("c"):
        pass
    assert [depth_of(line) for line in lines] == [0, 1, 2, 2, 1, 0]


def test_indentation_keeps_up_with_deep_nesting(run, lines):
    run()
    depth = 12
    spans = [et.span("s0", print_fn=lines.append).__enter__()]
    spans += [et.span(f"s{i}").__enter__() for i in range(1, depth)]
    for sp in reversed(spans):
        sp.end()
    assert [depth_of(line) for line in lines[:depth]] == list(range(depth))


def test_a_new_thread_starts_a_fresh_tree(run, lines):
    """Threads do not inherit the stack, so a worker must not be indented."""
    run()
    inner = []

    def work():
        with et.span("worker", print_fn=inner.append):
            pass

    with et.span("main", print_fn=lines.append):
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
    assert [depth_of(line) for line in inner] == [0, 0]


def test_concurrent_tasks_are_each_indented_by_their_own_depth(run, lines):
    run()

    async def leg(name):
        async with et.span(name), et.span("sub"):
            await asyncio.sleep(0.01)

    async def main():
        async with et.span("gather", print_fn=lines.append):
            await asyncio.gather(leg("a"), leg("b"))

    asyncio.run(main())
    by_name = {}
    for line in lines:
        parts = line.split()
        by_name.setdefault(parts[1], []).append(depth_of(line))
    assert by_name["gather"] == [0, 0]
    assert by_name["a"] == [1, 1] and by_name["b"] == [1, 1]
    assert by_name["sub"] == [2, 2, 2, 2]


def test_a_recursive_span_indents_by_recursion_depth(run, lines):
    run()

    @et.span("fib", print_fn=lines.append)
    def fib(n):
        return n if n < 2 else fib(n - 1) + fib(n - 2)

    fib(2)
    assert [depth_of(line) for line in lines] == [0, 1, 1, 1, 1, 0]


def test_a_span_ended_out_of_order_does_not_resurrect_itself(run):
    """Resetting the ContextVar token would restore the whole stack with it."""
    from expr_tracker.spans import _STACK

    run()
    outer = et.start_span("outer")
    inner = et.start_span("inner")
    outer.end()
    assert [sp.name for sp in _STACK.get()] == ["inner"]
    inner.end()
    assert _STACK.get() == ()
    after = et.start_span("after")
    assert after.path == "after"  # not "outer/after"
    after.end()


def test_out_of_order_ends_keep_the_indentation_honest(run, lines):
    run()
    outer = et.start_span("outer", print_fn=lines.append)
    inner = et.start_span("inner")
    outer.end()
    inner.end()
    later = et.start_span("later", print_fn=lines.append)
    later.end()
    assert [depth_of(line) for line in lines] == [0, 1, 0, 1, 0, 0]


def test_a_slashed_name_does_not_fake_a_level(run, lines):
    """`depth` is nesting, not a count of slashes in the path."""
    r = run()
    with et.span("data/load", print_fn=lines.append), et.span("decode"):
        pass
    et.log({"loss": 1.0})
    et.finish()
    assert [depth_of(line) for line in lines] == [0, 1, 1, 0]
    records = {rec["name"]: rec["depth"] for rec in span_records(r)}
    assert records == {"data/load": 0, "data/load/decode": 1}


def test_the_indent_survives_a_log_prefix(run, lines):
    """A tab is measured from the start of the line, so behind a logger prefix
    the first level collapses to whatever is left of the tab stop. Spaces do
    not care where the line begins."""
    run()
    with et.span("a", print_fn=lines.append), et.span("b"), et.span("c"):
        pass
    assert "\t" not in "".join(lines)
    for prefix in ("", "x" * 11, "x" * 71, "x" * 72):
        columns = [_visual_indent(prefix + line) for line in lines]
        relative = [column - len(prefix) for column in columns]
        assert relative == [0, 2, 4, 4, 2, 0], prefix


def _visual_indent(line, tabsize=8):
    """Where the marker lands once a terminal has expanded the tabs."""
    column = 0
    for index, char in enumerate(line):
        if char == "\t":
            column += tabsize - (column % tabsize)
        elif line.startswith(("-> ", "<- "), index):
            return column
        else:
            column += 1
    return column


def test_children_inherit_the_handler(run, lines):
    run()
    with et.span("outer", print_fn=lines.append), et.span("inner"):
        pass
    assert any("inner" in line for line in lines)


def test_a_child_can_override_the_handler(run, lines):
    other = []
    run()
    with (
        et.span("outer", print_fn=lines.append),
        et.span("inner", print_fn=other.append),
        et.span("deepest"),
    ):
        pass
    assert not any("inner" in line for line in lines)
    assert [line.split()[1] for line in other] == [
        "inner",
        "deepest",
        "deepest",
        "inner",
    ]


def test_a_silent_child_stays_silent_under_a_printing_parent(run, lines):
    """A parent handler is inherited, so opting a subtree out needs a no-op."""
    run()
    with (
        et.span("outer", print_fn=lines.append),
        et.span("inner", print_fn=lambda _: None),
    ):
        pass
    assert [line.split()[1] for line in lines] == ["outer", "outer"]


def test_reports_the_duration_it_measured(run, lines):
    run()
    with et.span("slow", print_fn=lines.append):
        time.sleep(0.02)
    measured = float(lines[1].rsplit("  ", 1)[1].removesuffix("ms"))
    assert 15 < measured < 200


def test_marks_a_failed_span(run, lines):
    run()
    with pytest.raises(ValueError), et.span("boom", print_fn=lines.append):
        raise ValueError("x")
    assert lines[1].endswith("!ValueError")


def test_shows_plugin_metrics_on_the_end_line(run, lines):
    run()
    with et.span("m", print_fn=lines.append, plugins=[lambda s: {"hits": 3}]):
        pass
    assert "hits=3" in lines[1]


def test_trims_trailing_zeros_from_floats(run, lines):
    run()
    with et.span(
        "m", print_fn=lines.append, plugins=[lambda s: {"a": 1.5000, "b": 2.0}]
    ):
        pass
    assert "a=1.5" in lines[1] and "b=2 " in lines[1] + " "


def test_a_broken_handler_does_not_break_the_span(run):
    run()

    def explode(line):
        raise RuntimeError("printer is down")

    with et.span("safe", print_fn=explode) as sp:
        pass
    assert sp.duration_ms > 0
    assert span_records(et.get_run())[0]["name"] == "safe"


def test_prints_without_a_run(lines):
    with et.span("orphan", print_fn=lines.append) as sp:
        pass
    assert len(lines) == 2
    assert sp.duration_ms >= 0


def test_the_decorator_takes_a_handler(run, lines):
    run()

    @et.span("work", print_fn=lines.append)
    def work():
        return 7

    assert work() == 7
    assert len(lines) == 2


def test_async_spans_print(run, lines):
    run()

    async def main():
        async with et.span("io", print_fn=lines.append):
            await asyncio.sleep(0)

    asyncio.run(main())
    assert len(lines) == 2


def test_start_span_takes_a_handler(run, lines):
    run()
    sp = et.start_span("manual", print_fn=lines.append)
    sp.end()
    assert len(lines) == 2


def test_the_run_supplies_a_default_handler(run, lines):
    run(span_print_fn=lines.append)
    with et.span("a"):
        pass
    assert len(lines) == 2


def test_a_span_argument_beats_the_run_default(run, lines):
    mine = []
    run(span_print_fn=lines.append)
    with et.span("a", print_fn=mine.append):
        pass
    assert not lines and len(mine) == 2


def test_printing_is_off_by_default(run, capsys):
    run()
    with et.span("quiet"):
        pass
    assert capsys.readouterr().out == ""


def test_sibling_spans_restart_the_indentation(run, lines):
    run()
    with et.span("root", print_fn=lines.append):
        with et.span("one"):
            pass
        with et.span("two"):
            pass
    assert [depth_of(line) for line in lines] == [0, 1, 1, 1, 1, 0]


# --------------------------------------------------------------------------- #
# the plugin protocol
# --------------------------------------------------------------------------- #


class Recorder:
    """A plugin that records its calls and reports a fixed metric."""

    def __init__(self, value=1.0, key="thing"):
        self.value, self.key = value, key
        self.calls = []

    def start(self, span):
        self.calls.append(("start", span.path))

    def end(self, span):
        self.calls.append(("end", span.path))
        return {self.key: self.value}


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


# --------------------------------------------------------------------------- #
# CpuTime
# --------------------------------------------------------------------------- #


def test_cpu_time_sees_a_busy_span_as_busy(run):
    run()
    with et.span("busy", plugins=[CpuTime()]) as sp:
        deadline = time.perf_counter() + 0.05
        while time.perf_counter() < deadline:
            pass
    assert sp.metrics["cpu_percent"] > 60
    assert sp.metrics["cpu_time_ms"] > 20


def test_cpu_time_sees_a_sleeping_span_as_idle(run):
    run()
    with et.span("idle", plugins=[CpuTime()]) as sp:
        time.sleep(0.05)
    assert sp.metrics["cpu_percent"] < 30


def test_cpu_time_aggregates_by_duration_not_call_count(run, monkeypatch):
    now = used = 0.0
    monkeypatch.setattr(time, "perf_counter", lambda: now)
    monkeypatch.setattr(time, "process_time", lambda: used)
    run()
    plugin = CpuTime()
    with et.span("work", plugins=[plugin]):
        now += 1
        used += 0.25
    with et.span("work", plugins=[plugin]):
        now += 3
        used += 2.25
    row = et.history(1)[0]
    assert row["cpu_time_ms/work"] == 2500
    assert row["cpu_percent/work"] == 62.5


def test_cpu_time_nests_without_double_counting(run):
    run()
    plugin = CpuTime()
    with et.span("outer", plugins=[plugin]) as outer, et.span("inner") as inner:
        deadline = time.perf_counter() + 0.03
        while time.perf_counter() < deadline:
            pass
    assert outer.metrics["cpu_time_ms"] >= inner.metrics["cpu_time_ms"] * 0.9


def test_cpu_time_ignores_an_unbalanced_end():
    plugin = CpuTime()
    span = types.SimpleNamespace(duration_ms=1.0, path="a")
    assert plugin.end(span) == {}


def test_cpu_time_survives_a_zero_duration_span():
    plugin = CpuTime()
    span = types.SimpleNamespace(duration_ms=0.0, path="a")
    plugin.start(span)
    assert plugin.end(span)["cpu_percent"] == 0.0


def test_cpu_time_keeps_threads_apart(run):
    run()
    plugin = CpuTime()
    depths = {}

    def worker(name):
        with et.span(name, plugins=[plugin]) as sp:
            time.sleep(0.01)
        depths[name] = sp.metrics

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(depths) == 4
    assert all("cpu_time_ms" in m for m in depths.values())
    assert plugin._active == {}


def test_cpu_time_pairs_non_lifo_spans_by_identity(monkeypatch, resource_span):
    plugin = CpuTime()
    a, b = resource_span(), resource_span()
    assert a == b and a is not b  # unhashable, value-equal objects still differ
    clock = iter([0.0, 1.0, 2.0, 4.0])
    monkeypatch.setattr(time, "process_time", lambda: next(clock))
    plugin.start(a)
    plugin.start(b)
    assert plugin.end(a) == {"cpu_time_ms": 2000.0, "cpu_percent": 200.0}
    assert plugin.end(b) == {"cpu_time_ms": 3000.0, "cpu_percent": 300.0}
    assert plugin.end(a) == {}
    assert plugin._active == {}


def test_cpu_time_pairs_overlapping_async_tasks(run, monkeypatch):
    run()
    now = 1.0
    plugin = CpuTime()
    monkeypatch.setattr(time, "process_time", lambda: now)
    monkeypatch.setattr(time, "perf_counter", lambda: now)

    async def main():
        a_started, b_started, a_ended = (asyncio.Event() for _ in range(3))

        async def first():
            nonlocal now
            async with et.span("a", plugins=[plugin]) as span:
                a_started.set()
                await b_started.wait()
                now = 3.0
            a_ended.set()
            return span.metrics

        async def second():
            nonlocal now
            await a_started.wait()
            now = 2.0
            async with et.span("b", plugins=[plugin]) as span:
                b_started.set()
                await a_ended.wait()
                now = 5.0
            return span.metrics

        return await asyncio.gather(first(), second())

    a, b = asyncio.run(main())
    assert a == {"cpu_time_ms": 2000.0, "cpu_percent": 100.0}
    assert b == {"cpu_time_ms": 3000.0, "cpu_percent": 100.0}
    assert plugin._active == {}


def test_cpu_time_can_end_on_another_thread(monkeypatch, resource_span):
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(time, "process_time", lambda: next(clock))
    plugin, span = CpuTime(), resource_span()
    plugin.start(span)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(plugin.end, span).result(5)["cpu_time_ms"] == 2000.0
    assert plugin._active == {}


@pytest.mark.parametrize("hook", ["start", "end"])
def test_cpu_time_failures_do_not_consume_other_spans(monkeypatch, resource_span, hook):
    plugin = CpuTime()
    a, b = resource_span(), resource_span()
    monkeypatch.setattr(time, "process_time", lambda: 1.0)
    plugin.start(a)
    if hook == "end":
        plugin.start(b)

    def fail():
        raise RuntimeError("clock failed")

    monkeypatch.setattr(time, "process_time", fail)
    with pytest.raises(RuntimeError, match="clock failed"):
        getattr(plugin, hook)(b)
    assert plugin.end(b) == {}
    monkeypatch.setattr(time, "process_time", lambda: 3.0)
    assert plugin.end(a)["cpu_time_ms"] == 2000.0
    assert plugin._active == {}


# --------------------------------------------------------------------------- #
# TorchMemory, against a controllable fake allocator
# --------------------------------------------------------------------------- #


class FakeCuda:
    """A torch.cuda stand-in with the real high-water-mark semantics."""

    def __init__(self, available=True):
        self.allocated = 0
        self.peak = 0
        self.device = 0
        self._available = available

    def current_device(self):
        return self.device

    def is_available(self):
        return self._available

    def allocate(self, n):
        self.allocated += n
        self.peak = max(self.peak, self.allocated)

    def free(self, n):
        self.allocated -= n

    def memory_allocated(self, device=None):
        return self.allocated

    def max_memory_allocated(self, device=None):
        return self.peak

    def reset_peak_memory_stats(self, device=None):
        self.peak = self.allocated


class FakeDevice:
    def __init__(self, device):
        if isinstance(device, FakeDevice):
            self.type, self.index = device.type, device.index
        else:
            self.type, _, index = device.partition(":")
            self.index = int(index) if index else None


class FastFakeCuda(FakeCuda):
    """Exposes the cheap nested-stats call that TorchMemory prefers."""

    def __init__(self, empty=False):
        super().__init__()
        self.empty = empty
        self.nested_calls = 0

    def memory_stats_as_nested_dict(self, device=None):
        self.nested_calls += 1
        if self.empty:
            return {}
        return {
            "allocated_bytes": {"all": {"current": self.allocated, "peak": self.peak}}
        }

    def memory_allocated(self, device=None):
        raise AssertionError("the slow path should not be used")

    def max_memory_allocated(self, device=None):
        raise AssertionError("the slow path should not be used")


@pytest.fixture
def fake_torch(monkeypatch):
    cuda = FakeCuda()
    module = types.ModuleType("torch")
    module.cuda = cuda
    module.device = FakeDevice
    monkeypatch.setitem(sys.modules, "torch", module)
    return cuda


MB = 1048576


@pytest.fixture
def fast_torch(monkeypatch):
    cuda = FastFakeCuda()
    module = types.ModuleType("torch")
    module.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", module)
    return cuda


def test_torch_memory_prefers_the_nested_stats_call(run, fast_torch):
    run()
    with et.span("a", plugins=[TorchMemory()]) as sp:
        fast_torch.allocate(100 * MB)
        fast_torch.free(100 * MB)
    assert sp.metrics["gpu_mem_peak_mb"] == 100.0
    assert fast_torch.nested_calls == 2  # one per hook, not two


def test_torch_memory_falls_back_when_the_stats_are_empty(run, monkeypatch):
    """An uninitialised device returns {}, so the public helpers must answer."""
    cuda = FastFakeCuda(empty=True)
    cuda.memory_allocated = lambda device=None: cuda.allocated
    cuda.max_memory_allocated = lambda device=None: cuda.peak
    module = types.ModuleType("torch")
    module.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", module)
    run()
    with et.span("a", plugins=[TorchMemory()]) as sp:
        cuda.allocate(64 * MB)
        cuda.free(64 * MB)
    assert sp.metrics["gpu_mem_peak_mb"] == 64.0


def test_torch_memory_reports_the_peak_and_the_delta(run, fake_torch):
    run()
    plugin = TorchMemory()
    with et.span("a", plugins=[plugin]) as sp:
        fake_torch.allocate(100 * MB)
        fake_torch.free(60 * MB)
    assert sp.metrics["gpu_mem_peak_mb"] == 100.0
    assert sp.metrics["gpu_mem_delta_mb"] == 40.0


def test_torch_memory_aggregates_repeated_spans_by_peak(run, fake_torch):
    run()
    plugin = TorchMemory()
    for megabytes in (512, 256):
        with et.span("work", plugins=[plugin]):
            fake_torch.allocate(megabytes * MB)
            fake_torch.free(megabytes * MB)
    row = et.history(1)[0]
    assert row["gpu_mem_peak_mb/work"] == 512.0
    assert row["gpu_mem_delta_mb/work"] == 0.0


def test_torch_memory_carries_a_peak_that_predates_a_child(run, fake_torch):
    """A child resets the one global counter; the parent must not lose its peak."""
    run()
    plugin = TorchMemory()
    with et.span("outer", plugins=[plugin]) as outer:
        fake_torch.allocate(512 * MB)
        fake_torch.free(512 * MB)
        with et.span("inner") as inner:
            fake_torch.allocate(8 * MB)
            fake_torch.free(8 * MB)
    assert inner.metrics["gpu_mem_peak_mb"] == 8.0
    assert outer.metrics["gpu_mem_peak_mb"] == 512.0


def test_torch_memory_carries_across_two_levels(run, fake_torch):
    run()
    plugin = TorchMemory()
    with et.span("l1", plugins=[plugin]) as l1:
        fake_torch.allocate(300 * MB)
        fake_torch.free(300 * MB)
        with et.span("l2") as l2:
            fake_torch.allocate(200 * MB)
            fake_torch.free(200 * MB)
            with et.span("l3") as l3:
                fake_torch.allocate(50 * MB)
                fake_torch.free(50 * MB)
    assert (l3.metrics["gpu_mem_peak_mb"], l2.metrics["gpu_mem_peak_mb"]) == (
        50.0,
        200.0,
    )
    assert l1.metrics["gpu_mem_peak_mb"] == 300.0


def test_torch_memory_lets_a_childs_peak_win(run, fake_torch):
    run()
    plugin = TorchMemory()
    with et.span("outer", plugins=[plugin]) as outer:
        fake_torch.allocate(10 * MB)
        fake_torch.free(10 * MB)
        with et.span("inner"):
            fake_torch.allocate(400 * MB)
            fake_torch.free(400 * MB)
    assert outer.metrics["gpu_mem_peak_mb"] == 400.0


def test_torch_memory_counts_memory_already_held(run, fake_torch):
    """max_memory_allocated is absolute, so a resident tensor counts."""
    fake_torch.allocate(256 * MB)
    run()
    plugin = TorchMemory()
    with et.span("a", plugins=[plugin]) as sp:
        fake_torch.allocate(16 * MB)
        fake_torch.free(16 * MB)
    assert sp.metrics["gpu_mem_peak_mb"] == 272.0
    assert sp.metrics["gpu_mem_delta_mb"] == 0.0


def test_torch_memory_keeps_sibling_spans_independent(run, fake_torch):
    run()
    plugin = TorchMemory()
    with et.span("root", plugins=[plugin]):
        with et.span("one") as one:
            fake_torch.allocate(64 * MB)
            fake_torch.free(64 * MB)
        with et.span("two") as two:
            fake_torch.allocate(32 * MB)
            fake_torch.free(32 * MB)
    assert one.metrics["gpu_mem_peak_mb"] == 64.0
    assert two.metrics["gpu_mem_peak_mb"] == 32.0


def test_torch_memory_degrades_when_cuda_is_absent(run, monkeypatch):
    module = types.ModuleType("torch")
    module.cuda = FakeCuda(available=False)
    monkeypatch.setitem(sys.modules, "torch", module)
    run()
    with et.span("a", plugins=[TorchMemory()]) as sp:
        pass
    assert sp.metrics == {} and sp.duration_ms > 0


def test_torch_memory_degrades_without_torch(run, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    run()
    with et.span("a", plugins=[TorchMemory()]) as sp:
        pass
    assert sp.metrics == {}


def test_torch_memory_ignores_an_unbalanced_end(fake_torch):
    assert TorchMemory().end(types.SimpleNamespace(path="a")) == {}


def test_torch_memory_passes_the_device_through(run, fake_torch):
    seen = []
    fake_torch.memory_allocated = lambda device=None: seen.append(device) or 0
    run()
    with et.span("a", plugins=[TorchMemory(device=3)]):
        pass
    assert seen == [3, 3]


@pytest.mark.parametrize("separate_plugins", [False, True])
@pytest.mark.parametrize("end_on_worker", [False, True])
def test_torch_memory_pairs_overlapping_spans(
    fake_torch, resource_span, separate_plugins, end_on_worker
):
    a, b = resource_span(), resource_span()
    first = TorchMemory()
    second = TorchMemory() if separate_plugins else first
    first.start(a)
    fake_torch.allocate(100 * MB)
    second.start(b)
    fake_torch.allocate(50 * MB)
    if end_on_worker:
        with ThreadPoolExecutor(max_workers=1) as pool:
            measured_a = pool.submit(first.end, a).result(5)
    else:
        measured_a = first.end(a)
    fake_torch.allocate(100 * MB)
    measured_b = second.end(b)
    assert measured_a == {"gpu_mem_peak_mb": 150.0, "gpu_mem_delta_mb": 150.0}
    assert measured_b == {"gpu_mem_peak_mb": 250.0, "gpu_mem_delta_mb": 150.0}
    assert first._active == second._active == pl._TORCH_WINDOWS == {}


def test_torch_memory_shares_resets_across_tasks_and_device_aliases(run, fake_torch):
    run()

    async def main():
        a_started, b_started, a_ended = (asyncio.Event() for _ in range(3))

        async def first():
            async with et.span("a", plugins=[TorchMemory()]) as span:
                fake_torch.allocate(512 * MB)
                fake_torch.free(512 * MB)
                a_started.set()
                await b_started.wait()
            a_ended.set()
            return span.metrics

        async def second():
            await a_started.wait()
            async with et.span("b", plugins=[TorchMemory("cuda:0")]) as span:
                fake_torch.allocate(64 * MB)
                fake_torch.free(64 * MB)
                b_started.set()
                await a_ended.wait()
            return span.metrics

        return await asyncio.gather(first(), second())

    a, b = asyncio.run(main())
    assert a["gpu_mem_peak_mb"] == 512.0
    assert b["gpu_mem_peak_mb"] == 64.0
    assert pl._TORCH_WINDOWS == {}


@pytest.mark.parametrize("device", [None, 0, "cuda", "cuda:0", FakeDevice("cuda:0")])
def test_torch_memory_coordinates_device_aliases(fake_torch, resource_span, device):
    a, b = resource_span(), resource_span()
    first, second = TorchMemory(device), TorchMemory(0)
    first.start(a)
    fake_torch.allocate(512 * MB)
    fake_torch.free(512 * MB)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(second.start, b).result(5)
    fake_torch.allocate(64 * MB)
    fake_torch.free(64 * MB)
    assert first.end(a)["gpu_mem_peak_mb"] == 512.0
    assert second.end(b)["gpu_mem_peak_mb"] == 64.0
    assert pl._TORCH_WINDOWS == {}


def test_torch_memory_pins_the_start_device_and_separates_devices(
    fake_torch, resource_span
):
    devices = [FakeCuda(), FakeCuda()]
    fake_torch.memory_allocated = lambda device: devices[device].allocated
    fake_torch.max_memory_allocated = lambda device: devices[device].peak
    fake_torch.reset_peak_memory_stats = lambda device: devices[
        device
    ].reset_peak_memory_stats()
    plugin = TorchMemory()
    a, b = resource_span(), resource_span()
    plugin.start(a)
    devices[0].allocate(512 * MB)
    devices[0].free(512 * MB)
    fake_torch.device = 1
    plugin.start(b)
    devices[1].allocate(32 * MB)
    assert plugin.end(a) == {"gpu_mem_peak_mb": 512.0, "gpu_mem_delta_mb": 0.0}
    assert plugin.end(b) == {"gpu_mem_peak_mb": 32.0, "gpu_mem_delta_mb": 32.0}
    assert pl._TORCH_WINDOWS == {}


def test_torch_memory_serializes_resets_with_other_span_ends(
    fake_torch, monkeypatch, resource_span
):
    a, b = resource_span(), resource_span()
    first, second = TorchMemory(), TorchMemory()
    first.start(a)
    fake_torch.allocate(512 * MB)
    fake_torch.free(512 * MB)
    entered, release, attempted = (threading.Event() for _ in range(3))
    reset = fake_torch.reset_peak_memory_stats

    def blocking_reset(device):
        entered.set()
        assert release.wait(5)
        reset(device)

    def end():
        attempted.set()
        return first.end(a)

    monkeypatch.setattr(fake_torch, "reset_peak_memory_stats", blocking_reset)
    with ThreadPoolExecutor(max_workers=2) as pool:
        starting = pool.submit(second.start, b)
        try:
            assert entered.wait(5)
            ending = pool.submit(end)
            assert attempted.wait(5)
            assert not ending.done()
        finally:
            release.set()
        starting.result(5)
        assert ending.result(5)["gpu_mem_peak_mb"] == 512.0
    fake_torch.allocate(64 * MB)
    fake_torch.free(64 * MB)
    assert second.end(b)["gpu_mem_peak_mb"] == 64.0
    assert pl._TORCH_WINDOWS == {}


@pytest.mark.parametrize("failure", ["read", "reset", "partial_reset", "end"])
def test_torch_memory_cleans_failed_spans_without_losing_other_peaks(
    fake_torch, monkeypatch, resource_span, failure
):
    plugin = TorchMemory()
    a, b = resource_span(), resource_span()
    plugin.start(a)
    fake_torch.allocate(512 * MB)
    fake_torch.free(512 * MB)
    if failure == "end":
        plugin.start(b)

    def fail(device):
        if failure == "partial_reset":
            fake_torch.peak = fake_torch.allocated
        raise RuntimeError("allocator failed")

    method = "reset_peak_memory_stats" if "reset" in failure else "memory_allocated"
    with monkeypatch.context() as patch:
        patch.setattr(fake_torch, method, fail)
        with pytest.raises(RuntimeError, match="allocator failed"):
            (plugin.end if failure == "end" else plugin.start)(b)
    assert plugin.end(b) == {}
    fake_torch.allocate(32 * MB)
    fake_torch.free(32 * MB)
    assert plugin.end(a)["gpu_mem_peak_mb"] == 512.0
    assert plugin._active == pl._TORCH_WINDOWS == {}


# --------------------------------------------------------------------------- #
# GpuStats, against a fake NVML
# --------------------------------------------------------------------------- #


class FakeNvml(types.ModuleType):
    def __init__(self, util=50.0, used=2 * MB, fail=False):
        super().__init__("pynvml")
        self.util, self.used, self.fail = util, used, fail
        self.inits = 0
        self.reads = 0

    def nvmlInit(self):
        self.inits += 1
        if self.fail:
            raise RuntimeError("no driver")

    def nvmlDeviceGetHandleByIndex(self, index):
        return f"handle{index}"

    def nvmlDeviceGetUtilizationRates(self, handle):
        self.reads += 1
        return types.SimpleNamespace(gpu=self.util)

    def nvmlDeviceGetMemoryInfo(self, handle):
        return types.SimpleNamespace(used=self.used)


@pytest.fixture
def fake_nvml(monkeypatch):
    module = FakeNvml()
    monkeypatch.setitem(sys.modules, "pynvml", module)
    monkeypatch.setattr(pl, "_SAMPLERS", {})
    yield module
    stop_samplers()


def test_gpu_stats_averages_the_samples_in_the_window(run, fake_nvml):
    run()
    with et.span("a", plugins=[GpuStats(interval=0.01)]) as sp:
        time.sleep(0.08)
    assert sp.metrics["gpu_percent"] == 50.0
    assert sp.metrics["gpu_mem_used_mb"] == 2.0


def test_gpu_stats_takes_the_peak_memory_not_the_last(run, fake_nvml):
    run()
    plugin = GpuStats(interval=0.01)
    with et.span("a", plugins=[plugin]) as sp:
        time.sleep(0.03)
        fake_nvml.used = 900 * MB
        time.sleep(0.03)
        fake_nvml.used = 1 * MB
        time.sleep(0.03)
    assert sp.metrics["gpu_mem_used_mb"] == 900.0


def test_gpu_stats_falls_back_to_a_live_read_for_a_short_span(run, fake_nvml):
    run()
    with et.span("a", plugins=[GpuStats(interval=10.0)]) as sp:
        pass
    assert sp.metrics["gpu_percent"] == 50.0


def test_gpu_stats_ignores_samples_older_than_the_span(run, fake_nvml):
    """The backward scan must stop at the span start, not average the history."""
    run()
    plugin = GpuStats(interval=0.01)
    with et.span("warm", plugins=[plugin]):
        time.sleep(0.05)
    fake_nvml.util = 99.0
    with et.span("later", plugins=[plugin]) as sp:
        time.sleep(0.05)
    assert sp.metrics["gpu_percent"] == 99.0
    assert len(plugin._sampler.samples) > 5  # older 50.0 samples are still there


def test_only_one_thread_wins_a_concurrent_start(fake_nvml):
    """The winner holds the lock through nvmlInit; the losers must not restart."""
    plugin = GpuStats(interval=0.01)
    inside = threading.Event()
    release = threading.Event()
    real_init = fake_nvml.nvmlInit

    def blocking_init():
        inside.set()
        release.wait(5)
        real_init()

    fake_nvml.nvmlInit = blocking_init
    winner = threading.Thread(target=plugin._sampler.ensure_running)
    winner.start()
    assert inside.wait(5)  # the winner is now parked inside the lock

    losers = [threading.Thread(target=plugin._sampler.ensure_running) for _ in range(4)]
    for t in losers:
        t.start()
    time.sleep(0.05)  # let every loser queue on the lock
    release.set()
    for t in [winner, *losers]:
        t.join(5)
    assert len([t for t in threading.enumerate() if t.name.startswith("et-nvml-")]) == 1


def test_gpu_stats_shares_one_sampler_per_device_and_interval(fake_nvml):
    a, b = GpuStats(0, 0.05), GpuStats(0, 0.05)
    c = GpuStats(1, 0.05)
    assert a._sampler is b._sampler
    assert a._sampler is not c._sampler


def test_gpu_stats_starts_one_thread_only(run, fake_nvml):
    run()
    plugin = GpuStats(interval=0.01)
    for _ in range(5):
        with et.span("a", plugins=[plugin]):
            pass
    alive = [t for t in threading.enumerate() if t.name.startswith("et-nvml-")]
    assert len(alive) == 1


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_gpu_stats_rejects_a_bad_interval(interval):
    with pytest.raises(ValueError):
        GpuStats(interval=interval)


def test_gpu_stats_degrades_without_pynvml(run, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(pl, "_SAMPLERS", {})
    run()
    with et.span("a", plugins=[GpuStats(interval=0.01)]) as sp:
        pass
    assert sp.metrics == {} and sp.duration_ms > 0


def test_gpu_stats_degrades_when_nvml_will_not_start(run, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", FakeNvml(fail=True))
    monkeypatch.setattr(pl, "_SAMPLERS", {})
    run()
    with et.span("a", plugins=[GpuStats(interval=0.01)]) as sp:
        pass
    assert sp.metrics == {}


def test_a_sampler_that_dies_mid_run_does_not_hang(run, fake_nvml):
    run()
    plugin = GpuStats(interval=0.01)
    with et.span("warm", plugins=[plugin]):
        time.sleep(0.03)

    def explode(handle):
        raise RuntimeError("device fell off the bus")

    fake_nvml.nvmlDeviceGetUtilizationRates = explode
    time.sleep(0.05)
    assert not plugin._sampler._thread.is_alive()


def test_stop_samplers_joins_every_thread(fake_nvml):
    GpuStats(interval=0.01)._sampler.ensure_running()
    GpuStats(index=1, interval=0.01)._sampler.ensure_running()
    stop_samplers()
    assert not [t for t in threading.enumerate() if t.name.startswith("et-nvml-")]
    assert pl._SAMPLERS == {}


def test_stop_samplers_is_safe_when_none_are_running(fake_nvml):
    stop_samplers()
    stop_samplers()


def test_a_stopped_sampler_restarts_on_the_next_span(run, fake_nvml):
    run()
    plugin = GpuStats(interval=0.01)
    with et.span("a", plugins=[plugin]):
        pass
    plugin._sampler.stop()
    with et.span("b", plugins=[plugin]):
        time.sleep(0.03)
    assert plugin._sampler._thread.is_alive()


def test_gpu_stats_keeps_the_full_window_after_history_eviction(
    fake_nvml, resource_span
):
    plugin = GpuStats(interval=3600)
    span = resource_span()
    plugin.start(span)
    sampler = plugin._sampler
    window = next(iter(sampler._active.values()))
    for tick in range(1, 10001):
        sampler._record(
            float(tick),
            90.0 if tick <= 100 else 10.0,
            (900 if tick == 1 else 2) * MB,
        )
    assert len(sampler.samples) == sampler.samples.maxlen == 64
    assert len(sampler._active) == 1
    assert next(iter(sampler._active.values())) is window
    assert window.count == 10000 and not hasattr(window, "__dict__")
    assert plugin.end(span) == {"gpu_percent": 10.8, "gpu_mem_used_mb": 900.0}
    assert sampler._active == {}


@pytest.mark.parametrize("separate_plugins", [False, True])
def test_gpu_stats_tracks_concurrent_windows_independently(
    fake_nvml, resource_span, separate_plugins
):
    first = GpuStats(interval=3600)
    second = GpuStats(interval=3600) if separate_plugins else first
    a, b = resource_span(1), resource_span(3)
    first.start(a)
    sampler = first._sampler
    sampler._record(2, 10, 900 * MB)
    second.start(b)
    sampler._record(4, 30, 50 * MB)
    assert first.end(a) == {"gpu_percent": 20.0, "gpu_mem_used_mb": 900.0}
    assert len(sampler._active) == 1
    sampler._record(5, 50, 70 * MB)
    assert second.end(b) == {"gpu_percent": 40.0, "gpu_mem_used_mb": 70.0}
    assert sampler._active == {}
    assert first.end(a) == {} and second.end(b) == {}
    assert fake_nvml.reads == 0


def test_gpu_stats_two_plugins_can_measure_the_same_span(fake_nvml, resource_span):
    a, b = GpuStats(interval=3600), GpuStats(interval=3600)
    span = resource_span()
    a.start(span)
    b.start(span)
    a._sampler._record(2, 70, 12 * MB)
    assert (
        a.end(span)
        == b.end(span)
        == {
            "gpu_percent": 70.0,
            "gpu_mem_used_mb": 12.0,
        }
    )
    assert a._sampler._active == {}


def test_gpu_stats_pairs_overlapping_async_tasks(run, fake_nvml):
    run()
    plugin = GpuStats(interval=3600)

    async def main():
        a_started, b_started, a_ended = (asyncio.Event() for _ in range(3))

        async def first():
            async with et.span("a", plugins=[plugin]) as span:
                plugin._sampler._record(time.time(), 10, 900 * MB)
                a_started.set()
                await b_started.wait()
            a_ended.set()
            return span.metrics

        async def second():
            await a_started.wait()
            async with et.span("b", plugins=[plugin]) as span:
                plugin._sampler._record(time.time(), 30, 50 * MB)
                b_started.set()
                await a_ended.wait()
                plugin._sampler._record(time.time(), 50, 70 * MB)
            return span.metrics

        return await asyncio.gather(first(), second())

    a, b = asyncio.run(main())
    assert a == {"gpu_percent": 20.0, "gpu_mem_used_mb": 900.0}
    assert b == {"gpu_percent": 40.0, "gpu_mem_used_mb": 70.0}
    assert plugin._sampler._active == {}


def test_gpu_stats_excludes_samples_during_start_hooks(run, fake_nvml):
    run()
    plugin = GpuStats(interval=3600)

    class SlowStart:
        def start(self, span):
            plugin._sampler._record(time.time(), 100, 900 * MB)

    with et.span("work", plugins=[plugin, SlowStart()]) as span:
        plugin._sampler._record(time.time(), 20, 5 * MB)
    assert span.metrics == {"gpu_percent": 20.0, "gpu_mem_used_mb": 5.0}


def test_gpu_stats_window_remains_a_bounded_diagnostic_api(fake_nvml):
    sampler = GpuStats(interval=3600)._sampler
    for tick in range(100):
        sampler._record(tick, tick, tick * MB)
    assert sampler.window(97, 98) == [(98, 98, 98 * MB), (97, 97, 97 * MB)]
    assert sampler.window(0, 1) == []


def test_gpu_stats_end_does_not_scan_sample_history(fake_nvml, resource_span):
    class NoScanDeque(deque):
        def __iter__(self):
            raise AssertionError("span metrics must not scan history")

        def __getitem__(self, index):
            raise AssertionError("span metrics must not index history")

    plugin, span = GpuStats(interval=3600), resource_span()
    plugin.start(span)
    plugin._sampler.samples = NoScanDeque(maxlen=64)
    for tick in range(1, 1001):
        plugin._sampler._record(tick, 20, 8 * MB)
    assert plugin.end(span) == {"gpu_percent": 20.0, "gpu_mem_used_mb": 8.0}


@pytest.mark.parametrize("hook", ["start", "end"])
def test_gpu_stats_failure_does_not_consume_other_windows(
    fake_nvml, monkeypatch, resource_span, hook
):
    plugin = GpuStats(interval=3600)
    a, b = resource_span(), resource_span()
    plugin.start(a)
    plugin._sampler._record(2, 10, 900 * MB)
    if hook == "end":
        plugin.start(b)

    def fail():
        raise RuntimeError("NVML failed")

    with monkeypatch.context() as patch:
        patch.setattr(
            plugin._sampler, "_ensure_running" if hook == "start" else "read", fail
        )
        with pytest.raises(RuntimeError, match="NVML failed"):
            getattr(plugin, hook)(b)
    assert plugin.end(b) == {}
    assert plugin.end(a) == {"gpu_percent": 10.0, "gpu_mem_used_mb": 900.0}
    assert plugin._sampler._active == {}


def test_gpu_stats_recovers_from_thread_start_failure(
    fake_nvml, monkeypatch, resource_span
):
    plugin, a, b = GpuStats(interval=3600), resource_span(), resource_span()

    def fail(thread):
        raise RuntimeError("cannot start thread")

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", fail)
        with pytest.raises(RuntimeError, match="cannot start thread"):
            plugin.start(a)
    assert plugin.end(a) == {}
    assert plugin._sampler._active == {}
    plugin.start(b)
    assert plugin.end(b)["gpu_percent"] == 50.0


def test_gpu_stats_failed_initialization_does_not_register_a_window(
    fake_nvml, resource_span
):
    plugin, a, b = GpuStats(interval=3600), resource_span(), resource_span()
    fake_nvml.fail = True
    with pytest.raises(RuntimeError, match="no driver"):
        plugin.start(a)
    assert plugin.end(a) == {}
    assert plugin._sampler._active == {}
    assert plugin._sampler._thread is None and fake_nvml.reads == 0
    fake_nvml.fail = False
    plugin.start(b)
    assert plugin.end(b)["gpu_percent"] == 50.0


def test_gpu_stats_sampling_failure_discards_incomplete_windows(
    fake_nvml, monkeypatch, resource_span
):
    plugin, span = GpuStats(interval=0.001), resource_span()
    armed, entered, release = (threading.Event() for _ in range(3))
    warnings = []

    def fail(handle):
        if not armed.is_set():
            return types.SimpleNamespace(gpu=fake_nvml.util)
        entered.set()
        assert release.wait(5)
        raise RuntimeError("device disappeared")

    with monkeypatch.context() as patch:
        patch.setattr(fake_nvml, "nvmlDeviceGetUtilizationRates", fail)
        patch.setattr(pl.logger, "warning", warnings.append)
        try:
            plugin.start(span)
            armed.set()
            assert entered.wait(5)
        finally:
            release.set()
        plugin._sampler._thread.join(5)
        assert not plugin._sampler._thread.is_alive()
    assert plugin._sampler._active == {}
    assert plugin.end(span) == {}
    assert warnings == ["NVML sampling stopped: device disappeared"]
    later = resource_span()
    plugin.start(later)
    assert plugin.end(later)["gpu_percent"] == 50.0


@pytest.mark.parametrize("boundary", ["start", "end"])
def test_gpu_stats_serializes_samples_with_span_boundaries(
    fake_nvml, monkeypatch, resource_span, boundary
):
    plugin, span = GpuStats(interval=3600), resource_span()
    sampler = plugin._sampler
    sampler.ensure_running()
    if boundary == "end":
        plugin.start(span)
    entered, release, attempted = (threading.Event() for _ in range(3))

    def blocking_read():
        entered.set()
        assert release.wait(5)
        return 10.0, 900 * MB

    def sample():
        with sampler._lock:
            util, used = sampler.read()
            sampler._record(2, util, used)

    def hook():
        attempted.set()
        return getattr(plugin, boundary)(span)

    with ThreadPoolExecutor(max_workers=2) as pool:
        with monkeypatch.context() as patch:
            patch.setattr(sampler, "read", blocking_read)
            reading = pool.submit(sample)
            try:
                assert entered.wait(5)
                changing = pool.submit(hook)
                assert attempted.wait(5)
                assert not changing.done()
            finally:
                release.set()
            reading.result(5)
            measured = changing.result(5)
        if boundary == "start":
            measured = plugin.end(span)
            assert measured == {"gpu_percent": 50.0, "gpu_mem_used_mb": 2.0}
        else:
            assert measured == {"gpu_percent": 10.0, "gpu_mem_used_mb": 900.0}
    assert sampler._active == {}


def test_stop_samplers_waits_for_inflight_reads_and_serializes_restart(
    fake_nvml, monkeypatch, resource_span
):
    plugin, span = GpuStats(interval=0.001), resource_span()
    sampler = plugin._sampler
    armed, entered, release, attempted = (threading.Event() for _ in range(4))

    def blocking_read():
        if not armed.is_set():
            return 50.0, 2 * MB
        entered.set()
        assert release.wait(5)
        return 100.0, 900 * MB

    monkeypatch.setattr(sampler, "read", blocking_read)
    plugin.start(span)
    armed.set()
    old_thread = sampler._thread
    replacement = resource_span()

    def restart():
        attempted.set()
        plugin.start(replacement)

    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            assert entered.wait(5)
            stopping = pool.submit(stop_samplers)
            assert sampler._stop.wait(5)
            assert not stopping.done() and old_thread.is_alive()
            starting = pool.submit(restart)
            assert attempted.wait(5)
            assert not starting.done()
        finally:
            release.set()
        stopping.result(5)
        starting.result(5)
    assert not old_thread.is_alive()
    assert sampler._thread is None and sampler._active == {}
    assert not sampler.samples
    assert plugin.end(span) == {}
    assert plugin._sampler is GpuStats(interval=0.001)._sampler
    assert plugin._sampler is not sampler
    assert plugin.end(replacement)["gpu_percent"] == 50.0


def test_stopping_a_sampler_invalidates_active_windows(fake_nvml, resource_span):
    plugin = GpuStats(interval=3600)
    spans = [resource_span() for _ in range(3)]
    for span in spans:
        plugin.start(span)
    plugin._sampler._record(2, 100, 900 * MB)
    old_thread = plugin._sampler._thread
    plugin._sampler.stop()
    assert not old_thread.is_alive()
    assert plugin._sampler._active == {}
    assert all(plugin.end(span) == {} for span in spans)
    assert fake_nvml.reads == 0
    plugin.start(spans[0])
    assert plugin.end(spans[0]) == {"gpu_percent": 50.0, "gpu_mem_used_mb": 2.0}


@pytest.mark.parametrize("kind", ["cpu", "torch", "gpu"])
def test_resource_plugins_keep_only_active_identity_state(
    fake_torch, fake_nvml, resource_span, kind
):
    plugin = {
        "cpu": CpuTime,
        "torch": TorchMemory,
        "gpu": lambda: GpuStats(interval=3600),
    }[kind]()
    spans = [resource_span() for _ in range(100)]
    for span in spans:
        plugin.start(span)
    active = plugin._sampler._active if kind == "gpu" else plugin._active
    assert len(active) == 100
    for index, span in enumerate(spans):
        assert plugin.end(span)
        assert len(active) == 99 - index
    assert pl._TORCH_WINDOWS == {}


@pytest.mark.parametrize("kind", ["cpu", "torch", "gpu"])
def test_resource_plugins_duplicate_start_does_not_replace_measurement(
    fake_torch, fake_nvml, resource_span, kind
):
    plugin = {
        "cpu": CpuTime,
        "torch": TorchMemory,
        "gpu": lambda: GpuStats(interval=3600),
    }[kind]()
    span = resource_span()
    plugin.start(span)
    active = plugin._sampler._active if kind == "gpu" else plugin._active
    opening = next(iter(active.values()))
    plugin.start(span)
    assert len(active) == 1 and next(iter(active.values())) is opening
    assert plugin.end(span)
    assert plugin.end(span) == {} and active == {}


@pytest.mark.benchmark
@pytest.mark.parametrize("kind", ["cpu", "torch", "gpu"])
def test_resource_plugin_hook_overhead(fake_torch, fake_nvml, resource_span, kind):
    if sys.gettrace() is not None:
        pytest.skip("timing is meaningless under tracing")
    plugin = {
        "cpu": CpuTime,
        "torch": TorchMemory,
        "gpu": lambda: GpuStats(interval=3600),
    }[kind]()
    span = resource_span()

    def measure():
        plugin.start(span)
        plugin.end(span)

    measure()
    times = sorted(timeit.repeat(measure, number=2000, repeat=5))
    per_span = times[len(times) // 2] / 2000
    print(f"\n{kind} resource hooks: {per_span * 1e6:.2f}us/span (fake GPU)")
    assert per_span < 100e-6


# --------------------------------------------------------------------------- #
# the real device, when there is one
# --------------------------------------------------------------------------- #


def has_nvml():
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetCount() > 0
    except Exception:
        return False


def has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


needs_nvml = pytest.mark.skipif(not has_nvml(), reason="no NVIDIA device")
needs_cuda = pytest.mark.skipif(not has_cuda(), reason="no CUDA torch")


@needs_nvml
def test_gpu_stats_reads_a_real_device(run):
    run()
    with et.span("a", plugins=[GpuStats(interval=0.02)]) as sp:
        time.sleep(0.1)
    assert 0 <= sp.metrics["gpu_percent"] <= 100
    assert sp.metrics["gpu_mem_used_mb"] >= 0


@needs_cuda
def test_torch_memory_carries_a_real_peak(run):
    import torch

    run()
    quarter = 64 * 1024 * 1024  # 256 MB of float32
    with et.span("outer", plugins=[TorchMemory()]) as outer:
        big = torch.empty(quarter, device="cuda")
        del big
        torch.cuda.empty_cache()
        with et.span("inner") as inner:
            small = torch.empty(quarter // 32, device="cuda")
            del small
    assert outer.metrics["gpu_mem_peak_mb"] > inner.metrics["gpu_mem_peak_mb"]
    assert outer.metrics["gpu_mem_peak_mb"] >= 256
