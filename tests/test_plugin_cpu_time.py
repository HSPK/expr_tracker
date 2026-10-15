"""CPU-time plugin ownership and measurement."""

import asyncio
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

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
from expr_tracker.plugins import CpuTime


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
    from expr_tracker._native import NativeSpan

    from expr_tracker import spans as spans_module

    monkeypatch.setattr(
        spans_module, "NativeSpan", lambda: NativeSpan(clock=lambda: now)
    )
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
    from expr_tracker._native import NativeSpan

    from expr_tracker import spans as spans_module

    monkeypatch.setattr(
        spans_module, "NativeSpan", lambda: NativeSpan(clock=lambda: now)
    )

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
