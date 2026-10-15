"""GPU sampler windows, boundaries and shutdown."""

import asyncio
import sys
import threading
import time
import types
from collections import deque
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
from expr_tracker import plugins as pl
from expr_tracker.plugins import GpuStats, stop_samplers


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
