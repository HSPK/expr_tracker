"""Torch-memory plugin peaks, devices and concurrency."""

import asyncio
import sys
import threading
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
from expr_tracker import plugins as pl
from expr_tracker.plugins import TorchMemory


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
