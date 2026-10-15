"""Shared resource-plugin lifecycle, overhead and real devices."""

import sys
import time
import timeit

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
from expr_tracker.plugins import CpuTime, GpuStats, TorchMemory


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
