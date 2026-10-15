"""Shared fixtures and helpers for the corresponding split regression suite."""

import contextlib
import json
import sys
import types

import pytest

import expr_tracker as et
from expr_tracker import plugins as pl
from expr_tracker.history.naming import spans_filename
from expr_tracker.plugins import stop_samplers
from expr_tracker.spans import INDENT


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
    from expr_tracker._native import NativeSpan

    from expr_tracker import spans as spans_module

    monkeypatch.setattr(
        spans_module, "NativeSpan", lambda: NativeSpan(clock=lambda: now)
    )

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
