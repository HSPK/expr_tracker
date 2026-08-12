"""Span print_fn output and the plugin protocol, including the built-ins."""

import asyncio
import contextlib
import json
import sys
import threading
import time
import types

import pytest

import expr_tracker as et
from expr_tracker import plugins as pl
from expr_tracker.history.naming import spans_filename
from expr_tracker.plugins import CpuTime, GpuStats, TorchMemory, stop_samplers


@pytest.fixture
def run(tmp_path):
    created = []

    def factory(**options):
        options.setdefault("max_open_seconds", None)
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


def span_records(run_obj):
    path = run_obj.history.log_dir / spans_filename(None, True)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    assert [line.count("\t") for line in lines] == [0, 1, 2, 2, 1, 0]
    assert lines[2].startswith("\t\t-> c")


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
    assert [line.count("\t") for line in lines] == [0, 1, 1, 1, 1, 0]


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
    assert row["outer/thing"] == 2.5
    assert row["outer/inner/thing"] == 2.5


def test_plugin_metrics_reach_the_span_file(run):
    r = run()
    with et.span("a", plugins=[Recorder(4.0)]):
        pass
    et.log({"loss": 1.0})
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
    assert et.history(1)[0]["a/n"] == 1


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
    assert row["a/x"] == 1 and row["a/y"] == 2


def test_a_later_plugin_wins_a_key_collision(run):
    run()
    with et.span("a", plugins=[Recorder(1, "x"), Recorder(2, "x")]):
        pass
    et.log({"loss": 1.0})
    assert et.history(1)[0]["a/x"] == 2


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
    assert et.history(1)[0]["a/ok"] == 1


def test_a_plugin_failing_in_end_does_not_break_the_span(run):
    run()

    def broken(span):
        raise RuntimeError("nope")

    with et.span("a", plugins=[broken, Recorder(5, "good")]) as sp:
        pass
    et.log({"loss": 1.0})
    assert sp.duration_ms > 0
    assert et.history(1)[0]["a/good"] == 5


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


def test_repeated_spans_sum_their_plugin_metrics(run):
    """Plugin metrics ride the same accumulate path as duration_ms."""
    run()
    for _ in range(3):
        with et.span("a", plugins=[Recorder(2.0)]):
            pass
    et.log({"loss": 1.0})
    row = et.history(1)[0]
    assert row["a/thing"] == 6.0 and row["a/count"] == 3


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
    assert plugin._stack  # per-thread frames, all balanced back to empty
    assert all(not frames for frames in plugin._stack.values())


# --------------------------------------------------------------------------- #
# TorchMemory, against a controllable fake allocator
# --------------------------------------------------------------------------- #


class FakeCuda:
    """A torch.cuda stand-in with the real high-water-mark semantics."""

    def __init__(self, available=True):
        self.allocated = 0
        self.peak = 0
        self._available = available

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


def test_gpu_stats_rejects_a_bad_interval():
    with pytest.raises(ValueError):
        GpuStats(interval=0)
    with pytest.raises(ValueError):
        GpuStats(interval=-1)


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
