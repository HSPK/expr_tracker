"""Span plugins: measure a resource across a span and attach it to the record.

A plugin is any object with ``start(span)`` and ``end(span) -> dict``; a plain
``fn(span) -> dict`` works as an end-only plugin. Whatever ``end`` returns is
merged into the span's metrics under the span's own path, so a plugin returning
``{"cpu_percent": 91.2}`` on a ``forward`` span writes ``forward/cpu_percent``.

Plugins are opt-in because they are not free. A bare span costs ~13us; see each
plugin's docstring for what it adds.
"""

from __future__ import annotations

import atexit
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from loguru import logger

from ._process import check_process, process_id, register_fork_reset

__all__ = ["CpuTime", "GpuStats", "TorchMemory", "stop_samplers"]


class CpuTime:
    """CPU utilisation over the span, from :func:`time.process_time`.

    Reports ``cpu_percent`` (process CPU time / wall time, so >100% means the
    span used more than one core) and ``cpu_time_ms``. Exact, not sampled, with
    one process-clock read per hook and state only for active spans. Counts
    every thread in the process, so concurrent work outside the span inflates it.
    """

    __slots__ = ("_active", "_lock", "_owner_pid")

    def __init__(self):
        self._owner_pid = process_id()
        self._active: dict[int, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def start(self, span) -> None:
        check_process(self._owner_pid)
        with self._lock:
            if id(span) not in self._active:
                self._active[id(span)] = (span, time.process_time())

    def end(self, span) -> dict[str, float]:
        check_process(self._owner_pid)
        with self._lock:
            opening = self._active.pop(id(span), None)
            if opening is None:
                return {}
            used_ms = (time.process_time() - opening[1]) * 1000.0
        percent = 100.0 * used_ms / span.duration_ms if span.duration_ms > 0 else 0.0
        return {"cpu_time_ms": round(used_ms, 3), "cpu_percent": round(percent, 1)}


@dataclass(slots=True)
class _TorchWindow:
    span: Any
    cuda: Any
    device: int
    opening: float
    peak: float


_TORCH_WINDOWS: dict[tuple[int, int], dict[int, _TorchWindow]] = {}
_TORCH_LOCK = threading.Lock()


class TorchMemory:
    """Peak CUDA memory the allocator held during the span, via torch.

    Reports ``gpu_mem_peak_mb`` and ``gpu_mem_delta_mb`` (end minus start, i.e.
    what the span kept). Exact, ~10us per span.

    ``torch.cuda.max_memory_allocated`` is one global high-water mark, so a
    naive reset would let one span erase another's history. Active spans share
    per-device peak accounting across plugin instances, threads and tasks.
    Each reset first carries the old peak into every existing span. Resets made
    outside this plugin cannot be accounted for.
    """

    __slots__ = ("_active", "_device", "_owner_pid", "_torch")

    def __init__(self, device: Any = None):
        self._owner_pid = process_id()
        self._device = device
        self._torch = None
        self._active: dict[int, _TorchWindow] = {}

    def _cuda(self):
        if self._torch is None:
            import torch  # raised to _safely() and logged once per span otherwise

            self._torch = torch
        if not self._torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return self._torch.cuda

    def _device_index(self, cuda) -> int:
        device = self._device
        if device is not None and not isinstance(device, int):
            device = self._torch.device(device)
            if device.type != "cuda":
                raise ValueError("TorchMemory requires a CUDA device")
            device = device.index
        return cuda.current_device() if device is None else device

    def _read(self, cuda, device: int) -> tuple[float, float]:
        """(allocated, peak) in bytes.

        One nested-stats call costs ~17us; the two public helpers rebuild and
        flatten the whole stats dict separately and cost ~90us each.
        """
        nested = getattr(cuda, "memory_stats_as_nested_dict", None)
        if nested is not None:
            stats = nested(device=device).get("allocated_bytes")
            if stats:
                return float(stats["all"]["current"]), float(stats["all"]["peak"])
        return (
            float(cuda.memory_allocated(device)),
            float(cuda.max_memory_allocated(device)),
        )

    def start(self, span) -> None:
        check_process(self._owner_pid)
        with _TORCH_LOCK:
            if id(span) in self._active:
                return
            cuda = self._cuda()
            device = self._device_index(cuda)
            opening, raw_peak = self._read(cuda, device)
            key = (id(cuda), device)
            for window in _TORCH_WINDOWS.get(key, {}).values():
                window.peak = max(window.peak, raw_peak)
            cuda.reset_peak_memory_stats(device)
            window = _TorchWindow(span, cuda, device, opening, opening)
            _TORCH_WINDOWS.setdefault(key, {})[id(window)] = window
            self._active[id(span)] = window

    def end(self, span) -> dict[str, float]:
        check_process(self._owner_pid)
        with _TORCH_LOCK:
            window = self._active.pop(id(span), None)
            if window is None:
                return {}
            key = (id(window.cuda), window.device)
            windows = _TORCH_WINDOWS[key]
            del windows[id(window)]
            if not windows:
                del _TORCH_WINDOWS[key]
            closing, raw_peak = self._read(window.cuda, window.device)
            peak = max(raw_peak, window.peak)
        return {
            "gpu_mem_peak_mb": round(peak / 1048576, 3),
            "gpu_mem_delta_mb": round((closing - window.opening) / 1048576, 3),
        }


@dataclass(slots=True)
class _NvmlWindow:
    span: Any
    count: int = 0
    total: float = 0.0
    peak: float = 0.0


class _NvmlSampler:
    """One background thread polling NVML, shared by every :class:`GpuStats`.

    Sampling in the background is the only honest way to report utilisation:
    NVML's counter is a rolling average over its own window, so reading it once
    at the end of a 3ms span says nothing about that span.
    """

    def __init__(self, index: int, interval: float):
        self._owner_pid = process_id()
        self.index = index
        self.interval = interval
        self.samples: deque[tuple[float, float, float]] = deque(
            maxlen=max(64, int(60.0 / interval))
        )
        self._handle = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._active: dict[tuple[GpuStats, int], _NvmlWindow] = {}

    def handle(self):
        check_process(self._owner_pid)
        with self._lock:
            if self._handle is None:
                import pynvml

                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
                self._pynvml = pynvml
            return self._handle

    def read(self) -> tuple[float, float]:
        check_process(self._owner_pid)
        with self._lock:
            handle = self.handle()
            rates = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(rates.gpu), float(memory.used)

    def ensure_running(self) -> None:
        check_process(self._owner_pid)
        with self._lifecycle_lock:
            self._ensure_running()

    def _ensure_running(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive() and not self._stop.is_set():
            return
        if thread is not None:
            thread.join()
        with self._lock:
            self.handle()  # fail loudly here, not once per sample
            self._stop.clear()
            thread = threading.Thread(
                target=self._run, name=f"et-nvml-{self.index}", daemon=True
            )
            thread.start()
            self._thread = thread

    def start(self, owner: GpuStats, span) -> None:
        check_process(self._owner_pid)
        with self._lifecycle_lock:
            self._ensure_running()
            with self._lock:
                if self._stop.is_set():
                    raise RuntimeError("NVML sampling stopped before span start")
                key = (owner, id(span))
                if key not in self._active:
                    self._active[key] = _NvmlWindow(span)

    def end(self, owner: GpuStats, span) -> tuple[float, float] | None:
        check_process(self._owner_pid)
        with self._lock:
            window = self._active.pop((owner, id(span)), None)
            if window is None or self._stop.is_set():
                return None
            if window.count:
                return window.total / window.count, window.peak
            return self.read()

    def _record(self, timestamp: float, util: float, used: float) -> None:
        with self._lock:
            self.samples.append((timestamp, util, used))
            for window in self._active.values():
                # Span.start() sets started_at only after all start hooks finish.
                started_at = window.span.started_at
                if started_at and timestamp >= started_at:
                    window.count += 1
                    window.total += util
                    window.peak = max(window.peak, used)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                if self._stop.is_set():
                    return
                try:
                    util, used = self.read()
                    if not self._stop.is_set():
                        self._record(time.time(), util, used)
                except Exception as e:
                    self._active.clear()
                    self.samples.clear()
                    self._stop.set()
                    logger.warning(f"NVML sampling stopped: {e}")
                    return

    def stop(self) -> None:
        check_process(self._owner_pid)
        with self._lifecycle_lock:
            self._stop.set()
            thread = self._thread
            if thread is not None:
                thread.join()
            with self._lock:
                self._active.clear()
                self.samples.clear()
                self._thread = None

    def window(self, start: float, end: float) -> list[tuple[float, float, float]]:
        """Samples inside [start, end], newest first.

        This bounded recent history is for diagnostics; span metrics use
        incremental aggregates and do not depend on its retention period.
        """
        picked = []
        with self._lock:
            samples = self.samples
            for index in range(len(samples) - 1, -1, -1):
                sample = samples[index]
                if sample[0] < start:
                    break
                if sample[0] <= end:
                    picked.append(sample)
        return picked


_SAMPLERS: dict[tuple[int, float], _NvmlSampler] = {}
_SAMPLERS_LOCK = threading.RLock()


def _reset_after_fork():
    global _SAMPLERS, _SAMPLERS_LOCK, _TORCH_WINDOWS, _TORCH_LOCK
    _SAMPLERS = {}
    _SAMPLERS_LOCK = threading.RLock()
    _TORCH_WINDOWS = {}
    _TORCH_LOCK = threading.Lock()


register_fork_reset(_reset_after_fork)


def _sampler(index: int, interval: float) -> _NvmlSampler:
    key = (index, interval)
    with _SAMPLERS_LOCK:
        sampler = _SAMPLERS.get(key)
        if sampler is None:
            sampler = _SAMPLERS[key] = _NvmlSampler(index, interval)
    return sampler


def stop_samplers() -> None:
    """Stop every background NVML sampler. Called for you at run teardown."""
    with _SAMPLERS_LOCK:
        for sampler in _SAMPLERS.values():
            sampler.stop()
        _SAMPLERS.clear()


class GpuStats:
    """GPU utilisation and device memory over the span, sampled via NVML.

    Reports ``gpu_percent`` (mean utilisation across the span) and
    ``gpu_mem_used_mb`` (peak device memory). Needs ``nvidia-ml-py``:
    ``pip install expr-tracker[gpu]``.

    A background thread does the sampling, so the span itself pays almost
    nothing. Device memory covers the whole GPU, not just this process; use
    :class:`TorchMemory` for what your own allocator holds.

    Each active span keeps a running sum, count and peak, so long spans retain
    their full sampling window without retaining every sample. Each sample
    updates all active spans; ending a sampled span takes constant time.
    Stopping or failing the sampler discards unfinished measurements.

    Spans shorter than ``interval`` may contain no sample. They fall back to one
    live reading (~14us), which for utilisation is NVML's own rolling average
    rather than a measurement of the span.
    """

    __slots__ = ("_owner_pid", "_sampler")

    def __init__(self, index: int = 0, interval: float = 0.1):
        self._owner_pid = process_id()
        interval = float(interval)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be finite and > 0")
        self._sampler = _sampler(int(index), interval)

    def start(self, span) -> None:
        check_process(self._owner_pid)
        with _SAMPLERS_LOCK:
            self._sampler = _sampler(self._sampler.index, self._sampler.interval)
            self._sampler.start(self, span)

    def end(self, span) -> dict[str, float]:
        check_process(self._owner_pid)
        measured = self._sampler.end(self, span)
        if measured is None:
            return {}
        util, used = measured
        return {
            "gpu_percent": round(util, 1),
            "gpu_mem_used_mb": round(used / 1048576, 3),
        }


atexit.register(stop_samplers)
