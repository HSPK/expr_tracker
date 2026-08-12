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
import threading
import time
from collections import deque
from typing import Any

from loguru import logger

__all__ = ["CpuTime", "GpuStats", "TorchMemory", "stop_samplers"]


class CpuTime:
    """CPU utilisation over the span, from :func:`time.process_time`.

    Reports ``cpu_percent`` (process CPU time / wall time, so >100% means the
    span used more than one core) and ``cpu_time_ms``. Exact, not sampled, and
    the cheapest thing here at ~1.5us per span. Counts every thread in the
    process, so concurrent work outside the span inflates it.
    """

    __slots__ = ("_stack",)

    def __init__(self):
        self._stack: dict[int, list[float]] = {}

    def _frames(self) -> list[float]:
        return self._stack.setdefault(threading.get_ident(), [])

    def start(self, span) -> None:
        self._frames().append(time.process_time())

    def end(self, span) -> dict[str, float]:
        frames = self._frames()
        if not frames:
            return {}
        used_ms = (time.process_time() - frames.pop()) * 1000.0
        percent = 100.0 * used_ms / span.duration_ms if span.duration_ms > 0 else 0.0
        return {"cpu_time_ms": round(used_ms, 3), "cpu_percent": round(percent, 1)}


class TorchMemory:
    """Peak CUDA memory the allocator held during the span, via torch.

    Reports ``gpu_mem_peak_mb`` and ``gpu_mem_delta_mb`` (end minus start, i.e.
    what the span kept). Exact, ~10us per span.

    ``torch.cuda.max_memory_allocated`` is one global high-water mark, so a
    naive reset would let a child span erase its parent's history. This keeps
    its own stack and rolls each child's peak back into the enclosing span.
    """

    __slots__ = ("_device", "_stack", "_torch")

    def __init__(self, device: Any = None):
        self._device = device
        self._torch = None
        self._stack: dict[int, list[tuple[float, float]]] = {}

    def _cuda(self):
        if self._torch is None:
            import torch  # raised to _safely() and logged once per span otherwise

            self._torch = torch
        if not self._torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return self._torch.cuda

    def _frames(self) -> list[list[float]]:
        return self._stack.setdefault(threading.get_ident(), [])

    def _read(self, cuda) -> tuple[float, float]:
        """(allocated, peak) in bytes.

        One nested-stats call costs ~17us; the two public helpers rebuild and
        flatten the whole stats dict separately and cost ~90us each.
        """
        nested = getattr(cuda, "memory_stats_as_nested_dict", None)
        if nested is not None:
            stats = nested(device=self._device).get("allocated_bytes")
            if stats:
                return float(stats["all"]["current"]), float(stats["all"]["peak"])
        return (
            float(cuda.memory_allocated(self._device)),
            float(cuda.max_memory_allocated(self._device)),
        )

    def start(self, span) -> None:
        cuda = self._cuda()
        opening, outer_peak = self._read(cuda)
        # [peak the enclosing span had reached, our opening bytes, carried peak]
        self._frames().append([outer_peak, opening, 0.0])
        cuda.reset_peak_memory_stats(self._device)

    def end(self, span) -> dict[str, float]:
        frames = self._frames()
        if not frames:
            return {}
        outer_peak, opening, carried = frames.pop()
        closing, raw_peak = self._read(self._cuda())
        # The counter was reset for us, so it holds our peak alone; our own
        # children left anything that predated them in `carried`
        peak = max(raw_peak, carried)
        if frames:
            # Hand our parent back the peak it had before we reset the counter
            frames[-1][2] = max(frames[-1][2], outer_peak)
        return {
            "gpu_mem_peak_mb": round(peak / 1048576, 3),
            "gpu_mem_delta_mb": round((closing - opening) / 1048576, 3),
        }


class _NvmlSampler:
    """One background thread polling NVML, shared by every :class:`GpuStats`.

    Sampling in the background is the only honest way to report utilisation:
    NVML's counter is a rolling average over its own window, so reading it once
    at the end of a 3ms span says nothing about that span.
    """

    def __init__(self, index: int, interval: float):
        self.index = index
        self.interval = interval
        self.samples: deque[tuple[float, float, float]] = deque(
            maxlen=max(64, int(60.0 / interval))
        )
        self._handle = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def handle(self):
        if self._handle is None:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            self._pynvml = pynvml
        return self._handle

    def read(self) -> tuple[float, float]:
        handle = self.handle()
        rates = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(rates.gpu), float(memory.used)

    def ensure_running(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            return  # the common case, off the lock
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self.handle()  # fail loudly here, not once per sample
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name=f"et-nvml-{self.index}", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                util, used = self.read()
            except Exception as e:
                logger.warning(f"NVML sampling stopped: {e}")
                return
            with self._lock:
                self.samples.append((time.time(), util, used))

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.interval * 2 + 1.0)
        self._thread = None

    def window(self, start: float, end: float) -> list[tuple[float, float, float]]:
        """Samples inside [start, end], newest first.

        Scanned backwards so a span pays for its own window, not for the whole
        minute of history the deque keeps.
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
_SAMPLERS_LOCK = threading.Lock()


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
        samplers = list(_SAMPLERS.values())
        _SAMPLERS.clear()
    for sampler in samplers:
        sampler.stop()


class GpuStats:
    """GPU utilisation and device memory over the span, sampled via NVML.

    Reports ``gpu_percent`` (mean utilisation across the span) and
    ``gpu_mem_used_mb`` (peak device memory). Needs ``nvidia-ml-py``:
    ``pip install expr-tracker[gpu]``.

    A background thread does the sampling, so the span itself pays almost
    nothing. Device memory covers the whole GPU, not just this process; use
    :class:`TorchMemory` for what your own allocator holds.

    Spans shorter than ``interval`` may contain no sample. They fall back to one
    live reading (~14us), which for utilisation is NVML's own rolling average
    rather than a measurement of the span.
    """

    __slots__ = ("_sampler",)

    def __init__(self, index: int = 0, interval: float = 0.1):
        if interval <= 0:
            raise ValueError("interval must be > 0")
        self._sampler = _sampler(int(index), float(interval))

    def start(self, span) -> None:
        self._sampler.ensure_running()

    def end(self, span) -> dict[str, float]:
        samples = self._sampler.window(span.started_at, time.time())
        if samples:
            util = sum(s[1] for s in samples) / len(samples)
            used = max(s[2] for s in samples)
        else:
            util, used = self._sampler.read()
        return {
            "gpu_percent": round(util, 1),
            "gpu_mem_used_mb": round(used / 1048576, 3),
        }


atexit.register(stop_samplers)
