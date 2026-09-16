"""Focused trace CPU/memory scaling; run with ``pytest tests/test_trace_perf.py -s``.

Timing assertions allow shared-runner noise. Python peaks are measured separately
from uninstrumented timings; fresh-process RSS also covers native SQLite buffers.
"""

import gc
import json
import statistics
import subprocess
import sys
import time
import tracemalloc
from functools import partial

import pytest

from expr_tracker.trace import _LanePool, assign_lanes, read_spans, write_trace

pytestmark = pytest.mark.benchmark


def timed(call, repeats=3):
    samples = []
    for _ in range(repeats):
        started = time.process_time()
        call()
        samples.append(time.process_time() - started)
    return statistics.median(samples)


def python_peak(call):
    gc.collect()
    tracemalloc.start()
    try:
        call()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def source_file(tmp_path, count):
    path = tmp_path / f"spans-{count}.jsonl"
    with path.open("w", encoding="utf-8") as target:
        # Reverse completion order requires sorting; step filters cannot help.
        for index in reversed(range(count)):
            target.write(
                json.dumps(
                    {
                        "name": "work",
                        "start": 1_700_000_000 + index,
                        "dur_ms": 1,
                        "depth": 0,
                        "_step": 0,
                        "args": {"payload": "x" * 1024},
                    }
                )
                + "\n"
            )
    return path


def test_crossing_lane_cpu_scaling():
    results = []
    for count in (4000, 16000, 64000):
        spans = [{"start": index / count, "dur_ms": 2000} for index in range(count)]
        seconds = timed(partial(assign_lanes, spans))
        lanes = assign_lanes(spans)
        assert len(set(lanes.values())) == count
        results.append(seconds)
        print(f"\ncrossing lanes: n={count:,}, CPU seconds={seconds:.4f}")
    # A 16x workload would take ~256x with the previous quadratic scan.
    assert results[-1] < results[0] * 45 + 0.1


def test_crossing_lanes_improve_over_the_previous_quadratic_scan():
    spans = [{"start": index / 4000, "dur_ms": 2000} for index in range(4000)]

    def previous_pack():
        lanes = []
        for record in spans:
            start = record["start"]
            end = start + record["dur_ms"] / 1000
            for stack in lanes:
                while stack and stack[-1] <= start:
                    stack.pop()
                if not stack or end <= stack[-1]:
                    stack.append(end)
                    break
            else:
                lanes.append([end])
        return len(lanes)

    assert previous_pack() == len(spans)
    old = timed(previous_pack, repeats=1)
    new = timed(partial(assign_lanes, spans))
    print(
        f"\n4,000 crossing spans: previous CPU={old:.4f}s, "
        f"new CPU={new:.4f}s, speedup={old / new:.1f}x"
    )
    assert new < old / 3


def test_deep_nesting_and_long_worker_state_do_not_accumulate_history():
    pool = _LanePool()
    for depth in range(20_000):
        assert pool.place(0, 1000, depth) == 0
    assert len(pool.stacks) == 1
    assert pool.place(1_000_000, 1, 0) == 0
    assert len(pool.ends) == 1

    pool = _LanePool()
    pool.place(0, 1_000_000_000, 0)
    for index in range(100_000):
        assert pool.place((index + 1) * 1_000_000, 1, 0) == 1
    assert len(pool.stacks) == 2
    assert len(pool.ends) == 2
    assert sum(map(len, pool.available.values())) <= 68
    print("\n100,000-step-zero worker: two lanes, bounded stale heap state")


def test_filtered_read_memory_is_independent_of_file_size(tmp_path):
    results = []
    for count in (4000, 32000):
        source = source_file(tmp_path, count)
        bounds = (1_700_000_000 + count - 10, None)
        call = partial(read_spans, source, time_range=bounds)
        assert len(call()) == 10
        seconds = timed(call, repeats=1)
        peak = python_peak(call)
        results.append((seconds, peak))
        print(
            f"\nfiltered read: n={count:,}, retained=10, CPU seconds={seconds:.4f}, "
            f"Python peak={peak / 1024:.1f} KiB"
        )
    assert results[-1][1] < results[0][1] * 2 + 128 * 1024
    assert results[-1][0] < results[0][0] * 16 + 0.1


def test_streamed_export_cpu_and_python_memory_scaling(tmp_path):
    output = tmp_path / "trace.json"
    results = []
    for count in (4000, 32000):
        source = source_file(tmp_path, count)
        call = partial(write_trace, source, output)
        started = time.perf_counter()
        assert call() == count
        wall = time.perf_counter() - started
        seconds = timed(call)
        peak = python_peak(call)
        results.append((seconds, peak))
        print(
            f"\nstreamed export: n={count:,}, CPU seconds={seconds:.4f}, wall={wall:.4f}, "
            f"Python peak={peak / 1024:.1f} KiB, output={output.stat().st_size / 2**20:.1f} MiB"
        )
    assert results[-1][1] < results[0][1] * 2 + 512 * 1024
    assert results[-1][1] < 4 * 2**20
    assert results[-1][0] < results[0][0] * 18 + 0.5


def test_crossing_export_reports_overlap_memory_cost(tmp_path):
    source = tmp_path / "spans.jsonl"
    output = tmp_path / "trace.json"
    count = 16_000
    with source.open("w", encoding="utf-8") as target:
        for index in reversed(range(count)):
            target.write(
                json.dumps(
                    {
                        "start": index / count,
                        "dur_ms": 2000,
                        "depth": 0,
                        "_step": 0,
                    }
                )
                + "\n"
            )
    call = partial(write_trace, source, output)
    assert call() == count
    seconds = timed(call, repeats=1)
    peak = python_peak(call)
    with output.open(encoding="utf-8") as result:
        events = [
            event for event in json.load(result)["traceEvents"] if event["ph"] == "X"
        ]
    assert len({event["tid"] for event in events}) == count
    print(
        f"\ncrossing export: n={count:,}, CPU={seconds:.4f}s, "
        f"Python peak={peak / 2**20:.2f} MiB (O(overlap), deliberately not constant)"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="resource RSS is Unix-specific")
def test_fresh_process_export_rss_includes_sqlite_buffers(tmp_path):
    script = """
import json, resource, sys, time
from expr_tracker.trace import write_trace
def rss():
    if sys.platform.startswith("linux"):
        # ru_maxrss can inherit pytest's pre-exec high-water mark on Linux.
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
before = rss()
started = time.perf_counter()
count = write_trace(sys.argv[1], sys.argv[2])
print(json.dumps({
    "count": count, "seconds": time.perf_counter() - started,
    "rss_before": before,
    "rss_peak": rss(),
}))
"""
    results = []
    for count in (4000, 32000):
        source = source_file(tmp_path, count)
        completed = subprocess.run(
            [sys.executable, "-c", script, str(source), str(tmp_path / "trace.json")],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        assert result["count"] == count
        results.append(result)
        print(f"\nfresh export process: {result}")
    unit = 1 if sys.platform == "darwin" else 1024
    growth = (results[-1]["rss_peak"] - results[0]["rss_peak"]) * unit
    assert growth < 16 * 2**20
