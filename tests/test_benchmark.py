"""Benchmarks: throughput, tail latency, query cost, and stability over time.

Thresholds are deliberately loose so the suite is not machine-dependent; the
assertions that matter are the *relative* ones -- cached beats disk, and the cost
per log does not grow with the number of steps already logged.

Run with ``-m benchmark -s`` to see the report.
"""

import gc
import statistics
import sys
import time
import tracemalloc

import pytest

from expr_tracker.history import HistoryStore

pytestmark = pytest.mark.benchmark

PAYLOAD = {
    "train/loss": 0.5,
    "train/acc": 0.9,
    "lr": 1e-4,
    "grad_norm": 1.2,
    "step_time": 0.03,
}


@pytest.fixture(autouse=True)
def skip_under_tracing():
    if sys.gettrace() is not None:
        pytest.skip("timing is meaningless under coverage/debugger tracing")


@pytest.fixture
def make(tmp_path):
    created = []

    def factory(name="bench", **options):
        store = HistoryStore()
        options.setdefault("max_open_seconds", None)
        store.init(project="bench", name=name, dir=str(tmp_path), **options)
        created.append(store)
        return store

    yield factory
    for store in created:
        store.finish()


def percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    take = lambda q: ordered[min(len(ordered) - 1, int(len(ordered) * q))]  # noqa: E731
    return {
        "p50": take(0.50),
        "p90": take(0.90),
        "p99": take(0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def report(label: str, stats: dict[str, float], unit: str = "us"):
    scale = 1e6 if unit == "us" else 1e3
    body = "  ".join(f"{k}={v * scale:8.1f}" for k, v in stats.items())
    print(f"\n{label:<38} {body} ({unit})")


def time_calls(count, call) -> list[float]:
    samples = []
    for i in range(count):
        started = time.perf_counter()
        call(i)
        samples.append(time.perf_counter() - started)
    return samples


# ------------------------------------------------------------------ write


def test_log_throughput_and_tail_latency(make):
    store = make("throughput", buffer_size=200)
    samples = time_calls(20_000, lambda i: store.log(dict(PAYLOAD)))
    store.flush(commit_open=True)

    stats = percentiles(samples)
    throughput = len(samples) / sum(samples)
    report("log() latency", stats)
    print(f"{'log() throughput':<38} {throughput:,.0f} calls/s")

    assert stats["p50"] < 200e-6
    assert stats["p99"] < 5e-3  # a buffer flush must not stall the loop
    assert stats["max"] < 0.5
    assert throughput > 5_000


def test_log_cost_does_not_grow_with_history(make):
    """The killer regression: an O(n) write path only shows up at scale."""
    store = make("growth", buffer_size=200, cache_bytes=64 * 1024 * 1024)
    window = 10_000
    first = time_calls(window, lambda i: store.log(dict(PAYLOAD)))
    time_calls(60_000, lambda i: store.log(dict(PAYLOAD)))
    last = time_calls(window, lambda i: store.log(dict(PAYLOAD)))
    store.flush(commit_open=True)

    early, late = percentiles(first), percentiles(last)
    report("log() @ first 10k", early)
    report("log() @ after 70k", late)
    assert late["p50"] < early["p50"] * 3 + 20e-6


def test_eviction_does_not_slow_down_logging(make):
    """A cache at its budget evicts on every write; that must stay cheap."""
    roomy = make("roomy", cache_bytes=256 * 1024 * 1024, buffer_size=200)
    tight = make("tight", cache_bytes=64 * 1024, buffer_size=200)
    samples = {}
    for name, store in (("roomy", roomy), ("tight", tight)):
        samples[name] = percentiles(
            time_calls(20_000, lambda i, s=store: s.log(dict(PAYLOAD)))
        )
        store.flush(commit_open=True)
        report(f"log() cache={name}", samples[name])

    assert tight.stats()["evicted_rows"] > 10_000
    assert samples["tight"]["p50"] < samples["roomy"]["p50"] * 3 + 20e-6


def test_flush_pauses_are_bounded(make):
    """Large buffers amortise IO, but a single log() must never wait long for it."""
    store = make("flush", buffer_size=2_000)
    samples = time_calls(20_000, lambda i: store.log(dict(PAYLOAD)))
    store.flush(commit_open=True)

    stats = percentiles(samples)
    report("log() buffer=2000", stats)
    assert stats["max"] < 0.5
    assert stats["p99"] < 20e-3


# ------------------------------------------------------------------ read


def test_cached_queries_are_much_faster_than_disk_queries(make):
    rows = 50_000
    cached = make("cached", cache_bytes=256 * 1024 * 1024, buffer_size=500)
    evicted = make("evicted", cache_bytes=1, buffer_size=500)
    for store in (cached, evicted):
        for _ in range(rows):
            store.log(dict(PAYLOAD))
        store.flush(commit_open=True)

    assert cached.stats()["evicted_rows"] == 0
    assert evicted.stats()["cached_rows"] <= 1

    results = {}
    for name, store in (("cache", cached), ("disk", evicted)):
        results[name] = percentiles(time_calls(200, lambda i, s=store: s.get(100)))
        report(f"get(100) from {name}", results[name])
    assert cached.stats()["disk_queries"] == 0
    assert evicted.stats()["disk_queries"] == 200
    assert results["cache"]["p50"] < results["disk"]["p50"]
    assert results["cache"]["p50"] < 5e-3


def test_tail_query_cost_is_independent_of_history_size(make):
    """get(50) must cost the same at 1k steps and at 100k."""
    store = make("tailcost", cache_bytes=256 * 1024 * 1024, buffer_size=500)
    measurements = {}
    for target in (1_000, 10_000, 100_000):
        while store.current_step < target:
            store.log(dict(PAYLOAD))
        store.flush(commit_open=True)
        measurements[target] = percentiles(time_calls(200, lambda i: store.get(50)))
        report(f"get(50) @ {target:,} steps", measurements[target])

    small, large = measurements[1_000]["p50"], measurements[100_000]["p50"]
    assert large < small * 5 + 200e-6


def test_range_query_cost_scales_with_the_range_not_the_run(make):
    store = make("rangecost", cache_bytes=1, buffer_size=500)  # force disk reads
    for _ in range(50_000):
        store.log(dict(PAYLOAD))
    store.flush(commit_open=True)

    near = percentiles(time_calls(100, lambda i: store.get(-1, step_range=(10, 60))))
    far = percentiles(
        time_calls(100, lambda i: store.get(-1, step_range=(45_000, 45_050)))
    )
    report("get(range) near the start", near)
    report("get(range) near the end", far)
    # position independence: the index must seek, not scan from the start
    assert far["p50"] < near["p50"] * 3 + 1e-3


def test_full_history_read_is_linear_and_bounded(make):
    store = make("fullread", cache_bytes=256 * 1024 * 1024, buffer_size=500)
    for _ in range(100_000):
        store.log(dict(PAYLOAD))
    store.flush(commit_open=True)

    started = time.perf_counter()
    rows = store.get(-1)
    elapsed = time.perf_counter() - started
    print(f"\n{'get(-1) over 100k steps':<38} {elapsed * 1e3:8.1f} ms")
    assert len(rows) == 100_000
    assert elapsed < 10


# ------------------------------------------------------------------ memory


def test_memory_stays_within_the_cache_budget(make):
    budget = 4 * 1024 * 1024
    store = make("memory", cache_bytes=budget, buffer_size=500)
    gc.collect()
    tracemalloc.start()
    try:
        for _ in range(200_000):
            store.log(dict(PAYLOAD))
        store.flush(commit_open=True)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    stats = store.stats()
    print(
        f"\n{'memory after 200k logs':<38} "
        f"live={current / 1e6:6.1f} MB  peak={peak / 1e6:6.1f} MB  "
        f"cache={stats['cached_bytes'] / 1e6:5.1f} MB"
    )
    assert stats["cached_bytes"] <= budget
    assert current < budget * 4  # bookkeeping overhead, not a second copy of it all
    assert stats["rows_on_disk"] == 200_000


def test_memory_is_flat_across_repeated_runs(make, tmp_path):
    """Init/finish cycles must not accumulate anything."""
    gc.collect()
    tracemalloc.start()
    try:
        for i in range(30):
            store = HistoryStore()
            store.init(project="bench", name=f"cycle{i}", dir=str(tmp_path))
            for _ in range(500):
                store.log(dict(PAYLOAD))
            store.finish()
            if i == 4:
                gc.collect()
                baseline = tracemalloc.get_traced_memory()[0]
        gc.collect()
        final = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()

    print(
        f"\n{'memory across 30 runs':<38} "
        f"after 5={baseline / 1e6:6.2f} MB  after 30={final / 1e6:6.2f} MB"
    )
    assert final < baseline * 3 + 5e6


def test_series_memory_is_bounded_by_the_alert_window(make):
    store = make("series", alert_window=100, cache_bytes=1024 * 1024)
    for step in range(50_000):
        store.log({f"m{step % 50}": float(step)})
    store.flush(commit_open=True)
    points = sum(len(store.series.points(f"m{i}")) for i in range(50))
    print(f"\n{'series points for 50 metrics':<38} {points}")
    assert points <= 50 * 100


# ------------------------------------------------------------------ alerts


def test_alert_evaluation_cost_is_bounded(make, tmp_path):
    from expr_tracker.run import Run

    rules = [
        "diff(loss) > 50 => warning: spike",
        "zscore(loss[50]) > 4 => error: z",
        "mean(loss[20]) > 100 => warning: mean",
        "stalled(loss[100]) => warning: stalled",
        "isnan(loss) or isinf(loss) => critical: nan",
    ]
    plain = Run(project="bench", name="norules", dir=str(tmp_path), backends=[])
    ruled = Run(
        project="bench",
        name="rules",
        dir=str(tmp_path),
        backends=[],
        alert_rules=rules,
    )
    try:
        without = percentiles(time_calls(20_000, lambda i: plain.log({"loss": 0.5})))
        with_rules = percentiles(time_calls(20_000, lambda i: ruled.log({"loss": 0.5})))
    finally:
        plain.finish()
        ruled.finish()

    report("log() without rules", without)
    report(f"log() with {len(rules)} rules", with_rules)
    overhead = with_rules["p50"] - without["p50"]
    print(f"{'per-rule overhead':<38} {overhead / len(rules) * 1e6:8.2f} us")
    assert overhead < 500e-6
