"""Stress tests: sustained volume, concurrency, tiny budgets and failing sinks.

Marked ``slow``; run with ``-m slow`` or skip with ``-m "not slow"``.
"""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from expr_tracker.history import HistoryStore, read_history
from expr_tracker.run import Run

pytestmark = pytest.mark.slow


def make_store(tmp_path, *, name="run", **options):
    store = HistoryStore()
    store.init(project="stress", name=name, dir=str(tmp_path), **options)
    return store


def run_dir(tmp_path, name="run"):
    return tmp_path / "stress" / name


def test_high_volume_stays_within_the_cache_budget(tmp_path):
    """100k steps through a 2 MB cache: bounded memory, complete file."""
    store = make_store(tmp_path, cache_bytes=2 * 1024 * 1024, buffer_size=256)
    try:
        for step in range(100_000):
            store.log({"loss": step * 0.001, "lr": 0.1, "grad": step % 7})
        store.flush(commit_open=True)
        stats = store.stats()
        assert stats["cached_bytes"] <= 2 * 1024 * 1024
        assert stats["cached_rows"] < 100_000  # eviction actually happened
        assert stats["rows_on_disk"] == 100_000

        assert [r["_step"] for r in store.get(3)] == [99_997, 99_998, 99_999]
        old = store.get(-1, step_range=(0, 5))  # served from disk
        assert [r["_step"] for r in old] == [0, 1, 2, 3, 4]
        assert len(store.get(-1)) == 100_000
    finally:
        store.finish()
    assert len(read_history(run_dir(tmp_path), -1)) == 100_000


def test_concurrent_writers_lose_nothing(tmp_path):
    """32 threads logging the same steps: every step lands exactly once."""
    store = make_store(tmp_path, step_policy="allow", cache_bytes=1024 * 1024)
    threads, per_thread = 32, 200
    barrier = threading.Barrier(threads)

    def worker(worker_id: int):
        barrier.wait()
        for i in range(per_thread):
            store.log({f"w{worker_id}": i}, step=i, commit=False)

    try:
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(worker, range(threads)))
        store.flush(commit_open=True)

        rows = store.get(-1)
        assert [r["_step"] for r in rows] == list(range(per_thread))
        # every worker's value for every step survived the merges
        for row in rows:
            values = {k: v for k, v in row.items() if k.startswith("w")}
            assert values == {f"w{w}": row["_step"] for w in range(threads)}
    finally:
        store.finish()


def test_readers_and_writers_do_not_deadlock(tmp_path):
    """Queries running against a live writer stay consistent and never block it."""
    store = make_store(tmp_path, cache_bytes=64 * 1024, buffer_size=64)
    stop = threading.Event()
    seen: list[int] = []
    errors: list[Exception] = []

    def reader():
        while not stop.is_set():
            try:
                rows = store.get(random.choice([1, 10, 100, -1]))
                steps = [r["_step"] for r in rows]
                assert steps == sorted(set(steps))  # ordered and unique
                seen.append(len(steps))
            except Exception as e:
                errors.append(e)

    try:
        readers = [threading.Thread(target=reader) for _ in range(4)]
        for thread in readers:
            thread.start()
        for step in range(20_000):
            store.log({"loss": float(step)})
        stop.set()
        for thread in readers:
            thread.join(timeout=10)
            assert not thread.is_alive()
        assert not errors
        assert seen  # the readers really did run
    finally:
        stop.set()
        store.finish()


def test_cache_thrash_with_a_minimal_budget(tmp_path):
    """A cache too small for even one row must still answer every query."""
    store = make_store(tmp_path, cache_bytes=1, buffer_size=1)
    try:
        for step in range(500):
            store.log({"loss": float(step)})
        store.flush(commit_open=True)
        assert store.stats()["cached_rows"] <= 1
        assert [r["_step"] for r in store.get(5)] == [495, 496, 497, 498, 499]
        assert len(store.get(-1)) == 500
        assert store.get(-1, step_range=(100, 110))[0]["loss"] == 100.0
    finally:
        store.finish()


def test_wide_and_sparse_metric_space(tmp_path):
    """2k distinct metric names across 2k steps: projection stays correct."""
    store = make_store(tmp_path, cache_bytes=8 * 1024 * 1024)
    try:
        for step in range(2_000):
            store.log({f"m{step % 2_000}": float(step), "always": 1.0}, step=step)
        store.flush(commit_open=True)
        # dropna drops rows where *all* selected metrics are missing, so a
        # metric present everywhere keeps every row
        rows = store.get(-1, metrics=["m7", "always"], dropna=True)
        assert len(rows) == 2_000
        assert [r["_step"] for r in store.get(-1, metrics=["m7"], dropna=True)] == [7]
        assert len(store.get(-1, metrics=["always"])) == 2_000
        assert len(store.series.points("m1999")) == 1
    finally:
        store.finish()


def test_write_failure_then_recovery(tmp_path, monkeypatch):
    """A disk that fails for a while must not lose rows once it comes back."""
    store = make_store(tmp_path, buffer_size=1)
    try:
        for step in range(50):
            store.log({"loss": float(step)})
        store.flush()
        assert store.stats()["rows_on_disk"] == 50

        healthy = store.writer._write
        monkeypatch.setattr(
            store.writer,
            "_write",
            lambda payload: (_ for _ in ()).throw(OSError("disk full")),
        )
        for step in range(50, 100):
            store.log({"loss": float(step)})
        store.flush()
        assert store.stats()["rows_on_disk"] == 50  # nothing landed
        assert [r["_step"] for r in store.get(3)] == [97, 98, 99]  # still queryable

        monkeypatch.setattr(store.writer, "_write", healthy)
        for step in range(100, 150):
            store.log({"loss": float(step)})
        store.flush(commit_open=True)

        steps = [r["_step"] for r in store.get(-1)]
        assert steps == list(range(150))  # the buffered rows were replayed in order
    finally:
        store.finish()
    assert [r["_step"] for r in read_history(run_dir(tmp_path), -1)] == list(range(150))


def test_many_rules_and_channels(tmp_path):
    """40 rules over 20 channels for 2k steps: bounded work, correct fan-out."""
    received: list[list] = [[] for _ in range(20)]
    config = {
        "channels": [
            {
                "type": "callable",
                "name": f"c{i}",
                "options": {"handler": received[i].append},
                "policy": {
                    "async_send": False,
                    "dedup_window": 0,
                    "rate_limit_per_minute": None,
                },
            }
            for i in range(20)
        ]
    }
    run = Run(
        project="stress",
        name="rules",
        dir=str(tmp_path),
        backends=[],
        alert=config,
        alert_rules=[f"loss > {i * 100} => warning: rule {i}" for i in range(40)],
    )
    try:
        started = time.perf_counter()
        for step in range(2_000):
            run.log({"loss": float(step)})
        elapsed = time.perf_counter() - started

        assert sum(len(box) for box in received) > 0
        assert len({len(box) for box in received}) == 1  # every channel got the same
        assert elapsed < 60
    finally:
        run.finish()


def test_repeated_run_lifecycles_do_not_leak_threads(tmp_path):
    """200 init/finish cycles must not accumulate timers or writer threads."""
    before = threading.active_count()
    for i in range(200):
        store = make_store(tmp_path, name=f"run{i}", max_open_seconds=5)
        store.log({"loss": 1.0}, step=0)
        store.finish()
    time.sleep(0.5)
    assert threading.active_count() <= before + 2


def test_alternating_commit_modes_keep_steps_unique(tmp_path):
    """Mixing implicit, explicit and timed-out commits still yields unique steps."""
    store = make_store(tmp_path, max_open_seconds=0.01, buffer_size=8)
    try:
        for step in range(1_000):
            mode = step % 4
            if mode == 0:
                store.log({"a": step})
            elif mode == 1:
                store.log({"b": step}, commit=False)
                store.log({"c": step})
            elif mode == 2:
                store.log({"d": step}, step=store.current_step)
                time.sleep(0.02)  # let the open-row timer fire
                store.log({"e": step}, step=store.current_step)
            else:
                store.log({"f": step}, commit=True)
        store.flush(commit_open=True)

        steps = [r["_step"] for r in store.get(-1)]
        assert steps == sorted(set(steps))
        assert len(read_history(run_dir(tmp_path), -1)) == len(steps)
    finally:
        store.finish()
