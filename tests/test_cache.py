"""Cache behaviour: does it actually serve reads, and are its boundaries exact?

Disk access is counted two ways: ``stats()["disk_queries"]`` (what the store reports)
and a wrapper around ``open()`` (ground truth), so a broken counter cannot hide a
broken cache.
"""

import builtins
import json

import pytest

from expr_tracker.history import HistoryStore


@pytest.fixture
def counting_open(monkeypatch):
    """Count reads of the metrics file, whoever performs them."""
    counter = {"reads": 0}
    real_open = builtins.open

    def spy(file, mode="r", *args, **kwargs):
        if "r" in mode and "metrics" in str(file):
            counter["reads"] += 1
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    return counter


@pytest.fixture
def make(tmp_path):
    created = []

    def factory(name="run", **options):
        store = HistoryStore()
        options.setdefault("max_open_seconds", None)
        store.init(project="cache", name=name, dir=str(tmp_path), **options)
        created.append(store)
        return store

    yield factory
    for store in created:
        store.finish()


def without_time(rows):
    """Two runs log at different wall-clock times, so compare everything else."""
    return [{k: v for k, v in row.items() if k != "_time"} for row in rows]


def fill(store, count, start=0, size=1):
    for step in range(start, start + count):
        store.log({"loss": float(step), "pad": "x" * size})
    store.flush(commit_open=True)


# ------------------------------------------------------------------ hits


def test_a_query_inside_the_cache_never_touches_the_file(make, counting_open):
    store = make(cache_bytes=10 * 1024 * 1024)
    fill(store, 200)
    counting_open["reads"] = 0

    for n in (1, 10, 50, 200, -1):
        assert len(store.get(n)) == (200 if n in (-1, 200) else n)
    assert store.stats()["disk_queries"] == 0
    assert counting_open["reads"] == 0
    assert store.stats()["queries"] == 5


def test_a_range_query_inside_the_cache_never_touches_the_file(make, counting_open):
    store = make(cache_bytes=10 * 1024 * 1024)
    fill(store, 200)
    counting_open["reads"] = 0

    rows = store.get(-1, step_range=(50, 60))
    assert [r["_step"] for r in rows] == list(range(50, 60))
    assert store.stats()["disk_queries"] == 0
    assert counting_open["reads"] == 0


def test_the_cache_is_bypassed_only_for_what_it_lost(make, counting_open):
    """Evicting the oldest rows must not push newer queries onto the disk."""
    store = make(cache_bytes=2000)
    fill(store, 400, size=40)
    stats = store.stats()
    assert stats["evicted_rows"] > 0 and stats["cached_rows"] < 400
    cached = stats["cached_rows"]

    counting_open["reads"] = 0
    store.get(max(1, cached - 1))  # comfortably inside the cache
    assert store.stats()["disk_queries"] == 0
    assert counting_open["reads"] == 0

    store.get(400)  # needs the evicted prefix
    assert store.stats()["disk_queries"] == 1
    assert counting_open["reads"] > 0


def test_the_open_row_is_served_without_a_disk_read(make, counting_open):
    store = make(cache_bytes=10 * 1024 * 1024)
    fill(store, 10)
    store.log({"loss": 99.0}, step=10)  # uncommitted
    counting_open["reads"] = 0

    rows = store.get(1)
    assert rows[0]["_step"] == 10 and rows[0]["loss"] == 99.0
    assert counting_open["reads"] == 0
    assert store.get(1, include_open=False)[0]["_step"] == 9


# ------------------------------------------------------------------ boundaries


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9, 10, 11, 20])
def test_results_are_identical_either_side_of_the_boundary(make, n):
    """A tiny cache and a huge one must return exactly the same rows."""
    cached = make("cached", cache_bytes=10 * 1024 * 1024)
    evicted = make("evicted", cache_bytes=1)
    for store in (cached, evicted):
        fill(store, 20)

    assert without_time(cached.get(n)) == without_time(evicted.get(n))
    assert without_time(cached.get(-1)) == without_time(evicted.get(-1))
    assert without_time(cached.get(-1, step_range=(4, 12))) == without_time(
        evicted.get(-1, step_range=(4, 12))
    )
    assert evicted.stats()["disk_queries"] > 0
    assert cached.stats()["disk_queries"] == 0


def test_a_query_spanning_the_boundary_stitches_both_halves(make):
    store = make(cache_bytes=600)
    fill(store, 100, size=40)
    cached = store.stats()["cached_rows"]
    assert 0 < cached < 100

    span = cached + 5  # a few rows from disk, the rest from memory
    rows = store.get(span)
    assert [r["_step"] for r in rows] == list(range(100 - span, 100))
    assert all(r["loss"] == float(r["_step"]) for r in rows)


def test_a_range_straddling_the_boundary_is_complete(make):
    store = make(cache_bytes=600)
    fill(store, 100, size=40)
    oldest_cached = store.get(store.stats()["cached_rows"])[0]["_step"]

    rows = store.get(-1, step_range=(oldest_cached - 5, oldest_cached + 5))
    assert [r["_step"] for r in rows] == list(
        range(oldest_cached - 5, oldest_cached + 5)
    )


def test_the_byte_budget_is_honoured_exactly(make):
    store = make(cache_bytes=4096)
    fill(store, 500, size=30)
    stats = store.stats()
    assert stats["cached_bytes"] <= 4096
    assert stats["cache_limit_bytes"] == 4096
    # the budget is not wasted either: the cache stays reasonably full
    assert stats["cached_bytes"] > 4096 // 2


def test_the_row_budget_is_honoured(make):
    store = make(cache_rows=25, cache_bytes=10 * 1024 * 1024)
    fill(store, 200)
    assert store.stats()["cached_rows"] <= 25
    assert [r["_step"] for r in store.get(200)] == list(range(200))


def test_undurable_rows_are_never_evicted(make, monkeypatch):
    """When the disk is unavailable, the budget yields rather than the data."""
    store = make(cache_bytes=1, buffer_size=1000, buffer_interval=None)
    monkeypatch.setattr(
        store.writer, "_write", lambda payload: (_ for _ in ()).throw(OSError("full"))
    )
    for step in range(50):
        store.log({"loss": float(step)})

    stats = store.stats()
    assert stats["rows_on_disk"] == 0
    assert stats["evicted_rows"] == 0  # nothing durable, so nothing droppable
    assert stats["cached_bytes"] > stats["cache_limit_bytes"]  # budget deliberately
    assert [r["_step"] for r in store.get(-1)] == list(range(50))

    monkeypatch.undo()
    store.flush(commit_open=True)
    assert [r["_step"] for r in store.get(-1)] == list(range(50))


def test_eviction_only_drops_rows_that_reached_the_disk(make):
    store = make(cache_bytes=1, buffer_size=1000, buffer_interval=None)
    for step in range(50):
        store.log({"loss": float(step)})
        stats = store.stats()
        assert stats["evicted_rows"] <= stats["rows_on_disk"]
    store.flush(commit_open=True)
    assert [r["_step"] for r in store.get(-1)] == list(range(50))


def test_eviction_flushes_when_only_undurable_rows_remain(make):
    store = make(cache_bytes=1, buffer_size=1000, buffer_interval=None)
    fill(store, 20)
    assert store.stats()["rows_on_disk"] == 20  # the writer was forced to flush
    assert store.stats()["cached_rows"] <= 1


# ------------------------------------------------------------------ warm vs cold


def test_a_cold_read_matches_a_warm_read(make, tmp_path):
    store = make("cold", cache_bytes=10 * 1024 * 1024)
    fill(store, 150)
    warm = store.get(-1)
    store.finish()

    reopened = HistoryStore()
    reopened.init(project="cache", name="cold", dir=str(tmp_path))
    try:
        assert reopened.get(-1) == warm
        assert reopened.stats()["disk_queries"] == 1  # nothing is cached yet
    finally:
        reopened.finish()


def test_resumed_rows_are_read_from_disk_then_cached(make, tmp_path, counting_open):
    store = make("resume", cache_bytes=10 * 1024 * 1024)
    fill(store, 30)
    store.finish()

    reopened = HistoryStore()
    reopened.init(project="cache", name="resume", dir=str(tmp_path))
    try:
        counting_open["reads"] = 0
        assert len(reopened.get(-1)) == 30
        assert counting_open["reads"] > 0  # the old rows live only on disk

        fill(reopened, 10, start=30)
        counting_open["reads"] = 0
        assert [r["_step"] for r in reopened.get(5)] == [35, 36, 37, 38, 39]
        assert counting_open["reads"] == 0  # the new rows are cached
    finally:
        reopened.finish()


# ------------------------------------------------------------------ accounting


def test_cache_bytes_track_the_lines_actually_stored(make):
    store = make(cache_bytes=10 * 1024 * 1024)
    fill(store, 40)
    stats = store.stats()

    metrics_file = store.log_fp
    on_disk = sum(len(line) + 1 for line in metrics_file.read_bytes().splitlines())
    assert stats["cached_bytes"] == on_disk
    assert stats["cached_rows"] == 40


def test_counters_survive_a_reinitialisation(make, tmp_path):
    store = make("reinit", cache_bytes=10 * 1024 * 1024)
    fill(store, 10)
    store.get(5)
    assert store.stats()["queries"] == 1

    store.init(project="cache", name="reinit2", dir=str(tmp_path))
    assert store.stats()["queries"] == 0
    assert store.stats()["evicted_rows"] == 0
    assert store.stats()["cached_rows"] == 0


def test_evicted_rows_count_matches_the_gap(make):
    store = make(cache_bytes=800)
    fill(store, 200, size=40)
    stats = store.stats()
    assert stats["evicted_rows"] == 200 - stats["cached_rows"]
    assert stats["disk_prefix"] is True


def test_a_cache_large_enough_never_reports_a_disk_prefix(make):
    store = make(cache_bytes=10 * 1024 * 1024)
    fill(store, 300)
    stats = store.stats()
    assert stats["disk_prefix"] is False
    assert stats["evicted_rows"] == 0


# ------------------------------------------------------------------ patch lines


def test_patched_steps_survive_eviction(make):
    """A step split across two rows must merge correctly after eviction."""
    store = make(cache_bytes=1, max_open_seconds=None)
    store.log({"a": 1}, step=0, commit=True)
    store.log({"b": 2}, step=0, commit=True)  # a patch line for step 0
    fill(store, 30, start=1)

    rows = store.get(-1)
    assert rows[0] == {**rows[0], "a": 1, "b": 2}
    assert [r["_step"] for r in rows] == list(range(31))
    assert len(store.get(1, step_range=(0, 1))) == 1


def test_a_patched_step_at_the_boundary_is_not_half_merged(make):
    store = make(cache_bytes=200, step_policy="allow")
    fill(store, 20, size=20)
    for step in range(20):  # patch every step, pushing the old rows out
        store.log({"extra": step}, step=step, commit=True)
    store.flush(commit_open=True)

    rows = store.get(-1)
    assert len(rows) == 20
    assert all("extra" in r and "loss" in r for r in rows)


# ------------------------------------------------------------------ consistency


def test_every_prefix_length_returns_a_consistent_suffix(make):
    """get(n) must always be the last n steps of get(-1), whatever is cached."""
    store = make(cache_bytes=1500)
    fill(store, 120, size=25)
    everything = store.get(-1)
    assert len(everything) == 120

    for n in (1, 2, 7, 30, 60, 119, 120, 500):
        assert store.get(n) == everything[-n:]


def test_range_queries_agree_with_the_full_history(make):
    store = make(cache_bytes=1500)
    fill(store, 120, size=25)
    everything = {r["_step"]: r for r in store.get(-1)}

    for start, end in [(0, 1), (0, 120), (37, 38), (50, 90), (110, 130), (200, 300)]:
        rows = store.get(-1, step_range=(start, end))
        expected = [everything[s] for s in range(start, min(end, 120))]
        assert rows == expected


def test_open_ended_ranges(make):
    store = make(cache_bytes=1500)
    fill(store, 60, size=25)
    assert [r["_step"] for r in store.get(-1, step_range=(None, 3))] == [0, 1, 2]
    assert [r["_step"] for r in store.get(-1, step_range=(57, None))] == [57, 58, 59]
    assert len(store.get(-1, step_range=(None, None))) == 60


def test_json_is_parsed_only_for_the_rows_returned(make):
    """A tail query must not decode the whole cache."""
    store = make(cache_bytes=10 * 1024 * 1024)
    fill(store, 500)
    decoded = []
    original = json.loads

    def counting_loads(raw, *args, **kwargs):
        decoded.append(raw)
        return original(raw, *args, **kwargs)

    json.loads = counting_loads
    try:
        store.get(5)
    finally:
        json.loads = original
    assert len(decoded) <= 12  # the 5 rows plus the merge-mode margin
