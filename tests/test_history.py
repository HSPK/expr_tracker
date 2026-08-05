"""HistoryStore: cache eviction, disk fallback, step ranges and offline reads."""

import json
import threading
import time

import pytest

from expr_tracker.history import HistoryStore, JsonlReader, read_history


def build(tmp_path, name, count, **kwargs):
    store = HistoryStore()
    store.init(
        project="p",
        name=name,
        dir=str(tmp_path),
        max_open_seconds=None,
        buffer_size=1,
        **kwargs,
    )
    for i in range(count):
        store.log({"loss": float(i), "lr": 0.1})
    store.flush()
    return store


def test_tail_from_cache_only(tmp_path):
    s = build(tmp_path, "cache", 20)
    rows = s.get(5)
    assert [r["_step"] for r in rows] == [15, 16, 17, 18, 19]
    assert not s.stats()["disk_prefix"]
    s.finish()


def test_cache_eviction_falls_back_to_disk(tmp_path):
    s = build(tmp_path, "evict", 50, cache_bytes=200)
    stats = s.stats()
    assert stats["disk_prefix"] and stats["cached_rows"] < 50
    assert [r["_step"] for r in s.get(30)] == list(range(20, 50))
    assert [r["_step"] for r in s.get(-1)] == list(range(50))
    s.finish()


def test_cache_never_evicts_unflushed_rows(tmp_path):
    s = HistoryStore().init(
        project="p",
        name="unflushed",
        dir=str(tmp_path),
        max_open_seconds=None,
        buffer_size=1000,
        buffer_interval=None,
        max_buffer_seconds=None,
        cache_bytes=1,
    )
    for i in range(10):
        s.log({"v": i})
    # Budget is one byte, yet only durable rows may be evicted
    assert [r["_step"] for r in s.get(-1)] == list(range(10))
    s.finish()


def test_step_range_uses_index(tmp_path):
    s = build(tmp_path, "range", 60, cache_bytes=200)
    rows = s.get(-1, step_range=(10, 15))
    assert [r["_step"] for r in rows] == [10, 11, 12, 13, 14]
    assert [r["_step"] for r in s.get(-1, step_range=(55, None))] == [
        55,
        56,
        57,
        58,
        59,
    ]
    s.finish()


def test_include_open_row(tmp_path):
    s = HistoryStore().init(
        project="p", name="open", dir=str(tmp_path), max_open_seconds=None
    )
    s.log({"a": 1})
    s.log({"b": 2}, step=1)  # still the open row
    assert [r["_step"] for r in s.get(5)] == [0, 1]
    assert [r["_step"] for r in s.get(5, include_open=False)] == [0]
    assert s.get(1)[0]["b"] == 2
    s.finish()


def test_metrics_projection_and_dropna(tmp_path):
    s = HistoryStore().init(
        project="p", name="proj", dir=str(tmp_path), max_open_seconds=None
    )
    s.log({"loss": 1.0})
    s.log({"acc": 0.5})
    s.log({"loss": 2.0})
    s.finish()
    rows = s.get(-1, metrics=["loss"])
    assert [set(r) for r in rows] == [
        {"_step", "_time", "loss"},
        {"_step", "_time"},
        {"_step", "_time", "loss"},
    ]
    assert [r["_step"] for r in s.get(-1, metrics=["loss"], dropna=True)] == [0, 2]
    filled = s.get(-1, metrics=["loss"], fill_missing=True)
    assert filled[1]["loss"] is None
    assert "_step" not in s.get(1, include_meta=False)[0]


def test_n_zero_and_negative(tmp_path):
    s = build(tmp_path, "counts", 5)
    assert s.get(0) == []
    assert len(s.get(None)) == 5
    assert len(s.get(-1)) == 5
    s.finish()


def test_offline_read(tmp_path):
    s = build(tmp_path, "offline", 12)
    s.finish()
    assert [r["_step"] for r in read_history(s.log_dir, 3)] == [9, 10, 11]
    assert len(read_history(s.log_fp, -1)) == 12
    assert [r["_step"] for r in read_history(s.log_dir, -1, step_range=(2, 5))] == [
        2,
        3,
        4,
    ]


def test_reader_tail_handles_missing_trailing_newline(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(json.dumps({"_step": i, "v": i}) for i in range(5)), encoding="utf-8"
    )
    reader = JsonlReader(path)
    assert [r["_step"] for r in reader.tail(2)] == [3, 4]
    assert len(reader.read_all()) == 5


def test_reader_handles_utf8_and_corruption(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_step": 0, "名字": "值" * 100}, ensure_ascii=False) + "\n")
        f.write("{not json}\n")
        f.write(json.dumps({"_step": 2, "v": 2}) + "\n")
    reader = JsonlReader(path)
    records = reader.read_all()
    assert [r["_step"] for r in records] == [0, 2]
    assert records[0]["名字"].startswith("值")


def test_reader_offset_of_step(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"_step": i, "v": i}) + "\n" for i in range(0, 100, 2))
    reader = JsonlReader(path)
    assert reader.offset_of_step(0) == 0
    assert [r["_step"] for r in reader.read_steps(50, 56)] == [50, 52, 54]
    assert reader.offset_of_step(1000) == reader.size


def test_pandas_output(tmp_path):
    pd = pytest.importorskip("pandas")
    s = build(tmp_path, "pandas", 4)
    frame = s.get(-1, output_type="pandas")
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns)[:2] == ["_step", "_time"]
    assert frame["loss"].tolist() == [0.0, 1.0, 2.0, 3.0]
    s.finish()


def test_unsupported_output_type(tmp_path):
    s = build(tmp_path, "bad_output", 1)
    with pytest.raises(ValueError, match="Unsupported output_type"):
        s.get(1, output_type="parquet")
    s.finish()


@pytest.mark.parametrize("cache_bytes", [1, 100, 400, 10_000])
@pytest.mark.parametrize("buffer_size", [1, 5, 50])
def test_cache_disk_boundary_is_consistent(tmp_path, cache_bytes, buffer_size):
    """Wherever eviction lands, queries must match what was written, field by field."""
    s = HistoryStore().init(
        project="p",
        name=f"mix{cache_bytes}-{buffer_size}",
        dir=str(tmp_path),
        max_open_seconds=None,
        cache_bytes=cache_bytes,
        buffer_size=buffer_size,
    )
    truth = {}
    total = 60
    for i in range(total):
        payload = {"loss": float(i), "extra": i} if i % 3 else {"loss": float(i)}
        s.log(payload)
        truth[i] = {"_step": i, **payload}

    everything = s.get(-1)
    assert [r["_step"] for r in everything] == list(range(total))
    for record in everything:
        assert {k: v for k, v in record.items() if k != "_time"} == truth[
            record["_step"]
        ]
    for n in (1, 17, total, total + 50):
        assert [r["_step"] for r in s.get(n)] == list(range(max(0, total - n), total))
    assert [r["_step"] for r in s.get(-1, step_range=(10, 25))] == list(range(10, 25))
    s.finish()


def test_patch_lines_survive_eviction(tmp_path):
    """Patch lines from timeout commits must survive eviction and merge back."""
    s = HistoryStore().init(
        project="p",
        name="patch_evict",
        dir=str(tmp_path),
        max_open_seconds=0.01,
        cache_bytes=150,
        buffer_size=1,
    )
    truth = {}
    for i in range(20):
        s.log({"a": i}, step=i)
        truth[i] = {"_step": i, "a": i}
        if i % 4 == 0:
            time.sleep(0.03)  # trigger the open-row timeout commit
            s.log({"b": -i}, step=i)  # late data for the same step: a patch line
            truth[i]["b"] = -i
    s.flush(commit_open=True)

    records = s.get(-1)
    assert [r["_step"] for r in records] == list(range(20))
    for record in records:
        assert {k: v for k, v in record.items() if k != "_time"} == truth[
            record["_step"]
        ]
    assert [r["_step"] for r in s.get(6)] == list(range(14, 20))
    s.finish()


def test_concurrent_log_and_query(tmp_path):
    """Concurrent writes, queries and live timers must not raise or lose data."""
    s = HistoryStore().init(
        project="p",
        name="race",
        dir=str(tmp_path),
        max_open_seconds=0.01,
        buffer_size=7,
        buffer_interval=0.001,
        max_buffer_seconds=0.02,
        cache_bytes=500,
    )
    errors: list = []

    def writer(tid):
        try:
            for i in range(150):
                s.log({f"m{tid}": i})
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def reader():
        try:
            for _ in range(100):
                s.get(10)
                s.get(-1, step_range=(0, 50))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    s.finish()

    assert not errors
    rows = s.get(-1)
    assert [r["_step"] for r in rows] == list(range(600))


@pytest.mark.parametrize("patches", [1, 2, 5])
@pytest.mark.parametrize("cache_bytes", [1, 300, 10**7])
def test_get_n_is_step_accurate_with_patch_lines(tmp_path, patches, cache_bytes):
    """Several rows per step must not make get(n) short-change or half-merge steps."""
    s = HistoryStore().init(
        project="p",
        name=f"steps{patches}-{cache_bytes}",
        dir=str(tmp_path),
        max_open_seconds=None,
        cache_bytes=cache_bytes,
        buffer_size=1,
    )
    truth: dict[int, dict] = {}
    total = 12
    for step in range(total):
        for k in range(patches):
            s.log({f"k{k}": step * 10 + k}, step=step)
            truth.setdefault(step, {"_step": step})[f"k{k}"] = step * 10 + k
            if patches > 1:
                s.flush(commit_open=True)  # force one physical row per key
    s.flush(commit_open=True)

    for n in (1, 2, 7, total, total + 5):
        rows = s.get(n)
        assert [r["_step"] for r in rows] == list(range(max(0, total - n), total))
        for row in rows:  # every returned step must be complete
            assert {k: v for k, v in row.items() if k != "_time"} == truth[row["_step"]]
    assert [r["_step"] for r in s.get(-1)] == list(range(total))
    s.finish()


def test_offline_tail_is_step_accurate_with_patch_lines(tmp_path):
    s = HistoryStore().init(
        project="p",
        name="offline_patch",
        dir=str(tmp_path),
        max_open_seconds=None,
        buffer_size=1,
    )
    for step in range(8):
        for k in range(4):
            s.log({f"k{k}": step * 10 + k}, step=step)
            s.flush(commit_open=True)
    s.finish()

    rows = read_history(s.log_dir, 3)
    assert [r["_step"] for r in rows] == [5, 6, 7]
    for row in rows:
        assert all(f"k{k}" in row for k in range(4)), row


def test_use_before_init_is_rejected():
    """Logging into an uninitialised store used to buffer forever with no eviction."""
    store = HistoryStore()
    with pytest.raises(RuntimeError, match="init"):
        store.log({"v": 1})
    with pytest.raises(RuntimeError, match="init"):
        store.get(5)


@pytest.mark.parametrize("needs_merge", [False, True])
def test_cache_range_scans_from_either_end(tmp_path, needs_merge):
    s = HistoryStore().init(
        project="p",
        name=f"scan{needs_merge}",
        dir=str(tmp_path),
        max_open_seconds=None,
        buffer_size=1,
    )
    for step in range(30):
        s.log({"v": step}, step=step)
        if needs_merge:
            s.flush(commit_open=True)
            s.log({"w": step}, step=step)  # patch line: steps repeat
        s.flush(commit_open=True)
    assert s._needs_merge is needs_merge

    for start, end in [(0, 3), (14, 17), (27, 30), (0, 30), (5, 5), (100, 200)]:
        rows = s.get(-1, step_range=(start, end))
        assert [r["_step"] for r in rows] == [i for i in range(30) if start <= i < end]
    assert [r["_step"] for r in s.get(-1, step_range=(None, 4))] == [0, 1, 2, 3]
    assert [r["_step"] for r in s.get(-1, step_range=(26, None))] == [26, 27, 28, 29]
    s.finish()


def test_query_cost_does_not_grow_with_the_cache(tmp_path):
    """get(n) must snapshot only the rows it returns, not the whole cache."""
    s = HistoryStore().init(
        project="p",
        name="snapshot",
        dir=str(tmp_path),
        max_open_seconds=None,
        buffer_size=500,
    )
    for i in range(5_000):
        s.log({"v": i})
    view = s._view_tail(10)
    # one row per step here, so exactly ten entries; `complete` records that an
    # eleventh, older step was seen, which proves nothing is truncated
    assert view.complete is True
    assert len(view.rows) == 10
    assert [entry[1] for entry in view.rows] == list(range(4_990, 5_000))
    assert [r["_step"] for r in s.get(10)] == list(range(4_990, 5_000))
    s.finish()


def test_scan_helpers_are_pure_and_exact():
    """The cache scans are plain functions over a sequence, so they test directly."""
    from expr_tracker.history.store import _nearer_front, _scan_range, _scan_tail

    # (row, step, line); two rows share step 2 the way a patch line does
    cache = [(0, 0, b"a"), (1, 1, b"b"), (2, 2, b"c"), (3, 2, b"d"), (4, 3, b"e")]

    rows, complete = _scan_tail(cache, 2)
    assert [r[0] for r in rows] == [2, 3, 4] and complete is True
    rows, complete = _scan_tail(cache, 10)
    assert len(rows) == 5 and complete is False

    assert [r[0] for r in _scan_range(cache, (1, 3), forward=True)] == [1, 2, 3]
    assert [r[0] for r in _scan_range(cache, (1, 3), forward=False)] == [1, 2, 3]
    assert _scan_range(cache, (None, 1), forward=True) == [cache[0]]
    assert _scan_range(cache, (3, None), forward=False) == [cache[4]]

    assert _nearer_front((0, 2), oldest=0, newest=3) is True
    assert _nearer_front((3, None), oldest=0, newest=3) is False


def test_open_row_inclusion_rules(tmp_path):
    s = HistoryStore().init(
        project="p", name="openrows", dir=str(tmp_path), max_open_seconds=None
    )
    s.log({"a": 1})
    s.log({"b": 2}, step=1)  # open, not committed
    assert [r["_step"] for r in s.get(5)] == [0, 1]
    assert [r["_step"] for r in s.get(5, include_open=False)] == [0]
    assert [r["_step"] for r in s.get(-1, step_range=(0, 1))] == [0]
    assert [r["_step"] for r in s.get(-1, step_range=(1, 2))] == [1]
    s.finish()


def test_unknown_history_options_are_rejected(tmp_path):
    """A typo used to be silently ignored, leaving the default in place."""
    with pytest.raises(TypeError, match="Unknown history options"):
        HistoryStore().init(project="p", name="typo", dir=str(tmp_path), cache_byte=1)


def test_option_validation_and_clamping():
    from expr_tracker.history.store import HistoryOptions

    with pytest.raises(ValueError, match="Unknown step_policy"):
        HistoryOptions(step_policy="descending")
    options = HistoryOptions(cache_bytes=-5, cache_rows=0, max_open_seconds=3)
    assert options.cache_bytes == 0
    assert options.cache_rows == 1
    assert options.max_open_seconds == 3.0


def test_reinit_resets_every_piece_of_run_state(tmp_path):
    """_reset is the single place run state is initialised, so nothing lingers."""
    s = HistoryStore()
    s.init(project="p", name="one", dir=str(tmp_path), max_open_seconds=None)
    s.log({"v": 1}, step=7)
    s.log({"v": 2}, step=7)  # sets _needs_merge on the first run
    s.init(project="p", name="two", dir=str(tmp_path), max_open_seconds=None)
    try:
        assert s.current_step == 0
        assert s.get(-1) == []
        assert s._needs_merge is False
        assert s._has_disk_prefix is False
        assert s.series.points("v") == ()
    finally:
        s.finish()


def test_metrics_filename_follows_the_shared_rank_helper(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "4")
    sharded = HistoryStore().init(project="p", name="r", dir=str(tmp_path))
    assert sharded.log_fp.name == "metrics.rank4.jsonl"
    sharded.finish()
    plain = HistoryStore().init(
        project="p", name="r2", dir=str(tmp_path), rank_aware=False
    )
    assert plain.log_fp.name == "metrics.jsonl"
    plain.finish()


def test_every_output_format_orders_columns_the_same(tmp_path):
    """`_step`/`_time` first, then metrics in first-seen order, in all formats."""
    pytest.importorskip("pandas")
    s = HistoryStore().init(
        project="p", name="order", dir=str(tmp_path), max_open_seconds=None
    )
    s.log({"loss": 1.0, "acc": 0.5, "_step": 999})  # reserved name must be ignored
    expected = ["_step", "_time", "loss", "acc"]
    assert list(s.get(1)[0]) == expected
    assert list(s.get(1, output_type="pandas").columns) == expected
    assert list(s.get(1, metrics=["loss"])[0]) == ["_step", "_time", "loss"]
    assert s.get(1)[0]["_step"] == 0  # the logged _step was still ignored
    s.log({"loss": 2.0}, step=5)
    assert list(s.open_record()) == ["_step", "_time", "loss"]
    s.finish()
