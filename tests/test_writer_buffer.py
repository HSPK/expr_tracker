"""Write buffering: adaptive batching, timers, retry after failure, buffer cap."""

import json
import threading
import time

from expr_tracker.history import HistoryStore
from expr_tracker.history.writer import JsonlWriter


def count_lines(path):
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def make(tmp_path, name, **kwargs):
    store = HistoryStore()
    store.init(
        project="p", name=name, dir=str(tmp_path), max_open_seconds=None, **kwargs
    )
    return store


def test_low_frequency_writes_through(tmp_path):
    s = make(tmp_path, "low", buffer_size=50, buffer_interval=0.2)
    for i in range(4):
        s.log({"v": i})
        time.sleep(0.25)
    assert count_lines(s.log_fp) == 4
    assert s.writer.buffer == []
    s.finish()


def test_high_frequency_is_buffered(tmp_path):
    s = make(
        tmp_path, "high", buffer_size=10, buffer_interval=0.2, max_buffer_seconds=None
    )
    for i in range(9):
        s.log({"v": i})
    # The first record is written straight through; the other 8 stay buffered
    assert count_lines(s.log_fp) == 1
    assert len(s.writer.buffer) == 8
    for i in range(2):
        s.log({"v": 100 + i})
    assert count_lines(s.log_fp) == 11
    assert s.writer.buffer == []
    s.finish()


def test_timer_flushes_stale_records(tmp_path):
    s = make(
        tmp_path, "timer", buffer_size=1000, buffer_interval=1.0, max_buffer_seconds=0.3
    )
    for i in range(5):
        s.log({"v": i})
    assert count_lines(s.log_fp) == 1
    time.sleep(0.7)
    assert count_lines(s.log_fp) == 5
    assert s.writer.buffer == []
    s.finish()


def test_concurrent_logging_does_not_lose_records(tmp_path):
    s = make(
        tmp_path,
        "threads",
        buffer_size=25,
        buffer_interval=0.05,
        max_buffer_seconds=0.2,
    )

    def worker():
        for i in range(200):
            s.log({"v": i})

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    s.finish()
    assert count_lines(s.log_fp) == 800


def test_write_failure_retries_without_duplicates(tmp_path):
    s = make(
        tmp_path,
        "io_fail",
        buffer_size=2,
        buffer_interval=None,
        max_buffer_seconds=None,
    )
    s.log({"v": 0})
    assert count_lines(s.log_fp) == 1

    good = s.writer.path
    blocked = s.log_dir / "blocked"
    blocked.mkdir()
    s.writer.path = blocked  # writing to a directory always fails

    s.log({"v": 1})
    s.log({"v": 2})
    assert len(s.writer.buffer) == 2
    assert count_lines(good) == 1

    s.writer.path = good
    s.finish()
    with open(good, "rb") as f:
        steps = [json.loads(line)["_step"] for line in f if line.strip()]
    assert steps == [0, 1, 2]


def test_buffer_is_capped_when_writes_keep_failing(tmp_path):
    s = make(
        tmp_path,
        "capped",
        buffer_size=2,
        buffer_interval=None,
        max_buffer_seconds=None,
        max_pending_records=4,
    )
    blocked = s.log_dir / "blocked"
    blocked.mkdir()
    good, s.writer.path = s.writer.path, blocked
    for i in range(20):
        s.log({"v": i})
    assert len(s.writer.buffer) <= 4
    assert json.loads(s.writer.buffer[-1][2])["v"] == 19
    s.writer.path = good
    s.finish()


def test_numpy_values_are_serialized(tmp_path):
    np = __import__("numpy")
    s = make(tmp_path, "numpy", buffer_size=1)
    s.log(
        {
            "int32": np.int32(7),
            "float32": np.float32(0.5),
            "bool": np.bool_(True),
            "array": np.array([[1, 2], [3, 4]]),
            "date": np.datetime64("2024-01-01"),
        }
    )
    s.finish()
    record = s.get(1)[0]
    assert record["int32"] == 7
    assert record["float32"] == 0.5
    assert record["bool"] is True
    assert record["array"] == [[1, 2], [3, 4]]
    assert record["date"] == "2024-01-01"


def test_unserializable_value_does_not_block_other_records(tmp_path):
    class Weird:
        __slots__ = ()

        def __repr__(self):
            return "<weird>"

    s = make(tmp_path, "poison", buffer_size=2, buffer_interval=None)
    s.log({"v": 0})
    s.log({"bad": Weird(), "good": 1})
    s.log({"v": 2})
    s.finish()
    records = s.get(-1)
    assert [r["_step"] for r in records] == [0, 1, 2]
    assert records[1]["bad"] == "<weird>" and records[1]["good"] == 1


def test_init_without_name_generates_one(tmp_path):
    s = HistoryStore().init(project="p", dir=str(tmp_path))
    assert s.name
    s.log({"v": 0})
    s.finish()
    assert count_lines(s.log_fp) == 1


def test_meta_is_written_and_reused(tmp_path):
    s = make(tmp_path, "meta", buffer_size=1)
    for i in range(5):
        s.log({"v": i})
    s.finish()
    meta = json.loads(s.writer.meta_path.read_text())
    assert meta["schema"] == 2
    assert meta["lines"] == 5
    assert meta["last_step"] == 4
    assert meta["size"] == s.log_fp.stat().st_size

    writer = JsonlWriter(s.log_fp)
    assert writer.lines == 5
    assert writer.last_step == 4


def test_stale_meta_is_repaired(tmp_path):
    s = make(tmp_path, "stale", buffer_size=1)
    for i in range(3):
        s.log({"v": i})
    s.finish()
    # Simulate a crash: meta lags behind the actual file
    meta = json.loads(s.writer.meta_path.read_text())
    meta["size"], meta["lines"], meta["last_step"] = 0, 0, None
    meta["index"] = []
    s.writer.meta_path.write_text(json.dumps(meta))

    writer = JsonlWriter(s.log_fp)
    assert writer.lines == 3
    assert writer.last_step == 2


def test_stale_meta_with_index_anchor_is_repaired(tmp_path):
    """Rescanning from the last anchor must not duplicate anchors or invent dup steps."""
    s = make(tmp_path, "anchor", buffer_size=1)
    for i in range(2500):
        s.log({"v": i})
    s.finish()
    meta = json.loads(s.writer.meta_path.read_text())
    assert len(meta["index"]) == 3  # one anchor every 1000 lines
    meta["size"] = 0  # simulate a crash
    s.writer.meta_path.write_text(json.dumps(meta))

    writer = JsonlWriter(s.log_fp)
    assert writer.lines == 2500
    assert writer.last_step == 2499
    assert writer.has_duplicate_steps is False
    assert writer.sorted is True
    assert [entry[1] for entry in writer.index] == [0, 1000, 2000]


def test_truncated_file_resets_last_step(tmp_path):
    s = make(tmp_path, "truncated", buffer_size=1)
    for i in range(5):
        s.log({"v": i})
    s.finish()
    s.log_fp.write_bytes(b"")  # file emptied externally while meta still describes it

    writer = JsonlWriter(s.log_fp)
    assert writer.lines == 0
    assert writer.last_step is None


def test_rescan_reproduces_the_meta_based_state(tmp_path):
    """Resume by rescan and resume by sidecar must agree on every counter."""
    path = tmp_path / "metrics.jsonl"
    w = JsonlWriter(path, buffer_size=100, index_every=50)
    for i in range(1000):
        w.append(i, i, (json.dumps({"_step": i}) + "\n").encode())
    w.close()

    from_meta = JsonlWriter(path, index_every=50)
    meta = json.loads(w.meta_path.read_text())
    meta["size"] = 0  # force the rescan path
    w.meta_path.write_text(json.dumps(meta))
    rescanned = JsonlWriter(path, index_every=50)

    for attr in ("lines", "last_step", "sorted", "has_duplicate_steps", "size"):
        assert getattr(from_meta, attr) == getattr(rescanned, attr), attr
    assert from_meta.index == rescanned.index
    assert rescanned.has_duplicate_steps is False  # the anchor line is not a duplicate
