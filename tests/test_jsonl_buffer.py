import tempfile
import threading
import time
from pathlib import Path

import jsonlines

from expr_tracker.jsonl import JsonlTracker


def _count_lines(fp: Path) -> int:
    if not fp.exists():
        return 0
    with open(fp, "rb") as f:
        return sum(1 for _ in f)


def _new_tracker(name: str, **kwargs):
    tracker = JsonlTracker()
    tracker.init(project="p", name=name, dir=tempfile.mkdtemp(), **kwargs)
    return tracker


def test_low_frequency_writes_through():
    """间隔大于 buffer_interval 的低频 log 应该直接写盘"""
    tracker = _new_tracker("low_freq", buffer_size=50, buffer_interval=0.2)
    for i in range(4):
        tracker.log({"v": i})
        time.sleep(0.25)
    assert _count_lines(tracker.log_fp) == 4
    assert tracker.buffer == []
    tracker.finish()


def test_high_frequency_is_buffered():
    """高频 log 攒批，直到 buffer_size 满才写盘"""
    tracker = _new_tracker(
        "high_freq", buffer_size=10, buffer_interval=0.2, max_buffer_seconds=None
    )
    for i in range(9):
        tracker.log({"v": i})

    # 第一条直接落盘，其余 8 条留在 buffer 中
    assert _count_lines(tracker.log_fp) == 1
    assert len(tracker.buffer) == 8

    for i in range(2):
        tracker.log({"v": 100 + i})
    assert _count_lines(tracker.log_fp) == 11
    assert tracker.buffer == []
    tracker.finish()


def test_max_buffer_seconds_flushes_stale_records():
    """高频写入突然停止时，后台定时器应在超时后写盘"""
    tracker = _new_tracker(
        "timer", buffer_size=1000, buffer_interval=1.0, max_buffer_seconds=0.5
    )
    for i in range(5):
        tracker.log({"v": i})
    assert _count_lines(tracker.log_fp) == 1

    time.sleep(0.9)
    assert _count_lines(tracker.log_fp) == 5
    assert tracker.buffer == []
    tracker.finish()


def test_steps_are_continuous_across_resume():
    tracker = _new_tracker(
        "resume", buffer_size=3, buffer_interval=None, max_buffer_seconds=None
    )
    log_dir = tracker.log_dir.parent.parent
    for i in range(6):
        tracker.log({"v": i})
    tracker.finish()
    assert _count_lines(tracker.log_fp) == 6

    resumed = JsonlTracker()
    resumed.init(project="p", name="resume", dir=str(log_dir))
    assert resumed.current_step == 6
    resumed.log({"v": 6})
    resumed.finish()

    with jsonlines.open(resumed.log_fp) as reader:
        assert [record["_step"] for record in reader] == list(range(7))


def test_concurrent_logging_does_not_lose_records():
    tracker = _new_tracker(
        "threads", buffer_size=25, buffer_interval=0.05, max_buffer_seconds=0.2
    )

    def worker():
        for i in range(200):
            tracker.log({"v": i})

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    tracker.finish()

    assert _count_lines(tracker.log_fp) == 800


def test_numpy_values_are_serialized():
    """numpy 标量/数组不是 int/float 子类，必须被编码后才能写入 jsonl"""
    try:
        import numpy as np
    except ImportError:  # numpy 是可选依赖
        return
    tracker = _new_tracker("numpy", buffer_size=1)
    tracker.log(
        {
            "int32": np.int32(7),
            "float32": np.float32(0.5),
            "bool": np.bool_(True),
            "array": np.array([[1, 2], [3, 4]]),
            "date": np.datetime64("2024-01-01"),
        }
    )
    tracker.finish()

    with jsonlines.open(tracker.log_fp) as reader:
        record = next(iter(reader))
    assert record["int32"] == 7
    assert record["float32"] == 0.5
    assert record["bool"] is True
    assert record["array"] == [[1, 2], [3, 4]]
    assert record["date"] == "2024-01-01"
    assert tracker.buffer == []


def test_unserializable_value_does_not_block_other_records():
    """无法序列化的值降级为 repr，不能污染 buffer 或丢掉后续记录"""

    class Weird:
        __slots__ = ()

        def __repr__(self):
            return "<weird>"

    tracker = _new_tracker("poison", buffer_size=2, buffer_interval=None)
    tracker.log({"v": 0})
    tracker.log({"bad": Weird(), "good": 1})
    tracker.log({"v": 2})
    tracker.finish()

    with jsonlines.open(tracker.log_fp) as reader:
        records = list(reader)
    assert [record["_step"] for record in records] == [0, 1, 2]
    assert records[1] == {"_step": 1, "bad": "<weird>", "good": 1}
    assert tracker.buffer == []


def test_write_failure_retries_without_duplicates():
    """写盘失败时整批回到 buffer，恢复后补写且不产生重复行"""
    tracker = _new_tracker(
        "io_fail", buffer_size=2, buffer_interval=None, max_buffer_seconds=None
    )
    tracker.log({"v": 0})
    assert _count_lines(tracker.log_fp) == 1

    good_fp = tracker.log_fp
    blocked = tracker.log_dir / "blocked"
    blocked.mkdir()
    tracker.log_fp = blocked  # 写入目录必然失败

    tracker.log({"v": 1})
    tracker.log({"v": 2})
    assert len(tracker.buffer) == 2
    assert _count_lines(good_fp) == 1

    tracker.log_fp = good_fp
    tracker.finish()

    with jsonlines.open(good_fp) as reader:
        assert [record["_step"] for record in reader] == [0, 1, 2]


def test_buffer_is_capped_when_writes_keep_failing():
    """持续写失败时 buffer 不应无限增长"""
    tracker = _new_tracker(
        "capped",
        buffer_size=2,
        buffer_interval=None,
        max_buffer_seconds=None,
        max_pending_records=4,
    )
    blocked = tracker.log_dir / "blocked"
    blocked.mkdir()
    tracker.log_fp = blocked

    for i in range(20):
        tracker.log({"v": i})
    assert len(tracker.buffer) <= 4
    # 保留的是最新的记录
    assert tracker.buffer[-1]["v"] == 19


def test_init_without_name_generates_one():
    tracker = JsonlTracker()
    tracker.init(project="p", dir=tempfile.mkdtemp())
    assert tracker.name
    tracker.log({"v": 0})
    tracker.finish()
    assert _count_lines(tracker.log_fp) == 1


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"{_name} OK")
    print("ALL OK")
