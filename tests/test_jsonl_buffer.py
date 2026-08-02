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


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"{_name} OK")
    print("ALL OK")
