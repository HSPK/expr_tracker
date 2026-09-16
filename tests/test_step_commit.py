"""Step commit semantics: merging, auto-increment, step policy, patch lines, resume."""

import json
import threading
import time
from unittest.mock import Mock

import pytest

from expr_tracker.history import HistoryStore


def read_lines(store):
    if not store.log_fp.exists():
        return []
    with open(store.log_fp, "rb") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_auto_increment_commits_each_call(store):
    s = store()
    for i in range(3):
        s.log({"v": i})
    s.flush()
    records = read_lines(s)
    assert [r["_step"] for r in records] == [0, 1, 2]
    assert [r["v"] for r in records] == [0, 1, 2]
    assert all("_time" in r for r in records)


def test_same_step_merges_into_one_row(store):
    s = store()
    s.log({"loss": 1.0}, step=5)
    s.log({"acc": 0.5}, step=5)
    s.flush(commit_open=True)
    records = read_lines(s)
    assert records == [
        {"_step": 5, "_time": records[0]["_time"], "loss": 1.0, "acc": 0.5}
    ]


def test_step_advance_commits_previous_row(store):
    s = store()
    s.log({"loss": 1.0}, step=0)
    assert read_lines(s) == []  # still the open row
    s.log({"loss": 2.0}, step=1)
    s.flush()
    assert [r["_step"] for r in read_lines(s)] == [0]
    s.flush(commit_open=True)
    assert [r["_step"] for r in read_lines(s)] == [0, 1]


def test_commit_false_accumulates(store):
    s = store()
    s.log({"a": 1}, commit=False)
    s.log({"b": 2})
    s.flush()
    records = read_lines(s)
    assert len(records) == 1
    assert records[0]["a"] == 1 and records[0]["b"] == 2
    assert s.current_step == 1


@pytest.mark.parametrize("step", [0, 10])
def test_implicit_logs_reuse_an_explicit_open_step(store, step):
    s = store()
    assert s.log({"a": 1}, step=step) == step
    assert s.log({"b": 2}, commit=False) == step
    assert read_lines(s) == []
    assert s.current_step == step
    assert s.log({"c": 3}) == step
    assert s.current_step == step + 1
    assert s.log({"d": 4}) == step + 1
    s.flush()
    rows = [{k: v for k, v in row.items() if k != "_time"} for row in read_lines(s)]
    assert rows == [
        {"_step": step, "a": 1, "b": 2, "c": 3},
        {"_step": step + 1, "d": 4},
    ]


def test_implicit_logs_follow_a_jump_that_committed_the_previous_row(store):
    s = store()
    s.log({"a": 1}, commit=False)
    s.log({"b": 2}, step=10)
    assert s.log({"c": 3}) == 10
    assert s.current_step == 11
    s.flush()
    assert [row["_step"] for row in read_lines(s)] == [0, 10]
    assert s.get(1)[0]["b"] == 2 and s.get(1)[0]["c"] == 3
    assert s.log({"rejected": True}, step=9) is None


def test_implicit_logs_reuse_an_allowed_backward_open_step(store):
    s = store(step_policy="allow")
    s.log({"a": 1}, step=10, commit=True)
    s.log({"b": 2}, step=3)
    assert s.log({"c": 3}) == 3
    assert s.current_step == 4
    s.flush()
    assert [row["_step"] for row in read_lines(s)] == [10, 3]
    assert s.get(-1)[0]["b"] == 2 and s.get(-1)[0]["c"] == 3


def test_backward_step_is_dropped_by_default(store):
    s = store()
    s.log({"v": 0}, step=10)
    s.log({"v": 1}, step=5)
    s.flush(commit_open=True)
    assert [r["_step"] for r in read_lines(s)] == [10]


def test_backward_step_allowed_by_policy(store):
    s = store(name="allow", step_policy="allow")
    s.log({"v": 0}, step=10)
    s.log({"v": 1}, step=5)
    s.flush(commit_open=True)
    assert sorted(r["_step"] for r in read_lines(s)) == [5, 10]
    # Reads merge and sort by step
    assert [r["_step"] for r in s.get(-1)] == [5, 10]


@pytest.mark.parametrize("step", [None, 3])
@pytest.mark.parametrize("commit_via", ["log", "advance", "flush", "finish", "exit"])
def test_long_step_waits_for_a_commit(store, monkeypatch, step, commit_via):
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    timer = Mock()
    monkeypatch.setattr(threading, "Timer", timer)
    committed = []
    s = store(buffer_size=1, on_commit=committed.append)
    expected_step = 0 if step is None else step

    s.log({"a": 1}, step=step, commit=False)
    now += 24 * 60 * 60
    s.flush()
    s.log({"b": 2}, step=step, commit=False)

    timer.assert_not_called()
    assert s.current_step == expected_step
    assert read_lines(s) == []
    assert s.get(-1, include_open=False) == []
    assert committed == []
    assert s.open_record() == {
        "_step": expected_step,
        "_time": now - 24 * 60 * 60,
        "a": 1,
        "b": 2,
    }

    if commit_via == "log":
        s.log({}, step=step, commit=True)
    elif commit_via == "advance":
        s.log({"c": 3}, step=expected_step + 1)
    elif commit_via == "flush":
        s.flush(commit_open=True)
    elif commit_via == "finish":
        s.finish()
    else:
        s._atexit()

    assert s.current_step == expected_step + 1
    assert read_lines(s) == committed
    assert len(committed) == 1
    assert committed[0]["_step"] == expected_step
    assert committed[0]["a"] == 1 and committed[0]["b"] == 2


def test_explicit_commit_writes_patch_line(store):
    s = store()
    s.log({"a": 1}, step=3)
    s.flush(commit_open=True)
    assert [r["_step"] for r in read_lines(s)] == [3]
    s.log({"b": 2}, step=3)  # late data for the same step: a patch line
    s.flush(commit_open=True)
    assert len(read_lines(s)) == 2
    merged = s.get(-1)
    assert merged == [{"_step": 3, "_time": merged[0]["_time"], "a": 1, "b": 2}]


def test_resume_continues_from_last_step(tmp_path):
    first = HistoryStore().init(project="p", name="resume", dir=str(tmp_path))
    for i in range(6):
        first.log({"v": i})
    first.finish()

    second = HistoryStore().init(project="p", name="resume", dir=str(tmp_path))
    assert second.current_step == 6
    second.log({"v": 6})
    second.finish()
    assert [r["_step"] for r in read_lines(second)] == list(range(7))


def test_resume_backfills_metric_series(tmp_path):
    first = HistoryStore().init(project="p", name="series", dir=str(tmp_path))
    for i in range(5):
        first.log({"loss": float(i)})
    first.finish()

    second = HistoryStore().init(project="p", name="series", dir=str(tmp_path))
    assert [p[2] for p in second.series.points("loss")] == [0.0, 1.0, 2.0, 3.0, 4.0]
    second.finish()


def test_duplicate_key_keeps_latest(store):
    s = store()
    s.log({"v": 1}, step=0)
    s.log({"v": 2}, step=0)
    s.flush(commit_open=True)
    assert read_lines(s)[0]["v"] == 2


def test_open_row_is_reused_until_the_step_moves(store):
    """_aim_open_row keeps one row per step, whichever way the step is chosen."""
    s = store()
    s.log({"a": 1}, commit=False)
    s.log({"b": 2})  # no step given: must land on the same open row
    s.log({"c": 3}, step=5)
    s.log({"d": 4}, step=5)  # same explicit step: same row again
    s.log({"e": 5}, step=6)  # step moves: previous row is committed
    s.flush(commit_open=True)
    rows = [{k: v for k, v in r.items() if k != "_time"} for r in s.get(-1)]
    assert rows == [
        {"_step": 0, "a": 1, "b": 2},
        {"_step": 5, "c": 3, "d": 4},
        {"_step": 6, "e": 5},
    ]
