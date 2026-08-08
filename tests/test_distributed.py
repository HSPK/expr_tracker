"""Distributed training: per-rank shards and rank-aware alerting.

Ranks come from ``RANK``/``LOCAL_RANK``. Each non-zero rank writes its own file so
concurrent appends from several processes cannot interleave.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from expr_tracker.history import HistoryStore, read_history, resolve_run_path
from expr_tracker.history.naming import current_rank, metrics_filename
from expr_tracker.run import Run


@pytest.fixture
def rank(monkeypatch):
    def set_rank(value, variable="RANK"):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        if value is not None:
            monkeypatch.setenv(variable, str(value))

    return set_rank


def store_for(tmp_path, name="run", **options):
    store = HistoryStore()
    options.setdefault("max_open_seconds", None)
    store.init(project="dist", name=name, dir=str(tmp_path), **options)
    return store


# ------------------------------------------------------------------ rank source


@pytest.mark.parametrize("variable", ["RANK", "LOCAL_RANK"])
def test_the_rank_comes_from_either_variable(rank, variable):
    rank(3, variable)
    assert current_rank() == 3


def test_rank_defaults_to_zero_when_unset(rank):
    rank(None)
    assert current_rank() == 0


@pytest.mark.parametrize("value", ["", "abc", "1.5", "rank2", " "])
def test_an_unusable_rank_falls_back_to_zero(rank, value):
    rank(value)
    assert current_rank() == 0


def test_rank_takes_priority_over_local_rank(monkeypatch):
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert current_rank() == 7


def test_a_negative_rank_is_treated_as_the_main_shard(rank):
    rank(-1)
    assert current_rank() == -1
    assert metrics_filename(None, rank_aware=True) == "metrics.jsonl"


# ------------------------------------------------------------------ file names


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "metrics.jsonl"),
        (1, "metrics.rank1.jsonl"),
        (7, "metrics.rank7.jsonl"),
        (None, "metrics.jsonl"),
    ],
)
def test_each_rank_gets_its_own_file(rank, value, expected):
    rank(value)
    assert metrics_filename(None, rank_aware=True) == expected


def test_rank_awareness_can_be_switched_off(rank):
    rank(5)
    assert metrics_filename(None, rank_aware=False) == "metrics.jsonl"


def test_a_worker_rank_writes_to_its_own_shard(rank, tmp_path):
    rank(2)
    store = store_for(tmp_path)
    try:
        store.log({"loss": 1.0})
        store.flush(commit_open=True)
        assert store.log_fp.name == "metrics.rank2.jsonl"
    finally:
        store.finish()
    assert (tmp_path / "dist" / "run" / "metrics.rank2.jsonl").is_file()
    assert not (tmp_path / "dist" / "run" / "metrics.jsonl").exists()


def test_disabling_rank_awareness_shares_one_file(rank, tmp_path):
    rank(3)
    store = store_for(tmp_path, rank_aware=False)
    try:
        store.log({"loss": 1.0})
        store.flush(commit_open=True)
        assert store.log_fp.name == "metrics.jsonl"
    finally:
        store.finish()


# ------------------------------------------------------------------ isolation


def test_two_ranks_do_not_interfere(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    main = store_for(tmp_path, "shared")
    monkeypatch.setenv("RANK", "1")
    worker = store_for(tmp_path, "shared")
    try:
        for step in range(20):
            main.log({"loss": float(step)})
            worker.log({"loss": float(step) * 10})
        main.flush(commit_open=True)
        worker.flush(commit_open=True)

        assert [r["loss"] for r in main.get(-1)] == [float(s) for s in range(20)]
        assert [r["loss"] for r in worker.get(-1)] == [float(s) * 10 for s in range(20)]
    finally:
        main.finish()
        worker.finish()

    run_dir = tmp_path / "dist" / "shared"
    assert {p.name for p in run_dir.glob("metrics*.jsonl")} == {
        "metrics.jsonl",
        "metrics.rank1.jsonl",
    }
    for path in run_dir.glob("metrics*.jsonl"):
        for line in path.read_text().splitlines():
            json.loads(line)  # no interleaved half-lines


def test_each_shard_keeps_its_own_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    main = store_for(tmp_path, "meta")
    monkeypatch.setenv("RANK", "1")
    worker = store_for(tmp_path, "meta")
    try:
        for step in range(5):
            main.log({"a": step})
        for step in range(9):
            worker.log({"b": step})
        main.flush(commit_open=True)
        worker.flush(commit_open=True)
    finally:
        main.finish()
        worker.finish()

    run_dir = tmp_path / "dist" / "meta"
    assert json.loads((run_dir / "metrics.meta.json").read_text())["lines"] == 5
    assert json.loads((run_dir / "metrics.rank1.meta.json").read_text())["lines"] == 9


def test_offline_reading_defaults_to_the_main_shard(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    main = store_for(tmp_path, "read")
    monkeypatch.setenv("RANK", "1")
    worker = store_for(tmp_path, "read")
    try:
        for _ in range(4):
            main.log({"who": "main"})
            worker.log({"who": "worker"})
        main.flush(commit_open=True)
        worker.flush(commit_open=True)
    finally:
        main.finish()
        worker.finish()

    run_dir = tmp_path / "dist" / "read"
    assert resolve_run_path(run_dir).name == "metrics.jsonl"
    assert all(r["who"] == "main" for r in read_history(run_dir, -1))
    # a shard is still readable when addressed directly
    shard = read_history(run_dir / "metrics.rank1.jsonl", -1)
    assert all(r["who"] == "worker" for r in shard)


def test_a_worker_rank_resumes_its_own_shard(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "2")
    first = store_for(tmp_path, "resume")
    for step in range(6):
        first.log({"loss": float(step)})
    first.finish()

    second = store_for(tmp_path, "resume")
    try:
        assert second.current_step == 6
        assert len(second.get(-1)) == 6
    finally:
        second.finish()


# ------------------------------------------------------------------ alerts


def alert_config(sink):
    return {
        "channels": [
            {
                "type": "callable",
                "name": "c",
                "options": {"handler": sink.append},
                "policy": {"async_send": False, "dedup_window": 0},
            }
        ]
    }


def test_only_the_alerting_rank_sends(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    main_sink: list = []
    main = Run(
        project="dist",
        name="alert0",
        dir=str(tmp_path),
        backends=[],
        alert=alert_config(main_sink),
        alert_rules=["loss > 0 => warning: fire"],
    )
    monkeypatch.setenv("RANK", "1")
    worker_sink: list = []
    worker = Run(
        project="dist",
        name="alert1",
        dir=str(tmp_path),
        backends=[],
        alert=alert_config(worker_sink),
        alert_rules=["loss > 0 => warning: fire"],
    )
    try:
        main.log({"loss": 1.0})
        worker.log({"loss": 1.0})
        assert len(main_sink) == 1
        assert worker_sink == []  # rank 1 stays quiet
        assert main.rank == 0 and worker.rank == 1
    finally:
        main.finish()
        worker.finish()


def test_alerting_can_be_pinned_to_another_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    sink: list = []
    run = Run(
        project="dist",
        name="alert3",
        dir=str(tmp_path),
        backends=[],
        alert=alert_config(sink),
        alert_rules=["loss > 0 => warning: fire"],
        alert_on_rank=3,
    )
    try:
        run.log({"loss": 1.0})
        assert len(sink) == 1
    finally:
        run.finish()


def test_alerting_on_every_rank_is_possible(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "4")
    sink: list = []
    run = Run(
        project="dist",
        name="alertall",
        dir=str(tmp_path),
        backends=[],
        alert=alert_config(sink),
        alert_rules=["loss > 0 => warning: fire"],
        alert_on_rank=None,
    )
    try:
        run.log({"loss": 1.0})
        assert len(sink) == 1
    finally:
        run.finish()


def test_a_silenced_rank_still_records_history_and_rules(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    sink: list = []
    run = Run(
        project="dist",
        name="silent",
        dir=str(tmp_path),
        backends=[],
        alert=alert_config(sink),
        alert_rules=["loss > 0 => warning: fire"],
    )
    try:
        for _ in range(10):
            run.log({"loss": 1.0})
        assert len(run.history_query(-1)) == 10  # history is never silenced
        assert run.alerts is not None  # rules can still be added and inspected
        assert run.info()["rank"] == 1
        assert sink == []
    finally:
        run.finish()


# ------------------------------------------------------------------ processes


def test_separate_processes_write_separate_shards(tmp_path):
    """The real deployment: N processes, one per rank, sharing a run directory."""
    script = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {str(Path("src").resolve())!r})
        import expr_tracker as et
        rank = int(os.environ["RANK"])
        et.init(project="dist", name="mp", dir={str(tmp_path)!r}, backends=[])
        for step in range(50):
            et.log({{"loss": float(step * 100 + rank)}})
        et.finish()
        """
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script], env={**dict(os.environ), "RANK": str(rank)}
        )
        for rank in range(4)
    ]
    for process in processes:
        assert process.wait(timeout=120) == 0

    run_dir = tmp_path / "dist" / "mp"
    shards = sorted(p.name for p in run_dir.glob("metrics*.jsonl"))
    assert shards == [
        "metrics.jsonl",
        "metrics.rank1.jsonl",
        "metrics.rank2.jsonl",
        "metrics.rank3.jsonl",
    ]
    for rank, shard in enumerate(shards):
        rows = read_history(run_dir / shard, -1)
        assert [r["_step"] for r in rows] == list(range(50))
        assert rows[7]["loss"] == 7 * 100 + rank
