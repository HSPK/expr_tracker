"""Streams: independent producers writing their own file under one run.

A data worker and a training loop have unrelated step semantics — batch 100 and
training step 100 mean different things — so each gets its own file, cursor and
resume state, while sharing the run directory.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import expr_tracker as et
from expr_tracker.history import HistoryStore, list_streams, read_history
from expr_tracker.history.naming import (
    metrics_filename,
    parse_stream,
    sidecar_filename,
    validate_stream,
)
from expr_tracker.history.reader import resolve_run_path
from expr_tracker.run import Run

SRC = str(Path("src").resolve())


@pytest.fixture
def store(tmp_path):
    created = []

    def factory(stream=None, name="r", **options):
        instance = HistoryStore()
        options.setdefault("max_open_seconds", None)
        instance.init(
            project="s", name=name, dir=str(tmp_path), stream=stream, **options
        )
        created.append(instance)
        return instance

    yield factory
    for instance in created:
        instance.finish()


def run_dir(tmp_path, name="r"):
    return tmp_path / "s" / name


# ------------------------------------------------------------------ naming


@pytest.mark.parametrize(
    ("stream", "rank", "expected"),
    [
        (None, 0, "metrics.jsonl"),
        ("data", 0, "metrics.data.jsonl"),
        (None, 2, "metrics.rank2.jsonl"),
        ("data", 2, "metrics.data.rank2.jsonl"),
        ("eval-1", 0, "metrics.eval-1.jsonl"),
    ],
)
def test_filenames_compose_stream_and_rank(monkeypatch, stream, rank, expected):
    monkeypatch.setenv("RANK", str(rank))
    assert metrics_filename(stream, rank_aware=True) == expected


def test_rank_awareness_can_be_disabled_per_stream(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    assert metrics_filename("data", rank_aware=False) == "metrics.data.jsonl"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("metrics.jsonl", None),
        ("metrics.data.jsonl", "data"),
        ("metrics.rank2.jsonl", None),
        ("metrics.data.rank1.jsonl", "data"),
        ("metrics.eval-1.rank9.jsonl", "eval-1"),
    ],
)
def test_a_filename_reports_its_stream(filename, expected):
    assert parse_stream(filename) == expected


def test_sidecars_are_named_per_stream():
    assert sidecar_filename("summary", None, "json") == "summary.json"
    assert sidecar_filename("summary", "data", "json") == "summary.data.json"
    assert sidecar_filename("config", "data", "json") == "config.data.json"


@pytest.mark.parametrize("name", ["train", "data", "eval-1", "a_b", "x9", "A"])
def test_valid_stream_names(name):
    assert validate_stream(name) == name


@pytest.mark.parametrize(
    "name", ["a/b", "a.b", "../etc", "", " ", "a b", "_lead", "-lead", "a\\b", "é"]
)
def test_a_stream_name_must_be_filename_safe(name):
    with pytest.raises(ValueError, match="Invalid stream name"):
        validate_stream(name)


@pytest.mark.parametrize("name", ["rank0", "rank1", "rank42"])
def test_a_stream_may_not_impersonate_a_rank_shard(name):
    """`metrics.rank1.jsonl` already means rank 1, not a stream called rank1."""
    with pytest.raises(ValueError, match="collides with the rank shard"):
        validate_stream(name)


def test_an_invalid_stream_is_rejected_at_init(tmp_path):
    with pytest.raises(ValueError, match="Invalid stream name"):
        HistoryStore().init(project="s", name="r", dir=str(tmp_path), stream="a/b")


# ------------------------------------------------------------------ isolation


def test_each_stream_writes_its_own_file(store, tmp_path):
    train, data = store(), store("data")
    for step in range(5):
        train.log({"train/loss": float(step)})
    for batch in range(3):
        data.log({"data/produce_ms": float(batch)})
    train.flush(commit_open=True)
    data.flush(commit_open=True)

    assert train.log_fp.name == "metrics.jsonl"
    assert data.log_fp.name == "metrics.data.jsonl"
    assert [r["_step"] for r in train.get(-1)] == [0, 1, 2, 3, 4]
    assert [r["_step"] for r in data.get(-1)] == [0, 1, 2]
    assert all("data/produce_ms" not in r for r in train.get(-1))
    assert all("train/loss" not in r for r in data.get(-1))


def test_step_cursors_are_independent(store):
    """The whole point: batch 100 and training step 100 are unrelated."""
    train, data = store(), store("data")
    for _ in range(10):
        train.log({"loss": 1.0})
    data.log({"produce_ms": 1.0})
    assert train.current_step == 10
    assert data.current_step == 1  # not dragged along by the training loop


def test_interleaved_writes_do_not_drop_each_other(store):
    """One cursor would treat the other producer's step as a backward write."""
    train, data = store(), store("data")
    for step in range(20):
        train.log({"loss": float(step)})
        data.log({"produce_ms": float(step)})
    train.flush(commit_open=True)
    data.flush(commit_open=True)
    assert len(train.get(-1)) == 20 and len(data.get(-1)) == 20


def test_each_stream_resumes_its_own_cursor(store, tmp_path):
    train, data = store(), store("data")
    for _ in range(7):
        train.log({"loss": 1.0})
    for _ in range(3):
        data.log({"produce_ms": 1.0})
    train.finish()
    data.finish()

    assert store().current_step == 7
    assert store("data").current_step == 3


def test_streams_compose_with_rank_shards(store, tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    data = store("data")
    data.log({"produce_ms": 1.0})
    data.flush(commit_open=True)
    assert data.log_fp.name == "metrics.data.rank1.jsonl"


def test_sidecars_do_not_clobber_each_other(tmp_path):
    """Two processes in one run directory must not overwrite each other's state."""
    train = Run(
        project="s",
        name="r",
        dir=str(tmp_path),
        backends=[],
        max_open_seconds=None,
        config={"lr": 0.1},
    )
    train.summary["best"] = 1
    train.log({"loss": 1.0})
    train.finish()

    data = Run(
        project="s",
        name="r",
        dir=str(tmp_path),
        backends=[],
        stream="data",
        max_open_seconds=None,
        config={"workers": 8},
    )
    data.summary["rows"] = 999
    data.log({"produce_ms": 1.0})
    data.finish()

    directory = run_dir(tmp_path)
    assert json.loads((directory / "summary.json").read_text())["best"] == 1
    assert json.loads((directory / "summary.data.json").read_text())["rows"] == 999
    assert json.loads((directory / "config.json").read_text())["lr"] == 0.1
    assert json.loads((directory / "config.data.json").read_text())["workers"] == 8


# ------------------------------------------------------------------ resolution


def test_the_default_file_wins_over_a_stream(store, tmp_path):
    """`metrics.data.jsonl` sorts before `metrics.jsonl`; order must not decide."""
    train, data = store(), store("data")
    train.log({"loss": 1.0})
    data.log({"produce_ms": 2.0})
    train.finish()
    data.finish()

    directory = run_dir(tmp_path)
    assert resolve_run_path(directory).name == "metrics.jsonl"
    assert resolve_run_path(directory, "data").name == "metrics.data.jsonl"
    assert read_history(directory, -1)[0]["loss"] == 1.0


def test_resolution_finds_a_rank_shard_when_rank_zero_is_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "2")
    instance = HistoryStore()
    instance.init(project="s", name="r", dir=str(tmp_path), max_open_seconds=None)
    instance.log({"loss": 1.0})
    instance.finish()
    assert resolve_run_path(run_dir(tmp_path)).name == "metrics.rank2.jsonl"


def test_an_unknown_stream_lists_what_exists(store, tmp_path):
    train = store()
    train.log({"loss": 1.0})
    train.finish()
    with pytest.raises(FileNotFoundError, match=r"available: metrics\.jsonl"):
        resolve_run_path(run_dir(tmp_path), "nope")


def test_streams_are_listed(store, tmp_path, monkeypatch):
    for stream in (None, "data", "eval"):
        instance = store(stream)
        instance.log({"m": 1.0})
        instance.finish()
    assert list_streams(run_dir(tmp_path)) == [None, "data", "eval"]


def test_listing_ignores_rank_shards(store, tmp_path, monkeypatch):
    train = store()
    train.log({"m": 1.0})
    train.finish()
    monkeypatch.setenv("RANK", "1")
    shard = store("data")
    shard.log({"m": 1.0})
    shard.finish()
    assert list_streams(run_dir(tmp_path)) == [None, "data"]


# ------------------------------------------------------------------ public api


@pytest.fixture
def active(tmp_path):
    created = []

    def factory(stream=None, **options):
        options.setdefault("max_open_seconds", None)
        created.append(
            et.init(
                project="s",
                name="r",
                dir=str(tmp_path),
                backends=[],
                stream=stream,
                **options,
            )
        )
        return created[-1]

    yield factory
    if et.get_run() is not None:
        et.finish()


def test_the_run_reports_its_stream(active):
    assert active().stream is None
    et.finish()
    assert active("data").stream == "data"


def test_history_defaults_to_the_running_stream(active):
    active("data")
    for i in range(4):
        et.log({"produce_ms": float(i)})
    assert [r["_step"] for r in et.history(-1)] == [0, 1, 2, 3]


def test_history_can_read_another_stream(tmp_path, active):
    active()
    for i in range(5):
        et.log({"train/loss": float(i)})
    et.finish()

    active("data")
    et.log({"data/produce_ms": 9.0})
    assert [r["train/loss"] for r in et.history(-1, stream=None)] == [0, 1, 2, 3, 4]
    assert et.history(-1)[0]["data/produce_ms"] == 9.0


def test_history_of_a_named_stream_from_the_default_run(tmp_path, active):
    active("data")
    for i in range(3):
        et.log({"produce_ms": float(i)})
    et.finish()

    active()
    et.log({"loss": 1.0})
    assert len(et.history(-1, stream="data")) == 3
    assert len(et.history(-1)) == 1


def test_offline_reads_select_the_stream(tmp_path, active):
    active()
    et.log({"loss": 1.0})
    et.finish()
    active("data")
    et.log({"produce_ms": 2.0})
    et.finish()

    directory = str(run_dir(tmp_path))
    assert et.history(-1, run=directory)[0]["loss"] == 1.0
    assert et.history(-1, run=directory, stream="data")[0]["produce_ms"] == 2.0


def test_alerts_are_independent_per_stream(tmp_path):
    """Each process alerts on what it can see, which is its own stream."""
    received: list = []
    channel = {
        "channels": [
            {
                "type": "callable",
                "name": "c",
                "options": {"handler": received.append},
                "policy": {"async_send": False, "dedup_window": 0},
            }
        ]
    }
    data = Run(
        project="s",
        name="r",
        dir=str(tmp_path),
        backends=[],
        stream="data",
        max_open_seconds=None,
        alert=channel,
        alert_rules=["produce_ms > 100 => warning: data pipeline slow"],
    )
    try:
        data.log({"produce_ms": 5.0})
        assert received == []
        data.log({"produce_ms": 500.0})
        assert [m.text for m in received] == ["data pipeline slow"]
    finally:
        data.finish()


# ------------------------------------------------------------------ backends


class Recorder:
    def __init__(self):
        self.calls = []

    def init(self, **kwargs):
        self.calls.append(kwargs)

    def log(self, data, step=None, commit=None):
        pass

    def finish(self, **kwargs):
        pass


def test_a_stream_becomes_a_grouped_backend_run(tmp_path):
    backend = Recorder()
    run = Run(
        project="s",
        name="r",
        dir=str(tmp_path),
        backends=[backend],
        stream="data",
        max_open_seconds=None,
    )
    try:
        kwargs = backend.calls[0]
        assert run.backend_run_name == "r-data"
        assert kwargs["name"] == "r-data" and kwargs["id"] == "r-data"
        assert kwargs["group"] == "r"  # ties the streams back together
        assert kwargs["job_type"] == "data"
    finally:
        run.finish()


def test_the_default_stream_is_not_grouped(tmp_path):
    backend = Recorder()
    run = Run(
        project="s",
        name="r",
        dir=str(tmp_path),
        backends=[backend],
        max_open_seconds=None,
    )
    try:
        assert run.backend_run_name == "r"
        assert backend.calls[0]["name"] == "r"
        assert "group" not in backend.calls[0]
        assert "job_type" not in backend.calls[0]
    finally:
        run.finish()


def test_trackio_streams_are_grouped(tmp_path, monkeypatch):
    module = Recorder()
    monkeypatch.setitem(sys.modules, "trackio", module)
    run = Run(
        project="s",
        name="r",
        dir=str(tmp_path),
        backends=["trackio"],
        stream="data",
        max_open_seconds=None,
    )
    try:
        kwargs = module.calls[0]
        assert kwargs["name"] == "r-data" and kwargs["group"] == "r"
    finally:
        run.finish()


def test_backend_kwargs_still_win(tmp_path):
    backend = Recorder()
    run = Run(
        project="s",
        name="r",
        dir=str(tmp_path),
        backends=[backend],
        stream="data",
        max_open_seconds=None,
        backend_kwargs={"recorder": {"group": "mine"}},
    )
    try:
        assert backend.calls[0]["group"] == "mine"
    finally:
        run.finish()


# ------------------------------------------------------------------ processes


def test_separate_processes_share_a_run_directory(tmp_path):
    """The real deployment: a training process and a data worker process."""
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {SRC!r})
        import expr_tracker as et
        stream = sys.argv[1] or None
        et.init(project="s", name="r", dir={str(tmp_path)!r}, backends=[], stream=stream)
        for step in range(30):
            et.log({{f"{{stream or 'train'}}/value": float(step)}})
        et.finish()
    """)
    processes = [
        subprocess.Popen([sys.executable, "-c", script, name]) for name in ("", "data")
    ]
    for process in processes:
        assert process.wait(timeout=120) == 0

    directory = run_dir(tmp_path)
    assert list_streams(directory) == [None, "data"]
    train = read_history(directory, -1)
    data = read_history(directory, -1, stream="data")
    assert [r["_step"] for r in train] == list(range(30))
    assert [r["_step"] for r in data] == list(range(30))
    assert train[5]["train/value"] == 5.0
    assert data[5]["data/value"] == 5.0
    assert all("data/value" not in r for r in train)


def test_a_worker_can_resume_its_own_stream_across_processes(tmp_path):
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {SRC!r})
        import expr_tracker as et
        et.init(project="s", name="r", dir={str(tmp_path)!r}, backends=[], stream="data")
        for step in range(10):
            et.log({{"produce_ms": float(step)}})
        print(et.get_run().step)
        et.finish()
    """)
    for expected_start in (0, 10):
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert int(result.stdout.strip().splitlines()[-1]) == expected_start + 10

    rows = read_history(run_dir(tmp_path), -1, stream="data")
    assert [r["_step"] for r in rows] == list(range(20))
