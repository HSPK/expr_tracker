"""Failure modes: the tracker degrades, it never takes the training loop down.

Every test here asserts two things -- the call survives, and the data that *could*
be kept was kept.
"""

import errno
import json
import os
import stat
import threading

import pytest

import expr_tracker as et
from expr_tracker.alerts.expr import ExprError, ExprSyntaxError
from expr_tracker.history import HistoryStore, read_history
from expr_tracker.run import Run


@pytest.fixture
def make(tmp_path):
    created = []

    def factory(name="run", **options):
        store = HistoryStore()
        options.setdefault("max_open_seconds", None)
        store.init(project="fail", name=name, dir=str(tmp_path), **options)
        created.append(store)
        return store

    yield factory
    for store in created:
        store.finish()


def breaks(exception):
    def raise_it(*args, **kwargs):
        raise exception

    return raise_it


# ------------------------------------------------------------------ disk


@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.ENOSPC, "No space left on device"),
        OSError(errno.EACCES, "Permission denied"),
        OSError(errno.EIO, "I/O error"),
        OSError(errno.EROFS, "Read-only file system"),
        RuntimeError("something exotic"),
    ],
)
def test_logging_survives_any_write_error(make, monkeypatch, error):
    store = make(buffer_size=1)
    monkeypatch.setattr(store.writer, "_write", breaks(error))
    for step in range(20):
        store.log({"loss": float(step)})
    store.flush(commit_open=True)

    assert [r["_step"] for r in store.get(-1)] == list(range(20))  # memory holds
    monkeypatch.undo()
    store.flush()
    assert [r["_step"] for r in read_history(store.log_dir, -1)] == list(range(20))


def test_a_read_only_directory_does_not_stop_the_run(tmp_path, monkeypatch):
    run_dir = tmp_path / "fail" / "readonly"
    run_dir.mkdir(parents=True)
    store = HistoryStore()
    store.init(project="fail", name="readonly", dir=str(tmp_path), buffer_size=1)
    try:
        os.chmod(run_dir, stat.S_IRUSR | stat.S_IXUSR)
        try:
            for step in range(10):
                store.log({"loss": float(step)})
            store.flush(commit_open=True)
            assert [r["_step"] for r in store.get(-1)] == list(range(10))
        finally:
            os.chmod(run_dir, stat.S_IRWXU)
    finally:
        store.finish()


def test_a_deleted_file_mid_run_is_recreated(make):
    store = make(buffer_size=1)
    for step in range(5):
        store.log({"loss": float(step)})
    store.flush()
    store.log_fp.unlink()

    for step in range(5, 10):
        store.log({"loss": float(step)})
    store.flush(commit_open=True)
    assert store.log_fp.is_file()
    assert [r["_step"] for r in store.get(-1)] == list(range(10))


def test_a_disk_read_failure_falls_back_to_the_cache(make, monkeypatch):
    store = make(cache_bytes=400)
    for step in range(100):
        store.log({"loss": float(step), "pad": "x" * 30})
    store.flush(commit_open=True)
    assert store.stats()["evicted_rows"] > 0

    monkeypatch.setattr(store.writer, "reader", breaks(OSError("gone")))
    rows = store.get(-1)  # needs the disk prefix, which is now unreadable
    assert rows and [r["_step"] for r in rows] == sorted(r["_step"] for r in rows)
    assert rows[-1]["_step"] == 99


def test_a_flush_failure_does_not_block_finish(make, monkeypatch):
    store = make(buffer_size=1)
    store.log({"loss": 1.0})
    monkeypatch.setattr(store.writer, "flush", breaks(OSError("nope")))
    store.finish()  # must not raise


def test_a_corrupt_sidecar_does_not_stop_a_resume(make, tmp_path):
    store = make("meta")
    for step in range(10):
        store.log({"loss": float(step)})
    store.finish()
    (tmp_path / "fail" / "meta" / "metrics.meta.json").write_text("{ broken")

    resumed = HistoryStore()
    resumed.init(project="fail", name="meta", dir=str(tmp_path))
    try:
        assert resumed.current_step == 10
        assert len(resumed.get(-1)) == 10
    finally:
        resumed.finish()


def test_a_config_write_failure_is_not_fatal(tmp_path, monkeypatch):
    run_dir = tmp_path / "fail" / "cfg"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").mkdir()  # a directory where the file should go
    store = HistoryStore()
    store.init(project="fail", name="cfg", dir=str(tmp_path), config={"lr": 0.1})
    try:
        store.log({"loss": 1.0})
        store.flush(commit_open=True)
        assert len(store.get(-1)) == 1
    finally:
        store.finish()


# ------------------------------------------------------------------ encoding


def test_an_object_with_attributes_is_encoded_as_its_dict(make):
    """The vendored encoder mirrors fastapi: objects become their ``__dict__``."""

    class Point:
        def __init__(self):
            self.x, self.y = 1, 2

    store = make()
    store.log({"good": 1.0, "obj": Point()})
    store.flush(commit_open=True)
    row = store.get(-1)[0]
    assert row["good"] == 1.0 and row["obj"] == {"x": 1, "y": 2}


def test_a_value_the_encoder_cannot_handle_falls_back_to_repr(make):
    store = make()
    store.log({"good": 1.0, "bad": object()})
    store.flush(commit_open=True)
    row = store.get(-1)[0]
    assert row["good"] == 1.0
    assert row["bad"].startswith("<object object at")


def test_a_value_whose_repr_explodes_is_still_logged(make):
    class Hostile:
        __slots__ = ()  # no __dict__, so the encoder must fall back to repr

        def __repr__(self):
            raise ValueError("no repr for you")

    store = make()
    store.log({"good": 1.0, "bad": Hostile()})
    store.flush(commit_open=True)
    row = store.get(-1)[0]
    assert row["good"] == 1.0 and isinstance(row["bad"], str)


def test_a_self_referential_value_does_not_hang(make):
    loop: dict = {}
    loop["self"] = loop
    store = make()
    store.log({"loop": loop, "ok": 1})
    store.flush(commit_open=True)
    assert store.get(-1)[0]["ok"] == 1


def test_logging_a_non_dict_is_rejected_cleanly(make):
    store = make()
    for bad in ([1, 2], "text", 42, None):
        store.log(bad)
    store.flush(commit_open=True)
    assert all(isinstance(r["_step"], int) for r in store.get(-1))


def test_the_encoder_warns_once_per_key(make, monkeypatch):
    warnings = []
    store = make()
    monkeypatch.setattr(
        store._codec, "warn_once", lambda key, msg: warnings.append(key)
    )
    for _ in range(10):
        store.log({"bad": object()})
    store.flush(commit_open=True)
    assert warnings == ["bad"] * 10  # the codec is asked once per value...
    assert len(set(warnings)) == 1  # ...and logs for one key only


# ------------------------------------------------------------------ alerts


def test_an_exploding_alert_backend_does_not_stop_training(tmp_path):
    def explode(message):
        raise RuntimeError("channel down")

    run = Run(
        project="fail",
        name="alerts",
        dir=str(tmp_path),
        backends=[],
        alert={
            "channels": [
                {
                    "type": "callable",
                    "name": "c",
                    "options": {"handler": explode},
                    "policy": {
                        "async_send": False,
                        "dedup_window": 0,
                        "max_retries": 0,
                    },
                }
            ]
        },
        alert_rules=["loss > 0 => warning: always"],
    )
    try:
        for step in range(20):
            run.log({"loss": float(step + 1)})
        assert len(run.history_query(-1)) == 20
        assert run.info()["alerts"]["channels"]["c"]["failed"] > 0
    finally:
        run.finish()


def test_an_invalid_rule_is_rejected_without_breaking_the_run(tmp_path):
    run = Run(project="fail", name="rules", dir=str(tmp_path), backends=[])
    try:
        for bad in ["loss >", "((", "unknown_fn(loss) > 1", "loss[abc] > 1"]:
            with pytest.raises((ExprError, ExprSyntaxError)):
                run.alerts.add_rule(bad)
        # an empty right-hand side is legal: level and message are both optional
        assert run.alerts.add_rule("loss > 1 =>").rule.condition == "loss > 1"
        run.log({"loss": 1.0})
        assert len(run.history_query(-1)) == 1
    finally:
        run.finish()


def test_a_rule_over_a_missing_metric_never_fires(tmp_path):
    received = []
    run = Run(
        project="fail",
        name="missing",
        dir=str(tmp_path),
        backends=[],
        alert={
            "channels": [
                {
                    "type": "callable",
                    "name": "c",
                    "options": {"handler": received.append},
                    "policy": {"async_send": False, "dedup_window": 0},
                }
            ]
        },
        alert_rules=["zscore(nothing[10]) > 2 => error: never"],
    )
    try:
        for step in range(30):
            run.log({"loss": float(step)})
        assert received == []
    finally:
        run.finish()


def test_a_broken_on_commit_callback_does_not_lose_data(make, monkeypatch):
    store = make()
    store._on_commit = breaks(RuntimeError("subscriber down"))
    for step in range(10):
        store.log({"loss": float(step)})
    store.flush(commit_open=True)
    assert [r["_step"] for r in store.get(-1)] == list(range(10))


# ------------------------------------------------------------------ lifecycle


def test_logging_after_finish_is_refused_not_fatal(make):
    store = make()
    store.log({"loss": 1.0})
    store.finish()
    assert store.log({"loss": 2.0}) is None
    assert len(store.get(-1)) == 1


def test_querying_before_init_is_a_clear_error():
    store = HistoryStore()
    with pytest.raises(RuntimeError, match="init"):
        store.get(5)
    with pytest.raises(RuntimeError, match="init"):
        store.log({"loss": 1.0})


def test_finish_is_safe_to_call_twice(make):
    store = make()
    store.log({"loss": 1.0})
    store.finish()
    store.finish()
    assert len(read_history(store.log_dir, -1)) == 1


def test_reinitialising_does_not_strand_the_previous_run(make, tmp_path):
    store = make("first")
    for step in range(10):
        store.log({"loss": float(step)})
    store.init(project="fail", name="second", dir=str(tmp_path))
    store.log({"loss": 100.0})
    store.flush(commit_open=True)

    assert len(read_history(tmp_path / "fail" / "first", -1)) == 10
    assert len(read_history(tmp_path / "fail" / "second", -1)) == 1


def test_a_second_init_without_finish_is_refused(tmp_path):
    et.init(project="fail", name="a", dir=str(tmp_path), backends=[])
    try:
        with pytest.raises(RuntimeError, match="already initialized"):
            et.init(project="fail", name="b", dir=str(tmp_path), backends=[])
        et.log({"loss": 1.0})  # the first run is untouched
        assert len(et.history(-1)) == 1
    finally:
        et.finish()


# ------------------------------------------------------------------ concurrency


def test_a_second_writer_on_the_same_file_does_not_corrupt_it(tmp_path):
    first = HistoryStore()
    first.init(project="fail", name="shared", dir=str(tmp_path), buffer_size=1)
    second = HistoryStore()
    second.init(project="fail", name="shared", dir=str(tmp_path), buffer_size=1)
    try:
        for step in range(10):
            first.log({"a": float(step)})
            second.log({"b": float(step)})
        first.flush(commit_open=True)
        second.flush(commit_open=True)
    finally:
        first.finish()
        second.finish()

    # lines may interleave, but every line must still be valid json
    path = tmp_path / "fail" / "shared" / "metrics.jsonl"
    steps = [json.loads(line)["_step"] for line in path.read_text().splitlines()]
    assert len(steps) == 20


def test_concurrent_finish_and_log_do_not_deadlock(make):
    store = make()
    errors: list[Exception] = []

    def spam():
        try:
            for step in range(500):
                store.log({"loss": float(step)})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=spam) for _ in range(4)]
    for thread in threads:
        thread.start()
    store.finish()
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive()
    assert not errors


def test_queries_during_finish_do_not_raise(make):
    store = make()
    for step in range(50):
        store.log({"loss": float(step)})
    results: list = []

    def query():
        for _ in range(200):
            try:
                results.append(len(store.get(10)))
            except RuntimeError:
                results.append(-1)  # closed stores may refuse, but must not crash

    thread = threading.Thread(target=query)
    thread.start()
    store.finish()
    thread.join(timeout=15)
    assert not thread.is_alive() and results


# ------------------------------------------------------------------ artifacts


def test_logging_a_missing_artifact_path_is_a_clear_error(tmp_path):
    run = Run(project="fail", name="art", dir=str(tmp_path), backends=[])
    try:
        with pytest.raises(FileNotFoundError):
            run.log_artifact(str(tmp_path / "nope.bin"), name="m", type="model")
        run.log({"loss": 1.0})
        assert len(run.history_query(-1)) == 1
    finally:
        run.finish()


def test_a_corrupt_artifact_index_is_survivable(tmp_path):
    source = tmp_path / "f.bin"
    source.write_bytes(b"x")
    run = Run(project="fail", name="art2", dir=str(tmp_path), backends=[])
    run.log_artifact(str(source), name="m", type="model")
    index = run.artifacts.root / "index.jsonl"
    run.finish()
    index.write_text(index.read_text() + "{ broken\n")

    resumed = Run(project="fail", name="art3", dir=str(tmp_path), backends=[])
    try:
        assert resumed.use_artifact("m:latest").name == "m"
    finally:
        resumed.finish()
