"""Scenarios that cross a process or a subsystem boundary.

Each of these works today but had nothing pinning it: a live run read from another
process, alerts firing while the cache evicts, and resume on a file that already
needs merging and a halved index.
"""

import contextlib
import json
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

from expr_tracker.history import HistoryStore, read_history
from expr_tracker.run import Run

SRC = str(Path("src").resolve())


def store_for(tmp_path, name="r", **options):
    store = HistoryStore()
    options.setdefault("max_open_seconds", None)
    store.init(project="sc", name=name, dir=str(tmp_path), **options)
    return store


def alert_config(sink):
    return {
        "channels": [
            {
                "type": "callable",
                "name": "c",
                "options": {"handler": sink.append},
                "policy": {
                    "async_send": False,
                    "dedup_window": 0,
                    "rate_limit_per_minute": None,
                },
            }
        ]
    }


# ------------------------------------------------------------------ live reads


def test_a_running_run_can_be_read_from_another_process(tmp_path):
    """Watching a training job from a notebook while it writes."""
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {SRC!r})
        import expr_tracker as et
        et.init(project="sc", name="live", dir={str(tmp_path)!r},
                backends=[], buffer_size=1)
        for step in range(40):
            et.log({{"loss": float(step)}})
            time.sleep(0.03)
        et.finish()
    """)
    process = subprocess.Popen([sys.executable, "-c", script])
    run_dir = tmp_path / "sc" / "live"
    try:
        seen: list[int] = []
        deadline = time.monotonic() + 30
        while process.poll() is None and time.monotonic() < deadline:
            if run_dir.exists():
                rows = read_history(run_dir, -1)
                steps = [r["_step"] for r in rows]
                assert steps == sorted(set(steps))  # never torn or out of order
                seen.append(len(steps))
            time.sleep(0.1)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.kill()

    assert seen and max(seen) > 0
    assert seen == sorted(seen)  # a reader only ever sees more, never less
    assert len(read_history(run_dir, -1)) == 40


def test_a_live_tail_query_never_returns_a_partial_line(tmp_path):
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {SRC!r})
        import expr_tracker as et
        et.init(project="sc", name="tail", dir={str(tmp_path)!r},
                backends=[], buffer_size=1)
        for step in range(60):
            et.log({{"loss": float(step), "pad": "x" * 200}})
            time.sleep(0.02)
        et.finish()
    """)
    process = subprocess.Popen([sys.executable, "-c", script])
    run_dir = tmp_path / "sc" / "tail"
    try:
        deadline = time.monotonic() + 30
        while process.poll() is None and time.monotonic() < deadline:
            if run_dir.exists():
                for row in read_history(run_dir, 5):
                    assert row["pad"] == "x" * 200  # a torn read would truncate it
            time.sleep(0.05)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.kill()
    assert len(read_history(run_dir, -1)) == 60


def test_two_readers_and_a_writer_agree(tmp_path):
    store = store_for(tmp_path, "readers", buffer_size=1)
    results: list[list[int]] = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            # FileNotFoundError until the first flush lands
            with contextlib.suppress(FileNotFoundError):
                results.append([r["_step"] for r in read_history(store.log_dir, 10)])

    threads = [threading.Thread(target=reader) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        for step in range(100):
            store.log({"loss": float(step)})
        store.flush(commit_open=True)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=10)
        store.finish()

    assert results
    for steps in results:
        assert steps == sorted(set(steps))


# ------------------------------------------------------------------ subsystems


def test_alerts_keep_firing_while_the_cache_evicts(tmp_path):
    """Eviction moves rows to disk; the alert series must not follow them out."""
    received: list = []
    run = Run(
        project="sc",
        name="evict-alerts",
        dir=str(tmp_path),
        backends=[],
        max_open_seconds=None,
        cache_bytes=1,  # every row is evicted immediately
        alert_window=50,
        alert=alert_config(received),
        alert_rules=["zscore(loss[20]) > 2.5 => error: spike at {step}"],
    )
    try:
        for step in range(200):
            run.log({"loss": 1.0 + (50.0 if step == 150 else 0.0)})
        run.history.flush(commit_open=True)

        assert run.info()["history"]["evicted_rows"] > 100
        assert received, "eviction must not starve the alert window"
        assert any("150" in m.text for m in received)
        assert len(run.history_query(-1)) == 200
    finally:
        run.finish()


def test_window_functions_survive_eviction(tmp_path):
    store = store_for(tmp_path, "window", cache_bytes=1, alert_window=30)
    try:
        for step in range(300):
            store.log({"loss": float(step)})
        store.flush(commit_open=True)
        points = store.series.points("loss")
        assert len(points) == 30  # the alert window is independent of the cache
        assert points[-1][2] == 299.0
        assert store.stats()["cached_rows"] <= 1
    finally:
        store.finish()


def test_alerts_fire_on_patched_steps(tmp_path):
    """A timed-out open row is committed, then patched; the rule sees both."""
    received: list = []
    run = Run(
        project="sc",
        name="patched",
        dir=str(tmp_path),
        backends=[],
        step_policy="allow",
        max_open_seconds=None,
        alert=alert_config(received),
        alert_rules=[
            {
                "condition": "a > 0 and b > 0",
                "message": "both",
                "mode": "level",
                "cooldown": None,
            }
        ],
    )
    try:
        run.log({"a": 1.0}, step=0, commit=True)
        assert received == []  # b is missing, so the rule cannot hold yet
        run.log({"b": 1.0}, step=0, commit=True)
        assert len(received) == 1
    finally:
        run.finish()


def test_resume_over_a_file_that_needs_merging(tmp_path):
    """Patch lines plus a resume: the reader must merge across the boundary."""
    first = store_for(tmp_path, "gnarly", step_policy="allow")
    for step in range(300):
        first.log({"loss": float(step)}, step=step, commit=True)
    for step in range(0, 300, 3):  # patch every third step
        first.log({"extra": step}, step=step, commit=True)
    first.finish()

    writer_lines = len(
        (tmp_path / "sc" / "gnarly" / "metrics.jsonl").read_text().splitlines()
    )
    assert writer_lines == 400

    second = store_for(tmp_path, "gnarly", step_policy="allow")
    try:
        assert second.current_step == 300
        rows = second.get(-1)
        assert [r["_step"] for r in rows] == list(range(300))
        assert sum("extra" in r for r in rows) == 100  # patches merged back in

        second.log({"loss": 300.0})
        second.flush(commit_open=True)
        assert [r["_step"] for r in second.get(-1)][-1] == 300
        assert len(second.get(-1)) == 301  # resume did not reuse an existing step
        assert [r["_step"] for r in second.get(-1, step_range=(150, 153))] == [
            150,
            151,
            152,
        ]
    finally:
        second.finish()


def test_a_tail_query_on_an_out_of_order_file_follows_write_order(tmp_path):
    """With step_policy="allow", get(n) means the n most recently *written* steps.

    Ordering by step number instead would need a full scan of an unsorted file, so
    only get(-1) and step_range are step-ordered.
    """
    store = store_for(tmp_path, "unsorted", step_policy="allow")
    try:
        for step in (0, 1, 2, 3, 4):
            store.log({"loss": float(step)}, step=step, commit=True)
        store.log({"patch": 1}, step=1, commit=True)  # written last, low step
        store.flush(commit_open=True)

        assert [r["_step"] for r in store.get(-1)] == [0, 1, 2, 3, 4]
        assert [r["_step"] for r in store.get(2)] == [1, 4]  # write order, merged
        assert [r["_step"] for r in store.get(-1, step_range=(3, 5))] == [3, 4]
    finally:
        store.finish()


def test_a_resumed_run_keeps_alerting(tmp_path):
    received: list = []
    first = Run(
        project="sc",
        name="resume-alerts",
        dir=str(tmp_path),
        backends=[],
        max_open_seconds=None,
        alert=alert_config(received),
        alert_rules=["loss > 10 => error: high"],
    )
    for step in range(10):
        first.log({"loss": float(step)})
    first.finish()
    assert received == []

    second = Run(
        project="sc",
        name="resume-alerts",
        dir=str(tmp_path),
        backends=[],
        max_open_seconds=None,
        alert=alert_config(received),
        alert_rules=["diff(loss[2]) > 5 => error: jump"],
    )
    try:
        second.log({"loss": 50.0})  # the series was backfilled, so diff() works
        assert len(received) == 1
        assert len(second.history_query(-1)) == 11
    finally:
        second.finish()


def test_summary_artifacts_and_alerts_all_survive_a_crash(tmp_path):
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {SRC!r})
        import expr_tracker as et
        et.init(project="sc", name="crash", dir={str(tmp_path)!r}, backends=[])
        source = {str(tmp_path)!r} + "/payload.bin"
        open(source, "wb").write(b"weights")
        et.log_artifact(source, name="model", type="model")
        et.summary()["best"] = 0.9
        for step in range(20):
            et.log({{"loss": float(step)}})
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr

    run_dir = tmp_path / "sc" / "crash"
    assert len(read_history(run_dir, -1)) == 20
    assert json.loads((run_dir / "summary.json").read_text())["best"] == 0.9
    assert (run_dir / "artifacts.jsonl").read_text().count("\n") == 1


# ------------------------------------------------------------------ parameters


class Recorder:
    def __init__(self):
        self.calls = []

    def init(self, **kwargs):
        self.calls.append(("init", kwargs))

    def log(self, data, step=None, commit=None):
        self.calls.append(("log", kwargs_of(data, step, commit)))

    def finish(self, **kwargs):
        self.calls.append(("finish", kwargs))


def kwargs_of(data, step, commit):
    return {"data": data, "step": step, "commit": commit}


def test_backend_kwargs_reach_the_named_backend(tmp_path):
    backend = Recorder()
    run = Run(
        project="sc",
        name="kwargs",
        dir=str(tmp_path),
        backends=[backend],
        backend_kwargs={"recorder": {"group": "ablation", "job_type": "train"}},
    )
    try:
        init_kwargs = backend.calls[0][1]
        assert init_kwargs["group"] == "ablation"
        assert init_kwargs["job_type"] == "train"
    finally:
        run.finish()


def test_backend_kwargs_for_another_backend_are_ignored(tmp_path):
    backend = Recorder()
    run = Run(
        project="sc",
        name="kwargs2",
        dir=str(tmp_path),
        backends=[backend],
        backend_kwargs={"wandb": {"group": "not-for-me"}},
    )
    try:
        assert "group" not in backend.calls[0][1]
    finally:
        run.finish()


def test_no_backend_kwargs_is_fine(tmp_path):
    backend = Recorder()
    run = Run(project="sc", name="kwargs3", dir=str(tmp_path), backends=[backend])
    try:
        assert backend.calls[0][0] == "init"
    finally:
        run.finish()


def test_entity_reaches_the_backend(tmp_path):
    backend = Recorder()
    run = Run(
        project="sc",
        name="entity",
        dir=str(tmp_path),
        backends=[backend],
        entity="my-team",
    )
    try:
        assert backend.calls[0][1]["entity"] == "my-team"
    finally:
        run.finish()


def test_notes_and_tags_reach_the_backend(tmp_path):
    backend = Recorder()
    run = Run(
        project="sc",
        name="meta",
        dir=str(tmp_path),
        backends=[backend],
        notes="a note",
        tags=["x", "y"],
    )
    try:
        init_kwargs = backend.calls[0][1]
        assert init_kwargs["notes"] == "a note" and init_kwargs["tags"] == ["x", "y"]
    finally:
        run.finish()


def test_backend_kwargs_clashing_with_a_core_field_drops_the_backend(tmp_path):
    """A duplicate keyword is a config error; the run degrades instead of dying."""
    backend = Recorder()
    run = Run(
        project="sc",
        name="clash",
        dir=str(tmp_path),
        backends=[backend],
        backend_kwargs={"recorder": {"project": "somewhere-else"}},
    )
    try:
        assert run.backends == {}
        run.log({"loss": 1.0})
        assert len(run.history_query(-1)) == 1
    finally:
        run.finish()
