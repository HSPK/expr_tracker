"""End-to-end scenarios: a realistic run from start to finish, and recovery paths."""

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import expr_tracker as et
from expr_tracker.history import read_history


@pytest.fixture
def alerts():
    """A synchronous channel config plus the list it delivers into."""
    received: list = []
    config = {
        "channels": [
            {
                "type": "callable",
                "name": "test",
                "options": {"handler": received.append},
                "policy": {
                    "async_send": False,
                    "dedup_window": 0,
                    "rate_limit_per_minute": None,
                },
            }
        ]
    }
    return config, received


def test_full_training_run(tmp_path, alerts):
    """One run: metrics, an eval phase on the same step, artifacts, alerts, summary."""
    config, received = alerts
    checkpoint = tmp_path / "ckpt.pt"
    root = str(tmp_path / "runs")

    run = et.init(
        project="demo",
        name="train",
        dir=root,
        backends=[],
        config={"lr": 0.1, "arch": "resnet"},
        alert=config,
        alert_rules=[
            "zscore(loss[20]) > 3 => error: loss spike {loss:.3f} @ step {step}",
            "isnan(loss) or isinf(loss) => critical: non-finite loss",
        ],
    )
    try:
        for step in range(60):
            loss = 1.0 / (step + 1) + (5.0 if step == 50 else 0.0)
            et.log({"train/loss": loss, "loss": loss, "lr": 0.1}, step=step)
            if step % 20 == 0:  # eval writes to the same step
                et.log({"eval/acc": 0.5 + step / 200}, step=step)
            if step % 25 == 0:
                checkpoint.write_bytes(f"weights-{step}".encode())
                et.log_artifact(str(checkpoint), name="model", type="model")
        et.summary()["best_acc"] = 0.75

        rows = et.history(-1)
        assert [r["_step"] for r in rows] == list(range(60))
        # the eval metric landed on the same row as the training metric
        assert rows[0]["eval/acc"] == 0.5 and rows[0]["train/loss"] == 1.0
        assert all("eval/acc" not in r for r in rows if r["_step"] % 20)
        assert [m.level.value for m in received] == ["error"]
        assert "loss spike" in received[0].text

        assert et.use_artifact("model:latest").version == 2
        assert dict(et.summary())["best_acc"] == 0.75
        info = et.info()
        assert info["history"]["rows_on_disk"] > 0
        assert info["alerts"]["channels"]["test"]["sent"] == 1
    finally:
        et.finish()

    run_dir = Path(run.history.log_dir)
    assert json.loads((run_dir / "config.json").read_text())["lr"] == 0.1
    assert json.loads((run_dir / "summary.json").read_text())["best_acc"] == 0.75
    assert len(read_history(run_dir, -1)) == 60
    lineage = [
        json.loads(x) for x in (run_dir / "artifacts.jsonl").read_text().splitlines()
    ]
    assert [entry["action"] for entry in lineage] == ["log", "log", "log", "use"]


def test_resume_continues_the_same_run(tmp_path):
    root = str(tmp_path / "runs")
    et.init(project="demo", name="resumed", dir=root, backends=[])
    for step in range(10):
        et.log({"loss": float(step)})
    et.finish()

    et.init(project="demo", name="resumed", dir=root, backends=[])
    try:
        assert et.get_run().step == 10  # the cursor picked up where it left off
        assert len(et.history(-1)) == 10  # and the old rows are still visible
        et.log({"loss": 10.0})
        rows = et.history(-1)
        assert [r["_step"] for r in rows] == list(range(11))
    finally:
        et.finish()


def test_alert_rules_added_after_init_and_removed(tmp_path, alerts):
    config, received = alerts
    et.init(project="demo", name="rules", dir=str(tmp_path), backends=[], alert=config)
    try:
        rule = et.add_alert_rule("loss > 10 => warning: too high")
        et.log({"loss": 50})
        assert len(received) == 1
        assert [r.name for r in et.list_alert_rules()] == [rule.name]

        assert et.remove_alert_rule(rule.name) is True
        et.log({"loss": 1})
        et.log({"loss": 99})
        assert len(received) == 1  # the rule is gone, nothing new fires
    finally:
        et.finish()


def test_offline_analysis_of_a_finished_run(tmp_path):
    et.init(project="demo", name="offline", dir=str(tmp_path), backends=[])
    for step in range(30):
        et.log({"loss": float(step), "acc": step / 30})
    run_dir = et.info()["history"]["log_dir"]
    et.finish()

    assert et.get_run() is None  # nothing is active any more
    assert [r["_step"] for r in et.history(5, run=run_dir)] == [25, 26, 27, 28, 29]
    assert len(et.history(-1, run=run_dir)) == 30
    assert [r["_step"] for r in et.history(-1, run=run_dir, step_range=(3, 6))] == [
        3,
        4,
        5,
    ]
    frame = et.history(-1, run=run_dir, output_type="pandas")
    assert list(frame.columns) == ["_step", "_time", "loss", "acc"]


def test_two_projects_do_not_share_artifacts(tmp_path):
    source = tmp_path / "f.bin"
    source.write_bytes(b"content")
    root = str(tmp_path / "runs")

    et.init(project="one", name="r", dir=root, backends=[])
    et.log_artifact(str(source), name="model", type="model")
    et.finish()

    et.init(project="two", name="r", dir=root, backends=[])
    try:
        with pytest.raises(FileNotFoundError):
            et.use_artifact("model:latest")
    finally:
        et.finish()


def test_crash_without_finish_still_persists(tmp_path):
    """A process that dies without finish() must leave a readable, complete run."""
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(Path("src").resolve())!r})
        import expr_tracker as et
        et.init(project="demo", name="crash", dir={str(tmp_path)!r}, backends=[])
        for step in range(25):
            et.log({{"loss": float(step)}})
        et.log({{"loss": 99.0}}, step=25)   # left as an uncommitted open row
        print("done", flush=True)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr

    rows = read_history(Path(tmp_path) / "demo" / "crash", -1)
    assert [r["_step"] for r in rows] == list(range(26))
    assert rows[-1]["loss"] == 99.0  # the atexit hook committed the open row


def test_torn_file_is_repaired_on_the_next_run(tmp_path):
    et.init(project="demo", name="torn", dir=str(tmp_path), backends=[])
    for step in range(5):
        et.log({"loss": float(step)})
    metrics_file = Path(et.info()["history"]["metrics_file"])
    et.finish()

    with open(metrics_file, "ab") as f:  # simulate a kill mid-write
        f.write(b'{"_step": 5, "loss": 5.0, "par')

    et.init(project="demo", name="torn", dir=str(tmp_path), backends=[])
    try:
        et.log({"loss": 5.0})
        rows = et.history(-1)
        assert [r["_step"] for r in rows] == list(range(6))
        assert rows[-1]["loss"] == 5.0
    finally:
        et.finish()
    # every line on disk is valid json
    for line in metrics_file.read_text().splitlines():
        json.loads(line)


def test_cli_reads_a_run_produced_by_the_api(tmp_path):
    et.init(project="demo", name="cli", dir=str(tmp_path), backends=[])
    for step in range(8):
        et.log({"loss": 1.0 / (step + 1)})
    run_dir = et.info()["history"]["log_dir"]
    et.finish()

    runner = __import__("click.testing", fromlist=["CliRunner"]).CliRunner()
    from expr_tracker import cli

    table = runner.invoke(cli.main, ["history", run_dir, "-n", "3"])
    assert table.exit_code == 0 and table.output.count("\n") >= 5

    replay = runner.invoke(
        cli.main, ["rules", "test", "loss > 0.3 => warn: high", "--run", run_dir]
    )
    assert replay.exit_code == 0
    assert "replayed 8 steps" in replay.output


def test_metrics_logged_at_different_frequencies(tmp_path):
    """Sparse metrics must not shift each other's windows or the row layout."""
    et.init(project="demo", name="sparse", dir=str(tmp_path), backends=[])
    try:
        for step in range(30):
            payload = {"loss": float(step)}
            if step % 10 == 0:
                payload["eval/acc"] = step / 30
            et.log(payload, step=step)
        et.get_run().history.flush(commit_open=True)

        rows = et.history(-1)
        assert len(rows) == 30
        assert sum("eval/acc" in r for r in rows) == 3
        # a sparse metric keeps its own series, so diff() sees consecutive evals
        series = et.get_run().history.series
        assert [p[0] for p in series.points("eval/acc")] == [0, 10, 20]
        assert len(series.points("loss")) == 30

        dense = et.history(-1, metrics=["eval/acc"], dropna=True)
        assert [r["_step"] for r in dense] == [0, 10, 20]
    finally:
        et.finish()


def test_step_policy_allow_accepts_out_of_order_writes(tmp_path):
    et.init(
        project="demo", name="ooo", dir=str(tmp_path), backends=[], step_policy="allow"
    )
    try:
        et.log({"a": 1}, step=5)
        et.log({"b": 2}, step=1)
        et.log({"c": 3}, step=5)
        et.get_run().history.flush(commit_open=True)
        rows = et.history(-1)
        assert [r["_step"] for r in rows] == [1, 5]
        assert rows[1]["a"] == 1 and rows[1]["c"] == 3  # both merged into step 5
    finally:
        et.finish()


def test_open_row_timeout_commits_without_finish(tmp_path):
    et.init(
        project="demo",
        name="timeout",
        dir=str(tmp_path),
        backends=[],
        max_open_seconds=0.1,
    )
    try:
        et.log({"loss": 1.0}, step=3)  # explicit step: not committed immediately
        deadline = time.monotonic() + 3
        while et.info()["history"]["rows_on_disk"] == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert et.info()["history"]["rows_on_disk"] == 1
    finally:
        et.finish()
