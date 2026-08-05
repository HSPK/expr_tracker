"""End to end: init/log/history/alert together, always-on jsonl, threading, CLI."""

import threading

import pytest
from click.testing import CliRunner

import expr_tracker as et
from expr_tracker import cli
from expr_tracker.alerts import reset_alert_config
from expr_tracker.run import current_run


def test_jsonl_history_is_always_on(run):
    """Local history stays available even with no backends configured."""
    r = run(backends=[])
    for i in range(3):
        et.log({"loss": float(i)})
    assert [row["_step"] for row in et.history(10)] == [0, 1, 2]
    assert (r.history.log_fp.parent / "metrics.jsonl").name == "metrics.jsonl"
    info = et.info()
    assert info["history"]["rows_on_disk"] >= 0
    assert info["jsonl"]["log_dir"]


def test_jsonl_in_backends_is_not_duplicated(run):
    r = run(backends=["jsonl"])
    assert r.backends == {}
    et.log({"v": 1})
    assert len(et.history(10)) == 1


def test_double_init_raises(run):
    run()
    with pytest.raises(RuntimeError, match="already initialized"):
        run()


def test_log_without_init_raises():
    assert current_run() is None
    with pytest.raises(RuntimeError, match="not initialized"):
        et.log({"v": 1})


def test_log_from_worker_thread(run):
    """The global run must be visible from worker threads (a ContextVar is not)."""
    run(backends=[])
    errors: list = []

    def worker():
        try:
            for i in range(5):
                et.log({"v": i})
        except Exception as e:  # pragma: no cover
            errors.append(e)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert not errors
    assert len(et.history(10)) == 5


def test_history_pandas_output(run):
    pd = pytest.importorskip("pandas")
    run(backends=[])
    for i in range(3):
        et.log({"loss": float(i)})
    frame = et.history(-1, output_type="pd")
    assert isinstance(frame, pd.DataFrame) and len(frame) == 3


def test_history_offline_mode(run, tmp_path):
    r = run(backends=[])
    for i in range(4):
        et.log({"loss": float(i)})
    et.finish()
    rows = et.history(2, run=str(r.history.log_dir))
    assert [row["_step"] for row in rows] == [2, 3]


def test_alert_rules_fire_through_public_api(run, collector):
    channel, messages = collector
    run(backends=[], alert={"channels": [channel()]})
    et.add_alert_rule("diff(m1) > 50 or m1 > 5 => warn: m1 failure {m1}")
    for value in (1, 2, 100):
        et.log({"m1": value})
    assert [m.text for m in messages] == ["m1 failure 100"]
    assert messages[0].fields["step"] == 2
    assert messages[0].level.value == "warning"


def test_alert_rules_from_init(run, collector):
    channel, messages = collector
    run(
        backends=[],
        alert={"channels": [channel()]},
        alert_rules=["loss > 10 => error: too high"],
    )
    et.log({"loss": 50})
    assert len(messages) == 1
    assert [r.name for r in et.list_alert_rules()]


def test_manual_alert_uses_run_channels(run, collector):
    channel, messages = collector
    run(backends=[], alert={"channels": [channel()]})
    et.alert(title="hello", text="world", level="error")
    assert messages[0].title == "hello"


def test_manual_alert_deprecated_backends_kwarg(run, collector):
    channel, messages = collector
    run(backends=[], alert={"channels": [channel(name="lark")]})
    et.alert(title="hi", text="x", backends=["lark"])
    assert len(messages) == 1


def test_alert_without_channels_is_silent():
    reset_alert_config()
    et.alert(title="nobody", text="listens")  # must not raise


def test_same_step_multi_log_evaluates_rule_once(run, collector):
    channel, messages = collector
    run(
        backends=[],
        alert={"channels": [channel()]},
        alert_rules=[{"condition": "loss > 1", "mode": "level", "cooldown": None}],
    )
    et.log({"loss": 10}, step=0)
    et.log({"acc": 0.1}, step=0)
    et.log({"loss": 10}, step=1)
    et.finish()
    assert len(messages) == 2  # once per committed step, not once per log call


# ---------------------------------------------------------------------- CLI


def test_cli_history_formats(run):
    r = run(backends=[])
    for i in range(3):
        et.log({"loss": float(i)})
    et.finish()
    runner = CliRunner()
    path = str(r.history.log_dir)
    table = runner.invoke(cli.main, ["history", path, "-n", "2"])
    assert table.exit_code == 0 and "_step" in table.output
    js = runner.invoke(cli.main, ["history", path, "-n", "-1", "--format", "json"])
    assert js.exit_code == 0 and '"loss"' in js.output
    csv = runner.invoke(
        cli.main, ["history", path, "--format", "csv", "--metrics", "loss"]
    )
    assert csv.exit_code == 0 and csv.output.splitlines()[0] == "_step,_time,loss"
    ranged = runner.invoke(cli.main, ["history", path, "--step-range", "1:2"])
    assert ranged.exit_code == 0


def test_cli_rules_explain():
    result = CliRunner().invoke(
        cli.main, ["rules", "explain", "diff(m1)>50 | m1 > 5 => warn: x"]
    )
    assert result.exit_code == 0
    assert "(diff(m1) > 50) or (m1 > 5)" in result.output
    assert "metrics   : m1" in result.output


def test_cli_rules_explain_reports_errors():
    result = CliRunner().invoke(cli.main, ["rules", "explain", "mean(loss) > 1"])
    assert result.exit_code != 0


def test_cli_rules_test_replays_history(run):
    r = run(backends=[])
    for value in (1, 2, 100, 1, 200):
        et.log({"m1": value})
    et.finish()
    result = CliRunner().invoke(
        cli.main,
        [
            "rules",
            "test",
            "m1 > 50 => warn: high {m1}",
            "--run",
            str(r.history.log_dir),
        ],
    )
    assert result.exit_code == 0
    assert "replayed 5 steps, 2 alert(s)" in result.output
    assert "step=2" in result.output and "step=4" in result.output
