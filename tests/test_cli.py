"""The full CLI surface: every command, every format, and their error paths."""

import json

import pytest
from click.testing import CliRunner

from expr_tracker import cli
from expr_tracker.history import HistoryStore


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def run_dir(tmp_path):
    """A finished run with predictable metrics."""
    store = HistoryStore()
    store.init(project="p", name="r", dir=str(tmp_path), max_open_seconds=None)
    for step in range(10):
        payload = {"loss": 1.0 / (step + 1), "lr": 0.1}
        if step % 3 == 0:
            payload["eval/acc"] = step / 10
        store.log(payload)
    store.finish()
    return str(store.log_dir)


def invoke(runner, *args):
    result = runner.invoke(cli.main, list(args))
    assert result.exit_code == 0, result.output
    return result.output


# ------------------------------------------------------------------ history


def test_history_table_is_the_default(runner, run_dir):
    output = invoke(runner, "history", run_dir, "-n", "3")
    lines = output.strip().splitlines()
    assert "_step" in lines[0] and "loss" in lines[0]
    assert set(lines[1]) <= {"-", " "}  # the rule under the header
    assert len(lines) == 5  # header, rule, three rows


def test_history_json_round_trips(runner, run_dir):
    rows = json.loads(
        invoke(runner, "history", run_dir, "-n", "-1", "--format", "json")
    )
    assert [r["_step"] for r in rows] == list(range(10))
    assert rows[0]["lr"] == 0.1


def test_history_csv_has_a_header_and_one_line_per_step(runner, run_dir):
    output = invoke(runner, "history", run_dir, "-n", "4", "--format", "csv")
    lines = output.strip().splitlines()
    assert lines[0].startswith("_step,")
    assert len(lines) == 5


def test_history_can_select_metrics(runner, run_dir):
    output = invoke(
        runner, "history", run_dir, "-n", "-1", "--format", "csv", "--metrics", "loss"
    )
    header = output.strip().splitlines()[0]
    assert header == "_step,_time,loss"
    assert "lr" not in output


def test_history_accepts_several_metrics(runner, run_dir):
    output = invoke(
        runner, "history", run_dir, "--format", "csv", "--metrics", "loss,lr"
    )
    assert output.strip().splitlines()[0] == "_step,_time,loss,lr"


def test_history_step_range_is_end_exclusive(runner, run_dir):
    rows = json.loads(
        invoke(runner, "history", run_dir, "--step-range", "2:5", "--format", "json")
    )
    assert [r["_step"] for r in rows] == [2, 3, 4]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [(":3", [0, 1, 2]), ("7:", [7, 8, 9]), (":", list(range(10)))],
)
def test_history_open_ended_step_ranges(runner, run_dir, spec, expected):
    rows = json.loads(
        invoke(runner, "history", run_dir, "--step-range", spec, "--format", "json")
    )
    assert [r["_step"] for r in rows] == expected


def test_history_reads_a_file_as_well_as_a_directory(runner, run_dir):
    from pathlib import Path

    path = str(Path(run_dir) / "metrics.jsonl")
    assert json.loads(invoke(runner, "history", path, "--format", "json"))


def test_history_handles_sparse_metrics(runner, run_dir):
    output = invoke(runner, "history", run_dir, "-n", "-1", "--format", "csv")
    header, *rows = output.strip().splitlines()
    assert "eval/acc" in header
    assert rows[1].endswith(",")  # step 1 has no eval/acc, so the cell is empty


def test_history_of_an_empty_selection_is_not_an_error(runner, run_dir):
    output = invoke(runner, "history", run_dir, "-n", "0", "--format", "json")
    assert json.loads(output) == []


@pytest.mark.parametrize("spec", ["abc", "1:2:3", "x:y", "1-2"])
def test_history_rejects_a_malformed_step_range(runner, run_dir, spec):
    result = runner.invoke(cli.main, ["history", run_dir, "--step-range", spec])
    assert result.exit_code != 0


def test_history_rejects_a_missing_run(runner, tmp_path):
    result = runner.invoke(cli.main, ["history", str(tmp_path / "nope")])
    assert result.exit_code != 0
    assert "does not exist" in result.output.lower()


def test_history_rejects_an_unknown_format(runner, run_dir):
    result = runner.invoke(cli.main, ["history", run_dir, "--format", "xml"])
    assert result.exit_code != 0


def test_history_of_a_directory_without_metrics(runner, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(cli.main, ["history", str(empty)])
    assert result.exit_code != 0


# ------------------------------------------------------------------ alert


@pytest.fixture
def channel(monkeypatch):
    """Route the process-wide dispatcher into a list."""
    from expr_tracker.alerts import configure_alert, reset_alert_config

    received: list = []
    configure_alert(
        channels=[
            {
                "type": "callable",
                "name": "c",
                "options": {"handler": received.append},
                "policy": {
                    "async_send": False,
                    "dedup_window": 0,
                    "rate_limit_per_minute": None,
                },
            }
        ]
    )
    yield received
    reset_alert_config()


def test_alert_sends_a_message(runner, channel):
    invoke(runner, "alert", "training diverged")
    assert len(channel) == 1
    assert channel[0].text == "training diverged"
    assert channel[0].title == "Alert"
    assert channel[0].level.value == "info"


def test_alert_accepts_a_title_and_level(runner, channel):
    invoke(runner, "alert", "gpu on fire", "--title", "Node 3", "--level", "critical")
    assert channel[0].title == "Node 3"
    assert channel[0].level.value == "critical"


def test_alert_can_target_named_channels(runner, channel):
    invoke(runner, "alert", "hello", "--channel", "c")
    assert len(channel) == 1
    invoke(runner, "alert", "hello", "--channel", "does-not-exist")
    assert len(channel) == 1  # unknown channel, nothing sent, still exit 0


def test_alert_rejects_an_unknown_level(runner, channel):
    result = runner.invoke(cli.main, ["alert", "x", "--level", "nonsense"])
    assert result.exit_code != 0


def test_alert_without_a_message_fails(runner, channel):
    assert runner.invoke(cli.main, ["alert"]).exit_code != 0


# ------------------------------------------------------------------ rules


def test_rules_explain_shows_the_parse(runner):
    output = invoke(runner, "rules", "explain", "mean(loss[20]) > 1 => error: high")
    assert "condition : mean(loss[20]) > 1" in output
    assert "level     : error" in output
    assert "message   : high" in output
    assert "metrics   : loss" in output
    assert "functions : mean" in output


def test_rules_explain_normalises_the_legacy_comma_form(runner):
    output = invoke(runner, "rules", "explain", "diff(m1)>50 | m1 > 5, warn, m1 bad")
    assert "level     : warning" in output
    assert "message   : m1 bad" in output
    assert "m1" in output


def test_rules_explain_reports_no_functions(runner):
    output = invoke(runner, "rules", "explain", "loss > 1")
    assert "functions : -" in output


@pytest.mark.parametrize(
    "expression", ["loss >", "((", "unknown_fn(loss) > 1", "loss[abc] > 1"]
)
def test_rules_explain_rejects_bad_expressions(runner, expression):
    result = runner.invoke(cli.main, ["rules", "explain", expression])
    assert result.exit_code != 0


def test_rules_test_replays_history(runner, run_dir):
    output = invoke(
        runner, "rules", "test", "loss > 0.3 => warning: high", "--run", run_dir
    )
    assert "replayed 10 steps" in output
    assert "alert(s)" in output
    assert "step=" in output


def test_rules_test_reports_no_alerts(runner, run_dir):
    output = invoke(
        runner, "rules", "test", "loss > 1000 => warning: never", "--run", run_dir
    )
    assert "replayed 10 steps, 0 alert(s)" in output


def test_rules_test_can_limit_the_replay(runner, run_dir):
    output = invoke(
        runner, "rules", "test", "loss > 0 => warning: x", "--run", run_dir, "-n", "3"
    )
    assert "replayed 3 steps" in output


def test_rules_test_supports_window_functions(runner, run_dir):
    output = invoke(
        runner,
        "rules",
        "test",
        "diff(loss[2]) < 0 => warning: falling",
        "--run",
        run_dir,
    )
    assert "replayed 10 steps" in output


def test_rules_test_requires_a_run(runner):
    assert runner.invoke(cli.main, ["rules", "test", "loss > 1"]).exit_code != 0


def test_rules_test_rejects_a_missing_run(runner, tmp_path):
    result = runner.invoke(
        cli.main, ["rules", "test", "loss > 1", "--run", str(tmp_path / "nope")]
    )
    assert result.exit_code != 0


# ------------------------------------------------------------------ shell


def test_help_lists_every_command(runner):
    output = invoke(runner, "--help")
    for command in ("alert", "history", "rules"):
        assert command in output


def test_rules_help_lists_subcommands(runner):
    output = invoke(runner, "rules", "--help")
    assert "explain" in output and "test" in output


def test_an_unknown_command_fails(runner):
    assert runner.invoke(cli.main, ["nope"]).exit_code != 0
