"""The calls made on every training step: et.log, et.history, summary, alerts.

These paths run millions of times per run, so they get contract-level tests: the
exact wandb-compatible signatures, the defaults, and the interactions between them.
"""

import math

import pytest

import expr_tracker as et


@pytest.fixture
def run(tmp_path):
    """An initialised run that is always torn down."""
    created = []

    def factory(**options):
        options.setdefault("project", "hot")
        options.setdefault("name", "run")
        options.setdefault("dir", str(tmp_path))
        options.setdefault("backends", [])
        options.setdefault("max_open_seconds", None)
        created.append(et.init(**options))
        return created[-1]

    yield factory
    if et.get_run() is not None:
        et.finish()


@pytest.fixture
def alerts():
    received: list = []
    config = {
        "channels": [
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
    }
    return config, received


# ------------------------------------------------------------------ et.log


def test_log_advances_the_step_implicitly(run):
    run()
    for expected in range(5):
        assert et.get_run().step == expected
        et.log({"loss": float(expected)})
    assert [r["_step"] for r in et.history(-1)] == list(range(5))


def test_log_merges_repeated_calls_for_one_step(run):
    run()
    et.log({"a": 1}, commit=False)
    et.log({"b": 2}, commit=False)
    et.log({"c": 3})
    rows = et.history(-1)
    assert len(rows) == 1
    assert (rows[0]["a"], rows[0]["b"], rows[0]["c"]) == (1, 2, 3)


def test_an_explicit_step_does_not_commit_by_default(run):
    run()
    et.log({"a": 1}, step=7)
    assert et.info()["history"]["rows_on_disk"] == 0
    et.log({"b": 2}, step=7)
    et.log({"c": 3}, step=8)  # advancing the step commits step 7
    assert et.info()["history"]["rows_on_disk"] == 1
    rows = et.history(-1)
    assert [r["_step"] for r in rows] == [7, 8]
    assert rows[0]["a"] == 1 and rows[0]["b"] == 2


def test_commit_true_writes_immediately(run):
    run()
    et.log({"a": 1}, step=3, commit=True)
    assert et.info()["history"]["rows_on_disk"] == 1


def test_an_empty_payload_is_harmless(run):
    run()
    et.log({})
    et.log({"loss": 1.0})
    assert len(et.history(-1)) <= 2


def test_the_step_never_moves_backwards(run):
    run()
    et.log({"loss": 1.0}, step=10)
    et.log({"loss": 2.0}, step=3)
    assert [r["_step"] for r in et.history(-1)] == [10]


def test_logging_the_same_metric_repeatedly_keeps_the_last_value(run):
    run()
    for value in (1.0, 2.0, 3.0):
        et.log({"loss": value}, step=0, commit=False)
    et.get_run().history.flush(commit_open=True)
    assert et.history(-1)[0]["loss"] == 3.0


@pytest.mark.parametrize(
    "value", [0, -1, 1.5, True, False, None, "text", [1, 2], {"a": 1}]
)
def test_log_accepts_every_json_shape(run, value):
    run()
    et.log({"v": value})
    assert et.history(-1)[0]["v"] == value


def test_nan_and_inf_reach_the_history(run):
    run()
    et.log({"a": float("nan"), "b": float("inf")})
    row = et.history(-1)[0]
    assert math.isnan(row["a"]) and math.isinf(row["b"])


def test_metric_names_with_slashes_are_preserved(run):
    run()
    et.log({"train/loss": 1.0, "eval/acc": 0.5})
    row = et.history(-1)[0]
    assert row["train/loss"] == 1.0 and row["eval/acc"] == 0.5


# ------------------------------------------------------------------ et.history


def test_history_defaults_to_fifty_steps(run):
    run()
    for step in range(80):
        et.log({"loss": float(step)})
    assert len(et.history()) == 50
    assert et.history()[-1]["_step"] == 79


@pytest.mark.parametrize("n", [0, 1, 5, 50, 79, 80, 200])
def test_history_returns_the_newest_n_steps(run, n):
    run()
    for step in range(80):
        et.log({"loss": float(step)})
    rows = et.history(n)
    assert len(rows) == min(n, 80)
    if rows:
        assert rows[-1]["_step"] == 79


def test_history_minus_one_returns_everything(run):
    run()
    for step in range(120):
        et.log({"loss": float(step)})
    assert len(et.history(-1)) == 120


def test_history_includes_the_open_row(run):
    run()
    et.log({"loss": 1.0}, step=0)  # uncommitted
    assert et.history(1)[0]["loss"] == 1.0


def test_history_output_types(run):
    pd = pytest.importorskip("pandas")
    run()
    for step in range(10):
        et.log({"loss": float(step)})
    assert isinstance(et.history(5), list)
    assert isinstance(et.history(5, output_type="pandas"), pd.DataFrame)
    assert isinstance(et.history(5, output_type="pd"), pd.DataFrame)


def test_history_of_a_fresh_run_is_empty(run):
    run()
    assert et.history(-1) == []
    assert et.history(50) == []


def test_history_metrics_and_meta_selection(run):
    run()
    for step in range(5):
        et.log({"loss": float(step), "lr": 0.1})
    bare = et.history(-1, metrics=["loss"], include_meta=False)
    assert bare[0] == {"loss": 0.0}


def test_history_is_a_snapshot_not_a_live_view(run):
    run()
    for step in range(5):
        et.log({"loss": float(step)})
    rows = et.history(-1)
    et.log({"loss": 99.0})
    assert len(rows) == 5  # the earlier result is unchanged


# ------------------------------------------------------------------ summary


def test_the_summary_tracks_the_last_value(run):
    run()
    for step in range(10):
        et.log({"loss": float(step), "acc": step / 10})
    summary = dict(et.summary())
    assert summary["loss"] == 9.0 and summary["acc"] == 0.9


def test_an_explicit_summary_value_is_not_overwritten(run):
    run()
    et.log({"acc": 0.1})
    et.summary()["acc"] = 0.99
    et.log({"acc": 0.2})
    assert dict(et.summary())["acc"] == 0.99


def test_the_summary_survives_finish(run, tmp_path):
    import json

    run()
    et.log({"loss": 1.0})
    et.summary()["best"] = 42
    log_dir = et.info()["history"]["log_dir"]
    et.finish()
    saved = json.loads((tmp_path / "hot" / "run" / "summary.json").read_text())
    assert saved["best"] == 42 and saved["loss"] == 1.0
    assert log_dir.endswith("hot/run")


def test_a_dropped_step_does_not_touch_the_summary(run):
    run()
    et.log({"loss": 1.0}, step=5)
    et.log({"loss": 99.0}, step=1)
    assert dict(et.summary())["loss"] == 1.0


# ------------------------------------------------------------------ alerts


def test_a_rule_fires_on_the_committed_step(run, alerts):
    config, received = alerts
    run(alert=config, alert_rules=["loss > 10 => error: too high {loss}"])
    et.log({"loss": 1.0})
    assert received == []
    et.log({"loss": 50.0})
    assert len(received) == 1
    assert "50" in received[0].text
    assert received[0].level.value == "error"


def test_a_rule_sees_the_merged_row_not_each_call(run, alerts):
    config, received = alerts
    run(alert=config, alert_rules=["a > 0 and b > 0 => warning: both"])
    et.log({"a": 1}, commit=False)
    assert received == []
    et.log({"b": 1})  # only now is the row complete
    assert len(received) == 1


def test_alerts_are_evaluated_once_per_step(run, alerts):
    config, received = alerts
    run(
        alert=config,
        alert_rules=[{"condition": "loss > 0", "mode": "level", "cooldown": None}],
    )
    for _ in range(10):
        et.log({"loss": 1.0})
    assert len(received) == 10


def test_window_functions_see_history_across_steps(run, alerts):
    config, received = alerts
    run(alert=config, alert_rules=["diff(loss[2]) > 100 => error: spike"])
    for value in (1.0, 2.0, 3.0):
        et.log({"loss": value})
    assert received == []
    et.log({"loss": 500.0})
    assert len(received) == 1


def test_a_rule_added_mid_run_applies_to_later_steps(run, alerts):
    config, received = alerts
    run(alert=config)
    et.log({"loss": 100.0})
    assert received == []
    et.add_alert_rule("loss > 10 => warning: high")
    et.log({"loss": 100.0})
    assert len(received) == 1


def test_the_message_template_sees_metrics_and_step(run, alerts):
    config, received = alerts
    run(alert=config, alert_rules=["loss > 1 => warning: step {step} loss {loss:.2f}"])
    et.log({"loss": 2.5}, step=4, commit=True)
    assert received[0].text == "step 4 loss 2.50"


def test_alert_fields_carry_the_run_identity(run, alerts):
    config, received = alerts
    run(alert=config, alert_rules=["loss > 1 => warning: x"])
    et.log({"loss": 2.0})
    fields = received[0].fields
    assert fields["project"] == "hot" and fields["run"] == "run"
    assert fields["step"] == 0


# ------------------------------------------------------------------ combined


def test_a_realistic_step_touches_every_sink_consistently(run, alerts):
    """One log() must land in history, summary, alerts and the file identically."""
    config, received = alerts
    run(alert=config, alert_rules=["loss > 100 => error: diverged"])
    for step in range(50):
        et.log({"loss": float(step), "lr": 0.1})
    et.log({"loss": 999.0})
    et.get_run().history.flush(commit_open=True)

    rows = et.history(-1)
    assert len(rows) == 51
    assert rows[-1]["loss"] == 999.0
    assert dict(et.summary())["loss"] == 999.0
    assert len(received) == 1
    assert received[0].fields["step"] == 50

    stats = et.info()["history"]
    assert stats["rows_on_disk"] == 51
    assert stats["last_step"] == 50


def test_the_hot_path_does_not_leak_state_between_runs(run, tmp_path):
    run(name="first")
    for step in range(10):
        et.log({"loss": float(step)})
    et.summary()["best"] = 1
    et.finish()

    run(name="second")
    assert et.get_run().step == 0
    assert et.history(-1) == []
    assert dict(et.summary()) == {}
