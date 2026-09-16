"""Numeric-series failures are isolated without changing persisted metric values."""

import json
import math

import pytest
from loguru import logger

import expr_tracker as et
from expr_tracker.alerts.expr import UNKNOWN, EvalContext, evaluate, parse
from expr_tracker.history import HistoryStore, MetricSeries
from expr_tracker.history.writer import JsonlWriter


@pytest.mark.parametrize("value", [10**1000, -(10**1000)])
def test_unrepresentable_integer_does_not_hide_other_metrics(run, collector, value):
    channel, messages = collector
    instance = run(
        alert={"watchdog_interval": 0, "channels": [channel()]},
        alert_rules=["loss > 0 => warning: valid loss"],
    )
    et.log({"before": 2.0, "huge": value, "loss": 1.0})
    assert et.history(1)[0]["huge"] == value
    assert instance.history.series.points("before")[0][2] == 2.0
    assert instance.history.series.points("loss")[0][2] == 1.0
    assert instance.history.series.has("huge")
    assert instance.history.series.points("huge") == ()
    context = EvalContext(instance.history.series)
    for expression in ("huge", "mean(huge[2])", "age(huge)", "isnan(huge)"):
        assert evaluate(parse(expression), context) is UNKNOWN
    assert [message.text for message in messages] == ["valid loss"]


def test_numeric_conversion_warning_is_once_per_name_and_clear_resets_state():
    series = MetricSeries()
    warnings = []
    handle = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        for step in range(3):
            series.add(step, float(step), {"huge": 10**1000, "good": step})
        assert len(warnings) == 1
        assert "huge" in warnings[0] and "float" in warnings[0]
        assert len(series.points("good")) == 3
        series.clear()
        assert series.names() == set()
        assert series.points("good") == ()
        series.add(4, 4.0, {"huge": 10**1000, "good": 4})
        assert len(warnings) == 2
        assert series.latest("good") == (4, 4.0, 4.0)
    finally:
        logger.remove(handle)


def test_nonfinite_floats_remain_numeric_diagnostics():
    series = MetricSeries()
    series.add(0, 0.0, {"nan": float("nan"), "inf": float("inf"), "loss": 1.0})
    assert math.isnan(series.latest("nan")[2])
    assert math.isinf(series.latest("inf")[2])
    assert series.latest("loss")[2] == 1.0


@pytest.mark.parametrize("failure", [TypeError, ValueError])
def test_broken_numeric_subclass_conversion_is_isolated(failure):
    class Number(int):
        def __float__(self):
            raise failure("cannot convert")

    series = MetricSeries()
    series.add(0, 0.0, {"bad": Number(1), "good": 2})
    assert series.has("bad")
    assert series.latest("bad") is None
    assert series.latest("good") == (0, 0.0, 2.0)


@pytest.mark.parametrize(
    "timestamp", ["broken", "", float("nan"), float("inf"), -float("inf"), 10**1000, [], {}, True]
)
def test_backfill_continues_after_an_invalid_timestamp(timestamp):
    series = MetricSeries()
    series.backfill([
        {"_step": 0, "_time": 1.0, "loss": 10.0},
        {"_step": 1, "_time": timestamp, "loss": 99.0},
        {"_step": 2, "_time": 2.0, "loss": 30.0},
        {"loss": "no step"},
    ])
    assert series.points("loss") == ((0, 1.0, 10.0), (2, 2.0, 30.0))


def test_backfill_keeps_legacy_missing_and_numeric_string_timestamps():
    series = MetricSeries()
    series.backfill([
        {"_step": 0, "loss": 1.0},
        {"_step": 1, "_time": "2.5", "loss": 2.0},
        {"_step": 2, "_time": 0.0, "loss": 3.0},
    ])
    assert [point[1] for point in series.points("loss")] == [0.0, 2.5, 0.0]


def test_resume_restores_valid_samples_after_corrupt_timestamps(tmp_path):
    rows = [
        {"_step": 0, "_time": 1.0, "loss": 10.0},
        {"_step": 1, "_time": "broken", "loss": 99.0},
        {"_step": 2, "_time": 2.0, "loss": 30.0},
    ]
    writer = JsonlWriter(tmp_path / "metrics.jsonl")
    for index, row in enumerate(rows):
        writer.append(index, index, (json.dumps(row) + "\n").encode())
    writer.close()
    store = HistoryStore().init(project="p", name="resume", run_dir=str(tmp_path))
    try:
        assert store.current_step == 3
        assert store.get(-1) == rows
        assert evaluate(parse("mean(loss[2])"), EvalContext(store.series)) == 20.0
    finally:
        store.finish()
