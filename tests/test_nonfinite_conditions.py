"""Non-finite literals and ranged samples obey the same three-valued logic."""

import pytest

import expr_tracker as et
from expr_tracker.alerts.expr import UNKNOWN, EvalContext, evaluate, parse
from expr_tracker.history import MetricSeries


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_ranged_samples_are_unknown_conditions(value):
    series = MetricSeries()
    series.add(0, 0.0, {"value": value})
    context = EvalContext(series)
    for expression in ("value[1] or false", "value[1] and true", "not value[1]"):
        assert evaluate(parse(expression), context) is UNKNOWN
    assert evaluate(parse("value[1] or true"), context) is True
    assert evaluate(parse("value[1] and false"), context) is False


@pytest.mark.parametrize(
    "expression", ["1e999", "-1e999", "1e999 or false", "not 1e999", "no_data(1e999)"]
)
def test_out_of_range_literals_do_not_become_true_or_false(expression):
    assert evaluate(parse(expression), EvalContext(MetricSeries())) is UNKNOWN


def test_nonfinite_values_only_fire_explicit_diagnostic_rules(run, collector):
    channel, messages = collector
    run(
        alert={"channels": [channel()]},
        alert_rules=[
            "1e999 => warning: invalid literal",
            "loss[1] => warning: invalid sample",
            "isinf(loss) => warning: explicit infinity",
        ],
    )
    et.log({"loss": float("inf")})
    assert [message.text for message in messages] == ["explicit infinity"]


def test_unknown_negation_does_not_report_a_false_recovery(run, collector):
    channel, messages = collector
    run(
        alert={"channels": [channel()]},
        alert_rules=[{
            "condition": "not loss[1]",
            "level": "warning",
            "message": "zero",
            "notify_recovery": True,
        }],
    )
    et.log({"loss": 0.0})
    assert len(messages) == 1
    et.log({"loss": float("inf")})
    assert len(messages) == 1
