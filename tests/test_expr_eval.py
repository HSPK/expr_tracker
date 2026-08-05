"""Expression evaluation: window function maths, three-valued logic, validation."""

import math

import pytest

from expr_tracker.alerts.expr import (
    UNKNOWN,
    EvalContext,
    ExprError,
    evaluate,
    explain,
    parse,
    truthy,
    validate,
)
from expr_tracker.history import MetricSeries


def ctx(
    values=None, *, step=None, now=100.0, started_at=0.0, last_commit=None, **series
):
    metrics = MetricSeries()
    data = dict(values or {})
    data.update(series)
    length = max((len(v) for v in data.values()), default=0)
    for i in range(length):
        record = {k: v[i] for k, v in data.items() if i < len(v)}
        metrics.add(i, float(i), record)
    return EvalContext(
        metrics,
        step=length - 1 if step is None else step,
        now=now,
        started_at=started_at,
        last_commit_time=last_commit,
    )


def value(source, context):
    return evaluate(parse(source), context)


def test_bare_metric_is_latest():
    assert value("loss", ctx(loss=[1.0, 2.0, 3.0])) == 3.0


def test_diff_default_window():
    assert value("diff(loss)", ctx(loss=[1.0, 5.0])) == 4.0
    assert value("diff(loss[3])", ctx(loss=[1.0, 5.0, 9.0])) == 8.0


def test_aggregations():
    context = ctx(loss=[1.0, 2.0, 3.0, 4.0])
    assert value("mean(loss[4])", context) == 2.5
    assert value("sum(loss[4])", context) == 10.0
    assert value("median(loss[4])", context) == 2.5
    assert value("min(loss[4])", context) == 1.0
    assert value("max(loss[4])", context) == 4.0
    assert value("count(loss[4])", context) == 4.0
    assert value("first(loss[4])", context) == 1.0
    assert value("last(loss)", context) == 4.0
    assert value("std(loss[4])", context) == pytest.approx(1.118033988)


def test_scalar_min_max_are_not_rolling():
    context = ctx(loss=[5.0])
    assert value("min(loss, 2)", context) == 2.0
    assert value("max(loss, 2)", context) == 5.0


def test_rate_and_pct_change():
    context = ctx(loss=[10.0, 20.0, 30.0])
    assert value("rate(loss[3])", context) == 10.0
    assert value("pct_change(loss[3])", context) == 2.0


def test_slope_and_zscore():
    assert value("slope(loss[4])", ctx(loss=[0.0, 2.0, 4.0, 6.0])) == pytest.approx(2.0)
    context = ctx(loss=[1.0, 1.0, 1.0, 10.0])
    assert value("zscore(loss[4])", context) > 1.5


def test_ema_and_stalled_and_monotonic():
    assert value("ema(loss[3], 1)", ctx(loss=[1.0, 2.0, 3.0])) == 3.0
    assert value("stalled(loss[3])", ctx(loss=[2.0, 2.0, 2.0])) is True
    assert value("stalled(loss[3], 0.5)", ctx(loss=[2.0, 2.2, 2.4])) is True
    assert value("increasing(loss[3])", ctx(loss=[1.0, 2.0, 3.0])) is True
    assert value("decreasing(loss[3])", ctx(loss=[1.0, 2.0, 3.0])) is False


def test_time_window():
    metrics = MetricSeries()
    for i in range(10):
        metrics.add(i, float(i), {"loss": float(i)})
    context = EvalContext(metrics, step=9, now=9.0, started_at=0.0)
    assert evaluate(parse("count(loss[3s])"), context) == 4.0


def test_scalar_functions():
    context = ctx(loss=[-4.0])
    assert value("abs(loss)", context) == 4.0
    assert value("sqrt(16)", context) == 4.0
    assert value("log(exp(1))", context) == pytest.approx(1.0)


def test_predicates():
    context = ctx(loss=[float("nan")], other=[1.0])
    assert value("isnan(loss)", context) is True
    assert value("has(loss)", context) is True
    assert value("has(missing)", context) is False
    assert value("isinf(other)", context) is False


def test_context_functions():
    context = ctx(loss=[1.0], step=7, now=100.0, started_at=40.0, last_commit=50.0)
    assert value("step()", context) == 7.0
    assert value("elapsed()", context) == 60.0
    assert value("no_data(30s)", context) is True
    assert value("no_data(120s)", context) is False
    assert value("age(loss)", context) == pytest.approx(100.0)


# ---------------------------------------------------------------------- kleene


def test_missing_metric_is_unknown():
    assert value("missing > 1", ctx(loss=[1.0])) is UNKNOWN


def test_insufficient_history_is_unknown():
    assert value("diff(loss) > 1", ctx(loss=[1.0])) is UNKNOWN
    assert value("std(loss[10]) > 1", ctx(loss=[1.0])) is UNKNOWN


def test_nan_is_unknown():
    assert value("loss > 1", ctx(loss=[float("nan")])) is UNKNOWN


def test_division_by_zero_is_unknown():
    assert value("loss / 0 > 1", ctx(loss=[1.0])) is UNKNOWN


@pytest.mark.parametrize(
    "source,expected",
    [
        ("missing > 1 or loss > 0", True),  # UNKNOWN or True -> True
        ("missing > 1 and loss < 0", False),  # UNKNOWN and False -> False
        ("missing > 1 or loss < 0", UNKNOWN),
        ("missing > 1 and loss > 0", UNKNOWN),
        ("not missing > 1", UNKNOWN),
    ],
)
def test_kleene_logic(source, expected):
    assert value(source, ctx(loss=[1.0])) is expected


def test_truthy_only_fires_on_true():
    context = ctx(loss=[1.0])
    assert truthy(value("loss > 0", context)) is True
    assert truthy(value("loss > 5", context)) is False
    assert truthy(value("missing > 5", context)) is False


def test_metric_name_dot_resolves_to_slash():
    metrics = MetricSeries()
    metrics.add(0, 0.0, {"eval/acc": 0.4})
    context = EvalContext(metrics, step=0)
    assert evaluate(parse("eval.acc < 0.5"), context) is True


# ---------------------------------------------------------------------- validate


def test_validate_unknown_function():
    with pytest.raises(ExprError, match="Unknown function"):
        validate(parse("nope(loss[2])"))


def test_validate_requires_window():
    with pytest.raises(ExprError, match="requires an explicit window"):
        validate(parse("mean(loss)"))
    validate(parse("mean(loss[10])"))
    validate(parse("diff(loss)"))  # diff has a default window


def test_validate_arity():
    with pytest.raises(ExprError, match="argument"):
        validate(parse("step(1)"))


def test_validate_first_arg_must_be_metric():
    with pytest.raises(ExprError, match="expects a metric"):
        validate(parse("mean(1 + 2)"))


def test_explain_renders_values():
    context = ctx(m1=[1.0, 70.0])
    text = explain(parse("diff(m1) > 50 or m1 > 5"), context)
    assert text == "diff(m1)=69 > 50 or m1=70 > 5"


def test_explain_marks_unknown():
    assert "?" in explain(parse("missing > 1"), ctx(loss=[1.0]))


def test_isnan_survives_nan_values():
    assert not math.isnan(0.0)
    assert value("isnan(loss) or loss > 100", ctx(loss=[float("nan")])) is True
