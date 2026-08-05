"""Unit tests for every DSL function and the evaluator's degradation rules.

An expression must never raise: anything undecidable evaluates to UNKNOWN.
"""

import math

import pytest

from expr_tracker.alerts.expr import UNKNOWN, EvalContext, evaluate, parse
from expr_tracker.alerts.expr.functions import arity, default_window
from expr_tracker.history import MetricSeries


def series_of(points: dict[str, list], times: list[float] | None = None):
    """A series holding one column per metric, one step per list position."""
    series = MetricSeries()
    length = max((len(v) for v in points.values()), default=0)
    for i in range(length):
        record = {k: v[i] for k, v in points.items() if i < len(v)}
        series.add(i, times[i] if times else float(i), record)
    return series


def context(points: dict[str, list] | None = None, **kwargs):
    points = points or {}
    kwargs.setdefault("step", max((len(v) for v in points.values()), default=1) - 1)
    kwargs.setdefault("now", 1000.0)
    kwargs.setdefault("started_at", 900.0)
    kwargs.setdefault(
        "record", {name: values[-1] for name, values in points.items() if values}
    )
    return EvalContext(series_of(points), **kwargs)


def value(expression: str, ctx: EvalContext):
    return evaluate(parse(expression), ctx)


def approx(expression: str, ctx: EvalContext, expected: float):
    result = value(expression, ctx)
    assert result == pytest.approx(expected), f"{expression} -> {result}"


# ------------------------------------------------------------------ aggregates


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("mean(m)", 3.0),
        ("std(m)", math.sqrt(2.0)),
        ("var(m)", 2.0),
        ("median(m)", 3.0),
        ("sum(m)", 15.0),
        ("first(m)", 1.0),
        ("count(m)", 5.0),
        ("min(m)", 1.0),
        ("max(m)", 5.0),
    ],
)
def test_aggregates_over_the_full_series(expression, expected):
    approx(expression, context({"m": [1, 2, 3, 4, 5]}), expected)


def test_last_defaults_to_a_single_point():
    ctx = context({"m": [1, 2, 3]})
    approx("last(m)", ctx, 3.0)
    approx("last(m[3])", ctx, 3.0)


def test_median_of_an_even_window_averages_the_middle():
    approx("median(m)", context({"m": [1, 2, 3, 10]}), 2.5)


def test_min_and_max_are_dual_purpose():
    ctx = context({"a": [5], "b": [2]})
    approx("min(a, b)", ctx, 2.0)  # scalar form
    approx("max(a, b, 9)", ctx, 9.0)
    approx("min(a)", ctx, 5.0)  # window form


# ------------------------------------------------------------------ differences


def test_diff_rate_and_pct_change():
    ctx = context({"m": [10, 30]})
    approx("diff(m)", ctx, 20.0)
    approx("rate(m)", ctx, 20.0)  # 20 over one step
    approx("pct_change(m)", ctx, 2.0)


def test_pct_change_from_zero_is_unknown():
    assert value("pct_change(m)", context({"m": [0, 5]})) is UNKNOWN


def test_rate_needs_a_step_span():
    series = MetricSeries()
    series.add(4, 1.0, {"m": 1.0})
    series.add(4, 2.0, {"m": 9.0})  # same step twice: no span
    assert value("rate(m)", EvalContext(series, step=4)) is UNKNOWN


def test_slope_and_zscore():
    ctx = context({"m": [1, 2, 3, 4]})
    approx("slope(m)", ctx, 1.0)
    approx("zscore(m)", ctx, 1.3416407865)


def test_slope_is_unknown_without_spread_in_x():
    series = MetricSeries()
    series.add(7, 1.0, {"m": 1.0})
    series.add(7, 2.0, {"m": 2.0})
    assert value("slope(m)", EvalContext(series, step=7)) is UNKNOWN


def test_zscore_is_unknown_for_a_flat_window():
    assert value("zscore(m)", context({"m": [3, 3, 3]})) is UNKNOWN


# ------------------------------------------------------------------ shape tests


def test_ema_weights_recent_points():
    approx("ema(m)", context({"m": [0, 10]}), 3.0)
    approx("ema(m, 1)", context({"m": [0, 10]}), 10.0)


@pytest.mark.parametrize("alpha", ["0", "-0.5", "1.5"])
def test_ema_rejects_an_out_of_range_alpha(alpha):
    assert value(f"ema(m, {alpha})", context({"m": [1, 2]})) is UNKNOWN


def test_stalled_with_and_without_tolerance():
    assert value("stalled(m)", context({"m": [2, 2, 2]})) is True
    assert value("stalled(m)", context({"m": [2, 2, 3]})) is False
    assert value("stalled(m, 1)", context({"m": [2, 2, 3]})) is True


def test_stalled_is_unknown_with_nan():
    assert value("stalled(m)", context({"m": [1.0, float("nan")]})) is UNKNOWN


def test_increasing_and_decreasing_are_strict():
    assert value("increasing(m)", context({"m": [1, 2, 3]})) is True
    assert value("increasing(m)", context({"m": [1, 1, 3]})) is False
    assert value("decreasing(m)", context({"m": [3, 2, 1]})) is True
    assert value("decreasing(m)", context({"m": [3, 3, 1]})) is False


# ------------------------------------------------------------------ empty input


@pytest.mark.parametrize(
    "expression",
    [
        "mean(m)",
        "std(m)",
        "var(m)",
        "median(m)",
        "first(m)",
        "last(m)",
        "diff(m)",
        "rate(m)",
        "pct_change(m)",
        "slope(m)",
        "zscore(m)",
        "ema(m)",
        "stalled(m)",
        "increasing(m)",
        "decreasing(m)",
        "min(m)",
        "max(m)",
    ],
)
def test_every_window_function_degrades_on_an_empty_series(expression):
    assert value(expression, context()) is UNKNOWN


def test_count_of_an_empty_series_is_zero():
    approx("count(m)", context(), 0.0)


@pytest.mark.parametrize(
    "expression", ["std(m)", "var(m)", "diff(m)", "slope(m)", "zscore(m)"]
)
def test_functions_needing_two_points_reject_one(expression):
    assert value(expression, context({"m": [1]})) is UNKNOWN


def test_non_finite_values_propagate_as_unknown():
    assert value("mean(m)", context({"m": [1.0, float("nan")]})) is UNKNOWN
    assert value("mean(m)", context({"m": [1.0, float("inf")]})) is UNKNOWN


# ------------------------------------------------------------------ scalar funcs


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("abs(0 - 3)", 3.0),
        ("exp(0)", 1.0),
        ("sqrt(9)", 3.0),
        ("floor(1.7)", 1.0),
        ("ceil(1.2)", 2.0),
        ("log(1)", 0.0),
        ("log(8, 2)", 3.0),
    ],
)
def test_scalar_functions(expression, expected):
    approx(expression, context(), expected)


@pytest.mark.parametrize("expression", ["sqrt(0 - 1)", "log(0)", "exp(100000)"])
def test_scalar_functions_swallow_domain_errors(expression):
    assert value(expression, context()) is UNKNOWN


def test_scalar_functions_propagate_unknown_arguments():
    assert value("abs(mean(missing))", context()) is UNKNOWN


# ------------------------------------------------------------------ special


def test_step_and_elapsed():
    ctx = context({"m": [1, 2]}, step=41, now=1000.0, started_at=940.0)
    approx("step()", ctx, 41.0)
    approx("elapsed()", ctx, 60.0)


def test_step_is_unknown_before_any_data():
    assert value("step()", EvalContext(MetricSeries(), step=None)) is UNKNOWN


def test_elapsed_defaults_to_zero_without_a_start_time():
    assert value("elapsed()", EvalContext(MetricSeries(), started_at=None)) == 0.0


def test_has_reflects_the_current_record():
    ctx = context({"m": [1]}, record={"m": 1, "other": 2})
    assert value("has(m)", ctx) is True
    assert value("has(missing)", ctx) is False


def test_isnan_and_isinf():
    ctx = context({"a": [float("nan")], "b": [float("inf")], "c": [1.0]})
    assert value("isnan(a)", ctx) is True
    assert value("isinf(b)", ctx) is True
    assert value("isnan(c)", ctx) is False
    assert value("isinf(c)", ctx) is False


def test_isnan_of_a_missing_metric_is_unknown():
    assert value("isnan(missing)", context()) is UNKNOWN


def test_age_and_no_data_measure_staleness():
    series = MetricSeries()
    series.add(0, 100.0, {"m": 1.0})
    ctx = EvalContext(series, step=0, now=160.0, last_commit_time=100.0)
    approx("age(m)", ctx, 60.0)  # age() takes a metric
    assert value("no_data(30s)", ctx) is True  # no_data() takes a duration
    assert value("no_data(120s)", ctx) is False
    assert value("no_data(mean(missing))", ctx) is UNKNOWN


def test_no_data_falls_back_to_the_start_time():
    ctx = EvalContext(MetricSeries(), now=160.0, started_at=100.0)
    assert value("no_data(30s)", ctx) is True


def test_age_of_an_unseen_metric_is_unknown():
    assert value("age(missing)", context()) is UNKNOWN


# ------------------------------------------------------------------ three-valued


def test_kleene_or_short_circuits_past_unknown():
    ctx = context({"m": [5]})
    assert value("mean(missing) > 1 or m > 1", ctx) is True
    assert value("mean(missing) > 1 or m > 100", ctx) is UNKNOWN


def test_kleene_and_short_circuits_on_false():
    ctx = context({"m": [5]})
    assert value("m > 100 and mean(missing) > 1", ctx) is False
    assert value("m > 1 and mean(missing) > 1", ctx) is UNKNOWN


def test_not_of_unknown_stays_unknown():
    assert value("not (mean(missing) > 1)", context()) is UNKNOWN
    assert value("not (m > 1)", context({"m": [0]})) is True


def test_comparisons_with_unknown_are_unknown():
    assert value("mean(missing) > 1", context()) is UNKNOWN
    assert value("1 < mean(missing)", context()) is UNKNOWN


def test_arithmetic_with_unknown_is_unknown():
    ctx = context({"m": [2]})
    assert value("m + mean(missing)", ctx) is UNKNOWN
    assert value("mean(missing) * 2", ctx) is UNKNOWN


def test_division_by_zero_is_unknown():
    ctx = context({"m": [2]})
    assert value("m / 0", ctx) is UNKNOWN
    assert value("m % 0", ctx) is UNKNOWN


def test_arithmetic_and_precedence():
    ctx = context({"m": [10]})
    approx("m + 2 * 3", ctx, 16.0)
    approx("(m + 2) * 3", ctx, 36.0)
    approx("m - 4 - 3", ctx, 3.0)
    approx("m % 3", ctx, 1.0)
    approx("0 - m", ctx, -10.0)


def test_chained_comparison_reads_left_to_right():
    assert value("1 < 2", context()) is True
    assert value("2 <= 2 and 2 >= 2", context()) is True
    assert value("1 == 1 and 1 != 2", context()) is True


def test_booleans_compare_as_numbers():
    ctx = context({"m": [1, 2, 3]})
    assert value("increasing(m) == 1", ctx) is True


def test_unknown_has_no_truth_value():
    with pytest.raises(TypeError):
        bool(UNKNOWN)
    assert repr(UNKNOWN) == "UNKNOWN"
    assert type(UNKNOWN)() is UNKNOWN  # a singleton


# ------------------------------------------------------------------ windows


def test_point_window_limits_the_tail():
    ctx = context({"m": [1, 2, 3, 100]})
    approx("mean(m[2])", ctx, 51.5)
    approx("mean(m[4])", ctx, 26.5)
    approx("mean(m[99])", ctx, 26.5)  # more than exists is not an error


def test_duration_window_selects_by_time():
    series = series_of({"m": [0.0, 1.0, 2.0]}, times=[0.0, 10.0, 100.0])
    ctx = EvalContext(series, step=2, now=100.0)
    approx("count(m[30s])", ctx, 1.0)
    approx("count(m[200s])", ctx, 3.0)


def test_metric_names_use_dots_for_slashes():
    series = series_of({"train/loss": [4.0]})
    ctx = EvalContext(series, step=0, record={"train/loss": 4.0})
    approx("train.loss", ctx, 4.0)
    approx("mean(train.loss)", ctx, 4.0)


# ------------------------------------------------------------------ metadata


def test_arity_and_default_window_tables():
    assert arity("mean") == (1, 1)
    assert arity("ema") == (1, 2)
    assert arity("min") == (1, None)
    assert arity("abs") == (1, 1)
    assert arity("step") == (0, 0)
    with pytest.raises(KeyError):
        arity("nope")

    assert default_window("diff") == 2
    assert default_window("last") == 1
    assert default_window("mean") is None
    assert default_window("abs") is None
