"""Formal properties of the expression language.

The strongest invariant is the round trip: rendering an AST and reparsing it must
give the same AST. That is what stops a precedence bug from silently changing what
a rule means when it is stored, replayed by the CLI, or shown to a user.
"""

import itertools

import pytest

from expr_tracker.alerts.expr import (
    UNKNOWN,
    EvalContext,
    M,
    evaluate,
    parse,
    parse_rule,
    validate,
)
from expr_tracker.history import MetricSeries

EXPRESSIONS = [
    # comparisons
    "a > 1",
    "a >= 1",
    "a < 1",
    "a <= 1",
    "a == 1",
    "a != 1",
    "a > b",
    "1 < a",
    # arithmetic and its precedence
    "a + b > 1",
    "a - b > 1",
    "a * b > 1",
    "a / b > 1",
    "a % b > 1",
    "a + b * c > 1",
    "(a + b) * c > 1",
    "a - b - c > 1",
    "a / b / c > 1",
    "a * b % c > 1",
    "a + b + c + d > 1",
    "0 - a > 1",
    "a > 1 + 2 * 3",
    # boolean structure
    "a > 1 and b > 2",
    "a > 1 or b > 2",
    "a > 1 and b > 2 or c > 3",
    "a > 1 or b > 2 and c > 3",
    "a > 1 and (b > 2 or c > 3)",
    "(a > 1 or b > 2) and c > 3",
    "not a > 1",
    "not (a > 1 and b > 2)",
    "not not a > 1",
    "not a > 1 and b > 2",
    "a > 1 and b > 2 and c > 3",
    # windows and calls
    "mean(a[5]) > 1",
    "mean(a[30s]) > 1",
    "diff(a) > 1",
    "zscore(a[20]) > 3",
    "ema(a[10], 0.5) > 1",
    "stalled(a[10], 0.01)",
    "min(a, b) > max(c, 1)",
    "mean(a[5]) + std(a[5]) > 1",
    "abs(a - b) > 1",
    "log(a, 2) > 1",
    "increasing(a[5])",
    "isnan(a) or isinf(a)",
    "has(a) and a > 1",
    "no_data(30s)",
    "age(a) > 60",
    "elapsed() > 3600",
    "step() > 100",
    # names
    "train.loss > 1",
    '"odd name" > 1',
    "'a-b' + 'c d' > 1",
    "train/loss > 1",
    "val/m1/acc@16 > 2",
    "mean(train/loss[20]) > 1",
    # literals
    "a > 1e-8",
    "a > .5",
    "a > 1E+3",
]


def reparse(source: str):
    return parse(source)


@pytest.mark.parametrize("source", EXPRESSIONS)
def test_rendering_and_reparsing_is_stable(source):
    """parse -> to_source -> parse must reach a fixed point immediately."""
    once = parse(source).to_source()
    twice = parse(once).to_source()
    assert once == twice


@pytest.mark.parametrize("source", EXPRESSIONS)
def test_rendering_preserves_the_referenced_metrics(source):
    node = parse(source)
    assert parse(node.to_source()).metrics() == node.metrics()


@pytest.mark.parametrize("source", EXPRESSIONS)
def test_rendering_preserves_the_functions_used(source):
    node = parse(source)
    assert parse(node.to_source()).functions() == node.functions()


@pytest.mark.parametrize("source", EXPRESSIONS)
def test_every_expression_validates(source):
    validate(parse(source))


@pytest.mark.parametrize("source", EXPRESSIONS)
def test_evaluation_never_raises(source):
    """Whatever the data, an expression degrades to UNKNOWN instead of raising."""
    series = MetricSeries()
    for step in range(6):
        series.add(step, float(step), {"a": float(step), "b": 2.0, "c": 3.0, "d": 4.0})
    ctx = EvalContext(series, step=5, now=100.0, started_at=0.0, record={"a": 5.0})
    result = evaluate(parse(source), ctx)
    assert result is UNKNOWN or isinstance(result, (bool, float, int))


@pytest.mark.parametrize("source", EXPRESSIONS)
def test_evaluation_agrees_after_a_round_trip(source):
    series = MetricSeries()
    for step in range(6):
        series.add(step, float(step), {"a": float(step), "b": 2.0, "c": 3.0, "d": 4.0})

    def evaluate_with(text):
        ctx = EvalContext(series, step=5, now=100.0, started_at=0.0, record={"a": 5.0})
        return evaluate(parse(text), ctx)

    first = evaluate_with(source)
    second = evaluate_with(parse(source).to_source())
    assert first is second or first == second


# ------------------------------------------------------------------ precedence


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1 + 2 * 3 > 0", "(1 + (2 * 3)) > 0"),
        ("(1 + 2) * 3 > 0", "((1 + 2) * 3) > 0"),
        ("8 / 4 / 2 > 0", "((8 / 4) / 2) > 0"),
        ("8 - 4 - 2 > 0", "((8 - 4) - 2) > 0"),
        ("2 * 3 % 4 > 0", "((2 * 3) % 4) > 0"),
        ("a > 1 and b > 2 or c > 3", "(a > 1) and (b > 2) or (c > 3)"),
        ("not a > 1 and b > 2", "(not (a > 1)) and (b > 2)"),
    ],
)
def test_rendering_makes_precedence_explicit(source, expected):
    """The rendered form is fully parenthesised, so nothing can be misread."""
    assert parse(source).to_source() == expected


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ("1 + 2 * 3", 7.0),
        ("(1 + 2) * 3", 9.0),
        ("8 / 4 / 2", 1.0),
        ("8 - 4 - 2", 2.0),
        ("2 * 3 % 4", 2.0),
        ("7 % 4 * 2", 6.0),
        ("0 - 2 + 5", 3.0),
    ],
)
def test_arithmetic_precedence_matches_python(source, value):
    assert evaluate(parse(source), EvalContext(MetricSeries())) == value
    assert eval(source) == value


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1 > 0 or 0 > 1 and 0 > 1", True),  # `and` binds tighter than `or`
        ("(1 > 0 or 0 > 1) and 0 > 1", False),
        ("not 0 > 1 and 1 > 0", True),  # `not` binds tighter than `and`
        ("not (0 > 1 and 1 > 0)", True),
    ],
)
def test_boolean_precedence(source, expected):
    assert evaluate(parse(source), EvalContext(MetricSeries())) is expected


def test_the_pipe_form_does_not_parse_as_python_would():
    """Python reads `a > 50 | b > 5` as a > (50|b) > 5; this DSL must not."""
    node = parse("diff(m1) > 50 | m1 > 5")
    assert node.to_source() == "(diff(m1) > 50) or (m1 > 5)"


# ------------------------------------------------------------------ builder


BUILT = {
    "a > 1": lambda: M.a > 1,
    "a >= 1": lambda: M.a >= 1,
    "a < 1": lambda: M.a < 1,
    "a <= 1": lambda: M.a <= 1,
    "a == 1": lambda: M.a == 1,
    "a != 1": lambda: M.a != 1,
    "a > b": lambda: M.a > M.b,
    "(a + 1) > 0": lambda: M.a + 1 > 0,
    "(a - 1) > 0": lambda: M.a - 1 > 0,
    "(a * 2) > 0": lambda: M.a * 2 > 0,
    "(a / 2) > 0": lambda: M.a / 2 > 0,
    "(a % 2) > 0": lambda: M.a % 2 > 0,
    "(1 + a) > 0": lambda: 1 + M.a > 0,
    "(1 - a) > 0": lambda: 1 - M.a > 0,
    "(2 * a) > 0": lambda: 2 * M.a > 0,
    "(2 / a) > 0": lambda: 2 / M.a > 0,
    "(a > 1) and (b > 2)": lambda: (M.a > 1) & (M.b > 2),
    "(a > 1) or (b > 2)": lambda: (M.a > 1) | (M.b > 2),
    "not (a > 1)": lambda: ~(M.a > 1),
    "mean(a[5]) > 1": lambda: M.a[5].mean() > 1,
    "mean(a[30s]) > 1": lambda: M.a["30s"].mean() > 1,
    "train.loss > 1": lambda: M.train.loss > 1,
    '"odd name" > 1': lambda: M["odd name"] > 1,
    "train/loss > 1": lambda: M["train/loss"] > 1,
    "diff(a[2]) > 1": lambda: M.a[2].diff() > 1,
    "zscore(a[20]) > 3": lambda: M.a[20].zscore() > 3,
}


@pytest.mark.parametrize(("expected", "build"), list(BUILT.items()), ids=list(BUILT))
def test_the_builder_renders_the_same_source_as_the_parser(expected, build):
    assert build().to_source() == expected


@pytest.mark.parametrize(("expected", "build"), list(BUILT.items()), ids=list(BUILT))
def test_a_built_expression_reparses_unchanged(expected, build):
    node = build()
    assert parse(node.to_source()).to_source() == node.to_source()


def test_a_built_rule_is_accepted_wherever_a_string_is(tmp_path):
    import expr_tracker as et

    received: list = []
    run = et.init(
        project="dsl",
        name="built",
        dir=str(tmp_path),
        backends=[],
        max_open_seconds=None,
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
    )
    try:
        et.add_alert_rule((M.loss > 10) & (M.lr < 1), level="error", message="built")
        run.log({"loss": 50.0, "lr": 0.1})
        assert [m.text for m in received] == ["built"]
    finally:
        et.finish()


def test_the_builder_composes_with_parsed_rules():
    rule = parse_rule(M.loss > 10, level="error", message="m")
    assert rule.condition == "loss > 10"
    assert rule.level.value == "error"


# ------------------------------------------------------------------ operators


OPERATORS = ["+", "-", "*", "/", "%"]
COMPARISONS = [">", ">=", "<", "<=", "==", "!="]


@pytest.mark.parametrize(
    ("left", "right"), list(itertools.product(OPERATORS, OPERATORS))
)
def test_every_arithmetic_pair_parses_and_renders_stably(left, right):
    source = f"a {left} b {right} c > 0"
    assert parse(parse(source).to_source()).to_source() == parse(source).to_source()


@pytest.mark.parametrize("comparison", COMPARISONS)
def test_every_comparison_parses(comparison):
    node = parse(f"a {comparison} 1")
    assert node.to_source() == f"a {comparison} 1"
    assert node.metrics() == {"a"}


@pytest.mark.parametrize(
    "source", ["mean(a) > 1", "std(a) > 1", "zscore(a) > 1", "ema(a) > 1"]
)
def test_aggregates_require_an_explicit_window(source):
    """Without a window the answer would silently depend on buffer size."""
    from expr_tracker.alerts.expr import ExprError

    with pytest.raises(ExprError, match="requires an explicit window"):
        validate(parse(source))


@pytest.mark.parametrize("source", ["diff(a) > 1", "last(a) > 1", "rate(a) > 1"])
def test_some_functions_carry_a_sensible_default_window(source):
    validate(parse(source))


@pytest.mark.parametrize("comparison", COMPARISONS)
def test_every_comparison_evaluates(comparison):
    series = MetricSeries()
    series.add(0, 0.0, {"a": 1.0})
    ctx = EvalContext(series, step=0, record={"a": 1.0})
    assert isinstance(evaluate(parse(f"a {comparison} 1"), ctx), bool)
