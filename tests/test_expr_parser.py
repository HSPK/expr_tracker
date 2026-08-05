"""Lexing and grammar: precedence, window selectors, metric names, rule splitting."""

import pytest

from expr_tracker.alerts.expr import M, parse, parse_rule, split_rule
from expr_tracker.alerts.expr.lexer import ExprSyntaxError, parse_duration, tokenize
from expr_tracker.alerts.expr.nodes import BoolOp, Compare, MetricRef, RangeRef


def test_or_binds_looser_than_comparison():
    """Python parses `a > 50 | b > 5` as a > (50|b) > 5; this DSL must not."""
    node = parse("diff(m1)>50 | m1 > 5")
    assert isinstance(node, BoolOp) and node.op == "or"
    assert all(isinstance(value, Compare) for value in node.values)
    assert node.to_source() == "(diff(m1) > 50) or (m1 > 5)"


def test_and_binds_tighter_than_or():
    node = parse("a > 1 or b > 2 and c > 3")
    assert node.op == "or"
    assert node.values[1].op == "and"


@pytest.mark.parametrize(
    "source",
    ["a > 1 or b > 2", "a > 1 || b > 2", "a > 1 | b > 2"],
)
def test_or_aliases(source):
    assert parse(source).op == "or"


@pytest.mark.parametrize(
    "source", ["a > 1 and b > 2", "a > 1 && b > 2", "a > 1 & b > 2"]
)
def test_and_aliases(source):
    assert parse(source).op == "and"


def test_chained_comparison_expands_to_and():
    node = parse("1 < loss < 10")
    assert node.op == "and" and len(node.values) == 2


def test_window_selector():
    node = parse("mean(loss[20]) > 1")
    window = node.left.args[0]
    assert isinstance(window, RangeRef) and window.count == 20
    assert parse("mean(loss[30s])").args[0].duration == 30.0
    assert parse("mean(loss[5m])").args[0].duration == 300.0


def test_backtick_metric_names():
    node = parse("`train/loss` > 1")
    assert isinstance(node.left, MetricRef) and node.left.name == "train/loss"
    assert node.to_source() == "`train/loss` > 1"


def test_dotted_metric_names_are_plain():
    assert parse("eval.acc < 0.5").left.name == "eval.acc"


def test_arithmetic_precedence():
    assert parse("a + b * 2 > 1").left.to_source() == "a + (b * 2)"


def test_unary_and_not():
    assert parse("-loss > -1").to_source() == "-loss > -1"
    assert parse("not has(loss)").to_source() == "not has(loss)"


def test_duration_literal_and_helper():
    assert parse("no_data(10m)").args[0].value == 600.0
    assert parse_duration("1.5h") == 5400.0
    assert parse_duration("250ms") == 0.25


def test_tokenizer_rejects_unknown_characters():
    with pytest.raises(ExprSyntaxError, match="Unexpected character"):
        tokenize("a $ b")


@pytest.mark.parametrize(
    "source",
    [
        "",
        "a >",
        "mean(loss[20]",
        "mean(loss[])",
        "a > 1 or",
        "loss[20] > 1 extra",
        "`unterminated",
    ],
)
def test_syntax_errors(source):
    with pytest.raises((ExprSyntaxError, ValueError)):
        parse(source)


def test_no_python_injection_surface():
    """The grammar is a closed whitelist: no string literals, attributes or free calls."""
    from expr_tracker.alerts.expr import ExprError, validate

    for source in ["__import__('os')", "open('x')", 'eval("1")']:
        with pytest.raises(ExprSyntaxError):
            parse(source)
    with pytest.raises(ExprError):
        validate(parse("open(1)"))
    # A dot is just part of a metric name, never attribute access
    assert parse("loss.__class__").name == "loss.__class__"


# ---------------------------------------------------------------------- splitting


def test_arrow_form():
    assert split_rule("diff(m1) > 50 or m1 > 5 => warn: m1 failure") == (
        "diff(m1) > 50 or m1 > 5",
        "warn",
        "m1 failure",
    )


def test_arrow_without_message():
    assert split_rule("loss > 1 => error") == ("loss > 1", "error", "")


def test_arrow_without_level():
    assert split_rule("loss > 1 => something broke") == (
        "loss > 1",
        None,
        "something broke",
    )


def test_comma_form_with_function_args():
    """The comma inside `mean(m1, 10)` must not be treated as a field separator."""
    assert split_rule("stalled(m1[20], 0.1) > 0, warn, a, b") == (
        "stalled(m1[20], 0.1) > 0",
        "warn",
        "a, b",
    )


def test_message_may_contain_colon():
    condition, level, message = split_rule("loss > 1 => error: loss: too high")
    assert (condition, level, message) == ("loss > 1", "error", "loss: too high")


def test_parse_rule_defaults():
    rule = parse_rule("diff(m1) > 50 => warn: m1 failure")
    assert rule.level.value == "warning"
    assert rule.message == "m1 failure"
    assert rule.name.startswith("rule_")
    assert rule.mode == "edge"


def test_parse_rule_from_builder():
    rule = parse_rule((M.m1.diff() > 50) | (M.m1 > 5), level="error")
    assert rule.condition == "(diff(m1) > 50) or (m1 > 5)"
    assert rule.level.value == "error"


def test_builder_round_trip():
    node = M["train/loss"][50].zscore() > 4
    assert node.to_source() == "zscore(`train/loss`[50]) > 4"
    assert parse(node.to_source()).to_source() == node.to_source()


def test_builder_rejects_python_boolean_ops():
    with pytest.raises(TypeError, match="boolean context"):
        bool(M.loss > 1)


def test_builder_unknown_function():
    """After a window only functions are valid; on a metric it is a nested name."""
    with pytest.raises(AttributeError, match="known functions"):
        M.loss[5].nope()
    assert M.loss.nope.to_source() == "loss.nope"
