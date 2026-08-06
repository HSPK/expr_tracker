"""Unit tests for the Python expression builder (``M``) and AST nodes."""

import pytest

from expr_tracker.alerts.expr import M, evaluate, parse
from expr_tracker.alerts.expr.eval import EvalContext
from expr_tracker.alerts.expr.nodes import (
    BinOp,
    BoolOp,
    Call,
    Compare,
    Literal,
    MetricRef,
    Not,
    RangeRef,
    UnaryOp,
)
from expr_tracker.history import MetricSeries


def context(**series):
    metrics = MetricSeries()
    length = max((len(v) for v in series.values()), default=0)
    for i in range(length):
        metrics.add(i, float(i), {k: v[i] for k, v in series.items() if i < len(v)})
    return EvalContext(metrics, step=length - 1, now=float(length), started_at=0.0)


# ---------------------------------------------------------------- construction


def test_metric_reference_forms():
    assert M.loss.to_source() == "loss"
    assert M["train/loss"].to_source() == "train/loss"
    assert M["odd name"].to_source() == '"odd name"'
    assert M.train.loss.to_source() == "train.loss"  # dotted path becomes one name


def test_window_selector_forms():
    assert M.loss[20].to_source() == "loss[20]"
    assert M.loss["30s"].to_source() == "loss[30s]"
    assert M.loss[2.5].to_source() == "loss[2.5s]"
    with pytest.raises(TypeError, match="Invalid window"):
        M.loss[None]


@pytest.mark.parametrize(
    "node,source",
    [
        (M.a > 1, "a > 1"),
        (M.a >= 1, "a >= 1"),
        (M.a < 1, "a < 1"),
        (M.a <= 1, "a <= 1"),
        (M.a == 1, "a == 1"),
        (M.a != 1, "a != 1"),
        (M.a + 1, "a + 1"),
        (M.a - 1, "a - 1"),
        (M.a * 2, "a * 2"),
        (M.a / 2, "a / 2"),
        (M.a % 2, "a % 2"),
        (-M.a, "-a"),
        (+M.a, "a"),
        (1 + M.a, "1 + a"),
        (1 - M.a, "1 - a"),
        (2 * M.a, "2 * a"),
        (2 / M.a, "2 / a"),
        (~(M.a > 1), "not (a > 1)"),
        ((M.a > 1) & (M.b > 2), "(a > 1) and (b > 2)"),
        ((M.a > 1) | (M.b > 2), "(a > 1) or (b > 2)"),
    ],
)
def test_operator_overloads_render(node, source):
    assert node.to_source() == source


def test_builder_output_reparses_identically():
    for node in (
        (M.m1.diff() > 50) | (M.m1 > 5),
        M["train/loss"][50].zscore() > 4,
        M.loss[10].mean() + M.loss[10].std() * 2 > 1,
        ~M.loss[5].stalled(),
    ):
        assert parse(node.to_source()).to_source() == node.to_source()


def test_builder_and_parser_agree_numerically():
    ctx = context(loss=[1.0, 2.0, 3.0, 10.0])
    for node in (M.loss[4].mean(), M.loss.diff(), M.loss[4].max(), M.loss[4].slope()):
        assert evaluate(node, ctx) == evaluate(parse(node.to_source()), ctx)


def test_string_is_lifted_to_a_metric_reference():
    node = M.loss > "baseline"
    assert node.to_source() == "loss > baseline"


def test_unsupported_operand_type():
    with pytest.raises(TypeError, match="Cannot use"):
        _ = M.loss > object()


def test_nodes_reject_python_truthiness():
    for node in (M.loss > 1, M.loss, M.loss[3].mean()):
        with pytest.raises(TypeError, match="boolean context"):
            bool(node)


def test_unknown_attribute_extends_the_metric_name():
    """On a metric it is a nested name; after a window only functions make sense."""
    assert (
        M.loss.definitely_not_a_function.to_source() == "loss.definitely_not_a_function"
    )
    with pytest.raises(AttributeError, match="known functions"):
        M.loss[3].definitely_not_a_function()
    with pytest.raises(AttributeError):
        _ = M.loss._private
    with pytest.raises(AttributeError):
        _ = M.loss[3]._private


def test_metric_factory_rejects_dunder():
    with pytest.raises(AttributeError):
        _ = M.__wrapped__


# ---------------------------------------------------------------- introspection


def test_metrics_and_functions_are_collected():
    node = parse("mean(a[10]) > b + max(c, 2) or isnan(`d/e`)")
    assert node.metrics() == {"a", "b", "c", "d/e"}
    assert node.functions() == {"mean", "max", "isnan"}


def test_children_walk_covers_every_node_type():
    node = parse("not (mean(a[3]) + -b > 1) and c[2s] > 2")
    seen = set()
    stack = [node]
    while stack:
        current = stack.pop()
        seen.add(type(current).__name__)
        stack.extend(current.children())
    assert {"BoolOp", "Not", "Compare", "BinOp", "UnaryOp", "Call", "RangeRef"} <= seen


def test_repr_shows_the_source():
    assert repr(M.loss > 1) == "Compare(loss > 1)"


def test_literal_rendering():
    assert Literal(1.0).to_source() == "1"
    assert Literal(1.5).to_source() == "1.5"


def test_node_types_are_what_the_parser_builds():
    assert isinstance(parse("a or b"), BoolOp)
    assert isinstance(parse("not a"), Not)
    assert isinstance(parse("a > 1"), Compare)
    assert isinstance(parse("a + 1"), BinOp)
    assert isinstance(parse("-a"), UnaryOp)
    assert isinstance(parse("mean(a[2])"), Call)
    assert isinstance(parse("a[2]"), RangeRef)
    assert isinstance(parse("a"), MetricRef)
    assert isinstance(parse("1"), Literal)
