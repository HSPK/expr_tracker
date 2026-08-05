"""Kleene three-valued evaluator.

Missing data, too few points, NaN and division by zero all yield ``UNKNOWN``; only
an exact ``True`` counts as a hit.
"""

from __future__ import annotations

import math
import time
from difflib import get_close_matches

from ...history.series import MetricSeries
from .functions import (
    DUAL_FUNCS,
    FUNCTIONS,
    SCALAR_FUNCS,
    UNKNOWN,
    WINDOW_FUNCS,
    Window,
    _Unknown,
    arity,
    default_window,
)
from .nodes import (
    BinOp,
    BoolOp,
    Call,
    Compare,
    Literal,
    MetricRef,
    Node,
    Not,
    RangeRef,
    UnaryOp,
)


class ExprError(ValueError):
    """Semantic error: unknown function, wrong arity, missing window and the like."""


class EvalContext:
    """Runtime state required to evaluate an expression."""

    def __init__(
        self,
        series: MetricSeries,
        *,
        step: int | None = None,
        now: float | None = None,
        started_at: float | None = None,
        last_commit_time: float | None = None,
        record: dict | None = None,
    ):
        self.series = series
        self.step = step
        self.now = now if now is not None else time.time()
        self.started_at = started_at if started_at is not None else self.now
        self.last_commit_time = last_commit_time
        self.record = record or {}

    def resolve(self, name: str) -> str | None:
        """Resolve a metric name: exact match first, then ``.`` rewritten to ``/``."""
        if self.series.has(name):
            return name
        alternative = name.replace(".", "/")
        if alternative != name and self.series.has(alternative):
            return alternative
        return None


def evaluate(node: Node, ctx: EvalContext):
    """Evaluate an AST node to ``float``, ``bool`` or ``UNKNOWN``."""
    if isinstance(node, Literal):
        return node.value
    if isinstance(node, MetricRef):
        return _latest(node.name, ctx)
    if isinstance(node, RangeRef):
        window = _window(node, ctx)
        return UNKNOWN if window is None or not len(window) else window.values[-1]
    if isinstance(node, Call):
        return _call(node, ctx)
    if isinstance(node, Not):
        value = _truth(evaluate(node.operand, ctx))
        return UNKNOWN if value is UNKNOWN else not value
    if isinstance(node, BoolOp):
        return _bool_op(node, ctx)
    if isinstance(node, Compare):
        return _compare(node, ctx)
    if isinstance(node, UnaryOp):
        value = evaluate(node.operand, ctx)
        return UNKNOWN if value is UNKNOWN else -_number(value)
    if isinstance(node, BinOp):
        return _bin_op(node, ctx)
    raise ExprError(f"Cannot evaluate node {node!r}")


def truthy(value) -> bool:
    """Only a definite True counts as a hit."""
    return _truth(value) is True


def tristate(value):
    """Reduce any evaluation result to ``True``, ``False`` or ``UNKNOWN``."""
    return _truth(value)


# ---------------------------------------------------------------------- internals


def _truth(value):
    if value is UNKNOWN:
        return UNKNOWN
    if isinstance(value, bool):
        return value
    number = float(value)
    return UNKNOWN if math.isnan(number) else number != 0


def _number(value) -> float:
    return 1.0 if value is True else 0.0 if value is False else float(value)


def _latest(name: str, ctx: EvalContext):
    resolved = ctx.resolve(name)
    if resolved is None:
        return UNKNOWN
    point = ctx.series.latest(resolved)
    if point is None:
        return UNKNOWN
    value = point[2]
    return UNKNOWN if math.isnan(value) or math.isinf(value) else value


def _window(node: Node, ctx: EvalContext, fallback: int | None = None) -> Window | None:
    if isinstance(node, RangeRef):
        name, count, duration = node.ref.name, node.count, node.duration
    elif isinstance(node, MetricRef):
        name, count, duration = node.name, fallback, None
    else:
        return None
    resolved = ctx.resolve(name)
    if resolved is None:
        return Window(name, ())
    points = ctx.series.window_points(
        resolved, count=count, duration=duration, now=ctx.now
    )
    return Window(resolved, points)


def _bool_op(node: BoolOp, ctx: EvalContext):
    values = [_truth(evaluate(value, ctx)) for value in node.values]
    if node.op == "or":
        if any(value is True for value in values):
            return True
        return UNKNOWN if any(value is UNKNOWN for value in values) else False
    if any(value is False for value in values):
        return False
    return UNKNOWN if any(value is UNKNOWN for value in values) else True


def _compare(node: Compare, ctx: EvalContext):
    left, right = evaluate(node.left, ctx), evaluate(node.right, ctx)
    if left is UNKNOWN or right is UNKNOWN:
        return UNKNOWN
    a, b = _number(left), _number(right)
    if not (math.isfinite(a) and math.isfinite(b)):
        return UNKNOWN
    return {
        ">": a > b,
        ">=": a >= b,
        "<": a < b,
        "<=": a <= b,
        "==": a == b,
        "!=": a != b,
    }[node.op]


def _bin_op(node: BinOp, ctx: EvalContext):
    left, right = evaluate(node.left, ctx), evaluate(node.right, ctx)
    if left is UNKNOWN or right is UNKNOWN:
        return UNKNOWN
    a, b = _number(left), _number(right)
    try:
        if node.op == "+":
            result = a + b
        elif node.op == "-":
            result = a - b
        elif node.op == "*":
            result = a * b
        elif node.op == "/":
            result = a / b
        elif node.op == "%":
            result = a % b
        else:
            raise ExprError(f"Unknown operator {node.op!r}")
    except ZeroDivisionError:
        return UNKNOWN
    return UNKNOWN if math.isnan(result) or math.isinf(result) else result


def _call(node: Call, ctx: EvalContext):
    """Call a function; unexpected implementation errors degrade to UNKNOWN."""
    try:
        return _call_impl(node, ctx)
    except ExprError:
        raise
    except Exception:
        return UNKNOWN


def _call_impl(node: Call, ctx: EvalContext):
    kind = FUNCTIONS.get(node.func)
    if kind is None:
        raise ExprError(
            f"Unknown function {node.func!r}; available: {', '.join(sorted(FUNCTIONS))}"
        )
    if kind == "special":
        return _special(node, ctx)
    if kind == "scalar":
        impl, _, _ = SCALAR_FUNCS[node.func]
        args = [evaluate(arg, ctx) for arg in node.args]
        if any(arg is UNKNOWN for arg in args):
            return UNKNOWN
        return impl(*[_number(arg) for arg in args])
    if kind == "dual":
        window_impl, scalar_impl = DUAL_FUNCS[node.func]
        if isinstance(node.args[0], (MetricRef, RangeRef)) and len(node.args) == 1:
            window = _window(node.args[0], ctx)
            return window_impl(window) if window is not None else UNKNOWN
        args = [evaluate(arg, ctx) for arg in node.args]
        if any(arg is UNKNOWN for arg in args):
            return UNKNOWN
        return scalar_impl(*[_number(arg) for arg in args])
    impl, fallback, _, _ = WINDOW_FUNCS[node.func]
    window = _window(node.args[0], ctx, fallback)
    if window is None:
        raise ExprError(
            f"{node.func}() expects a metric as its first argument, "
            f"got {node.args[0].to_source()}"
        )
    extra = [evaluate(arg, ctx) for arg in node.args[1:]]
    if any(arg is UNKNOWN for arg in extra):
        return UNKNOWN
    return impl(window, *[_number(arg) for arg in extra])


def _special(node: Call, ctx: EvalContext):
    name = node.func
    if name == "step":
        return UNKNOWN if ctx.step is None else float(ctx.step)
    if name == "elapsed":
        return max(0.0, ctx.now - ctx.started_at)
    if name == "no_data":
        threshold = evaluate(node.args[0], ctx)
        if threshold is UNKNOWN:
            return UNKNOWN
        reference = ctx.last_commit_time or ctx.started_at
        return (ctx.now - reference) >= _number(threshold)
    metric = node.args[0]
    if not isinstance(metric, (MetricRef, RangeRef)):
        raise ExprError(f"{name}() expects a metric name, got {metric.to_source()}")
    metric_name = metric.name if isinstance(metric, MetricRef) else metric.ref.name
    resolved = ctx.resolve(metric_name)
    if name == "has":
        return resolved is not None
    if resolved is None:
        return UNKNOWN
    point = ctx.series.latest(resolved)
    if point is None:
        return UNKNOWN
    if name == "age":
        return max(0.0, ctx.now - point[1])
    if name == "isnan":
        return math.isnan(point[2])
    if name == "isinf":
        return math.isinf(point[2])
    raise ExprError(f"Unknown function {name!r}")


# ---------------------------------------------------------------------- validate


def validate(node: Node):
    """Static checks: function exists, arity matches, window supplied when required."""
    for child in node.children():
        validate(child)
    if not isinstance(node, Call):
        return
    kind = FUNCTIONS.get(node.func)
    if kind is None:
        matches = get_close_matches(node.func, sorted(FUNCTIONS), n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(matches)}?" if matches else ""
        raise ExprError(f"Unknown function {node.func!r}.{hint}")
    low, high = arity(node.func)
    if len(node.args) < low or (high is not None and len(node.args) > high):
        expected = (
            f"{low}" if high == low else f"{low}..{high if high is not None else 'N'}"
        )
        raise ExprError(
            f"{node.func}() takes {expected} argument(s), got {len(node.args)}"
        )
    if kind == "window":
        first = node.args[0]
        if not isinstance(first, (MetricRef, RangeRef)):
            raise ExprError(
                f"{node.func}() expects a metric as its first argument, "
                f"got {first.to_source()}"
            )
        if isinstance(first, MetricRef) and default_window(node.func) is None:
            raise ExprError(
                f"{node.func}() requires an explicit window, e.g. "
                f"{node.func}({first.to_source()}[20])"
            )
    if (
        kind == "dual"
        and not isinstance(node.args[0], (MetricRef, RangeRef))
        and len(node.args) < 2
    ):
        raise ExprError(
            f"{node.func}() needs a window (e.g. {node.func}(loss[20])) "
            f"or at least two scalar arguments"
        )
    if (
        kind == "special"
        and node.func in ("has", "age", "isnan", "isinf")
        and not isinstance(node.args[0], (MetricRef, RangeRef))
    ):
        raise ExprError(f"{node.func}() expects a metric name")


def explain(node: Node, ctx: EvalContext) -> str:
    """Render a condition with observed values, e.g. ``diff(m1)=63.2 > 50``."""
    if isinstance(node, BoolOp):
        return f" {node.op} ".join(explain(value, ctx) for value in node.values)
    if isinstance(node, Not):
        return f"not {explain(node.operand, ctx)}"
    if isinstance(node, Compare):
        return f"{_annotate(node.left, ctx)} {node.op} {_annotate(node.right, ctx)}"
    return _annotate(node, ctx)


def _annotate(node: Node, ctx: EvalContext) -> str:
    source = node.to_source()
    if isinstance(node, Literal):
        return source
    value = evaluate(node, ctx)
    return f"{source}={_format(value)}"


def _format(value) -> str:
    if isinstance(value, _Unknown):
        return "?"
    if isinstance(value, bool):
        return str(value).lower()
    number = float(value)
    return f"{number:.6g}"
