"""Behavioral regressions for the Rust alert core."""

import math
import threading
import time

import pytest
from expr_tracker._native import AlertExpression

from expr_tracker.alerts import (
    AlertConfig,
    AlertMessage,
    ChannelConfig,
    Dispatcher,
    WebhookPolicy,
)
from expr_tracker.alerts.backends.base import SendError
from expr_tracker.alerts.expr import UNKNOWN, EvalContext, M, evaluate, explain, parse
from expr_tracker.alerts.expr.functions import WINDOW_FUNCS, Window
from expr_tracker.alerts.expr.lexer import ExprSyntaxError, tokenize
from expr_tracker.alerts.expr.rule import find_top_level, split_rule
from expr_tracker.history import MetricSeries


@pytest.mark.parametrize(
    ("a", "b"),
    [(1, 1e20), (-1, 1e20), (1, -1e20), (-7, 4), (7, -4), (0, -4)],
)
def test_modulo_retains_python_numeric_semantics(a, b):
    series = MetricSeries()
    series.add(0, 0.0, {"a": a, "b": b})
    actual = evaluate(parse("a % b"), EvalContext(series))
    assert actual == a % b
    if actual == 0:
        assert math.copysign(1, actual) == math.copysign(1, b)


def test_window_functions_do_not_narrow_arbitrary_precision_steps():
    start = 10**1000
    window = Window("loss", [(start, 1.0, 1.0), (start + 2, 2.0, 9.0)])
    assert WINDOW_FUNCS["mean"][0](window) == 5.0
    assert WINDOW_FUNCS["diff"][0](window) == 8.0
    assert WINDOW_FUNCS["rate"][0](window) == 4.0
    assert WINDOW_FUNCS["slope"][0](window) is UNKNOWN


def test_statistical_mean_preserves_small_contributions():
    window = Window("loss", [(0, 0.0, 1e16), (1, 1.0, 1.0), (2, 2.0, -1e16)])
    assert WINDOW_FUNCS["mean"][0](window) == 1 / 3


@pytest.mark.parametrize("quote", ['"', "'", "`"])
@pytest.mark.parametrize(
    "name", ["file\udcff", "\ud800", "\U000f0000", "\U000f0000\ue001"]
)
def test_quoted_unicode_metric_names_are_lossless(quote, name):
    series = MetricSeries()
    series.add(0, 0.0, {name: 3.0})
    source = f"{quote}{name}{quote} > 0"
    node = parse(source)
    assert node.metrics() == {name}
    assert evaluate(node, EvalContext(series)) is True
    assert AlertExpression(node).evaluate(EvalContext(series)) is True
    assert name in explain(node, EvalContext(series))
    rule = source + " => warning: failed on file\udcff"
    assert find_top_level(rule, "=>") == rule.index("=>")
    assert split_rule(rule) == (source, "warning", "failed on file\udcff")
    assert tokenize(source)[1].position == source.index(">")


def test_surrogate_syntax_diagnostic_positions_are_python_character_offsets():
    source = "'file\udcff' + )"
    with pytest.raises(ExprSyntaxError) as caught:
        parse(source)
    assert caught.value.source == source
    assert caught.value.position == source.index(")")


@pytest.mark.parametrize(
    "source",
    [
        "x",
        "x[3]",
        "mean(x[3])",
        "std(x[3])",
        "var(x[3])",
        "median(x[3])",
        "sum(x[3])",
        "min(x[3])",
        "max(x[3])",
        "first(x[3])",
        "last(x)",
        "count(x[3])",
        "diff(x)",
        "rate(x)",
        "pct_change(x)",
        "slope(x[3])",
        "zscore(x[3])",
        "ema(x[3], 0.2)",
        "stalled(x[3])",
        "increasing(x[3])",
        "decreasing(x[3])",
        "abs(x)",
        "sqrt(x)",
        "log(x, 2)",
        "exp(x)",
        "floor(x)",
        "ceil(x)",
        "min(x, 2)",
        "max(x, 2)",
        "has(x)",
        "age(x)",
        "isnan(x)",
        "isinf(x)",
        "no_data(5)",
        "elapsed()",
        "step()",
        "x > 1 or x < 0",
        "has(x) and not x > 2",
        "(x + 2) / 3",
        "x % 4",
    ],
)
@pytest.mark.parametrize(
    "values",
    [[1.0, 2.0, 3.0], [float("nan"), 2.0, 3.0], [1.0, 2.0, float("inf")], []],
)
def test_compiled_plan_matches_public_evaluator(source, values):
    series = MetricSeries()
    for step, value in enumerate(values):
        series.add(step, float(step), {"x": value})
    context = EvalContext(
        series,
        step=2,
        now=10.0,
        started_at=0.0,
        last_commit_time=2.0,
    )
    node = parse(source)
    expected = evaluate(node, context)
    actual = AlertExpression(node).evaluate(context)
    if expected is UNKNOWN:
        assert actual is UNKNOWN
    elif isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected


def test_generic_evaluation_observes_mutated_ast_and_context():
    series = MetricSeries()
    series.add(0, 0.0, {"x": 1.0, "y": 9.0})
    context = EvalContext(series, now=0.0)
    node = parse("mean(x[2]) > 5")
    assert evaluate(node, context) is False
    node.left.args[0].ref.name = "y"
    assert evaluate(node, context) is True
    node.right.value = 10.0
    assert evaluate(node, context) is False
    node.right.value = 5.0
    replacement = MetricSeries()
    replacement.add(0, 0.0, {"y": 2.0})
    context.series = replacement
    assert evaluate(node, context) is False


def test_generic_evaluation_observes_custom_resolver():
    series = MetricSeries()
    series.add(0, 0.0, {"actual": 9.0})
    context = EvalContext(series)
    context.resolve = lambda name: "actual"
    assert evaluate(parse("mean(alias[2])"), context) == 9.0


@pytest.mark.parametrize("count", [-3, -1, 0, 1, 3])
def test_generic_fluent_windows_preserve_signed_counts(count):
    series = MetricSeries()
    series.add(0, 0.0, {"x": 7.0})
    context = EvalContext(series)
    result = evaluate(M.x[count], context)
    if count <= 0:
        assert result is UNKNOWN
    else:
        assert result == 7.0


@pytest.mark.parametrize("method", ["has", "latest", "window_points"])
def test_generic_evaluation_observes_overridden_series_methods(method):
    series = MetricSeries()
    series.add(0, 0.0, {"x": 1.0})
    context = EvalContext(series)
    if method == "has":
        series.has = lambda name: False
        assert evaluate(parse("has(x)"), context) is False
    elif method == "latest":
        series.latest = lambda name: (0, 0.0, 9.0)
        assert evaluate(parse("x"), context) == 9.0
    else:
        series.window_points = lambda *args, **kwargs: [(0, 0.0, 9.0)]
        assert evaluate(parse("mean(x[2])"), context) == 9.0


@pytest.mark.parametrize(
    "character",
    [chr(code) for code in range(160)]
    + ["\ud800", "\udcff", "\uffff", "\U00010000", "\U0010ffff", "\U000f0000"],
)
def test_native_metric_key_encoding_matches_python_json(character):
    name = "metric/" + character + "\u4e2d\u6587"
    series = MetricSeries()
    series.add(0, 0.0, {name: 3.0})
    node = M[name]
    context = EvalContext(series)
    assert evaluate(node, context) == 3.0
    assert AlertExpression(node).evaluate(context) == 3.0


def test_surrogate_tags_channel_names_and_dedup_coalescing_are_lossless():
    received = []
    name = "channel\udcff"
    dispatcher = Dispatcher(
        AlertConfig(
            channels=[
                ChannelConfig(
                    type="callable",
                    name=name,
                    tags=["tag\ud800"],
                    options={"handler": received.append},
                    policy=WebhookPolicy(
                        async_send=False,
                        dedup_window=0.01,
                        rate_limit_per_minute=None,
                    ),
                )
            ]
        )
    )
    message = AlertMessage(
        "file\udcff",
        "body\ud800",
        dedup_key="key\udcff",
        tags=["tag\ud800"],
    )
    try:
        dispatcher.send(message, channels=[name])
        dispatcher.send(message, channels=[name])
        time.sleep(0.02)
        dispatcher.send(message, channels=[name])
        assert received[0].title == message.title
        assert received[1].text.startswith(message.text)
        assert "1 similar alerts suppressed" in received[1].text
    finally:
        dispatcher.close()


def test_backend_failure_preserves_exception_cause():
    def fail(message):
        raise SendError("backend rejected", retryable=False)

    dispatcher = Dispatcher(
        AlertConfig(
            channels=[
                ChannelConfig(
                    type="callable",
                    options={"handler": fail},
                    policy=WebhookPolicy(
                        async_send=False,
                        dedup_window=0,
                        rate_limit_per_minute=None,
                        fail_silently=False,
                    ),
                )
            ]
        )
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            dispatcher.send(AlertMessage("title", "body"))
        assert isinstance(caught.value.__cause__, SendError)
        assert dispatcher.stats()["callable"]["failed"] == 1
    finally:
        dispatcher.close()


def test_concurrent_blocking_producers_preserve_all_messages():
    received = []
    lock = threading.Lock()

    def collect(message):
        time.sleep(0.001)
        with lock:
            received.append(message.title)

    dispatcher = Dispatcher(
        AlertConfig(
            channels=[
                ChannelConfig(
                    type="callable",
                    options={"handler": collect},
                    policy=WebhookPolicy(
                        async_send=True,
                        queue_size=1,
                        on_queue_full="block",
                        dedup_window=0,
                        rate_limit_per_minute=None,
                    ),
                )
            ]
        )
    )
    threads = [
        threading.Thread(
            target=lambda index=i: [
                dispatcher.send(AlertMessage(f"{index}:{j}", "body")) for j in range(10)
            ]
        )
        for i in range(4)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
            assert not thread.is_alive()
    finally:
        dispatcher.close()
    assert len(received) == len(set(received)) == 40
    assert dispatcher.stats()["callable"]["dropped"] == 0
