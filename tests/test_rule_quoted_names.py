"""Rule metadata separators cannot split a lexer-supported quoted metric name."""

import pytest

import expr_tracker as et
from expr_tracker.alerts.expr import EvalContext, evaluate, parse, parse_rule
from expr_tracker.alerts.expr.lexer import QUOTES
from expr_tracker.history import MetricSeries

NAMES = ("a=>b", "a,b", "a(b", "a]b", "a`b", 'a"b', "a'b")


@pytest.mark.parametrize(
    ("quote", "name"),
    [(quote, name) for quote in QUOTES for name in NAMES if quote not in name],
)
@pytest.mark.parametrize("metadata", [" => warning: changed", ", warning, changed"])
def test_all_quote_styles_protect_rule_metadata_separators(quote, name, metadata):
    condition = f"{quote}{name}{quote} > 0"
    rule = parse_rule(condition + metadata)
    assert rule.condition == condition
    assert rule.level.value == "warning"
    assert rule.message == "changed"
    series = MetricSeries()
    series.add(0, 0.0, {name: 1})
    assert evaluate(parse(rule.condition), EvalContext(series)) is True


def test_quoted_rule_names_work_through_run_alert_registration(run, collector):
    channel, messages = collector
    run(
        alert={"channels": [channel()]},
        alert_rules=[
            '"a=>b" > 0 => warning: arrow',
            "'a,b' > 0, warning, comma",
            '"a(b" > 0 => warning: bracket',
        ],
    )
    et.log({"a=>b": 1, "a,b": 1, "a(b": 1})
    assert [message.text for message in messages] == ["arrow", "comma", "bracket"]
