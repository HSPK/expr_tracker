"""Rule parsing for ``<expr> => [level][: message]``, plus the legacy comma form."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..models import AlertRule
from .nodes import Node
from .parser import parse

ARROW = "=>"


def parse_rule(text: str | Node | dict | AlertRule, **overrides: Any) -> AlertRule:
    """Parse one line of rule text into an :class:`AlertRule`.

    Accepts::

        diff(m1) > 50 or m1 > 5 => warn: m1 failure
        zscore(loss[50]) > 4 => error
        diff(m1) > 50, warn, m1 failure          # legacy comma form
    """
    if isinstance(text, AlertRule):
        if not overrides:
            return text
        values = asdict(text)
        if text.auto_named:
            values["name"] = None
        return AlertRule(**{**values, **overrides})
    if isinstance(text, dict):
        return AlertRule.from_dict({**text, **overrides})
    if isinstance(text, Node):
        return AlertRule(**{"condition": text.to_source(), **overrides})
    condition, level, message = split_rule(str(text))
    values: dict[str, Any] = {"condition": condition}
    if level is not None:
        values["level"] = level
    if message:
        values["message"] = message
    values.update(overrides)
    return AlertRule(**values)


def split_rule(text: str) -> tuple[str, str | None, str]:
    """Split rule metadata without cutting through brackets or quoted names."""
    from ..._native import alert_split_rule

    return alert_split_rule(text)


def compile_condition(condition: str) -> Node:
    return parse(condition)


def find_top_level(text: str, token: str) -> int:
    """Find the first ``token`` outside brackets and quoted names."""
    from ..._native import alert_top_level

    return alert_top_level(text, token, False)[0]


def split_top_level(text: str, sep: str) -> list[str]:
    from ..._native import alert_top_level

    return alert_top_level(text, sep, True)[1]
