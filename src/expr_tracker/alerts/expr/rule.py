"""Rule parsing for ``<expr> => [level][: message]``, plus the legacy comma form."""

from __future__ import annotations

from typing import Any

from ..models import AlertLevel, AlertRule
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
        return text
    if isinstance(text, dict):
        return AlertRule.from_dict({**text, **overrides})
    if isinstance(text, Node):
        return AlertRule(condition=text.to_source(), **overrides)
    condition, level, message = split_rule(str(text))
    values: dict[str, Any] = {"condition": condition}
    if level is not None:
        values["level"] = level
    if message:
        values["message"] = message
    values.update(overrides)
    return AlertRule(**values)


def split_rule(text: str) -> tuple[str, str | None, str]:
    """Split into ``(condition, level, message)``, respecting brackets and backticks."""
    source = text.strip()
    if not source:
        raise ValueError("Empty alert rule")
    arrow = find_top_level(source, ARROW)
    if arrow >= 0:
        condition = source[:arrow].strip()
        rest = source[arrow + len(ARROW) :].strip()
        return condition, *_split_meta(rest)
    parts = split_top_level(source, ",")
    condition = parts[0].strip()
    level, message = None, ""
    if len(parts) >= 2:
        candidate = parts[1].strip()
        if _is_level(candidate):
            level = candidate
            message = ",".join(parts[2:]).strip()
        else:
            message = ",".join(parts[1:]).strip()
    if not condition:
        raise ValueError(f"Alert rule has no condition: {text!r}")
    return condition, level, message


def compile_condition(condition: str) -> Node:
    return parse(condition)


def _split_meta(rest: str) -> tuple[str | None, str]:
    if not rest:
        return None, ""
    if ":" in rest:
        head, _, tail = rest.partition(":")
        head = head.strip()
        if not head:
            return None, tail.strip()
        if _is_level(head):
            return head, tail.strip()
        return None, rest
    return (rest, "") if _is_level(rest) else (None, rest)


def _is_level(text: str) -> bool:
    try:
        AlertLevel.parse(text)
    except ValueError:
        return False
    return True


def find_top_level(text: str, token: str) -> int:
    """Find the first ``token`` outside brackets and backticks."""
    depth = 0
    quoted = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "`":
            quoted = not quoted
        elif not quoted:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif depth == 0 and text.startswith(token, i):
                return i
        i += 1
    return -1


def split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    quoted = False
    start = 0
    for i, ch in enumerate(text):
        if ch == "`":
            quoted = not quoted
        elif not quoted:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch == sep and depth == 0:
                parts.append(text[start:i])
                start = i + 1
    parts.append(text[start:])
    return parts
