"""Expression tokenizer."""

from __future__ import annotations

from dataclasses import dataclass

NUMBER = "NUMBER"
DURATION = "DURATION"
NAME = "NAME"
OP = "OP"
EOF = "EOF"

KEYWORDS = {"and", "or", "not", "true", "false"}
QUOTES = "\"'`"
TWO_CHAR_OPS = {">=", "<=", "==", "!=", "&&", "||"}
ONE_CHAR_OPS = set("><+-*/%()[],|&!~")
DURATION_UNITS = (
    ("ms", 0.001),
    ("min", 60.0),
    ("s", 1.0),
    ("m", 60.0),
    ("h", 3600.0),
    ("d", 86400.0),
)


class ExprSyntaxError(ValueError):
    def __init__(self, message: str, source: str, position: int):
        self.source = source
        self.position = position
        super().__init__(
            f"{message} (at column {position + 1})\n  {source}\n  {' ' * position}^"
        )


@dataclass
class Token:
    kind: str
    value: object
    position: int

    def __repr__(self) -> str:
        return f"{self.kind}:{self.value!r}@{self.position}"


def tokenize(source: str) -> list[Token]:
    from ..._native import alert_tokenize

    return alert_tokenize(source)


def parse_duration(text: str) -> float:
    """Convert ``'30s'``, ``'5m'`` or ``'1.5h'`` to seconds."""
    stripped = text.strip()
    for unit, scale in sorted(DURATION_UNITS, key=lambda item: -len(item[0])):
        if stripped.endswith(unit):
            try:
                return float(stripped[: -len(unit)]) * scale
            except ValueError:
                break
    try:
        return float(stripped)
    except ValueError as e:
        raise ValueError(
            f"Invalid duration {text!r}; expected e.g. '30s', '5m', '2h'."
        ) from e


def is_name_char(ch: str) -> bool:
    return ch.isalnum() or ch in "_.@"


def scan_name(source: str, i: int) -> int:
    """Consume a metric name, which may contain ``/`` as in ``val/m1/acc@16``.

    A ``/`` only joins the name when a name character follows it immediately, so
    division has to be spaced: ``a / b`` divides, ``a/b`` is one metric.
    """
    from ..._native import alert_scan_name

    return alert_scan_name(source, i)
