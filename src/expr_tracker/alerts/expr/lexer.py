"""Expression tokenizer."""

from __future__ import annotations

from dataclasses import dataclass

NUMBER = "NUMBER"
DURATION = "DURATION"
NAME = "NAME"
OP = "OP"
EOF = "EOF"

KEYWORDS = {"and", "or", "not", "true", "false"}
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
    tokens: list[Token] = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        start = i
        if ch == "`":
            end = source.find("`", i + 1)
            if end < 0:
                raise ExprSyntaxError("Unterminated `identifier`", source, i)
            tokens.append(Token(NAME, source[i + 1 : end], start))
            i = end + 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            i = _scan_number(source, i)
            text = source[start:i]
            unit_end, seconds = _scan_duration_unit(source, i, float(text))
            if seconds is not None:
                tokens.append(Token(DURATION, seconds, start))
                i = unit_end
            else:
                tokens.append(Token(NUMBER, float(text), start))
            continue
        if ch.isalpha() or ch == "_":
            while i < n and (source[i].isalnum() or source[i] in "_."):
                i += 1
            word = source[start:i]
            tokens.append(
                Token(
                    OP if word in KEYWORDS and word not in ("true", "false") else NAME,
                    word,
                    start,
                )
            )
            continue
        if source[i : i + 2] in TWO_CHAR_OPS:
            tokens.append(Token(OP, source[i : i + 2], start))
            i += 2
            continue
        if ch in ONE_CHAR_OPS:
            tokens.append(Token(OP, ch, start))
            i += 1
            continue
        raise ExprSyntaxError(f"Unexpected character {ch!r}", source, i)
    tokens.append(Token(EOF, None, n))
    return tokens


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


def _scan_number(source: str, i: int) -> int:
    n = len(source)
    while i < n and (source[i].isdigit() or source[i] == "."):
        i += 1
    if i < n and source[i] in "eE":
        j = i + 1
        if j < n and source[j] in "+-":
            j += 1
        if j < n and source[j].isdigit():
            i = j
            while i < n and source[i].isdigit():
                i += 1
    return i


def _scan_duration_unit(source: str, i: int, value: float) -> tuple[int, float | None]:
    n = len(source)
    for unit, scale in sorted(DURATION_UNITS, key=lambda item: -len(item[0])):
        end = i + len(unit)
        if source[i:end] == unit and (
            end >= n or not (source[end].isalnum() or source[end] in "_.")
        ):
            return end, value * scale
    return i, None
