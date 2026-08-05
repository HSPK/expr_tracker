"""Pratt-style recursive descent parser.

Precedence, lowest to highest::

    or < and < not < comparison < + - < * / % < unary < call / window selector

This differs from Python, where ``|`` binds tighter than comparison and
``diff(m1) > 50 | m1 > 5`` would parse as ``diff(m1) > (50 | m1) > 5``. Here it
parses the intuitive way, as ``(diff(m1) > 50) or (m1 > 5)``.
"""

from __future__ import annotations

from .lexer import DURATION, EOF, NAME, NUMBER, ExprSyntaxError, Token, tokenize
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

OR_TOKENS = {"or", "||", "|"}
AND_TOKENS = {"and", "&&", "&"}
NOT_TOKENS = {"not", "!", "~"}
COMPARE_OPS = {">", ">=", "<", "<=", "==", "!="}


def parse(source: str) -> Node:
    """Parse expression text into an AST."""
    return _Parser(source).parse()


class _Parser:
    def __init__(self, source: str):
        self.source = source
        self.tokens = tokenize(source)
        self.pos = 0

    # ------------------------------------------------------------------ helpers

    @property
    def current(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def match(self, *values: str) -> bool:
        token = self.current
        return token.kind == "OP" and token.value in values

    def expect(self, value: str) -> Token:
        if not self.match(value):
            raise ExprSyntaxError(
                f"Expected {value!r} but found {self.current.value!r}",
                self.source,
                self.current.position,
            )
        return self.advance()

    def error(self, message: str) -> ExprSyntaxError:
        return ExprSyntaxError(message, self.source, self.current.position)

    # ------------------------------------------------------------------ grammar

    def parse(self) -> Node:
        if self.current.kind == EOF:
            raise self.error("Empty expression")
        node = self.parse_or()
        if self.current.kind != EOF:
            raise self.error(f"Unexpected token {self.current.value!r}")
        return node

    def parse_or(self) -> Node:
        values = [self.parse_and()]
        while self.match(*OR_TOKENS):
            self.advance()
            values.append(self.parse_and())
        return values[0] if len(values) == 1 else BoolOp("or", values)

    def parse_and(self) -> Node:
        values = [self.parse_not()]
        while self.match(*AND_TOKENS):
            self.advance()
            values.append(self.parse_not())
        return values[0] if len(values) == 1 else BoolOp("and", values)

    def parse_not(self) -> Node:
        if self.match(*NOT_TOKENS):
            self.advance()
            return Not(self.parse_not())
        return self.parse_compare()

    def parse_compare(self) -> Node:
        left = self.parse_arith()
        comparisons: list[Node] = []
        while self.match(*COMPARE_OPS):
            op = str(self.advance().value)
            right = self.parse_arith()
            comparisons.append(Compare(op, left, right))
            left = right
        if not comparisons:
            return left
        return comparisons[0] if len(comparisons) == 1 else BoolOp("and", comparisons)

    def parse_arith(self) -> Node:
        node = self.parse_term()
        while self.match("+", "-"):
            op = str(self.advance().value)
            node = BinOp(op, node, self.parse_term())
        return node

    def parse_term(self) -> Node:
        node = self.parse_unary()
        while self.match("*", "/", "%"):
            op = str(self.advance().value)
            node = BinOp(op, node, self.parse_unary())
        return node

    def parse_unary(self) -> Node:
        if self.match("-", "+"):
            op = str(self.advance().value)
            operand = self.parse_unary()
            return operand if op == "+" else UnaryOp("-", operand)
        return self.parse_postfix()

    def parse_postfix(self) -> Node:
        node = self.parse_primary()
        if self.match("["):
            if not isinstance(node, MetricRef):
                raise self.error("Window selector [...] can only follow a metric name")
            self.advance()
            count, duration = self.parse_window()
            self.expect("]")
            node = RangeRef(node, count, duration)
        return node

    def parse_window(self) -> tuple[int | None, float | None]:
        token = self.current
        if token.kind == DURATION:
            self.advance()
            return None, float(token.value)  # type: ignore[arg-type]
        if token.kind == NUMBER:
            self.advance()
            value = float(token.value)  # type: ignore[arg-type]
            if value <= 0 or value != int(value):
                raise ExprSyntaxError(
                    "Window size must be a positive integer or a duration like 30s",
                    self.source,
                    token.position,
                )
            return int(value), None
        raise self.error("Expected a window size such as [20] or [30s]")

    def parse_primary(self) -> Node:
        token = self.current
        if self.match("("):
            self.advance()
            node = self.parse_or()
            self.expect(")")
            return node
        if token.kind in (NUMBER, DURATION):
            self.advance()
            return Literal(float(token.value))  # type: ignore[arg-type]
        if token.kind == NAME:
            self.advance()
            name = str(token.value)
            if name in ("true", "false"):
                return Literal(1.0 if name == "true" else 0.0)
            if self.match("("):
                return Call(name, self.parse_args())
            return MetricRef(name)
        raise self.error(f"Unexpected token {token.value!r}")

    def parse_args(self) -> list[Node]:
        self.expect("(")
        args: list[Node] = []
        if self.match(")"):
            self.advance()
            return args
        while True:
            args.append(self.parse_or())
            if self.match(","):
                self.advance()
                continue
            self.expect(")")
            return args
