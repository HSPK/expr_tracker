"""Expression AST nodes, built either by the parser or by the Python builder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


class Node:
    """Base node. Operator overloading lets ``M.loss[50].mean() > 5`` build an AST."""

    __hash__ = object.__hash__

    # comparison
    def __gt__(self, other):
        return Compare(">", self, _lift(other))

    def __ge__(self, other):
        return Compare(">=", self, _lift(other))

    def __lt__(self, other):
        return Compare("<", self, _lift(other))

    def __le__(self, other):
        return Compare("<=", self, _lift(other))

    def __eq__(self, other):  # type: ignore[override]
        return Compare("==", self, _lift(other))

    def __ne__(self, other):  # type: ignore[override]
        return Compare("!=", self, _lift(other))

    # logical
    def __and__(self, other):
        return BoolOp("and", [self, _lift(other)])

    def __or__(self, other):
        return BoolOp("or", [self, _lift(other)])

    def __invert__(self):
        return Not(self)

    # arithmetic
    def __add__(self, other):
        return BinOp("+", self, _lift(other))

    def __radd__(self, other):
        return BinOp("+", _lift(other), self)

    def __sub__(self, other):
        return BinOp("-", self, _lift(other))

    def __rsub__(self, other):
        return BinOp("-", _lift(other), self)

    def __mul__(self, other):
        return BinOp("*", self, _lift(other))

    def __rmul__(self, other):
        return BinOp("*", _lift(other), self)

    def __truediv__(self, other):
        return BinOp("/", self, _lift(other))

    def __rtruediv__(self, other):
        return BinOp("/", _lift(other), self)

    def __mod__(self, other):
        return BinOp("%", self, _lift(other))

    def __neg__(self):
        return UnaryOp("-", self)

    def __pos__(self):
        return self

    def __bool__(self):
        raise TypeError(
            "Alert expression nodes cannot be used in Python boolean context; "
            "use & | ~ instead of and/or/not."
        )

    # ------------------------------------------------------------------
    def to_source(self) -> str:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def metrics(self) -> set[str]:
        return set()

    def functions(self) -> set[str]:
        return set()

    def children(self) -> Sequence[Node]:
        return ()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_source()})"


@dataclass(eq=False, repr=False)
class Literal(Node):
    value: float

    def to_source(self) -> str:
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        text = repr(float(self.value))
        return text.removesuffix(".0")


@dataclass(eq=False, repr=False)
class MetricRef(Node):
    name: str

    def to_source(self) -> str:
        return self.name if _is_plain(self.name) else _quote(self.name)

    def metrics(self) -> set[str]:
        return {self.name}

    def __getitem__(self, window) -> RangeRef:
        return RangeRef(self, *_parse_window(window))

    def __getattr__(self, item) -> Node:
        from .functions import FUNCTIONS

        if item.startswith("_"):
            raise AttributeError(item)
        if item in FUNCTIONS:
            return _method(self, item)
        # Not a function, so it extends the name: M.train.loss is the metric
        # "train.loss", which resolves to "train/loss" exactly as the DSL does.
        # A metric whose name collides with a function needs M["mean"] instead.
        return MetricRef(f"{self.name}.{item}")


@dataclass(eq=False, repr=False)
class RangeRef(Node):
    ref: MetricRef
    count: int | None = None
    duration: float | None = None

    def to_source(self) -> str:
        window = (
            f"{self.count}"
            if self.count is not None
            else _format_duration(self.duration)
        )
        return f"{self.ref.to_source()}[{window}]"

    def metrics(self) -> set[str]:
        return self.ref.metrics()

    def children(self):
        return (self.ref,)

    def __getattr__(self, item) -> Node:
        return _method(self, item)


@dataclass(eq=False, repr=False)
class Call(Node):
    func: str
    args: list[Node] = field(default_factory=list)

    def to_source(self) -> str:
        return f"{self.func}({', '.join(a.to_source() for a in self.args)})"

    def metrics(self) -> set[str]:
        return set().union(*(a.metrics() for a in self.args)) if self.args else set()

    def functions(self) -> set[str]:
        inner = set().union(*(a.functions() for a in self.args)) if self.args else set()
        return {self.func} | inner

    def children(self):
        return tuple(self.args)


@dataclass(eq=False, repr=False)
class UnaryOp(Node):
    op: str
    operand: Node

    def to_source(self) -> str:
        return f"{self.op}{_wrap(self.operand)}"

    def metrics(self):
        return self.operand.metrics()

    def functions(self):
        return self.operand.functions()

    def children(self):
        return (self.operand,)


@dataclass(eq=False, repr=False)
class BinOp(Node):
    op: str
    left: Node
    right: Node

    def to_source(self) -> str:
        return f"{_wrap(self.left)} {self.op} {_wrap(self.right)}"

    def metrics(self):
        return self.left.metrics() | self.right.metrics()

    def functions(self):
        return self.left.functions() | self.right.functions()

    def children(self):
        return (self.left, self.right)


@dataclass(eq=False, repr=False)
class Compare(Node):
    op: str
    left: Node
    right: Node

    def to_source(self) -> str:
        return f"{_wrap(self.left)} {self.op} {_wrap(self.right)}"

    def metrics(self):
        return self.left.metrics() | self.right.metrics()

    def functions(self):
        return self.left.functions() | self.right.functions()

    def children(self):
        return (self.left, self.right)


@dataclass(eq=False, repr=False)
class BoolOp(Node):
    op: str  # and | or
    values: list[Node]

    def to_source(self) -> str:
        return f" {self.op} ".join(_wrap(v, self.op) for v in self.values)

    def metrics(self):
        return set().union(*(v.metrics() for v in self.values))

    def functions(self):
        return set().union(*(v.functions() for v in self.values))

    def children(self):
        return tuple(self.values)


@dataclass(eq=False, repr=False)
class Not(Node):
    operand: Node

    def to_source(self) -> str:
        return f"not {_wrap(self.operand)}"

    def metrics(self):
        return self.operand.metrics()

    def functions(self):
        return self.operand.functions()

    def children(self):
        return (self.operand,)


# ---------------------------------------------------------------------- builder


class _MetricFactory:
    """Builds metric references: ``M.loss`` or ``M["train/loss"]``."""

    def __getattr__(self, name: str) -> MetricRef:
        if name.startswith("_"):
            raise AttributeError(name)
        return MetricRef(name)

    def __getitem__(self, name: str) -> MetricRef:
        return MetricRef(name)


M = _MetricFactory()


def _method(target: Node, name: str):
    from .functions import FUNCTIONS

    if name.startswith("_") or name not in FUNCTIONS:
        raise AttributeError(
            f"{type(target).__name__!r} has no attribute {name!r}; "
            "known functions: " + ", ".join(sorted(FUNCTIONS))
        )

    def build(*args):
        return Call(name, [target, *(_lift(a) for a in args)])

    return build


def _lift(value) -> Node:
    if isinstance(value, Node):
        return value
    if isinstance(value, (int, float, bool)):
        return Literal(float(value))
    if isinstance(value, str):
        return MetricRef(value)
    raise TypeError(f"Cannot use {value!r} in an alert expression")


def _parse_window(window) -> tuple[int | None, float | None]:
    if isinstance(window, int):
        return window, None
    if isinstance(window, float):
        return None, window
    if isinstance(window, str):
        from .lexer import parse_duration

        return None, parse_duration(window)
    raise TypeError(
        f"Invalid window {window!r}; use an int (points) or '30s' (duration)"
    )


_PRECEDENCE = {"or": 1, "and": 2}


def _wrap(node: Node, parent_op: str | None = None) -> str:
    source = node.to_source()
    if isinstance(node, (Literal, MetricRef, RangeRef, Call, UnaryOp)):
        return source
    if (
        isinstance(node, BoolOp)
        and parent_op
        and _PRECEDENCE.get(node.op, 9) >= _PRECEDENCE.get(parent_op, 9)
    ):
        return source
    return f"({source})"


def _is_plain(name: str) -> bool:
    """Whether the name can be written bare and lex back to exactly itself."""
    from .lexer import is_name_char

    if not name or not (name[0].isalpha() or name[0] == "_"):
        return False
    for index, ch in enumerate(name):
        if is_name_char(ch):
            continue
        # A `/` only joins the name when a name character follows it
        if ch == "/" and index + 1 < len(name) and is_name_char(name[index + 1]):
            continue
        return False
    return True


def _quote(name: str) -> str:
    """Wrap a name in whichever quote it does not already contain."""
    for quote in ('"', "'", "`"):
        if quote not in name:
            return f"{quote}{name}{quote}"
    return f'"{name}"'  # pragma: no cover - a name with all three quote styles


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "0"
    for unit, size in (("h", 3600), ("m", 60), ("s", 1)):
        if seconds >= size and seconds % size == 0:
            return f"{int(seconds // size)}{unit}"
    return f"{seconds}s"
