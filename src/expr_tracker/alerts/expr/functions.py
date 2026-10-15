"""Function metadata and Python values for the native expression engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..._native import alert_scalar, alert_window

Point = tuple[int, float, float]


class _Unknown:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "UNKNOWN"

    def __bool__(self):
        raise TypeError("UNKNOWN has no truth value")


UNKNOWN = _Unknown()


@dataclass
class Window:
    name: str
    points: Sequence[Point]

    @property
    def values(self):
        return [p[2] for p in self.points]

    @property
    def steps(self):
        return [p[0] for p in self.points]

    @property
    def times(self):
        return [p[1] for p in self.points]

    def __len__(self):
        return len(self.points)


def _window_function(name):
    def call(window, *args):
        return alert_window(name, list(window.points), list(args))

    return call


def _scalar_function(name):
    def call(*args):
        return alert_scalar(name, list(args))

    return call


_WINDOW_META = {
    "mean": (None, 1, 1),
    "std": (None, 1, 1),
    "var": (None, 1, 1),
    "median": (None, 1, 1),
    "sum": (None, 1, 1),
    "first": (None, 1, 1),
    "last": (1, 1, 1),
    "count": (None, 1, 1),
    "diff": (2, 1, 1),
    "rate": (2, 1, 1),
    "pct_change": (2, 1, 1),
    "slope": (None, 1, 1),
    "zscore": (None, 1, 1),
    "ema": (None, 1, 2),
    "stalled": (None, 1, 2),
    "increasing": (None, 1, 1),
    "decreasing": (None, 1, 1),
}
WINDOW_FUNCS = {
    name: (_window_function(name), *meta) for name, meta in _WINDOW_META.items()
}
DUAL_FUNCS = {
    name: (_window_function(name), _scalar_function(name)) for name in ("min", "max")
}
SCALAR_FUNCS = {
    name: (_scalar_function(name), 1, 2 if name == "log" else 1)
    for name in ("abs", "log", "exp", "sqrt", "floor", "ceil")
}
SPECIAL_FUNCS = {
    "step": (0, 0),
    "elapsed": (0, 0),
    "no_data": (1, 1),
    "age": (1, 1),
    "has": (1, 1),
    "isnan": (1, 1),
    "isinf": (1, 1),
}
FUNCTIONS = {
    **dict.fromkeys(WINDOW_FUNCS, "window"),
    **dict.fromkeys(DUAL_FUNCS, "dual"),
    **dict.fromkeys(SCALAR_FUNCS, "scalar"),
    **dict.fromkeys(SPECIAL_FUNCS, "special"),
}


def arity(name):
    from ..._native import alert_function_meta

    return alert_function_meta(name)[1:3]


def default_window(name):
    from ..._native import alert_function_meta

    return alert_function_meta(name)[3]


# Private names remain aliases for integrations that used the old function table.
for _name in WINDOW_FUNCS:
    globals()["_" + _name] = WINDOW_FUNCS[_name][0]
