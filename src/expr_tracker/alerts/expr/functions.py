"""Expression function library. Window functions read a metric's own series, so
sparse records do not shift the window."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Point = tuple[int, float, float]


class _Unknown:
    """Three-valued UNKNOWN: missing data, short window, NaN or division by zero."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNKNOWN"

    def __bool__(self) -> bool:
        raise TypeError("UNKNOWN has no truth value")


UNKNOWN = _Unknown()


@dataclass
class Window:
    name: str
    points: Sequence[Point]

    @property
    def values(self) -> list[float]:
        return [p[2] for p in self.points]

    @property
    def steps(self) -> list[int]:
        return [p[0] for p in self.points]

    @property
    def times(self) -> list[float]:
        return [p[1] for p in self.points]

    def __len__(self) -> int:
        return len(self.points)


def _finite(value):
    if isinstance(value, _Unknown):
        return UNKNOWN
    if isinstance(value, bool):
        return value
    number = float(value)
    return UNKNOWN if math.isnan(number) or math.isinf(number) else number


def _need(window: Window, count: int):
    return len(window) >= count


# statistics uses exact fractions, too slow for the per-step hot path: use plain floats
def _fmean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _pvariance(values: Sequence[float]) -> float:
    mean = _fmean(values)
    return math.fsum((value - mean) ** 2 for value in values) / len(values)


def _pmedian(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


# ---------------------------------------------------------------------- window


def _mean(w: Window):
    return _finite(_fmean(w.values)) if _need(w, 1) else UNKNOWN


def _std(w: Window):
    return _finite(math.sqrt(_pvariance(w.values))) if _need(w, 2) else UNKNOWN


def _var(w: Window):
    return _finite(_pvariance(w.values)) if _need(w, 2) else UNKNOWN


def _median(w: Window):
    return _finite(_pmedian(w.values)) if _need(w, 1) else UNKNOWN


def _sum(w: Window):
    return _finite(sum(w.values)) if _need(w, 1) else UNKNOWN


def _min(w: Window):
    return _finite(min(w.values)) if _need(w, 1) else UNKNOWN


def _max(w: Window):
    return _finite(max(w.values)) if _need(w, 1) else UNKNOWN


def _first(w: Window):
    return _finite(w.values[0]) if _need(w, 1) else UNKNOWN


def _last(w: Window):
    return _finite(w.values[-1]) if _need(w, 1) else UNKNOWN


def _count(w: Window):
    return float(len(w))


def _diff(w: Window):
    return _finite(w.values[-1] - w.values[0]) if _need(w, 2) else UNKNOWN


def _rate(w: Window):
    if not _need(w, 2):
        return UNKNOWN
    span = w.steps[-1] - w.steps[0]
    if span <= 0:
        return UNKNOWN
    return _finite((w.values[-1] - w.values[0]) / span)


def _pct_change(w: Window):
    if not _need(w, 2) or w.values[0] == 0:
        return UNKNOWN
    return _finite((w.values[-1] - w.values[0]) / abs(w.values[0]))


def _slope(w: Window):
    if not _need(w, 2):
        return UNKNOWN
    xs, ys = w.steps, w.values
    mean_x, mean_y = _fmean(xs), _fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return UNKNOWN
    return _finite(
        sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        / denominator
    )


def _zscore(w: Window):
    if not _need(w, 2):
        return UNKNOWN
    values = w.values
    mean = _fmean(values)
    spread = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))
    if spread == 0:
        return UNKNOWN
    return _finite((values[-1] - mean) / spread)


def _ema(w: Window, alpha: float = 0.3):
    if not _need(w, 1) or not 0 < alpha <= 1:
        return UNKNOWN
    value = w.values[0]
    for point in w.values[1:]:
        value = alpha * point + (1 - alpha) * value
    return _finite(value)


def _stalled(w: Window, eps: float = 0.0):
    if not _need(w, 2):
        return UNKNOWN
    values = w.values
    if any(math.isnan(v) for v in values):
        return UNKNOWN
    return max(values) - min(values) <= eps


def _increasing(w: Window):
    if not _need(w, 2):
        return UNKNOWN
    return all(b > a for a, b in itertools.pairwise(w.values))


def _decreasing(w: Window):
    if not _need(w, 2):
        return UNKNOWN
    return all(b < a for a, b in itertools.pairwise(w.values))


# ---------------------------------------------------------------------- scalar


def _scalar(fn):
    def wrapper(*args):
        try:
            return _finite(fn(*args))
        except Exception:
            return UNKNOWN

    return wrapper


WINDOW_FUNCS: dict[str, tuple[Callable, int | None, int, int | None]] = {
    # name: (impl, default_window, min_args, max_args)
    "mean": (_mean, None, 1, 1),
    "std": (_std, None, 1, 1),
    "var": (_var, None, 1, 1),
    "median": (_median, None, 1, 1),
    "sum": (_sum, None, 1, 1),
    "first": (_first, None, 1, 1),
    "last": (_last, 1, 1, 1),
    "count": (_count, None, 1, 1),
    "diff": (_diff, 2, 1, 1),
    "rate": (_rate, 2, 1, 1),
    "pct_change": (_pct_change, 2, 1, 1),
    "slope": (_slope, None, 1, 1),
    "zscore": (_zscore, None, 1, 1),
    "ema": (_ema, None, 1, 2),
    "stalled": (_stalled, None, 1, 2),
    "increasing": (_increasing, None, 1, 1),
    "decreasing": (_decreasing, None, 1, 1),
}

DUAL_FUNCS = {"min": (_min, min), "max": (_max, max)}

SCALAR_FUNCS: dict[str, tuple[Callable, int, int | None]] = {
    "abs": (_scalar(abs), 1, 1),
    "log": (_scalar(math.log), 1, 2),
    "exp": (_scalar(math.exp), 1, 1),
    "sqrt": (_scalar(math.sqrt), 1, 1),
    "floor": (_scalar(math.floor), 1, 1),
    "ceil": (_scalar(math.ceil), 1, 1),
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

FUNCTIONS: dict[str, str] = {
    **dict.fromkeys(WINDOW_FUNCS, "window"),
    **dict.fromkeys(DUAL_FUNCS, "dual"),
    **dict.fromkeys(SCALAR_FUNCS, "scalar"),
    **dict.fromkeys(SPECIAL_FUNCS, "special"),
}


def arity(name: str) -> tuple[int, int | None]:
    if name in WINDOW_FUNCS:
        _, _, low, high = WINDOW_FUNCS[name]
        return low, high
    if name in DUAL_FUNCS:
        return 1, None
    if name in SCALAR_FUNCS:
        _, low, high = SCALAR_FUNCS[name]
        return low, high
    if name in SPECIAL_FUNCS:
        return SPECIAL_FUNCS[name]
    raise KeyError(name)


def default_window(name: str) -> int | None:
    if name in WINDOW_FUNCS:
        return WINDOW_FUNCS[name][1]
    return None
