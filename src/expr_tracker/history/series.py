"""Bounded per-metric numeric buffers backing alert evaluation (never hits disk).

Lists with amortised trimming beat ``deque`` here: ``points[-count:]`` is O(count),
while a deque cannot be sliced and would copy the whole buffer on every window read.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable, Sequence

from loguru import logger

Point = tuple[int, float, float]  # (step, timestamp, value)

DEFAULT_WINDOW = 4096


class MetricSeries:
    """Keep the last ``window`` numeric points per metric name.

    Non-numeric metrics only register their name, so ``has()`` still sees them.
    """

    def __init__(self, window: int = DEFAULT_WINDOW):
        self.window = max(2, int(window))
        self._trim_at = self.window * 2
        self._series: dict[str, list[Point]] = {}
        self._non_numeric: set[str] = set()
        self._lock = threading.RLock()

    def ensure_capacity(self, window: int):
        """Grow the retained window so a rule asking for ``window`` points is exact."""
        window = int(window)
        with self._lock:
            if window <= self.window:
                return
            self.window = window
            self._trim_at = window * 2

    def add(self, step: int, ts: float, record: dict):
        with self._lock:
            for key, value in record.items():
                if key.startswith("_"):
                    continue
                try:
                    number = _as_float(value)
                except (OverflowError, TypeError, ValueError):
                    # Valid JSON integers can exceed the alert series' float range.
                    if key not in self._non_numeric:
                        logger.warning(
                            f"Metric {key!r} cannot be represented as a float; "
                            "skipping its numeric series."
                        )
                    self._non_numeric.add(key)
                    continue
                if number is None:
                    self._non_numeric.add(key)
                    continue
                series = self._series.get(key)
                if series is None:
                    series = self._series[key] = []
                series.append((step, ts, number))
                if len(series) > self._trim_at:
                    del series[: len(series) - self.window]

    def backfill(self, records: Iterable[dict]):
        """Preload from history on resume so windows work across restarts."""
        invalid_timestamps = 0
        for record in records:
            step = record.get("_step")
            if isinstance(step, int):
                timestamp = _timestamp(record.get("_time"))
                if timestamp is None:
                    invalid_timestamps += 1
                    continue
                self.add(step, timestamp, record)
        if invalid_timestamps:
            logger.warning(
                f"Skipped {invalid_timestamps} history rows with invalid timestamps "
                "during alert backfill."
            )

    # ------------------------------------------------------------------ queries

    def names(self) -> set[str]:
        with self._lock:
            return set(self._series) | set(self._non_numeric)

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._series or name in self._non_numeric

    def points(self, name: str) -> Sequence[Point]:
        with self._lock:
            series = self._series.get(name)
            return tuple(series[-self.window :]) if series else ()

    def window_points(
        self,
        name: str,
        count: int | None = None,
        duration: float | None = None,
        now: float | None = None,
    ) -> Sequence[Point]:
        """Take the last ``count`` points, or those within ``duration`` seconds."""
        with self._lock:
            series = self._series.get(name)
            if not series:
                return ()
            start = max(0, len(series) - (count if count is not None else self.window))
            if duration is not None:
                cutoff = (now if now is not None else series[-1][1]) - duration
                index = len(series)
                while index > start and series[index - 1][1] >= cutoff:
                    index -= 1
                start = index
            return series[start:]

    def latest(self, name: str) -> Point | None:
        with self._lock:
            series = self._series.get(name)
            return series[-1] if series else None

    def clear(self):
        with self._lock:
            self._series.clear()
            self._non_numeric.clear()


def _as_float(value) -> float | None:
    """Coerce to float, keeping NaN/inf so ``isnan()``/``isinf()`` can see them."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _timestamp(value) -> float | None:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return timestamp if math.isfinite(timestamp) else None
