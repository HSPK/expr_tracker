"""SDK numeric/name normalization for Rust-owned bounded metric windows."""

from __future__ import annotations

import json
import math
from functools import lru_cache

from loguru import logger

from .._native import NativeMetricSeries

Point = tuple[int, float, float]
DEFAULT_WINDOW = 4096


def _key(name):
    if type(name) is str:
        return _string_key(name)
    return json.dumps(name, ensure_ascii=True)


@lru_cache(maxsize=4096)
def _string_key(name):
    return json.dumps(name, ensure_ascii=True)


def _point(point):
    return (int(point[0]), point[1], point[2])


class MetricSeries:
    def __init__(self, window=DEFAULT_WINDOW):
        self._native = NativeMetricSeries(max(2, int(window)))

    @property
    def window(self):
        return self._native.window

    def ensure_capacity(self, window):
        self._native.ensure_capacity(max(0, int(window)))

    def add(self, step, ts, record):
        if type(record) is dict:
            warnings = self._native.add_record(str(step), float(ts), record)
        else:
            normalized = []
            for key, value in record.items():
                if key.startswith("_"):
                    continue
                try:
                    normalized.append((_key(key), _as_float(value), False))
                except (OverflowError, TypeError, ValueError):
                    normalized.append((_key(key), None, True))
            warnings = self._native.add(str(step), float(ts), normalized)
        for key in warnings:
            logger.warning(
                f"Metric {json.loads(key)!r} cannot be represented as a float; "
                "skipping its numeric series."
            )

    _native_add = add

    def backfill(self, records):
        invalid = 0
        for record in records:
            step = record.get("_step")
            if isinstance(step, int):
                timestamp = _timestamp(record.get("_time"))
                if timestamp is None:
                    invalid += 1
                    continue
                self.add(step, timestamp, record)
        if invalid:
            logger.warning(
                f"Skipped {invalid} history rows with invalid timestamps "
                "during alert backfill."
            )

    def names(self):
        return {json.loads(name) for name in self._native.names()}

    def has(self, name):
        return self._native.has(_key(name))

    def points(self, name):
        return tuple(map(_point, self._native.points(_key(name))))

    def window_points(self, name, count=None, duration=None, now=None):
        points = self._native.window_points(_key(name), count, duration, now)
        return list(map(_point, points)) if points is not None else ()

    def latest(self, name):
        point = self._native.latest(_key(name))
        return _point(point) if point is not None else None

    def clear(self):
        self._native.clear()


def _as_float(value):
    if isinstance(value, (bool, int, float)):
        return float(value)
    return None


def _timestamp(value):
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return timestamp if math.isfinite(timestamp) else None
