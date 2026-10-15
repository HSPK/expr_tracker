"""Python context and exception adapters for the native alert evaluator."""

from __future__ import annotations

import time

from ..._native import alert_evaluate, alert_explain, alert_tristate, alert_validate
from ...history.series import MetricSeries
from .functions import UNKNOWN


class ExprError(ValueError):
    """Semantic error in an alert expression."""


class EvalContext:
    """Runtime state passed to Rust through the public MetricSeries interface."""

    _native_series_methods = (
        MetricSeries.has,
        MetricSeries.latest,
        MetricSeries.window_points,
    )

    def __init__(
        self,
        series,
        *,
        step=None,
        now=None,
        started_at=None,
        last_commit_time=None,
        record=None,
    ):
        self.series = series
        self.step = step
        self.now = now if now is not None else time.time()
        self.started_at = started_at if started_at is not None else self.now
        self.last_commit_time = last_commit_time
        self.record = record or {}

    def resolve(self, name):
        from ..._native import alert_resolve

        return alert_resolve(self.series, name)

    _native_resolve = resolve


evaluate = alert_evaluate
validate = alert_validate
explain = alert_explain


def tristate(value):
    result = alert_tristate(value)
    return UNKNOWN if result is None else result


def truthy(value):
    return tristate(value) is True
