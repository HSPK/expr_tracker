"""Performance guard: log() overhead must stay small enough for training loops."""

import sys
import time

import pytest

from expr_tracker.history import HistoryStore

RULES = [
    "diff(loss) > 50 => warn: spike",
    "zscore(loss[50]) > 4 => error: z",
    "mean(loss[20]) > 100 => warn: mean",
    "stalled(loss[100]) => warn: stalled",
    "isnan(loss) or isinf(loss) => critical: nan",
    "slope(loss[30]) > 1 => warn: slope",
    "acc < 0.1 => warn: acc",
    "lr < 1e-8 => warn: lr",
    "grad_norm > 1e4 => error: grad",
    "count(loss[10]) < 1 => warn: gap",
]


def build(tmp_path, rules):
    from expr_tracker.alerts.dispatch import Dispatcher
    from expr_tracker.alerts.engine import AlertEngine
    from expr_tracker.alerts.expr import EvalContext
    from expr_tracker.alerts.models import AlertConfig

    store = HistoryStore()
    engine = None
    if rules:
        dispatcher = Dispatcher(AlertConfig())

        def context(record):
            record = record or {}
            return EvalContext(
                store.series,
                step=record.get("_step"),
                started_at=store.started_at,
                last_commit_time=record.get("_time"),
                record=record,
            )

        engine = AlertEngine(dispatcher, context, rules=rules, watchdog_interval=0)
    store.init(
        project="p",
        name="perf",
        dir=str(tmp_path),
        max_open_seconds=None,
        buffer_size=200,
        on_commit=engine.on_step if engine else None,
    )
    return store


def measure(store, iterations=2000):
    payload = {
        "loss": 1.0,
        "acc": 0.9,
        "lr": 1e-4,
        "grad_norm": 1.0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
    }
    start = time.perf_counter()
    for _ in range(iterations):
        store.log(dict(payload))
    elapsed = time.perf_counter() - start
    store.finish()
    return elapsed / iterations


@pytest.mark.parametrize("rules", [[], RULES])
def test_log_overhead_stays_small(tmp_path, rules):
    if sys.gettrace() is not None:  # coverage/debugger tracing skews the numbers
        pytest.skip("timing is meaningless under tracing")
    store = build(tmp_path / ("with" if rules else "without"), rules)
    per_call = measure(store)
    assert per_call < 300e-6, f"log() took {per_call * 1e6:.1f}us per call"
