import json
import math
import threading

import pytest

from expr_tracker.history.reader import _decode
from expr_tracker.history.series import MetricSeries


def test_scalar_local_commits_bypass_python_serialization(run, monkeypatch):
    instance = run()

    def forbidden(*args, **kwargs):
        pytest.fail("primitive local records should not use SDK value encoding")

    monkeypatch.setattr(instance.history._codec, "encode", forbidden)
    for step in range(3):
        instance.log({"loss": step + 0.5, "count": step, "ok": True, "text": "value"})
    instance.history.flush()
    assert instance.summary["loss"] == 2.5
    assert [point[2] for point in instance.history.series.points("loss")] == [
        0.5,
        1.5,
        2.5,
    ]
    assert [record["count"] for record in instance.history.get(-1)] == [0, 1, 2]


@pytest.mark.parametrize("width", [5, 50, 500, 2000])
def test_wide_scalar_records_preserve_every_field(run, width):
    instance = run()
    metrics = {f"metric_{index}": index + 0.5 for index in range(width)}
    instance.log(metrics)
    instance.history.flush()
    row = instance.history.get(1)[0]
    assert {key: row[key] for key in metrics} == metrics
    assert dict(instance.summary) == metrics
    assert all(
        instance.history.series.latest(key)[2] == value
        for key, value in metrics.items()
    )


def test_local_record_snapshot_survives_mutation_during_flush(run, monkeypatch):
    instance = run(buffer_size=1)
    writer = instance.history.writer
    entered, release = threading.Event(), threading.Event()
    original = writer._write
    metrics = {"loss": 1.5, "payload": "x" * (128 << 10)}
    errors = []

    def write(payload):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("write was not released")
        original(payload)

    def produce():
        try:
            instance.log(metrics)
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(writer, "_write", write)
    producer = threading.Thread(target=produce)
    producer.start()
    try:
        assert entered.wait(5)
        metrics.update(loss=9.5, payload="changed", added=1)
    finally:
        release.set()
        producer.join(5)
    assert not producer.is_alive()
    assert not errors
    row = instance.history.get(1)[0]
    assert row["loss"] == instance.summary["loss"] == 1.5
    assert row["payload"] == "x" * (128 << 10)
    assert "added" not in row and "added" not in instance.summary
    assert json.loads(instance.history.log_fp.read_text()) == row


def test_backend_added_during_flush_receives_that_commit(run, monkeypatch):
    instance = run(buffer_size=1)
    observed = []

    class Backend:
        def log(self, metrics, step=None):
            observed.append((step, dict(metrics)))

    original = instance.history.writer._write

    def write(payload):
        original(payload)
        instance.backends["late"] = Backend()

    monkeypatch.setattr(instance.history.writer, "_write", write)
    instance.log({"loss": 7.0})
    assert observed == [(0, {"loss": 7.0})]
    assert instance.history.series.points("loss") == (
        (0, instance.history.get(1)[0]["_time"], 7.0),
    )
    instance.backends.clear()


def test_rule_added_during_flush_evaluates_that_commit(run, collector, monkeypatch):
    channel, messages = collector
    instance = run(alert={"channels": [channel()]}, buffer_size=1)
    original = instance.history.writer._write

    def write(payload):
        original(payload)
        instance.alerts.add_rule("loss > 5 => warning: high")

    monkeypatch.setattr(instance.history.writer, "_write", write)
    instance.log({"loss": 7.0})
    assert [message.text for message in messages] == ["high"]
    assert len(instance.history.series.points("loss")) == 1


def test_foreign_pinned_key_keeps_reentrant_summary_and_backend_order(run):
    instance = run()
    observed = []

    class Backend:
        def log(self, metrics, step=None):
            observed.append((step, metrics["loss"]))

    class Key(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            instance.backends["late"] = Backend()
            return super().__eq__(other)

    instance.summary[Key("loss")] = 99
    instance.log({"loss": 7.0})
    assert observed == [(0, 7.0)]
    assert instance.summary["loss"] == 99
    instance.backends.clear()


def test_numeric_conversion_reenters_series_without_holding_a_native_lock():
    series = MetricSeries()

    class Value(float):
        def __float__(self):
            series.add(9, 1.0, {"nested": 2.0})
            return 3.0

    series.add(10, 2.0, {"value": Value(1.0)})
    assert series.latest("nested") == (9, 1.0, 2.0)
    assert series.latest("value") == (10, 2.0, 3.0)


def test_bulk_decode_isolates_bad_rows_and_keeps_extended_json():
    rows = _decode(['{"_step": 0, "x": NaN}', '{"bad":', '{"_step": 1, "x": Infinity}'])
    assert [row["_step"] for row in rows] == [0, 1]
    assert math.isnan(rows[0]["x"]) and math.isinf(rows[1]["x"])


def test_default_query_results_are_independent_snapshots(store):
    history = store()
    history.log({"nested": {"items": [1, 2]}, "loss": 0.5})
    first = history.get(1)
    first[0]["loss"] = 7
    first[0]["nested"]["items"].append(3)
    assert history.get(1)[0]["nested"] == {"items": [1, 2]}
    assert history.get(1)[0]["loss"] == 0.5


@pytest.mark.parametrize("observer", ["backend", "rules", "callback"])
def test_observed_scalar_commits_use_native_admission(
    run, collector, monkeypatch, observer
):
    channel, messages = collector
    instance = run(alert={"channels": [channel()]})
    observed = []

    class Backend:
        def log(self, metrics, step=None):
            observed.append((step, dict(metrics)))

    if observer == "backend":
        instance.backends["test"] = Backend()
    elif observer == "rules":
        instance.alerts.add_rule("loss > 1 => warning: high")
    else:
        instance.history.on_commit = lambda row: observed.append(
            (row["_step"], {key: row[key] for key in ("loss", "count")})
        )

    def forbidden(*args, **kwargs):
        pytest.fail("observers should not force Python scalar serialization")

    monkeypatch.setattr(instance.history._codec, "encode", forbidden)
    instance.log({"loss": 2.5, "count": 7})
    assert instance.history.series.latest("loss")[2] == 2.5
    if observer == "rules":
        assert [message.text for message in messages] == ["high"]
    else:
        assert observed == [(0, {"loss": 2.5, "count": 7})]
    instance.backends.clear()


@pytest.mark.parametrize("method", ["ingest", "_ingest"])
def test_custom_ingest_hooks_are_not_bypassed(store, monkeypatch, method):
    history = store()
    observed = []
    original = getattr(history, method)

    def hook(*args, **kwargs):
        observed.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(history, method, hook)
    history.log({"loss": 2.5})
    assert observed == [True]
    assert history.get(1)[0]["loss"] == 2.5


def test_numpy_float64_snapshot_uses_native_scalar_transport(run, monkeypatch):
    np = pytest.importorskip("numpy")
    instance = run()
    metrics = {f"metric_{index}": np.float64(index + 0.5) for index in range(100)}
    metrics.update(nan=np.float64("nan"), inf=np.float64("inf"), neg=np.float64("-inf"))
    observed = []
    instance.history.on_commit = observed.append

    def forbidden(*args, **kwargs):
        pytest.fail("numpy.float64 should use native scalar admission")

    monkeypatch.setattr(instance.history._codec, "encode", forbidden)
    instance.log(metrics)
    instance.history.flush()
    row = json.loads(instance.history.log_fp.read_text())
    for index in range(100):
        key = f"metric_{index}"
        assert (
            row[key]
            == observed[0][key]
            == instance.history.series.latest(key)[2]
            == index + 0.5
        )
        assert type(observed[0][key]) is float
        assert type(metrics[key]) is np.float64
    assert math.isnan(row["nan"]) and math.isnan(observed[0]["nan"])
    assert row["inf"] == float("inf") and row["neg"] == float("-inf")


def test_fresh_scalar_serialization_handles_metadata_and_escaped_keys(store):
    history = store(buffer_size=1)
    metrics = {
        "\x00name": 5,
        'quote"\n\\': "text\t\x00",
        "\u4e2d\u6587\U0001f600": -(10**1000),
        "bool": True,
        "none": None,
        "negative_zero": -0.0,
        "_step": 999,
        "_time": 999,
    }
    assert history.log(metrics) == 0
    row = json.loads(history.log_fp.read_text())
    assert row["_step"] == 0 and row["_time"] != 999
    assert {key: row[key] for key in metrics if key not in ("_step", "_time")} == {
        key: value for key, value in metrics.items() if key not in ("_step", "_time")
    }
    assert math.copysign(1, row["negative_zero"]) == -1


def test_series_deque_preserves_retained_capacity_sparse_metrics_and_time_windows():
    series = MetricSeries(window=4)
    dense, sparse = [], []
    for step in range(29):
        record = {"dense": step}
        point = (step, step * 2.0, float(step))
        dense.append(point)
        if len(dense) > series.window * 2:
            dense = dense[-series.window :]
        if step % 3 == 0:
            record["sparse"] = step
            sparse.append(point)
            if len(sparse) > series.window * 2:
                sparse = sparse[-series.window :]
        series.add(step, step * 2.0, record)
        assert series.points("dense") == tuple(dense[-series.window :])
        assert series.points("sparse") == tuple(sparse[-series.window :])
        assert series.window_points("dense", count=3, duration=3) == dense[-2:]
        if step == 14:
            series.ensure_capacity(7)
            assert series.points("dense") == tuple(dense[-7:])
            assert series.points("sparse") == tuple(sparse[-7:])
    assert series.latest("dense") == dense[-1]
    assert series.latest("sparse") == sparse[-1]
    series.clear()
    assert series.names() == set()


@pytest.mark.parametrize("base", [int, float])
def test_exposing_native_snapshot_invalidates_validation_before_reentrant_conversion(
    base,
):
    from expr_tracker._native import NativeHistoryState, NativeMetricSeries

    state = NativeHistoryState(64 << 20, 100_000, False)
    series = NativeMetricSeries()
    record = state.ingest_values({"loss": 1.0}, None, True, False, {}, 1.0, 1.0)[1][0]
    observed = []

    class Value(base):
        def __float__(self):
            observed.append(series.names())
            return 7.0

    record.dictionary()["loss"] = Value(2)
    assert state.publish_local([record], series, None) == []
    assert series.latest('"loss"') == ("0", 1.0, 7.0)
    assert observed


@pytest.mark.parametrize("key", [1, "\ud800"])
def test_exposing_native_snapshot_invalidates_validation_before_key_changes(key):
    from expr_tracker._native import NativeHistoryState, NativeMetricSeries

    state = NativeHistoryState(64 << 20, 100_000, False)
    series = NativeMetricSeries()
    record = state.ingest_values({"loss": 1.0}, None, True, False, {}, 1.0, 1.0)[1][0]
    record.dictionary()[key] = 2.0
    assert state.publish_local([record], series, None) is None
    assert series.names() == set()


def test_validated_huge_integer_falls_back_before_any_observer_updates():
    from expr_tracker._native import (
        NativeHistoryState,
        NativeMetricSeries,
        NativeSummary,
    )

    state = NativeHistoryState(64 << 20, 100_000, False)
    series = NativeMetricSeries()
    summary = NativeSummary()
    records = state.ingest_values(
        {"loss": 1.0, "huge": 10**1000}, None, True, False, {}, 1.0, 1.0
    )[1]
    assert state.publish_local(records, series, summary) is None
    assert series.names() == set()
    assert summary.snapshot() == {}


def test_publication_capability_is_resolved_after_flush_without_a_run_dependency(
    store, monkeypatch
):
    history = store(buffer_size=1)
    observed = []
    flushed = []

    def callback(record):
        observed.append(record["_step"])

    def unavailable():
        assert flushed == [True]
        return None

    history.on_commit = callback
    history._set_inline_publication(callback, unavailable)
    original = history.writer._write

    def write(payload):
        original(payload)
        flushed.append(True)

    monkeypatch.setattr(history.writer, "_write", write)
    assert history.log({"loss": 2.5}) == 0
    assert observed == [0]
    assert history.series.latest("loss")[2] == 2.5
    assert "_local_run" not in history.__dict__


def test_summary_hook_installed_during_flush_receives_that_commit(run, monkeypatch):
    instance = run(buffer_size=1)
    observed = []
    original_write = instance.history.writer._write
    original_observe = instance.summary.observe

    def observe(metrics):
        observed.append(dict(metrics))
        original_observe(metrics)

    def write(payload):
        original_write(payload)
        monkeypatch.setattr(instance.summary, "observe", observe)

    monkeypatch.setattr(instance.history.writer, "_write", write)
    instance.log({"loss": 7.0})
    assert observed == [{"loss": 7.0}]
    assert instance.summary["loss"] == 7.0
    assert len(instance.history.series.points("loss")) == 1


def test_replaced_summary_is_resolved_at_publication_time(run, monkeypatch):
    from expr_tracker.summary import Summary

    instance = run(buffer_size=1)
    previous = instance.summary
    replacement = Summary()
    original = instance.history.writer._write

    def write(payload):
        original(payload)
        instance.summary = replacement

    monkeypatch.setattr(instance.history.writer, "_write", write)
    instance.log({"loss": 7.0})
    assert replacement["loss"] == 7.0
    assert dict(previous) == {}


def test_replaced_commit_callback_is_not_bypassed(run, monkeypatch):
    instance = run(buffer_size=1)
    observed = []
    original = instance.history.writer._write

    def write(payload):
        original(payload)
        instance.history.on_commit = lambda record: observed.append(record["_step"])

    monkeypatch.setattr(instance.history.writer, "_write", write)
    instance.log({"loss": 7.0})
    assert observed == [0]
    assert dict(instance.summary) == {}
    assert len(instance.history.series.points("loss")) == 1


def test_history_reinitialization_clears_previous_publication_capability(store):
    history = store()
    resolved = []

    def callback(record):
        pass

    def unavailable():
        resolved.append(True)
        return None

    history.on_commit = callback
    history._set_inline_publication(callback, unavailable)
    history.log({"loss": 1.0})
    assert resolved == [True]
    history.init(
        project="p", name="reinitialized", dir=str(history.log_dir.parent.parent)
    )
    history.log({"loss": 2.0})
    assert resolved == [True]
    assert history._inline_callback is None and history._inline_summary is None
