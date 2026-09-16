"""Plugin values and attributes share log() encoding without slowing bare spans."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import expr_tracker as et


@pytest.mark.parametrize("value", [np.int64(2), np.float32(2.5)])
@pytest.mark.parametrize("span_metrics", [False, True])
def test_numpy_plugin_values_are_normalized_before_aggregation(run, value, span_metrics):
    instance = run(span_metrics=span_metrics)
    for _ in range(2):
        with et.span("work", plugins=[lambda span: {"items": value}]) as span:
            pass
        assert type(span.metrics["items"]) in (int, float)
    et.log({"loss": 1.0})
    et.finish()
    rows = instance.history_query(-1)
    assert len(rows) == 1 and rows[0]["loss"] == 1.0
    if span_metrics:
        assert rows[0]["items/work"] == 2 * value
    else:
        assert "items/work" not in rows[0]
    spans = [json.loads(line) for line in instance.history.spans_fp.read_text().splitlines()]
    assert len(spans) == 2
    assert all(record["metrics"]["items"] == value for record in spans)


def test_span_attributes_use_supported_log_value_encoders(run):
    instance = run()
    with et.span(
        "work",
        batch=np.int64(3),
        values=np.array([1, 2]),
        path=Path("batch.bin"),
        when=datetime(2026, 1, 1),
    ) as span:
        pass
    et.finish()
    record = json.loads(instance.history.spans_fp.read_text())
    assert record["args"] == {
        "batch": 3,
        "values": [1, 2],
        "path": "batch.bin",
        "when": "2026-01-01T00:00:00",
    }
    assert isinstance(span.attributes["values"], np.ndarray)


def test_bad_plugin_values_do_not_poison_other_metrics_or_span_events(run):
    class Unsupported:
        __slots__ = ()

        def __repr__(self):
            return "unsupported-plugin-value"

    instance = run()
    with et.span("work", plugins=[lambda span: {"bad": Unsupported(), "good": 2}]):
        pass
    et.log({"loss": 1.0})
    et.finish()
    row = instance.history_query(1)[0]
    assert row["good/work"] == 2 and row["loss"] == 1.0
    assert row["bad/work"] == "unsupported-plugin-value"
    event = json.loads(instance.history.spans_fp.read_text())
    assert event["metrics"] == {"bad": "unsupported-plugin-value", "good": 2}


@pytest.mark.parametrize("name", ["items", "gpu_percent", "gpu_mem_peak_mb"])
def test_plugin_type_changes_do_not_abort_recording(run, name):
    instance = run()
    for value in (1, "unavailable", 3):
        with et.span("work", plugins=[lambda span, value=value: {name: value}]):
            pass
    et.log({"loss": 1.0})
    et.finish()
    row = instance.history_query(1)[0]
    assert row[f"{name}/work"] == 3
    assert row["loss"] == 1.0
    assert len(instance.history.spans_fp.read_text().splitlines()) == 3


def test_normalized_plugin_keys_cannot_override_reserved_metrics(run):
    instance = run(span_count=True)
    with et.span(
        "work",
        plugins=[lambda span: {b"time_ms": 999, b"count": 999, b"good": np.int64(2)}],
    ) as span:
        pass
    et.log({})
    row = instance.history_query(1)[0]
    assert row["time_ms/work"] == span.duration_ms
    assert row["count/work"] == 1
    assert row["good/work"] == 2
    assert span.metrics == {"good": 2}


def test_bare_spans_do_not_run_the_value_encoder(run, monkeypatch):
    instance = run()

    def forbidden(*args, **kwargs):
        pytest.fail("bare spans should bypass value encoding")

    monkeypatch.setattr(instance.history, "_encode_span_data", forbidden)
    with et.span("work"):
        pass
