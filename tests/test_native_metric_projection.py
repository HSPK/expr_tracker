"""Observer projection preserves the independent shallow metrics snapshot."""

import math

import pytest
from expr_tracker._native import history_metric_values


def test_native_projection_preserves_order_identity_and_unusual_values():
    key = "".join(["unicode-", "\u4e2d\u6587", "\0"])
    payload = {"nested": [1]}
    record = {
        "_step": 7,
        "_time": 1.0,
        "_hidden": "excluded",
        key: payload,
        "": None,
        "nan": float("nan"),
        "inf": float("inf"),
        "huge": 10**1000,
        3: "excluded",
    }
    metrics = history_metric_values(record)
    assert metrics is not None and metrics is not record
    assert list(metrics) == [key, "", "nan", "inf", "huge"]
    assert next(iter(metrics)) is key
    assert metrics[key] is payload
    assert metrics[""] is None
    assert math.isnan(metrics["nan"]) and metrics["inf"] == float("inf")
    assert metrics["huge"] is record["huge"]
    metrics["added"] = 1
    assert "added" not in record


def test_empty_and_metadata_only_records_produce_an_empty_snapshot():
    for record in ({}, {"_step": 0, "_time": 1.0, "_hidden": 1}):
        metrics = history_metric_values(record)
        assert metrics == {} and metrics is not record


def test_dictionary_subclass_keeps_its_items_hook(run):
    calls = []

    class Record(dict):
        def items(self):
            calls.append("items")
            return [("_step", 0), ("from_hook", 3)]

    record = Record(_step=0, original=1)
    assert history_metric_values(record) is None
    assert calls == []
    instance = run()
    instance._on_commit(record)
    assert calls == ["items"]
    assert dict(instance.summary) == {"from_hook": 3}


def test_string_subclass_keeps_its_prefix_hook(run, monkeypatch):
    calls, snapshots = [], []

    class Key(str):
        def startswith(self, prefix):
            calls.append(prefix)
            return False

    key = Key("_custom")
    record = {"_step": 0, key: 3}
    assert history_metric_values(record) is None
    assert calls == []
    instance = run()
    monkeypatch.setattr(instance.summary, "observe", snapshots.append)
    instance._on_commit(record)
    assert calls == ["_"]
    assert snapshots == [{key: 3}]
    assert next(iter(snapshots[0])) is key


def test_surrogate_keys_keep_lossless_python_projection(run):
    record = {"_step": 0, "visible\ud800": 3, "_hidden\ud800": 4}
    assert history_metric_values(record) is None
    instance = run()
    instance._on_commit(record)
    assert dict(instance.summary) == {"visible\ud800": 3}


def test_prefix_hook_mutation_keeps_legacy_iteration_error(run):
    record = {"_step": 0}

    class Key(str):
        def startswith(self, prefix):
            record["added"] = 1
            return False

    record[Key("loss")] = 3
    instance = run()
    with pytest.raises(RuntimeError, match="dictionary changed size"):
        instance._on_commit(record)
    assert dict(instance.summary) == {}


def test_summary_and_backends_share_snapshot_but_alerts_keep_the_row(run, monkeypatch):
    instance = run()
    snapshots, alert_rows = [], []
    payload = {"nested": [1]}
    record = {"_step": 7, "_time": 1.0, "_hidden": 3, "payload": payload}

    def observe(metrics):
        snapshots.append(metrics)
        metrics["summary_hook"] = 1

    class Backend:
        def log(self, metrics, step=None):
            assert step == 7
            snapshots.append(metrics)
            metrics["backend_hook"] = 2

    class Alerts:
        def on_step(self, row):
            alert_rows.append(row)

        def close(self):
            pass

    monkeypatch.setattr(instance.summary, "observe", observe)
    instance.backends = {"first": Backend(), "second": Backend()}
    instance.alerts = Alerts()
    instance._on_commit(record)
    assert len(snapshots) == 3 and all(row is snapshots[0] for row in snapshots)
    assert snapshots[0] == {
        "payload": payload,
        "summary_hook": 1,
        "backend_hook": 2,
    }
    assert snapshots[0]["payload"] is payload
    assert alert_rows == [record] and alert_rows[0] is record
    assert record == {"_step": 7, "_time": 1.0, "_hidden": 3, "payload": payload}
