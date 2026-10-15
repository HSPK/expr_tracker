"""Trace spool cleanup, atomic export and diagnostic values."""

import json
from pathlib import Path

import pytest
from _trace_helpers import (
    lanes_are_stacks as lanes_are_stacks,
)
from _trace_helpers import (
    make as make,
)
from _trace_helpers import (
    observe_native_trace as observe_native_trace,
)
from _trace_helpers import (
    range_file as range_file,
)
from _trace_helpers import (
    recorded as recorded,
)
from _trace_helpers import (
    run as run,
)
from _trace_helpers import (
    runner as runner,
)
from _trace_helpers import (
    spans_in as spans_in,
)
from _trace_helpers import (
    write_records as write_records,
)

from expr_tracker.trace import (
    assign_lanes,
    build_trace,
    read_spans,
    write_trace,
)


def test_failed_export_preserves_destination_and_cleans_up(tmp_path, monkeypatch):
    output = tmp_path / "trace.json"
    output.write_text("original", encoding="utf-8")
    closed = []

    def fail(*args, **kwargs):
        try:
            raise OSError("disk unavailable")
        finally:
            closed.append(True)

    observe_native_trace(monkeypatch, fail)
    with pytest.raises(OSError, match="disk unavailable"):
        write_trace(tmp_path, output)
    assert output.read_text() == "original"
    assert list(tmp_path.iterdir()) == [output]
    assert closed == [True]


def test_sqlite_spool_is_cleaned_up_on_layout_failure(tmp_path, monkeypatch):
    source = write_records(tmp_path / "spans.jsonl", [make(0, 1)])
    temporary_paths = []

    def fail(phase, payload):
        if phase == "spool":
            temporary_paths.append(str(Path(payload).parent))
        if phase == "layout":
            raise OSError("layout failure")

    observe_native_trace(monkeypatch, fail)
    with pytest.raises(OSError, match="layout failure"):
        write_trace(source, tmp_path / "out.json")
    assert temporary_paths
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("stream_output", [False, True])
def test_trace_uses_and_cleans_the_requested_spool_directory(
    tmp_path, monkeypatch, stream_output
):
    source = write_records(tmp_path / "spans.jsonl", [make(0, 1), make(2, 1)])
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    sentinel = scratch / "keep"
    sentinel.write_text("unrelated")
    spools = []

    def stage(phase, payload):
        if phase != "spool":
            return
        spool = Path(payload)
        assert spool.parent.parent == scratch
        spools.append(spool)

    observe_native_trace(monkeypatch, stage)
    if stream_output:
        output = tmp_path / "out.json"
        assert write_trace(source, output, temp_dir=scratch) == 2
        trace = json.loads(output.read_text())
    else:
        trace = build_trace(source, temp_dir=str(scratch))
    assert len(spans_in(trace)) == 2
    assert spools and all(not path.exists() for path in spools)
    assert list(scratch.iterdir()) == [sentinel]


@pytest.mark.parametrize("kind", ["missing", "file", "sqlite_failure"])
def test_bad_spool_storage_preserves_output_and_cleans_up(tmp_path, monkeypatch, kind):
    import sqlite3

    source = write_records(tmp_path / "spans.jsonl", [make(0, 1)])
    output = tmp_path / "out.json"
    output.write_text("original")
    scratch = tmp_path / "scratch"
    if kind == "file":
        scratch.write_text("not a directory")
    elif kind == "sqlite_failure":
        scratch.mkdir()

        def fail(*args):
            raise sqlite3.OperationalError("database or disk is full")

        observe_native_trace(monkeypatch, fail)
    with pytest.raises((OSError, sqlite3.Error)):
        write_trace(source, output, temp_dir=scratch)
    assert output.read_text() == "original"
    assert not list(tmp_path.glob(".expr-trace-*.tmp"))
    if scratch.is_dir():
        assert not list(scratch.iterdir())


@pytest.mark.parametrize(
    ("value", "expected"),
    [(float("nan"), "NaN"), (float("inf"), "Infinity"), (float("-inf"), "-Infinity")],
)
def test_nonfinite_arguments_are_exported_as_diagnostic_strings(
    tmp_path, value, expected
):
    record = {
        **make(0, 1),
        "args": {"nested": [{"value": value}]},
        "metrics": {"gpu_percent": value},
    }
    source = write_records(tmp_path / "spans.jsonl", [record])
    original = read_spans(source)[0]["metrics"]["gpu_percent"]
    assert original == value or (original != original and value != value)
    trace = build_trace(source)
    args = spans_in(trace)[0]["args"]
    assert args["nested"] == [{"value": expected}]
    assert args["gpu_percent"] == expected

    output = tmp_path / "out.json"
    assert write_trace(source, output) == 1

    def reject_constant(token):
        raise ValueError(f"non-standard JSON constant: {token}")

    assert json.loads(output.read_text(), parse_constant=reject_constant) == trace


def test_overflowing_json_exponents_remain_diagnostic_strings(tmp_path):
    source = tmp_path / "spans.jsonl"
    source.write_bytes(
        b'{"start":0,"dur_ms":1,"args":{"nested":[1e999,-1e999]},'
        b'"metrics":{"value":1e999},"_step":1e999}\n'
    )
    trace = build_trace(source)
    args = spans_in(trace)[0]["args"]
    assert args["nested"] == ["Infinity", "-Infinity"]
    assert args["value"] == args["step"] == "Infinity"
    output = tmp_path / "out.json"
    assert write_trace(source, output) == 1
    assert json.loads(output.read_text()) == trace


def test_output_can_replace_input_only_after_it_has_been_read(tmp_path):
    source = write_records(tmp_path / "spans.jsonl", [make(0, 1)])
    expected = build_trace(source)
    assert write_trace(source, source) == 1
    assert json.loads(source.read_text()) == expected


def test_submicrosecond_spans_are_packed_using_exported_coordinates(tmp_path):
    records = [
        {**make(1_700_000_000, 0.0003), "depth": 0},
        {**make(1_700_000_000.0000002, 0.0001), "depth": 1},
    ]
    source = write_records(tmp_path / "spans.jsonl", records)
    trace = build_trace(source)
    assert lanes_are_stacks(trace)
    assert len(set(assign_lanes(records).values())) == 2
    assert len({event["tid"] for event in spans_in(trace)}) == 2


def test_unrepresentable_relative_timestamps_fail_without_invalid_json(tmp_path):
    source = write_records(tmp_path / "spans.jsonl", [make(-1e302, 1), make(1e302, 1)])
    with pytest.raises(ValueError, match="finite trace range"):
        write_trace(source, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()
