"""Trace filtering, snapshot reads and interval layout."""

import json
import random

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

from expr_tracker import trace as trace_module
from expr_tracker.trace import (
    assign_lanes,
    build_trace,
    read_spans,
    write_trace,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start", None),
        ("start", True),
        ("start", []),
        ("start", "bad"),
        ("start", float("nan")),
        ("start", float("inf")),
        ("start", "1e999"),
        ("start", 10**400),
        ("start", 1e308),
        ("dur_ms", -1),
        ("dur_ms", False),
        ("dur_ms", {}),
        ("dur_ms", "-inf"),
        ("dur_ms", 1e308),
        ("args", []),
        ("metrics", "bad"),
        ("track", []),
        ("track", None),
        ("track", float("nan")),
        ("depth", -1),
        ("depth", "1"),
        ("depth", True),
    ],
)
def test_malformed_span_fields_are_skipped(tmp_path, field, value):
    invalid = {**make(1.0, 1), field: value}
    valid = make(2.0, 1)
    source = write_records(tmp_path / "spans.jsonl", [invalid, valid])
    assert read_spans(source) == [valid]
    assert len(spans_in(build_trace(source))) == 1


def test_invalid_utf8_and_truncated_json_do_not_hide_later_records(tmp_path):
    source = tmp_path / "spans.jsonl"
    source.write_bytes(
        b'\xff\n{"start":\n[]\n{}\n\n{"start": 1, "dur_ms": 2}\n{"start":'
    )
    assert read_spans(source) == [{"start": 1, "dur_ms": 2}]


def test_reader_never_reads_the_entire_file(tmp_path, monkeypatch):
    source = write_records(tmp_path / "spans.jsonl", [make(0, 1), make(1, 1)])

    def forbidden(*args, **kwargs):
        pytest.fail("span input must be parsed one line at a time")

    monkeypatch.setattr(type(source), "read_text", forbidden)
    assert len(read_spans(source)) == 2
    assert len(spans_in(build_trace(source, time_range=(1, 2)))) == 1


def test_numeric_strings_sort_numerically_and_preserve_reader_records(tmp_path):
    records = [
        {"name": "late", "start": "10", "dur_ms": "20000"},
        {"name": "early", "start": "2", "dur_ms": "30000"},
    ]
    source = write_records(tmp_path / "spans.jsonl", records)
    assert read_spans(source) == records
    trace = build_trace(source)
    assert {event["tid"] for event in spans_in(trace)} == {0}
    assert [event["ts"] for event in spans_in(trace)] == [8_000_000, 0]
    assert assign_lanes(records) == {1: 0, 0: 0}


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((10, 20), {"crosses-low", "encloses-range", "inside", "crosses-high"}),
        ((None, 10), {"ends-at-low", "crosses-low", "encloses-range"}),
        ((20, None), {"encloses-range", "crosses-high", "starts-at-high"}),
        (
            (None, None),
            {
                "ends-at-low",
                "crosses-low",
                "encloses-range",
                "inside",
                "crosses-high",
                "starts-at-high",
            },
        ),
        ((10, 10), set()),
        ((20, 10), set()),
        ((30, 40), set()),
    ],
)
def test_time_ranges_select_overlaps_without_clipping(
    range_file, tmp_path, bounds, expected
):
    records = read_spans(range_file, time_range=bounds)
    assert {record["name"] for record in records} == expected
    trace = build_trace(range_file, time_range=bounds, step_range=(0, 1))
    events = spans_in(trace)
    assert {event["name"] for event in events} == expected
    original = {record["name"]: record for record in read_spans(range_file)}
    for event in events:
        assert event["dur"] == original[event["name"]]["dur_ms"] * 1000
    output = tmp_path / "out.json"
    assert write_trace(range_file, output, time_range=bounds, step_range=(0, 1)) == len(
        expected
    )
    assert json.loads(output.read_text()) == trace


def test_no_time_filter_keeps_zero_duration_spans(range_file):
    assert "empty" in {record["name"] for record in read_spans(range_file)}
    assert "empty" in {event["name"] for event in spans_in(build_trace(range_file))}


def test_step_and_time_filters_both_must_match(range_file):
    assert (
        spans_in(build_trace(range_file, time_range=(10, 20), step_range=(1, 2))) == []
    )


@pytest.mark.parametrize("step", [None, [], {}, "no", float("inf"), True])
def test_step_filter_skips_invalid_step_values(tmp_path, step):
    source = write_records(tmp_path / "spans.jsonl", [{**make(1, 1), "_step": step}])
    assert read_spans(source, step_range=(0, 1)) == []
    assert spans_in(build_trace(source, step_range=(0, 1))) == []


def test_step_filter_preserves_large_integer_precision(tmp_path):
    step = 2**53 + 1
    record = {**make(0, 1), "_step": step}
    source = write_records(tmp_path / "spans.jsonl", [record])
    assert read_spans(source, step_range=(step, step + 1)) == [record]
    assert read_spans(source, step_range=(step - 1, step)) == []


@pytest.mark.parametrize(
    "bounds",
    [(float("nan"), None), (None, float("inf")), ("1", 2), (True, 2), (10**400, None)],
)
def test_invalid_time_bounds_fail_even_without_input(tmp_path, bounds):
    for call in (
        lambda: read_spans(tmp_path / "missing", time_range=bounds),
        lambda: build_trace(tmp_path, time_range=bounds),
        lambda: write_trace(tmp_path, tmp_path / "out.json", time_range=bounds),
    ):
        with pytest.raises(ValueError, match="time_range"):
            call()
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("bounds", [(2, 2), (3, 2)])
def test_empty_step_range_keeps_only_process_metadata(range_file, bounds):
    trace = build_trace(range_file, step_range=bounds)
    assert [event["name"] for event in trace["traceEvents"]] == ["process_name"]


def test_later_stream_can_define_global_origin_and_input_order_is_preserved(tmp_path):
    write_records(
        tmp_path / "spans.jsonl",
        [
            {"name": "late", **make(150, 1000, track=2)},
            {"name": "middle", **make(120, 1000, track=1)},
        ],
    )
    write_records(
        tmp_path / "spans.data.jsonl",
        [
            {"name": "first", **make(100, 30000)},
        ],
    )
    write_records(tmp_path / "spans.empty.jsonl", [make(1, 1)])
    trace = build_trace(tmp_path, time_range=(110, 160))
    events = spans_in(trace)
    assert [event["name"] for event in events] == ["late", "middle", "first"]
    assert [event["ts"] for event in events] == [50_000_000, 20_000_000, 0]
    assert [
        event["args"]["name"]
        for event in trace["traceEvents"]
        if event["name"] == "process_name"
    ] == ["default", "data", "empty"]
    output = tmp_path / "out.json"
    assert write_trace(tmp_path, output, time_range=(110, 160)) == 3
    assert json.loads(output.read_text()) == trace


def test_known_top_level_spans_do_not_become_temporal_children():
    records = [{**make(0, 100), "depth": 0}, {**make(0.01, 10), "depth": 0}]
    assert len(set(assign_lanes(records).values())) == 2


def test_siblings_do_not_nest_and_real_children_still_do():
    records = [
        {**make(0, 100), "depth": 0},
        {**make(0.01, 80), "depth": 1},
        {**make(0.02, 10), "depth": 1},
        {**make(0.021, 1), "depth": 2},
    ]
    lanes = assign_lanes(records)
    assert lanes[0] == lanes[1]
    assert lanes[1] != lanes[2]
    assert lanes[3] in (lanes[1], lanes[2])


def test_equal_intervals_use_depth_to_put_parent_before_child(tmp_path):
    records = [{**make(0, 100), "depth": depth} for depth in (2, 1, 0)]
    assert set(assign_lanes(records).values()) == {0}
    source = write_records(tmp_path / "spans.jsonl", records)
    assert {event["tid"] for event in spans_in(build_trace(source))} == {0}


def test_expiring_child_exposes_parent_and_reuses_lanes():
    records = [
        {**make(0, 100), "depth": 0},
        {**make(0.01, 10), "depth": 1},
        {**make(0.02, 20), "depth": 1},
        {**make(0.1, 10), "depth": 0},
    ]
    assert set(assign_lanes(records).values()) == {0}


def test_packing_random_intervals_has_valid_stacks_and_matches_export(tmp_path):
    rng = random.Random(12)
    records = [
        {
            "name": str(index),
            **make(rng.randrange(100) / 10, rng.randrange(1000), rng.randrange(3)),
            "depth": rng.choice([None, 0, 1, 2, 3]),
        }
        for index in range(1000)
    ]
    source = write_records(tmp_path / "spans.jsonl", records)
    trace = build_trace(source)
    assert lanes_are_stacks(trace)
    expected = assign_lanes(records)
    assert {int(event["name"]): event["tid"] for event in spans_in(trace)} == expected
    output = tmp_path / "out.json"
    assert write_trace(source, output) == len(records)
    assert json.loads(output.read_text()) == trace


def test_write_trace_does_not_materialize_build_trace(tmp_path, monkeypatch):
    source = write_records(tmp_path / "spans.jsonl", [make(0, 1)])

    def forbidden(*args, **kwargs):
        pytest.fail("write_trace must not build an in-memory trace")

    monkeypatch.setattr(trace_module, "build_trace", forbidden)
    monkeypatch.setattr(trace_module, "read_spans", forbidden)
    assert write_trace(source, tmp_path / "out.json") == 1


def test_trace_query_and_export_include_original_and_parts(tmp_path):
    source = write_records(tmp_path / "spans.jsonl", [{"name": "first", **make(1, 1)}])
    parts = source.with_name(source.name + ".parts")
    parts.mkdir()
    write_records(parts / "00000002.jsonl", [{"name": "third", **make(3, 1)}])
    write_records(parts / "00000001.jsonl", [{"name": "second", **make(2, 1)}])
    assert [span["name"] for span in read_spans(source)] == ["first", "second", "third"]
    trace = build_trace(source)
    assert [span["name"] for span in spans_in(trace)] == ["first", "second", "third"]
    output = tmp_path / "out.json"
    assert write_trace(source, output) == 3
    assert json.loads(output.read_text()) == trace
