"""Chrome Trace export: lane layout, filtering and the CLI."""

import json
import random
import threading
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

import expr_tracker as et
from expr_tracker import cli
from expr_tracker import trace as trace_module
from expr_tracker.trace import (
    assign_lanes,
    build_trace,
    read_spans,
    span_files,
    write_trace,
)


@pytest.fixture
def run(tmp_path):
    created = []

    def factory(**options):
        created.append(
            et.init(
                project="tr",
                name=options.pop("name", "r"),
                dir=str(tmp_path),
                backends=[],
                **options,
            )
        )
        return created[-1]

    yield factory
    if et.get_run() is not None:
        et.finish()


@pytest.fixture
def recorded(tmp_path, run):
    """A run with nested spans on the default stream and one on "data"."""
    run()
    for _ in range(3):
        with et.span("forward"):
            with et.span("attention"):
                time.sleep(0.002)
            with et.span("mlp"):
                time.sleep(0.001)
        et.log({"loss": 1.0})
    et.finish()

    run(stream="data")
    for _ in range(2):
        with et.span("produce", rows=32):
            time.sleep(0.001)
        et.log({"rows": 32})
    et.finish()
    return tmp_path / "tr" / "r"


def spans_in(trace):
    return [e for e in trace["traceEvents"] if e["ph"] == "X"]


def lanes_are_stacks(trace) -> bool:
    """A Chrome Trace track must be properly nested to render at all."""
    tracks: dict[tuple, list] = {}
    for event in spans_in(trace):
        start, end = event["ts"], event["ts"] + event["dur"]
        tracks.setdefault((event["pid"], event["tid"]), []).append((start, end))
    for intervals in tracks.values():
        intervals.sort(key=lambda interval: (interval[0], -interval[1]))
        stack = []
        for start, end in intervals:
            while stack and stack[-1] <= start:
                stack.pop()
            if stack and end > stack[-1]:
                return False
            stack.append(end)
    return True


# ------------------------------------------------------------------ lanes


def make(start, dur_ms, track=0):
    return {"start": start, "dur_ms": dur_ms, "track": track}


def test_nested_spans_share_a_lane():
    spans = [make(0.0, 100), make(0.01, 50)]  # child inside parent
    assert set(assign_lanes(spans).values()) == {0}


def test_sequential_spans_share_a_lane():
    spans = [make(0.0, 10), make(0.02, 10), make(0.04, 10)]
    assert set(assign_lanes(spans).values()) == {0}


def test_partially_overlapping_spans_are_split():
    """Two events that cross cannot sit on one track."""
    spans = [make(0.0, 30), make(0.01, 30)]  # overlap without containment
    lanes = assign_lanes(spans)
    assert lanes[0] != lanes[1]


def test_different_threads_never_share_a_lane():
    """Unrelated threads must not look like one nesting inside the other."""
    spans = [make(0.0, 100, track=1), make(0.01, 10, track=2)]
    lanes = assign_lanes(spans)
    assert lanes[0] != lanes[1]


def test_a_deep_stack_stays_on_one_lane():
    spans = [make(0.0, 100), make(0.001, 80), make(0.002, 60), make(0.003, 40)]
    assert set(assign_lanes(spans).values()) == {0}


def test_lanes_of_an_empty_list():
    assert assign_lanes([]) == {}


def test_many_concurrent_spans_get_their_own_lanes():
    spans = [make(index * 0.001, 100) for index in range(5)]
    assert len(set(assign_lanes(spans).values())) == 5


# ------------------------------------------------------------------ building


def test_the_trace_has_one_process_per_stream(recorded):
    trace = build_trace(recorded)
    names = [
        e["args"]["name"] for e in trace["traceEvents"] if e["name"] == "process_name"
    ]
    assert names == ["default", "data"]  # the default producer comes first


def test_every_span_becomes_a_complete_event(recorded):
    events = spans_in(build_trace(recorded))
    assert len(events) == 3 * 3 + 2
    assert all(e["ph"] == "X" and e["dur"] > 0 for e in events)


def test_nested_names_are_preserved(recorded):
    names = {e["name"] for e in spans_in(build_trace(recorded))}
    assert {"forward", "forward/attention", "forward/mlp", "produce"} <= names


def test_timestamps_are_relative_to_the_first_span(recorded):
    events = spans_in(build_trace(recorded))
    assert min(e["ts"] for e in events) == 0.0
    assert all(e["ts"] >= 0 for e in events)


def test_streams_share_one_timeline(recorded):
    """Both producers are offset from the same origin, or they cannot be compared."""
    events = spans_in(build_trace(recorded))
    by_pid = {}
    for event in events:
        by_pid.setdefault(event["pid"], []).append(event["ts"])
    assert len(by_pid) == 2
    assert min(min(v) for v in by_pid.values()) == 0.0


def test_the_step_travels_with_the_span(recorded):
    events = spans_in(build_trace(recorded))
    steps = sorted({e["args"]["step"] for e in events if e["cat"] == "default"})
    assert steps == [0, 1, 2]


def test_attributes_are_carried_through(recorded):
    events = spans_in(build_trace(recorded))
    produce = next(e for e in events if e["name"] == "produce")
    assert produce["args"]["rows"] == 32


def test_the_category_is_the_stream(recorded):
    events = spans_in(build_trace(recorded))
    assert {e["cat"] for e in events} == {"default", "data"}


def test_lanes_are_valid_stacks(recorded):
    assert lanes_are_stacks(build_trace(recorded))


def test_the_trace_declares_its_time_unit(recorded):
    assert build_trace(recorded)["displayTimeUnit"] == "ms"


def test_an_error_is_carried_through(run, tmp_path):
    run()
    with pytest.raises(ValueError), et.span("risky"):
        raise ValueError("boom")
    et.log({"loss": 1.0})
    et.finish()
    events = spans_in(build_trace(tmp_path / "tr" / "r"))
    assert events[0]["args"]["error"] == "ValueError"


def test_concurrent_threads_land_on_separate_lanes(run, tmp_path):
    run()
    barrier = threading.Barrier(3)

    def worker(name):
        barrier.wait()
        with et.span(name):
            time.sleep(0.02)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    with et.span("main"):
        time.sleep(0.02)
    for thread in threads:
        thread.join(timeout=10)
    et.log({"loss": 1.0})
    et.finish()

    trace = build_trace(tmp_path / "tr" / "r")
    events = spans_in(trace)
    assert len({e["tid"] for e in events}) == 3  # one lane per thread
    assert lanes_are_stacks(trace)


def test_lanes_are_labelled_by_thread_when_there_are_several(run, tmp_path):
    run()

    def worker():
        with et.span("other"):
            time.sleep(0.005)

    thread = threading.Thread(target=worker)
    thread.start()
    with et.span("main"):
        time.sleep(0.005)
    thread.join(timeout=10)
    et.log({"loss": 1.0})
    et.finish()

    trace = build_trace(tmp_path / "tr" / "r")
    labels = [
        e["args"]["name"] for e in trace["traceEvents"] if e["name"] == "thread_name"
    ]
    assert len(labels) == 2 and all(x.startswith("thread ") for x in labels)


# ------------------------------------------------------------------ selection


def test_a_single_stream_can_be_exported(recorded):
    events = spans_in(build_trace(recorded, ["data"]))
    assert {e["name"] for e in events} == {"produce"}


def test_the_default_stream_is_selected_with_none(recorded):
    events = spans_in(build_trace(recorded, [None]))
    assert "produce" not in {e["name"] for e in events}


def test_several_streams_can_be_named(recorded):
    events = spans_in(build_trace(recorded, [None, "data"]))
    assert {e["cat"] for e in events} == {"default", "data"}


def test_an_unknown_stream_is_an_error(recorded):
    with pytest.raises(FileNotFoundError, match=r"spans\.nope\.jsonl"):
        build_trace(recorded, ["nope"])


@pytest.mark.parametrize(
    ("bounds", "expected"), [((0, 1), [0]), ((1, 3), [1, 2]), ((None, 2), [0, 1])]
)
def test_a_step_range_filters_spans(recorded, bounds, expected):
    events = spans_in(build_trace(recorded, [None], step_range=bounds))
    assert sorted({e["args"]["step"] for e in events}) == expected


def test_a_step_range_that_matches_nothing(recorded):
    assert spans_in(build_trace(recorded, [None], step_range=(99, 100))) == []


def test_a_span_file_can_be_given_directly(recorded):
    events = spans_in(build_trace(recorded / "spans.data.jsonl"))
    assert {e["name"] for e in events} == {"produce"}


# ------------------------------------------------------------------ robustness


def test_a_run_without_spans_gives_an_empty_trace(run, tmp_path):
    run(spans=False)
    et.log({"loss": 1.0})
    et.finish()
    trace = build_trace(tmp_path / "tr" / "r")
    assert spans_in(trace) == []
    assert trace["traceEvents"] == [] or all(
        e["ph"] == "M" for e in trace["traceEvents"]
    )


def test_corrupt_lines_are_skipped(recorded):
    path = recorded / "spans.jsonl"
    path.write_text(path.read_text() + "not json\n" + json.dumps({"no": "span"}) + "\n")
    assert len(read_spans(path)) == 9  # the malformed lines are gone


def test_reading_a_missing_file_is_empty(tmp_path):
    assert read_spans(tmp_path / "nope.jsonl") == []


def test_span_files_lists_every_stream(recorded):
    names = [p.name for p in span_files(recorded)]
    assert names == ["spans.jsonl", "spans.data.jsonl"]


# ------------------------------------------------------------------ writing


def test_write_trace_reports_the_span_count(recorded, tmp_path):
    output = tmp_path / "out" / "trace.json"
    assert write_trace(recorded, output) == 11
    assert output.is_file()
    assert json.loads(output.read_text())["displayTimeUnit"] == "ms"


def test_write_trace_creates_the_parent_directory(recorded, tmp_path):
    output = tmp_path / "deep" / "nested" / "trace.json"
    write_trace(recorded, output)
    assert output.is_file()


def test_the_written_trace_is_valid_json(recorded, tmp_path):
    output = tmp_path / "trace.json"
    write_trace(recorded, output)
    trace = json.loads(output.read_text())
    assert lanes_are_stacks(trace)
    assert all("ts" in e and "dur" in e for e in spans_in(trace))


# ------------------------------------------------------------------ cli


@pytest.fixture
def runner():
    return CliRunner()


def test_the_cli_writes_a_trace(runner, recorded, tmp_path):
    output = tmp_path / "cli.json"
    result = runner.invoke(cli.main, ["trace", str(recorded), "-o", str(output)])
    assert result.exit_code == 0, result.output
    assert "wrote 11 span(s)" in result.output
    assert output.is_file()


def test_the_cli_defaults_to_trace_json(runner, recorded, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.main, ["trace", str(recorded)])
    assert result.exit_code == 0
    assert (tmp_path / "trace.json").is_file()


def test_the_cli_can_select_a_stream(runner, recorded, tmp_path):
    output = tmp_path / "one.json"
    result = runner.invoke(
        cli.main, ["trace", str(recorded), "-o", str(output), "--stream", "data"]
    )
    assert result.exit_code == 0 and "wrote 2 span(s)" in result.output


def test_the_cli_names_the_default_stream(runner, recorded, tmp_path):
    output = tmp_path / "default.json"
    result = runner.invoke(
        cli.main, ["trace", str(recorded), "-o", str(output), "--stream", "default"]
    )
    assert result.exit_code == 0 and "wrote 9 span(s)" in result.output


def test_the_cli_accepts_a_step_range(runner, recorded, tmp_path):
    output = tmp_path / "range.json"
    result = runner.invoke(
        cli.main,
        [
            "trace",
            str(recorded),
            "-o",
            str(output),
            "--step-range",
            "0:1",
            "--stream",
            "default",
        ],
    )
    assert result.exit_code == 0 and "wrote 3 span(s)" in result.output


def test_the_cli_rejects_a_missing_run(runner, tmp_path):
    result = runner.invoke(cli.main, ["trace", str(tmp_path / "nope")])
    assert result.exit_code != 0


def test_the_cli_rejects_an_unknown_stream(runner, recorded, tmp_path):
    result = runner.invoke(
        cli.main,
        ["trace", str(recorded), "--stream", "nope", "-o", str(tmp_path / "x")],
    )
    assert result.exit_code != 0


def test_the_cli_is_listed_in_help(runner):
    result = runner.invoke(cli.main, ["--help"])
    assert result.exit_code == 0 and "trace" in result.output


def test_blank_lines_are_skipped(recorded):
    path = recorded / "spans.jsonl"
    path.write_text(path.read_text() + "\n   \n\n")
    assert len(read_spans(path)) == 9


def write_records(path, records):
    with path.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record) + "\n")
    return path


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


@pytest.fixture
def range_file(tmp_path):
    records = [
        ("ends-at-low", 8, 2000),
        ("crosses-low", 9, 2000),
        ("encloses-range", 5, 20000),
        ("inside", 10, 1000),
        ("crosses-high", 19, 2000),
        ("starts-at-high", 20, 1000),
        ("empty", 15, 0),
    ]
    return write_records(
        tmp_path / "spans.jsonl",
        [
            {"name": name, **make(start, duration), "_step": 0}
            for name, start, duration in records
        ],
    )


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


def test_failed_export_preserves_destination_and_cleans_up(tmp_path, monkeypatch):
    output = tmp_path / "trace.json"
    output.write_text("original", encoding="utf-8")
    closed = []

    def fail(*args, **kwargs):
        try:
            yield {"ph": "X"}
            raise OSError("disk unavailable")
        finally:
            closed.append(True)

    monkeypatch.setattr(trace_module, "_trace_events", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        write_trace(tmp_path, output)
    assert output.read_text() == "original"
    assert list(tmp_path.iterdir()) == [output]
    assert closed == [True]


def test_sqlite_spool_is_cleaned_up_on_layout_failure(tmp_path, monkeypatch):
    source = write_records(tmp_path / "spans.jsonl", [make(0, 1)])
    original = trace_module.TemporaryDirectory
    temporary_paths = []

    def temporary(**kwargs):
        if kwargs.get("dir") is None:
            kwargs["dir"] = tmp_path
        context = original(**kwargs)
        temporary_paths.append(context.name)
        return context

    def fail(*args):
        raise OSError("layout failure")

    monkeypatch.setattr(trace_module, "TemporaryDirectory", temporary)
    monkeypatch.setattr(trace_module._LanePool, "place", fail)
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
    original = trace_module._stage_spans
    spools = []

    def stage(db, *args):
        spool = Path(db.execute("PRAGMA database_list").fetchone()[2])
        assert spool.parent.parent == scratch
        spools.append(spool)
        return original(db, *args)

    monkeypatch.setattr(trace_module, "_stage_spans", stage)
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

        monkeypatch.setattr(trace_module, "_stage_spans", fail)
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
def test_nonfinite_arguments_are_exported_as_diagnostic_strings(tmp_path, value, expected):
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
