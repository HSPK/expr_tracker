"""Chrome Trace export: lane layout, filtering and the CLI."""

import json
import threading
import time

import pytest
from click.testing import CliRunner

import expr_tracker as et
from expr_tracker import cli
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
        options.setdefault("max_open_seconds", None)
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
        intervals.sort()
        for index, (_start, end) in enumerate(intervals):
            for other_start, other_end in intervals[index + 1 :]:
                if other_start < end < other_end:  # partial overlap
                    return False
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
