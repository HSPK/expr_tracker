"""Shared fixtures and helpers for the corresponding split regression suite."""

import json
import time

import pytest
from click.testing import CliRunner

import expr_tracker as et
from expr_tracker import trace as trace_module


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


def make(start, dur_ms, track=0):
    return {"start": start, "dur_ms": dur_ms, "track": track}


@pytest.fixture
def runner():
    return CliRunner()


def write_records(path, records):
    with path.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record) + "\n")
    return path


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


def observe_native_trace(monkeypatch, observer):
    """Inject faults at the native lifecycle boundary, not removed Python loops."""
    for name in ("trace_build", "trace_write"):
        native = getattr(trace_module, name)

        def wrap(*args, _native=native, **kwargs):
            return _native(*args, **kwargs, observer=observer)

        monkeypatch.setattr(trace_module, name, wrap)
