"""The shipped examples must actually run, and mean what they claim to mean."""

import importlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def pipeline():
    # On sys.path rather than loaded by location: spawn hands its own sys.path
    # to each child, which then has to import the workers by module name.
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    return importlib.import_module("multiprocess_pipeline")


def run_pipeline(module, tmp_path, **overrides):
    argv = {
        "--producers": "2",
        "--trainers": "2",
        "--steps": "3",
        "--produce-ms": "2",
        "--train-ms": "2",
        "--jitter": "0",
        "--dir": str(tmp_path),
        "--name": "run",
    }
    argv.update({k: str(v) for k, v in overrides.items()})
    flat = [item for pair in argv.items() for item in pair]
    return module.main(flat), tmp_path / "pipeline" / "run"


def test_every_worker_gets_its_own_stream(pipeline, tmp_path):
    from expr_tracker.history import list_streams

    _, run_dir = run_pipeline(pipeline, tmp_path)
    assert sorted(list_streams(run_dir)) == [
        "producer0",
        "producer1",
        "trainer0",
        "trainer1",
    ]


def test_the_trace_has_one_lane_per_worker(pipeline, tmp_path):
    output, _ = run_pipeline(pipeline, tmp_path)
    events = json.loads(output.read_text())["traceEvents"]
    names = {e["args"]["name"] for e in events if e.get("name") == "process_name"}
    assert names == {"producer0", "producer1", "trainer0", "trainer1"}


def test_the_trace_holds_the_whole_span_tree(pipeline, tmp_path):
    output, _ = run_pipeline(pipeline, tmp_path)
    events = json.loads(output.read_text())["traceEvents"]
    spans = {e["name"] for e in events if e["ph"] == "X"}
    assert spans == {
        "produce",
        "produce/read",
        "produce/decode",
        "produce/enqueue",
        "step",
        "step/wait_for_batch",
        "step/forward",
        "step/backward",
    }
    assert min(e["ts"] for e in events if e["ph"] == "X") == 0.0  # rebased


def test_the_trainers_consume_exactly_what_they_asked_for(pipeline, tmp_path):
    from expr_tracker.history import read_history

    _, run_dir = run_pipeline(pipeline, tmp_path, **{"--steps": 4})
    batches = []
    for worker in ("trainer0", "trainer1"):
        rows = read_history(run_dir, -1, stream=worker)
        assert len(rows) == 4
        batches += [row["train/batch"] for row in rows]
    assert sorted(batches) == list(range(8))  # every batch exactly once


def test_the_producers_stop_at_what_was_asked_for(pipeline, tmp_path):
    from expr_tracker.history import read_history

    _, run_dir = run_pipeline(pipeline, tmp_path, **{"--steps": 4})
    made = sum(
        read_history(run_dir, -1, stream=worker)[-1]["produce/made"]
        for worker in ("producer0", "producer1")
    )
    assert made == 8  # 4 steps x 2 trainers, no overrun


def test_slow_trainers_show_up_as_backpressure(pipeline, tmp_path):
    """A full queue must block the producers inside the enqueue span."""
    _, run_dir = run_pipeline(
        pipeline,
        tmp_path,
        **{"--produce-ms": 1, "--train-ms": 40, "--steps": 4, "--staleness": 1},
    )
    stages = stage_totals(pipeline, run_dir)
    assert stages["enqueue"] > stages["wait_for_batch"]
    assert stages["enqueue"] > stages["read"]


def test_slow_producers_show_up_as_starvation(pipeline, tmp_path):
    _, run_dir = run_pipeline(
        pipeline, tmp_path, **{"--produce-ms": 40, "--train-ms": 1, "--steps": 4}
    )
    stages = stage_totals(pipeline, run_dir)
    assert stages["wait_for_batch"] > stages["enqueue"]
    assert stages["wait_for_batch"] > stages["forward"] + stages["backward"]


def stage_totals(module, run_dir):
    from expr_tracker.trace import read_spans, span_files

    totals = {}
    for path in span_files(run_dir):
        for record in read_spans(path):
            leaf = record["name"].rsplit("/", 1)[-1]
            totals[leaf] = totals.get(leaf, 0.0) + float(record["dur_ms"])
    return totals


def test_staleness_bounds_how_far_ahead_producers_get(pipeline, tmp_path):
    """With one slot per trainer, no batch can be produced far ahead of use."""
    from expr_tracker.history import read_history

    _, run_dir = run_pipeline(
        pipeline,
        tmp_path,
        **{"--produce-ms": 1, "--train-ms": 20, "--steps": 5, "--staleness": 1},
    )
    ages = [
        row["train/queue_age_ms"]
        for worker in ("trainer0", "trainer1")
        for row in read_history(run_dir, -1, stream=worker)
    ]
    # One slot per trainer, so a batch is consumed within a step or two of
    # being made. Unbounded, the producers would finish first and the last
    # batch would sit through the whole run.
    assert max(ages) < 20 * 3


def test_the_summary_reports_both_sides(pipeline, tmp_path, capsys):
    _, run_dir = run_pipeline(pipeline, tmp_path)
    capsys.readouterr()
    pipeline.summarise(run_dir)
    out = capsys.readouterr().out
    assert "producers" in out and "trainers" in out
    assert "backpressure" in out and "starvation" in out


class DeadProcess:
    """A worker that starts, joins and turns out to have failed."""

    def __init__(self, target=None, args=(), daemon=None):
        self.name = getattr(target, "__name__", "worker")
        self.exitcode = 1

    def start(self):
        pass

    def join(self):
        pass


def test_a_worker_failure_is_not_swallowed(pipeline, tmp_path, monkeypatch):
    """A silent partial run would be worse than a loud failure."""
    real = mp.get_context("spawn")
    fake = type(
        "Ctx", (), {"Queue": real.Queue, "Value": real.Value, "Process": DeadProcess}
    )()
    monkeypatch.setattr(pipeline.mp, "get_context", lambda method: fake)
    with pytest.raises(SystemExit, match="workers failed"):
        run_pipeline(pipeline, tmp_path, **{"--steps": 1})
