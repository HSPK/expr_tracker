"""The shipped examples must actually run, and mean what they claim to mean."""

import importlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def example(name: str):
    # On sys.path rather than loaded by location: spawn hands its own sys.path
    # to each child, which then has to import the workers by module name.
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    return importlib.import_module(name)


@pytest.fixture(autouse=True)
def no_leftover_run():
    """An example that failed to finish would break the next one."""
    yield
    import expr_tracker as et

    if et.get_run() is not None:  # pragma: no cover - only on a broken example
        et.finish()


@pytest.fixture(scope="module")
def pipeline():
    return example("multiprocess_pipeline")


@pytest.fixture(scope="module")
def quickstart():
    return example("quickstart")


@pytest.fixture(scope="module")
def alert_rules():
    return example("alert_rules")


@pytest.fixture(scope="module")
def profile_step():
    return example("profile_step")


@pytest.fixture(scope="module")
def early_stopping():
    return example("early_stopping")


@pytest.fixture(scope="module")
def checkpoints():
    return example("checkpoints")


def argv(**options):
    return [item for pair in options.items() for item in (pair[0], str(pair[1]))]


def test_every_example_is_listed_in_the_readme():
    """A shipped example nobody can find is not much use."""
    readme = (EXAMPLES.parent / "README.md").read_text(encoding="utf-8")
    for path in sorted(EXAMPLES.glob("*.py")):
        assert path.name in readme, f"{path.name} is missing from the README"


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


# ------------------------------------------------------------------ quickstart


def test_quickstart_keeps_one_row_per_training_step(quickstart, tmp_path):
    """The eval log uses commit=False, so it must not create steps of its own."""
    from expr_tracker.history import read_history

    run_dir = quickstart.main(argv(**{"--steps": 40, "--dir": tmp_path}))
    rows = read_history(run_dir, -1)
    assert len(rows) == 40
    assert [row["_step"] for row in rows] == list(range(40))


def test_quickstart_merges_eval_into_the_training_row(quickstart, tmp_path):
    from expr_tracker.history import read_history

    run_dir = quickstart.main(argv(**{"--steps": 40, "--dir": tmp_path}))
    with_eval = [r for r in read_history(run_dir, -1) if "eval/accuracy" in r]
    assert [row["_step"] for row in with_eval] == [0, 10, 20, 30]
    assert all("train/loss" in row for row in with_eval)  # same row, not its own


def test_quickstart_records_the_best_accuracy(quickstart, tmp_path):
    import json

    run_dir = quickstart.main(argv(**{"--steps": 40, "--dir": tmp_path}))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["best_step"] in (0, 10, 20, 30)
    assert 0 < summary["best_accuracy"] <= 1


# ------------------------------------------------------------------ alert_rules


def test_a_healthy_run_raises_nothing(alert_rules, tmp_path):
    """Warm-up must not produce false alarms; this is the whole claim."""
    fired = alert_rules.main(argv(**{"--fault": "none", "--dir": tmp_path}))
    assert fired == []


def test_a_loss_spike_is_caught(alert_rules, tmp_path):
    fired = alert_rules.main(
        argv(**{"--fault": "spike", "--at": 50, "--dir": tmp_path})
    )
    assert [m.level.value for m in fired] == ["error"]
    assert "spike" in fired[0].text


def test_a_non_finite_loss_is_critical_and_nothing_else(alert_rules, tmp_path):
    """zscore of NaN is UNKNOWN, so only the isnan rule may fire."""
    fired = alert_rules.main(argv(**{"--fault": "nan", "--dir": tmp_path}))
    assert [m.level.value for m in fired] == ["critical"]


def test_a_stalled_loss_is_caught(alert_rules, tmp_path):
    fired = alert_rules.main(argv(**{"--fault": "stall", "--dir": tmp_path}))
    assert {m.level.value for m in fired} == {"warning"}
    assert any("flat" in m.text for m in fired)


def test_for_steps_delays_the_accuracy_rule(alert_rules, tmp_path):
    """The rule needs three consecutive steps, so it fires at 50 + 3."""
    fired = alert_rules.main(
        argv(**{"--fault": "stall", "--at": 50, "--dir": tmp_path})
    )
    regression = [m for m in fired if "accuracy_regression" in m.title]
    assert len(regression) == 1
    assert "step 53" in regression[0].text


# ------------------------------------------------------------------ profile_step


def test_profiling_records_the_span_tree_as_metrics(profile_step, tmp_path):
    from expr_tracker.history import read_history

    output = profile_step.main(argv(**{"--steps": 5, "--dir": tmp_path}))
    row = read_history(output.parent, 1)[0]
    for key in (
        "step/duration_ms",
        "step/data/read/duration_ms",
        "step/data/collate/duration_ms",
        "step/forward/duration_ms",
        "step/backward/duration_ms",
    ):
        assert key in row, key
    assert row["step/duration_ms"] >= row["step/forward/duration_ms"]


def test_profiling_attaches_the_cpu_plugin_to_every_span(profile_step, tmp_path):
    from expr_tracker.history import read_history

    output = profile_step.main(argv(**{"--steps": 5, "--dir": tmp_path}))
    row = read_history(output.parent, 1)[0]
    assert row["step/forward/cpu_percent"] > 50  # a spin loop
    assert row["step/data/read/cpu_percent"] < 50  # a sleep


def test_profiling_writes_a_loadable_trace(profile_step, tmp_path):
    output = profile_step.main(argv(**{"--steps": 4, "--dir": tmp_path}))
    events = json.loads(output.read_text())["traceEvents"]
    spans = [e for e in events if e["ph"] == "X"]
    assert len(spans) == 4 * 6  # step, data, read, collate, forward, backward
    assert {e["name"] for e in spans} >= {"step", "step/data/read"}


def test_printing_spans_is_opt_in(profile_step, tmp_path, capsys):
    profile_step.main(argv(**{"--steps": 2, "--dir": tmp_path}))
    assert "-> step" not in capsys.readouterr().out
    profile_step.main([*argv(**{"--steps": 2, "--dir": tmp_path}), "--print-spans"])
    printed = capsys.readouterr().out
    assert "-> step" in printed and "\t-> data" in printed


# ------------------------------------------------------------------ early_stopping


def test_the_loop_decays_then_stops(early_stopping, tmp_path):
    summary = early_stopping.main(argv(**{"--dir": tmp_path}))
    assert summary["decays"] == 2
    assert summary["stopped_early"] is True
    assert summary["steps_run"] < 600


def test_a_short_run_never_reaches_a_plateau(early_stopping, tmp_path):
    summary = early_stopping.main(argv(**{"--steps": 60, "--dir": tmp_path}))
    assert summary["decays"] == 0
    assert "stopped_early" not in summary


def test_patience_controls_how_soon_it_acts(early_stopping, tmp_path):
    impatient = early_stopping.main(argv(**{"--patience": 2, "--dir": tmp_path}))
    patient = early_stopping.main(argv(**{"--patience": 8, "--dir": tmp_path}))
    assert impatient["steps_run"] <= patient["steps_run"]


# ------------------------------------------------------------------ checkpoints


def test_checkpoints_version_and_deduplicate(checkpoints, tmp_path, capsys):
    checkpoints.main(argv(**{"--steps": 40, "--every": 10, "--dir": tmp_path}))
    out = capsys.readouterr().out
    assert "model:v0" in out and "model:v3" in out
    assert "v3 (deduplicated)" in out  # the same bytes made no new version


def test_the_best_checkpoint_is_restored_by_alias(checkpoints, tmp_path, capsys):
    checkpoints.main(argv(**{"--steps": 40, "--every": 10, "--dir": tmp_path}))
    out = capsys.readouterr().out
    assert "restored model:best (v3) -> weights@30" in out
