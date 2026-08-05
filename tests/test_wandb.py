"""The wandb backend, exercised against the real package in offline mode.

Offline needs no API key and no network, so these run anywhere. The assertions use
wandb's own live run state (``wandb.run.step`` / ``.config`` / ``.summary``), which
is what a user would see in the UI.
"""

import json
from pathlib import Path

import pytest

import expr_tracker as et
from expr_tracker.run import Run, _accepts_commit, get_backend

wandb = pytest.importorskip("wandb")


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    """Keep wandb entirely local: no key, no network, no shared state."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wb"))
    monkeypatch.setenv("WANDB_CONSOLE", "off")
    (tmp_path / "wb").mkdir(parents=True, exist_ok=True)
    yield
    if wandb.run is not None:  # a failed assertion must not leak the singleton
        wandb.finish()


@pytest.fixture
def run(tmp_path):
    created = []

    def factory(**options):
        options.setdefault("project", "wbtest")
        options.setdefault("name", "r")
        options.setdefault("dir", str(tmp_path / "et"))
        options.setdefault("backends", ["wandb"])
        options.setdefault("max_open_seconds", None)
        created.append(Run(**options))
        return created[-1]

    yield factory
    for instance in created:
        instance.finish()


def offline_run_dir(tmp_path) -> Path:
    """``dir=`` is forwarded to wandb, so its files live beside our run."""
    matches = sorted((tmp_path / "et" / "wandb").glob("offline-run-*"))
    assert matches, "wandb did not create an offline run directory"
    return matches[-1]


# ------------------------------------------------------------------ contract


def test_the_real_wandb_matches_our_assumptions():
    import inspect

    parameters = inspect.signature(wandb.init).parameters
    for name in ("project", "name", "entity", "dir", "notes", "tags", "resume", "id"):
        assert name in parameters, f"wandb.init lost {name!r}"
    assert "config" in parameters

    log_parameters = inspect.signature(wandb.log).parameters
    assert "step" in log_parameters and "commit" in log_parameters


def test_wandb_is_detected_as_taking_commit():
    assert _accepts_commit(wandb) is True


def test_get_backend_logs_in_and_returns_the_module(monkeypatch):
    calls = {}
    monkeypatch.setattr(wandb, "login", lambda **kwargs: calls.update(kwargs))
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setenv("WANDB_HOST", "https://example.invalid")
    assert get_backend("wandb") is wandb
    assert calls == {"key": "k", "host": "https://example.invalid"}


# ------------------------------------------------------------------ init


def test_init_maps_our_metadata_onto_wandb(run):
    run(
        name="mapped",
        config={"lr": 0.1, "arch": "resnet"},
        notes="a note",
        tags=["alpha", "beta"],
    )
    assert wandb.run is not None
    assert wandb.run.id == "mapped"  # the run name doubles as the wandb id
    assert wandb.run.name == "mapped"
    assert wandb.run.tags == ("alpha", "beta")
    assert wandb.run.notes == "a note"
    assert dict(wandb.run.config) == {"lr": 0.1, "arch": "resnet"}


def test_an_empty_config_is_still_valid(run):
    run(name="noconfig")
    assert dict(wandb.run.config) == {}


def test_the_run_directory_is_created_where_we_asked(run, tmp_path):
    """Our ``dir`` wins over WANDB_DIR, so both sides land under one tree."""
    run(name="dir")
    assert Path(wandb.run.dir).is_relative_to(tmp_path / "et")
    assert offline_run_dir(tmp_path).name.endswith("dir")


def test_url_is_reported_without_raising(run):
    instance = run(name="url")
    assert instance.url is None or isinstance(instance.url, str)
    assert "wandb" in instance.info()


# ------------------------------------------------------------------ steps


def test_wandb_steps_track_our_steps(run):
    instance = run(name="steps")
    for expected in range(5):
        instance.log({"loss": float(expected)})
        assert wandb.run.step == expected + 1  # wandb counts the next step
    assert [r["_step"] for r in instance.history_query(-1)] == list(range(5))


def test_several_logs_for_one_step_stay_one_wandb_step(run):
    """The regression this guards: forwarding step=None made wandb advance twice."""
    instance = run(name="merge")
    instance.log({"a": 1}, commit=False)
    instance.log({"b": 2}, commit=False)
    instance.log({"c": 3})
    assert wandb.run.step == 1
    assert len(instance.history_query(-1)) == 1


def test_an_explicit_step_defers_the_commit_on_wandb_too(run):
    instance = run(name="explicit")
    instance.log({"a": 1}, step=7)
    assert wandb.run.step == 7  # positioned, not yet advanced
    instance.log({"b": 2}, step=7)
    assert wandb.run.step == 7
    instance.log({"c": 3}, step=8, commit=True)
    assert wandb.run.step == 9


def test_a_dropped_backward_step_never_reaches_wandb(run):
    instance = run(name="dropped")
    instance.log({"loss": 1.0}, step=5, commit=True)
    assert wandb.run.step == 6
    instance.log({"loss": 2.0}, step=1)  # rejected locally
    assert wandb.run.step == 6  # and never forwarded
    assert [r["_step"] for r in instance.history_query(-1)] == [5]


def test_wandb_summary_holds_the_last_committed_values(run):
    instance = run(name="summary")
    for step in range(4):
        instance.log({"loss": float(step), "acc": step / 10})
    live = {k: v for k, v in dict(wandb.run.summary).items() if not k.startswith("_")}
    assert live["loss"] == 3.0
    assert live["acc"] == pytest.approx(0.3)
    assert dict(instance.summary)["loss"] == 3.0  # our summary agrees


def test_our_history_and_wandb_agree_on_every_step(run):
    instance = run(name="agree")
    for step in range(20):
        instance.log({"loss": float(step)})
        if step % 5 == 0:
            instance.log({"eval": float(step)}, step=step)
    instance.history.flush(commit_open=True)
    assert wandb.run.step == 20
    assert len(instance.history_query(-1)) == 20


# ------------------------------------------------------------------ features


def test_define_metric_reaches_wandb(run):
    instance = run(name="define")
    instance.define_metric("loss", summary="min")
    instance.define_metric("acc", summary="max", step_metric="loss")
    instance.log({"loss": 1.0, "acc": 0.5})
    assert wandb.run.step == 1  # the run is still healthy afterwards


def test_log_artifact_reaches_wandb(run, tmp_path):
    source = tmp_path / "model.bin"
    source.write_bytes(b"weights")
    instance = run(name="artifact")
    logged = instance.log_artifact(str(source), name="model", type="model")
    assert logged.version == 0
    assert (logged.dir / "model.bin").read_bytes() == b"weights"
    assert wandb.run is not None  # a rejected artifact would not kill the run


def test_finish_accepts_an_exit_code(run, tmp_path):
    instance = run(name="exit")
    instance.log({"loss": 1.0})
    instance.finish(exit_code=1)
    assert wandb.run is None
    assert offline_run_dir(tmp_path).is_dir()


def test_a_run_survives_wandb_failing_mid_loop(run, monkeypatch):
    instance = run(name="broken")
    instance.log({"loss": 1.0})
    monkeypatch.setattr(
        wandb, "log", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
    )
    for step in range(5):
        instance.log({"loss": float(step)})
    assert len(instance.history_query(-1)) == 6  # local history is untouched


# ------------------------------------------------------------------ end to end


def test_a_full_offline_run_is_recorded_on_both_sides(tmp_path):
    et.init(
        project="wbtest",
        name="e2e",
        dir=str(tmp_path / "et"),
        backends=["wandb"],
        config={"lr": 0.1},
        max_open_seconds=None,
    )
    try:
        for step in range(30):
            et.log({"train/loss": 1.0 / (step + 1), "lr": 0.1})
            if step % 10 == 0:
                et.log({"eval/acc": step / 30}, step=step)
        et.summary()["best_acc"] = 0.9
        assert wandb.run.step == 30
    finally:
        et.finish()

    rows = et.history(-1, run=str(tmp_path / "et" / "wbtest" / "e2e"))
    assert [r["_step"] for r in rows] == list(range(30))
    assert sum("eval/acc" in r for r in rows) == 3
    saved = json.loads(
        (tmp_path / "et" / "wbtest" / "e2e" / "summary.json").read_text()
    )
    assert saved["best_acc"] == 0.9
    assert offline_run_dir(tmp_path).is_dir()


def test_jsonl_only_runs_do_not_start_wandb(tmp_path):
    et.init(project="wbtest", name="local", dir=str(tmp_path / "et"), backends=[])
    try:
        et.log({"loss": 1.0})
        assert wandb.run is None
    finally:
        et.finish()
