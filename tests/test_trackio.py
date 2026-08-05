"""Trackio backend integration, exercised against the real package when installed."""

import inspect

import pytest

from expr_tracker.run import Run, _accepts_commit, _trackio_resume, get_backend

trackio = pytest.importorskip("trackio")


@pytest.fixture
def local_trackio(tmp_path, monkeypatch):
    """Keep trackio entirely on disk and out of the network."""
    monkeypatch.setenv("TRACKIO_DIR", str(tmp_path / "trackio"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return trackio


# ------------------------------------------------------------------ contract


def test_the_real_trackio_matches_our_assumptions(local_trackio):
    """Our special-casing depends on this signature; fail loudly if it changes."""
    parameters = inspect.signature(trackio.init).parameters
    assert "project" in parameters and "name" in parameters
    assert "config" in parameters and "resume" in parameters
    # the fields we deliberately fold into config because trackio has no slot
    assert not {"entity", "notes", "tags", "dir", "id"} & set(parameters)

    log_parameters = inspect.signature(trackio.log).parameters
    assert "metrics" in log_parameters and "step" in log_parameters
    assert "commit" not in log_parameters


def test_trackio_does_not_take_a_commit_argument(local_trackio):
    assert _accepts_commit(trackio) is False


def test_get_backend_returns_the_module(local_trackio):
    assert get_backend("trackio") is trackio
    assert get_backend("TrackIO") is trackio


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("must", "must"),
        ("allow", "allow"),
        ("never", "never"),
        ("auto", "allow"),
        (True, "allow"),
        (False, "never"),
        (None, "never"),
        ("nonsense", "allow"),
    ],
)
def test_resume_is_mapped_onto_what_trackio_accepts(given, expected):
    assert _trackio_resume(given) == expected


def test_every_mapped_resume_value_is_accepted_by_trackio(local_trackio):
    """trackio validates resume strictly, so the mapping must never produce junk."""
    source = inspect.getsource(trackio.init)
    for value in ("must", "allow", "never"):
        assert f'"{value}"' in source


# ------------------------------------------------------------------ wiring


class RecordingTrackio:
    """Stands in for the module so assertions do not depend on trackio internals."""

    def __init__(self):
        self.calls = []

    def init(self, **kwargs):
        self.calls.append(("init", kwargs))

    def log(self, metrics, step=None):
        self.calls.append(("log", {"metrics": metrics, "step": step}))

    def finish(self):
        self.calls.append(("finish", {}))

    def kinds(self):
        return [name for name, _ in self.calls]


@pytest.fixture
def fake_trackio(monkeypatch):
    import sys

    module = RecordingTrackio()
    monkeypatch.setitem(sys.modules, "trackio", module)
    return module


def test_run_initialises_trackio_with_a_folded_config(tmp_path, fake_trackio):
    run = Run(
        project="p",
        name="r",
        dir=str(tmp_path),
        backends=["trackio"],
        config={"lr": 0.1},
        notes="a note",
        tags=["x"],
        entity="team",
        resume="allow",
    )
    try:
        kwargs = fake_trackio.calls[0][1]
        assert kwargs["project"] == "p" and kwargs["name"] == "r"
        assert kwargs["resume"] == "allow"  # forwarded, not buried in config
        assert kwargs["config"]["lr"] == 0.1
        assert kwargs["config"]["trackio.notes"] == "a note"
        assert kwargs["config"]["trackio.tags"] == ["x"]
        assert kwargs["config"]["trackio.entity"] == "team"
        assert "entity" not in kwargs and "dir" not in kwargs
    finally:
        run.finish()


@pytest.mark.parametrize("resume", ["auto", True, None, False, "must"])
def test_wandb_style_resume_values_never_break_trackio(tmp_path, fake_trackio, resume):
    run = Run(
        project="p", name="r", dir=str(tmp_path), backends=["trackio"], resume=resume
    )
    try:
        assert run.backends  # the backend survived initialisation
        assert fake_trackio.calls[0][1]["resume"] in ("must", "allow", "never")
    finally:
        run.finish()


def test_trackio_receives_the_resolved_step(tmp_path, fake_trackio):
    run = Run(project="p", name="r", dir=str(tmp_path), backends=["trackio"])
    try:
        run.log({"loss": 1.0})
        run.log({"acc": 0.5}, step=0, commit=False)
        run.log({"acc2": 0.6}, step=0)
        logs = [call for name, call in fake_trackio.calls if name == "log"]
        assert [entry["step"] for entry in logs] == [0, 0, 0]
        assert logs[0]["metrics"] == {"loss": 1.0}
    finally:
        run.finish()


def test_a_dropped_step_never_reaches_trackio(tmp_path, fake_trackio):
    run = Run(project="p", name="r", dir=str(tmp_path), backends=["trackio"])
    try:
        run.log({"loss": 1.0}, step=5)
        run.log({"loss": 2.0}, step=1)  # backwards: dropped locally
        logs = [call for name, call in fake_trackio.calls if name == "log"]
        assert [entry["step"] for entry in logs] == [5]
    finally:
        run.finish()


def test_trackio_is_finished_without_arguments(tmp_path, fake_trackio):
    run = Run(project="p", name="r", dir=str(tmp_path), backends=["trackio"])
    run.log({"loss": 1.0})
    run.finish()
    assert fake_trackio.kinds() == ["init", "log", "finish"]
    assert fake_trackio.calls[-1][1] == {}


def test_a_failing_trackio_does_not_stop_local_history(tmp_path, monkeypatch):
    import sys

    class Broken(RecordingTrackio):
        def log(self, metrics, step=None):
            raise RuntimeError("trackio is down")

    monkeypatch.setitem(sys.modules, "trackio", Broken())
    run = Run(project="p", name="r", dir=str(tmp_path), backends=["trackio"])
    try:
        for step in range(5):
            run.log({"loss": float(step)})
        assert len(run.history_query(-1)) == 5
    finally:
        run.finish()


def test_trackio_and_history_agree_on_steps(tmp_path, fake_trackio):
    run = Run(project="p", name="r", dir=str(tmp_path), backends=["trackio"])
    try:
        for step in range(20):
            run.log({"loss": float(step)})
            if step % 5 == 0:
                run.log({"eval": float(step)}, step=step)
        run.history.flush(commit_open=True)

        logged = [call["step"] for name, call in fake_trackio.calls if name == "log"]
        history = [row["_step"] for row in run.history_query(-1)]
        assert set(logged) == set(history)  # no step exists on one side only
        assert len(history) == 20
    finally:
        run.finish()


# ------------------------------------------------------------------ end to end


def test_a_real_trackio_run_records_our_metrics(local_trackio, tmp_path):
    """Full loop against the installed trackio, kept local via TRACKIO_DIR."""
    run = Run(
        project="et_trackio_test",
        name="e2e",
        dir=str(tmp_path / "et"),
        backends=["trackio"],
        config={"lr": 0.1},
        resume="never",
    )
    try:
        if not run.backends:
            pytest.skip("trackio could not start in this environment")
        for step in range(10):
            run.log({"loss": 1.0 / (step + 1), "acc": step / 10})
    finally:
        run.finish()

    rows = run.history_query(-1)
    assert [r["_step"] for r in rows] == list(range(10))
    assert rows[-1]["acc"] == 0.9
