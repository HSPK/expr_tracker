"""Unit tests for run-level wiring: backend selection, forwarding and failure isolation."""

import json
import sys
import types

import pytest

import expr_tracker as et
from expr_tracker.run import Run, _accepts_commit, get_backend


class Wandb:
    """Named so that ``Run`` registers it under the "wandb" key."""

    run = None

    def init(self, **kwargs): ...
    def log(self, data, step=None, commit=None): ...
    def finish(self, **kwargs): ...


class FakeBackend:
    """Minimal wandb-shaped backend that records every call."""

    def __init__(self, fail: set[str] = frozenset()):
        self.calls: list[tuple] = []
        self.fail = fail

    def _record(self, _call, *args, **kwargs):
        self.calls.append((_call, args, kwargs))
        if _call in self.fail:
            raise RuntimeError(f"{_call} boom")

    def init(self, **kwargs):
        self._record("init", **kwargs)

    def log(self, data, step=None, commit=None):
        self._record("log", data, step=step, commit=commit)

    def define_metric(self, name, **kwargs):
        self._record("define_metric", name, **kwargs)

    def log_artifact(self, path, **kwargs):
        self._record("log_artifact", path, **kwargs)

    def finish(self, **kwargs):
        self._record("finish", **kwargs)

    def names(self):
        return [call[0] for call in self.calls]


@pytest.fixture
def fake_module(monkeypatch):
    """Install importable stand-ins for the optional wandb/trackio dependencies."""

    def make(name):
        module = types.ModuleType(name)
        module.login = lambda **kwargs: module.__dict__.setdefault(
            "login_kwargs", kwargs
        )
        monkeypatch.setitem(sys.modules, name, module)
        return module

    return make


def test_get_backend_passes_objects_through():
    backend = FakeBackend()
    assert get_backend(backend) is backend


def test_get_backend_jsonl_is_not_a_backend():
    assert get_backend("jsonl") is None
    assert get_backend("JSONL") is None


def test_get_backend_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("mlflow")


def test_get_backend_wandb_logs_in(fake_module, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "key-123")
    monkeypatch.setenv("WANDB_HOST", "https://example.invalid")
    module = fake_module("wandb")
    assert get_backend("wandb") is module
    assert module.login_kwargs == {
        "key": "key-123",
        "host": "https://example.invalid",
    }


def test_get_backend_trackio(fake_module):
    module = fake_module("trackio")
    assert get_backend("trackio") is module


@pytest.mark.parametrize("name", ["wandb", "trackio"])
def test_get_backend_reports_missing_dependency(name, monkeypatch):
    monkeypatch.setitem(sys.modules, name, None)  # forces ImportError
    with pytest.raises(ImportError, match=f'pip install "expr_tracker\\[{name}\\]"'):
        get_backend(name)


def test_unavailable_backend_is_skipped_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    run = Run(project="p", name="r", dir=str(tmp_path), backends=["wandb"])
    try:
        assert run.backends == {}
        run.log({"loss": 1.0})  # history still works on its own
        assert len(run.history_query(-1)) == 1
    finally:
        run.finish()


def test_backend_init_failure_drops_the_backend(tmp_path):
    backend = FakeBackend(fail={"init"})
    run = Run(project="p", name="r", dir=str(tmp_path), backends=[backend])
    try:
        assert run.backends == {}
        run.log({"loss": 1.0})
        assert backend.names() == ["init"]  # no log reached a dead backend
    finally:
        run.finish()


def test_backend_receives_init_log_and_finish(tmp_path):
    backend = FakeBackend()
    run = Run(
        project="p",
        name="r",
        dir=str(tmp_path),
        backends=[backend],
        config={"lr": 0.1},
        tags=["a"],
        notes="hello",
    )
    run.log({"loss": 1.0}, step=0)
    run.define_metric("loss", summary="min")
    run.finish()

    assert backend.names() == ["init", "log", "define_metric", "finish"]
    init_kwargs = backend.calls[0][2]
    assert init_kwargs["project"] == "p" and init_kwargs["config"] == {"lr": 0.1}
    assert init_kwargs["tags"] == ["a"] and init_kwargs["notes"] == "hello"
    assert backend.calls[1][1] == ({"loss": 1.0},)
    # an explicit step defers the commit locally, so the backend defers it too
    assert backend.calls[1][2] == {"step": 0, "commit": False}


def test_backend_is_named_after_its_class(tmp_path):
    backend = FakeBackend()
    run = Run(project="p", name="r", dir=str(tmp_path), backends=[backend])
    try:
        assert list(run.backends) == ["fakebackend"]
    finally:
        run.finish()


def test_duplicate_backends_are_initialised_once(tmp_path):
    backend = FakeBackend()
    run = Run(project="p", name="r", dir=str(tmp_path), backends=[backend, backend])
    try:
        assert backend.names() == ["init"]
    finally:
        run.finish()


def test_failures_in_log_define_and_finish_are_contained(tmp_path):
    backend = FakeBackend(fail={"log", "define_metric", "finish", "log_artifact"})
    source = tmp_path / "f.bin"
    source.write_bytes(b"x")
    run = Run(project="p", name="r", dir=str(tmp_path), backends=[backend])
    run.log({"loss": 1.0})
    run.define_metric("loss")
    artifact = run.log_artifact(str(source), name="m", type="model")
    run.finish()

    assert artifact.version == 0  # the local artifact still landed
    assert len(run.history_query(-1)) == 1  # and so did the local history
    assert backend.names() == ["init", "log", "define_metric", "log_artifact", "finish"]


def test_trackio_init_folds_metadata_into_config(tmp_path, fake_module):
    module = fake_module("trackio")
    backend = FakeBackend()
    module.init, module.log, module.finish = backend.init, backend.log, backend.finish
    run = Run(
        project="p",
        name="r",
        dir=str(tmp_path),
        backends=["trackio"],
        notes="n",
        tags=["t"],
    )
    try:
        config = backend.calls[0][2]["config"]
        assert config["trackio.notes"] == "n" and config["trackio.tags"] == ["t"]
        assert "entity" not in backend.calls[0][2]  # trackio.init has no entity kwarg
    finally:
        run.finish()


def test_url_prefers_wandb_and_tolerates_absence(tmp_path):
    backend = Wandb()
    run = Run(project="p", name="r", dir=str(tmp_path), backends=[backend])
    try:
        assert run.url is None  # no .run attribute yet
        backend.run = types.SimpleNamespace(url="https://wandb.test/r")
        assert run.url == "https://wandb.test/r"
        assert run.info()["wandb"]["url"] == "https://wandb.test/r"
    finally:
        run.finish()


def test_info_reports_every_section(tmp_path):
    backend = FakeBackend()
    run = Run(
        project="p",
        name="r",
        dir=str(tmp_path),
        backends=[backend],
        alert={"channels": []},
        alert_rules=["loss > 1 => warn"],
    )
    try:
        run.log({"loss": 5.0})
        run.summary["best"] = 1
        info = run.info()
        assert info["rank"] == 0
        assert info["summary"]["best"] == 1
        assert info["fakebackend"] == {}
        assert info["jsonl"]["log_dir"] == info["history"]["log_dir"]
        assert info["artifacts"]["root"].endswith("/p/artifacts")
        assert len(info["alerts"]["rules"]) == 1
    finally:
        run.finish()


def test_lineage_failure_does_not_break_log_artifact(tmp_path, monkeypatch):
    source = tmp_path / "f.bin"
    source.write_bytes(b"x")
    run = Run(project="p", name="r", dir=str(tmp_path), backends=[])
    try:
        (run.history.log_dir / "artifacts.jsonl").mkdir()  # make the append fail
        artifact = run.log_artifact(str(source), name="m", type="model")
        assert artifact.version == 0
    finally:
        run.finish()


def test_lineage_records_use_as_well_as_log(tmp_path):
    source = tmp_path / "f.bin"
    source.write_bytes(b"x")
    run = Run(project="p", name="r", dir=str(tmp_path), backends=[])
    try:
        run.log_artifact(str(source), name="m", type="model")
        run.log({"loss": 1.0})
        run.use_artifact("m:latest")
        lines = (run.history.log_dir / "artifacts.jsonl").read_text().splitlines()
        records = [json.loads(line) for line in lines]
        assert [r["action"] for r in records] == ["log", "use"]
        assert records[1]["step"] == 1  # lineage remembers when it was consumed
    finally:
        run.finish()


def test_summary_failure_does_not_stop_backend_logging(tmp_path, monkeypatch):
    backend = FakeBackend()
    run = Run(project="p", name="r", dir=str(tmp_path), backends=[backend])
    try:
        monkeypatch.setattr(
            run.summary,
            "observe",
            lambda data: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        run.log({"loss": 1.0})
        assert backend.names() == ["init", "log"]
    finally:
        run.finish()


def test_rejected_step_reaches_no_sink(tmp_path):
    backend = FakeBackend()
    run = Run(project="p", name="r", dir=str(tmp_path), backends=[backend])
    try:
        run.log({"loss": 1.0}, step=5)
        run.log({"loss": 2.0}, step=1)  # backwards: dropped
        assert [c[1][0] for c in run.backends["fakebackend"].calls[1:]] == [
            {"loss": 1.0}
        ]
        assert dict(run.summary)["loss"] == 1.0
    finally:
        run.finish()


def test_finish_is_idempotent(tmp_path):
    backend = FakeBackend()
    run = Run(project="p", name="r", dir=str(tmp_path), backends=[backend])
    run.finish()
    run.finish()
    assert backend.names().count("finish") == 1


def test_accepts_commit_detection():
    class Explicit:
        def log(self, data, step=None, commit=None): ...

    class Kwargs:
        def log(self, data, **kwargs): ...

    class Neither:
        def log(self, data, step=None): ...

    assert _accepts_commit(Explicit()) is True
    assert _accepts_commit(Kwargs()) is True
    assert _accepts_commit(Neither()) is False
    assert _accepts_commit(types.SimpleNamespace(log=print)) is False


def test_commit_is_only_sent_to_backends_that_accept_it(tmp_path):
    class NoCommit(FakeBackend):
        def log(self, data, step=None):
            self._record("log", data, step=step)

    backend = NoCommit()
    run = Run(project="p", name="r", dir=str(tmp_path), backends=[backend])
    try:
        run.log({"loss": 1.0}, commit=True)
        assert backend.calls[1][2] == {"step": 0}  # resolved, not None
    finally:
        run.finish()


@pytest.mark.parametrize(
    "call",
    [
        lambda: et.info(),
        lambda: et.log({"loss": 1.0}),
        lambda: et.summary(),
        lambda: et.finish(),
        lambda: et.history(5),
        lambda: et.define_metric("loss"),
        lambda: et.use_artifact("m:latest"),
    ],
)
def test_module_helpers_refuse_to_run_without_init(call):
    assert et.get_run() is None
    with pytest.raises(RuntimeError, match="not initialized"):
        call()
