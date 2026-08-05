"""Artifact store, run summary, and wandb API compatibility."""

import inspect
import json
from pathlib import Path

import pytest

import expr_tracker as et
from expr_tracker.artifacts import Artifact, ArtifactStore, coerce_artifact
from expr_tracker.summary import Summary


@pytest.fixture
def payload(tmp_path):
    """A small tree to log: one file plus a directory."""
    (tmp_path / "ckpt.pt").write_bytes(b"weights-v1")
    config = tmp_path / "cfg"
    config.mkdir()
    (config / "a.yaml").write_text("lr: 0.1")
    (config / "nested").mkdir()
    (config / "nested" / "b.yaml").write_text("wd: 0.01")
    return tmp_path


# ---------------------------------------------------------------- artifact store


def test_log_artifact_from_path(run, payload):
    r = run(backends=[])
    artifact = et.log_artifact(str(payload / "ckpt.pt"), name="model", type="model")
    assert artifact.qualified_name == "model:v0"
    assert artifact.files() == ["ckpt.pt"]
    assert (artifact.dir / "ckpt.pt").read_bytes() == b"weights-v1"
    assert artifact.digest.startswith("sha256:")
    assert artifact.run == r.name and artifact.step == 0


def test_log_artifact_from_object(run, payload):
    run(backends=[])
    artifact = Artifact("bundle", type="config", metadata={"seed": 1})
    artifact.add_dir(payload / "cfg").add_reference("s3://bucket/data.tar")
    logged = et.log_artifact(artifact, aliases=["release"])
    assert logged.files() == ["a.yaml", "nested/b.yaml", "data.tar"]
    assert logged.metadata == {"seed": 1}
    assert "release" in logged.aliases
    assert (logged.dir / "nested" / "b.yaml").read_text() == "wd: 0.01"
    # references are recorded but never materialised
    assert not (logged.dir / "data.tar").exists()


def test_identical_contents_reuse_a_version(run, payload):
    run(backends=[])
    first = et.log_artifact(str(payload / "ckpt.pt"), name="model", type="model")
    second = et.log_artifact(str(payload / "ckpt.pt"), name="model", type="model")
    assert first.version == second.version == 0

    (payload / "ckpt.pt").write_bytes(b"weights-v2")
    third = et.log_artifact(str(payload / "ckpt.pt"), name="model", type="model")
    assert third.version == 1


def test_use_artifact_resolution(run, payload, tmp_path):
    run(backends=[])
    et.log_artifact(
        str(payload / "ckpt.pt"), name="model", type="model", aliases=["best"]
    )
    (payload / "ckpt.pt").write_bytes(b"weights-v2")
    et.log_artifact(str(payload / "ckpt.pt"), name="model", type="model")

    assert et.use_artifact("model").qualified_name == "model:v1"
    assert et.use_artifact("model:latest").qualified_name == "model:v1"
    assert et.use_artifact("model:v0").qualified_name == "model:v0"
    assert et.use_artifact("model:best").qualified_name == "model:v0"
    with pytest.raises(FileNotFoundError):
        et.use_artifact("nope")
    with pytest.raises(ValueError, match="has type"):
        et.use_artifact("model", type="dataset")


def test_artifacts_are_shared_across_runs_of_a_project(tmp_path, payload):
    et.init(project="p", name="producer", dir=str(tmp_path), backends=[])
    et.log_artifact(str(payload / "ckpt.pt"), name="model", type="model")
    et.finish()

    et.init(project="p", name="consumer", dir=str(tmp_path), backends=[])
    try:
        artifact = et.use_artifact("model:latest")
        assert artifact.run == "producer"
        restored = artifact.download(tmp_path / "restored")
        assert (restored / "ckpt.pt").read_bytes() == b"weights-v1"
    finally:
        et.finish()


def test_artifact_lineage_is_recorded(run, payload):
    r = run(backends=[])
    et.log_artifact(str(payload / "ckpt.pt"), name="model", type="model")
    et.use_artifact("model:latest")
    lines = [
        json.loads(line)
        for line in (r.history.log_dir / "artifacts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [entry["action"] for entry in lines] == ["log", "use"]
    assert all(entry["name"] == "model" for entry in lines)


def test_copy_and_reference_modes(run, payload):
    run(backends=[])
    copied = et.log_artifact(
        str(payload / "ckpt.pt"), name="copied", type="model", mode="copy"
    )
    assert (copied.dir / "ckpt.pt").exists()
    referenced = et.log_artifact(
        str(payload / "ckpt.pt"), name="ref", type="model", mode="reference"
    )
    assert referenced.dir is not None and not referenced.dir.exists()
    with pytest.raises(ValueError, match="Unknown artifact mode"):
        et.log_artifact(str(payload / "ckpt.pt"), name="bad", mode="teleport")


def test_artifact_input_validation(payload):
    with pytest.raises(ValueError, match="name must not be empty"):
        Artifact("")
    with pytest.raises(FileNotFoundError):
        Artifact("a").add_file(payload / "missing.bin")
    with pytest.raises(NotADirectoryError):
        Artifact("a").add_dir(payload / "missing_dir")
    with pytest.raises(RuntimeError, match="has not been logged"):
        Artifact("a").download()


def test_coerce_artifact_defaults(payload):
    artifact = coerce_artifact(str(payload / "ckpt.pt"))
    assert artifact.name == "ckpt" and artifact.type == "dataset"
    directory = coerce_artifact(payload / "cfg", name="cfg", type="config")
    assert directory.files() == ["a.yaml", "nested/b.yaml"]


def test_store_tolerates_corrupt_index(tmp_path, payload):
    store = ArtifactStore(root=tmp_path / "artifacts")
    store.log(Artifact("a").add_file(payload / "ckpt.pt"))
    with open(store.index_path, "a", encoding="utf-8") as f:
        f.write("{not json}\n")
    assert [a.name for a in store.entries()] == ["a"]
    assert store.resolve("a").version == 0
    assert store.resolve("missing") is None


# ---------------------------------------------------------------- summary


def test_summary_tracks_last_value(run):
    run(backends=[])
    et.log({"loss": 1.0, "acc": 0.1})
    et.log({"loss": 0.5})
    assert dict(et.summary()) == {"loss": 0.5, "acc": 0.1}


def test_explicit_summary_entries_are_pinned(run):
    run(backends=[])
    et.log({"loss": 1.0})
    et.summary()["loss"] = 0.0
    et.log({"loss": 9.0})
    assert et.summary()["loss"] == 0.0


def test_summary_is_persisted_and_reloaded(tmp_path):
    r = et.init(project="p", name="s", dir=str(tmp_path), backends=[])
    et.log({"loss": 0.25})
    path = r.history.log_dir / "summary.json"
    et.finish()
    assert json.loads(path.read_text())["loss"] == 0.25
    assert Summary(path)["loss"] == 0.25


def test_summary_mapping_operations(tmp_path):
    summary = Summary(tmp_path / "summary.json")
    summary["a"] = 1
    assert list(summary) == ["a"] and len(summary) == 1
    del summary["a"]
    assert len(summary) == 0
    summary.observe({"_step": 3, "b": 2})
    assert "_step" not in summary and summary["b"] == 2


# ---------------------------------------------------------------- wandb compat


def test_log_signature_matches_wandb():
    wandb = pytest.importorskip("wandb")
    assert (
        list(inspect.signature(et.log).parameters)
        == list(inspect.signature(wandb.sdk.wandb_run.Run.log).parameters)[1:]
    )


def test_public_api_covers_core_wandb_names():
    for name in (
        "init",
        "log",
        "finish",
        "alert",
        "log_artifact",
        "use_artifact",
        "define_metric",
        "Artifact",
        "summary",
    ):
        assert hasattr(et, name), name


def test_run_exposes_wandb_like_attributes(run, tmp_path):
    r = run(backends=[])
    assert et.get_run() is r
    assert r.project == "p" and r.name
    assert Path(r.dir).is_dir()
    assert r.step == 0
    et.log({"v": 1})
    assert r.step == 1
    assert r.url is None  # no wandb backend configured
    assert r.tags == [] and r.entity is None


def test_define_metric_is_a_safe_noop(run):
    run(backends=[])
    et.define_metric("loss", summary="min")  # must not raise without a backend


def test_finish_accepts_wandb_arguments(tmp_path):
    et.init(project="p", name="exit", dir=str(tmp_path), backends=[])
    et.finish(exit_code=0, quiet=True)
    assert et.get_run() is None


def test_backend_receives_commit_argument(tmp_path):
    calls = []

    class Backend:
        def init(self, **kwargs):
            pass

        def log(self, data, step=None, commit=None):
            calls.append((dict(data), step, commit))

        def finish(self):
            pass

    et.init(project="p", name="fwd", dir=str(tmp_path), backends=[Backend()])
    try:
        et.log({"v": 1}, step=3, commit=False)
    finally:
        et.finish()
    assert calls == [({"v": 1}, 3, False)]


def test_legacy_backend_without_commit_still_works(tmp_path):
    calls = []

    class LegacyBackend:
        def init(self, **kwargs):
            pass

        def log(self, data, step=None):
            calls.append((dict(data), step))

        def finish(self):
            pass

    et.init(project="p", name="legacy", dir=str(tmp_path), backends=[LegacyBackend()])
    try:
        et.log({"v": 1}, step=2)
    finally:
        et.finish()
    assert calls == [({"v": 1}, 2)]


def test_run_construction_rolls_back_on_failure(tmp_path):
    """A bad alert rule must not leave a half-open run behind."""
    import threading

    from expr_tracker.alerts.expr import ExprError
    from expr_tracker.run import current_run

    before = {t.name for t in threading.enumerate()}
    with pytest.raises(ExprError):
        et.init(
            project="p",
            name="bad",
            dir=str(tmp_path),
            backends=[],
            alert_rules=["no_data(1s) => error: hung", "mean(loss) => warn: bad"],
        )
    assert current_run() is None
    leaked = {t.name for t in threading.enumerate()} - before
    assert not [n for n in leaked if n.startswith("et-alert")]


def test_rejected_step_reaches_no_sink(tmp_path):
    """A backward step must be skipped by history, summary and remote backends alike."""
    forwarded = []

    class Backend:
        def init(self, **kwargs):
            pass

        def log(self, data, step=None, commit=None):
            forwarded.append(dict(data))

        def finish(self):
            pass

    et.init(project="p", name="mono", dir=str(tmp_path), backends=[Backend()])
    try:
        et.log({"loss": 1.0}, step=10)
        et.log({"loss": 99.0}, step=5)  # rejected under the default step policy
        assert forwarded == [{"loss": 1.0}]
        assert dict(et.summary()) == {"loss": 1.0}
    finally:
        et.finish()


def test_alerts_are_disabled_on_non_zero_ranks(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    r = et.init(
        project="p",
        name="ddp",
        dir=str(tmp_path),
        backends=[],
        alert_rules=["loss > 1 => error: x"],
    )
    try:
        assert r.rank == 3
        assert r.history.log_fp.name == "metrics.rank3.jsonl"
        assert et.list_alert_rules() == []
    finally:
        et.finish()


def test_rule_window_grows_the_metric_buffer(tmp_path):
    r = et.init(project="p", name="win", dir=str(tmp_path), backends=[], alert_window=8)
    try:
        et.add_alert_rule("mean(loss[300]) > 1 => warn: x")
        assert r.history.series.window >= 300
        for i in range(400):
            et.log({"loss": float(i)})
        assert len(r.history.series.points("loss")) == 300
    finally:
        et.finish()
