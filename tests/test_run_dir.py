"""Where a run's files land: the default layout, and naming it outright."""

import contextlib
from pathlib import Path

import pytest

import expr_tracker as et
from expr_tracker.history import HistoryStore
from expr_tracker.history.naming import (
    DEFAULT_ROOT,
    resolve_artifact_root,
    resolve_log_dir,
)


@pytest.fixture
def run(tmp_path):
    created = []

    def factory(**options):
        options.setdefault("max_open_seconds", None)
        created.append(
            et.init(project="proj", name="job", backends=[], resume="allow", **options)
        )
        return created[-1]

    yield factory
    for _ in created:
        with contextlib.suppress(Exception):
            et.finish()


# ------------------------------------------------------------------ resolution


def test_dir_is_a_root_holding_projects(tmp_path):
    assert resolve_log_dir("proj", "job", str(tmp_path)) == tmp_path / "proj" / "job"


def test_the_default_root_is_used_when_no_dir_is_given():
    assert resolve_log_dir("proj", "job") == Path(DEFAULT_ROOT) / "proj" / "job"


def test_run_dir_names_the_directory_outright(tmp_path):
    assert resolve_log_dir("proj", "job", run_dir=str(tmp_path)) == tmp_path


def test_dir_and_run_dir_together_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        resolve_log_dir("proj", "job", str(tmp_path), str(tmp_path))


def test_artifacts_sit_beside_the_projects_runs(tmp_path):
    root = resolve_artifact_root("proj", str(tmp_path))
    assert root == tmp_path / "proj" / "artifacts"
    assert root.parent == resolve_log_dir("proj", "job", str(tmp_path)).parent


def test_artifacts_move_inside_a_named_run_dir(tmp_path):
    assert (
        resolve_artifact_root("proj", run_dir=str(tmp_path)) == tmp_path / "artifacts"
    )


# ------------------------------------------------------------------ the run


def test_the_default_layout_nests_project_and_name(run, tmp_path):
    instance = run(dir=str(tmp_path))
    assert instance.history.log_dir == tmp_path / "proj" / "job"
    assert instance.artifacts.root == tmp_path / "proj" / "artifacts"


def test_a_named_run_dir_is_used_exactly(run, tmp_path):
    target = tmp_path / "somewhere" / "else"
    instance = run(run_dir=str(target))
    instance.log({"loss": 1.0})
    instance.history.flush(commit_open=True)
    assert instance.history.log_dir == target
    assert (target / "metrics.jsonl").is_file()
    assert not (target / "proj").exists()  # no silent nesting


def test_a_named_run_dir_keeps_its_artifacts_inside(run, tmp_path):
    target = tmp_path / "solo"
    instance = run(run_dir=str(target))
    source = tmp_path / "ckpt.pt"
    source.write_text("weights")
    instance.log_artifact(str(source), name="model")
    assert instance.artifacts.root == target / "artifacts"
    assert (target / "artifacts").is_dir()


def test_the_run_reports_the_directory_it_chose(run, tmp_path):
    instance = run(dir=str(tmp_path))
    assert instance.dir == str(tmp_path / "proj" / "job")


def test_passing_both_is_rejected_before_anything_is_written(run, tmp_path):
    with pytest.raises(ValueError, match="not both"):
        run(dir=str(tmp_path), run_dir=str(tmp_path / "x"))
    assert not (tmp_path / "x").exists()


def test_the_resolved_directory_is_announced(tmp_path, caplog):
    """The nesting must not be something you have to go and look up."""
    from loguru import logger

    lines: list = []
    handle = logger.add(lambda m: lines.append(m), level="INFO")
    try:
        store = HistoryStore()
        store.init(project="proj", name="job", dir=str(tmp_path))
        store.finish()
    finally:
        logger.remove(handle)
    assert any(str(tmp_path / "proj" / "job") in line for line in lines)


def test_a_resume_says_where_it_picked_up(tmp_path):
    from loguru import logger

    first = HistoryStore()
    first.init(project="proj", name="job", dir=str(tmp_path), max_open_seconds=None)
    first.log({"loss": 1.0})
    first.finish()

    lines: list = []
    handle = logger.add(lambda m: lines.append(m), level="INFO")
    try:
        second = HistoryStore()
        second.init(project="proj", name="job", dir=str(tmp_path))
        second.finish()
    finally:
        logger.remove(handle)
    assert any("resuming at step 1" in line for line in lines)


# ------------------------------------------------------------------ sharing


def test_runs_of_one_project_share_their_artifacts(run, tmp_path):
    """The nesting earns its keep: dedup across runs needs a project-level store."""
    source = tmp_path / "ckpt.pt"
    source.write_text("identical bytes")

    first = run(dir=str(tmp_path))
    logged = first.log_artifact(str(source), name="model")
    et.finish()

    second = et.init(
        project="proj", name="other", dir=str(tmp_path), backends=[], resume="allow"
    )
    try:
        again = second.log_artifact(str(source), name="model")
        assert again.version == logged.version  # deduplicated, not a new version
        assert second.use_artifact("model:latest").version == logged.version
    finally:
        et.finish()


def test_separate_run_dirs_do_not_share_artifacts(tmp_path):
    """The trade the escape hatch makes, stated plainly."""
    source = tmp_path / "ckpt.pt"
    source.write_text("identical bytes")
    versions = []
    for name in ("a", "b"):
        instance = et.init(
            project="proj",
            name=name,
            run_dir=str(tmp_path / name),
            backends=[],
            resume="allow",
        )
        try:
            versions.append(instance.log_artifact(str(source), name="model").version)
        finally:
            et.finish()
    assert versions == [0, 0]  # each store starts over
    assert (tmp_path / "a" / "artifacts").is_dir()
    assert (tmp_path / "b" / "artifacts").is_dir()


def test_a_named_run_dir_still_resumes(tmp_path):
    target = tmp_path / "exact"
    for _ in range(2):
        instance = et.init(
            project="proj", name="job", run_dir=str(target), backends=[], resume="allow"
        )
        try:
            instance.log({"loss": 1.0})
        finally:
            et.finish()
    rows = et.history(-1, run=str(target))
    assert [row["_step"] for row in rows] == [0, 1]


def test_streams_and_ranks_still_work_in_a_named_run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "2")
    target = tmp_path / "exact"
    instance = et.init(
        project="proj",
        name="job",
        run_dir=str(target),
        backends=[],
        stream="data",
        max_open_seconds=None,
    )
    try:
        instance.log({"rows": 1})
    finally:
        et.finish()
    assert (target / "metrics.data.rank2.jsonl").is_file()
