"""Timestamp isolation must preserve strict copy assertions, not hide failures."""

import os
import threading
from pathlib import Path

import _artifact_metadata as metadata
import pytest
from _artifact_metadata import (
    assert_exact_metadata,
)
from _artifact_metadata import (
    noatime_artifact_fs as noatime_artifact_fs,
)

from expr_tracker.artifacts import Artifact, ArtifactStore

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="POSIX permissions and nanosecond timestamps"
)
ACCESSED_NS = 1_234_567_889_987_654_321
MODIFIED_NS = 1_234_567_890_123_456_789


def publish(root):
    source = root / "source"
    source.write_bytes(b"weights-v1")
    artifact = Artifact("model").add_file(source)
    os.chmod(source, 0o640)
    os.utime(source, ns=(ACCESSED_NS, MODIFIED_NS))
    return source, ArtifactStore(root / "store").log(artifact, mode="copy")


def test_external_reads_cannot_change_isolated_copy_timestamps(noatime_artifact_fs):
    root = noatime_artifact_fs
    source, logged = publish(root)
    published = logged.get_path("source")
    ready, stopping = threading.Event(), threading.Event()
    errors = []

    def read():
        try:
            while not stopping.is_set():
                assert source.read_bytes() == published.read_bytes() == b"weights-v1"
                ready.set()
        except BaseException as error:
            errors.append(error)
            ready.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        assert ready.wait(5), "reader did not start"
        assert not errors
        for index in range(10):
            assert_exact_metadata(
                source, published, accessed_ns=ACCESSED_NS, modified_ns=MODIFIED_NS
            )
            downloaded = logged.download(root / f"download-{index}") / "source"
            assert downloaded.read_bytes() == b"weights-v1"
            assert_exact_metadata(
                published,
                downloaded,
                accessed_ns=ACCESSED_NS,
                modified_ns=MODIFIED_NS,
            )
    finally:
        stopping.set()
        reader.join(5)
    assert not reader.is_alive()
    assert not errors


@pytest.mark.parametrize("stage", ["publication", "download"])
@pytest.mark.parametrize("mutation", ["omit_atime", "truncate_nanoseconds"])
def test_exact_timestamp_oracle_rejects_bad_copies(
    noatime_artifact_fs, stage, mutation
):
    root = noatime_artifact_fs
    source, logged = publish(root)
    target = logged.get_path("source")
    if stage == "download":
        source = target
        target = logged.download(root / "download") / "source"
    assert_exact_metadata(
        source, target, accessed_ns=ACCESSED_NS, modified_ns=MODIFIED_NS
    )
    corrupted_ns = (
        0 if mutation == "omit_atime" else ACCESSED_NS // 1_000_000_000 * 1_000_000_000
    )
    os.utime(target, ns=(corrupted_ns, MODIFIED_NS))
    with pytest.raises(AssertionError, match="access time"):
        assert_exact_metadata(
            source, target, accessed_ns=ACCESSED_NS, modified_ns=MODIFIED_NS
        )


def test_timestamp_isolation_failure_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata.sys, "platform", "linux")
    monkeypatch.setattr(metadata, "_reads_preserve_atime", lambda root: False)

    def forbidden(path):
        raise PermissionError("no inode flags")

    monkeypatch.setattr(metadata, "_set_noatime", forbidden)
    with pytest.raises(RuntimeError, match="noatime"):
        metadata._prepare_noatime(tmp_path)


def test_timestamp_isolation_checks_child_inheritance(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata.sys, "platform", "linux")
    monkeypatch.setattr(metadata, "_reads_preserve_atime", lambda root: False)
    monkeypatch.setattr(metadata, "_set_noatime", lambda path: None)
    with pytest.raises(RuntimeError, match="noatime"):
        metadata._prepare_noatime(tmp_path)


def test_timestamp_isolation_only_changes_its_private_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata.sys, "platform", "linux")
    root = tmp_path / "owned"
    root.mkdir()
    child = root / "nested"
    child.mkdir()
    sample = child / "sample"
    sample.write_bytes(b"owned")
    outside = tmp_path / "outside"
    outside.write_bytes(b"unrelated")
    probes = iter([False, True])
    changed = []
    monkeypatch.setattr(metadata, "_reads_preserve_atime", lambda root: next(probes))
    monkeypatch.setattr(metadata, "_set_noatime", changed.append)
    metadata._prepare_noatime(root)
    assert set(changed) == {root, child, sample}
    assert outside.read_bytes() == b"unrelated"
    assert all(Path(path).is_relative_to(root) for path in changed)
