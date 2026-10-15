"""Read-stable timestamp checks on private, pytest-owned files."""

import array
import os
import stat
import sys
import tempfile
from pathlib import Path

import pytest


def _reads_preserve_atime(root):
    with tempfile.TemporaryDirectory(prefix="atime-probe-", dir=root) as directory:
        path = Path(directory) / "sample"
        path.write_bytes(b"probe")
        timestamp = 1_234_567_888_222_333_444
        os.utime(path, ns=(timestamp, timestamp + 1_000_000_000))
        path.read_bytes()
        return path.stat().st_atime_ns == timestamp


def _set_noatime(path):
    import fcntl

    # Linux FS_IOC_GETFLAGS/SETFLAGS encode the native sizeof(long).
    size = array.array("L").itemsize
    get_flags = (2 << 30) | (size << 16) | (ord("f") << 8) | 1
    set_flags = (1 << 30) | (size << 16) | (ord("f") << 8) | 2
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        flags = array.array("L", [0])
        fcntl.ioctl(descriptor, get_flags, flags, True)
        flags[0] |= 0x80  # FS_NOATIME_FL, inherited by ext4 children.
        fcntl.ioctl(descriptor, set_flags, flags, True)
    finally:
        os.close(descriptor)


def _prepare_noatime(root):
    if _reads_preserve_atime(root):
        return
    if sys.platform != "linux":
        raise RuntimeError(
            "Exact artifact timestamps require a noatime test filesystem; "
            "configure pytest --basetemp on one."
        )
    try:
        for path in [root, *sorted(root.rglob("*"))]:
            _set_noatime(path)
    except OSError as error:
        raise RuntimeError(
            "Cannot set noatime on private test files; use a noatime filesystem "
            "for pytest --basetemp."
        ) from error
    if not _reads_preserve_atime(root):
        raise RuntimeError(
            "Test files did not inherit noatime; configure pytest --basetemp "
            "on a filesystem supporting noatime."
        )


@pytest.fixture
def noatime_artifact_fs(tmp_path):
    _prepare_noatime(tmp_path)
    return tmp_path


def assert_exact_metadata(source, target, *, accessed_ns, modified_ns):
    expected, actual = source.stat(), target.stat()
    assert stat.S_IMODE(actual.st_mode) == stat.S_IMODE(expected.st_mode) == 0o640, (
        "permissions were not preserved"
    )
    assert actual.st_mtime_ns == expected.st_mtime_ns == modified_ns, (
        "modification time was not preserved exactly"
    )
    assert actual.st_atime_ns == expected.st_atime_ns == accessed_ns, (
        "access time was not preserved exactly"
    )
