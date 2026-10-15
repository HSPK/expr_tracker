"""Forked children discard inherited clients without touching parent locks/files."""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")

PRELUDE = """
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
import expr_tracker as et

root = Path(sys.argv[1])

def wait_child(pid, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        waited, status = os.waitpid(pid, os.WNOHANG)
        if waited:
            assert os.waitstatus_to_exitcode(status) == 0, status
            return
        time.sleep(0.01)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    raise AssertionError("forked child hung")

def rejected(call):
    try:
        call()
    except RuntimeError as error:
        assert "after fork" in str(error), str(error)
    else:
        raise AssertionError("inherited object was accepted")
"""


def run_fork_script(tmp_path, source):
    result = subprocess.run(
        [sys.executable, "-c", PRELUDE + textwrap.dedent(source), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_child_initializes_independently_and_does_not_flush_parent_at_exit(tmp_path):
    run_fork_script(
        tmp_path,
        """
        from expr_tracker.plugins import CpuTime
        parent = et.init(project="p", run_dir=str(root / "parent"), backends=[])
        et.log({"committed": 1})
        et.log({"pending": 2}, commit=False)
        parent.summary["marker"] = "saved"
        parent.summary.save()
        parent.summary["marker"] = "unsaved"
        plugin = CpuTime()
        outer = et.start_span("outer", plugins=[plugin])
        pid = os.fork()
        if pid == 0:
            assert et.get_run() is None
            rejected(lambda: parent.log({"child": 1}))
            rejected(lambda: parent.history.get(-1))
            rejected(parent.history.flush)
            rejected(parent.history.stats)
            rejected(parent.history.writer.flush)
            rejected(parent.summary.save)
            rejected(parent.finish)
            rejected(outer.end)
            rejected(lambda: plugin.start(outer))
            child = et.init(project="p", run_dir=str(root / "child"), backends=[])
            with et.span("work", plugins=[CpuTime()]):
                pass
            et.finish()
            records = [json.loads(line) for line in child.history.spans_fp.read_text().splitlines()]
            assert records[0]["name"] == "work" and records[0]["depth"] == 0
            sys.exit(0)
        wait_child(pid)
        assert len(parent.history.log_fp.read_text().splitlines()) == 1
        assert json.loads(parent.summary.path.read_text())["marker"] == "saved"
        assert et.get_run() is parent
        outer.end()
        et.finish()
        assert len(parent.history.log_fp.read_text().splitlines()) == 2
        assert json.loads(parent.summary.path.read_text())["marker"] == "unsaved"
        """,
    )


def test_child_does_not_acquire_inherited_locked_state(tmp_path):
    run_fork_script(
        tmp_path,
        """
        from contextlib import ExitStack
        from types import SimpleNamespace
        from expr_tracker import alerts, plugins, tracker
        import expr_tracker.run as run_module
        parent = et.init(project="p", run_dir=str(root / "parent"), backends=[])
        with et.span("warm"):
            pass
        parent.history.flush()
        cpu = plugins.CpuTime()
        locks = [
            run_module._lock, tracker._lifecycle, alerts._lock,
            plugins._SAMPLERS_LOCK, plugins._TORCH_LOCK,
            parent.history._write_lock, parent.history._lock,
            parent._finish_lock, cpu._lock,
        ]
        ready, release = threading.Event(), threading.Event()
        def hold():
            with ExitStack() as stack:
                for lock in locks:
                    stack.enter_context(lock)
                ready.set()
                release.wait(10)
        holder = threading.Thread(target=hold)
        holder.start()
        assert ready.wait(5)
        pid = os.fork()
        if pid == 0:
            assert et.get_run() is None
            rejected(parent.finish)
            rejected(parent.history.flush)
            rejected(parent.history.span_writer.flush)
            rejected(parent.summary.save)
            rejected(lambda: cpu.start(SimpleNamespace(duration_ms=1)))
            child = et.init(project="p", run_dir=str(root / "child"), backends=[])
            plugins.GpuStats(interval=0.2)
            with et.span("independent", plugins=[plugins.CpuTime()]):
                pass
            et.finish()
            sys.exit(0)
        try:
            wait_child(pid)
        finally:
            release.set()
            holder.join(5)
            et.finish()
        assert not holder.is_alive()
        """,
    )


def test_child_exit_does_not_wait_for_inherited_alert_queue(tmp_path):
    run_fork_script(
        tmp_path,
        """
        entered, release = threading.Event(), threading.Event()
        def handler(message):
            entered.set()
            release.wait(10)
        parent = et.init(
            project="p", run_dir=str(root / "parent"), backends=[],
            alert={"channels": [{
                "type": "callable", "name": "sink", "options": {"handler": handler},
                "policy": {"async_send": True},
            }]},
        )
        et.alert("parent alert")
        assert entered.wait(5)
        pid = os.fork()
        if pid == 0:
            sys.exit(0)
        try:
            wait_child(pid, timeout=3)
        finally:
            release.set()
            et.finish()
        """,
    )


def test_child_does_not_inherit_a_writer_diagnostic_context(tmp_path):
    run_fork_script(
        tmp_path,
        """
        from loguru import logger
        from expr_tracker.history import writer as writer_module
        logger.remove()
        # Model a fork after entering diagnostic delivery but before a sink lock.
        writer_module._diagnostic_emission.active = True
        pid = os.fork()
        if pid == 0:
            assert not writer_module.in_diagnostic_callback()
            child = et.init(project="p", run_dir=str(root / "child"), backends=[])
            with et.span("child"):
                pass
            et.finish()
            os._exit(0)
        writer_module._diagnostic_emission.active = False
        wait_child(pid)
        assert (root / "child" / "spans.jsonl").is_file()
        """,
    )
