"""CPU-bound cached readers must not starve a metric producer."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import expr_tracker


def test_four_cached_readers_do_not_starve_writer(tmp_path):
    source = Path(expr_tracker.__file__).resolve().parent.parent
    script = textwrap.dedent(
        """
        import faulthandler
        import os
        import sys
        import threading

        faulthandler.dump_traceback_later(8, exit=True)
        sys.path.insert(0, sys.argv[2])
        from expr_tracker.history import HistoryStore
        from loguru import logger
        logger.remove()
        cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        if cpus:
            os.sched_setaffinity(0, {cpus[-1]})
        store = HistoryStore()
        store.init(project="test", run_dir=sys.argv[1], buffer_size=200)
        metrics = {"loss": 0.5, "acc": 0.9, "lr": 0.001, "norm": 1.2, "time": 0.03}
        for _ in range(100):
            store.log(metrics)
        stop = threading.Event()
        ready = threading.Barrier(5)
        reads = [0] * 4
        errors = []
        def reader(index):
            if len(cpus) > 1:
                os.sched_setaffinity(0, {cpus[index % (len(cpus) - 1)]})
            ready.wait()
            try:
                while not stop.is_set():
                    rows = store.get(100)
                    assert len(rows) == 100
                    assert rows == sorted(rows, key=lambda row: row["_step"])
                    reads[index] += 1
            except Exception as error:
                errors.append(repr(error))
                stop.set()
        readers = [threading.Thread(target=reader, args=(index,)) for index in range(4)]
        for thread in readers:
            thread.start()
        ready.wait()
        try:
            for _ in range(1000):
                assert store.log(metrics) is not None
        finally:
            stop.set()
            for thread in readers:
                thread.join(2)
        assert all(not thread.is_alive() for thread in readers)
        assert not errors, errors
        assert sum(reads) > 0
        store.finish()
        assert store.stats()["rows_logged"] == 1100
        assert store.stats()["rows_on_disk"] == 1100
        faulthandler.cancel_dump_traceback_later()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(source)],
        capture_output=True,
        text=True,
        timeout=12,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("phase", ["write", "rollback"])
def test_io_error_formatting_can_inspect_writer_without_lock_inversion(tmp_path, phase):
    source = Path(expr_tracker.__file__).resolve().parent.parent
    script = textwrap.dedent(
        """
        import faulthandler
        import sys

        faulthandler.dump_traceback_later(8, exit=True)
        sys.path.insert(0, sys.argv[2])
        from expr_tracker.history.writer import JsonlWriter
        from loguru import logger
        logger.remove()
        writer = JsonlWriter(
            sys.argv[1] + "/metrics.jsonl", buffer_size=1, max_buffer_seconds=None
        )
        observed = []
        class ReentrantError(OSError):
            def __str__(self):
                observed.append(writer.pending_bytes)
                return "injected error"
        def fail_write(payload):
            if sys.argv[3] == "write":
                raise ReentrantError()
            raise OSError("injected write error")
        def fail_rollback(size):
            raise ReentrantError()
        writer._write = fail_write
        if sys.argv[3] == "rollback":
            writer._truncate_partial_write = fail_rollback
        assert writer.append_with_status(0, 0, b'{"_step":0}\\n')
        assert observed and all(size > 0 for size in observed)
        del writer._write
        if sys.argv[3] == "rollback":
            del writer._truncate_partial_write
        writer.flush()
        writer.close()
        assert writer.lines == 1
        faulthandler.cancel_dump_traceback_later()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(source), phase],
        capture_output=True,
        text=True,
        timeout=12,
    )
    assert result.returncode == 0, result.stdout + result.stderr
