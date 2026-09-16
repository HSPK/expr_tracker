"""Reusable writer workers, bounded async buffering, and shutdown barriers."""

import json
import threading
import time

import pytest

from expr_tracker import _process
from expr_tracker.history import writer as writer_module
from expr_tracker.history.writer import JsonlWriter


def line_for(step):
    return json.dumps({"_step": step}).encode() + b"\n"


def steps_on_disk(writer):
    if not writer.path.exists():
        return []
    return [json.loads(line)["_step"] for line in writer.path.read_bytes().splitlines()]


@pytest.fixture
def writer(tmp_path):
    instances = []

    def make(**kwargs):
        instance = JsonlWriter(tmp_path / f"records-{len(instances)}.jsonl", **kwargs)
        instances.append(instance)
        return instance

    yield make
    for instance in instances:
        instance.close()


def test_synchronous_enqueue_keeps_its_flush_due_contract(writer):
    instance = writer(buffer_size=2, buffer_interval=None)
    assert instance._worker is None
    assert instance.enqueue(0, 0, line_for(0)) is True
    assert instance._worker is None
    assert not instance.path.exists()
    instance.flush()
    assert steps_on_disk(instance) == [0]
    assert instance.enqueue(1, 1, line_for(1)) is False
    assert instance.enqueue(2, 2, line_for(2)) is True
    instance.flush()
    assert steps_on_disk(instance) == [0, 1, 2]
    assert instance._worker is None


@pytest.mark.parametrize("max_buffer_seconds", [None, 5.0])
def test_synchronous_worker_is_lazy(writer, max_buffer_seconds):
    instance = writer(
        buffer_size=50,
        buffer_interval=None,
        max_buffer_seconds=max_buffer_seconds,
    )
    assert instance._worker is None
    instance.append(0, 0, line_for(0))
    assert steps_on_disk(instance) == [0]
    assert instance._worker is None
    instance.append(1, 1, line_for(1))
    if max_buffer_seconds is None:
        assert instance._worker is None
    else:
        assert instance._worker.is_alive()
    instance.close()
    assert steps_on_disk(instance) == [0, 1]
    if instance._worker is not None:
        assert not instance._worker.is_alive()


@pytest.mark.parametrize("async_write", [False, True])
def test_six_thousand_appends_use_one_worker(writer, monkeypatch, async_write):
    instance = writer(async_write=async_write)
    created = []
    original_start = threading.Thread.start

    def record_start(thread):
        if thread.name.startswith("expr-tracker-writer-"):
            created.append(thread)
        return original_start(thread)

    def reject_timer(*args, **kwargs):
        pytest.fail("writer must not create per-batch Timer threads")

    monkeypatch.setattr(threading.Thread, "start", record_start)
    monkeypatch.setattr(threading, "Timer", reject_timer)
    for row in range(6000):
        instance.append(row, row, line_for(row))
    instance.close()
    assert created == [instance._worker]
    assert not instance._worker.is_alive()
    assert steps_on_disk(instance) == list(range(6000))
    assert instance.lines == 6000
    assert instance.dropped == 0


@pytest.mark.parametrize("async_write", [False, True])
def test_idle_deadlines_reuse_the_same_worker(writer, monkeypatch, async_write):
    instance = writer(
        buffer_size=1000,
        buffer_interval=None,
        max_buffer_seconds=0.01,
        async_write=async_write,
    )
    flushed = threading.Event()
    original_flush = instance.flush
    expected_lines = 2

    def notify_flush():
        original_flush()
        if instance.lines == expected_lines:
            flushed.set()

    monkeypatch.setattr(instance, "flush", notify_flush)
    instance.append(0, 0, line_for(0))
    instance.flush()
    worker = None
    for row in range(1, 6):
        expected_lines = row + 1
        flushed.clear()
        instance.append(row, row, line_for(row))
        assert flushed.wait(3), "idle records never reached their deadline"
        assert steps_on_disk(instance) == list(range(row + 1))
        if worker is None:
            worker = instance._worker
        assert instance._worker is worker
    instance.close()
    assert not worker.is_alive()


def test_async_enqueue_and_append_do_not_wait_for_disk(writer, monkeypatch):
    instance = writer(
        buffer_size=2, buffer_interval=None, max_buffer_seconds=None, async_write=True
    )
    writing, release, produced = (threading.Event() for _ in range(3))
    original_write = instance._write
    writers = []

    def slow_write(payload):
        writers.append(threading.current_thread())
        writing.set()
        assert release.wait(5)
        original_write(payload)

    def produce():
        instance.append(1, 1, line_for(1))
        assert instance.enqueue(2, 2, line_for(2)) is True
        produced.set()

    monkeypatch.setattr(instance, "_write", slow_write)
    producer = threading.Thread(target=produce)
    try:
        assert instance.enqueue(0, 0, line_for(0)) is True
        assert writing.wait(3)
        producer.start()
        assert produced.wait(1), "producer waited for the worker's disk I/O"
        assert instance.lines == 0
    finally:
        release.set()
        if producer.ident is not None:
            producer.join(3)
    instance.flush()
    assert steps_on_disk(instance) == [0, 1, 2]
    assert writers[0] is instance._worker
    assert producer not in writers


def test_async_flush_waits_for_inflight_and_queued_records(writer, monkeypatch):
    instance = writer(
        buffer_size=100, buffer_interval=None, max_buffer_seconds=None, async_write=True
    )
    writing, release, flushing, flushed = (threading.Event() for _ in range(4))
    original_write = instance._write

    def slow_write(payload):
        writing.set()
        assert release.wait(5)
        original_write(payload)

    def flush():
        flushing.set()
        instance.flush()
        flushed.set()

    monkeypatch.setattr(instance, "_write", slow_write)
    flusher = threading.Thread(target=flush)
    try:
        instance.append(0, 0, line_for(0))
        assert writing.wait(3)
        for row in range(1, 20):
            instance.append(row, row, line_for(row))
        flusher.start()
        assert flushing.wait(3)
        assert not flushed.wait(0.03)
    finally:
        release.set()
        if flusher.ident is not None:
            flusher.join(3)
    assert flushed.is_set()
    assert not flusher.is_alive()
    assert steps_on_disk(instance) == list(range(20))
    assert instance.flushed_row == 19
    assert instance.buffer == []


@pytest.mark.parametrize("async_write", [False, True])
def test_close_rejects_new_records_and_waits_for_all_closers(
    writer, monkeypatch, async_write
):
    instance = writer(buffer_size=1, async_write=async_write)
    writing, release = (threading.Event() for _ in range(2))
    original_write = instance._write
    writes = []

    def slow_write(payload):
        writes.append(payload)
        writing.set()
        assert release.wait(5)
        original_write(payload)

    monkeypatch.setattr(instance, "_write", slow_write)
    producer = threading.Thread(target=lambda: instance.append(0, 0, line_for(0)))
    closers = [threading.Thread(target=instance.close) for _ in range(2)]
    try:
        producer.start()
        assert writing.wait(3)
        instance.enqueue(1, 1, line_for(1))
        closers[0].start()
        with instance._condition:
            assert instance._condition.wait_for(lambda: instance._closing, timeout=3)
        closers[1].start()
        assert instance.enqueue(2, 2, line_for(2)) is False
        instance.append(3, 3, line_for(3))
        for closer in closers:
            closer.join(0.03)
            assert closer.is_alive()
        assert not instance._closed
    finally:
        release.set()
        producer.join(3)
        for closer in closers:
            if closer.ident is not None:
                closer.join(3)
    assert all(not thread.is_alive() for thread in [producer, *closers])
    assert instance._closed
    assert steps_on_disk(instance) == [0, 1]
    meta = json.loads(instance.meta_path.read_text())
    assert meta["lines"] == 2
    assert meta["size"] == instance.path.stat().st_size
    instance.close()
    instance.flush()
    instance.schedule_timer()
    assert instance.enqueue(4, 4, line_for(4)) is False
    instance.append(5, 5, line_for(5))
    assert len(writes) == 2
    assert instance.buffer == []
    if instance._worker is not None:
        assert not instance._worker.is_alive()


def test_async_append_does_not_wait_for_metadata_io(writer, monkeypatch):
    instance = writer(async_write=True, buffer_size=1)
    saving, release, produced = (threading.Event() for _ in range(3))
    original_save = instance._save_meta

    def slow_meta(force=False):
        saving.set()
        assert release.wait(5)
        original_save(force=force)

    def produce():
        for row in range(1, 100):
            instance.append(row, row, line_for(row))
        produced.set()

    monkeypatch.setattr(instance, "_save_meta", slow_meta)
    producer = threading.Thread(target=produce)
    try:
        instance.append(0, 0, line_for(0))
        assert saving.wait(3)
        producer.start()
        assert produced.wait(1), "producer waited for a sidecar write"
    finally:
        release.set()
        if producer.ident is not None:
            producer.join(3)
    instance.close()
    assert steps_on_disk(instance) == list(range(100))
    assert json.loads(instance.meta_path.read_text())["lines"] == 100


def test_metadata_writes_are_serialized_with_data_flushes(writer, monkeypatch):
    instance = writer(max_buffer_seconds=None)
    saving, release, second_write, second_flush = (
        threading.Event() for _ in range(4)
    )
    original_save, original_write = instance._save_meta, instance._write

    def slow_meta(force=False):
        saving.set()
        assert release.wait(5)
        original_save(force=force)

    def write(payload):
        if payload == line_for(1):
            second_write.set()
        original_write(payload)

    def flush_second():
        second_flush.set()
        instance.flush()

    monkeypatch.setattr(instance, "_save_meta", slow_meta)
    monkeypatch.setattr(instance, "_write", write)
    instance.enqueue(0, 0, line_for(0))
    first = threading.Thread(target=instance.flush)
    second = threading.Thread(target=flush_second)
    try:
        first.start()
        assert saving.wait(3)
        instance.enqueue(1, 1, line_for(1))
        second.start()
        assert second_flush.wait(3)
        assert not second_write.wait(0.03)
    finally:
        release.set()
        first.join(3)
        if second.ident is not None:
            second.join(3)
    assert not first.is_alive() and not second.is_alive()
    instance.close()
    assert steps_on_disk(instance) == [0, 1]
    assert json.loads(instance.meta_path.read_text())["lines"] == 2


@pytest.mark.parametrize("async_write", [False, True])
def test_concurrent_producers_and_flushes_preserve_admission_order(writer, async_write):
    instance = writer(
        buffer_size=7,
        buffer_interval=None,
        max_buffer_seconds=0.005,
        async_write=async_write,
        index_every=10,
    )
    allocation = threading.Lock()
    ready = threading.Barrier(6)
    accepted = []

    def produce():
        ready.wait()
        for _ in range(150):
            with allocation:
                row = len(accepted)
                accepted.append(row)
                instance.append(row, row, line_for(row))

    def flush():
        ready.wait()
        for _ in range(100):
            instance.flush()

    threads = [threading.Thread(target=produce) for _ in range(4)]
    threads.append(threading.Thread(target=flush))
    for thread in threads:
        thread.start()
    ready.wait()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    instance.close()
    assert steps_on_disk(instance) == accepted == list(range(600))
    assert instance.sorted and not instance.has_duplicate_steps
    assert instance.flushed_row == 599
    assert [row["_step"] for row in instance.reader().read_steps(275, 285)] == list(
        range(275, 285)
    )


@pytest.mark.parametrize("inflight", [1, 4])
@pytest.mark.parametrize("recovered", [False, True])
def test_healthy_saturation_waits_without_dropping_records(
    writer, monkeypatch, inflight, recovered
):
    instance = writer(
        buffer_size=1, max_pending_records=4, max_buffer_seconds=None, async_write=True
    )
    writing, release, waiting, produced = (threading.Event() for _ in range(4))
    original_write = instance._write
    original_wait = instance._condition.wait
    errors = []

    if recovered:
        def fail(payload):
            raise OSError("temporary failure")

        with monkeypatch.context() as patch:
            patch.setattr(instance, "_write", fail)
            instance.enqueue(-1, -1, line_for(-1))
            instance.flush()
        instance.flush()
        assert instance._retry_deadline is None

    def slow_write(payload):
        writing.set()
        assert release.wait(5)
        original_write(payload)

    def wait(timeout=None):
        if threading.current_thread() is producer:
            waiting.set()
        return original_wait(timeout)

    def produce():
        for row in range(4, 30):
            instance.append(row, row, line_for(row))
            with instance._lock:
                assert len(instance.buffer) + instance._inflight_records <= 4
        produced.set()

    producer = threading.Thread(target=produce)
    monkeypatch.setattr(instance, "_write", slow_write)
    monkeypatch.setattr(instance._condition, "wait", wait)
    monkeypatch.setattr(writer_module.logger, "error", errors.append)
    try:
        with instance._lock:
            for row in range(inflight):
                instance.append(row, row, line_for(row))
        assert writing.wait(3)
        for row in range(inflight, 4):
            instance.append(row, row, line_for(row))
        producer.start()
        assert waiting.wait(3), "producer did not apply bounded backpressure"
        assert not produced.is_set()
        with instance._lock:
            assert len(instance.buffer) + instance._inflight_records == 4
    finally:
        release.set()
        if producer.ident is not None:
            producer.join(3)
    assert produced.is_set()
    assert not producer.is_alive()
    instance.close()
    assert steps_on_disk(instance) == ([-1] if recovered else []) + list(range(30))
    assert instance.dropped == 0
    assert errors == []


def test_close_wakes_and_rejects_blocked_admissions(writer, monkeypatch):
    instance = writer(buffer_size=1, max_pending_records=2, async_write=True)
    writing, release, waiting = (threading.Event() for _ in range(3))
    original_write = instance._write
    original_wait = instance._condition.wait
    blocked = set()
    results = []
    warnings = []

    def slow_write(payload):
        writing.set()
        assert release.wait(5)
        original_write(payload)

    def wait(timeout=None):
        thread = threading.current_thread()
        if thread in producers:
            blocked.add(thread)
            if len(blocked) == len(producers):
                waiting.set()
        return original_wait(timeout)

    def produce(row):
        results.append(instance.enqueue(row, row, line_for(row)))

    producers = [threading.Thread(target=produce, args=(row,)) for row in range(2, 5)]
    closer = threading.Thread(target=instance.close)
    monkeypatch.setattr(instance, "_write", slow_write)
    monkeypatch.setattr(instance._condition, "wait", wait)
    monkeypatch.setattr(writer_module.logger, "warning", warnings.append)
    try:
        instance.append(0, 0, line_for(0))
        assert writing.wait(3)
        instance.append(1, 1, line_for(1))
        for producer in producers:
            producer.start()
        assert waiting.wait(3)
        closer.start()
        for producer in producers:
            producer.join(3)
            assert not producer.is_alive(), "shutdown stranded a blocked admission"
        assert results == [False, False, False]
        assert instance.enqueue(5, 5, line_for(5)) is False
        assert len(warnings) == 4
        assert closer.is_alive(), "close did not wait for its in-flight write"
    finally:
        release.set()
        for thread in [*producers, closer]:
            if thread.ident is not None:
                thread.join(3)
    assert not closer.is_alive()
    assert instance._closed and not instance._worker.is_alive()
    assert steps_on_disk(instance) == [0, 1]
    assert instance.dropped == 0
    assert instance.buffer == []


def test_failed_writes_wake_blocked_producers_and_bound_retry_drops(writer, monkeypatch):
    instance = writer(
        buffer_size=1, max_pending_records=4, max_buffer_seconds=30, async_write=True
    )
    writing, release, waiting, produced = (threading.Event() for _ in range(4))
    original_wait = instance._condition.wait
    attempts = []
    errors = []

    def fail(payload):
        attempts.append(payload)
        writing.set()
        assert release.wait(5)
        raise OSError("disk stays broken")

    def wait(timeout=None):
        if threading.current_thread() is producer:
            waiting.set()
        return original_wait(timeout)

    def produce():
        for row in range(4, 14):
            instance.append(row, row, line_for(row))
            with instance._lock:
                assert len(instance.buffer) + instance._inflight_records <= 4
        produced.set()

    producer = threading.Thread(target=produce)
    monkeypatch.setattr(instance, "_write", fail)
    monkeypatch.setattr(instance._condition, "wait", wait)
    monkeypatch.setattr(writer_module.logger, "error", errors.append)
    try:
        with instance._lock:
            for row in range(4):
                instance.append(row, row, line_for(row))
        assert writing.wait(3)
        producer.start()
        assert waiting.wait(3)
        assert instance.dropped == 0
    finally:
        release.set()
        if producer.ident is not None:
            producer.join(3)
    assert produced.is_set(), "a failed write did not release healthy backpressure"
    assert not producer.is_alive()
    assert len(instance.buffer) == 4
    assert [row for _, row, _ in instance.buffer] == [10, 11, 12, 13]
    assert instance.dropped == 10
    assert sum("dropped 1 oldest queued records" in error for error in errors) == 10
    instance.close()
    assert len(attempts) == 2
    assert not instance._worker.is_alive()
    assert len(instance.buffer) == 4


def test_saturated_enqueue_only_writer_makes_progress_without_deadlines(writer):
    instance = writer(buffer_size=2, max_pending_records=2, max_buffer_seconds=None)
    produced = threading.Event()

    def produce():
        for row in range(20):
            instance.enqueue(row, row, line_for(row))
        produced.set()

    producer = threading.Thread(target=produce)
    producer.start()
    producer.join(3)
    assert not producer.is_alive(), "capacity waits never requested a flush"
    assert produced.is_set()
    instance.close()
    assert steps_on_disk(instance) == list(range(20))
    assert instance.dropped == 0


def test_async_failed_partial_write_retries_in_order(writer, monkeypatch):
    instance = writer(
        buffer_size=2, max_buffer_seconds=0.01, async_write=True, index_every=1
    )
    failed, recovered = (threading.Event() for _ in range(2))
    original_write = instance._write
    calls = []

    def fail_once(payload):
        calls.append(payload)
        if len(calls) == 1:
            with open(instance.path, "ab") as stream:
                stream.write(payload[: len(payload) // 2])
            failed.set()
            raise OSError("partial write")
        original_write(payload)
        recovered.set()

    monkeypatch.setattr(instance, "_write", fail_once)
    with instance._lock:
        instance.append(0, 0, line_for(0))
        instance.append(1, 1, line_for(1))
    assert failed.wait(3)
    instance.append(2, 2, line_for(2))
    assert recovered.wait(3), "failed batch was never retried by the worker"
    instance.close()
    assert steps_on_disk(instance) == [0, 1, 2]
    assert instance.lines == 3
    assert instance.flushed_row == 2
    assert instance.dropped == 0
    assert not instance.has_duplicate_steps
    assert [entry[1] for entry in instance.index] == [0, 1, 2]


@pytest.mark.parametrize("max_buffer_seconds", [None, 0.0])
def test_permanent_failure_does_not_spin_or_retry_after_close(
    writer, monkeypatch, max_buffer_seconds
):
    instance = writer(async_write=True, max_buffer_seconds=max_buffer_seconds)
    failed = threading.Event()
    calls = []
    errors = []

    def fail(payload):
        calls.append(time.monotonic())
        failed.set()
        raise OSError("disk stays broken")

    monkeypatch.setattr(instance, "_write", fail)
    monkeypatch.setattr(writer_module.logger, "error", errors.append)
    instance.append(0, 0, line_for(0))
    assert failed.wait(3)
    time.sleep(0.03)
    assert len(calls) == 1, "worker hot-looped on a broken disk"
    instance.close()
    assert len(calls) == 2
    assert not instance._worker.is_alive()
    assert len(instance.buffer) == 1
    assert instance.lines == 0
    assert any("1 unwritten records" in error for error in errors)
    instance.flush()
    instance.close()
    assert len(calls) == 2


def test_empty_close_creates_no_worker_and_is_idempotent(writer):
    instance = writer(async_write=True)
    instance.close()
    instance.close()
    assert instance._worker is None
    assert instance._closed
    assert instance.enqueue(0, 0, line_for(0)) is False
    assert not instance.path.exists()


def test_flush_cannot_retry_after_the_final_close_barrier(writer, monkeypatch):
    instance = writer(max_buffer_seconds=None)
    saved, released, flushed = (threading.Event() for _ in range(3))
    original_lock, original_save = instance._write_lock, instance._save_meta
    attempts = []

    def fail(payload):
        attempts.append(payload)
        raise OSError("unwritable disk")

    def save(force=False):
        original_save(force=force)
        if force:
            saved.set()

    class PausingLock:
        def __enter__(self):
            original_lock.acquire()

        def __exit__(self, *args):
            original_lock.release()
            if threading.current_thread() is closer and saved.is_set():
                released.set()
                assert flushed.wait(3)

    def flush_after_final_save():
        assert released.wait(3)
        instance.flush()
        flushed.set()

    monkeypatch.setattr(instance, "_write", fail)
    monkeypatch.setattr(instance, "_save_meta", save)
    monkeypatch.setattr(instance, "_write_lock", PausingLock())
    instance.enqueue(0, 0, line_for(0))
    closer = threading.Thread(target=instance.close)
    flusher = threading.Thread(target=flush_after_final_save)
    closer.start()
    flusher.start()
    closer.join(5)
    flusher.join(5)
    assert not closer.is_alive() and not flusher.is_alive()
    assert len(attempts) == 1
    assert instance._closed


@pytest.mark.parametrize(
    "entry",
    [
        "enqueue", "enqueue_with_status", "append", "flush", "close",
        "schedule_timer", "reader", "buffer", "queued_bytes", "inflight_bytes",
        "pending_bytes",
    ],
)
def test_inherited_writer_rejects_public_calls_before_touching_locks(
    writer, monkeypatch, entry
):
    instance = writer(max_buffer_seconds=None)
    owner_pid = _process.process_id()
    assert instance._owner_pid == owner_pid
    instance.enqueue(0, 0, line_for(0))
    buffered = instance.buffer
    queued_bytes = instance.pending_bytes

    class InheritedLock:
        def __enter__(self):
            pytest.fail("inherited writer touched a lock before checking its process")

        def __exit__(self, *args):
            pytest.fail("inherited writer touched a lock before checking its process")

    with monkeypatch.context() as patch:
        patch.setattr(_process, "_PID", owner_pid + 1)
        for name in ("_lock", "_condition", "_write_lock"):
            patch.setattr(instance, name, InheritedLock())
        with pytest.raises(RuntimeError, match="after fork"):
            public_entry = getattr(instance, entry)
            if entry in ("enqueue", "enqueue_with_status", "append"):
                public_entry(1, 1, line_for(1))
            elif callable(public_entry):
                public_entry()

    assert instance._owner_pid == owner_pid
    assert instance.buffer == buffered
    assert instance.pending_bytes == queued_bytes
    assert instance.dropped == instance.dropped_bytes == 0
    assert instance.flushed_row == -1
    assert not instance._closing
    assert not instance.path.exists()
    instance.close()
    assert steps_on_disk(instance) == [0]


def test_writer_process_guards_use_cached_pid_without_syscalls(writer, monkeypatch):
    def reject_getpid():
        pytest.fail("writer hot paths must use the cached process identity")

    with monkeypatch.context() as patch:
        patch.setattr(writer_module.os, "getpid", reject_getpid)
        instance = writer(max_buffer_seconds=None)
        for row in range(10):
            instance.append(row, row, line_for(row))
        instance.flush()
        instance.schedule_timer()
        assert len(instance.reader().read_all()) == 10
        assert instance.buffer == []
        assert instance.queued_bytes == instance.inflight_bytes == instance.pending_bytes == 0
        instance.close()
