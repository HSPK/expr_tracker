"""Byte capacity, admission status, failure accounting, and bounded-memory stress."""

import gc
import json
import statistics
import sys
import threading
import time
import tracemalloc

import pytest

from expr_tracker.history import writer as writer_module
from expr_tracker.history.writer import (
    DEFAULT_MAX_PENDING_BYTES,
    DEFAULT_MAX_RECORD_BYTES,
    JsonlWriter,
)


def line_for(step, size):
    prefix = json.dumps({"_step": step}).encode()
    assert len(prefix) < size
    return prefix + b" " * (size - len(prefix) - 1) + b"\n"


def steps_on_disk(instance):
    return [
        json.loads(line)["_step"] for line in instance.path.read_bytes().splitlines()
    ]


def assert_bytes(instance, queued, inflight):
    with instance._lock:
        assert instance.queued_bytes == queued
        assert instance.inflight_bytes == inflight
        assert instance.pending_bytes == queued + inflight
        assert sum(len(line) for _, _, line in instance.buffer) == queued
        assert len(instance.buffer) + instance._inflight_records <= instance.max_pending_records
        if instance.max_pending_bytes is not None:
            assert instance.pending_bytes <= instance.max_pending_bytes


@pytest.fixture
def writer(tmp_path):
    instances = []

    def make(**kwargs):
        options = {"buffer_interval": None, "max_buffer_seconds": None}
        options.update(kwargs)
        instance = JsonlWriter(tmp_path / f"bytes-{len(instances)}.jsonl", **options)
        instances.append(instance)
        return instance

    yield make
    for instance in instances:
        instance.close()


def test_byte_defaults_and_enqueue_status_preserve_flush_due_contract(writer, monkeypatch):
    default = writer()
    assert default.max_pending_bytes == DEFAULT_MAX_PENDING_BYTES == 64 * 1024 * 1024
    assert default.max_record_bytes == DEFAULT_MAX_RECORD_BYTES == 8 * 1024 * 1024
    assert default.async_write is False

    instance = writer(max_pending_bytes=128, max_record_bytes=64)
    errors = []
    monkeypatch.setattr(writer_module.logger, "error", errors.append)
    assert instance.enqueue_with_status(0, 0, line_for(0, 64)) == (True, True)
    instance.flush()
    assert instance.enqueue_with_status(1, 1, line_for(1, 32)) == (True, False)
    assert instance.enqueue(2, 2, line_for(2, 32)) is False
    assert instance.enqueue_with_status(99, 99, line_for(99, 65)) == (False, False)
    assert instance.enqueue(100, 100, line_for(100, 65)) is False
    assert instance.dropped == 2 and instance.dropped_bytes == 130
    assert len(errors) == 2
    assert instance.flushed_row == 0
    assert_bytes(instance, 64, 0)
    snapshot = instance.buffer
    snapshot.clear()
    assert_bytes(instance, 64, 0)
    instance.close()
    assert steps_on_disk(instance) == [0, 1, 2]
    assert instance.flushed_row == 2
    assert_bytes(instance, 0, 0)
    assert instance.enqueue_with_status(3, 3, line_for(3, 32)) == (False, False)
    assert instance.dropped == 2


@pytest.mark.parametrize("name", ["max_pending_bytes", "max_record_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "1024", float("inf"), float("nan")])
def test_byte_limits_require_positive_integers_or_none(tmp_path, name, value):
    path = tmp_path / "not-created" / "records.jsonl"
    with pytest.raises(ValueError, match=f"{name} must be a positive integer or None"):
        JsonlWriter(path, **{name: value})
    assert not path.parent.exists()


@pytest.mark.parametrize(
    ("pending_limit", "record_limit"),
    [(None, None), (None, 64), (64, None), (64, 64)],
)
def test_byte_limits_can_be_disabled_independently(writer, pending_limit, record_limit):
    instance = writer(max_pending_bytes=pending_limit, max_record_bytes=record_limit)
    size = DEFAULT_MAX_RECORD_BYTES + 1 if pending_limit is record_limit is None else 64
    assert instance.enqueue_with_status(0, 0, line_for(0, size)) == (True, True)
    assert_bytes(instance, size, 0)
    instance.close()
    assert instance.size == size
    assert_bytes(instance, 0, 0)


@pytest.mark.parametrize(
    ("pending_limit", "record_limit", "size"),
    [(96, 64, 65), (96, 128, 97), (96, None, 97), (None, 64, 65)],
)
def test_oversize_rejection_never_waits_even_at_capacity(
    writer, monkeypatch, pending_limit, record_limit, size
):
    instance = writer(max_pending_bytes=pending_limit, max_record_bytes=record_limit)
    instance.enqueue(0, 0, line_for(0, 48))
    instance.enqueue(1, 1, line_for(1, 48))
    errors = []

    def reject_wait(timeout=None):
        pytest.fail("an oversized record must never wait for capacity")

    monkeypatch.setattr(instance._condition, "wait", reject_wait)
    monkeypatch.setattr(writer_module.logger, "error", errors.append)
    assert instance.enqueue_with_status(9, 9, line_for(9, size)) == (False, False)
    assert instance.dropped == 1 and instance.dropped_bytes == size
    assert len(errors) == 1
    assert "Rejected oversized record" in errors[0]
    assert f"{size} bytes" in errors[0]
    assert f"max_pending_bytes={pending_limit}" in errors[0]
    assert f"max_record_bytes={record_limit}" in errors[0]
    assert instance.flushed_row == -1
    assert instance._worker is None
    assert_bytes(instance, 96, 0)
    instance.close()
    assert steps_on_disk(instance) == [0, 1]


@pytest.mark.parametrize("async_write", [False, True])
def test_mixed_sizes_wait_for_queued_and_inflight_bytes(writer, monkeypatch, async_write):
    instance = writer(
        buffer_size=50,
        max_pending_bytes=256,
        max_record_bytes=128,
        async_write=async_write,
    )
    writing, release, waiting, produced = (threading.Event() for _ in range(4))
    original_write, original_wait = instance._write, instance._condition.wait
    writing_threads = set()
    statuses = []

    def slow_write(payload):
        writing_threads.add(threading.current_thread())
        with instance._lock:
            assert instance.inflight_bytes == len(payload)
            assert instance.pending_bytes <= 256
        writing.set()
        assert release.wait(5)
        original_write(payload)

    def wait(timeout=None):
        if threading.current_thread() is producer:
            waiting.set()
        return original_wait(timeout)

    def produce():
        for row in range(4, 54):
            size = (128, 32, 96, 48, 64)[(row - 4) % 5]
            statuses.append(instance.enqueue_with_status(row, row, line_for(row, size)))
            assert instance.pending_bytes <= 256
        produced.set()

    producer = threading.Thread(target=produce)
    flusher = threading.Thread(target=instance.flush)
    monkeypatch.setattr(instance, "_write", slow_write)
    monkeypatch.setattr(instance._condition, "wait", wait)
    try:
        with instance._lock:
            instance.enqueue(0, 0, line_for(0, 64))
            instance.enqueue(1, 1, line_for(1, 96))
        flusher.start()
        assert writing.wait(3)
        assert_bytes(instance, 0, 160)
        instance.enqueue(2, 2, line_for(2, 48))
        instance.enqueue(3, 3, line_for(3, 48))
        assert_bytes(instance, 96, 160)
        producer.start()
        assert waiting.wait(3)
        assert not produced.is_set()
        assert instance.dropped == 0
    finally:
        release.set()
        for thread in (producer, flusher):
            if thread.ident is not None:
                thread.join(5)
    assert produced.is_set()
    assert not producer.is_alive() and not flusher.is_alive()
    assert all(accepted for accepted, _ in statuses)
    assert producer not in writing_threads
    instance.close()
    assert steps_on_disk(instance) == list(range(54))
    assert instance.flushed_row == 53
    assert instance.sorted and not instance.has_duplicate_steps
    assert instance.dropped == instance.dropped_bytes == 0
    assert_bytes(instance, 0, 0)


def test_partial_failure_requeues_bytes_and_wakes_waiters_to_drop_oldest(writer, monkeypatch):
    instance = writer(
        buffer_size=2,
        max_pending_bytes=256,
        max_record_bytes=128,
        max_buffer_seconds=30,
        async_write=True,
        index_every=1,
    )
    writing, release, waiting = (threading.Event() for _ in range(3))
    original_write, original_wait = instance._write, instance._condition.wait
    statuses, errors = [], []

    def fail(payload):
        with open(instance.path, "ab") as stream:
            stream.write(payload[:50])
        writing.set()
        assert release.wait(5)
        raise OSError("partial write")

    def wait(timeout=None):
        if threading.current_thread() is producer:
            waiting.set()
        return original_wait(timeout)

    def produce():
        statuses.append(instance.enqueue_with_status(4, 4, line_for(4, 128)))

    producer = threading.Thread(target=produce)
    monkeypatch.setattr(instance, "_write", fail)
    monkeypatch.setattr(instance._condition, "wait", wait)
    monkeypatch.setattr(writer_module.logger, "error", errors.append)
    try:
        with instance._lock:
            instance.enqueue(0, 0, line_for(0, 80))
            instance.enqueue(1, 1, line_for(1, 48))
        assert writing.wait(3)
        instance.enqueue(2, 2, line_for(2, 48))
        instance.enqueue(3, 3, line_for(3, 80))
        assert_bytes(instance, 128, 128)
        producer.start()
        assert waiting.wait(3)
    finally:
        release.set()
        if producer.ident is not None:
            producer.join(5)
    assert not producer.is_alive() and statuses == [(True, True)]
    assert instance.path.read_bytes() == b""
    assert instance.lines == 0 and instance.flushed_row == -1
    assert [row for _, row, _ in instance.buffer] == [2, 3, 4]
    assert instance.dropped == 2 and instance.dropped_bytes == 128
    assert_bytes(instance, 256, 0)

    assert instance.enqueue_with_status(5, 5, line_for(5, 96)) == (True, True)
    assert [row for _, row, _ in instance.buffer] == [4, 5]
    assert instance.dropped == 4 and instance.dropped_bytes == 256
    assert sum("dropped 2 oldest queued records (128 bytes)" in error for error in errors) == 2
    assert_bytes(instance, 224, 0)
    monkeypatch.setattr(instance, "_write", original_write)
    instance.close()
    assert steps_on_disk(instance) == [4, 5]
    assert instance.flushed_row == 5
    assert instance.size == 224
    assert not instance.has_duplicate_steps
    assert [entry[1:] for entry in instance.index] == [(0, 0), (1, 128)]
    assert_bytes(instance, 0, 0)


@pytest.mark.parametrize("retry_fails", [False, True])
def test_failure_overflow_cannot_evict_inflight_bytes(writer, monkeypatch, retry_fails):
    instance = writer(max_pending_bytes=256, max_record_bytes=192)
    original_write = instance._write
    errors = []

    def fail(payload):
        raise OSError("broken disk")

    monkeypatch.setattr(instance, "_write", fail)
    monkeypatch.setattr(writer_module.logger, "error", errors.append)
    instance.enqueue(0, 0, line_for(0, 160))
    instance.flush()
    assert_bytes(instance, 160, 0)
    writing, release = threading.Event(), threading.Event()

    def retry(payload):
        writing.set()
        assert release.wait(5)
        if retry_fails:
            fail(payload)
        original_write(payload)

    monkeypatch.setattr(instance, "_write", retry)
    flusher = threading.Thread(target=instance.flush)
    try:
        flusher.start()
        assert writing.wait(3)
        assert_bytes(instance, 0, 160)
        assert instance.enqueue_with_status(1, 1, line_for(1, 64))[0]
        assert instance.enqueue_with_status(2, 2, line_for(2, 128)) == (False, False)
        assert instance.enqueue(3, 3, line_for(3, 128)) is False
        assert instance.dropped == 3 and instance.dropped_bytes == 320
        assert_bytes(instance, 0, 160)
        assert instance.enqueue_with_status(4, 4, line_for(4, 96))[0]
        assert_bytes(instance, 96, 160)
    finally:
        release.set()
        flusher.join(5)
    assert not flusher.is_alive()
    assert_bytes(instance, 256 if retry_fails else 96, 0)
    assert instance.flushed_row == (-1 if retry_fails else 0)
    assert any("dropped 2 oldest queued records (192 bytes)" in error for error in errors)
    monkeypatch.setattr(instance, "_write", original_write)
    instance.close()
    assert steps_on_disk(instance) == [0, 4]
    assert instance.flushed_row == 4
    assert_bytes(instance, 0, 0)


def test_failure_overflow_enforces_count_and_byte_caps_together(writer, monkeypatch):
    instance = writer(
        buffer_size=1,
        max_pending_records=3,
        max_pending_bytes=160,
        max_record_bytes=128,
    )
    original_write = instance._write
    errors = []

    def fail(payload):
        raise OSError("offline")

    monkeypatch.setattr(instance, "_write", fail)
    monkeypatch.setattr(writer_module.logger, "error", errors.append)
    instance.enqueue(0, 0, line_for(0, 32))
    instance.flush()
    for row in range(1, 4):
        assert instance.enqueue_with_status(row, row, line_for(row, 32))[0]
    assert_bytes(instance, 96, 0)
    assert instance.dropped == 1 and instance.dropped_bytes == 32
    assert [row for _, row, _ in instance.buffer] == [1, 2, 3]
    assert instance.enqueue_with_status(4, 4, line_for(4, 128))[0]
    assert_bytes(instance, 160, 0)
    assert instance.dropped == 3 and instance.dropped_bytes == 96
    assert [row for _, row, _ in instance.buffer] == [3, 4]
    for row in range(5, 8):
        assert instance.enqueue_with_status(row, row, line_for(row, 16))[0]
    assert_bytes(instance, 48, 0)
    assert instance.dropped == 5 and instance.dropped_bytes == 256
    assert [row for _, row, _ in instance.buffer] == [5, 6, 7]
    assert any("dropped 2 oldest queued records (64 bytes)" in error for error in errors)
    monkeypatch.setattr(instance, "_write", original_write)
    instance.close()
    assert steps_on_disk(instance) == [5, 6, 7]
    assert instance.flushed_row == 7
    assert_bytes(instance, 0, 0)


@pytest.mark.parametrize("async_write", [False, True])
def test_concurrent_mixed_size_producers_and_flushers_keep_byte_accounting(
    writer, async_write
):
    instance = writer(
        buffer_size=7,
        max_pending_records=10,
        max_pending_bytes=512,
        max_record_bytes=256,
        async_write=async_write,
    )
    allocation = threading.Lock()
    start = threading.Barrier(6)
    accepted = []

    def produce():
        start.wait()
        for _ in range(100):
            with allocation:
                row = len(accepted)
                size = (32, 64, 256, 128)[row % 4]
                accepted.append(row)
                assert instance.enqueue_with_status(row, row, line_for(row, size))[0]
                assert instance.pending_bytes <= 512

    def flush():
        start.wait()
        for _ in range(100):
            instance.flush()
            assert instance.pending_bytes <= 512

    threads = [threading.Thread(target=produce) for _ in range(4)]
    threads.append(threading.Thread(target=flush))
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    instance.close()
    assert steps_on_disk(instance) == accepted == list(range(400))
    assert instance.size == 100 * (32 + 64 + 256 + 128)
    assert instance.flushed_row == 399
    assert instance.sorted and not instance.has_duplicate_steps
    assert instance.dropped == instance.dropped_bytes == 0
    assert_bytes(instance, 0, 0)


@pytest.mark.parametrize("final_failure", [False, True])
def test_close_wakes_byte_waiters_and_accounts_for_failed_final_flush(
    writer, monkeypatch, final_failure
):
    instance = writer(max_pending_bytes=256, max_record_bytes=192, async_write=True)
    writing, release, waiting = (threading.Event() for _ in range(3))
    original_write, original_wait = instance._write, instance._condition.wait
    blocked, results, errors = set(), [], []

    def write(payload):
        writing.set()
        assert release.wait(5)
        if final_failure:
            raise OSError("final failure")
        original_write(payload)

    def wait(timeout=None):
        thread = threading.current_thread()
        if thread in producers:
            blocked.add(thread)
            if len(blocked) == len(producers):
                waiting.set()
        return original_wait(timeout)

    def produce(row):
        results.append(instance.enqueue_with_status(row, row, line_for(row, 32)))

    producers = [threading.Thread(target=produce, args=(row,)) for row in range(2, 5)]
    closer = threading.Thread(target=instance.close)
    monkeypatch.setattr(instance, "_write", write)
    monkeypatch.setattr(instance._condition, "wait", wait)
    monkeypatch.setattr(writer_module.logger, "error", errors.append)
    try:
        instance.enqueue(0, 0, line_for(0, 192))
        assert writing.wait(3)
        instance.enqueue(1, 1, line_for(1, 64))
        for producer in producers:
            producer.start()
        assert waiting.wait(3)
        assert_bytes(instance, 64, 192)
        closer.start()
        for producer in producers:
            producer.join(3)
            assert not producer.is_alive()
        assert results == [(False, False)] * 3
        assert closer.is_alive()
    finally:
        release.set()
        for thread in [*producers, closer]:
            if thread.ident is not None:
                thread.join(5)
    assert not closer.is_alive()
    assert instance._closed and not instance._worker.is_alive()
    assert instance.dropped == instance.dropped_bytes == 0
    assert_bytes(instance, 256 if final_failure else 0, 0)
    if final_failure:
        assert instance.flushed_row == -1
        assert any("2 unwritten records" in error and "(256 bytes)" in error for error in errors)
    else:
        assert steps_on_disk(instance) == [0, 1]
        assert instance.flushed_row == 1
    instance.flush()
    instance.close()
    assert_bytes(instance, 256 if final_failure else 0, 0)


def test_repeated_batches_and_partial_retries_do_not_leak_byte_accounting(writer, monkeypatch):
    instance = writer(max_pending_bytes=8192, max_record_bytes=512, index_every=3)
    original_write = instance._write
    fail_next = False

    def write(payload):
        nonlocal fail_next
        if fail_next:
            fail_next = False
            with open(instance.path, "ab") as stream:
                stream.write(payload[:73])
            raise OSError("intermittent partial write")
        original_write(payload)

    monkeypatch.setattr(instance, "_write", write)
    total_bytes = 0
    for batch in range(40):
        batch_bytes = 0
        for row in range(batch * 15, (batch + 1) * 15):
            size = (32, 128, 512)[row % 3]
            assert instance.enqueue_with_status(row, row, line_for(row, size))[0]
            batch_bytes += size
        assert_bytes(instance, batch_bytes, 0)
        fail_next = batch % 3 == 0
        if fail_next:
            instance.flush()
            assert_bytes(instance, batch_bytes, 0)
            assert instance.size == total_bytes
            assert instance.flushed_row == batch * 15 - 1
        instance.flush()
        total_bytes += batch_bytes
        assert_bytes(instance, 0, 0)
        assert instance.size == total_bytes
    instance.close()
    assert steps_on_disk(instance) == list(range(600))
    assert instance.flushed_row == 599
    assert instance.dropped == instance.dropped_bytes == 0
    assert instance.path.stat().st_size == total_bytes


@pytest.mark.benchmark
def test_failure_overflow_bookkeeping_scales_linearly(writer, monkeypatch):
    if sys.gettrace() is not None:
        pytest.skip("timing is meaningless under coverage/debugger tracing")

    class MeasuredBytes(bytes):
        calls = 0

        def __len__(self):
            type(self).calls += 1
            return super().__len__()

    def fail(payload):
        raise OSError("offline")

    monkeypatch.setattr(writer_module.logger, "error", lambda message: None)
    line = MeasuredBytes(line_for(0, 32))
    timings = {}
    for capacity in (1000, 30_000):
        samples = []
        for _ in range(3):
            instance = writer(
                buffer_size=1,
                max_pending_records=100_000,
                max_pending_bytes=capacity * 32,
            )
            monkeypatch.setattr(instance, "_write", fail)
            instance.enqueue(0, 0, line)
            instance.flush()
            for row in range(1, capacity):
                instance.enqueue(row, row, line)
            MeasuredBytes.calls = 0
            start = time.perf_counter()
            for row in range(capacity, capacity + 40_000):
                instance.enqueue(row, row, line)
            samples.append(time.perf_counter() - start)
            # One len() for admission, one for each evicted record: no queue scans.
            assert MeasuredBytes.calls == 80_000
            assert instance.pending_bytes == capacity * 32
            assert instance.dropped == 40_000
            instance.close()
        timings[capacity] = statistics.median(samples)
    ratio = timings[30_000] / timings[1000]
    print(
        f"\n40,000 overflow admissions: capacity=1,000 {timings[1000]:.4f}s; "
        f"capacity=30,000 {timings[30_000]:.4f}s; ratio={ratio:.2f}; "
        "exactly 80,000 length calls per run"
    )
    assert ratio < 4


@pytest.mark.benchmark
@pytest.mark.parametrize("failing", [False, True])
def test_streaming_large_records_keeps_memory_bounded(writer, monkeypatch, failing):
    cap = 256 * 1024
    instance = writer(
        buffer_size=32,
        max_pending_bytes=cap,
        max_record_bytes=64 * 1024,
        async_write=not failing,
    )
    original_write = instance._write
    maximum_pending = 0

    def write(payload):
        nonlocal maximum_pending
        with instance._lock:
            assert instance.inflight_bytes == len(payload)
            maximum_pending = max(maximum_pending, instance.pending_bytes)
            assert instance.pending_bytes <= cap
        if failing:
            raise OSError("offline")
        time.sleep(0.0001)
        original_write(payload)

    monkeypatch.setattr(instance, "_write", write)
    monkeypatch.setattr(writer_module.logger, "error", lambda message: None)
    total_bytes = 0
    gc.collect()
    tracemalloc.start()
    try:
        for row in range(4096):
            size = (1024, 4096, 16 * 1024, 64 * 1024)[row % 4]
            assert instance.enqueue_with_status(row, row, line_for(row, size))[0]
            total_bytes += size
            if failing and row == 0:
                instance.flush()
            pending = instance.pending_bytes
            maximum_pending = max(maximum_pending, pending)
            assert pending <= cap
        instance.close()
        retained, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert total_bytes == 89_128_960
    assert maximum_pending >= cap - 64 * 1024
    assert peak < 4 * cap + 1024 * 1024
    assert retained < cap + 512 * 1024
    if failing:
        assert_bytes(instance, instance.queued_bytes, 0)
        assert instance.pending_bytes + instance.dropped_bytes == total_bytes
        assert len(instance.buffer) + instance.dropped == 4096
    else:
        assert_bytes(instance, 0, 0)
        assert instance.lines == 4096 and instance.flushed_row == 4095
        assert instance.dropped == instance.dropped_bytes == 0
        assert instance.path.stat().st_size == total_bytes
    print(
        f"\n{'failing' if failing else 'healthy'} 4,096-record stream: "
        f"encoded={total_bytes:,} bytes, max_pending={maximum_pending:,}/{cap:,}, "
        f"tracemalloc_peak={peak:,}, retained={retained:,}, dropped={instance.dropped}"
    )
