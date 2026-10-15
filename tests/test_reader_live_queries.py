"""Live reader cache, patch and concurrent-prefix regressions."""

import json
import random
import threading

import pytest
from _reader_edge_helpers import (
    reader_warnings as reader_warnings,
)
from _reader_edge_helpers import (
    write_meta as write_meta,
)
from _reader_edge_helpers import (
    write_rows as write_rows,
)

from expr_tracker.history import JsonlReader, read_history


@pytest.mark.parametrize("cache_rows", [1, 3, 100])
@pytest.mark.parametrize("resume", [False, True])
def test_live_tail_and_range_match_offline_recency(store, cache_rows, resume):
    history = store(cache_rows=cache_rows, step_policy="allow", buffer_size=1)
    for step, fields in [
        (100, {"original": 1, "value": "old"}),
        (2, {"low": 2}),
        (100, {"value": "new"}),
        (90, {"other": 3}),
        (1, {"low": 1}),
    ]:
        history.log(fields, step=step, commit=True)
    if resume:
        history.finish()
        history = store(cache_rows=cache_rows, step_policy="allow", buffer_size=1)
    for count, steps in [
        (1, [1]),
        (2, [1, 90]),
        (3, [1, 90, 100]),
        (20, [1, 2, 90, 100]),
    ]:
        rows = history.get(count)
        assert [row["_step"] for row in rows] == steps
        assert rows == read_history(history.log_fp, count)
        for bounds in [(None, None), (2, 101), (90, None), (None, 3)]:
            assert history.get(count, step_range=bounds) == read_history(
                history.log_fp, count, step_range=bounds
            )
    assert history.get(3)[-1]["original"] == 1
    assert history.get(-1) == read_history(history.log_fp, -1)


@pytest.mark.parametrize("cache_rows", [1, 2, 100])
@pytest.mark.parametrize("resume", [False, True])
def test_open_old_step_merges_its_original_before_tail_selection(
    store, cache_rows, resume
):
    history = store(cache_rows=cache_rows, step_policy="allow", buffer_size=1)
    history.log({"original": 1, "value": "old"}, step=0, commit=True)
    history.log({"one": 1}, step=1, commit=True)
    history.log({"two": 2}, step=2, commit=True)
    if resume:
        history.finish()
        history = store(cache_rows=cache_rows, step_policy="allow", buffer_size=1)
    history.log({"value": "open"}, step=0)
    expected = {
        "_step": 0,
        "_time": history.open_record()["_time"],
        "original": 1,
        "value": "open",
    }
    assert history.get(1) == [expected]
    assert history.get(2)[0] == expected
    assert [row["_step"] for row in history.get(2)] == [0, 2]
    assert history.get(1, include_open=False) == read_history(history.log_fp, 1)
    assert history.get(1, step_range=(0, 1)) == [expected]
    assert history.get(1, step_range=(0, 3)) == read_history(
        history.log_fp, 1, step_range=(0, 3)
    )
    history.flush(commit_open=True)
    assert history.get(1) == read_history(history.log_fp, 1) == [expected]


@pytest.mark.parametrize("cache_rows", [1, 6000])
def test_live_tail_has_no_patch_widening_limit(store, cache_rows):
    history = store(cache_rows=cache_rows, buffer_size=1000, buffer_interval=None)
    history.log({"zero": 0}, step=0, commit=True)
    history.log({"original": 1}, step=1, commit=True)
    for value in range(5000):
        history.log({"value": value}, step=1, commit=True)
    history.flush()
    for count in (1, 2):
        rows = history.get(count)
        assert rows == read_history(history.log_fp, count)
        assert rows[-1]["original"] == 1
        assert rows[-1]["value"] == 4999


def test_cache_miss_does_not_wait_for_commit_callback_or_flush_spans(
    store, monkeypatch
):
    history = store(cache_bytes=1, buffer_size=1)
    history.log({"value": 0})
    history.log({"value": 1})
    entered, release, read_done = (threading.Event() for _ in range(3))
    errors, results, span_flushes = [], [], []

    def callback(row):
        entered.set()
        assert release.wait(5), "callback was not released"

    def query():
        try:
            results.extend(history.get(2))
        except Exception as error:
            errors.append(error)
        finally:
            read_done.set()

    history.on_commit = callback
    producer = threading.Thread(target=lambda: history.log({"value": 2}))
    reader = threading.Thread(target=query)
    with monkeypatch.context() as patch:
        patch.setattr(history.span_writer, "flush", lambda: span_flushes.append(True))
        producer.start()
        try:
            assert entered.wait(5)
            reader.start()
            assert read_done.wait(1), "cache miss waited for a commit callback"
            assert producer.is_alive()
            assert not span_flushes
        finally:
            release.set()
            producer.join(5)
            if reader.ident is not None:
                reader.join(5)
    assert not producer.is_alive() and not reader.is_alive()
    assert not errors
    assert [row["_step"] for row in results] == [1, 2]
    assert history.stats()["disk_queries"] == 1


def test_cache_miss_uses_durable_prefix_without_flushing_pending_metrics(
    store, monkeypatch
):
    history = store(
        cache_rows=2, buffer_size=1000, buffer_interval=None, max_buffer_seconds=None
    )
    for value in range(3):
        history.log({"value": value})
    history.flush()
    history.log({"pending": 3})
    assert history.writer.pending_bytes > 0
    before = history.writer.size
    flushes = []
    with monkeypatch.context() as patch:
        patch.setattr(history.writer, "flush", lambda: flushes.append(True))
        assert [row["_step"] for row in history.get(-1)] == [0, 1, 2, 3]
        assert [row["_step"] for row in history.get(3)] == [1, 2, 3]
    assert not flushes
    assert history.writer.size == before
    assert history.writer.pending_bytes > 0


@pytest.mark.parametrize("cache_rows", [1, 3])
def test_rejected_writer_record_does_not_break_unordered_disk_boundary(
    store, cache_rows
):
    history = store(
        cache_rows=cache_rows,
        step_policy="allow",
        buffer_size=1,
        max_record_bytes=256,
        max_pending_bytes=1024,
    )
    for step in [100, 2, 90]:
        history.log({"original": step}, step=step, commit=True)
    assert history.log({"oversized": "x" * 1024}, step=101, commit=True) is None
    history.log({"patch": True}, step=2, commit=True)
    history.log({"last": True}, step=1, commit=True)
    assert history.writer.dropped == 1
    for count in (1, 2, 3, -1):
        assert history.get(count) == read_history(history.log_fp, count)
        assert history.get(count, step_range=(0, 101)) == read_history(
            history.log_fp, count, step_range=(0, 101)
        )
    assert history.get(2)[-1]["original"] == 2


def test_overflowed_writer_ordinals_still_merge_disk_originals(store, monkeypatch):
    history = store(
        cache_rows=1,
        step_policy="allow",
        buffer_size=1,
        buffer_interval=None,
        max_buffer_seconds=None,
        max_pending_records=2,
    )
    history.log({"original": 0}, step=0, commit=True)

    def fail_write(payload):
        raise OSError("disk unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(history.writer, "_write", fail_write)
        for step in [1, 2, 3]:
            history.log({"original": step}, step=step, commit=True)
    assert history.writer.dropped > 0
    history.log({"original": 4}, step=4, commit=True)
    history.log({"patch": True}, step=0, commit=True)
    for count in (1, 2, 10, -1):
        assert history.get(count) == read_history(history.log_fp, count)
    assert history.get(1)[0]["original"] == 0


@pytest.mark.parametrize("cache_rows", [1, 7, 1000])
def test_tail_and_range_selection_match_a_physical_write_model(store, cache_rows):
    history = store(
        cache_rows=cache_rows,
        step_policy="allow",
        buffer_size=20,
        buffer_interval=None,
        max_buffer_seconds=None,
    )
    rng = random.Random(42)
    for write in range(80):
        history.log(
            {f"value{write % 5}": write}, step=rng.randrange(-3, 8), commit=True
        )
    history.flush()
    committed = [json.loads(line) for line in history.log_fp.read_bytes().splitlines()]
    history.log({"open": True}, step=-2)
    for include_open in (False, True):
        physical = committed + ([history.open_record()] if include_open else [])
        merged, last_write = {}, {}
        for ordinal, row in enumerate(physical):
            step = row["_step"]
            merged.setdefault(step, {}).update(row)
            last_write[step] = ordinal
        for count in (0, 1, 2, 5, 20, -1, None):
            limit = None if count is None or count < 0 else count
            recent = sorted(last_write, key=last_write.get, reverse=True)[:limit]
            expected = [merged[step] for step in sorted(recent)]
            assert history.get(count, include_open=include_open) == expected
            if not include_open:
                assert read_history(history.log_fp, count) == expected
            for start, end in [(None, None), (-2, 3), (None, 0), (3, None), (4, 4)]:
                selected = [
                    step
                    for step in sorted(merged)
                    if (start is None or step >= start) and (end is None or step < end)
                ]
                if limit is not None:
                    selected = selected[-limit:] if limit else []
                expected = [merged[step] for step in selected]
                assert (
                    history.get(
                        count, step_range=(start, end), include_open=include_open
                    )
                    == expected
                )
                if not include_open:
                    assert (
                        read_history(history.log_fp, count, step_range=(start, end))
                        == expected
                    )


@pytest.mark.parametrize("timestamp", [None, "invalid", float("inf"), -float("inf"), 0])
def test_selected_patch_merge_preserves_raw_timestamps(tmp_path, timestamp):
    path = tmp_path / "metrics.jsonl"
    write_rows(
        path,
        [
            {"_step": 7, "_time": timestamp, "original": 1},
            {"_step": 8, "_time": 2},
            {"_step": 7, "patch": 2},
        ],
    )
    assert read_history(path, 1) == [
        {"_step": 7, "_time": timestamp, "original": 1, "patch": 2}
    ]


@pytest.mark.parametrize(
    "selection",
    [{"n": 1}, {"n": None}, {"n": None, "step_range": (0, 2)}],
    ids=["tail", "all", "range"],
)
@pytest.mark.parametrize("meta_state", ["missing", "stale", "updated"])
def test_append_after_size_check_reads_one_coherent_prefix(
    tmp_path, monkeypatch, selection, meta_state
):
    path = tmp_path / "metrics.jsonl"
    original = [{"_step": 0, "original": 1}, {"_step": 1, "other": 2}]
    write_rows(path, original)
    if meta_state != "missing":
        write_meta(path, index=[[0, 0, 0]])
    snapshot = JsonlReader._snapshot
    appended = False

    def append_after_stat(reader, end=None):
        nonlocal appended
        captured = snapshot(reader, end)
        if reader.path == path and not appended:
            appended = True
            with path.open("ab") as writer:
                writer.write(b'{"_step": 0, "patch": 3}\n')
            if meta_state == "updated":
                write_meta(path, index=[[0, 0, 0]], sorted=False)
        return captured

    monkeypatch.setattr(JsonlReader, "_snapshot", append_after_stat)
    expected = original[-1:] if selection["n"] == 1 else original
    assert read_history(path, **selection) == expected
    assert appended
    merged = [{"_step": 0, "original": 1, "patch": 3}, original[1]]
    assert read_history(path, **selection) == (
        merged[:1] if selection["n"] == 1 else merged
    )


@pytest.mark.parametrize("method", ["tail", "read_all", "read_steps"])
def test_reused_reader_refreshes_order_metadata_after_append(tmp_path, method):
    path = tmp_path / "metrics.jsonl"
    original = [{"_step": 0, "original": 1}, {"_step": 1, "other": 2}]
    write_rows(path, original)
    write_meta(path, index=[[0, 0, 0]])
    reader = JsonlReader(path)
    args = {"tail": (1,), "read_all": (), "read_steps": (0, 2)}[method]
    query = getattr(reader, method)
    assert query(*args) == (original[-1:] if method == "tail" else original)
    with path.open("ab") as writer:
        writer.write(b'{"_step": 0, "patch": 3}\n')
    merged = [{"_step": 0, "original": 1, "patch": 3}, original[1]]
    assert query(*args) == (merged[:1] if method == "tail" else merged)


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("count", [1, None])
def test_append_does_not_change_an_unterminated_snapshot_row(
    tmp_path, monkeypatch, complete, count
):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"_step": 0}\n{"_step": 1' + (b"}" if complete else b""))
    snapshot = JsonlReader._snapshot
    appended = False

    def append_after_stat(reader, end=None):
        nonlocal appended
        captured = snapshot(reader, end)
        if reader.path == path and not appended:
            appended = True
            with path.open("ab") as writer:
                writer.write((b"" if complete else b"}") + b'\n{"_step": 2}\n')
        return captured

    monkeypatch.setattr(JsonlReader, "_snapshot", append_after_stat)
    expected = [{"_step": 0}, {"_step": 1}] if complete else [{"_step": 0}]
    assert read_history(path, count) == (
        expected if count is None else expected[-count:]
    )


def test_concurrent_queries_on_one_reader_keep_independent_prefixes(
    tmp_path, monkeypatch
):
    path = tmp_path / "metrics.jsonl"
    write_rows(path, [{"_step": 0, "original": 1}, {"_step": 1}])
    write_meta(path, index=[[0, 0, 0]])
    reader = JsonlReader(path)
    entered, release = threading.Event(), threading.Event()
    results, errors = [], []
    snapshot = JsonlReader._snapshot

    def pause_after_validation(instance, end=None):
        captured = snapshot(instance, end)
        if threading.current_thread() is first:
            entered.set()
            assert release.wait(5), "captured reader was not released"
        return captured

    def first_query():
        try:
            results.append(reader.tail(1))
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(JsonlReader, "_snapshot", pause_after_validation)
    first = threading.Thread(target=first_query)
    first.start()
    try:
        assert entered.wait(5)
        with path.open("ab") as writer:
            writer.write(b'{"_step": 0, "patch": 3}\n')
        assert reader.tail(1) == [{"_step": 0, "original": 1, "patch": 3}]
    finally:
        release.set()
        first.join(5)
    assert not first.is_alive()
    assert not errors
    assert results == [[{"_step": 1}]]
    assert reader.merge
