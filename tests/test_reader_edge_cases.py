"""Reader regressions using on-disk histories and interrupted-write states."""

import json
import random
import sys
import threading
import time
import tracemalloc

import pytest

from expr_tracker.history import JsonlReader, read_history
from expr_tracker.history import reader as reader_module


def write_rows(path, rows):
    payload = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    path.write_bytes(payload)
    return payload


def write_meta(path, **overrides):
    meta = {
        "schema": 2,
        "size": path.stat().st_size,
        "sorted": True,
        "has_duplicate_steps": False,
        "index": [],
    }
    meta.update(overrides)
    path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def reader_warnings():
    records = []
    sink = reader_module.logger.add(
        lambda message: records.append(message.record), level="WARNING"
    )
    try:
        yield records
    finally:
        reader_module.logger.remove(sink)


@pytest.mark.parametrize("with_meta", [False, True])
@pytest.mark.parametrize("count", [1, 2, 3, 20])
def test_tail_selects_recent_writes_and_merges_nonadjacent_patches(
    tmp_path, with_meta, count
):
    path = tmp_path / "metrics.jsonl"
    rows = [
        {"_step": 100, "original": 1, "value": "old"},
        {"_step": 2, "low": 2},
        {"_step": 100, "value": "new"},
        {"_step": 90, "other": 3},
        {"_step": 1, "low": 1},
    ]
    write_rows(path, rows)
    if with_meta:
        write_meta(path, sorted=False)
    expected = [
        {"_step": 1, "low": 1},
        {"_step": 2, "low": 2},
        {"_step": 90, "other": 3},
        {"_step": 100, "original": 1, "value": "new"},
    ]
    selected = set([2, 100, 90, 1][-count:])
    recent = [row for row in expected if row["_step"] in selected]
    assert read_history(path, count) == recent
    assert list(read_history(path, count)[-1]) == list(recent[-1])
    assert read_history(path, count, step_range=(2, 101)) == expected[1:][-count:]


def test_tail_has_no_patch_count_cutoff(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_rows(
        path,
        [{"_step": 0}, {"_step": 1, "original": True}]
        + [{"_step": 1, "value": i} for i in range(5000)],
    )
    assert read_history(path, 1) == [{"_step": 1, "original": True, "value": 4999}]


@pytest.mark.parametrize(
    "garbage",
    [
        b"{torn",
        b'{"_step": 2, "value": "\xff"}\n',
        b'{"_step": 2.5}\n',
        b'{"_step": "2"}\n',
        b'{"_step": null}\n',
        b'{"missing": "step"}\n',
        b"[1, 2]\n",
        b"null\n",
    ],
)
def test_tail_counts_readable_steps_not_malformed_rows(tmp_path, garbage):
    path = tmp_path / "metrics.jsonl"
    payload = write_rows(path, [{"_step": 0}, {"_step": 1}])
    path.write_bytes(payload + garbage)
    assert read_history(path, 2) == [{"_step": 0}, {"_step": 1}]
    assert read_history(path, 1) == [{"_step": 1}]
    assert read_history(path, None) == [{"_step": 0}, {"_step": 1}]
    assert read_history(path, None, step_range=(0, 2)) == [
        {"_step": 0},
        {"_step": 1},
    ]
    # The physical-row API intentionally does not refill skipped rows.
    assert JsonlReader(path).tail_rows(1) == []


@pytest.mark.parametrize("chunk_size", [1, 7, 64])
def test_reverse_reads_skip_blanks_across_chunks(tmp_path, monkeypatch, chunk_size):
    path = tmp_path / "metrics.jsonl"
    rows = [{"_step": i, "value": "long" * 50} for i in range(3)]
    lines = [json.dumps(row).encode() for row in rows]
    path.write_bytes(b"\r\n \n".join(lines) + b"\n" + b" \r\n" * 100)
    monkeypatch.setattr(reader_module, "CHUNK_SIZE", chunk_size)
    reader = JsonlReader(path)
    assert reader.tail_raw(2) == [lines[1] + b"\r", lines[2]]
    assert reader.tail(2) == rows[-2:]
    assert reader.count_lines() == 3


@pytest.mark.parametrize("boundary_delta", [0, 1, 12])
def test_byte_bounds_never_include_a_crossing_row(tmp_path, boundary_delta):
    path = tmp_path / "metrics.jsonl"
    first = b'{"_step": 0}\n'
    second = b'{"_step": 1, "value": 2}\n'
    path.write_bytes(first + second)
    end = len(first) + boundary_delta
    reader = JsonlReader(path)
    assert list(reader.iter_raw(end=end)) == [first]
    assert reader.read_steps(None, None, end) == [{"_step": 0}]
    assert reader.tail_rows(2, end) == [{"_step": 0}]


def test_scan_does_not_complete_a_partial_line_past_its_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    first = b'{"_step": 0}\n'
    partial = b'{"_step": 1, "value": '
    path.write_bytes(first + partial)
    original_open = open
    appended = False

    def append_before_open(file, mode="r", *args, **kwargs):
        nonlocal appended
        if file == path and mode == "rb" and not appended:
            appended = True
            with original_open(path, "ab") as writer:
                writer.write(b"2}\n")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(reader_module, "open", append_before_open, raising=False)
    reader = JsonlReader(path, index=[], merge=False)
    assert reader.read_all() == [{"_step": 0}]
    assert reader.read_all() == [{"_step": 0}, {"_step": 1, "value": 2}]


@pytest.mark.parametrize(
    "meta",
    [
        None,
        [],
        42,
        {"index": None},
        {"index": [1]},
        {"index": [[0, 0]]},
        {"index": [["0", 0, 0]]},
        {"index": [[0, 0, -1]]},
        {"index": [[0, 0, 10**9]]},
        {"index": [[0, 2, 15], [1, 1, 0]]},
        {"index": [[2, 0, 0], [1, 1, 21]]},
        {"index": [[2, 0, 0], [2, 1, 21]]},
        {"sorted": "false"},
        {"has_duplicate_steps": "false"},
        {"schema": 999},
    ],
)
def test_malformed_sidecars_fall_back_to_data(tmp_path, meta):
    path = tmp_path / "metrics.jsonl"
    rows = [{"_step": 2, "a": 1}, {"_step": 1}, {"_step": 2, "b": 2}]
    write_rows(path, rows)
    if isinstance(meta, dict):
        write_meta(path, **meta)
    else:
        path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    expected = [{"_step": 1}, {"_step": 2, "a": 1, "b": 2}]
    assert read_history(path, None) == expected
    assert read_history(path, 2, step_range=(1, 3)) == expected


@pytest.mark.parametrize("sidecar", [b"{torn", b"\xff", b""])
def test_torn_sidecars_fall_back_to_data(tmp_path, sidecar):
    path = tmp_path / "metrics.jsonl"
    write_rows(path, [{"_step": 1}, {"_step": 0}])
    path.with_suffix(".meta.json").write_bytes(sidecar)
    assert read_history(path, 1) == [{"_step": 0}]


@pytest.mark.parametrize("change", ["append", "truncate"])
def test_stale_sidecars_rebuild_after_size_changes(tmp_path, change):
    path = tmp_path / "metrics.jsonl"
    first = b'{"_step": 0}\n'
    second = b'{"_step": 1}\n'
    path.write_bytes(first + second)
    write_meta(path, index=[[0, 0, 0], [1, 1, len(first)]])
    if change == "append":
        path.write_bytes(first + second + b'{"_step": 0, "patch": 1}\n')
        expected = [{"_step": 0, "patch": 1}, {"_step": 1}]
    else:
        path.write_bytes(first)
        expected = [{"_step": 0}]
    assert read_history(path, None) == expected
    assert read_history(path, 10, step_range=(0, 10)) == expected


def test_supplied_index_does_not_imply_sorted_unique_steps(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_rows(path, [{"_step": 2, "a": 1}, {"_step": 1}, {"_step": 2, "b": 2}])
    reader = JsonlReader(path, index=[(2, 0, 0)])
    assert reader.read_all() == [{"_step": 1}, {"_step": 2, "a": 1, "b": 2}]
    assert reader.index == [(2, 0, 0)]


def test_sparse_index_counts_corrupt_records_but_not_blanks(tmp_path):
    path = tmp_path / "metrics.jsonl"
    prefix = b"\n" + b"bad\n\n" * 1000
    first = b'{"_step": 5}\n'
    second = b'{"_step": 9}\n'
    path.write_bytes(prefix + first + b" \n" + second + b"\n")
    reader = JsonlReader(path)
    assert reader.index == [(5, 1000, len(prefix))]
    assert reader.offset_of_line(0) == 0
    assert reader.offset_of_line(1) == len(b"\nbad\n\n")
    assert reader.offset_of_line(1000) == len(prefix)
    assert reader.offset_of_line(1001) == len(prefix + first + b" \n")
    assert reader.offset_of_line(1002) == reader.size
    assert reader.offset_of_step(-1) == len(prefix)
    assert reader.offset_of_step(6) == len(prefix + first + b" \n")
    assert reader.offset_of_step(10) == reader.size
    assert reader.read_steps(6, 10) == [{"_step": 9}]


def test_corruption_at_sampling_intervals_does_not_disable_seeking(
    tmp_path, monkeypatch
):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(
        b"".join(
            b"bad\n"
            if step % 1000 == 0
            else json.dumps({"_step": step}).encode() + b"\n"
            for step in range(4000)
        )
    )
    reader = JsonlReader(path)
    assert [entry[1] for entry in reader.index] == [1, 1001, 2001, 3001]
    scanned = 0
    original_scan = JsonlReader._scan

    def counted_scan(instance, *args, **kwargs):
        nonlocal scanned
        for row in original_scan(instance, *args, **kwargs):
            scanned += 1
            yield row

    monkeypatch.setattr(JsonlReader, "_scan", counted_scan)
    assert reader.read_steps(3998, 4000) == [{"_step": 3998}, {"_step": 3999}]
    assert 0 < scanned <= 1000


def test_valid_sparse_sidecar_keeps_physical_patch_boundaries(tmp_path):
    path = tmp_path / "metrics.jsonl"
    lines = [
        b'{"_step": 5, "a": 1}\n',
        b'{"_step": 5, "b": 2}\n',
        b'{"_step": 6}\n',
    ]
    path.write_bytes(b"".join(lines))
    write_meta(
        path,
        has_duplicate_steps=True,
        index=[[5, 0, 0], [5, 1, len(lines[0])]],
    )
    reader = JsonlReader(path)
    assert reader.merge
    boundary = reader.offset_of_line(1)
    assert boundary == len(lines[0])
    assert reader.tail(1, boundary) == [{"_step": 5, "a": 1}]
    assert reader.read_steps(5, 6, boundary) == [{"_step": 5, "a": 1}]
    assert reader.read_all() == [{"_step": 5, "a": 1, "b": 2}, {"_step": 6}]


def test_metadata_does_not_override_explicit_reader_state(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [{"_step": 1}, {"_step": 2}]
    write_rows(path, rows)
    write_meta(path, index=[[1, 0, 0]])
    assert JsonlReader(path, index=[]).read_all() == rows
    assert JsonlReader(path, merge=False).index == [(1, 0, 0)]


@pytest.mark.parametrize("error", [PermissionError, OSError])
def test_metadata_io_errors_warn_and_scan_readable_data(
    tmp_path, monkeypatch, reader_warnings, error
):
    path = tmp_path / "metrics.jsonl"
    write_rows(
        path, [{"_step": 1, "original": 1}, {"_step": 2}, {"_step": 1, "patch": 2}]
    )
    write_meta(path, sorted=False)
    meta_path = path.with_suffix(".meta.json")
    original_open = open

    def deny_meta(file, *args, **kwargs):
        if file == meta_path:
            raise error("metadata unavailable")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(reader_module, "open", deny_meta, raising=False)
    merged = {"_step": 1, "original": 1, "patch": 2}
    assert read_history(path, 1) == [merged]
    assert read_history(path, None, step_range=(1, 3)) == [merged, {"_step": 2}]
    assert len(reader_warnings) == 2
    assert all(
        record["level"].name == "WARNING"
        and str(meta_path) in record["message"]
        and "metadata unavailable" in record["message"]
        and "scanning history file" in record["message"]
        for record in reader_warnings
    )


def test_directory_instead_of_sidecar_warns_and_scans_data(tmp_path, reader_warnings):
    path = tmp_path / "metrics.jsonl"
    write_rows(
        path, [{"_step": 2, "original": 1}, {"_step": 1}, {"_step": 2, "patch": 2}]
    )
    meta_path = path.with_suffix(".meta.json")
    meta_path.mkdir()
    assert read_history(path, 1) == [{"_step": 2, "original": 1, "patch": 2}]
    assert len(reader_warnings) == 1
    assert reader_warnings[0]["level"].name == "WARNING"
    assert str(meta_path) in reader_warnings[0]["message"]
    assert "scanning history file" in reader_warnings[0]["message"]


@pytest.mark.parametrize("error", [PermissionError, OSError])
def test_data_file_io_errors_still_propagate(tmp_path, monkeypatch, error):
    path = tmp_path / "metrics.jsonl"
    write_rows(path, [{"_step": 1}])
    write_meta(path, index=[[1, 0, 0]])
    original_open = open

    def deny_data(file, *args, **kwargs):
        if file == path:
            raise error("metrics denied")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(reader_module, "open", deny_data, raising=False)
    with pytest.raises(error, match="metrics denied"):
        read_history(path, 1)


@pytest.mark.parametrize("step_range", [(None, 0), (9, None), (5, 5), (9, 1)])
def test_empty_step_selections(tmp_path, step_range):
    path = tmp_path / "metrics.jsonl"
    write_rows(path, [{"_step": 2}, {"_step": 5}, {"_step": 8}])
    assert read_history(path, 5, step_range=step_range) == []


@pytest.mark.parametrize("payload", [b"", b"\n \r\n", b"{torn", b"null\n"])
def test_empty_and_unreadable_files_have_empty_results(tmp_path, payload):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(payload)
    reader = JsonlReader(path)
    assert reader.tail(0) == []
    assert reader.tail_raw(-1) == []
    assert reader.tail(3) == []
    assert reader.read_all() == []
    assert reader.read_steps(0, 5) == []
    assert reader.last_record() is None
    assert reader.offset_of_line(100) == reader.size
    assert reader.offset_of_step(100) == reader.size


def test_missing_low_level_file_is_empty(tmp_path):
    reader = JsonlReader(tmp_path / "missing.jsonl")
    assert reader.index == []
    assert reader.tail_raw(1) == []
    assert list(reader.iter_raw()) == []
    assert reader.last_record() is None


def test_huge_integer_steps_remain_exact(tmp_path):
    path = tmp_path / "metrics.jsonl"
    huge = 10**100
    rows = [{"_step": -huge}, {"_step": huge}, {"_step": huge + 1}]
    write_rows(path, rows)
    assert read_history(path, None, step_range=(huge, huge + 1)) == rows[1:2]
    assert read_history(path, 2) == rows[-2:]


@pytest.mark.skipif(
    not hasattr(sys, "get_int_max_str_digits") or not sys.get_int_max_str_digits(),
    reason="Python integer-string length limit is disabled",
)
def test_over_limit_integer_row_is_skipped(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(
        b'{"_step": 1}\n{"_step": ' + b"9" * (sys.get_int_max_str_digits() + 1) + b"}\n"
    )
    assert read_history(path, 1) == [{"_step": 1}]


def test_single_unterminated_valid_record_is_readable(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"_step": 5}')
    reader = JsonlReader(path)
    assert reader.last_record() == {"_step": 5}
    assert reader.tail(1, reader.size + 100) == [{"_step": 5}]
    assert reader.tail_raw(1, 0) == []


@pytest.mark.benchmark
def test_merged_tail_memory_is_bounded_by_selection(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_rows(
        path,
        ({"_step": row // 2, "payload": "x" * 1000} for row in range(20_000)),
    )
    reader = JsonlReader(path)
    assert reader.merge
    tracemalloc.start()
    started = time.perf_counter()
    try:
        rows = reader.tail(5)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    elapsed = time.perf_counter() - started
    assert [row["_step"] for row in rows] == list(range(9995, 10_000))
    assert peak < 2 * 1024 * 1024
    print(f"\nMerged tail over 20k rows: {elapsed:.3f}s, peak {peak / 1024:.1f} KiB")


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
    size_getter = JsonlReader.size.fget
    appended = False

    def append_after_stat(reader):
        nonlocal appended
        size = size_getter(reader)
        if reader.path == path and not appended:
            appended = True
            with path.open("ab") as writer:
                writer.write(b'{"_step": 0, "patch": 3}\n')
            if meta_state == "updated":
                write_meta(path, index=[[0, 0, 0]], sorted=False)
        return size

    monkeypatch.setattr(JsonlReader, "size", property(append_after_stat))
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
    size_getter = JsonlReader.size.fget
    appended = False

    def append_after_stat(reader):
        nonlocal appended
        size = size_getter(reader)
        if reader.path == path and not appended:
            appended = True
            with path.open("ab") as writer:
                writer.write((b"" if complete else b"}") + b'\n{"_step": 2}\n')
        return size

    monkeypatch.setattr(JsonlReader, "size", property(append_after_stat))
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
    valid_meta = reader_module._valid_meta

    def pause_after_validation(meta, size):
        valid = valid_meta(meta, size)
        entered.set()
        assert release.wait(5), "metadata read was not released"
        return valid

    def first_query():
        try:
            results.append(reader.tail(1))
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(reader_module, "_valid_meta", pause_after_validation)
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
