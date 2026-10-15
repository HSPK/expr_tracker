"""Offline reader corruption, metadata and physical-row boundaries."""

import json
import sys
import time
import tracemalloc

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
from expr_tracker.history import reader as reader_module


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


@pytest.mark.parametrize("query", ["read_all", "parse_rows"])
def test_native_reader_reports_corrupted_rows(tmp_path, reader_warnings, query):
    path = tmp_path / "metrics.jsonl"
    payload = write_rows(path, [{"_step": 1, "value": 2}]) + b"{broken\n"
    path.write_bytes(payload)
    reader = JsonlReader(path)
    if query == "read_all":
        rows = reader.read_all()
    else:
        rows = reader.parse_rows(payload.splitlines())
    assert rows == [{"_step": 1, "value": 2}]
    assert any(
        f"Skipping corrupted line in {path}" in record["message"]
        for record in reader_warnings
    )


@pytest.mark.parametrize("contents", ["{broken", "[]", "null"])
def test_native_reader_reports_corrupted_metadata(tmp_path, reader_warnings, contents):
    path = tmp_path / "metrics.jsonl"
    write_rows(path, [{"_step": 1}])
    path.with_suffix(".meta.json").write_text(contents)
    assert JsonlReader(path).read_all() == [{"_step": 1}]
    assert any(
        "metadata" in record["message"]
        and ("corrupted" in record["message"] or "malformed" in record["message"])
        for record in reader_warnings
    )


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
    reader = JsonlReader(path, index=[], merge=False)
    captured = reader._snapshot()
    with path.open("ab") as writer:
        writer.write(b"2}\n")
    assert captured.read_all() == [{"_step": 0}]
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
    reader._native.clear_stats()
    assert reader.read_steps(3998, 4000) == [{"_step": 3998}, {"_step": 3999}]
    assert 0 < reader._native.scanned_rows <= 1000


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
    meta_path.unlink()
    meta_path.mkdir()
    merged = {"_step": 1, "original": 1, "patch": 2}
    assert read_history(path, 1) == [merged]
    assert read_history(path, None, step_range=(1, 3)) == [merged, {"_step": 2}]
    assert len(reader_warnings) == 2
    assert all(
        record["level"].name == "WARNING"
        and str(meta_path) in record["message"]
        and "directory" in record["message"].lower()
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
    original_query = JsonlReader._query

    def deny_data(reader, *args):
        if reader.path == path:
            raise error("metrics denied")
        return original_query(reader, *args)

    monkeypatch.setattr(JsonlReader, "_query", deny_data)
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


def test_disabled_python_integer_limit_preserves_arbitrary_steps_and_values(tmp_path):
    previous = getattr(sys, "get_int_max_str_digits", lambda: 0)()
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    try:
        huge = 10**5000
        path = tmp_path / "metrics.jsonl"
        rows = [{"_step": huge, "value": huge + 1}, {"_step": huge + 1, "value": -huge}]
        write_rows(path, rows)
        assert read_history(path, None) == rows
        assert read_history(path, 1) == rows[-1:]
        assert read_history(path, None, step_range=(huge, huge + 1)) == rows[:1]
    finally:
        if hasattr(sys, "set_int_max_str_digits"):
            sys.set_int_max_str_digits(previous)


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
