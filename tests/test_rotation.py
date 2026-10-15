import json

import pytest

import expr_tracker as et
from expr_tracker.history import HistoryStore, JsonlReader, JsonlWriter, read_history
from expr_tracker.history.rotation import DEFAULT_MAX_FILE_BYTES, segments
from expr_tracker.trace import build_trace, read_spans


def line(_step, **fields):
    return json.dumps({"_step": _step, **fields}).encode() + b"\n"


def write(writer, records):
    start = writer.lines
    for row, record in enumerate(records, start):
        writer.enqueue(record["_step"], row, line(**record))
    writer.flush()


def test_default_and_validation(tmp_path):
    writer = JsonlWriter(tmp_path / "metrics.jsonl")
    assert writer.max_file_bytes == DEFAULT_MAX_FILE_BYTES == 100 << 20
    writer.close()
    for invalid in (0, -1, True, 1.5, "100"):
        with pytest.raises(ValueError, match="max_file_bytes"):
            JsonlWriter(tmp_path / "metrics.jsonl", max_file_bytes=invalid)
        with pytest.raises(ValueError, match="max_file_bytes"):
            HistoryStore().init("p", "n", dir=str(tmp_path), max_file_bytes=invalid)


@pytest.mark.parametrize("async_write", [False, True])
def test_batch_rotation_and_resume(tmp_path, async_write):
    path = tmp_path / "metrics.jsonl"
    records = [{"_step": step, "v": step} for step in range(20)]
    writer = JsonlWriter(
        path, max_file_bytes=80, max_buffer_seconds=None, async_write=async_write
    )
    write(writer, records)
    writer.close()
    assert len(segments(path)) > 2
    assert all(part.stat().st_size <= 80 for part in segments(path))
    reader = JsonlReader(path)
    assert reader.read_all() == records
    assert reader.tail(3) == records[-3:]
    assert reader.read_steps(3, 8) == records[3:8]
    assert reader.count_lines() == 20
    for row in range(21):
        boundary = reader.offset_of_line(row)
        assert reader.tail_rows(3, end=boundary) == records[max(0, row - 3) : row]
        assert list(reader.iter_raw(end=boundary)) == [
            line(**record) for record in records[:row]
        ]
        if boundary:
            assert list(reader.iter_raw(end=boundary - 1)) == [
                line(**record) for record in records[: row - 1]
            ]
    assert list(reader.iter_raw(reader.offset_of_line(7))) == [
        line(**record) for record in records[7:]
    ]
    resumed = JsonlWriter(path, max_file_bytes=None)
    assert resumed.lines == 20
    assert resumed.max_step == 19
    write(resumed, [{"_step": 20, "v": 20}])
    resumed.close()
    assert JsonlReader(path).read_all() == [*records, {"_step": 20, "v": 20}]


def test_boundary_oversize_and_disabled(tmp_path):
    raw = line(0, v="x")
    path = tmp_path / "metrics.jsonl"
    writer = JsonlWriter(path, max_file_bytes=len(raw))
    write(
        writer,
        [
            {"_step": 0, "v": "x"},
            {"_step": 1, "v": "x" * 100},
            {"_step": 2, "v": "x"},
        ],
    )
    writer.close()
    assert [part.stat().st_size for part in segments(path)] == [
        len(raw),
        len(line(1, v="x" * 100)),
        len(raw),
    ]
    disabled = JsonlWriter(tmp_path / "disabled.jsonl", max_file_bytes=None)
    write(disabled, [{"_step": i} for i in range(10)])
    disabled.close()
    assert len(segments(disabled.path)) == 1


def test_snapshot_and_patches_across_segments(tmp_path):
    path = tmp_path / "metrics.jsonl"
    writer = JsonlWriter(path, max_file_bytes=30)
    write(writer, [{"_step": 0, "a": 1}, {"_step": 1, "a": 2}])
    snapshot = writer.reader()._snapshot()
    size = snapshot.size
    write(writer, [{"_step": 0, "b": 3}, {"_step": 2, "a": 4}])
    assert snapshot.size == size
    assert snapshot.read_all() == [{"_step": 0, "a": 1}, {"_step": 1, "a": 2}]
    assert writer.reader().tail(2) == [
        {"_step": 0, "a": 1, "b": 3},
        {"_step": 2, "a": 4},
    ]
    writer.close()
    path.with_name("metrics.meta.json").unlink()
    resumed = JsonlWriter(path, max_file_bytes=30)
    assert resumed.lines == 4
    assert resumed.max_step == 2
    assert not resumed.sorted
    resumed.close()


def test_partial_write_rollback_across_segments(tmp_path, monkeypatch):
    writer = JsonlWriter(
        tmp_path / "metrics.jsonl", max_file_bytes=30, max_buffer_seconds=None
    )
    write(writer, [{"_step": 0}])
    original = writer._append
    calls = 0

    def fail(payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            original(payload[:5])
            raise OSError("disk unavailable")
        original(payload)

    monkeypatch.setattr(writer, "_append", fail)
    write(writer, [{"_step": i} for i in range(1, 5)])
    assert writer.reader().read_all() == [{"_step": 0}]
    monkeypatch.setattr(writer, "_append", original)
    writer.flush()
    writer.close()
    assert JsonlReader(writer.path).read_all() == [{"_step": i} for i in range(5)]


def test_native_vectored_rotation_failure_rolls_back_and_retries(tmp_path):
    path = tmp_path / "metrics.jsonl"
    raw = line(0)
    writer = JsonlWriter(
        path, max_file_bytes=len(raw), max_buffer_seconds=None, buffer_size=1000
    )
    records = [{"_step": step} for step in range(4)]
    write(writer, records[:1])
    blocked = path.with_name(path.name + ".parts") / "00000002.jsonl"
    blocked.parent.mkdir(exist_ok=True)
    blocked.mkdir()
    try:
        write(writer, records[1:])
        assert path.read_bytes() == raw
        assert not blocked.with_name("00000001.jsonl").exists()
        assert len(writer.buffer) == 3 and writer.lines == 1
        assert writer.pending_bytes == sum(len(line(**row)) for row in records[1:])
        blocked.rmdir()
        writer.flush()
        assert writer.reader().read_all() == records
        assert writer.pending_bytes == writer.dropped == 0
        assert all(part.stat().st_size == len(raw) for part in segments(path))
    finally:
        if blocked.is_dir():
            blocked.rmdir()
        writer.close()


def test_external_multiline_and_split_payloads_preserve_rotation_boundaries(tmp_path):
    writer = JsonlWriter(
        tmp_path / "metrics.jsonl", max_file_bytes=30, max_buffer_seconds=None
    )
    rows = [line(step) for step in range(6)]
    for ordinal, payload in enumerate(
        [
            rows[0] + rows[1],
            rows[2][:4],
            rows[2][4:] + rows[3],
            rows[4] + rows[5],
        ]
    ):
        writer.enqueue(ordinal, ordinal, payload)
    writer.flush()
    writer.close()
    assert b"".join(part.read_bytes() for part in segments(writer.path)) == b"".join(
        rows
    )
    assert all(part.stat().st_size <= 30 for part in segments(writer.path))


def test_live_offline_and_torn_tail(tmp_path):
    store = HistoryStore().init(
        "p",
        "n",
        dir=str(tmp_path),
        max_file_bytes=100,
        cache_rows=1,
        cache_bytes=100,
        max_buffer_seconds=None,
        buffer_size=1,
    )
    for step in range(12):
        store.log({"v": step}, step=step, commit=True)
    store.flush()
    assert [row["v"] for row in store.get(-1, output_type="records")] == list(range(12))
    path = store.log_fp
    store.finish()
    assert [row["v"] for row in read_history(path, -1, output_type="records")] == list(
        range(12)
    )
    with segments(path)[-1].open("ab") as source:
        source.write(b'{"_step":')
    resumed = JsonlWriter(path, max_file_bytes=100)
    assert resumed.lines == 12
    assert resumed.max_step == 11
    resumed.close()


def test_spans_include_rotated_records(tmp_path):
    path = tmp_path / "spans.jsonl"
    writer = JsonlWriter(path, max_file_bytes=100)
    records = [
        {"_step": i, "name": f"span{i}", "start": i, "dur_ms": 1} for i in range(5)
    ]
    write(writer, records)
    writer.close()
    assert [span["name"] for span in read_spans(path)] == [
        record["name"] for record in records
    ]
    assert [
        event["name"]
        for event in build_trace(tmp_path)["traceEvents"]
        if event["ph"] == "X"
    ] == [record["name"] for record in records]


@pytest.mark.parametrize("rank", [0, 2])
def test_public_history_stream_rank_spans_and_resume(run, monkeypatch, rank):
    monkeypatch.setenv("RANK", str(rank))
    instance = run(
        stream="data",
        max_file_bytes=100,
        cache_rows=1,
        buffer_size=1,
        max_buffer_seconds=None,
    )
    for step in range(8):
        with et.span("work"):
            pass
        et.log({"v": step})
    instance.history.flush()
    assert instance.history.writer.max_file_bytes == 100
    assert instance.history.span_writer.max_file_bytes == 100
    path = instance.history.log_fp
    span_path = instance.history.spans_fp
    assert len(segments(path)) > 1
    assert [row["v"] for row in et.history(-1, output_type="records")] == list(range(8))
    et.finish()
    assert len(read_spans(span_path)) == 8
    assert [
        row["v"]
        for row in et.history(-1, run=path.parent, stream="data", output_type="records")
    ] == list(range(8))
    resumed = run(stream="data", max_file_bytes=100, cache_rows=1, buffer_size=1)
    assert resumed.history.current_step == 8
    et.log({"v": 8})
    assert [row["v"] for row in et.history(-1, output_type="records")] == list(range(9))
