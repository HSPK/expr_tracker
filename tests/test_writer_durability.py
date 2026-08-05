"""Unit tests for writer/reader durability: torn files, bad meta, index limits."""

import json
import os

import pytest

from expr_tracker.history import JsonlWriter, read_history, resolve_run_path
from expr_tracker.history.writer import MAX_INDEX_ANCHORS


def line_for(step: int, **fields) -> bytes:
    return json.dumps({"_step": step, "_time": float(step), **fields}).encode() + b"\n"


def write_steps(writer: JsonlWriter, steps, row_start: int = 0):
    for offset, step in enumerate(steps):
        writer.enqueue(step, row_start + offset, line_for(step, v=step))
    writer.flush()


@pytest.fixture
def writer(tmp_path):
    created = []

    def factory(name="metrics.jsonl", **kwargs):
        instance = JsonlWriter(tmp_path / name, **kwargs)
        created.append(instance)
        return instance

    yield factory
    for instance in created:
        instance.close()


# ------------------------------------------------------------------ torn tails


@pytest.mark.parametrize(
    "tail",
    [
        b'{"_step": 3, "v": ',  # cut mid-object
        b'{"_step": 3, "v": 3}',  # complete but unterminated
        b"\x00\x00\x00",  # garbage
    ],
)
def test_a_torn_final_line_is_dropped_on_open(writer, tmp_path, tail):
    first = writer()
    write_steps(first, [0, 1, 2])
    first.close()
    with open(tmp_path / "metrics.jsonl", "ab") as f:
        f.write(tail)
    (tmp_path / "metrics.meta.json").unlink(missing_ok=True)  # force a disk rebuild

    second = writer()
    assert second.lines == 3
    assert second.last_step == 2
    for text in (tmp_path / "metrics.jsonl").read_text().splitlines():
        json.loads(text)


def test_an_empty_trailing_line_is_not_treated_as_torn(writer, tmp_path):
    first = writer()
    write_steps(first, [0, 1])
    first.close()
    assert (tmp_path / "metrics.jsonl").read_bytes().endswith(b"\n")
    assert writer().lines == 2


def test_a_file_of_only_garbage_yields_no_rows(writer, tmp_path):
    (tmp_path / "metrics.jsonl").write_bytes(b"not json at all")
    assert writer().lines == 0


# ------------------------------------------------------------------ sidecar


def test_a_stale_sidecar_is_ignored(writer, tmp_path):
    first = writer()
    write_steps(first, [0, 1, 2])
    first.close()
    meta = json.loads((tmp_path / "metrics.meta.json").read_text())
    meta["size"] = 999999  # no longer matches the file
    (tmp_path / "metrics.meta.json").write_text(json.dumps(meta))

    second = writer()
    assert second.lines == 3  # rebuilt from disk instead of trusted


def test_a_sidecar_from_another_schema_is_ignored(writer, tmp_path):
    first = writer()
    write_steps(first, [0, 1])
    first.close()
    meta = json.loads((tmp_path / "metrics.meta.json").read_text())
    meta["schema"] = "ancient"
    (tmp_path / "metrics.meta.json").write_text(json.dumps(meta))
    assert writer().lines == 2


@pytest.mark.parametrize("content", ["", "{", "[]", "null"])
def test_a_corrupt_sidecar_is_ignored(writer, tmp_path, content):
    first = writer()
    write_steps(first, [0, 1])
    first.close()
    (tmp_path / "metrics.meta.json").write_text(content)
    assert writer().lines == 2


def test_an_unwritable_sidecar_does_not_break_flushing(writer, tmp_path, monkeypatch):
    instance = writer()
    monkeypatch.setattr(
        instance, "meta_path", tmp_path / "missing-dir" / "metrics.meta.json"
    )
    write_steps(instance, [0, 1, 2])
    assert instance.lines == 3
    assert len(read_history(tmp_path, -1)) == 3


# ------------------------------------------------------------------ resume


def test_resume_continues_the_row_and_step_counters(writer):
    first = writer()
    write_steps(first, [0, 1, 2])
    first.close()

    second = writer()
    assert (second.lines, second.last_step) == (3, 2)
    write_steps(second, [3, 4], row_start=3)
    assert second.lines == 5
    assert [r["_step"] for r in second.reader().read_all()] == [0, 1, 2, 3, 4]


def test_resume_detects_pre_existing_duplicate_steps(writer, tmp_path):
    first = writer()
    write_steps(first, [0, 1, 1, 2])
    first.close()
    (tmp_path / "metrics.meta.json").unlink(missing_ok=True)

    second = writer()
    assert second.has_duplicate_steps is True
    assert second.reader().merge is True
    assert [r["_step"] for r in second.reader().read_all()] == [0, 1, 2]


def test_resume_detects_out_of_order_steps(writer, tmp_path):
    first = writer()
    write_steps(first, [5, 1, 9])
    first.close()
    (tmp_path / "metrics.meta.json").unlink(missing_ok=True)
    assert writer().sorted is False


# ------------------------------------------------------------------ write fail


def test_a_failed_write_is_retried_not_lost(writer, monkeypatch):
    instance = writer(buffer_size=1)
    write_steps(instance, [0])

    calls = []

    def failing(payload):
        calls.append(payload)
        raise OSError("nope")

    monkeypatch.setattr(instance, "_write", failing)
    write_steps(instance, [1, 2], row_start=1)
    assert instance.lines == 1 and calls

    monkeypatch.undo()
    instance.flush()
    assert instance.lines == 3
    assert [r["_step"] for r in instance.reader().read_all()] == [0, 1, 2]


def test_an_unrollable_partial_write_forces_merge_mode(writer, monkeypatch):
    instance = writer(buffer_size=1)
    write_steps(instance, [0])
    monkeypatch.setattr(
        instance, "_write", lambda payload: (_ for _ in ()).throw(OSError("nope"))
    )
    monkeypatch.setattr(instance, "_truncate_partial_write", lambda size: False)
    write_steps(instance, [1], row_start=1)
    assert instance.has_duplicate_steps is True  # reader must de-duplicate


def test_the_buffer_drops_the_oldest_records_when_it_overflows(writer, monkeypatch):
    instance = writer(buffer_size=1, max_pending_records=3)
    monkeypatch.setattr(
        instance, "_write", lambda payload: (_ for _ in ()).throw(OSError("nope"))
    )
    for step in range(10):
        instance.enqueue(step, step, line_for(step))
        instance.flush()
    assert len(instance.buffer) == 3
    assert instance.dropped == 7

    monkeypatch.undo()
    instance.flush()
    assert [r["_step"] for r in instance.reader().read_all()] == [7, 8, 9]


def test_a_deleted_parent_directory_is_recreated(writer, tmp_path):
    instance = writer("sub/metrics.jsonl", buffer_size=1)
    write_steps(instance, [0])
    for path in (tmp_path / "sub").iterdir():
        path.unlink(missing_ok=True)
    (tmp_path / "sub").rmdir()
    write_steps(instance, [1], row_start=1)
    assert (tmp_path / "sub" / "metrics.jsonl").is_file()


def test_enqueue_after_close_is_refused(writer):
    instance = writer()
    write_steps(instance, [0])
    instance.close()
    assert instance.enqueue(1, 1, line_for(1)) is False


# ------------------------------------------------------------------ index


def test_the_index_is_halved_instead_of_growing_forever(writer):
    instance = writer(index_every=1)
    total = MAX_INDEX_ANCHORS + 50
    for step in range(total):
        instance.enqueue(step, step, line_for(step))
    instance.flush()
    assert len(instance.index) <= MAX_INDEX_ANCHORS
    assert instance.index_every >= 2
    # a halved index must still locate rows correctly
    reader = instance.reader()
    assert [r["_step"] for r in reader.read_steps(total - 3, total)] == [
        total - 3,
        total - 2,
        total - 1,
    ]
    assert reader.tail(1)[0]["_step"] == total - 1


def test_a_rebuilt_index_survives_a_restart(writer, tmp_path):
    first = writer(index_every=4)
    write_steps(first, list(range(40)))
    first.close()
    (tmp_path / "metrics.meta.json").unlink(missing_ok=True)

    second = writer(index_every=4)
    assert [r["_step"] for r in second.reader().read_steps(20, 23)] == [20, 21, 22]


# ------------------------------------------------------------------ reader


def test_reading_a_missing_run_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_history(tmp_path / "nowhere", -1)
    with pytest.raises(FileNotFoundError):
        read_history(tmp_path, -1)  # a directory without metrics*.jsonl


def test_reading_a_file_with_interleaved_garbage(writer, tmp_path):
    instance = writer()
    write_steps(instance, [0, 1])
    instance.close()
    with open(tmp_path / "metrics.jsonl", "ab") as f:
        f.write(b"garbage\n")
        f.write(line_for(2))
    (tmp_path / "metrics.meta.json").unlink(missing_ok=True)

    rows = read_history(tmp_path, -1)
    assert [r["_step"] for r in rows] == [0, 1, 2]


def test_rows_without_a_step_are_skipped(writer, tmp_path):
    instance = writer()
    write_steps(instance, [0])
    instance.close()
    with open(tmp_path / "metrics.jsonl", "ab") as f:
        f.write(json.dumps({"no": "step"}).encode() + b"\n")
        f.write(b"[1, 2, 3]\n")  # valid json, wrong shape
    (tmp_path / "metrics.meta.json").unlink(missing_ok=True)
    assert [r["_step"] for r in read_history(tmp_path, -1)] == [0]


def test_resolve_run_path_accepts_a_directory_or_a_file(writer, tmp_path):
    instance = writer()
    write_steps(instance, [0])
    instance.close()
    assert resolve_run_path(tmp_path) == tmp_path / "metrics.jsonl"
    assert resolve_run_path(tmp_path / "metrics.jsonl") == tmp_path / "metrics.jsonl"


def test_resolve_run_path_rejects_a_directory_without_metrics(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_run_path(tmp_path)


def test_reading_a_truncated_file_mid_query(writer, tmp_path):
    """A file shrinking under a live reader must not raise."""
    instance = writer()
    write_steps(instance, list(range(20)))
    reader = instance.reader()
    assert len(reader.read_all()) == 20
    os.truncate(tmp_path / "metrics.jsonl", 30)
    assert isinstance(reader.tail(5), list)
