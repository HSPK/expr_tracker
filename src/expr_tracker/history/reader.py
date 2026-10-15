"""Python value/output adapters for the native segmented JSONL reader."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from .._native import NativeReader, history_merge, history_parse, history_recent_steps
from .codec import encode_line
from .frame import project_owned as project
from .frame import to_output
from .naming import RANK_PATTERN, metrics_filename

CHUNK_SIZE = 256 * 1024


def _integer_limit():
    return getattr(sys, "get_int_max_str_digits", lambda: 0)()


def _decode(records):
    result = []
    for offset in range(0, len(records), 1024):
        batch = records[offset : offset + 1024]
        try:
            result.extend(json.loads("[" + ",".join(batch) + "]"))
        except (ValueError, RecursionError) as error:
            logger.warning(
                f"Could not decode history batch; retrying individual rows: {error}"
            )
            for record in batch:
                try:
                    result.append(json.loads(record))
                except (ValueError, RecursionError) as error:
                    logger.warning(f"Skipping corrupted history record: {error}")
    return result


class JsonlReader:
    """Snapshot-consistent, lossless reader backed entirely by Rust."""

    def __init__(self, path, *, index=None, merge=None):
        self.path = Path(path)
        self._native = NativeReader(
            self.path,
            None if index is None else [(str(s), line, o) for s, line, o in index],
            merge,
            _integer_limit(),
        )

    def _snapshot(self, end=None):
        reader = object.__new__(JsonlReader)
        reader.path = self.path
        reader._native = self._native.snapshot(end)
        return reader

    @property
    def size(self):
        return self._native.size

    @property
    def merge(self):
        value = self._native.merge
        self._warnings()
        return value

    @property
    def index(self):
        value = [(int(s), line, o) for s, line, o in self._native.index]
        self._warnings()
        return value

    def _warnings(self):
        for warning in self._native.take_warnings():
            logger.warning(warning)

    def _query(self, *args):
        reader = self._snapshot()
        try:
            return _decode(reader._native.query(*args))
        finally:
            reader._warnings()

    def _scan(self, start=0, end=None):
        yield from self._native.scan(start, end)

    def iter_raw(self, start=0, end=None):
        yield from self._native.iter_raw(start, end)

    def tail_raw(self, count, end=None):
        return self._native.tail_raw(max(0, count), end)

    def _reverse_raw(self, end=None):
        yield from reversed(self.tail_raw(self.count_lines(), end))

    def offset_of_step(self, step):
        try:
            return self._native.offset_of_step(str(step))
        finally:
            self._warnings()

    def offset_of_line(self, line_no):
        return self._native.offset_of_line(max(0, line_no))

    def parse_rows(self, lines):
        try:
            return _decode(self._native.parse_rows(list(lines), _integer_limit()))
        finally:
            self._warnings()

    def _iter_rows(self, lines):
        yield from self.parse_rows(lines)

    def parse(self, lines):
        rows = self.parse_rows(lines)
        return merge_steps(rows) if self.merge else rows

    def tail_rows(self, count, end=None):
        return self.parse_rows(self.tail_raw(count, end))

    def tail(self, count, end=None):
        return self._query(max(0, count), None, None, end)

    def select_recent_steps(self, count, end=None, *, newer=()):
        selected = recent_steps(newer, count)
        try:
            return {
                int(step)
                for step in self._native.select_recent(
                    max(0, count), end, [str(step) for step in selected]
                )
            }
        finally:
            self._warnings()

    def read_selected_steps(self, selected, end=None):
        try:
            return _decode(self._native.selected([str(step) for step in selected], end))
        finally:
            self._warnings()

    def read_steps(self, start, end, stop_offset=None):
        return self._query(
            None,
            None if start is None else str(start),
            None if end is None else str(end),
            stop_offset,
        )

    def read_all(self):
        return self._query()

    def last_record(self):
        records = self.parse(self.tail_raw(1))
        return records[-1] if records else None

    def count_lines(self):
        return self._native.count_lines()


def recent_steps(steps: Iterable[int], count: int) -> set[int]:
    return {int(step) for step in history_recent_steps(steps, max(0, count))}


def merge_steps(records: Iterable[dict], *, selected=None) -> list[dict]:
    return _decode(
        history_merge(
            [encode_line(record).decode("utf-8") for record in records],
            None if selected is None else [str(step) for step in selected],
        )
    )


def _peek_step(raw):
    rows = history_parse([raw], _integer_limit())
    return json.loads(rows[0])["_step"] if rows else None


def read_history(
    run,
    n=50,
    *,
    stream=None,
    output_type="dict",
    metrics=None,
    step_range=None,
    include_meta=True,
    fill_missing=False,
    dropna=False,
):
    reader = JsonlReader(resolve_run_path(run, stream))
    limit = None if n is None or n < 0 else max(0, n)
    if limit == 0:
        records = []
    elif step_range is not None:
        records = reader.read_steps(*step_range)
        if limit is not None:
            records = records[-limit:]
    elif limit is None:
        records = reader.read_all()
    else:
        records = reader.tail(limit)
    return to_output(
        project(
            records,
            metrics=metrics,
            include_meta=include_meta,
            fill_missing=fill_missing,
            dropna=dropna,
        ),
        output_type,
    )


def resolve_run_path(run, stream=None):
    path = Path(run)
    if not path.is_dir():
        if not path.exists():
            raise FileNotFoundError(f"Run path does not exist: {path}")
        return path
    wanted = metrics_filename(stream, rank_aware=False)
    exact = path / wanted
    if exact.is_file():
        return exact
    prefix = wanted[: -len(".jsonl")]
    shards = sorted(path.glob(f"{prefix}.rank*.jsonl"))
    if shards:
        return shards[0]
    raise FileNotFoundError(f"No {wanted} found under {path}; {_available(path)}")


def _available(path):
    found = sorted(p.name for p in path.glob("metrics*.jsonl"))
    return f"available: {', '.join(found)}" if found else "the directory has none"


def list_streams(run):
    streams = set()
    for path in Path(run).glob("metrics*.jsonl"):
        parts = path.name.split(".")[1:-1]
        if parts and RANK_PATTERN.fullmatch(parts[-1]):
            parts = parts[:-1]
        streams.add(parts[0] if parts else None)
    return sorted(streams, key=lambda s: (s is not None, s or ""))
