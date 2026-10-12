"""Export recorded spans as a Chrome Trace, readable by Perfetto and chrome://tracing.

Writing the format rather than a viewer means the timeline can sit next to a
``torch.profiler`` trace in the same window, which is where the interesting
question usually is: what were the GPUs doing while the data loader stalled.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from heapq import heapify, heappop, heappush
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

from .history.naming import parse_stream, spans_filename
from .history.reader import JsonlReader

MICROSECONDS = 1_000_000.0


def span_files(run: str | Path, stream: str | list[str] | None = "*") -> list[Path]:
    """Span files of a run: every stream by default, or the ones named."""
    directory = Path(run)
    if directory.is_file():
        return [directory]
    if stream == "*":
        # Default producer first: "spans.data.jsonl" sorts before "spans.jsonl"
        return sorted(
            directory.glob("spans*.jsonl"),
            key=lambda p: (parse_stream(p.name) is not None, p.name),
        )
    wanted = [stream] if isinstance(stream, (str, type(None))) else list(stream)
    found = []
    for name in wanted:
        path = directory / spans_filename(name, rank_aware=False)
        if not path.is_file():
            raise FileNotFoundError(f"No {path.name} under {directory}")
        found.append(path)
    return found


def _time_bounds(
    time_range: tuple[float | None, float | None] | None,
) -> tuple[float | None, float | None] | None:
    if time_range is None:
        return None
    if len(time_range) != 2:
        raise ValueError("time_range must contain two epoch-second bounds")
    bounds = []
    for value in time_range:
        if value is None:
            bounds.append(None)
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("time_range bounds must be finite epoch seconds or None")
        else:
            try:
                bounds.append(_number(value))
            except (ValueError, OverflowError) as exc:
                raise ValueError(
                    "time_range bounds must be finite epoch seconds or None"
                ) from exc
    return bounds[0], bounds[1]


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("not a numeric timestamp")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("not a finite timestamp")
    return number


def _iter_spans(
    path: Path,
    *,
    step_range: tuple[int | None, int | None] | None = None,
    time_range: tuple[float | None, float | None] | None = None,
) -> Iterator[dict]:
    time_range = _time_bounds(time_range)
    for bounds in (step_range, time_range):
        if bounds is not None:
            low, high = bounds
            if low is not None and high is not None and low >= high:
                return
    if not path.is_file():
        return
    # Binary lines isolate invalid UTF-8 to one record, like malformed JSON.
    with closing(JsonlReader(path).iter_raw()) as source:
        for line in source:
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                start = _number(record.get("start"))
                duration = _number(record.get("dur_ms"))
                end = start + duration / 1000.0
                if duration < 0 or not all(
                    math.isfinite(value)
                    for value in (end, start * MICROSECONDS, duration * 1000.0)
                ):
                    continue
                if time_range is not None:
                    low, high = time_range
                    # Zero-duration spans are empty intervals, not point events.
                    if (
                        duration == 0
                        or (low is not None and end <= low)
                        or (high is not None and start >= high)
                    ):
                        continue
                if step_range is not None:
                    low, high = step_range
                    step = record.get("_step", 0)
                    if isinstance(step, bool) or not isinstance(step, int):
                        step = _number(step)
                    if (low is not None and step < low) or (
                        high is not None and step >= high
                    ):
                        continue
                if any(
                    record.get(key) is not None and not isinstance(record[key], dict)
                    for key in ("args", "metrics")
                ):
                    continue
                track = record.get("track", 0)
                if not isinstance(track, (int, float, str)) or (
                    isinstance(track, float) and not math.isfinite(track)
                ):
                    continue
                depth = record.get("depth")
                if depth is not None and (
                    isinstance(depth, bool) or not isinstance(depth, int) or depth < 0
                ):
                    continue
            except (ValueError, OverflowError, UnicodeError, RecursionError):
                continue
            yield record


def read_spans(
    path: Path,
    *,
    step_range: tuple[int | None, int | None] | None = None,
    time_range: tuple[float | None, float | None] | None = None,
) -> list[dict]:
    """Incrementally decode a file into a list, skipping malformed span records.

    Filters are applied before retaining records. ``step_range`` is [low, high).
    ``time_range`` uses absolute epoch seconds and selects intervals overlapping
    [low, high), without clipping them; either bound may be None. Equal or reversed
    bounds select nothing. Zero-duration spans are excluded by a time filter.
    Finite numeric strings for start/dur_ms are accepted without changing the
    returned records; negative durations and invalid timestamps are skipped.
    Memory is proportional to selected records, plus the largest input line.
    """
    return list(_iter_spans(path, step_range=step_range, time_range=time_range))


class _LanePool:
    """Online stack packing in trace microseconds, O(log n) amortized per event."""

    def __init__(self):
        self.stacks: list[list[tuple[float, int | None]]] = []
        self.versions: list[int] = []
        self.ends: list[tuple[float, int]] = []
        self.free: list[int] = []
        self.available: dict[int | None, list[tuple[float, int, int]]] = {}
        self.live: dict[int | None, int] = {}

    def _valid(self, item: tuple[float, int, int]) -> bool:
        _end, lane, version = item
        return self.versions[lane] == version

    def _clean(self, depth: int | None):
        heap = self.available[depth]
        # Lazy heap deletion must not retain a whole long-running worker's past.
        if len(heap) > 2 * self.live[depth] + 64:
            heap[:] = [item for item in heap if self._valid(item)]
            heapify(heap)
        while heap and not self._valid(heap[0]):
            heappop(heap)
        if not heap:
            del self.available[depth]
            del self.live[depth]

    def _remove_top(self, lane: int):
        depth = self.stacks[lane][-1][1]
        self.versions[lane] += 1
        self.live[depth] -= 1
        self._clean(depth)

    def _add_top(self, lane: int):
        end, depth = self.stacks[lane][-1]
        self.live[depth] = self.live.get(depth, 0) + 1
        heappush(
            self.available.setdefault(depth, []),
            (-end, lane, self.versions[lane]),
        )

    def place(self, start: float, duration: float, depth: int | None) -> int:
        end = start + duration * 1000.0
        if not math.isfinite(end):
            raise ValueError("selected span timestamps exceed finite trace range")
        while self.ends and self.ends[0][0] <= start:
            _expired, lane = heappop(self.ends)
            stack = self.stacks[lane]
            if not stack or stack[-1][0] > start:
                continue
            self._remove_top(lane)
            while stack and stack[-1][0] <= start:
                stack.pop()
            if stack:
                self._add_top(lane)
            else:
                heappush(self.free, lane)

        # Recorded siblings cannot nest, even when their wall-clock times do.
        # Unknown-depth legacy records retain temporal nesting among themselves.
        parent_depth = None if depth is None else depth - 1
        candidates = self.available.get(parent_depth)
        lane = None
        if candidates and -candidates[0][0] >= end:
            lane = candidates[0][1]
        if self.free and (lane is None or self.free[0] < lane):
            lane = heappop(self.free)
        elif lane is not None:
            self._remove_top(lane)
        else:
            lane = len(self.stacks)
            self.stacks.append([])
            self.versions.append(0)
        self.stacks[lane].append((end, depth))
        self._add_top(lane)
        heappush(self.ends, (end, lane))
        return lane


def _relative_start(start: float, origin: float) -> float:
    timestamp = start * MICROSECONDS - origin * MICROSECONDS
    if not math.isfinite(timestamp):
        raise ValueError("selected span timestamps exceed finite trace range")
    return timestamp


def _pack(
    spans: list[dict], indices: list[int], offset: int, origin: float = 0.0
) -> dict[int, int]:
    """Lay one track's spans onto lanes in O(n log n), each lane a valid stack."""
    order = sorted(
        indices,
        key=lambda i: (
            float(spans[i]["start"]),
            -float(spans[i]["dur_ms"]),
            spans[i].get("depth") is not None,
            spans[i].get("depth") or 0,
        ),
    )
    pool = _LanePool()
    placement: dict[int, int] = {}
    for index in order:
        record = spans[index]
        placement[index] = offset + pool.place(
            _relative_start(float(record["start"]), origin),
            float(record["dur_ms"]),
            record.get("depth"),
        )
    return placement


def assign_lanes(spans: list[dict]) -> dict[int, int]:
    """Place spans on lanes so each lane is a valid stack, keyed by list index.

    Threads come first: work on different threads is unrelated, and letting one
    contain the other would draw a nesting that never happened. Within a thread,
    crossing intervals get separate lanes. Known depths nest only immediately
    below the preceding depth; overlapping top-level spans and siblings never
    nest. Records without depth use temporal nesting with other legacy records.
    Sorting and packing take O(n log n) time and O(n) memory. Lane numbers are
    deterministic, but are not a persistent identity across different exports.
    """
    by_track: dict[Any, list[int]] = {}
    for index, record in enumerate(spans):
        by_track.setdefault(record.get("track", 0), []).append(index)
    placement: dict[int, int] = {}
    offset = 0
    origin = min((float(record["start"]) for record in spans), default=0.0)
    for track in sorted(by_track, key=str):
        lanes = _pack(spans, by_track[track], offset, origin)
        placement.update(lanes)
        offset = max(lanes.values(), default=offset - 1) + 1
    return placement


def build_trace(
    run: str | Path,
    stream: str | list[str] | None = "*",
    *,
    step_range: tuple[int | None, int | None] | None = None,
    time_range: tuple[float | None, float | None] | None = None,
    temp_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a Chrome Trace dict, one process per stream.

    ``step_range`` selects steps in [low, high). ``time_range`` is an absolute
    epoch-second interval [low, high), with None for either unbounded endpoint.
    Include every positive-duration span overlapping that interval, without
    clipping it. Spans ending at low or starting at high are excluded. Equal or
    reversed bounds select nothing; non-finite time bounds raise ValueError.
    Both filters must match. The origin is the earliest selected start across
    all streams, not the requested lower bound. Empty streams keep their metadata.

    Parsing/filtering is incremental and sorting uses temporary disk storage,
    but the returned dict necessarily retains all selected events in memory.
    Use write_trace to stream the result instead. ``temp_dir`` selects an existing
    directory for the temporary spool database; None uses the system default.
    """
    return {
        "traceEvents": list(
            _trace_events(
                run,
                stream,
                step_range=step_range,
                time_range=time_range,
                temp_dir=temp_dir,
            )
        ),
        "displayTimeUnit": "ms",
    }


def _stage_spans(
    db: sqlite3.Connection,
    run: str | Path,
    stream: str | list[str] | None,
    step_range: tuple[int | None, int | None] | None,
    time_range: tuple[float | None, float | None] | None,
) -> tuple[list[tuple[str, dict[Any, int]]], float]:
    db.executescript(
        """
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-2048;
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE spans (
            id INTEGER PRIMARY KEY, pid INTEGER, track INTEGER,
            start REAL, duration REAL, depth TEXT, record TEXT, lane INTEGER
        );
        """
    )
    streams: list[tuple[str, dict[Any, int]]] = []
    origin = math.inf
    for pid, path in enumerate(span_files(run, stream)):
        name = parse_stream(path.name) or "default"
        tracks: dict[Any, int] = {}
        streams.append((name, tracks))
        for record in _iter_spans(path, step_range=step_range, time_range=time_range):
            start = float(record["start"])
            origin = min(origin, start)
            track = record.get("track", 0)
            track_id = tracks.setdefault(track, len(tracks))
            depth = record.get("depth")
            db.execute(
                "INSERT INTO spans VALUES (NULL, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    pid,
                    track_id,
                    start,
                    float(record["dur_ms"]),
                    None if depth is None else str(depth),
                    json.dumps(record),
                ),
            )
    # Completion-order JSONL requires a start-time sort. SQLite spills the sort
    # to disk, and the index avoids sorting the (potentially large) JSON payloads.
    db.execute(
        "CREATE INDEX layout ON spans "
        "(pid, track, start, duration DESC, length(depth), depth, id)"
    )
    db.execute("CREATE INDEX output_order ON spans (pid, id)")
    db.commit()
    return streams, 0.0 if origin == math.inf else origin


def _metadata(name: str, pid: int, tid: int, label: str) -> dict:
    return {
        "name": name,
        "ph": "M",
        "pid": pid,
        "tid": tid,
        "args": {"name": label},
    }


def _trace_events(
    run: str | Path,
    stream: str | list[str] | None,
    *,
    step_range: tuple[int | None, int | None] | None,
    time_range: tuple[float | None, float | None] | None,
    temp_dir: str | Path | None = None,
) -> Iterator[dict]:
    time_range = _time_bounds(time_range)
    with (
        TemporaryDirectory(prefix="expr-trace-", dir=temp_dir) as temporary,
        closing(sqlite3.connect(str(Path(temporary) / "spans.db"))) as db,
    ):
        streams, origin = _stage_spans(db, run, stream, step_range, time_range)
        for pid, (name, tracks) in enumerate(streams):
            yield _metadata("process_name", pid, 0, name)
            offset = 0
            for track in sorted(tracks, key=str):
                pool = _LanePool()
                updates = []
                rows = db.execute(
                    "SELECT id, start, duration, depth FROM spans "
                    "WHERE pid = ? AND track = ? "
                    "ORDER BY start, duration DESC, length(depth), depth, id",
                    (pid, tracks[track]),
                )
                for index, start, duration, depth in rows:
                    lane = pool.place(
                        _relative_start(start, origin),
                        duration,
                        None if depth is None else int(depth),
                    )
                    updates.append((offset + lane, index))
                    if len(updates) == 1024:
                        db.executemany(
                            "UPDATE spans SET lane = ? WHERE id = ?", updates
                        )
                        updates.clear()
                db.executemany("UPDATE spans SET lane = ? WHERE id = ?", updates)
                label = f"thread {track}" if len(tracks) > 1 else name
                for lane in range(len(pool.stacks)):
                    yield _metadata("thread_name", pid, offset + lane, label)
                offset += len(pool.stacks)
                del pool
            rows = db.execute(
                "SELECT record, lane FROM spans WHERE pid = ? ORDER BY id", (pid,)
            )
            for payload, lane in rows:
                # Preserve diagnostic values without emitting non-standard JSON tokens.
                record = json.loads(payload, parse_constant=str)
                args = dict(record.get("args") or {})
                args.update(record.get("metrics") or {})
                args["step"] = record.get("_step")
                if record.get("error"):
                    args["error"] = record["error"]
                if len(tracks) > 1:
                    args["track"] = record.get("track", 0)
                yield {
                    "name": record.get("name", "span"),
                    "cat": name,
                    "ph": "X",
                    "pid": pid,
                    "tid": lane,
                    "ts": _relative_start(float(record["start"]), origin),
                    "dur": float(record["dur_ms"]) * 1000.0,
                    "args": args,
                }


def write_trace(
    run: str | Path,
    output: str | Path,
    stream: str | list[str] | None = "*",
    *,
    step_range: tuple[int | None, int | None] | None = None,
    time_range: tuple[float | None, float | None] | None = None,
    temp_dir: str | Path | None = None,
) -> int:
    """Stream a trace to disk and return its span count.

    ``time_range`` selects overlapping positive-duration intervals using absolute
    epoch seconds [low, high), with None for open bounds, and does not clip spans.
    Equal/reversed bounds are empty; non-finite bounds raise ValueError.
    ``step_range`` is also half-open; both filters must match. All streams share
    the earliest selected start as their origin, exactly as in build_trace.

    Records and the start-time sort are spooled to a temporary SQLite database;
    output is encoded one event at a time and atomically replaces the destination
    only on success. Temporary disk usage scales with selected input plus output.
    Python memory scales with the largest record, stream/thread metadata, peak
    lane count and active nesting/overlap, not all historical records. SQLite
    also uses its own page/sort buffers. Fully overlapping workloads still need
    O(n) lane state; this is not a constant-memory exporter.

    ``temp_dir`` selects an existing directory for the spool database. SQLite's
    internal temporary-file location remains environment/platform dependent.
    The temporary output stays beside the destination for atomic replacement.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    count = 0
    try:
        with (
            NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                delete=False,
                prefix=".expr-trace-",
                suffix=".tmp",
            ) as target,
            closing(
                _trace_events(
                    run,
                    stream,
                    step_range=step_range,
                    time_range=time_range,
                    temp_dir=temp_dir,
                )
            ) as events,
        ):
            temporary = Path(target.name)
            target.write('{"traceEvents": [')
            for index, event in enumerate(events):
                if index:
                    target.write(", ")
                target.write(json.dumps(event, allow_nan=False))
                count += event["ph"] == "X"
            target.write('], "displayTimeUnit": "ms"}')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count
