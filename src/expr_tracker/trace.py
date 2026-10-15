"""Native span queries and bounded-memory, atomic Chrome Trace exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._native import (
    NativeLanePool,
    trace_assign_lanes,
    trace_build,
    trace_read_spans,
    trace_write,
)
from .history.naming import parse_stream, spans_filename

MICROSECONDS = 1_000_000.0
_LanePool = NativeLanePool


def span_files(run: str | Path, stream: str | list[str] | None = "*") -> list[Path]:
    """Find a run's span streams; native reads include each file's .parts."""
    directory = Path(run)
    if directory.is_file():
        return [directory]
    if stream == "*":
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


def read_spans(
    path: Path,
    *,
    step_range: tuple[int | None, int | None] | None = None,
    time_range: tuple[float | None, float | None] | None = None,
) -> list[dict]:
    """Read valid spans with half-open step and overlapping epoch-time filters.

    Input is scanned incrementally, including original and .parts segments.
    Memory is proportional to selected records and the largest input record.
    Numeric timestamp strings are accepted; invalid records are skipped.
    A time filter excludes zero-duration spans and never clips intervals.
    """
    return trace_read_spans(Path(path), step_range, time_range)


def assign_lanes(spans: list[dict]) -> dict[int, int]:
    """Pack each thread's spans into valid stacks in O(n log n).

    Known depths nest only below their immediately preceding depth. Siblings,
    crossing spans, and unrelated threads occupy separate lanes.
    """
    return trace_assign_lanes(spans)


def _files(run, stream):
    return [
        (path, parse_stream(path.name) or "default") for path in span_files(run, stream)
    ]


def build_trace(
    run: str | Path,
    stream: str | list[str] | None = "*",
    *,
    step_range: tuple[int | None, int | None] | None = None,
    time_range: tuple[float | None, float | None] | None = None,
    temp_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a Chrome Trace with a shared earliest-selected-start origin.

    Native SQLite spools sorting/layout to disk. The returned dict necessarily
    retains selected events; use write_trace for bounded-memory output.
    temp_dir selects an existing spool directory (default: working directory).
    """
    return trace_build(
        _files(run, stream),
        step_range,
        time_range,
        Path(temp_dir) if temp_dir is not None else None,
    )


def write_trace(
    run: str | Path,
    output: str | Path,
    stream: str | list[str] | None = "*",
    *,
    step_range: tuple[int | None, int | None] | None = None,
    time_range: tuple[float | None, float | None] | None = None,
    temp_dir: str | Path | None = None,
) -> int:
    """Atomically export a Chrome Trace and return its selected span count.

    Rust scans records, filters, spools SQLite layout and emits one event at a
    time. Memory scales with the largest record, threads, peak overlap and active
    nesting, not historical record count. Failures preserve the old destination
    and remove spool/output scratch files. Spools default to beside the output.
    Atomic replacement does not imply fsync durability across a machine crash.
    """
    return trace_write(
        _files(run, stream),
        Path(output),
        step_range,
        time_range,
        Path(temp_dir) if temp_dir is not None else None,
    )
