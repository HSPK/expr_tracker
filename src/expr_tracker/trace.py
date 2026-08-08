"""Export recorded spans as a Chrome Trace, readable by Perfetto and chrome://tracing.

Writing the format rather than a viewer means the timeline can sit next to a
``torch.profiler`` trace in the same window, which is where the interesting
question usually is: what were the GPUs doing while the data loader stalled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .history.naming import parse_stream, spans_filename

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


def read_spans(path: Path) -> list[dict]:
    """Decode one span file, skipping anything malformed."""
    if not path.is_file():
        return []
    spans = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and "start" in record and "dur_ms" in record:
            spans.append(record)
    return spans


def _pack(spans: list[dict], indices: list[int], offset: int) -> dict[int, int]:
    """Lay one track's spans onto lanes, each lane a valid stack."""
    order = sorted(
        indices, key=lambda i: (spans[i]["start"], -float(spans[i]["dur_ms"]))
    )
    lanes: list[list[float]] = []  # per lane, the end times currently open
    placement: dict[int, int] = {}
    for index in order:
        start = float(spans[index]["start"])
        end = start + float(spans[index]["dur_ms"]) / 1000.0
        for lane, stack in enumerate(lanes):
            while stack and stack[-1] <= start:
                stack.pop()
            if not stack or end <= stack[-1]:  # free, or nests inside the top
                stack.append(end)
                placement[index] = offset + lane
                break
        else:
            lanes.append([end])
            placement[index] = offset + len(lanes) - 1
    return placement


def assign_lanes(spans: list[dict]) -> dict[int, int]:
    """Place spans on lanes so each lane is a valid stack, keyed by list index.

    Threads come first: work on different threads is unrelated, and letting one
    contain the other would draw a nesting that never happened. Within a thread,
    anything still overlapping -- concurrent asyncio tasks -- gets its own lane,
    because a Chrome Trace track must be properly nested to render at all.
    """
    by_track: dict[Any, list[int]] = {}
    for index, record in enumerate(spans):
        by_track.setdefault(record.get("track", 0), []).append(index)
    placement: dict[int, int] = {}
    offset = 0
    for track in sorted(by_track, key=str):
        lanes = _pack(spans, by_track[track], offset)
        placement.update(lanes)
        offset = max(lanes.values(), default=offset - 1) + 1
    return placement


def build_trace(
    run: str | Path,
    stream: str | list[str] | None = "*",
    *,
    step_range: tuple[int | None, int | None] | None = None,
) -> dict[str, Any]:
    """Chrome Trace Event Format for a run's spans, one process per stream."""
    events: list[dict] = []
    origin: float | None = None
    for pid, path in enumerate(span_files(run, stream)):
        spans = read_spans(path)
        if step_range is not None:
            low, high = step_range
            spans = [
                s
                for s in spans
                if (low is None or s.get("_step", 0) >= low)
                and (high is None or s.get("_step", 0) < high)
            ]
        name = parse_stream(path.name) or "default"
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": name},
            }
        )
        if not spans:
            continue
        lanes = assign_lanes(spans)
        tracks = sorted({s.get("track", 0) for s in spans}, key=str)
        # Label each lane with the thread it came from
        lane_track: dict[int, Any] = {}
        for index, record in enumerate(spans):
            lane_track.setdefault(lanes[index], record.get("track", 0))
        for tid, track in sorted(lane_track.items()):
            label = f"thread {track}" if len(tracks) > 1 else name
            events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": label},
                }
            )
        for index, record in enumerate(spans):
            start = float(record["start"])
            origin = start if origin is None else min(origin, start)
            args = dict(record.get("args") or {})
            args["step"] = record.get("_step")
            if record.get("error"):
                args["error"] = record["error"]
            if len(tracks) > 1:
                args["track"] = record.get("track", 0)
            events.append(
                {
                    "name": record.get("name", "span"),
                    "cat": name,
                    "ph": "X",
                    "pid": pid,
                    "tid": lanes[index],
                    "ts": start * MICROSECONDS,
                    "dur": float(record["dur_ms"]) * 1000.0,
                    "args": args,
                }
            )
    # Relative timestamps keep the numbers readable and the streams aligned
    if origin is not None:
        for event in events:
            if "ts" in event:
                event["ts"] -= origin * MICROSECONDS
    return {"traceEvents": events, "displayTimeUnit": "ms"}


def write_trace(
    run: str | Path,
    output: str | Path,
    stream: str | list[str] | None = "*",
    *,
    step_range: tuple[int | None, int | None] | None = None,
) -> int:
    """Write the trace and return how many spans it holds."""
    trace = build_trace(run, stream, step_range=step_range)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace), encoding="utf-8")
    return sum(1 for event in trace["traceEvents"] if event["ph"] == "X")
