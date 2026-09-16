"""JSONL reader: reverse tail, step/line lookup and ranged streaming, never
loading the whole file into memory.

Also hosts the offline entry points, which read a finished run without a live store.
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Iterable, Iterator, Sequence
from itertools import chain, islice
from pathlib import Path

from loguru import logger

from .frame import project, to_output
from .naming import RANK_PATTERN, metrics_filename

CHUNK_SIZE = 256 * 1024


class JsonlReader:
    """Read-only access to one ``metrics.jsonl``.

    ``index`` is a sparse list of ``(step, line, offset)`` anchors, loaded from the
    meta sidecar or rebuilt by scanning. With ``merge=True`` results are merged by
    ``_step``, which is required for files containing patch lines or out-of-order steps.

    Each read pins a byte prefix before inspecting metadata. Appends become visible
    to subsequent reads, never halfway through the current read's merge decision.
    Explicit ``index`` and ``merge`` overrides remain caller-provided hints.

    Naming convention: a ``_rows`` suffix means **physical rows, never merged**
    (``tail_rows``, ``parse_rows``); the plain name means **steps**, merged when the
    file needs it (``tail``, ``parse``). One step can span several rows, so mixing
    the two up silently returns half-merged records.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        index: Sequence[tuple[int, int, int]] | None = None,
        merge: bool | None = None,
    ):
        self.path = Path(path)
        self._index = list(index) if index is not None else None
        self._merge = merge
        self._meta_loaded = False
        self._snapshot_limit: int | None = None
        self._snapshot_file_size: int | None = None
        self._cached_snapshot: tuple[int, JsonlReader] | None = None

    def _snapshot(self, end: int | None = None) -> JsonlReader:
        """A query-local reader whose metadata and data share one byte boundary."""
        if self._snapshot_limit is not None and (
            end is None or end >= self._snapshot_limit
        ):
            return self
        size = self.size
        limit = size if end is None else max(0, min(end, size))
        source = self
        cached = self._cached_snapshot
        if cached is not None and cached[0] >= limit:
            source = cached[1]
        reader = JsonlReader(self.path)
        reader._snapshot_limit = limit
        reader._snapshot_file_size = (
            size if self._snapshot_file_size is None else self._snapshot_file_size
        )
        reader._index = source._index
        if reader._index and source._snapshot_limit != limit:
            reader._index = [entry for entry in reader._index if entry[2] < limit]
        reader._merge = source._merge
        reader._meta_loaded = source._meta_loaded
        self._cached_snapshot = (limit, reader)
        return reader

    # ------------------------------------------------------------------ metadata

    @property
    def size(self) -> int:
        if self._snapshot_limit is not None:
            return self._snapshot_limit
        return self.path.stat().st_size if self.path.exists() else 0

    @property
    def merge(self) -> bool:
        if self._snapshot_limit is None:
            return self._snapshot().merge
        if self._merge is None:
            self._load_meta()
        if self._merge is None:
            # Building the index also detects duplicate or out-of-order steps
            _ = self.index
        if self._merge is None:
            # Supplied anchors describe locations, not whether steps are unique.
            self._build_index()
        return bool(self._merge)

    @property
    def index(self) -> list[tuple[int, int, int]]:
        if self._snapshot_limit is None:
            return self._snapshot().index
        if self._index is None:
            self._load_meta()
        if self._index is None:
            self._index = self._build_index()
        return self._index

    def _load_meta(self):
        if self._meta_loaded:
            return
        self._meta_loaded = True
        meta_path = self.path.with_name(self.path.stem + ".meta.json")
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except FileNotFoundError:
            return
        except OSError as e:
            logger.warning(
                f"Could not read metadata from {meta_path}: {e}; scanning history file."
            )
            return
        except (ValueError, RecursionError) as e:
            logger.warning(f"Ignoring corrupted metadata in {meta_path}: {e}")
            return
        if not isinstance(meta, dict):
            logger.warning(f"Ignoring malformed metadata in {meta_path}")
            return
        size = self.size
        if meta.get("size") != size:
            return  # stale meta cannot be trusted; fall back to scanning
        if not _valid_meta(meta, size):
            logger.warning(f"Ignoring malformed metadata in {meta_path}")
            return
        if self._index is None:
            self._index = [tuple(e) for e in meta.get("index", [])]
        if self._merge is None:
            self._merge = bool(meta.get("has_duplicate_steps")) or not bool(
                meta.get("sorted", True)
            )

    def _build_index(self, every: int = 1000) -> list[tuple[int, int, int]]:
        index: list[tuple[int, int, int]] = []
        if not self.path.exists():
            return index
        line_no = 0
        next_anchor = 0
        previous: int | None = None
        duplicates = unsorted = False
        for offset, raw in self._scan():
            if not raw.strip():
                continue
            step = _peek_step(raw)
            if step is not None:
                if previous is not None:
                    duplicates = duplicates or step == previous
                    unsorted = unsorted or step < previous
                previous = step
                if line_no >= next_anchor:
                    index.append((step, line_no, offset))
                    next_anchor = line_no + every
            line_no += 1
        if self._merge is None:
            self._merge = duplicates or unsorted
        return index

    def _scan(self, start: int = 0, end: int | None = None):
        """Yield ``(byte offset, raw line)`` for every line, blanks included."""
        if not self.path.exists():
            return
        limit = self.size if end is None else end
        with open(self.path, "rb") as f:
            f.seek(start)
            offset = start
            while offset < limit:
                raw = f.readline(limit - offset + 1)
                if not raw:
                    break
                if offset + len(raw) > limit:
                    if limit != self._snapshot_file_size:
                        break
                    # At the captured EOF, preserve an unterminated row exactly as
                    # it was, even if an append has since completed or extended it.
                    raw = raw[: limit - offset]
                yield offset, raw
                offset += len(raw)

    # ------------------------------------------------------------------ reading

    def iter_raw(self, start: int = 0, end: int | None = None) -> Iterator[bytes]:
        reader = self._snapshot(end)
        for _, raw in reader._scan(start):
            if raw.strip():
                yield raw

    def tail_raw(self, count: int, end: int | None = None) -> list[bytes]:
        """Read the last ``count`` lines ending before byte offset ``end``."""
        if count <= 0:
            return []
        lines = list(islice(self._snapshot(end)._reverse_raw(), count))
        lines.reverse()
        return lines

    def _reverse_raw(self, end: int | None = None) -> Iterator[bytes]:
        """Yield nonblank lines backwards, joining long lines only once."""
        if not self.path.exists():
            return
        size = self.size
        pos = size if end is None else min(end, size)
        if pos <= 0:
            return
        fragments: list[bytes] = []
        with open(self.path, "rb") as f:
            f.seek(pos - 1)
            file_size = (
                size if self._snapshot_file_size is None else self._snapshot_file_size
            )
            skip_partial = pos < file_size and f.read(1) != b"\n"
            while pos > 0:
                read = min(CHUNK_SIZE, pos)
                pos -= read
                f.seek(pos)
                parts = f.read(read).split(b"\n")
                fragments.append(parts[-1])
                if len(parts) == 1:
                    continue
                raw = b"".join(reversed(fragments))
                if skip_partial:
                    skip_partial = False
                elif raw.strip():
                    yield raw
                for raw in reversed(parts[1:-1]):
                    if raw.strip():
                        yield raw
                fragments = [parts[0]]
        raw = b"".join(reversed(fragments))
        if not skip_partial and raw.strip():
            yield raw

    def offset_of_step(self, step: int) -> int:
        """Byte offset of the first line with ``_step >= step``, else the file size.

        Requires **strictly ascending** steps: the anchor search lands on an exact
        match, which would skip a step's earlier row if a step were anchored twice.
        The writer guarantees that whenever ``merge`` is false, so callers must use
        :meth:`offset_of_line` in ``merge`` mode.
        """
        reader = self._snapshot()
        index = reader.index
        start = 0
        if index:
            # The last anchor at or before the step; bisect_left would land one
            # anchor too early whenever the step is anchored exactly
            pos = bisect.bisect_right(index, step, key=_anchor_step) - 1
            if pos >= 0:
                start = index[pos][2]
        for offset, raw in reader._scan(start):
            if not raw.strip():
                continue
            value = _peek_step(raw)
            if value is not None and value >= step:
                return offset
        return reader.size

    def offset_of_line(self, line_no: int) -> int:
        """Byte offset of record ``line_no`` (0-based, blank lines excluded).

        Physical addressing is independent of ``_step``, so it stays exact for files
        with patch lines or out-of-order steps.
        """
        if line_no <= 0:
            return 0
        reader = self._snapshot()
        index = reader.index
        start_offset, count = 0, 0
        if index:
            pos = bisect.bisect_right(index, line_no, key=_anchor_line) - 1
            if pos >= 0:
                _, count, start_offset = index[pos]
        if count >= line_no:
            return start_offset
        for offset, raw in reader._scan(start_offset):
            if not raw.strip():
                continue
            if count >= line_no:
                return offset
            count += 1
        return reader.size

    # ------------------------------------------------------------------ records

    def parse(self, lines: Sequence[bytes]) -> list[dict]:
        """Decode lines, merging by step when the file needs it."""
        records = self.parse_rows(lines)
        return merge_steps(records) if self.merge else records

    def parse_rows(self, lines: Iterable[bytes]) -> list[dict]:
        """Decode lines one-for-one, never merging.

        Corrupt lines and rows without an integer ``_step`` are skipped, so every
        record a reader returns can be ordered and merged by step.
        """
        return list(self._iter_rows(lines))

    def _iter_rows(self, lines: Iterable[bytes]) -> Iterator[dict]:
        for raw in lines:
            try:
                value = json.loads(raw)
            except (ValueError, RecursionError) as e:
                logger.warning(f"Skipping corrupted line in {self.path}: {e}")
                continue
            if isinstance(value, dict) and isinstance(value.get("_step"), int):
                yield value

    def tail_rows(self, count: int, end: int | None = None) -> list[dict]:
        """The last ``count`` physical rows, unmerged."""
        return self.parse_rows(self.tail_raw(count, end))

    def tail(self, count: int, end: int | None = None) -> list[dict]:
        """The most recently written ``count`` distinct steps, merged and sorted.

        A reverse pass selects the keys. Files requiring a merge then scan forwards
        for all patches of those keys, retaining only the selected steps.
        """
        if count <= 0:
            return []
        reader = self._snapshot(end)
        if not reader.merge:
            rows = list(islice(reader._iter_rows(reader._reverse_raw()), count))
            rows.reverse()
            return rows
        return reader.read_selected_steps(reader.select_recent_steps(count))

    def select_recent_steps(
        self, count: int, end: int | None = None, *, newer: Iterable[int] = ()
    ) -> set[int]:
        """Select recent keys, first considering ``newer`` keys in reverse order."""
        reader = self._snapshot(end)
        disk_steps = (
            record["_step"] for record in reader._iter_rows(reader._reverse_raw())
        )
        return recent_steps(chain(newer, disk_steps), count)

    def read_selected_steps(
        self, selected: set[int], end: int | None = None
    ) -> list[dict]:
        """Merge every physical row of the selected keys below ``end``."""
        if not selected:
            return []
        reader = self._snapshot(end)
        return merge_steps(reader._iter_rows(reader.iter_raw()), selected=selected)

    def read_steps(
        self, start: int | None, end: int | None, stop_offset: int | None = None
    ) -> list[dict]:
        """Read records with ``start <= _step < end``; ``None`` means unbounded.

        ``stop_offset`` bounds the scan, which the live store uses to stay strictly
        below the cache boundary.
        """
        reader = self._snapshot(stop_offset)
        merge = reader.merge
        # Unordered files rule out seeking by step or stopping early
        offset = 0 if (start is None or merge) else reader.offset_of_step(start)
        records = []
        for raw in reader.iter_raw(offset):
            value = _peek_step(raw)
            if value is not None:
                if end is not None and value >= end and not merge:
                    break
                if (start is not None and value < start) or (
                    end is not None and value >= end
                ):
                    continue
            records.append(raw)
        return reader.parse(records)

    def read_all(self) -> list[dict]:
        reader = self._snapshot()
        return reader.parse(list(reader.iter_raw()))

    def last_record(self) -> dict | None:
        reader = self._snapshot()
        records = reader.parse(reader.tail_raw(1))
        return records[-1] if records else None

    def count_lines(self) -> int:
        return sum(1 for _ in self.iter_raw())


def _anchor_step(anchor: tuple[int, int, int]) -> int:
    return anchor[0]


def _anchor_line(anchor: tuple[int, int, int]) -> int:
    return anchor[1]


def _valid_meta(meta: dict, size: int) -> bool:
    if (
        meta.get("schema") != 2
        or type(meta.get("size")) is not int
        or type(meta.get("sorted")) is not bool
        or type(meta.get("has_duplicate_steps")) is not bool
        or not isinstance(meta.get("index"), list)
    ):
        return False
    previous_line = previous_offset = -1
    previous_step: int | None = None
    for entry in meta["index"]:
        if (
            not isinstance(entry, list)
            or len(entry) != 3
            or any(type(value) is not int for value in entry)
        ):
            return False
        step, line, offset = entry
        if line <= previous_line or not previous_offset < offset < size:
            return False
        if (
            meta["sorted"]
            and previous_step is not None
            and (
                step < previous_step
                or (step == previous_step and not meta["has_duplicate_steps"])
            )
        ):
            return False
        previous_step, previous_line, previous_offset = step, line, offset
    return True


def recent_steps(steps: Iterable[int], count: int) -> set[int]:
    """Take distinct keys from a newest-first stream without scanning its suffix."""
    selected: set[int] = set()
    if count <= 0:
        return selected
    for step in steps:
        selected.add(step)
        if len(selected) == count:
            break
    return selected


def merge_steps(
    records: Iterable[dict], *, selected: set[int] | None = None
) -> list[dict]:
    """Merge rows sharing a ``_step`` (last write wins) and sort by step."""
    merged: dict[int, dict] = {}
    for record in records:
        step = record.get("_step")
        if not isinstance(step, int):
            continue
        if selected is not None and step not in selected:
            continue
        if step in merged:
            merged[step].update(record)
        else:
            merged[step] = dict(record)
    return [merged[step] for step in sorted(merged)]


def _peek_step(raw: bytes) -> int | None:
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    step = value.get("_step") if isinstance(value, dict) else None
    return step if isinstance(step, int) else None


# ---------------------------------------------------------------------- offline
def read_history(
    run: str | Path,
    n: int | None = 50,
    *,
    stream: str | None = None,
    output_type: str = "dict",
    metrics: Sequence[str] | None = None,
    step_range: tuple[int | None, int | None] | None = None,
    include_meta: bool = True,
    fill_missing: bool = False,
    dropna: bool = False,
):
    """Read any run's history without ``init()``; ``run`` is a directory or file."""
    reader = JsonlReader(resolve_run_path(run, stream))
    limit = None if n is None or n < 0 else max(0, n)
    if limit == 0:
        records: list[dict] = []
    elif step_range is not None:
        records = reader.read_steps(*step_range)
        if limit is not None:
            records = records[-limit:]
    elif limit is None:
        records = reader.read_all()
    else:
        records = reader.tail(limit)
    rows = project(
        records,
        metrics=metrics,
        include_meta=include_meta,
        fill_missing=fill_missing,
        dropna=dropna,
    )
    return to_output(rows, output_type)


def resolve_run_path(run: str | Path, stream: str | None = None) -> Path:
    """The metrics file of a run directory, or the file itself.

    Names are matched exactly rather than by sort order: ``metrics.data.jsonl``
    sorts before ``metrics.jsonl``, so picking the first match would silently
    return a stream instead of the default producer.
    """
    path = Path(run)
    if not path.is_dir():
        if not path.exists():
            raise FileNotFoundError(f"Run path does not exist: {path}")
        return path
    wanted = metrics_filename(stream, rank_aware=False)
    exact = path / wanted
    if exact.is_file():
        return exact
    # Fall back to this stream's rank shards, which a worker rank writes instead
    prefix = wanted[: -len(".jsonl")]
    shards = sorted(path.glob(f"{prefix}.rank*.jsonl"))
    if shards:
        return shards[0]
    raise FileNotFoundError(f"No {wanted} found under {path}; {_available(path)}")


def _available(path: Path) -> str:
    found = sorted(p.name for p in path.glob("metrics*.jsonl"))
    return f"available: {', '.join(found)}" if found else "the directory has none"


def list_streams(run: str | Path) -> list[str | None]:
    """Every stream in a run directory; ``None`` is the default producer."""
    streams: set[str | None] = set()
    for path in Path(run).glob("metrics*.jsonl"):
        parts = path.name.split(".")[1:-1]  # drop "metrics" and "jsonl"
        if parts and RANK_PATTERN.fullmatch(parts[-1]):
            parts = parts[:-1]
        streams.add(parts[0] if parts else None)
    return sorted(streams, key=lambda s: (s is not None, s or ""))
