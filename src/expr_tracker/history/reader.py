"""JSONL reader: reverse tail, step/line lookup and ranged streaming, never
loading the whole file into memory.

Also hosts the offline entry points, which read a finished run without a live store.
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Iterable, Iterator, Sequence
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

    # ------------------------------------------------------------------ metadata

    @property
    def size(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0

    @property
    def merge(self) -> bool:
        if self._merge is None:
            self._load_meta()
        if self._merge is None:
            # Building the index also detects duplicate or out-of-order steps
            _ = self.index
        return bool(self._merge)

    @property
    def index(self) -> list[tuple[int, int, int]]:
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
        except Exception:
            return
        if meta.get("size") != self.size:
            return  # stale meta cannot be trusted; fall back to scanning
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
                if line_no % every == 0:
                    index.append((step, line_no, offset))
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
            for raw in f:
                if offset >= limit:
                    break
                yield offset, raw
                offset += len(raw)

    # ------------------------------------------------------------------ reading

    def iter_raw(self, start: int = 0, end: int | None = None) -> Iterator[bytes]:
        for _, raw in self._scan(start, end):
            if raw.strip():
                yield raw

    def tail_raw(self, count: int, end: int | None = None) -> list[bytes]:
        """Read the last ``count`` lines ending before byte offset ``end``."""
        if count <= 0 or not self.path.exists():
            return []
        pos = self.size if end is None else min(end, self.size)
        chunks: list[bytes] = []
        newlines = 0
        with open(self.path, "rb") as f:
            while pos > 0 and newlines <= count:
                read = min(CHUNK_SIZE, pos)
                pos -= read
                f.seek(pos)
                chunk = f.read(read)
                chunks.append(chunk)
                newlines += chunk.count(b"\n")
        data = b"".join(reversed(chunks))
        lines = [line for line in data.split(b"\n") if line.strip()]
        if pos > 0 and lines:
            lines = lines[1:]  # the first line may be cut by the chunk boundary
        return lines[-count:]

    def offset_of_step(self, step: int) -> int:
        """Byte offset of the first line with ``_step >= step``, else the file size.

        Requires **strictly ascending** steps: the anchor search lands on an exact
        match, which would skip a step's earlier row if a step were anchored twice.
        The writer guarantees that whenever ``merge`` is false, so callers must use
        :meth:`offset_of_line` in ``merge`` mode.
        """
        index = self.index
        start = 0
        if index:
            # The last anchor at or before the step; bisect_left would land one
            # anchor too early whenever the step is anchored exactly
            pos = bisect.bisect_right(index, step, key=_anchor_step) - 1
            if pos >= 0:
                start = index[pos][2]
        for offset, raw in self._scan(start):
            if not raw.strip():
                continue
            value = _peek_step(raw)
            if value is not None and value >= step:
                return offset
        return self.size

    def offset_of_line(self, line_no: int) -> int:
        """Byte offset of record ``line_no`` (0-based, blank lines excluded).

        Physical addressing is independent of ``_step``, so it stays exact for files
        with patch lines or out-of-order steps.
        """
        if line_no <= 0:
            return 0
        index = self.index
        start_offset, count = 0, 0
        if index:
            pos = bisect.bisect_right(index, line_no, key=_anchor_line) - 1
            if pos >= 0:
                _, count, start_offset = index[pos]
        if count >= line_no:
            return start_offset
        for offset, raw in self._scan(start_offset):
            if not raw.strip():
                continue
            if count >= line_no:
                return offset
            count += 1
        return self.size

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
        records = []
        for raw in lines:
            try:
                value = json.loads(raw)
            except Exception as e:
                logger.warning(f"Skipping corrupted line in {self.path}: {e}")
                continue
            if isinstance(value, dict) and isinstance(value.get("_step"), int):
                records.append(value)
        return records

    def tail_rows(self, count: int, end: int | None = None) -> list[dict]:
        """The last ``count`` physical rows, unmerged."""
        return self.parse_rows(self.tail_raw(count, end))

    def tail(self, count: int, end: int | None = None) -> list[dict]:
        """The newest ``count`` steps, merged.

        Rows can share a step, so the read widens until one whole extra step is in
        hand; that spare step is then dropped so the oldest result is never partial.
        """
        if count <= 0:
            return []
        if not self.merge:
            return self.tail_rows(count, end)
        want = count
        for _ in range(12):
            rows = self.tail_rows(want, end)
            merged = merge_steps(rows)
            if len(rows) < want or len(merged) > count:
                return merged[-count:]
            want *= 2
        return merge_steps(self.tail_rows(want, end))[-count:]

    def read_steps(
        self, start: int | None, end: int | None, stop_offset: int | None = None
    ) -> list[dict]:
        """Read records with ``start <= _step < end``; ``None`` means unbounded.

        ``stop_offset`` bounds the scan, which the live store uses to stay strictly
        below the cache boundary.
        """
        # Unordered files rule out seeking by step or stopping early
        offset = 0 if (start is None or self.merge) else self.offset_of_step(start)
        if stop_offset is not None:
            offset = min(offset, stop_offset)
        records = []
        for raw in self.iter_raw(offset, stop_offset):
            value = _peek_step(raw)
            if value is not None:
                if end is not None and value >= end and not self.merge:
                    break
                if (start is not None and value < start) or (
                    end is not None and value >= end
                ):
                    continue
            records.append(raw)
        return self.parse(records)

    def read_all(self) -> list[dict]:
        return self.parse(list(self.iter_raw()))

    def last_record(self) -> dict | None:
        records = self.parse(self.tail_raw(1))
        return records[-1] if records else None

    def count_lines(self) -> int:
        return sum(1 for _ in self.iter_raw())


def _anchor_step(anchor: tuple[int, int, int]) -> int:
    return anchor[0]


def _anchor_line(anchor: tuple[int, int, int]) -> int:
    return anchor[1]


def merge_steps(records: Sequence[dict]) -> list[dict]:
    """Merge rows sharing a ``_step`` (last write wins) and sort by step."""
    merged: dict[int, dict] = {}
    for record in records:
        step = record.get("_step")
        if not isinstance(step, int):
            continue
        if step in merged:
            merged[step].update(record)
        else:
            merged[step] = dict(record)
    return [merged[step] for step in sorted(merged)]


def _peek_step(raw: bytes) -> int | None:
    try:
        value = json.loads(raw)
    except Exception:
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
