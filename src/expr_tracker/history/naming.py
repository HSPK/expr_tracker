"""Run file naming: distributed rank shards and independent producer streams.

Kept separate from the store and the reader because both need it, and the store
already depends on the reader.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_ROOT = "./tracker/jsonl"
STREAM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
RANK_PATTERN = re.compile(r"rank\d+")


def current_rank() -> int:
    """Distributed rank from the usual environment variables, 0 when unset."""
    rank = os.getenv("RANK") or os.getenv("LOCAL_RANK") or "0"
    try:
        return int(rank)
    except ValueError:
        return 0


def validate_stream(stream: str) -> str:
    """Check a stream name is safe as a filename component.

    The name becomes part of ``metrics.<stream>[.rankN].jsonl``, so it must not
    contain a separator, and must not look like a rank shard.
    """
    if not STREAM_PATTERN.match(stream):
        raise ValueError(
            f"Invalid stream name {stream!r}; use letters, digits, '_' or '-', "
            "starting with a letter or digit."
        )
    if RANK_PATTERN.fullmatch(stream):
        raise ValueError(
            f"Stream name {stream!r} collides with the rank shard naming; "
            "pick another name."
        )
    return stream


def metrics_filename(stream: str | None, rank_aware: bool = True) -> str:
    """``metrics[.stream][.rankN].jsonl``.

    Streams separate independent producers; non-zero ranks get their own shard so
    concurrent appends cannot interleave. The two compose.
    """
    rank = current_rank() if rank_aware else 0
    parts = ["metrics"]
    if stream:
        parts.append(stream)
    if rank > 0:
        parts.append(f"rank{rank}")
    return ".".join(parts) + ".jsonl"


def spans_filename(stream: str | None, rank_aware: bool = True) -> str:
    """``spans[.stream][.rankN].jsonl``, alongside the metrics file."""
    return metrics_filename(stream, rank_aware).replace("metrics", "spans", 1)


def sidecar_filename(base: str, stream: str | None, suffix: str) -> str:
    """``<base>[.stream].<suffix>``, so producers do not overwrite each other."""
    return f"{base}.{stream}.{suffix}" if stream else f"{base}.{suffix}"


def parse_stream(filename: str) -> str | None:
    """The stream a metrics filename belongs to; ``None`` is the default producer."""
    parts = filename.split(".")[1:-1]  # drop "metrics" and "jsonl"
    if parts and RANK_PATTERN.fullmatch(parts[-1]):
        parts = parts[:-1]
    return parts[0] if parts else None


def resolve_log_dir(
    project: str, name: str, dir: str | None = None, run_dir: str | None = None
) -> Path:
    """Where this run's files go.

    ``dir`` is a root holding many projects, so a run lands in
    ``<dir>/<project>/<name>``. That layout is what lets artifacts be shared and
    deduplicated across a project's runs, and what lets a resume find its files
    from the project and name alone. ``run_dir`` opts out of it and names the
    directory outright, for when something else already chose the path.
    """
    if run_dir is not None:
        if dir is not None:
            raise ValueError(
                "Pass either dir (a root, giving <dir>/<project>/<name>) or "
                "run_dir (this run's directory), not both"
            )
        return Path(run_dir)
    return Path(dir or DEFAULT_ROOT) / project / name


def resolve_artifact_root(
    project: str, dir: str | None = None, run_dir: str | None = None
) -> Path:
    """Where artifacts live: beside the project's runs, or inside a lone run.

    A project-level store is what makes deduplication across runs possible, so
    ``run_dir`` gives up that sharing in exchange for a self-contained directory.
    """
    if run_dir is not None:
        return Path(run_dir) / "artifacts"
    return Path(dir or DEFAULT_ROOT) / project / "artifacts"
