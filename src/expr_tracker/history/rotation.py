"""Stable JSONL segments, addressed as one append-only byte stream."""

from pathlib import Path

DEFAULT_MAX_FILE_BYTES = 100 * 1024 * 1024


def validate_file_limit(max_file_bytes: int | None) -> None:
    if max_file_bytes is not None and (
        type(max_file_bytes) is not int or max_file_bytes <= 0
    ):
        raise ValueError("max_file_bytes must be a positive integer or None")


def segment_path(path: Path, number: int) -> Path:
    return path.with_name(path.name + ".parts") / f"{number:08d}.jsonl"


def segments(path: Path) -> list[Path]:
    """The original file stays in place; later segments never rename older data."""
    parts = path.with_name(path.name + ".parts")
    numbered = []
    if parts.exists():
        numbered.extend(
            part
            for part in parts.glob("*.jsonl")
            if part.stem.isascii() and part.stem.isdecimal()
        )
    return [path, *sorted(numbered, key=lambda part: int(part.stem))]
