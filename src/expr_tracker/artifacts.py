"""Local artifact store: versioned file bundles logged alongside a run.

Layout under ``<dir>/<project>/artifacts``::

    index.jsonl          one line per logged version (project-wide)
    <name>/v<N>/...      materialised files for that version

Files are copied by default. Explicit link mode uses hard links when possible,
falling back to copies. Rust atomically publishes complete versions and indexes
under an interprocess lock; references are indexed without materialization.
Atomic publication does not imply fsync durability across a machine crash.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from ._native import (
    NativeArtifactStore,
    artifact_copy,
    artifact_directory_entries,
    artifact_file_digest,
    artifact_file_entry,
    artifact_validate_path,
)

DEFAULT_TYPE = "dataset"
LATEST = "latest"
CHUNK = 1 << 20


@dataclass
class ArtifactEntry:
    """One file (or external reference) inside an artifact."""

    path: str
    size: int = 0
    digest: str = ""
    ref: str | None = None
    source: str | None = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "size": self.size,
            "digest": self.digest,
            "ref": self.ref,
        }


class Artifact:
    """A named, versioned bundle of files, mirroring the wandb Artifact API."""

    def __init__(
        self,
        name: str,
        type: str = DEFAULT_TYPE,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ):
        if not name:
            raise ValueError("Artifact name must not be empty")
        artifact_validate_path(Path(name))
        self.name = name
        self.type = type or DEFAULT_TYPE
        self.description = description
        self.metadata = dict(metadata or {})
        self.aliases = list(aliases or [])
        self.entries: list[ArtifactEntry] = []
        self.version: int | None = None
        self.digest: str | None = None
        self.dir: Path | None = None
        self.run: str | None = None
        self.step: int | None = None
        self.created_at: float | None = None

    # ------------------------------------------------------------------ building

    def add_file(
        self, local_path: str | os.PathLike, name: str | None = None
    ) -> Artifact:
        path = Path(local_path)
        self._add(path, name or path.name)
        return self

    def add_dir(
        self, local_dir: str | os.PathLike, name: str | None = None
    ) -> Artifact:
        root = Path(local_dir)
        for logical, size, digest, source in artifact_directory_entries(root, name):
            self.entries.append(
                ArtifactEntry(path=logical, size=size, digest=digest, source=source)
            )
        return self

    def add_reference(self, uri: str, name: str | None = None) -> Artifact:
        """Record an external URI without copying anything."""
        logical = name or uri.rsplit("/", 1)[-1] or uri
        artifact_validate_path(Path(logical))
        self.entries.append(ArtifactEntry(path=logical, ref=uri))
        return self

    def _add(self, path: Path, logical: str):
        artifact_validate_path(Path(logical))
        size, digest = artifact_file_entry(path)
        self.entries.append(
            ArtifactEntry(
                path=logical,
                size=size,
                digest=digest,
                source=str(path),
            )
        )

    # ------------------------------------------------------------------ reading

    @property
    def qualified_name(self) -> str:
        version = "draft" if self.version is None else f"v{self.version}"
        return f"{self.name}:{version}"

    def download(self, root: str | os.PathLike | None = None) -> Path:
        """Return the directory holding this version's files, copying if asked."""
        if self.dir is None:
            raise RuntimeError(f"Artifact {self.name!r} has not been logged yet")
        if root is None:
            return self.dir
        target = Path(root)
        artifact_copy(
            self.dir,
            target,
            [Path(entry.path) for entry in self.entries if not entry.ref],
        )
        return target

    def get_path(self, name: str) -> Path:
        if self.dir is None:
            raise RuntimeError(f"Artifact {self.name!r} has not been logged yet")
        artifact_validate_path(Path(name))
        return self.dir / name

    def files(self) -> list[str]:
        return [entry.path for entry in self.entries]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "type": self.type,
            "description": self.description,
            "metadata": self.metadata,
            "aliases": self.aliases,
            "digest": self.digest,
            "created_at": self.created_at,
            "run": self.run,
            "step": self.step,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict, root: Path) -> Artifact:
        artifact = cls(
            name=data["name"],
            type=data.get("type", DEFAULT_TYPE),
            description=data.get("description"),
            metadata=data.get("metadata"),
            aliases=data.get("aliases"),
        )
        artifact.version = data.get("version")
        artifact.digest = data.get("digest")
        artifact.created_at = data.get("created_at")
        artifact.run = data.get("run")
        artifact.step = data.get("step")
        artifact.entries = [
            ArtifactEntry(
                path=e["path"],
                size=e.get("size", 0),
                digest=e.get("digest", ""),
                ref=e.get("ref"),
            )
            for e in data.get("entries", [])
        ]
        if artifact.version is not None:
            artifact.dir = root / artifact.name / f"v{artifact.version}"
        return artifact

    def __repr__(self) -> str:
        return (
            f"Artifact({self.qualified_name}, type={self.type!r}, "
            f"files={len(self.entries)})"
        )


class ArtifactStore:
    """Project-scoped artifact storage shared by every run of that project."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._native = NativeArtifactStore(self.root)

    @staticmethod
    def _diagnostics(messages):
        for message in messages:
            logger.warning(message)

    @property
    def index_path(self) -> Path:
        return self.root / "index.jsonl"

    def log(
        self,
        artifact: Artifact,
        *,
        aliases: list[str] | None = None,
        mode: str = "copy",
        run: str | None = None,
        step: int | None = None,
    ) -> Artifact:
        """Store a new version, reusing an existing one with identical contents."""
        data = artifact.to_dict()
        data["aliases"] = [*artifact.aliases, *(aliases or [])]
        data["entries"] = [
            {**entry.to_dict(), "source": entry.source} for entry in artifact.entries
        ]
        result, diagnostics = self._native.log(
            data,
            mode,
            run,
            step,
            getattr(self, "_on_locked", None),
            getattr(self, "_link_errno", None),
        )
        self._diagnostics(diagnostics)
        artifact.aliases, artifact.digest = data["aliases"], data["digest"]
        logged = Artifact.from_dict(result, self.root)
        if not result["reused"]:
            artifact.__dict__.update(logged.__dict__)
            return artifact
        return logged

    def resolve(self, spec: str) -> Artifact | None:
        """Look up ``name``, ``name:latest`` or ``name:v3`` in the index."""
        result, diagnostics = self._native.resolve(spec)
        self._diagnostics(diagnostics)
        return Artifact.from_dict(result, self.root) if result is not None else None

    def entries(self) -> list[Artifact]:
        result, diagnostics = self._native.entries()
        self._diagnostics(diagnostics)
        return [Artifact.from_dict(data, self.root) for data in result]


def file_digest(path: Path) -> str:
    return artifact_file_digest(Path(path))


def coerce_artifact(
    artifact_or_path: Artifact | str | os.PathLike,
    name: str | None = None,
    type: str | None = None,
    metadata: dict | None = None,
) -> Artifact:
    """Accept an ``Artifact`` or a path, as ``wandb.log_artifact`` does."""
    if isinstance(artifact_or_path, Artifact):
        if type:
            artifact_or_path.type = type
        if metadata:
            artifact_or_path.metadata.update(metadata)
        return artifact_or_path
    path = Path(artifact_or_path)
    artifact = Artifact(
        name=name or path.stem or path.name,
        type=type or DEFAULT_TYPE,
        metadata=metadata,
    )
    if path.is_dir():
        artifact.add_dir(path)
    else:
        artifact.add_file(path)
    return artifact
