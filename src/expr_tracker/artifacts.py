"""Local artifact store: versioned file bundles logged alongside a run.

Layout under ``<dir>/<project>/artifacts``::

    index.jsonl          one line per logged version (project-wide)
    <name>/v<N>/...      materialised files for that version

Files are hard-linked when possible so checkpoints cost no extra disk space.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

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
        if not path.is_file():
            raise FileNotFoundError(f"Artifact file does not exist: {path}")
        self._add(path, name or path.name)
        return self

    def add_dir(
        self, local_dir: str | os.PathLike, name: str | None = None
    ) -> Artifact:
        root = Path(local_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"Artifact directory does not exist: {root}")
        prefix = f"{name.rstrip('/')}/" if name else ""
        for file in sorted(p for p in root.rglob("*") if p.is_file()):
            self._add(file, prefix + file.relative_to(root).as_posix())
        return self

    def add_reference(self, uri: str, name: str | None = None) -> Artifact:
        """Record an external URI without copying anything."""
        self.entries.append(
            ArtifactEntry(path=name or uri.rsplit("/", 1)[-1] or uri, ref=uri)
        )
        return self

    def _add(self, path: Path, logical: str):
        self.entries.append(
            ArtifactEntry(
                path=logical,
                size=path.stat().st_size,
                digest=file_digest(path),
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
        target.mkdir(parents=True, exist_ok=True)
        for entry in self.entries:
            if entry.ref:
                continue
            source = self.dir / entry.path
            destination = target / entry.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                shutil.copy2(source, destination)
        return target

    def get_path(self, name: str) -> Path:
        if self.dir is None:
            raise RuntimeError(f"Artifact {self.name!r} has not been logged yet")
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


@dataclass
class ArtifactStore:
    """Project-scoped artifact storage shared by every run of that project."""

    root: Path
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

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
        if mode not in ("link", "copy", "reference"):
            raise ValueError(
                f"Unknown artifact mode {mode!r}; use link, copy or reference"
            )
        artifact.aliases = sorted({*artifact.aliases, *(aliases or [])} - {LATEST})
        artifact.digest = self._digest(artifact)
        with self._lock:
            # One pass over the index answers both questions: has this exact content
            # been stored before, and what is the next free version?
            versions = [a for a in self.entries() if a.name == artifact.name]
            existing = next(
                (a for a in reversed(versions) if a.digest == artifact.digest), None
            )
            if existing is not None:
                existing.aliases = sorted({*existing.aliases, *artifact.aliases})
                self._append(existing, reused=True)
                return existing
            known = [a.version for a in versions if a.version is not None]
            artifact.version = max(known, default=-1) + 1
            artifact.dir = self.root / artifact.name / f"v{artifact.version}"
            artifact.created_at = time.time()
            artifact.run, artifact.step = run, step
            if mode != "reference":
                self._materialise(artifact, mode)
            self._append(artifact)
        return artifact

    def resolve(self, spec: str) -> Artifact | None:
        """Look up ``name``, ``name:latest`` or ``name:v3`` in the index."""
        name, _, selector = spec.partition(":")
        selector = selector or LATEST
        matches = [a for a in self.entries() if a.name == name]
        if not matches:
            return None
        if selector == LATEST:
            return matches[-1]
        if selector.startswith("v") and selector[1:].isdigit():
            version = int(selector[1:])
            return next((a for a in matches if a.version == version), None)
        return next((a for a in reversed(matches) if selector in a.aliases), None)

    def entries(self) -> list[Artifact]:
        if not self.index_path.exists():
            return []
        artifacts: list[Artifact] = []
        by_version: dict[tuple, Artifact] = {}
        with open(self.index_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception as e:
                    logger.warning(f"Skipping corrupted artifact index line: {e}")
                    continue
                key = (data.get("name"), data.get("version"))
                if data.get("reused"):
                    # A reuse only records new aliases; fold them into the version
                    known = by_version.get(key)
                    if known is not None:
                        added = data.get("aliases", [])
                        known.aliases = sorted({*known.aliases, *added})
                    continue
                artifact = Artifact.from_dict(data, self.root)
                by_version[key] = artifact
                artifacts.append(artifact)
        return artifacts

    # ------------------------------------------------------------------ internals

    def _digest(self, artifact: Artifact) -> str:
        hasher = hashlib.sha256()
        hasher.update(f"{artifact.type}\0".encode())
        for entry in sorted(artifact.entries, key=lambda e: e.path):
            hasher.update(f"{entry.path}\0{entry.digest}\0{entry.ref or ''}\0".encode())
        return f"sha256:{hasher.hexdigest()}"

    def _materialise(self, artifact: Artifact, mode: str):
        assert artifact.dir is not None
        artifact.dir.mkdir(parents=True, exist_ok=True)
        for entry in artifact.entries:
            if entry.ref or not entry.source:
                continue
            destination = artifact.dir / entry.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                continue
            source = Path(entry.source)
            try:
                if mode == "link":
                    os.link(source, destination)
                else:
                    shutil.copy2(source, destination)
            except OSError:
                shutil.copy2(source, destination)

    def _append(self, artifact: Artifact, reused: bool = False):
        self.root.mkdir(parents=True, exist_ok=True)
        record = artifact.to_dict()
        record["reused"] = reused
        try:
            with open(self.index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to update artifact index {self.index_path}: {e}")


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


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
