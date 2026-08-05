"""Run summary: the wandb-style ``run.summary`` mapping, persisted as JSON."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any

from loguru import logger

from .history.codec import RecordCodec


class Summary(MutableMapping):
    """Last-value-wins mapping of metrics, plus anything the user sets explicitly.

    Explicit assignments win over automatic updates, so ``run.summary["best_acc"] = x``
    is not overwritten by later ``log()`` calls.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else None
        self._values: dict[str, Any] = {}
        self._pinned: set[str] = set()
        self._codec = RecordCodec()
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        if self.path is None or not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._values.update(data)
        except Exception as e:
            logger.warning(f"Could not read summary from {self.path}: {e}")

    def observe(self, metrics: dict):
        """Record the latest value of each logged metric, skipping pinned keys."""
        if not isinstance(metrics, dict):
            return
        with self._lock:
            for key, value in metrics.items():
                if not isinstance(key, str) or key.startswith("_"):
                    continue
                if key in self._pinned:
                    continue
                self._values[key] = value

    def save(self):
        if self.path is None:
            return
        with self._lock:
            # Per-field fallback, so one unserialisable value cannot degrade the rest
            payload = self._codec.encode(self._values, kind="summary")
        temporary = self.path.with_suffix(".json.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(temporary, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(temporary, self.path)
        except Exception as e:
            logger.warning(f"Failed to write summary to {self.path}: {e}")

    # ------------------------------------------------------------------ mapping

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._values[key]

    def __setitem__(self, key: str, value: Any):
        with self._lock:
            self._values[key] = value
            self._pinned.add(key)

    def __delitem__(self, key: str):
        with self._lock:
            del self._values[key]
            self._pinned.discard(key)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(dict(self._values))

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def __repr__(self) -> str:
        return f"Summary({dict(self._values)!r})"
