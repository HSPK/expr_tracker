"""Run summary: the wandb-style ``run.summary`` mapping, persisted as JSON."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any

from loguru import logger

from ._native import NativeSummary
from ._process import check_process, process_id
from .history.codec import RecordCodec


class Summary(MutableMapping):
    """Last-value-wins mapping of metrics, plus anything the user sets explicitly.

    Explicit assignments win over automatic updates, so ``run.summary["best_acc"] = x``
    is not overwritten by later ``log()`` calls.
    """

    def __init__(self, path: Path | None = None):
        self._owner_pid = process_id()
        self.path = Path(path) if path is not None else None
        self._native = NativeSummary()
        self._codec = RecordCodec()
        self._load()

    def _load(self):
        if self.path is None:
            return
        try:
            self._native.load_path(self.path)
        except Exception as e:
            logger.warning(f"Could not read summary from {self.path}: {e}")

    def observe(self, metrics: dict):
        """Record the latest value of each logged metric, skipping pinned keys."""
        check_process(self._owner_pid)
        if not isinstance(metrics, dict):
            return
        self._native.observe(metrics)

    _native_observe = observe

    def save(self):
        check_process(self._owner_pid)
        if self.path is None:
            return
        payload = self._codec.encode(self._native.snapshot(), kind="summary")
        try:
            self._native.save(self.path, payload)
        except Exception as e:
            logger.warning(f"Failed to write summary to {self.path}: {e}")

    # ------------------------------------------------------------------ mapping

    def __getitem__(self, key: str) -> Any:
        check_process(self._owner_pid)
        return self._native.get(key)

    def __setitem__(self, key: str, value: Any):
        check_process(self._owner_pid)
        self._native.set(key, value)

    def __delitem__(self, key: str):
        check_process(self._owner_pid)
        self._native.delete(key)

    def __iter__(self) -> Iterator[str]:
        check_process(self._owner_pid)
        return iter(self._native.keys())

    def __len__(self) -> int:
        check_process(self._owner_pid)
        return self._native.len()

    def __repr__(self) -> str:
        return f"Summary({self._native.snapshot()!r})"
