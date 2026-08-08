"""Public API: ``init`` / ``log`` / ``history`` / ``finish`` / ``info``."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from .artifacts import Artifact
from .history import read_history
from .run import Run, current_run, require_run, set_run
from .run import current_run as current

Tracker = Run


_lifecycle = threading.RLock()


def init(
    project: str,
    name: str | None = None,
    entity: str | None = None,
    dir: str | None = None,
    notes: str | None = None,
    tags: list[str] | None = None,
    resume: bool | Literal["allow", "never", "must", "auto"] | None = "allow",
    config: dict | None = None,
    backends: Sequence[Literal["wandb", "jsonl", "trackio"]] = ("wandb", "jsonl"),
    backend_kwargs: dict[str, dict] | None = None,
    print_to_screen: bool = False,
    alert=None,
    alert_rules: Sequence = (),
    **kwargs,
) -> Run:
    """Initialise the tracker. Local jsonl history is always on, whatever ``backends``.

    Extra keyword arguments are forwarded to the history store (``cache_bytes``,
    ``alert_window``, ``max_open_seconds``, ``step_policy``, ``buffer_size``, ...).
    """
    # Check, construct and publish atomically: concurrent init would race otherwise
    with _lifecycle:
        if current_run() is not None:
            raise RuntimeError("Tracker is already initialized. Call finish() first.")
        run = Run(
            project=project,
            name=name,
            entity=entity,
            dir=dir,
            notes=notes,
            tags=tags,
            resume=resume,
            config=config,
            backends=backends,
            backend_kwargs=backend_kwargs,
            print_to_screen=print_to_screen,
            alert=alert,
            alert_rules=alert_rules,
            **kwargs,
        )
        set_run(run)
        return run


def log(data: dict, step: int | None = None, commit: bool | None = None):
    """Log metrics; the signature mirrors ``wandb.log``.

    Repeated calls for the same step are merged into a single row.
    """
    require_run().log(data, step=step, commit=commit)


class _Current:
    """Sentinel: ``stream`` was not given, so use the running stream.

    ``stream=None`` has to keep meaning the default, unnamed producer.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<current stream>"


CURRENT_STREAM = _Current()


def history(
    n: int | None = 50,
    *,
    stream: str | _Current | None = CURRENT_STREAM,
    output_type: str = "dict",
    metrics: Sequence[str] | None = None,
    step_range: tuple[int | None, int | None] | None = None,
    include_meta: bool = True,
    include_open: bool = True,
    fill_missing: bool = False,
    dropna: bool = False,
    run: str | Path | None = None,
):
    """Return the last ``n`` steps (``n=-1``/``None`` for everything).

    ``output_type="pandas"`` returns a DataFrame (pandas is an optional extra).
    Passing ``run`` reads that run directory or file offline, without ``init()``.
    ``stream`` selects an independent producer, ``None`` being the default one;
    another process's stream is read from its file, so it reflects what that
    process has flushed. Omit it to read whichever stream this process writes.
    """
    active = getattr(current(), "stream", None)
    if run is None and (stream is CURRENT_STREAM or stream == active):
        return require_run().history.get(
            n,
            output_type=output_type,
            metrics=metrics,
            step_range=step_range,
            include_meta=include_meta,
            include_open=include_open,
            fill_missing=fill_missing,
            dropna=dropna,
        )
    if run is None:
        run = require_run().history.log_dir
    return read_history(
        run,
        n,
        stream=None if stream is CURRENT_STREAM else stream,
        output_type=output_type,
        metrics=metrics,
        step_range=step_range,
        include_meta=include_meta,
        fill_missing=fill_missing,
        dropna=dropna,
    )


def finish(exit_code: int | None = None, quiet: bool | None = None):
    """Close the active run. ``exit_code``/``quiet`` exist for wandb compatibility."""
    with _lifecycle:
        run = require_run()
        try:
            run.finish(exit_code=exit_code, quiet=quiet)
        finally:
            set_run(None)


def info() -> dict:
    return require_run().info()


def get_run() -> Run | None:
    """The active run, or ``None``. Mirrors ``wandb.run``."""
    return current_run()


def define_metric(name: str, **kwargs):
    """Forwarded to backends that support it; a no-op for local history."""
    require_run().define_metric(name, **kwargs)


def log_artifact(
    artifact_or_path: Artifact | str,
    name: str | None = None,
    type: str | None = None,
    aliases: list[str] | None = None,
    metadata: dict | None = None,
    mode: str = "copy",
) -> Artifact:
    """Store a versioned file bundle, mirroring ``wandb.log_artifact``.

    ``mode`` controls local storage: ``"copy"`` (default) is safe when the source is
    later overwritten in place, ``"link"`` hard-links for zero extra disk but then
    shares the caller's inode, ``"reference"`` records paths without materialising.
    """
    return require_run().log_artifact(
        artifact_or_path,
        name=name,
        type=type,
        aliases=aliases,
        metadata=metadata,
        mode=mode,
    )


def use_artifact(artifact_or_name: Artifact | str, type: str | None = None) -> Artifact:
    """Resolve ``name``, ``name:latest``, ``name:v3`` or ``name:<alias>``."""
    return require_run().use_artifact(artifact_or_name, type=type)


def summary():
    """The run summary mapping (last value per metric, plus explicit entries)."""
    return require_run().summary
