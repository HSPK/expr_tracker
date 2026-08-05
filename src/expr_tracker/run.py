"""Run orchestration: global singleton, backend fan-out, history, alert engine."""

from __future__ import annotations

import atexit
import contextlib
import inspect
import json
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from .artifacts import Artifact, ArtifactStore, coerce_artifact
from .history import HistoryStore, current_rank, resolve_commit
from .summary import Summary

_lock = threading.RLock()
_current: Run | None = None

BackendName = Literal["wandb", "jsonl", "trackio"]


def get_backend(backend: str | Any):
    if not isinstance(backend, str):
        return backend
    name = backend.lower()
    if name == "wandb":
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                'The wandb backend needs wandb: pip install "expr_tracker[wandb]"'
            ) from e
        wandb.login(
            key=os.getenv("WANDB_API_KEY", None), host=os.getenv("WANDB_HOST", None)
        )
        return wandb
    if name == "trackio":
        try:
            import trackio
        except ImportError as e:
            raise ImportError(
                'The trackio backend needs trackio: pip install "expr_tracker[trackio]"'
            ) from e
        return trackio
    if name == "jsonl":
        return None  # jsonl is always on via HistoryStore, not an optional backend
    raise ValueError(f"Unknown backend: {backend!r}")


class Run:
    """A single experiment run. ``history`` is always present; backends are optional."""

    def __init__(
        self,
        project: str,
        name: str | None = None,
        entity: str | None = None,
        dir: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
        resume: bool | str | None = "allow",
        config: dict | None = None,
        backends: Sequence[str] = ("wandb", "jsonl"),
        backend_kwargs: dict[str, dict] | None = None,
        print_to_screen: bool = False,
        alert: Any = None,
        alert_rules: Sequence[Any] = (),
        alert_on_rank: int | None = 0,
        **history_kwargs,
    ):
        self.project = project
        self.config = dict(config or {})
        self.entity = entity
        self.notes = notes
        self.tags = list(tags or [])
        backend_kwargs = dict(backend_kwargs or {})
        self._finished = False
        # Probed once: retrying on TypeError would double-log a backend whose own
        # code raised TypeError for an unrelated reason
        self._takes_commit: dict[str, bool] = {}
        # Construction is transactional: anything already opened is closed again if
        # a later step (an invalid alert rule, say) raises.
        self._closers: list = []

        self.history = HistoryStore()
        self.history.init(
            project=project,
            name=name,
            config=self.config or None,
            dir=dir,
            print_to_screen=print_to_screen,
            on_commit=self._on_commit,
            **{**backend_kwargs.get("jsonl", {}), **history_kwargs},
        )
        self._closers.append(self.history.finish)
        self.name = self.history.name
        self.summary = Summary(self.history.log_dir / "summary.json")
        # History survives a crash through its own atexit hook; the summary is
        # just as much a record of the run, so persist it the same way
        atexit.register(self._save_summary_at_exit)
        self._closers.append(lambda: atexit.unregister(self._save_summary_at_exit))
        self.artifacts = ArtifactStore(
            root=Path(dir or "./tracker/jsonl") / project / "artifacts"
        )

        self.backends: dict[str, Any] = {}
        self.alerts = None
        try:
            for backend in dict.fromkeys(backends):
                if backend == "jsonl":
                    continue
                try:
                    instance = get_backend(backend)
                except Exception as e:
                    logger.warning(f"Skipping backend {backend!r}: {e}")
                    continue
                if instance is None:
                    continue
                name = (
                    backend
                    if isinstance(backend, str)
                    else type(instance).__name__.lower()
                )
                self.backends[name] = instance
                self._takes_commit[name] = _accepts_commit(instance)
                self._init_backend(
                    name,
                    instance,
                    entity=entity,
                    dir=dir,
                    notes=notes,
                    tags=tags,
                    resume=resume,
                    kwargs=backend_kwargs.get(name, {}),
                )
                self._closers.append(instance.finish)
            self.rank = current_rank()
            # Every rank would otherwise raise the same alert N times
            if alert_on_rank is not None and self.rank != alert_on_rank:
                alert, alert_rules = {"enabled": False}, ()
            self.alerts = self._build_alert_engine(alert, alert_rules)
            self._closers.append(self.alerts.close)
        except Exception:
            self._rollback()
            raise

    def _init_backend(
        self, name, instance, *, entity, dir, notes, tags, resume, kwargs
    ):
        try:
            if name == "trackio":
                # trackio.init has no entity/notes/tags/dir/id, so those ride along
                # in the config; resume is a real parameter and must be forwarded
                config = {
                    **self.config,
                    "trackio.notes": notes,
                    "trackio.tags": tags,
                    "trackio.entity": entity,
                }
                instance.init(
                    project=self.project,
                    name=self.name,
                    config=config,
                    resume=_trackio_resume(resume),
                    **kwargs,
                )
            else:
                instance.init(
                    project=self.project,
                    name=self.name,
                    entity=entity,
                    dir=dir,
                    notes=notes,
                    tags=tags,
                    resume=resume,
                    id=self.name,
                    config=self.config,
                    **kwargs,
                )
        except Exception as e:
            logger.warning(f"Failed to initialize backend {name}: {e}")
            self.backends.pop(name, None)

    def _save_summary_at_exit(self):
        with contextlib.suppress(Exception):  # pragma: no cover - shutdown
            self.summary.save()

    def _rollback(self):
        """Close everything opened so far, newest first, ignoring secondary errors."""
        for close in reversed(self._closers):
            try:
                close()
            except Exception as e:  # pragma: no cover - best-effort teardown
                logger.warning(f"Failed to roll back run component: {e}")
        self._closers.clear()

    def _build_alert_engine(self, alert, alert_rules):
        from .alerts import build_engine

        # Config and rule errors are programming errors: raise instead of degrading
        return build_engine(alert, alert_rules, run=self)

    # ------------------------------------------------------------------ logging

    def _on_commit(self, record: dict):
        if self.alerts is not None:
            self.alerts.on_step(record)

    def log(self, data: dict, step: int | None = None, commit: bool | None = None):
        """Log metrics, mirroring ``wandb.log(data, step=..., commit=...)``.

        All sinks share one timeline: if local history rejects the call (closed run,
        or a backward step), the summary and remote backends are skipped too.
        """
        resolved_step = self.history.log(data, step=step, commit=commit)
        if resolved_step is None:
            return
        try:
            self.summary.observe(data)
        except Exception as e:  # a sink must never break the training loop
            logger.warning(f"Failed to update summary: {e}")
        for name, backend in self.backends.items():
            # Forward the resolved step and commit so a backend's row layout
            # matches the local history instead of drifting on its own counter.
            extra = (
                {"commit": resolve_commit(step, commit)}
                if self._takes_commit.get(name)
                else {}
            )
            try:
                backend.log(data, step=resolved_step, **extra)
            except Exception as e:
                logger.warning(f"Failed to log metrics to {name}: {e}")

    def history_query(self, *args, **kwargs):
        return self.history.get(*args, **kwargs)

    @property
    def step(self) -> int:
        """The step the next ``log()`` without an explicit step would use."""
        return self.history.current_step

    @property
    def dir(self) -> str:
        return self.history.log_dir.as_posix() if self.history.log_dir else ""

    @property
    def url(self) -> str | None:
        backend = self.backends.get("wandb")
        try:
            return backend.run.url if backend is not None else None
        except Exception:
            return None

    def define_metric(self, name: str, **kwargs):
        """Forwarded to backends that support it; a no-op for local history."""
        for backend_name, backend in self.backends.items():
            define = getattr(backend, "define_metric", None)
            if define is None:
                continue
            try:
                define(name, **kwargs)
            except Exception as e:
                logger.warning(f"define_metric failed on {backend_name}: {e}")

    # ------------------------------------------------------------------ artifacts

    def log_artifact(
        self,
        artifact_or_path: Artifact | str,
        name: str | None = None,
        type: str | None = None,
        aliases: list[str] | None = None,
        metadata: dict | None = None,
        mode: str = "copy",
    ) -> Artifact:
        """Store a versioned file bundle, mirroring ``wandb.log_artifact``."""
        artifact = coerce_artifact(
            artifact_or_path, name=name, type=type, metadata=metadata
        )
        logged = self.artifacts.log(
            artifact, aliases=aliases, mode=mode, run=self.name, step=self.step
        )
        self._record_artifact("log", logged)
        for backend_name, backend in self.backends.items():
            log_artifact = getattr(backend, "log_artifact", None)
            if log_artifact is None:
                continue
            try:
                log_artifact(
                    str(logged.dir or artifact_or_path),
                    name=logged.name,
                    type=logged.type,
                    aliases=aliases,
                )
            except Exception as e:
                logger.warning(f"Failed to log artifact to {backend_name}: {e}")
        return logged

    def use_artifact(
        self, artifact_or_name: Artifact | str, type: str | None = None
    ) -> Artifact:
        """Resolve a previously logged artifact of this project."""
        if isinstance(artifact_or_name, Artifact):
            resolved = artifact_or_name
        else:
            resolved = self.artifacts.resolve(artifact_or_name)
            if resolved is None:
                raise FileNotFoundError(
                    f"No artifact matching {artifact_or_name!r} "
                    f"in {self.artifacts.root}"
                )
        if type is not None and resolved.type != type:
            raise ValueError(
                f"Artifact {resolved.qualified_name} has type "
                f"{resolved.type!r}, not {type!r}"
            )
        self._record_artifact("use", resolved)
        return resolved

    def _record_artifact(self, action: str, artifact: Artifact):
        """Append run-level lineage so a run records what it produced and consumed."""
        if self.history.log_dir is None:
            return
        record = {
            "_time": time.time(),
            "action": action,
            "name": artifact.name,
            "version": artifact.version,
            "type": artifact.type,
            "step": self.step,
        }
        try:
            with open(
                self.history.log_dir / "artifacts.jsonl", "a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to record artifact lineage: {e}")

    def info(self) -> dict:
        info: dict[str, Any] = {"history": self.history.stats()}
        info["jsonl"] = {"log_dir": info["history"]["log_dir"]}
        info["artifacts"] = {"root": self.artifacts.root.as_posix()}
        info["rank"] = self.rank
        info["summary"] = dict(self.summary)
        if self.alerts is not None:
            info["alerts"] = {
                "rules": self.alerts.stats(),
                "channels": self.alerts.dispatcher.stats(),
            }
        for name, backend in self.backends.items():
            if name == "wandb":
                try:
                    info["wandb"] = {"url": backend.run.url}
                except Exception:
                    info["wandb"] = {}
            else:
                info[name] = {}
        return info

    def finish(self, exit_code: int | None = None, quiet: bool | None = None):
        """Close the run. ``exit_code``/``quiet`` exist for wandb compatibility."""
        if self._finished:
            return
        self._finished = True
        # Commit the final open row (it may fire an alert) before closing dispatch
        try:
            self.history.flush(commit_open=True)
        except Exception as e:
            logger.warning(f"Failed to flush history on finish: {e}")
        self.summary.save()
        atexit.unregister(self._save_summary_at_exit)
        if self.alerts is not None:
            self.alerts.close()
        self.history.finish()
        for name, backend in self.backends.items():
            try:
                backend.finish(
                    exit_code=exit_code
                ) if name == "wandb" else backend.finish()
            except Exception as e:
                logger.warning(f"Failed to finish tracker for {name}: {e}")


# ---------------------------------------------------------------------- global state


def _trackio_resume(resume) -> str:
    """Map wandb-style resume values onto trackio's must/allow/never, which it
    validates strictly and would otherwise reject."""
    if resume == "must":
        return "must"
    if resume == "never" or resume is False or resume is None:
        return "never"
    return "allow"


def _accepts_commit(backend) -> bool:
    """Whether ``backend.log`` takes a ``commit`` argument, as wandb's does."""
    try:
        parameters = inspect.signature(backend.log).parameters
    except (TypeError, ValueError):  # builtins and C callables
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return True
    return "commit" in parameters


def set_run(run: Run | None):
    global _current
    with _lock:
        _current = run


def current_run() -> Run | None:
    with _lock:
        return _current


def require_run() -> Run:
    run = current_run()
    if run is None:
        raise RuntimeError("Tracker is not initialized. Call init() first.")
    return run
