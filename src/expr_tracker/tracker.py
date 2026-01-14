import os
from contextvars import ContextVar
from typing import Literal

from loguru import logger

_tracker: ContextVar["Tracker | None"] = ContextVar("tracker", default=None)


def get_backend(backend: str | object):
    if isinstance(backend, str):
        backend = backend.lower()
        if backend == "wandb":
            try:
                import wandb

                wandb.login(
                    key=os.getenv("WANDB_API_KEY", None),
                    host=os.getenv("WANDB_HOST", None),
                )
                return wandb
            except ImportError:
                raise ImportError("WandB backend is not installed.")
        elif backend == "jsonl":
            from .jsonl import JsonlTracker

            return JsonlTracker()
        elif backend == "trackio":
            try:
                import trackio

                return trackio
            except ImportError:
                raise ImportError("TrackIO backend is not installed.")
    else:
        return backend


class Tracker:
    def __init__(
        self,
        project: str,
        name: str | None = None,
        entity: str | None = None,
        dir: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
        resume: bool | Literal["allow", "never", "must", "auto"] | None = "allow",
        config: dict | None = None,
        backends: list[Literal["wandb", "jsonl", "trackio"]] = [
            "wandb",
            "trackio",
            "jsonl",
        ],
        backend_kwargs: dict[str, dict] | None = None,
        **kwargs,
    ):
        if kwargs:
            logger.info(
                f"Unrecognized keyword arguments passed to Tracker: {kwargs}. "
                "These will be ignored."
            )
        self.backend = {b: get_backend(b) for b in backends}
        if config is None:
            config = {}
        if backend_kwargs is None:
            backend_kwargs = {}
        for b, t in self.backend.items():
            if b == "trackio":
                t.init(
                    project=project,
                    name=name,
                    config=config.update(
                        {
                            "trackio.notes": notes,
                            "trackio.tags": tags,
                            "trackio.entity": entity,
                            "trackio.resume": resume,
                        }
                    ),
                    **backend_kwargs.get(b, {}),
                )
            else:
                t.init(
                    project=project,
                    name=name,
                    entity=entity,
                    dir=dir,
                    notes=notes,
                    tags=tags,
                    resume=resume,
                    id=name,
                    config=config,
                    **backend_kwargs.get(b, {}),
                )


def init(
    project: str,
    name: str | None = None,
    entity: str | None = None,
    dir: str | None = None,
    notes: str | None = None,
    tags: list[str] | None = None,
    resume: bool | Literal["allow", "never", "must", "auto"] | None = "allow",
    config: dict | None = None,
    backends: list[Literal["wandb", "jsonl", "trackio"]] = ["wandb", "jsonl"],
    backend_kwargs: dict[str, dict] | None = None,
    **kwargs,
) -> Tracker:
    """
    Initialize the tracker with the given parameters.
    """
    if _tracker.get() is not None:
        raise RuntimeError("Tracker is already initialized. Call finish() first.")

    tracker = Tracker(
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
        **kwargs,
    )
    _tracker.set(tracker)
    return tracker


def log(metrics: dict, step: int | None = None):
    tracker = _tracker.get()
    if tracker is None:
        raise RuntimeError("Tracker is not initialized. Call init() first.")
    for backend, t in tracker.backend.items():
        try:
            t.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to {backend}: {e}")


def finish():
    tracker = _tracker.get()
    if tracker is None:
        raise RuntimeError("Tracker is not initialized. Call init() first.")
    for backend, t in tracker.backend.items():
        try:
            t.finish()
        except Exception as e:
            logger.warning(f"Failed to finish tracker for {backend}: {e}")
    _tracker.set(None)


def info():
    tracker = _tracker.get()
    if tracker is None:
        raise RuntimeError("Tracker is not initialized. Call init() first.")
    info = {}
    for backend, t in tracker.backend.items():
        if backend == "wandb":
            info["wandb"] = {"url": t.run.url}
        if backend == "jsonl":
            info["jsonl"] = {"log_dir": t.log_dir.as_posix()}
    return info
