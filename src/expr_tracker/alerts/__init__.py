"""Public alert API: ``alert`` / ``configure_alert`` / ``add_alert_rule``."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from .backends import register_backend
from .dispatch import Dispatcher
from .engine import AlertEngine
from .expr import M, parse_rule
from .expr.eval import EvalContext
from .models import (
    AlertConfig,
    AlertLevel,
    AlertMessage,
    AlertRule,
    ChannelConfig,
    WebhookPolicy,
    default_channels_from_env,
    normalize_channels,
    normalize_rules,
)

_lock = threading.RLock()
_configured: AlertConfig | None = None
_default_dispatcher: Dispatcher | None = None
_warned_no_channel = False


# ---------------------------------------------------------------------- config


def load_config(source: Any = None) -> AlertConfig:
    """Resolve alert config: explicit argument > configure_alert > file > env."""
    if isinstance(source, AlertConfig):
        return source
    if isinstance(source, dict):
        return AlertConfig.from_dict(source)
    if isinstance(source, (str, Path)):
        return AlertConfig.from_dict(_read_config_file(source))
    if source is not None:
        raise TypeError(f"Unsupported alert config: {source!r}")
    with _lock:
        if _configured is not None:
            return _configured
    path = os.getenv("ET_ALERT_CONFIG")
    if path:
        return AlertConfig.from_dict(_read_config_file(path))
    return AlertConfig(channels=default_channels_from_env())


def configure_alert(
    channels: Sequence | None = None,
    *,
    policy: WebhookPolicy | dict | None = None,
    rules: Sequence | None = None,
    enabled: bool = True,
) -> AlertConfig:
    """Set the process-wide default used when ``et.init`` gets no ``alert=``."""
    global _configured, _default_dispatcher
    config = AlertConfig(
        channels=normalize_channels(channels) or default_channels_from_env(),
        default_policy=(
            WebhookPolicy.from_dict(policy)
            if isinstance(policy, dict)
            else (policy or WebhookPolicy())
        ),
        rules=normalize_rules(rules),
        enabled=enabled,
    )
    with _lock:
        _configured = config
        if _default_dispatcher is not None:
            _default_dispatcher.close(timeout=0.5)
        _default_dispatcher = None
    return config


def reset_alert_config():
    """Clear the process-wide alert config (for tests and reconfiguration)."""
    global _configured, _default_dispatcher, _warned_no_channel
    with _lock:
        if _default_dispatcher is not None:
            _default_dispatcher.close(timeout=0.5)
        _configured = None
        _default_dispatcher = None
        _warned_no_channel = False


def _read_config_file(path: str | Path) -> dict:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if file.suffix == ".toml":
        import tomllib

        data = tomllib.loads(text)
    elif file.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("Reading YAML alert config requires PyYAML") from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return data.get("alert", data) if isinstance(data, dict) else {}


def _dispatcher() -> Dispatcher:
    global _default_dispatcher
    with _lock:
        if _default_dispatcher is None:
            _default_dispatcher = Dispatcher(load_config())
        return _default_dispatcher


# ---------------------------------------------------------------------- engine


def build_engine(source: Any, rules: Sequence, run) -> AlertEngine:
    """Build the run's alert engine, even with no channels, so rules can be added."""
    config = load_config(source)
    dispatcher = Dispatcher(
        AlertConfig(
            channels=config.channels,
            default_policy=config.default_policy,
            enabled=config.enabled,
        )
    )
    history = run.history

    def context(record: dict | None) -> EvalContext:
        latest = record if record is not None else (history.open_record() or {})
        step = latest.get("_step") if latest else None
        if step is None:
            step = history.current_step
        # A committed step leaves no open row, so fall back to the store's last
        # commit time; otherwise no_data()/age() would think data never arrived
        last_commit = latest.get("_time") if record is not None else None
        return EvalContext(
            history.series,
            step=step if isinstance(step, int) else None,
            now=time.time(),
            started_at=history.started_at,
            last_commit_time=last_commit or history.last_commit_time,
            record=latest or {},
        )

    context.series = history.series  # lets the engine size metric buffers per rule
    return AlertEngine(
        dispatcher,
        context,
        rules=list(config.rules) + normalize_rules(rules),
        run_info={"project": run.project, "run": run.name},
        watchdog_interval=config.watchdog_interval,
    )


# ---------------------------------------------------------------------- manual


def alert(
    title: str,
    text: str = "",
    subtitle: str | None = None,
    traceback: str | None = None,
    level: str | AlertLevel = "info",
    channels: Sequence[str] | None = None,
    backends: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    mentions: Sequence[str] | None = None,
    fields: dict | None = None,
    link: str | None = None,
    source: str = "manual",
    dedup_key: str | None = None,
):
    """Send an alert. Delivery failures are logged, never raised to the caller.

    ``backends`` is the former name of ``channels`` and still works.
    """
    global _warned_no_channel
    if backends is not None and channels is None:
        channels = list(backends)
        logger.warning("alert(backends=...) is deprecated; use alert(channels=...).")
    message = AlertMessage(
        title=title,
        text=text,
        subtitle=subtitle,
        level=AlertLevel.parse(level),
        traceback=traceback,
        fields=dict(fields or {}),
        tags=list(tags or []),
        mentions=list(mentions or []),
        link=link,
        source=source,
        dedup_key=dedup_key,
    )
    dispatcher = _run_dispatcher() or _dispatcher()
    if not dispatcher.channels:
        if not _warned_no_channel:
            _warned_no_channel = True
            logger.warning(
                "No alert channels configured; set ET_LARK_WEBHOOK_URL or call "
                "et.configure_alert(...)."
            )
        return
    try:
        dispatcher.send(message, list(channels) if channels else None)
    except Exception as e:
        logger.warning(f"Failed to dispatch alert: {e}")


def _run_dispatcher() -> Dispatcher | None:
    from ..run import current_run

    run = current_run()
    if run is not None and run.alerts is not None:
        return run.alerts.dispatcher
    return None


# ---------------------------------------------------------------------- rules


def _engine() -> AlertEngine:
    from ..run import require_run

    return require_run().alerts


def add_alert_rule(rule, **overrides) -> AlertRule:
    """Add an alert rule from a string, ``AlertRule``, dict or builder expression."""
    return _engine().add_rule(rule, **overrides).rule


def remove_alert_rule(name: str) -> bool:
    return _engine().remove_rule(name)


def list_alert_rules() -> list[AlertRule]:
    from ..run import current_run

    run = current_run()
    if run is None or run.alerts is None:
        return []
    return run.alerts.list_rules()


__all__ = [
    "AlertConfig",
    "AlertEngine",
    "AlertLevel",
    "AlertMessage",
    "AlertRule",
    "ChannelConfig",
    "Dispatcher",
    "M",
    "WebhookPolicy",
    "add_alert_rule",
    "alert",
    "build_engine",
    "configure_alert",
    "list_alert_rules",
    "load_config",
    "parse_rule",
    "register_backend",
    "remove_alert_rule",
    "reset_alert_config",
]
