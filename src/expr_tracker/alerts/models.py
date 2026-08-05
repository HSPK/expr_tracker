"""Alert models: levels, messages, channels, policies and rules (plain dataclasses)."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

DEFAULT_WATCHDOG_INTERVAL = 30.0


class AlertLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @classmethod
    def parse(cls, value: AlertLevel | str | None) -> AlertLevel:
        if value is None:
            return cls.INFO
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower()
        alias = {
            "warn": cls.WARNING,
            "warning": cls.WARNING,
            "err": cls.ERROR,
            "error": cls.ERROR,
            "fatal": cls.CRITICAL,
            "crit": cls.CRITICAL,
            "critical": cls.CRITICAL,
            "info": cls.INFO,
            "debug": cls.DEBUG,
        }
        if key not in alias:
            raise ValueError(
                f"Unknown alert level {value!r}; expected one of "
                f"{', '.join(level.value for level in cls)}."
            )
        return alias[key]

    @property
    def rank(self) -> int:
        return _LEVEL_ORDER[self]

    @staticmethod
    def _rank_of(other):
        """Rank of a comparable operand, or ``None`` if it is not a level.

        Names are accepted because this is a ``str`` enum: returning
        ``NotImplemented`` for a string would fall back to ``str`` ordering, and
        ``level >= "error"`` would then silently miss ``critical``.
        """
        if isinstance(other, AlertLevel):
            return other.rank
        if isinstance(other, str):
            # Let parse() raise for junk: falling back to str ordering would
            # silently answer an alphabetical question instead of a severity one
            return AlertLevel.parse(other).rank
        return None

    def __ge__(self, other):  # type: ignore[override]
        rank = self._rank_of(other)
        return NotImplemented if rank is None else self.rank >= rank

    def __gt__(self, other):  # type: ignore[override]
        rank = self._rank_of(other)
        return NotImplemented if rank is None else self.rank > rank

    def __le__(self, other):  # type: ignore[override]
        rank = self._rank_of(other)
        return NotImplemented if rank is None else self.rank <= rank

    def __lt__(self, other):  # type: ignore[override]
        rank = self._rank_of(other)
        return NotImplemented if rank is None else self.rank < rank


_LEVEL_ORDER = {
    AlertLevel.DEBUG: 0,
    AlertLevel.INFO: 1,
    AlertLevel.WARNING: 2,
    AlertLevel.ERROR: 3,
    AlertLevel.CRITICAL: 4,
}


@dataclass
class AlertMessage:
    title: str
    text: str
    subtitle: str | None = None
    level: AlertLevel = AlertLevel.INFO
    traceback: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    link: str | None = None
    source: str | None = None
    dedup_key: str | None = None
    ts: float = field(default_factory=time.time)

    def __post_init__(self):
        self.level = AlertLevel.parse(self.level)

    def key(self) -> str:
        if self.dedup_key:
            return self.dedup_key
        raw = f"{self.source or ''}|{self.title}|{self.level.value}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["level"] = self.level.value
        return data


@dataclass
class WebhookPolicy:
    timeout: float = 10.0
    max_retries: int = 3
    backoff_initial: float = 0.5
    backoff_factor: float = 2.0
    backoff_max: float = 30.0
    retry_on_status: tuple[int, ...] = (408, 429, 500, 502, 503, 504)
    respect_retry_after: bool = True
    rate_limit_per_minute: int | None = 20
    on_rate_limited: str = "coalesce"  # drop | queue | coalesce
    dedup_window: float = 300.0
    async_send: bool = True
    queue_size: int = 1000
    on_queue_full: str = "drop_oldest"  # drop_oldest | drop_new | block
    fail_silently: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> WebhookPolicy:
        if not data:
            return cls()
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown webhook policy options: {sorted(unknown)}")
        values = dict(data)
        if "retry_on_status" in values:
            values["retry_on_status"] = tuple(values["retry_on_status"])
        return cls(**values)

    def merged(self, override: WebhookPolicy | None) -> WebhookPolicy:
        return override or self


@dataclass
class ChannelConfig:
    type: str
    name: str | None = None
    url: str | None = None
    url_env: str | None = None
    enabled: bool = True
    min_level: AlertLevel = AlertLevel.INFO
    levels: list[AlertLevel] | None = None
    tags: list[str] | None = None
    options: dict[str, Any] = field(default_factory=dict)
    policy: WebhookPolicy | None = None

    def __post_init__(self):
        self.type = str(self.type).lower()
        self.name = self.name or self.type
        self.min_level = AlertLevel.parse(self.min_level)
        if self.levels is not None:
            self.levels = [AlertLevel.parse(level) for level in self.levels]

    @classmethod
    def from_dict(cls, data: dict) -> ChannelConfig:
        if "type" not in data:
            raise ValueError(f"Channel config requires a 'type' key: {data!r}")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown channel options: {sorted(unknown)}")
        values = dict(data)
        values["policy"] = (
            WebhookPolicy.from_dict(values["policy"])
            if isinstance(values.get("policy"), dict)
            else values.get("policy")
        )
        return cls(**values)

    def resolve_url(self) -> str | None:
        if self.url:
            return self.url
        if self.url_env:
            return os.getenv(self.url_env)
        return None

    def accepts(self, message: AlertMessage) -> bool:
        if not self.enabled:
            return False
        if self.levels is not None:
            if message.level not in self.levels:
                return False
        elif message.level < self.min_level:
            return False
        return not (self.tags and not set(self.tags) & set(message.tags))


@dataclass
class AlertRule:
    condition: str
    name: str | None = None
    level: AlertLevel = AlertLevel.WARNING
    title: str | None = None
    message: str = ""
    mode: str = "edge"  # edge | level
    for_steps: int = 1
    cooldown: float | None = 300.0
    max_fires: int | None = None
    notify_recovery: bool = False
    channels: list[str] | None = None
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    def __post_init__(self):
        self.level = AlertLevel.parse(self.level)
        if self.mode not in ("edge", "level"):
            raise ValueError(
                f"Unknown alert rule mode {self.mode!r}; use 'edge' or 'level'."
            )
        self.for_steps = max(1, int(self.for_steps))
        if not self.name:
            self.name = f"rule_{self._auto_digest()}"
        if not self.message:
            self.message = "{name} triggered at step {step}: {expr}"

    def _auto_digest(self) -> str:
        """Digest everything that makes two rules different alerts.

        Hashing the condition alone would give "loss > 1 => warning: notify" and
        "loss > 1 => critical: page" the same name, so registering both would
        silently keep only the last.
        """
        identity = "\0".join(
            [
                self.condition,
                self.level.value,
                self.message,
                self.title or "",
                self.mode,
                str(self.for_steps),
                ",".join(self.channels or []),
                ",".join(self.tags),
            ]
        )
        return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:8]

    @classmethod
    def from_dict(cls, data: dict) -> AlertRule:
        values = dict(data)
        # Accept Prometheus-style key names
        for src, dst in (
            ("alert", "name"),
            ("expr", "condition"),
            ("for", "for_steps"),
        ):
            if src in values and dst not in values:
                values[dst] = values.pop(src)
            values.pop(src, None)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown alert rule options: {sorted(unknown)}")
        return cls(**values)


@dataclass
class AlertConfig:
    channels: list[ChannelConfig] = field(default_factory=list)
    default_policy: WebhookPolicy = field(default_factory=WebhookPolicy)
    rules: list[AlertRule] = field(default_factory=list)
    enabled: bool = True
    # How often time-based rules (no_data/age/elapsed) are re-checked without a
    # log() call; it bounds how quickly a hung run is noticed.
    watchdog_interval: float = DEFAULT_WATCHDOG_INTERVAL

    @classmethod
    def from_dict(cls, data: dict | None) -> AlertConfig:
        if not data:
            return cls()
        known = {
            "channels",
            "default_policy",
            "policy",
            "rules",
            "enabled",
            "watchdog_interval",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown alert config options: {sorted(unknown)}")
        policy = data.get("default_policy", data.get("policy"))
        return cls(
            channels=[_as_channel(c) for c in data.get("channels", [])],
            default_policy=(
                WebhookPolicy.from_dict(policy)
                if isinstance(policy, dict)
                else (policy or WebhookPolicy())
            ),
            rules=[_as_rule(r) for r in data.get("rules", [])],
            enabled=bool(data.get("enabled", True)),
            watchdog_interval=float(
                data.get("watchdog_interval", DEFAULT_WATCHDOG_INTERVAL)
            ),
        )


def _as_channel(value: ChannelConfig | dict | str) -> ChannelConfig:
    if isinstance(value, ChannelConfig):
        return value
    if isinstance(value, str):
        return ChannelConfig(type=value)
    return ChannelConfig.from_dict(value)


def _as_rule(value: AlertRule | dict | str) -> AlertRule:
    if isinstance(value, AlertRule):
        return value
    if isinstance(value, str):
        from .expr.rule import parse_rule

        return parse_rule(value)
    return AlertRule.from_dict(value)


def default_channels_from_env() -> list[ChannelConfig]:
    """Infer default channels from the environment (legacy ``WEBHOOK_URL`` included)."""
    channels: list[ChannelConfig] = []
    for env, kind in (
        ("ET_LARK_WEBHOOK_URL", "lark"),
        ("WEBHOOK_URL", "lark"),
        ("ET_SLACK_WEBHOOK_URL", "slack"),
        ("ET_DINGTALK_WEBHOOK_URL", "dingtalk"),
        ("ET_WECOM_WEBHOOK_URL", "wecom"),
    ):
        url = os.getenv(env)
        if url and not any(c.type == kind for c in channels):
            channels.append(ChannelConfig(type=kind, url=url))
    return channels


def normalize_channels(value: Sequence | None) -> list[ChannelConfig]:
    return [_as_channel(item) for item in (value or [])]


def normalize_rules(value: Sequence | None) -> list[AlertRule]:
    return [_as_rule(item) for item in (value or [])]
