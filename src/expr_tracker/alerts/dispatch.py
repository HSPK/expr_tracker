"""Python backend/config adapters for native alert delivery lanes."""

from __future__ import annotations

import atexit

from loguru import logger

from .._native import (
    AlertChannel,
    AlertDeduper,
    AlertTokenBucket,
    alert_flush_channels,
)
from .._process import check_process, process_id
from .backends import create_backend
from .models import AlertConfig, AlertMessage

MAX_DEDUP_KEYS = 10_000
TokenBucket = AlertTokenBucket
Deduper = AlertDeduper


class ChannelRuntime:
    def __init__(self, config, backend, policy):
        self.config, self.backend, self.policy = config, backend, policy
        self._native = AlertChannel(config, backend, policy)

    def _stat(self, index):
        return self._native.stats()[index]

    sent = property(lambda self: self._stat(0))
    failed = property(lambda self: self._stat(1))
    suppressed = property(lambda self: self._stat(2))
    dropped = property(lambda self: self._stat(3))
    pending = property(lambda self: self._stat(4))
    alive = property(lambda self: self._stat(5))
    unfinished_tasks = property(lambda self: self._stat(6))


class Dispatcher:
    """Deliver through independently scheduled Rust channel lanes."""

    def __init__(self, config=None):
        self._owner_pid = process_id()
        config = config or AlertConfig()
        self.enabled = config.enabled
        self.default_policy = config.default_policy
        self.channels = {}
        self._atexit_registered = False
        for channel in config.channels:
            self.add_channel(channel)

    def add_channel(self, config):
        check_process(self._owner_pid)
        policy = config.policy or self.default_policy
        config.policy = policy
        backend = create_backend(config)
        name = config.name or config.type
        if name in self.channels:
            raise ValueError(f"Duplicate alert channel name: {name!r}")
        self.channels[name] = ChannelRuntime(config, backend, policy)

    def channel_names(self):
        check_process(self._owner_pid)
        return list(self.channels)

    def is_worker_thread(self):
        check_process(self._owner_pid)
        return any(rt._native.is_worker_thread() for rt in self.channels.values())

    @property
    def async_send(self):
        check_process(self._owner_pid)
        return any(rt.policy.async_send for rt in self.channels.values())

    def send(self, message, channels=None):
        check_process(self._owner_pid)
        if not self.enabled:
            return
        for runtime in self._resolve(message, channels):
            if runtime.policy.async_send and not self._atexit_registered:
                atexit.register(self._atexit)
                self._atexit_registered = True
            runtime._native.send(message)

    def _resolve(self, message, channels):
        from .._native import alert_select_channels

        names, missing = alert_select_channels(list(self.channels), channels)
        if missing:
            logger.warning(f"Unknown alert channels: {missing}")
        return [
            self.channels[name]
            for name in names
            if self.channels[name]._native.accepts(message)
        ]

    def _ensure_worker(self, runtime):
        runtime._native.start()

    def _enqueue(self, runtime, message):
        runtime._native.send(message)

    def _deliver(self, runtime, message):
        runtime._native.deliver(message)

    def _send_with_retry(self, runtime, message):
        runtime._native.deliver(message)

    def flush(self, timeout=5.0):
        check_process(self._owner_pid)
        alert_flush_channels(
            [runtime._native for runtime in self.channels.values()],
            timeout,
        )

    def close(self, timeout=5.0):
        check_process(self._owner_pid)
        self.flush(timeout)
        for runtime in self.channels.values():
            runtime._native.close(timeout)
        if self._atexit_registered:
            atexit.unregister(self._atexit)
            self._atexit_registered = False

    def _atexit(self):
        if self._owner_pid == process_id():
            self.close()

    def stats(self):
        check_process(self._owner_pid)
        return {
            name: {
                "type": rt.config.type,
                "sent": rt.sent,
                "failed": rt.failed,
                "suppressed": rt.suppressed,
                "dropped": rt.dropped,
                "pending": rt.pending,
                "alive": rt.alive,
            }
            for name, rt in self.channels.items()
        }


def _with_suffix(message, suffix):
    clone = AlertMessage(**{**message.to_dict(), "level": message.level})
    clone.text = message.text + suffix
    return clone
