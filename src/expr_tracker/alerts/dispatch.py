"""Alert delivery: routing, rate limiting, dedup, retries, async queue."""

from __future__ import annotations

import atexit
import queue
import random
import threading
import time

from loguru import logger

from .backends import create_backend
from .backends.base import AlertBackend, SendError
from .models import AlertConfig, AlertMessage, ChannelConfig, WebhookPolicy

MAX_DEDUP_KEYS = 10_000


class TokenBucket:
    """Simple token bucket granting ``rate_per_minute`` tokens per minute."""

    def __init__(self, rate_per_minute: int | None):
        self.capacity = float(rate_per_minute or 0)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        if self.capacity <= 0:
            return True
        with self._lock:
            now = time.monotonic()
            self.tokens = min(
                self.capacity, self.tokens + (now - self.updated) * self.capacity / 60.0
            )
            self.updated = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False


class Deduper:
    """Suppress repeats of a key inside a window, then report how many were dropped."""

    def __init__(self, window: float):
        self.window = float(window or 0)
        self._seen: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Return ``(allowed, number suppressed since the last send)``."""
        if self.window <= 0:
            return True, 0
        now = now if now is not None else time.monotonic()
        with self._lock:
            last, suppressed = self._seen.get(key, (None, 0))
            if last is None or now - last >= self.window:
                self._seen[key] = (now, 0)
                if len(self._seen) > MAX_DEDUP_KEYS:
                    self._prune(now)
                return True, suppressed
            self._seen[key] = (last, suppressed + 1)
            return False, suppressed + 1

    def _prune(self, now: float):
        """Drop long-expired keys so dedup state stays bounded."""
        expired = [
            k for k, (last, _) in self._seen.items() if now - last >= self.window
        ]
        for key in expired:
            del self._seen[key]


class ChannelRuntime:
    """One channel plus its own queue and worker.

    Each channel gets a private lane so a slow or retrying channel (email, a rate
    limited webhook) cannot delay unrelated ones.
    """

    def __init__(
        self, config: ChannelConfig, backend: AlertBackend, policy: WebhookPolicy
    ):
        self.config = config
        self.backend = backend
        self.policy = policy
        self.bucket = TokenBucket(policy.rate_limit_per_minute)
        self.deduper = Deduper(policy.dedup_window)
        self.sent = 0
        self.failed = 0
        self.suppressed = 0
        self.dropped = 0
        self.queue: queue.Queue | None = None
        self.worker: threading.Thread | None = None

    @property
    def pending(self) -> int:
        return self.queue.qsize() if self.queue is not None else 0


class Dispatcher:
    """Deliver an :class:`AlertMessage` to every matching channel."""

    def __init__(self, config: AlertConfig | None = None):
        config = config or AlertConfig()
        self.enabled = config.enabled
        self.default_policy = config.default_policy
        self.channels: dict[str, ChannelRuntime] = {}
        for channel in config.channels:
            self.add_channel(channel)
        self._stopping = threading.Event()
        self._atexit_registered = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ config

    def add_channel(self, config: ChannelConfig):
        policy = config.policy or self.default_policy
        config.policy = policy
        backend = create_backend(config)
        name = config.name or config.type
        if name in self.channels:
            raise ValueError(f"Duplicate alert channel name: {name!r}")
        self.channels[name] = ChannelRuntime(config, backend, policy)

    def channel_names(self) -> list[str]:
        return list(self.channels)

    @property
    def async_send(self) -> bool:
        return any(rt.policy.async_send for rt in self.channels.values())

    # ------------------------------------------------------------------ sending

    def send(self, message: AlertMessage, channels: list[str] | None = None):
        if not self.enabled:
            return
        targets = self._resolve(message, channels)
        for runtime in targets:
            if runtime.policy.async_send:
                self._enqueue(runtime, message)
            else:
                self._deliver(runtime, message)

    def _resolve(
        self, message: AlertMessage, channels: list[str] | None
    ) -> list[ChannelRuntime]:
        if channels:
            missing = [name for name in channels if name not in self.channels]
            if missing:
                logger.warning(f"Unknown alert channels: {missing}")
            selected = [
                self.channels[name] for name in channels if name in self.channels
            ]
        else:
            selected = list(self.channels.values())
        return [rt for rt in selected if rt.config.accepts(message)]

    def _enqueue(self, runtime: ChannelRuntime, message: AlertMessage):
        self._ensure_worker(runtime)
        assert runtime.queue is not None
        try:
            runtime.queue.put_nowait(message)
            return
        except queue.Full:
            pass
        behaviour = runtime.policy.on_queue_full
        if behaviour == "block":
            runtime.queue.put(message)
            return
        if behaviour == "drop_oldest":
            try:
                runtime.queue.get_nowait()
                runtime.queue.task_done()
                runtime.dropped += 1
            except queue.Empty:
                pass
            try:
                runtime.queue.put_nowait(message)
                return
            except queue.Full:
                pass
        runtime.dropped += 1
        logger.warning(f"Alert queue for {runtime.config.name!r} is full; dropping.")

    def _ensure_worker(self, runtime: ChannelRuntime):
        with self._lock:
            if runtime.queue is None:
                runtime.queue = queue.Queue(maxsize=max(1, runtime.policy.queue_size))
            if runtime.worker is None or not runtime.worker.is_alive():
                self._stopping.clear()
                runtime.worker = threading.Thread(
                    target=self._run,
                    args=(runtime,),
                    name=f"et-alert-{runtime.config.name}",
                    daemon=True,
                )
                runtime.worker.start()
            if not self._atexit_registered:
                # Workers are daemon threads: drain the queues at exit or lose alerts
                atexit.register(self.close)
                self._atexit_registered = True

    def _run(self, runtime: ChannelRuntime):
        assert runtime.queue is not None
        while not self._stopping.is_set() or not runtime.queue.empty():
            try:
                message = runtime.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._deliver(runtime, message)
            except Exception as e:  # one bad message must not kill the worker
                logger.warning(f"Alert dispatch worker error: {e}")
            finally:
                runtime.queue.task_done()

    # ------------------------------------------------------------------ one message

    def _deliver(self, runtime: ChannelRuntime, message: AlertMessage):
        policy = runtime.policy
        allowed, suppressed = runtime.deduper.check(message.key())
        if not allowed:
            runtime.suppressed += 1
            return
        if not runtime.bucket.acquire():
            if policy.on_rate_limited != "queue":
                # drop/coalesce: skip this one; the count rides along on the next send
                runtime.suppressed += 1
                return
            interval = 60.0 / max(1, policy.rate_limit_per_minute or 1)
            deadline = time.monotonic() + max(interval, 1.0) * 60
            while not runtime.bucket.acquire():
                if time.monotonic() >= deadline or self._stopping.is_set():
                    runtime.suppressed += 1
                    return
                time.sleep(min(interval, 0.5))
        payload = message
        if suppressed:
            payload = _with_suffix(
                message, f"\n(+{suppressed} similar alerts suppressed)"
            )
        self._send_with_retry(runtime, payload)

    def _send_with_retry(self, runtime: ChannelRuntime, message: AlertMessage):
        policy = runtime.policy
        delay = policy.backoff_initial
        for attempt in range(policy.max_retries + 1):
            try:
                runtime.backend.send(message)
                runtime.sent += 1
                return
            except SendError as e:
                last_error: Exception = e
                retryable = e.retryable
                wait = (
                    e.retry_after
                    if (policy.respect_retry_after and e.retry_after)
                    else None
                )
            except Exception as e:  # unexpected backend failure
                last_error, retryable, wait = e, True, None
            if attempt >= policy.max_retries or not retryable:
                break
            sleep = wait if wait is not None else delay * (1 + random.random() * 0.25)
            time.sleep(min(sleep, policy.backoff_max))
            delay = min(delay * policy.backoff_factor, policy.backoff_max)
        runtime.failed += 1
        text = f"Failed to send alert via {runtime.config.name!r}: {last_error}"
        if policy.fail_silently:
            logger.warning(text)
        else:
            raise RuntimeError(text) from last_error

    # ------------------------------------------------------------------ lifecycle

    def flush(self, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(rt.pending == 0 for rt in self.channels.values()):
                return
            time.sleep(0.02)

    def close(self, timeout: float = 5.0):
        self.flush(timeout)
        self._stopping.set()
        for runtime in self.channels.values():
            worker = runtime.worker
            if worker is not None and worker.is_alive():
                worker.join(timeout=timeout)
            runtime.worker = None
        with self._lock:
            if self._atexit_registered:
                atexit.unregister(self.close)
                self._atexit_registered = False

    def stats(self) -> dict:
        return {
            name: {
                "type": rt.config.type,
                "sent": rt.sent,
                "failed": rt.failed,
                "suppressed": rt.suppressed,
                "dropped": rt.dropped,
                "pending": rt.pending,
                "alive": bool(rt.worker and rt.worker.is_alive()),
            }
            for name, rt in self.channels.items()
        }


def _with_suffix(message: AlertMessage, suffix: str) -> AlertMessage:
    clone = AlertMessage(**{**message.to_dict(), "level": message.level})
    clone.text = message.text + suffix
    return clone
