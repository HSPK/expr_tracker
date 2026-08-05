"""Dispatch layer: routing, rate limits, dedup, retries, async queue, level filters."""

import threading
import time

import pytest

from expr_tracker.alerts.backends.base import AlertBackend, SendError, register_backend
from expr_tracker.alerts.dispatch import Deduper, Dispatcher, TokenBucket
from expr_tracker.alerts.models import (
    AlertConfig,
    AlertLevel,
    AlertMessage,
    ChannelConfig,
    WebhookPolicy,
)


class FlakyBackend(AlertBackend):
    """Fails the first ``fail_times`` sends, to exercise retries."""

    type = "flaky"

    def __init__(self, config):
        super().__init__(config)
        self.attempts = 0

    def send(self, message):
        self.attempts += 1
        options = self.config.options
        if self.attempts <= options.get("fail_times", 0):
            raise SendError("boom", retryable=not options.get("fatal", False))
        options.setdefault("received", []).append(message)


register_backend("flaky", FlakyBackend)


def make(policy=None, **channel):
    received: list = []
    config = ChannelConfig(
        type="callable",
        name="test",
        options={"handler": received.append},
        policy=policy
        or WebhookPolicy(async_send=False, dedup_window=0, rate_limit_per_minute=None),
        **channel,
    )
    return Dispatcher(AlertConfig(channels=[config])), received


def message(title="t", level="info", **kwargs):
    return AlertMessage(title=title, text="body", level=level, **kwargs)


# ---------------------------------------------------------------------- pieces


def test_token_bucket_limits_rate():
    bucket = TokenBucket(2)
    assert bucket.acquire() and bucket.acquire()
    assert not bucket.acquire()


def test_token_bucket_disabled():
    bucket = TokenBucket(None)
    assert all(bucket.acquire() for _ in range(100))


def test_deduper_window():
    deduper = Deduper(10.0)
    assert deduper.check("k", now=0.0) == (True, 0)
    assert deduper.check("k", now=1.0) == (False, 1)
    assert deduper.check("k", now=2.0) == (False, 2)
    assert deduper.check("k", now=20.0) == (True, 2)


# ---------------------------------------------------------------------- routing


def test_min_level_filters():
    dispatcher, received = make(min_level=AlertLevel.WARNING)
    dispatcher.send(message(level="info"))
    dispatcher.send(message(level="error"))
    assert [m.level.value for m in received] == ["error"]


def test_explicit_level_allowlist():
    dispatcher, received = make(levels=[AlertLevel.ERROR])
    for level in ("info", "warning", "error", "critical"):
        dispatcher.send(message(level=level))
    assert [m.level.value for m in received] == ["error"]


def test_tag_routing():
    dispatcher, received = make(tags=["oncall"])
    dispatcher.send(message(tags=["daily"]))
    dispatcher.send(message(tags=["oncall"]))
    assert len(received) == 1


def test_disabled_channel():
    dispatcher, received = make(enabled=False)
    dispatcher.send(message())
    assert received == []


def test_channel_selection():
    dispatcher, received = make()
    dispatcher.send(message(), channels=["nope"])
    assert received == []
    dispatcher.send(message(), channels=["test"])
    assert len(received) == 1


def test_duplicate_channel_name_rejected():
    dispatcher, _ = make()
    with pytest.raises(ValueError, match="Duplicate"):
        dispatcher.add_channel(
            ChannelConfig(type="callable", name="test", options={"handler": print})
        )


# ---------------------------------------------------------------------- policy


def test_dedup_suppresses_and_coalesces():
    policy = WebhookPolicy(
        async_send=False, dedup_window=0.2, rate_limit_per_minute=None
    )
    dispatcher, received = make(policy=policy)
    for _ in range(4):
        dispatcher.send(message(title="same"))
    assert len(received) == 1
    assert dispatcher.stats()["test"]["suppressed"] == 3

    time.sleep(0.25)  # once the window closes the next send carries the summary
    dispatcher.send(message(title="same"))
    assert "3 similar alerts suppressed" in received[1].text


def test_dedup_key_separates_sources():
    policy = WebhookPolicy(
        async_send=False, dedup_window=1e6, rate_limit_per_minute=None
    )
    dispatcher, received = make(policy=policy)
    dispatcher.send(message(title="a"))
    dispatcher.send(message(title="b"))
    assert len(received) == 2


def test_rate_limit_drops_extra_messages():
    policy = WebhookPolicy(async_send=False, dedup_window=0, rate_limit_per_minute=2)
    dispatcher, received = make(policy=policy)
    for i in range(5):
        dispatcher.send(message(title=f"t{i}"))
    assert len(received) == 2
    assert dispatcher.stats()["test"]["suppressed"] == 3


def test_retry_until_success():
    config = ChannelConfig(
        type="flaky",
        name="flaky",
        url="http://example.invalid",
        options={"fail_times": 2, "received": []},
        policy=WebhookPolicy(
            async_send=False,
            dedup_window=0,
            rate_limit_per_minute=None,
            max_retries=3,
            backoff_initial=0.001,
            backoff_max=0.002,
        ),
    )
    dispatcher = Dispatcher(AlertConfig(channels=[config]))
    dispatcher.send(message())
    runtime = dispatcher.channels["flaky"]
    assert runtime.backend.attempts == 3
    assert runtime.sent == 1


def test_retry_gives_up_and_fails_silently():
    config = ChannelConfig(
        type="flaky",
        name="flaky",
        url="http://example.invalid",
        options={"fail_times": 99},
        policy=WebhookPolicy(
            async_send=False,
            dedup_window=0,
            rate_limit_per_minute=None,
            max_retries=1,
            backoff_initial=0.001,
        ),
    )
    dispatcher = Dispatcher(AlertConfig(channels=[config]))
    dispatcher.send(message())  # must not raise
    assert dispatcher.channels["flaky"].backend.attempts == 2
    assert dispatcher.stats()["flaky"]["failed"] == 1


def test_non_retryable_error_stops_immediately():
    config = ChannelConfig(
        type="flaky",
        name="flaky",
        url="http://example.invalid",
        options={"fail_times": 99, "fatal": True},
        policy=WebhookPolicy(
            async_send=False, dedup_window=0, rate_limit_per_minute=None, max_retries=5
        ),
    )
    dispatcher = Dispatcher(AlertConfig(channels=[config]))
    dispatcher.send(message())
    assert dispatcher.channels["flaky"].backend.attempts == 1


def test_fail_silently_false_raises():
    config = ChannelConfig(
        type="flaky",
        name="flaky",
        url="http://example.invalid",
        options={"fail_times": 99},
        policy=WebhookPolicy(
            async_send=False,
            dedup_window=0,
            rate_limit_per_minute=None,
            max_retries=0,
            fail_silently=False,
        ),
    )
    dispatcher = Dispatcher(AlertConfig(channels=[config]))
    with pytest.raises(RuntimeError, match=r"Failed to send alert"):
        dispatcher.send(message())


# ---------------------------------------------------------------------- async


def test_async_delivery_and_flush():
    policy = WebhookPolicy(async_send=True, dedup_window=0, rate_limit_per_minute=None)
    dispatcher, received = make(policy=policy)
    for i in range(10):
        dispatcher.send(message(title=f"t{i}"))
    dispatcher.close(timeout=3.0)
    assert len(received) == 10


def test_queue_full_drops_oldest():
    policy = WebhookPolicy(
        async_send=True, dedup_window=0, rate_limit_per_minute=None, queue_size=2
    )
    dispatcher, _ = make(policy=policy)
    runtime = dispatcher.channels["test"]
    dispatcher._ensure_worker(runtime)
    dispatcher._stopping.set()  # stop the worker so the queue backs up
    time.sleep(0.3)
    for i in range(10):
        dispatcher.send(message(title=f"t{i}"))
    assert runtime.pending <= 2
    assert runtime.dropped > 0


def test_backend_validation_errors_are_raised():
    with pytest.raises(ValueError, match="webhook URL"):
        Dispatcher(AlertConfig(channels=[ChannelConfig(type="slack")]))
    with pytest.raises(ValueError, match=r"options\.host"):
        Dispatcher(
            AlertConfig(channels=[ChannelConfig(type="email", options={"to": "a@b"})])
        )
    with pytest.raises(ValueError, match="Unknown alert backend"):
        Dispatcher(AlertConfig(channels=[ChannelConfig(type="carrier-pigeon")]))


def test_channels_have_independent_lanes():
    """A slow channel must not delay an unrelated one."""
    fast: list = []
    slow: list = []
    started = threading.Event()

    def slow_handler(msg):
        started.set()
        time.sleep(0.6)
        slow.append(msg)

    policy = WebhookPolicy(
        async_send=True, dedup_window=0, rate_limit_per_minute=None, max_retries=0
    )
    config = AlertConfig(
        channels=[
            ChannelConfig(
                type="callable",
                name="slow",
                options={"handler": slow_handler},
                policy=policy,
            ),
            ChannelConfig(
                type="callable",
                name="fast",
                options={"handler": fast.append},
                policy=policy,
            ),
        ]
    )
    dispatcher = Dispatcher(config)
    try:
        dispatcher.send(message(title="a"))
        assert started.wait(1.0)
        deadline = time.monotonic() + 1.0
        while not fast and time.monotonic() < deadline:
            time.sleep(0.01)
        assert [m.title for m in fast] == ["a"]  # delivered while slow is still busy
        assert slow == []
        assert dispatcher.channels["slow"].pending == 0
    finally:
        dispatcher.close(timeout=3.0)
    assert [m.title for m in slow] == ["a"]
