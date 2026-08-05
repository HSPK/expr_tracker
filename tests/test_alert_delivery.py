"""Unit tests for alert delivery: webhook transport, retries and queue policies."""

import json
import queue
import threading
import time
import urllib.error

import pytest

from expr_tracker.alerts import (
    AlertConfig,
    AlertLevel,
    AlertMessage,
    ChannelConfig,
    Dispatcher,
    WebhookPolicy,
)
from expr_tracker.alerts.backends.base import (
    SendError,
    post_json,
    render_text,
)
from expr_tracker.alerts.backends.base import (
    _redact as redact,
)


def message(title="t", text="x", level="warning", **kwargs):
    return AlertMessage(title=title, text=text, level=AlertLevel(level), **kwargs)


class Recorder:
    """A backend stand-in that can fail a set number of times before succeeding."""

    def __init__(self, failures=0, retryable=True, retry_after=None):
        self.sent: list = []
        self.attempts = 0
        self.failures = failures
        self.retryable = retryable
        self.retry_after = retry_after
        self.closed = False

    def __call__(self, msg):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise SendError(
                "boom", retryable=self.retryable, retry_after=self.retry_after
            )
        self.sent.append(msg)


def channel(handler, name="c", **overrides):
    policy = {
        "async_send": False,
        "dedup_window": 0,
        "rate_limit_per_minute": None,
        "backoff_initial": 0.001,
        "backoff_max": 0.002,
    }
    policy.update(overrides.pop("policy", {}))
    return ChannelConfig(
        type="callable",
        name=name,
        options={"handler": handler},
        policy=WebhookPolicy(**policy),
        **overrides,
    )


def dispatcher_for(handler, **policy):
    return Dispatcher(AlertConfig(channels=[channel(handler, policy=policy)]))


# ------------------------------------------------------------------ transport


class FakeHTTPError(urllib.error.HTTPError):
    def __init__(self, code, reason, headers=None):
        super().__init__("https://host/hook?token=secret", code, reason, headers, None)


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (408, True),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
    ],
)
def test_http_errors_are_classified_by_status(monkeypatch, code, retryable):
    def raise_http(*args, **kwargs):
        raise FakeHTTPError(code, "nope", {})

    monkeypatch.setattr("urllib.request.urlopen", raise_http)
    with pytest.raises(SendError) as excinfo:
        post_json("https://host/hook?token=secret", {"a": 1}, timeout=1)
    assert excinfo.value.retryable is retryable
    assert "secret" not in str(excinfo.value)  # the token never reaches the logs


def test_retry_after_is_taken_from_the_response(monkeypatch):
    def raise_http(*args, **kwargs):
        raise FakeHTTPError(429, "slow down", {"Retry-After": "2.5"})

    monkeypatch.setattr("urllib.request.urlopen", raise_http)
    with pytest.raises(SendError) as excinfo:
        post_json("https://host/hook", {}, timeout=1)
    assert excinfo.value.retry_after == 2.5


@pytest.mark.parametrize("header", [None, "", "later", "Wed, 21 Oct 2015 07:28:00 GMT"])
def test_an_unparsable_retry_after_is_ignored(monkeypatch, header):
    def raise_http(*args, **kwargs):
        raise FakeHTTPError(429, "slow down", {"Retry-After": header})

    monkeypatch.setattr("urllib.request.urlopen", raise_http)
    with pytest.raises(SendError) as excinfo:
        post_json("https://host/hook", {}, timeout=1)
    assert excinfo.value.retry_after is None


def test_network_errors_stay_retryable(monkeypatch):
    """Connectivity blips are transient, unlike a 4xx rejection."""

    def raise_url(*args, **kwargs):
        raise urllib.error.URLError("unreachable")

    monkeypatch.setattr("urllib.request.urlopen", raise_url)
    with pytest.raises(SendError) as excinfo:
        post_json("https://host/hook", {}, timeout=1)
    assert excinfo.value.retryable is True


def test_unexpected_transport_errors_become_send_errors(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", raise_timeout)
    with pytest.raises(SendError):
        post_json("https://host/hook", {}, timeout=1)


def test_a_successful_post_returns_the_body(monkeypatch):
    captured = {}

    class Response:
        def read(self):
            return b'{"ok": true}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["headers"] = request.headers
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    body = post_json(
        "https://host/hook", {"a": "中"}, timeout=3, headers={"X-Token": "v"}
    )
    assert body == '{"ok": true}'
    assert captured["body"] == {"a": "中"}
    assert captured["headers"]["X-token"] == "v"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://host/hook?token=secret", "https://host/…"),
        ("not a url", "<webhook>"),
        ("", "<webhook>"),
    ],
)
def test_redaction_keeps_only_the_host(url, expected):
    assert redact(url) == expected


def test_render_text_includes_fields_link_and_traceback():
    msg = message(
        text="body",
        fields={"step": 3},
        link="https://x",
        traceback="Traceback...",
    )
    rendered = render_text(msg)
    assert rendered.startswith("body")
    assert "step: 3" in rendered
    assert "link: https://x" in rendered
    assert rendered.endswith("Traceback...")
    assert "step: 3" not in render_text(msg, include_fields=False)


# ------------------------------------------------------------------ retries


def test_a_retryable_failure_is_retried(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    handler = Recorder(failures=2)
    dispatcher = dispatcher_for(handler, max_retries=3, backoff_initial=0.001)
    try:
        dispatcher.send(message())
        assert handler.attempts == 3 and len(handler.sent) == 1
        assert dispatcher.stats()["c"]["sent"] == 1
    finally:
        dispatcher.close()


def test_a_non_retryable_failure_is_not_retried():
    handler = Recorder(failures=1, retryable=False)
    dispatcher = dispatcher_for(handler, max_retries=5)
    try:
        dispatcher.send(message())
        assert handler.attempts == 1 and not handler.sent
        assert dispatcher.stats()["c"]["failed"] == 1
    finally:
        dispatcher.close()


def test_retries_are_capped(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    handler = Recorder(failures=99)
    dispatcher = dispatcher_for(handler, max_retries=2, backoff_initial=0.001)
    try:
        dispatcher.send(message())
        assert handler.attempts == 3  # the first try plus two retries
        assert dispatcher.stats()["c"]["failed"] == 1
    finally:
        dispatcher.close()


def test_retry_after_bounds_the_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(time, "sleep", waits.append)
    handler = Recorder(failures=1, retry_after=0.05)
    dispatcher = dispatcher_for(handler, max_retries=2, backoff_initial=10)
    try:
        dispatcher.send(message())
        assert waits and max(waits) <= 10
        assert len(handler.sent) == 1
    finally:
        dispatcher.close()


def test_an_unexpected_backend_error_is_contained():
    def explode(msg):
        raise ValueError("not a SendError")

    dispatcher = dispatcher_for(explode)
    try:
        dispatcher.send(message())
        assert dispatcher.stats()["c"]["failed"] == 1
    finally:
        dispatcher.close()


# ------------------------------------------------------------------ policies


def test_duplicates_are_suppressed_within_the_window():
    handler = Recorder()
    dispatcher = dispatcher_for(handler, dedup_window=60)
    try:
        for _ in range(5):
            dispatcher.send(message(title="same", text="same"))
        assert len(handler.sent) == 1
        assert dispatcher.stats()["c"]["suppressed"] == 4

        dispatcher.send(message(title="other", text="different"))
        assert len(handler.sent) == 2
    finally:
        dispatcher.close()


def test_a_suppressed_count_rides_along_on_the_next_send(monkeypatch):
    handler = Recorder()
    dispatcher = dispatcher_for(handler, dedup_window=0.05)
    try:
        dispatcher.send(message(title="same", text="same"))
        dispatcher.send(message(title="same", text="same"))
        time.sleep(0.08)
        dispatcher.send(message(title="same", text="same"))
        assert len(handler.sent) == 2
        assert "1 similar alerts suppressed" in handler.sent[1].text
    finally:
        dispatcher.close()


def test_rate_limited_messages_are_dropped_by_default():
    handler = Recorder()
    dispatcher = dispatcher_for(
        handler, rate_limit_per_minute=1, on_rate_limited="drop"
    )
    try:
        for i in range(5):
            dispatcher.send(message(text=f"m{i}"))
        assert len(handler.sent) == 1
        assert dispatcher.stats()["c"]["suppressed"] == 4
    finally:
        dispatcher.close()


def test_rate_limit_queue_mode_gives_up_when_stopping():
    handler = Recorder()
    dispatcher = dispatcher_for(
        handler, rate_limit_per_minute=1, on_rate_limited="queue"
    )
    try:
        dispatcher.send(message(text="first"))
        dispatcher._stopping.set()  # a shutdown must not block on the bucket
        started = time.monotonic()
        dispatcher.send(message(text="second"))
        assert time.monotonic() - started < 5
        assert len(handler.sent) == 1
    finally:
        dispatcher._stopping.clear()
        dispatcher.close()


def test_unknown_channel_names_are_ignored():
    handler = Recorder()
    dispatcher = dispatcher_for(handler)
    try:
        dispatcher.send(message(), channels=["nope"])
        assert not handler.sent
        dispatcher.send(message(), channels=["c"])
        assert len(handler.sent) == 1
    finally:
        dispatcher.close()


def test_a_disabled_dispatcher_sends_nothing():
    handler = Recorder()
    dispatcher = Dispatcher(AlertConfig(enabled=False, channels=[channel(handler)]))
    try:
        dispatcher.send(message())
        assert not handler.sent
    finally:
        dispatcher.close()


def test_duplicate_channel_names_are_rejected():
    noop = lambda m: None  # noqa: E731
    with pytest.raises(ValueError, match="Duplicate alert channel"):
        Dispatcher(AlertConfig(channels=[channel(noop), channel(noop)]))


def test_level_filtering_per_channel():
    handler = Recorder()
    dispatcher = Dispatcher(AlertConfig(channels=[channel(handler, min_level="error")]))
    try:
        dispatcher.send(message(level="warning"))
        assert not handler.sent
        dispatcher.send(message(level="critical"))
        assert len(handler.sent) == 1
    finally:
        dispatcher.close()


# ------------------------------------------------------------------ async queue


def test_async_delivery_drains_on_close():
    handler = Recorder()
    dispatcher = dispatcher_for(handler, async_send=True)
    try:
        for i in range(20):
            dispatcher.send(message(text=f"m{i}"))
    finally:
        dispatcher.close()
    assert len(handler.sent) == 20


def test_a_full_queue_drops_the_oldest_by_default():
    release = threading.Event()

    def slow(msg):
        release.wait(timeout=5)

    dispatcher = dispatcher_for(
        slow, async_send=True, queue_size=2, on_queue_full="drop_oldest"
    )
    try:
        for i in range(20):
            dispatcher.send(message(text=f"m{i}"))
        assert dispatcher.stats()["c"]["dropped"] > 0
    finally:
        release.set()
        dispatcher.close()


def test_a_full_queue_can_drop_the_newest():
    release = threading.Event()
    dispatcher = dispatcher_for(
        lambda msg: release.wait(timeout=5),
        async_send=True,
        queue_size=1,
        on_queue_full="drop_new",
    )
    try:
        for i in range(10):
            dispatcher.send(message(text=f"m{i}"))
        assert dispatcher.stats()["c"]["dropped"] > 0
    finally:
        release.set()
        dispatcher.close()


def test_a_worker_survives_a_poisoned_message():
    handler = Recorder(failures=1, retryable=False)
    dispatcher = dispatcher_for(handler, async_send=True)
    try:
        dispatcher.send(message(text="bad"))
        dispatcher.send(message(text="good"))
    finally:
        dispatcher.close()
    assert [m.text for m in handler.sent] == ["good"]


def test_close_is_idempotent():
    handler = Recorder()
    dispatcher = dispatcher_for(handler, async_send=True)
    dispatcher.send(message())
    dispatcher.close()
    dispatcher.close()
    assert len(handler.sent) == 1


def test_sending_after_close_does_not_hang():
    handler = Recorder()
    dispatcher = dispatcher_for(handler, async_send=True)
    dispatcher.close()
    dispatcher.send(message())  # must return promptly, delivered or dropped
    assert isinstance(dispatcher.stats()["c"], dict)


def test_queue_full_blocking_mode_eventually_delivers():
    handler = Recorder()
    dispatcher = dispatcher_for(
        handler, async_send=True, queue_size=1, on_queue_full="block"
    )
    try:
        for i in range(10):
            dispatcher.send(message(text=f"m{i}"))
    finally:
        dispatcher.close()
    assert len(handler.sent) == 10
    assert dispatcher.stats()["c"]["dropped"] == 0


def test_stats_expose_every_counter():
    handler = Recorder()
    dispatcher = dispatcher_for(handler)
    try:
        dispatcher.send(message())
        stats = dispatcher.stats()["c"]
        assert set(stats) >= {"sent", "failed", "suppressed", "dropped"}
        assert stats["sent"] == 1
    finally:
        dispatcher.close()


def test_queue_module_is_not_leaked_between_channels():
    """Two channels must not share a queue or a worker."""
    a, b = Recorder(), Recorder()
    dispatcher = Dispatcher(
        AlertConfig(
            channels=[
                channel(handler, name=name, policy={"async_send": True})
                for name, handler in (("a", a), ("b", b))
            ]
        )
    )
    try:
        runtimes = list(dispatcher.channels.values())
        dispatcher.send(message())
    finally:
        dispatcher.close()
    assert len(a.sent) == len(b.sent) == 1
    assert runtimes[0].queue is not runtimes[1].queue
    assert isinstance(runtimes[0].queue, queue.Queue)
