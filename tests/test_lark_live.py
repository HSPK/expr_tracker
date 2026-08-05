"""Lark (Feishu) channel against the real slark client.

Live delivery is opt-in: set ``ET_LARK_TEST_WEBHOOK`` to a bot webhook URL. Without
it the network tests skip and only the local behaviour is checked.
"""

import os

import pytest

from expr_tracker.alerts import AlertConfig, AlertLevel, AlertMessage, Dispatcher
from expr_tracker.alerts.backends import LarkBackend
from expr_tracker.alerts.backends.base import SendError, create_backend
from expr_tracker.alerts.models import ChannelConfig, WebhookPolicy

slark = pytest.importorskip("slark")

LIVE_URL = os.getenv("ET_LARK_TEST_WEBHOOK")
live_only = pytest.mark.skipif(
    not LIVE_URL, reason="set ET_LARK_TEST_WEBHOOK to a Feishu bot webhook"
)


def message(level="info", **kwargs):
    return AlertMessage(
        title=kwargs.pop("title", "expr_tracker test"),
        text=kwargs.pop("text", "body"),
        level=AlertLevel.parse(level),
        **kwargs,
    )


def channel(**overrides):
    overrides.setdefault("url", "https://open.feishu.cn/open-apis/bot/v2/hook/dummy")
    return ChannelConfig(
        type="lark",
        name="lark",
        policy=WebhookPolicy(
            async_send=False, dedup_window=0, rate_limit_per_minute=None
        ),
        **overrides,
    )


# ------------------------------------------------------------------ client


def test_the_backend_builds_a_real_slark_client():
    backend = create_backend(channel())
    assert isinstance(backend, LarkBackend)
    client = backend._lark()
    assert isinstance(client, slark.Lark)


def test_the_client_is_built_once_and_reused():
    backend = LarkBackend(channel())
    assert backend._client is None
    first = backend._lark()
    assert backend._lark() is first  # no client churn per alert


def test_options_are_forwarded_to_slark_except_headers():
    backend = LarkBackend(channel(options={"headers": {"X": "1"}, "timeout": 5}))
    client = backend._lark()  # headers would be an unexpected kwarg
    assert isinstance(client, slark.Lark)


def test_the_url_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ET_LARK_URL_FOR_TEST", "https://open.feishu.cn/hook/from-env")
    backend = LarkBackend(channel(url=None, url_env="ET_LARK_URL_FOR_TEST"))
    assert backend.url == "https://open.feishu.cn/hook/from-env"


def test_a_missing_url_is_reported_before_any_network_call():
    backend = LarkBackend(channel(url=None))
    with pytest.raises(SendError, match="no webhook URL") as excinfo:
        backend.send(message())
    assert excinfo.value.retryable is False


def test_an_unset_url_env_is_treated_as_missing(monkeypatch):
    monkeypatch.delenv("ET_LARK_URL_FOR_TEST", raising=False)
    backend = LarkBackend(channel(url=None, url_env="ET_LARK_URL_FOR_TEST"))
    with pytest.raises(SendError, match="no webhook URL"):
        backend.send(message())


# ------------------------------------------------------------------ card choice


class SpyWebhook:
    def __init__(self):
        self.calls = []

    def post_success_card(self, **kwargs):
        self.calls.append(("success", kwargs))

    def post_error_card(self, **kwargs):
        self.calls.append(("error", kwargs))


@pytest.fixture
def spy(monkeypatch):
    webhook = SpyWebhook()
    backend = LarkBackend(channel())
    backend._client = type("Client", (), {"webhook": webhook})()
    return backend, webhook


@pytest.mark.parametrize(
    ("level", "card"),
    [
        ("info", "success"),
        ("warning", "success"),
        ("error", "error"),
        ("critical", "error"),
    ],
)
def test_the_card_style_follows_the_level(spy, level, card):
    backend, webhook = spy
    backend.send(message(level=level))
    assert webhook.calls[0][0] == card


def test_an_error_card_carries_the_traceback(spy):
    backend, webhook = spy
    backend.send(message(level="error", traceback="Traceback: boom"))
    assert webhook.calls[0][1]["traceback"] == "Traceback: boom"


def test_an_error_card_without_a_traceback_sends_an_empty_string(spy):
    backend, webhook = spy
    backend.send(message(level="error"))
    assert webhook.calls[0][1]["traceback"] == ""


def test_the_title_carries_a_level_emoji(spy):
    backend, webhook = spy
    backend.send(message(level="critical", title="Diverged"))
    title = webhook.calls[0][1]["title"]
    assert title.endswith("Diverged") and title != "Diverged"


def test_the_subtitle_is_forwarded(spy):
    backend, webhook = spy
    backend.send(message(subtitle="run-42"))
    assert webhook.calls[0][1]["subtitle"] == "run-42"


def test_fields_and_links_reach_the_card_body(spy):
    backend, webhook = spy
    backend.send(message(fields={"step": 7}, link="https://example/run"))
    body = webhook.calls[0][1]["msg"]
    assert "step: 7" in body and "https://example/run" in body


def test_a_slark_failure_becomes_a_retryable_send_error(spy):
    backend, webhook = spy

    def explode(**kwargs):
        raise RuntimeError("feishu is down")

    webhook.post_success_card = explode
    with pytest.raises(SendError, match="Lark webhook failed") as excinfo:
        backend.send(message())
    assert excinfo.value.retryable is True


# ------------------------------------------------------------------ live


@live_only
def test_a_success_card_is_delivered():
    dispatcher = Dispatcher(AlertConfig(channels=[channel(url=LIVE_URL)]))
    try:
        dispatcher.send(
            message(
                title="expr_tracker · success card",
                text="pytest live check",
                fields={"suite": "test_lark", "level": "info"},
            )
        )
        stats = dispatcher.stats()["lark"]
        assert stats["sent"] == 1 and stats["failed"] == 0
    finally:
        dispatcher.close()


@live_only
def test_an_error_card_is_delivered():
    dispatcher = Dispatcher(AlertConfig(channels=[channel(url=LIVE_URL)]))
    try:
        dispatcher.send(
            message(
                level="error",
                title="expr_tracker · error card",
                text="pytest live check",
                traceback="Traceback (most recent call last):\n  demo only",
            )
        )
        assert dispatcher.stats()["lark"]["sent"] == 1
    finally:
        dispatcher.close()


@live_only
def test_a_bad_webhook_path_is_reported_as_a_failure():
    broken = LIVE_URL.rsplit("/", 1)[0] + "/00000000-0000-0000-0000-000000000000"
    dispatcher = Dispatcher(
        AlertConfig(
            channels=[
                ChannelConfig(
                    type="lark",
                    name="lark",
                    url=broken,
                    policy=WebhookPolicy(
                        async_send=False,
                        dedup_window=0,
                        rate_limit_per_minute=None,
                        max_retries=0,
                    ),
                )
            ]
        )
    )
    try:
        dispatcher.send(message(title="should not arrive"))
        stats = dispatcher.stats()["lark"]
        assert stats["sent"] == 0 and stats["failed"] == 1
    finally:
        dispatcher.close()


@live_only
def test_a_rule_alert_is_delivered_from_a_run(tmp_path):
    """The whole path: log a metric, a rule fires, Feishu receives the card."""
    import expr_tracker as et

    run = et.init(
        project="expr_tracker",
        name="lark-live",
        dir=str(tmp_path),
        backends=[],
        max_open_seconds=None,
        alert={
            "channels": [
                {
                    "type": "lark",
                    "name": "lark",
                    "url": LIVE_URL,
                    "policy": {
                        "async_send": False,
                        "dedup_window": 0,
                        "rate_limit_per_minute": None,
                    },
                }
            ]
        },
        alert_rules=[
            "loss > 10 => error: expr_tracker live rule, loss={loss:.2f} @ step {step}"
        ],
    )
    try:
        for step in range(5):
            run.log({"loss": float(step)})
        run.log({"loss": 99.0})
        assert run.info()["alerts"]["channels"]["lark"]["sent"] == 1
    finally:
        et.finish()
