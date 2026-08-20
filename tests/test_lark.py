"""The Lark (Feishu) channel: the card it builds and the webhook it posts to.

Live delivery is opt-in: set ``ET_LARK_TEST_WEBHOOK`` to a bot webhook URL.
Without it the network tests skip and only the local behaviour is checked.
"""

import json
import os

import pytest

from expr_tracker.alerts import AlertConfig, AlertLevel, AlertMessage, Dispatcher
from expr_tracker.alerts import backends as backend_module
from expr_tracker.alerts.backends import LarkBackend
from expr_tracker.alerts.backends.base import SendError, create_backend
from expr_tracker.alerts.backends.cards import build_card, card_payload
from expr_tracker.alerts.models import ChannelConfig, WebhookPolicy

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


class Posted(list):
    """What the backend sent, plus the reply the webhook will give back."""

    reply = '{"code": 0, "msg": "success"}'


@pytest.fixture
def posted(monkeypatch):
    sink = Posted()

    def fake_post(url, payload, timeout, headers=None):
        sink.append({"url": url, "payload": payload, "headers": headers})
        return sink.reply

    monkeypatch.setattr(backend_module, "post_json", fake_post)
    return sink


def send(level="info", **kwargs):
    LarkBackend(channel()).send(message(level=level, **kwargs))


def card_of(posted):
    return posted[0]["payload"]["card"]


# ------------------------------------------------------------------ the card


def test_the_card_is_the_layout_slark_used_to_build():
    """Frozen, so a later refactor cannot quietly reshape what Feishu renders."""
    assert build_card("MSG", "TITLE", "SUB") == {
        "header": {
            "title": {"tag": "plain_text", "content": "TITLE"},
            "subtitle": {"tag": "plain_text", "content": "SUB"},
            "template": "green",
            "ud_icon": {"token": "yes_filled"},
        },
        "elements": [
            {
                "tag": "column_set",
                "flex_mode": "none",
                "horizontal_spacing": "default",
                "horizontal_align": "left",
                "background_style": "default",
                "columns": [
                    {
                        "tag": "column",
                        "background_style": "default",
                        "elements": [
                            {
                                "tag": "div",
                                "text": {
                                    "tag": "plain_text",
                                    "content": "Message",
                                    "text_size": "heading",
                                    "text_align": "left",
                                    "text_color": "default",
                                },
                            }
                        ],
                        "width": "auto",
                        "weight": 1,
                        "vertical_align": "top",
                        "vertical_spacing": "default",
                    }
                ],
            },
            {
                "tag": "markdown",
                "content": "```txt\nMSG\n```",
                "text_align": "left",
                "text_size": "normal",
            },
        ],
    }


def test_a_failed_card_is_red_with_an_error_icon():
    header = build_card("m", "t", "s", failed=True)["header"]
    assert header["template"] == "red"
    assert header["ud_icon"] == {"token": "error_filled"}


def test_a_traceback_becomes_its_own_section():
    card = build_card("m", "t", "s", traceback="boom", failed=True)
    headings = [
        column["elements"][0]["text"]["content"]
        for element in card["elements"]
        if element["tag"] == "column_set"
        for column in element["columns"]
    ]
    assert headings == ["Message", "Traceback"]
    assert card["elements"][-1]["content"] == "```\nboom\n```"


@pytest.mark.parametrize("traceback", [None, "", "   ", "\n\t "])
def test_an_absent_traceback_adds_no_empty_section(traceback):
    """An empty code block on every metric alert would be pure noise."""
    card = build_card("m", "t", "s", traceback=traceback, failed=True)
    assert len(card["elements"]) == 2
    assert "Traceback" not in json.dumps(card)


def test_the_subtitle_defaults_to_now():
    subtitle = build_card("m", "t")["header"]["subtitle"]["content"]
    assert len(subtitle) == 19 and subtitle[4] == "-" and subtitle[13] == ":"


def test_an_explicit_empty_subtitle_is_kept():
    assert build_card("m", "t", "")["header"]["subtitle"]["content"] == ""


def test_the_message_is_fenced_so_lark_renders_it_verbatim():
    card = build_card("a *b* [c](d)", "t", "s")
    assert card["elements"][1]["content"] == "```txt\na *b* [c](d)\n```"


def test_the_envelope_marks_the_body_as_a_card():
    assert card_payload({"x": 1}) == {"msg_type": "interactive", "card": {"x": 1}}


def test_a_card_is_json_serialisable():
    json.dumps(card_payload(build_card("m", "t", "s", traceback="tb", failed=True)))


# ------------------------------------------------------------------ the backend


def test_the_backend_posts_the_card_to_the_webhook(posted):
    send()
    assert posted[0]["url"].endswith("/dummy")
    assert posted[0]["payload"]["msg_type"] == "interactive"


@pytest.mark.parametrize(
    ("level", "template"),
    [("info", "green"), ("warning", "green"), ("error", "red"), ("critical", "red")],
)
def test_the_card_style_follows_the_level(posted, level, template):
    send(level=level)
    assert card_of(posted)["header"]["template"] == template


def test_the_title_carries_a_level_emoji(posted):
    send(level="critical", title="Diverged")
    title = card_of(posted)["header"]["title"]["content"]
    assert title.endswith("Diverged") and title != "Diverged"


def test_the_subtitle_is_forwarded(posted):
    send(subtitle="run-42")
    assert card_of(posted)["header"]["subtitle"]["content"] == "run-42"


def test_fields_and_links_reach_the_card_body(posted):
    send(fields={"step": 7}, link="https://example/run")
    body = card_of(posted)["elements"][1]["content"]
    assert "step: 7" in body and "https://example/run" in body


def test_the_error_card_carries_the_traceback(posted):
    send(level="error", traceback="Traceback: boom")
    assert "Traceback: boom" in json.dumps(posted[0]["payload"])


def test_custom_headers_are_forwarded(posted):
    LarkBackend(channel(options={"headers": {"X-Token": "t"}})).send(message())
    assert posted[0]["headers"] == {"X-Token": "t"}


# ------------------------------------------------------------------ the reply


def test_a_rejection_inside_a_200_is_a_failure(posted):
    posted.reply = '{"code": 19021, "msg": "sign verification failed"}'
    with pytest.raises(SendError, match="code=19021") as excinfo:
        send()
    assert excinfo.value.retryable is False  # a bad signature stays bad


def test_a_throttle_is_retryable(posted):
    posted.reply = '{"code": 11232, "msg": "too many requests"}'
    with pytest.raises(SendError) as excinfo:
        send()
    assert excinfo.value.retryable is True


def test_the_legacy_status_code_field_is_understood(posted):
    posted.reply = '{"StatusCode": 19001, "StatusMessage": "param invalid"}'
    with pytest.raises(SendError, match="code=19001"):
        send()


def test_a_success_reply_raises_nothing(posted):
    for body in ('{"code": 0, "msg": "success"}', '{"StatusCode": 0}'):
        posted.reply = body
        send()


def test_an_unparsable_reply_is_accepted(posted):
    """Some proxies answer 200 with an empty body; the post itself succeeded."""
    for body in ("", "<html>ok</html>", "[]"):
        posted.reply = body
        send()


# ------------------------------------------------------------------ the url


def test_a_transport_failure_is_never_swallowed(posted, monkeypatch):
    """Decoding the reply must not shield a POST that failed outright."""

    def explode(url, payload, timeout, headers=None):
        raise SendError("HTTP 500 from the webhook", retryable=True)

    monkeypatch.setattr(backend_module, "post_json", explode)
    with pytest.raises(SendError, match="HTTP 500"):
        send()


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


def test_the_channel_type_resolves_to_this_backend():
    assert isinstance(create_backend(channel()), LarkBackend)


def test_the_channel_needs_no_third_party_client(monkeypatch, posted):
    """The point of the rewrite: sending must not import slark."""
    monkeypatch.setitem(__import__("sys").modules, "slark", None)
    send(level="error", traceback="tb")
    assert posted[0]["payload"]["msg_type"] == "interactive"


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
        alert=AlertConfig(channels=[channel(url=LIVE_URL)]),
        alert_rules=["loss > 10 => error: loss exploded to {loss}"],
    )
    try:
        et.log({"loss": 42.0})
        assert run.alerts.dispatcher.stats()["lark"]["sent"] == 1
    finally:
        et.finish()
