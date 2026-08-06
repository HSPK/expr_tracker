"""Channel payload construction and HTTP error mapping."""

import json
import urllib.error
import urllib.request

import pytest

from expr_tracker.alerts import backends as backend_module
from expr_tracker.alerts.backends import create_backend
from expr_tracker.alerts.backends.base import SendError, post_json
from expr_tracker.alerts.models import AlertLevel, AlertMessage, ChannelConfig


@pytest.fixture
def captured(monkeypatch):
    sink: list = []

    def fake_post(url, payload, timeout, headers=None):
        sink.append(
            {"url": url, "payload": payload, "timeout": timeout, "headers": headers}
        )
        return "ok"

    monkeypatch.setattr(backend_module, "post_json", fake_post)
    return sink


def build(kind, **options):
    return create_backend(ChannelConfig(type=kind, url="http://hook", options=options))


def message(level="warning", **kwargs):
    return AlertMessage(title="Title", text="Body", level=level, **kwargs)


def test_slack_payload(captured):
    build("slack", channel="#ops", username="et").send(message(mentions=["U1"]))
    payload = captured[0]["payload"]
    assert payload["channel"] == "#ops" and payload["username"] == "et"
    assert "*Title*" in payload["text"] and "<@U1>" in payload["text"]


def test_dingtalk_payload(captured):
    build("dingtalk").send(message(mentions=["13800000000"]))
    payload = captured[0]["payload"]
    assert payload["msgtype"] == "text"
    assert payload["at"]["atMobiles"] == ["13800000000"]
    assert "Title" in payload["text"]["content"]


def test_wecom_payload(captured):
    build("wecom").send(message(mentions=["alice"]))
    payload = captured[0]["payload"]
    assert payload["text"]["mentioned_list"] == ["alice"]


def test_generic_webhook_default_payload(captured):
    build("webhook").send(message(fields={"step": 3}))
    payload = captured[0]["payload"]
    assert payload["title"] == "Title" and payload["level"] == "warning"
    assert payload["fields"] == {"step": 3}


def test_generic_webhook_template(captured):
    template = {"content": "{level}: {title} - {text}", "nested": ["{title}"]}
    build("webhook", template=template).send(message())
    payload = captured[0]["payload"]
    assert payload["content"] == "warning: Title - Body"
    assert payload["nested"] == ["Title"]


def test_custom_headers_are_forwarded(captured):
    build("webhook", headers={"X-Token": "t"}).send(message())
    assert captured[0]["headers"] == {"X-Token": "t"}


def test_lark_uses_error_card_for_errors(monkeypatch):
    calls: list = []

    class FakeWebhook:
        def post_success_card(self, **kwargs):
            calls.append(("success", kwargs))

        def post_error_card(self, **kwargs):
            calls.append(("error", kwargs))

    class FakeLark:
        def __init__(self, **kwargs):
            self.webhook = FakeWebhook()

    backend = build("lark")
    monkeypatch.setattr(backend, "_lark", lambda: FakeLark())
    backend.send(message(level="info"))
    backend.send(message(level="error", traceback="tb"))
    assert [kind for kind, _ in calls] == ["success", "error"]
    assert calls[1][1]["traceback"] == "tb"
    assert calls[0][1]["title"].endswith("Title")  # level emoji prefix


def test_callable_backend_requires_handler():
    with pytest.raises(ValueError, match="handler"):
        create_backend(ChannelConfig(type="callable"))


@pytest.mark.parametrize(
    "status,retryable", [(429, True), (503, True), (400, False), (404, False)]
)
def test_post_json_maps_http_errors(monkeypatch, status, retryable):
    def raise_http(*args, **kwargs):
        raise urllib.error.HTTPError(
            "http://hook", status, "boom", {"Retry-After": "2"}, None
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_http)
    with pytest.raises(SendError) as info:
        post_json("http://hook", {}, timeout=1)
    assert info.value.retryable is retryable
    assert info.value.retry_after == 2.0


def test_post_json_maps_network_errors(monkeypatch):
    def raise_url(*args, **kwargs):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(urllib.request, "urlopen", raise_url)
    with pytest.raises(SendError, match="Network error"):
        post_json("http://hook", {}, timeout=1)


def test_post_json_does_not_leak_secrets(monkeypatch):
    def raise_url(*args, **kwargs):
        raise urllib.error.URLError("nope")

    monkeypatch.setattr(urllib.request, "urlopen", raise_url)
    url = "http://hook.example.com/services/T000/B000?token=supersecret"
    with pytest.raises(SendError) as info:
        post_json(url, {}, timeout=1)
    assert "supersecret" not in str(info.value)


def test_email_backend_sends(monkeypatch):
    sent: list = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            sent.append({"host": host, "port": port})

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self):
            sent.append("tls")

        def login(self, user, password):
            sent.append(("login", user))

        def send_message(self, mail):
            sent.append(mail)

    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    backend = create_backend(
        ChannelConfig(
            type="email",
            options={
                "host": "smtp.test",
                "port": 25,
                "tls": True,
                "user": "u",
                "password": "p",
                "to": ["a@b.c"],
            },
        )
    )
    backend.send(message(level="error"))
    mail = sent[-1]
    assert mail["Subject"] == "[error] Title"
    assert mail["To"] == "a@b.c"
    assert "tls" in sent


# ---------------------------------------------------------------------- subtitle


def test_the_subtitle_reaches_the_rendered_text():
    """Every channel but Lark renders through render_text, which used to drop it."""
    from expr_tracker.alerts.backends.base import render_text

    message = AlertMessage(
        title="T", text="body", subtitle="run-42", level=AlertLevel.INFO
    )
    rendered = render_text(message)
    assert "run-42" in rendered
    assert rendered.startswith("run-42")  # context first, then the body


def test_a_message_without_a_subtitle_is_unchanged():
    from expr_tracker.alerts.backends.base import render_text

    assert render_text(AlertMessage(title="T", text="body")) == "body"


@pytest.mark.parametrize("backend_type", ["slack", "dingtalk", "wecom"])
def test_text_channels_carry_the_subtitle(captured, backend_type):
    backend = create_backend(
        ChannelConfig(type=backend_type, name="c", url="https://hook.test/x")
    )
    backend.send(
        AlertMessage(title="T", text="body", subtitle="run-42", level=AlertLevel.ERROR)
    )
    assert "run-42" in json.dumps(captured[-1])


# ---------------------------------------------------------------------- html mail


def sent_mail(monkeypatch, message, **options):
    """Send through the real EmailBackend with SMTP stubbed out."""
    import smtplib

    box = {}

    class FakeSMTP:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self):
            box["tls"] = True

        def login(self, user, password):
            box["login"] = (user, password)

        def send_message(self, mail):
            box["mail"] = mail

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    backend = create_backend(
        ChannelConfig(
            type="email",
            name="m",
            options={"host": "smtp.test", "to": "a@b.c", **options},
        )
    )
    backend.send(message)
    return box


def test_email_is_multipart_with_a_text_fallback(monkeypatch):
    box = sent_mail(monkeypatch, AlertMessage(title="T", text="body"))
    mail = box["mail"]
    assert mail.is_multipart()
    assert [p.get_content_type() for p in mail.walk()] == [
        "multipart/alternative",
        "text/plain",
        "text/html",
    ]


def test_the_plain_part_carries_the_title_and_body(monkeypatch):
    message = AlertMessage(
        title="Loss spike", text="diverged", subtitle="run-42", fields={"step": 3}
    )
    box = sent_mail(monkeypatch, message)
    text = box["mail"].get_body(("plain",)).get_content()
    assert "Loss spike" in text and "diverged" in text
    assert "run-42" in text and "step: 3" in text


def test_the_html_part_carries_everything(monkeypatch):
    message = AlertMessage(
        title="Loss spike",
        text="diverged",
        subtitle="run-42",
        level=AlertLevel.ERROR,
        fields={"step": 3, "loss": 8.42},
        link="https://example.test/run",
        traceback="Traceback: demo",
    )
    box = sent_mail(monkeypatch, message)
    body = box["mail"].get_body(("html",)).get_content()
    for fragment in (
        "Loss spike",
        "run-42",
        "diverged",
        "step",
        "8.42",
        "https://example.test/run",
        "Traceback: demo",
        "error",
    ):
        assert fragment in body, fragment


@pytest.mark.parametrize(
    ("level", "color"),
    [
        ("info", "#0288d1"),
        ("warning", "#ed6c02"),
        ("error", "#d32f2f"),
        ("critical", "#b71c1c"),
    ],
)
def test_the_html_header_is_coloured_by_severity(monkeypatch, level, color):
    box = sent_mail(
        monkeypatch, AlertMessage(title="T", text="b", level=AlertLevel.parse(level))
    )
    assert color in box["mail"].get_body(("html",)).get_content()


def test_html_content_is_escaped(monkeypatch):
    """A metric name or message must never inject markup into the mail."""
    message = AlertMessage(
        title="<script>alert(1)</script>",
        text="a & b < c",
        fields={"<k>": "<v>"},
    )
    box = sent_mail(monkeypatch, message)
    body = box["mail"].get_body(("html",)).get_content()
    assert "<script>" not in body
    assert "&lt;script&gt;" in body and "a &amp; b &lt; c" in body
    assert "&lt;k&gt;" in body and "&lt;v&gt;" in body


def test_html_can_be_switched_off(monkeypatch):
    box = sent_mail(monkeypatch, AlertMessage(title="T", text="b"), html=False)
    mail = box["mail"]
    assert not mail.is_multipart()
    assert mail.get_content_type() == "text/plain"


def test_the_smtp_session_uses_the_configured_options(monkeypatch):
    box = sent_mail(
        monkeypatch,
        AlertMessage(title="T", text="b"),
        tls=True,
        user="me@x.y",
        password="secret",
        sender="from@x.y",
    )
    assert box["tls"] is True
    assert box["login"] == ("me@x.y", "secret")
    assert box["mail"]["From"] == "from@x.y"


def test_several_recipients(monkeypatch):
    box = sent_mail(
        monkeypatch, AlertMessage(title="T", text="b"), to=["a@x.y", "b@x.y"]
    )
    assert box["mail"]["To"] == "a@x.y, b@x.y"
