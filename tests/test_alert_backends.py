"""Channel payload construction and HTTP error mapping."""

import urllib.error
import urllib.request

import pytest

from expr_tracker.alerts import backends as backend_module
from expr_tracker.alerts.backends import create_backend
from expr_tracker.alerts.backends.base import SendError, post_json
from expr_tracker.alerts.models import AlertMessage, ChannelConfig


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
