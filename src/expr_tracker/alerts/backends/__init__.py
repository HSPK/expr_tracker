"""Built-in channels: lark, slack, dingtalk, wecom, generic webhook and email."""

from __future__ import annotations

import json
import smtplib
from email.message import EmailMessage

from ..models import AlertLevel, AlertMessage, ChannelConfig
from .base import (
    AlertBackend,
    SendError,
    create_backend,
    post_json,
    register_backend,
    render_html,
    render_text,
)

LEVEL_EMOJI = {
    AlertLevel.DEBUG: "🔍",
    AlertLevel.INFO: "ℹ️",
    AlertLevel.WARNING: "⚠️",
    AlertLevel.ERROR: "❌",
    AlertLevel.CRITICAL: "🚨",
}


class UrlBackend(AlertBackend):
    """Base for channels that need a webhook URL."""

    def validate(self):
        if not self.config.resolve_url():
            raise ValueError(
                f"Channel {self.config.name!r} ({self.config.type}) has no webhook "
                "URL; "
                "set 'url' or 'url_env'."
            )

    @property
    def url(self) -> str:
        url = self.config.resolve_url()
        if not url:
            raise SendError(
                f"Channel {self.config.name!r} has no webhook URL", retryable=False
            )
        return url

    @property
    def timeout(self) -> float:
        policy = self.config.policy
        return policy.timeout if policy else 10.0

    def post(self, payload: dict) -> str:
        return post_json(
            self.url, payload, self.timeout, self.config.options.get("headers")
        )

    def post_checked(self, payload: dict):
        """DingTalk and WeCom report failures via errcode inside an HTTP 200 body."""
        body = self.post(payload)
        try:
            result = json.loads(body)
        except Exception:
            return
        code = result.get("errcode")
        if code:
            raise SendError(
                f"{self.config.type} rejected the message: "
                f"errcode={code} errmsg={result.get('errmsg')!r}",
                retryable=code in (-1, 45009, 130101),
            )


class LarkBackend(UrlBackend):
    type = "lark"

    def __init__(self, config: ChannelConfig):
        super().__init__(config)
        self._client = None

    def _lark(self):
        if self._client is None:
            from slark import Lark

            options = {k: v for k, v in self.config.options.items() if k != "headers"}
            self._client = Lark(webhook=self.url, **options)
        return self._client

    def send(self, message: AlertMessage):
        title = f"{LEVEL_EMOJI.get(message.level, '')} {message.title}".strip()
        text = render_text(message)
        subtitle = message.subtitle
        try:
            client = self._lark()
        except ImportError as e:  # pragma: no cover - optional extra
            raise SendError(
                f'The lark channel needs slark: pip install "expr_tracker[lark]" ({e})',
                retryable=False,
            ) from e
        try:
            if message.level >= AlertLevel.ERROR:
                client.webhook.post_error_card(
                    msg=text,
                    traceback=message.traceback or "",
                    title=title,
                    subtitle=subtitle,
                )
            else:
                client.webhook.post_success_card(
                    msg=text, title=title, subtitle=subtitle
                )
        except Exception as e:
            raise SendError(f"Lark webhook failed: {e}") from e


class SlackBackend(UrlBackend):
    type = "slack"

    def send(self, message: AlertMessage):
        mentions = " ".join(f"<@{m}>" for m in message.mentions)
        header = f"{LEVEL_EMOJI.get(message.level, '')} *{message.title}*"
        body = "\n".join(
            part for part in (header, mentions, render_text(message)) if part
        )
        payload = {"text": body}
        if channel := self.config.options.get("channel"):
            payload["channel"] = channel
        if username := self.config.options.get("username"):
            payload["username"] = username
        self.post(payload)


class DingTalkBackend(UrlBackend):
    type = "dingtalk"

    def send(self, message: AlertMessage):
        content = _titled_text(message)
        payload = {"msgtype": "text", "text": {"content": content}}
        if message.mentions:
            payload["at"] = {"atMobiles": message.mentions}
        self.post_checked(payload)


class WeComBackend(UrlBackend):
    type = "wecom"

    def send(self, message: AlertMessage):
        content = _titled_text(message)
        payload = {"msgtype": "text", "text": {"content": content}}
        if message.mentions:
            payload["text"]["mentioned_list"] = message.mentions
        self.post_checked(payload)


class WebhookBackend(UrlBackend):
    """Generic JSON webhook; ``options.template`` shapes the body via ``{field}``."""

    type = "webhook"

    def send(self, message: AlertMessage):
        template = self.config.options.get("template")
        payload = (
            _render_template(template, message)
            if template is not None
            else message.to_dict()
        )
        self.post(payload)


class EmailBackend(AlertBackend):
    """SMTP email channel. ``options``: host, port, tls, user, password, sender, to."""

    type = "email"

    def validate(self):
        options = self.config.options
        if not options.get("host"):
            raise ValueError(
                f"Email channel {self.config.name!r} requires options.host"
            )
        if not options.get("to"):
            raise ValueError(f"Email channel {self.config.name!r} requires options.to")

    def send(self, message: AlertMessage):
        options = self.config.options
        recipients = options["to"]
        recipients = [recipients] if isinstance(recipients, str) else list(recipients)
        mail = EmailMessage()
        mail["Subject"] = f"[{message.level.value}] {message.title}"
        mail["From"] = options.get("sender") or options.get("user") or "expr-tracker"
        mail["To"] = ", ".join(recipients)
        # multipart/alternative: rich clients show the card, the rest see the text
        mail.set_content(f"{message.title}\n\n{render_text(message)}")
        if options.get("html", True):
            mail.add_alternative(render_html(message), subtype="html")
        policy_timeout = self.config.policy.timeout if self.config.policy else 10.0
        try:
            factory = smtplib.SMTP_SSL if options.get("ssl") else smtplib.SMTP
            with factory(
                options["host"],
                int(options.get("port", 465 if options.get("ssl") else 25)),
                timeout=policy_timeout,
            ) as smtp:
                if options.get("tls"):
                    smtp.starttls()
                if options.get("user"):
                    smtp.login(options["user"], options.get("password", ""))
                smtp.send_message(mail)
        except Exception as e:
            raise SendError(f"SMTP send failed: {e}") from e


class CallableBackend(AlertBackend):
    """Hand the message to ``options.handler``; useful for tests and custom sinks."""

    type = "callable"

    def validate(self):
        if not callable(self.config.options.get("handler")):
            raise ValueError("callable channel requires options.handler to be callable")

    def send(self, message: AlertMessage):
        self.config.options["handler"](message)


def _titled_text(message: AlertMessage) -> str:
    """Emoji + title + body, for channels that only take plain text."""
    emoji = LEVEL_EMOJI.get(message.level, "")
    return f"{emoji} {message.title}\n{render_text(message)}"


def _render_template(template, message: AlertMessage):
    data = message.to_dict()
    if isinstance(template, dict):
        return {
            key: _render_template(value, message) for key, value in template.items()
        }
    if isinstance(template, list):
        return [_render_template(item, message) for item in template]
    if isinstance(template, str):
        try:
            return template.format(**data)
        except Exception:
            return template
    return template


for _cls in (
    LarkBackend,
    SlackBackend,
    DingTalkBackend,
    WeComBackend,
    WebhookBackend,
    EmailBackend,
    CallableBackend,
):
    register_backend(_cls.type, _cls)


__all__ = [
    "AlertBackend",
    "CallableBackend",
    "DingTalkBackend",
    "EmailBackend",
    "LarkBackend",
    "SendError",
    "SlackBackend",
    "WeComBackend",
    "WebhookBackend",
    "create_backend",
    "register_backend",
    "render_text",
]
