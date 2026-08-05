"""Alert models and config: level aliases, channel validation, precedence, rendering."""

import json

import pytest

from expr_tracker.alerts import configure_alert, load_config, reset_alert_config
from expr_tracker.alerts.backends import LarkBackend, render_text
from expr_tracker.alerts.models import (
    AlertConfig,
    AlertLevel,
    AlertMessage,
    AlertRule,
    ChannelConfig,
    WebhookPolicy,
    default_channels_from_env,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("warn", AlertLevel.WARNING),
        ("WARNING", AlertLevel.WARNING),
        ("err", AlertLevel.ERROR),
        ("fatal", AlertLevel.CRITICAL),
        ("crit", AlertLevel.CRITICAL),
        (None, AlertLevel.INFO),
        (AlertLevel.DEBUG, AlertLevel.DEBUG),
    ],
)
def test_level_aliases(value, expected):
    assert AlertLevel.parse(value) is expected


def test_unknown_level_raises():
    with pytest.raises(ValueError, match="Unknown alert level"):
        AlertLevel.parse("panic")


def test_level_ordering():
    assert AlertLevel.DEBUG < AlertLevel.INFO < AlertLevel.WARNING
    assert AlertLevel.CRITICAL >= AlertLevel.ERROR


def test_message_dedup_key_is_stable():
    a = AlertMessage(title="t", text="x", source="rule:a")
    b = AlertMessage(title="t", text="y", source="rule:a")
    c = AlertMessage(title="t", text="x", source="rule:b")
    assert a.key() == b.key() != c.key()
    assert AlertMessage(title="t", text="x", dedup_key="fixed").key() == "fixed"


def test_render_text_includes_fields_and_traceback():
    message = AlertMessage(
        title="t", text="body", fields={"step": 3}, link="http://x", traceback="tb"
    )
    rendered = render_text(message)
    assert "body" in rendered and "step: 3" in rendered
    assert "http://x" in rendered and "tb" in rendered


def test_channel_url_from_env(monkeypatch):
    monkeypatch.setenv("MY_HOOK", "http://hook")
    channel = ChannelConfig(type="slack", url_env="MY_HOOK")
    assert channel.resolve_url() == "http://hook"


def test_channel_rejects_unknown_options():
    with pytest.raises(ValueError, match="Unknown channel options"):
        ChannelConfig.from_dict({"type": "lark", "nope": 1})


def test_policy_rejects_unknown_options():
    with pytest.raises(ValueError, match="Unknown webhook policy"):
        WebhookPolicy.from_dict({"timeout": 1, "nope": 2})


def test_rule_accepts_prometheus_style_keys():
    rule = AlertRule.from_dict(
        {"alert": "loss_spike", "expr": "loss > 1", "for": 3, "level": "error"}
    )
    assert rule.name == "loss_spike"
    assert rule.condition == "loss > 1"
    assert rule.for_steps == 3
    assert rule.level is AlertLevel.ERROR


def test_rule_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        AlertRule(condition="a > 1", mode="sometimes")


def test_default_channels_from_env(monkeypatch):
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    monkeypatch.setenv("ET_LARK_WEBHOOK_URL", "http://lark")
    monkeypatch.setenv("ET_SLACK_WEBHOOK_URL", "http://slack")
    channels = default_channels_from_env()
    assert {c.type for c in channels} == {"lark", "slack"}


def test_legacy_webhook_url_still_works(monkeypatch):
    monkeypatch.delenv("ET_LARK_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("WEBHOOK_URL", "http://legacy")
    channels = default_channels_from_env()
    assert channels[0].type == "lark" and channels[0].url == "http://legacy"


def test_load_config_from_file(tmp_path, monkeypatch):
    path = tmp_path / "alert.json"
    path.write_text(
        json.dumps(
            {
                "alert": {
                    "channels": [{"type": "slack", "url": "http://x"}],
                    "rules": ["loss > 1 => error: boom"],
                    "default_policy": {"timeout": 3},
                }
            }
        )
    )
    monkeypatch.setenv("ET_ALERT_CONFIG", str(path))
    reset_alert_config()
    config = load_config()
    assert config.channels[0].type == "slack"
    assert config.rules[0].level is AlertLevel.ERROR
    assert config.default_policy.timeout == 3


def test_configure_alert_takes_priority(monkeypatch):
    monkeypatch.setenv("ET_ALERT_CONFIG", "/does/not/exist.json")
    configure_alert([{"type": "slack", "url": "http://configured"}])
    try:
        assert load_config().channels[0].url == "http://configured"
    finally:
        reset_alert_config()


def test_explicit_config_wins_over_everything():
    config = load_config({"channels": [{"type": "wecom", "url": "http://direct"}]})
    assert config.channels[0].type == "wecom"


def test_alert_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown alert config"):
        AlertConfig.from_dict({"channelz": []})


def test_lark_backend_requires_url():
    with pytest.raises(ValueError, match="webhook URL"):
        LarkBackend(ChannelConfig(type="lark")).validate()
