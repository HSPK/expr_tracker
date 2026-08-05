"""Rule state machine: edge/level, for_steps, cooldown, recovery, max_fires, disabling."""

import time

import pytest

from expr_tracker.alerts.dispatch import Dispatcher
from expr_tracker.alerts.engine import AlertEngine
from expr_tracker.alerts.expr import EvalContext
from expr_tracker.alerts.models import AlertConfig, ChannelConfig, WebhookPolicy
from expr_tracker.history import MetricSeries


class Harness:
    """Minimal replay harness driving the engine."""

    def __init__(self, rule, **engine_kwargs):
        self.messages: list = []
        self.series = MetricSeries()
        self.now = 1000.0
        channel = ChannelConfig(
            type="callable",
            name="test",
            options={"handler": self.messages.append},
            policy=WebhookPolicy(
                async_send=False,
                dedup_window=0,
                rate_limit_per_minute=None,
                max_retries=0,
            ),
        )
        self.engine = AlertEngine(
            Dispatcher(AlertConfig(channels=[channel])),
            self._context,
            rules=[rule],
            run_info={"project": "p", "run": "r"},
            watchdog_interval=engine_kwargs.pop("watchdog_interval", 0),
            **engine_kwargs,
        )
        self.step = 0

    def _context(self, record):
        record = record or {}
        return EvalContext(
            self.series,
            step=record.get("_step"),
            now=self.now,
            started_at=1000.0,
            last_commit_time=record.get("_time"),
            record=record,
        )

    def feed(self, **metrics):
        record = {"_step": self.step, "_time": self.now, **metrics}
        self.series.add(self.step, self.now, record)
        self.engine.on_step(record)
        self.step += 1
        self.now += 1.0
        return record

    @property
    def titles(self):
        return [m.title for m in self.messages]


def test_edge_mode_fires_once_per_episode():
    h = Harness("loss > 5 => warn: high loss")
    for value in (1, 10, 11, 12):
        h.feed(loss=value)
    assert len(h.messages) == 1
    h.feed(loss=1)  # recovers
    h.feed(loss=10)  # breaches again, so it fires again
    assert len(h.messages) == 2


def test_level_mode_respects_cooldown():
    h = Harness({"condition": "loss > 5", "mode": "level", "cooldown": 1e6})
    for _ in range(5):
        h.feed(loss=10)
    assert len(h.messages) == 1


def test_level_mode_without_cooldown_fires_every_step():
    h = Harness({"condition": "loss > 5", "mode": "level", "cooldown": None})
    for _ in range(3):
        h.feed(loss=10)
    assert len(h.messages) == 3


def test_for_steps_suppresses_single_spike():
    h = Harness({"condition": "loss > 5", "for_steps": 3})
    h.feed(loss=10)
    h.feed(loss=1)
    assert h.messages == []
    for _ in range(3):
        h.feed(loss=10)
    assert len(h.messages) == 1


def test_recovery_notification():
    h = Harness({"condition": "loss > 5", "notify_recovery": True})
    h.feed(loss=10)
    h.feed(loss=1)
    assert len(h.messages) == 2
    assert h.messages[1].title.startswith("[recovered]")
    assert h.messages[1].level.value == "info"


def test_max_fires():
    h = Harness({"condition": "loss > 5", "max_fires": 1})
    for value in (10, 1, 10, 1, 10):
        h.feed(loss=value)
    assert len(h.messages) == 1


def test_unknown_never_fires_and_keeps_state():
    h = Harness("diff(loss) > 50 => warn: spike")
    h.feed(loss=1)  # a single point, so diff() is UNKNOWN
    assert h.messages == []
    h.feed(loss=100)
    assert len(h.messages) == 1


def test_rule_only_evaluated_when_its_metrics_change():
    h = Harness("loss > 5 => warn: high")
    h.feed(other=100)
    assert h.messages == []
    h.feed(loss=10)
    assert len(h.messages) == 1


def test_message_template_and_expr_placeholder():
    h = Harness("diff(m1) > 50 or m1 > 5 => warn: m1={m1} step={step} [{expr}]")
    h.feed(m1=1)
    h.feed(m1=70)
    text = h.messages[0].text
    assert "m1=70" in text and "step=1" in text
    assert "diff(m1)=69 > 50" in text


def test_unknown_placeholder_is_left_untouched():
    h = Harness("loss > 5 => warn: {missing} and {loss:.2f}")
    h.feed(loss=10)
    assert h.messages[0].text == "{missing} and 10.00"


def test_broken_rule_is_disabled_after_repeated_errors(monkeypatch):
    h = Harness("loss > 5 => warn: boom")

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("expr_tracker.alerts.engine.evaluate", explode)
    for _ in range(4):
        h.feed(loss=10)
    assert h.engine.list_rules()[0].enabled is False


def test_add_and_remove_rules():
    h = Harness("loss > 5 => warn: a")
    rule = h.engine.add_rule("acc < 0.1 => error: b")
    assert len(h.engine.list_rules()) == 2
    assert h.engine.remove_rule(rule.rule.name) is True
    assert len(h.engine.list_rules()) == 1


def test_invalid_rule_raises_at_registration():
    from expr_tracker.alerts.expr import ExprError

    h = Harness("loss > 5 => warn: a")
    with pytest.raises(ExprError, match="requires an explicit window"):
        h.engine.add_rule("mean(loss) > 1 => warn: needs window")


def test_watchdog_evaluates_time_based_rules():
    h = Harness(
        {"condition": "no_data(1s)", "level": "error"},
        watchdog_interval=0.05,
    )
    h.now = 1010.0  # pretend no data arrived for a while
    h.engine._ensure_watchdog()
    time.sleep(0.25)
    h.engine.close()
    assert len(h.messages) >= 1
