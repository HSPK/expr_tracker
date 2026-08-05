"""Alert delivery decisions: the watchdog thread, routing filters and extension points.

The watchdog is what notices a *hung* run -- no log() call ever arrives, so nothing
but the background thread can fire the rule.
"""

import threading
import time

import pytest

import expr_tracker as et
from expr_tracker.alerts import AlertConfig, AlertLevel, AlertMessage, Dispatcher
from expr_tracker.alerts.backends import CallableBackend
from expr_tracker.alerts.backends.base import (
    _REGISTRY as BACKENDS,
)
from expr_tracker.alerts.backends.base import (
    AlertBackend,
    SendError,
    create_backend,
    register_backend,
)
from expr_tracker.alerts.models import ChannelConfig, WebhookPolicy


def channel(handler, name="c", **overrides):
    policy = {
        "async_send": False,
        "dedup_window": 0,
        "rate_limit_per_minute": None,
        **overrides.pop("policy", {}),
    }
    return ChannelConfig(
        type="callable",
        name=name,
        options={"handler": handler},
        policy=WebhookPolicy(**policy),
        **overrides,
    )


def message(level="warning", tags=(), **kwargs):
    return AlertMessage(
        title=kwargs.pop("title", "t"),
        text=kwargs.pop("text", "x"),
        level=AlertLevel.parse(level),
        tags=list(tags),
        **kwargs,
    )


def wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ====================================================================== watchdog


@pytest.fixture
def watched(tmp_path):
    """A run whose time-based rules are polled quickly."""
    created = []

    def factory(rules, interval=0.05, **options):
        received: list = []
        alert = {
            "watchdog_interval": interval,
            "channels": [
                {
                    "type": "callable",
                    "name": "c",
                    "options": {"handler": received.append},
                    "policy": {
                        "async_send": False,
                        "dedup_window": 0,
                        "rate_limit_per_minute": None,
                    },
                }
            ],
        }
        run = et.init(
            project="wd",
            name=options.pop("name", "r"),
            dir=str(tmp_path),
            backends=[],
            alert=alert,
            alert_rules=rules,
            max_open_seconds=None,
            **options,
        )
        created.append(run)
        return run, received

    yield factory
    if et.get_run() is not None:
        et.finish()


def test_the_watchdog_fires_without_any_log_call(watched):
    """A hung run logs nothing, so only the background thread can notice."""
    run, received = watched(["no_data(0.2s) => error: training is hung"])
    assert wait_for(lambda: received)
    assert received[0].level.value == "error"
    assert received[0].text == "training is hung"
    assert run.history_query(-1) == []  # nothing was ever logged


def test_the_watchdog_stays_quiet_while_data_arrives(watched):
    run, received = watched(["no_data(2s) => error: hung"])
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        run.log({"loss": 1.0})
        time.sleep(0.05)
    assert received == []


def test_the_watchdog_fires_once_logging_stops(watched):
    run, received = watched(["no_data(0.3s) => error: hung"])
    for _ in range(5):
        run.log({"loss": 1.0})
        time.sleep(0.02)
    assert received == []
    assert wait_for(lambda: received)  # the flow stopped, the watchdog notices


def test_elapsed_rules_are_watchdog_driven(watched):
    _, received = watched(["elapsed() > 0.2 => warning: run is long"])
    assert wait_for(lambda: received)
    assert received[0].text == "run is long"


def test_age_rules_need_a_metric_and_stay_unknown_without_one(watched):
    run, received = watched(["age(loss) > 0.2 => warning: loss is stale"])
    time.sleep(0.4)
    assert received == []  # the metric was never seen, so the rule is UNKNOWN
    run.log({"loss": 1.0})
    assert wait_for(lambda: received)


def test_no_watchdog_thread_without_time_based_rules(watched):
    watched(["loss > 1 => warning: high"])
    time.sleep(0.2)
    names = [t.name for t in threading.enumerate()]
    assert not any(n.startswith("et-alert-watchdog") for n in names)


def test_the_watchdog_starts_when_a_time_rule_is_added_later(watched):
    _, received = watched(["loss > 1000 => warning: never"])
    assert not any(
        t.name.startswith("et-alert-watchdog") for t in threading.enumerate()
    )
    et.add_alert_rule("no_data(0.2s) => error: hung")
    assert wait_for(lambda: received)


def test_finish_stops_the_watchdog(watched):
    watched(["no_data(5s) => error: hung"])
    assert wait_for(
        lambda: any(
            t.name.startswith("et-alert-watchdog") for t in threading.enumerate()
        )
    )
    et.finish()
    assert wait_for(
        lambda: (
            not any(
                t.name.startswith("et-alert-watchdog") for t in threading.enumerate()
            )
        )
    )


def test_the_watchdog_interval_is_configurable(tmp_path):
    from expr_tracker.alerts import build_engine

    run = et.init(
        project="wd",
        name="interval",
        dir=str(tmp_path),
        backends=[],
        alert={"watchdog_interval": 7.5},
    )
    try:
        assert run.alerts.watchdog_interval == 7.5
    finally:
        et.finish()
    assert AlertConfig().watchdog_interval == 30.0  # the shipped default
    assert build_engine is not None


def test_a_disabled_rule_is_not_polled(watched):
    _, received = watched(["no_data(0.2s) => error: hung"])
    for rule in et.list_alert_rules():
        rule.enabled = False
    time.sleep(0.4)
    assert received == []


def test_removing_a_rule_stops_its_alerts(watched):
    _, received = watched(["no_data(0.2s) => error: hung"])
    assert wait_for(lambda: received)
    for rule in et.list_alert_rules():
        et.remove_alert_rule(rule.name)
    received.clear()
    time.sleep(0.4)
    assert received == []


# ====================================================================== routing


def dispatcher_with(*channels):
    return Dispatcher(AlertConfig(channels=list(channels)))


def test_min_level_filters_below_the_threshold():
    got: list = []
    dispatcher = dispatcher_with(channel(got.append, min_level="error"))
    try:
        for level in ("info", "warning", "error", "critical"):
            dispatcher.send(message(level=level))
        assert [m.level.value for m in got] == ["error", "critical"]
    finally:
        dispatcher.close()


def test_a_levels_allowlist_is_an_exact_set():
    got: list = []
    dispatcher = dispatcher_with(channel(got.append, levels=["warning", "critical"]))
    try:
        for level in ("info", "warning", "error", "critical"):
            dispatcher.send(message(level=level))
        assert [m.level.value for m in got] == ["warning", "critical"]
    finally:
        dispatcher.close()


def test_a_levels_allowlist_overrides_min_level():
    got: list = []
    dispatcher = dispatcher_with(
        channel(got.append, min_level="critical", levels=["info"])
    )
    try:
        dispatcher.send(message(level="info"))
        dispatcher.send(message(level="critical"))
        assert [m.level.value for m in got] == ["info"]
    finally:
        dispatcher.close()


def test_a_disabled_channel_receives_nothing():
    got: list = []
    dispatcher = dispatcher_with(channel(got.append, enabled=False))
    try:
        dispatcher.send(message(level="critical"))
        assert got == []
    finally:
        dispatcher.close()


def test_channel_tags_must_intersect_the_message_tags():
    got: list = []
    dispatcher = dispatcher_with(channel(got.append, tags=["gpu", "infra"]))
    try:
        dispatcher.send(message(tags=["gpu"]))
        dispatcher.send(message(tags=["infra", "other"]))
        dispatcher.send(message(tags=["unrelated"]))
        dispatcher.send(message())  # no tags at all
        assert len(got) == 2
    finally:
        dispatcher.close()


def test_a_channel_without_tags_accepts_everything():
    got: list = []
    dispatcher = dispatcher_with(channel(got.append))
    try:
        dispatcher.send(message(tags=["anything"]))
        dispatcher.send(message())
        assert len(got) == 2
    finally:
        dispatcher.close()


def test_each_channel_filters_independently():
    errors: list = []
    everything: list = []
    gpu: list = []
    dispatcher = dispatcher_with(
        channel(errors.append, name="errors", min_level="error"),
        channel(everything.append, name="all"),
        channel(gpu.append, name="gpu", tags=["gpu"]),
    )
    try:
        dispatcher.send(message(level="info"))
        dispatcher.send(message(level="error", tags=["gpu"]))
        assert len(errors) == 1 and len(everything) == 2 and len(gpu) == 1
    finally:
        dispatcher.close()


def test_a_rule_can_target_specific_channels(tmp_path):
    primary: list = []
    secondary: list = []
    run = et.init(
        project="wd",
        name="targeted",
        dir=str(tmp_path),
        backends=[],
        alert={
            "channels": [
                {
                    "type": "callable",
                    "name": name,
                    "options": {"handler": sink.append},
                    "policy": {"async_send": False, "dedup_window": 0},
                }
                for name, sink in (("primary", primary), ("secondary", secondary))
            ]
        },
        alert_rules=[
            {
                "condition": "loss > 1",
                "channels": ["primary"],
                "message": "only primary",
            }
        ],
    )
    try:
        run.log({"loss": 5.0})
        assert len(primary) == 1 and secondary == []
    finally:
        et.finish()


def test_a_rule_without_channels_reaches_all_of_them(tmp_path):
    a: list = []
    b: list = []
    run = et.init(
        project="wd",
        name="broadcast",
        dir=str(tmp_path),
        backends=[],
        alert={
            "channels": [
                {
                    "type": "callable",
                    "name": name,
                    "options": {"handler": sink.append},
                    "policy": {"async_send": False, "dedup_window": 0},
                }
                for name, sink in (("a", a), ("b", b))
            ]
        },
        alert_rules=["loss > 1 => warning: everyone"],
    )
    try:
        run.log({"loss": 5.0})
        assert len(a) == 1 and len(b) == 1
    finally:
        et.finish()


def test_rule_tags_reach_the_channel_filter(tmp_path):
    gpu: list = []
    run = et.init(
        project="wd",
        name="tagged",
        dir=str(tmp_path),
        backends=[],
        alert={
            "channels": [
                {
                    "type": "callable",
                    "name": "gpu",
                    "options": {"handler": gpu.append},
                    "tags": ["gpu"],
                    "policy": {"async_send": False, "dedup_window": 0},
                }
            ]
        },
        alert_rules=[
            {"condition": "loss > 1", "tags": ["gpu"], "message": "gpu problem"},
            {"condition": "loss > 2", "tags": ["cpu"], "message": "cpu problem"},
        ],
    )
    try:
        run.log({"loss": 5.0})
        assert [m.text for m in gpu] == ["gpu problem"]
    finally:
        et.finish()


# ====================================================================== backends


def test_a_custom_backend_can_be_registered():
    sent: list = []

    class Recorder(AlertBackend):
        type = "recorder"

        def send(self, msg):
            sent.append(msg)

    register_backend("recorder", Recorder)
    try:
        assert BACKENDS["recorder"] is Recorder
        backend = create_backend(ChannelConfig(type="recorder", name="r"))
        assert isinstance(backend, Recorder)
        backend.send(message())
        assert len(sent) == 1
    finally:
        BACKENDS.pop("recorder", None)


def test_a_custom_backend_works_end_to_end(tmp_path):
    sent: list = []

    class Collector(AlertBackend):
        type = "collector"

        def send(self, msg):
            sent.append(msg)

    register_backend("collector", Collector)
    try:
        run = et.init(
            project="wd",
            name="custom",
            dir=str(tmp_path),
            backends=[],
            alert={
                "channels": [
                    {
                        "type": "collector",
                        "name": "col",
                        "policy": {"async_send": False, "dedup_window": 0},
                    }
                ]
            },
            alert_rules=["loss > 1 => error: custom sink"],
        )
        try:
            run.log({"loss": 5.0})
            assert [m.text for m in sent] == ["custom sink"]
        finally:
            et.finish()
    finally:
        BACKENDS.pop("collector", None)


def test_registering_over_a_builtin_replaces_it():
    original = BACKENDS["callable"]
    sent: list = []

    class Replacement(AlertBackend):
        type = "callable"

        def send(self, msg):
            sent.append(msg)

    register_backend("callable", Replacement)
    try:
        backend = create_backend(ChannelConfig(type="callable", name="c"))
        assert isinstance(backend, Replacement)
    finally:
        register_backend("callable", original)
    assert BACKENDS["callable"] is original


def test_validate_runs_when_a_backend_is_created():
    class Strict(AlertBackend):
        type = "strict"

        def validate(self):
            raise ValueError("needs options.token")

        def send(self, msg):
            pass

    register_backend("strict", Strict)
    try:
        with pytest.raises(ValueError, match=r"needs options\.token"):
            create_backend(ChannelConfig(type="strict", name="s"))
    finally:
        BACKENDS.pop("strict", None)


def test_an_unknown_channel_type_is_rejected():
    with pytest.raises((ValueError, KeyError), match=r"Unknown alert backend"):
        create_backend(ChannelConfig(type="teleport", name="t"))


def test_the_builtin_types_are_all_registered():
    assert {
        "lark",
        "slack",
        "dingtalk",
        "wecom",
        "webhook",
        "email",
        "callable",
    } <= set(BACKENDS)
    assert BACKENDS["callable"] is CallableBackend


def test_a_url_backend_without_a_url_reports_clearly():
    from expr_tracker.alerts.backends import SlackBackend

    backend = SlackBackend(ChannelConfig(type="slack", name="s"))
    with pytest.raises(SendError, match="no webhook URL"):
        backend.send(message())
