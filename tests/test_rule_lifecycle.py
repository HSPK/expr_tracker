"""Rule state machine details and the remaining configuration surface."""

import json

import pytest

import expr_tracker as et
from expr_tracker.alerts import AlertLevel, alert
from expr_tracker.alerts.models import AlertRule, WebhookPolicy


@pytest.fixture
def run(tmp_path):
    """A run whose alerts land in a list, with rules given per test."""
    created = []

    def factory(rules, **options):
        received: list = []
        instance = et.init(
            project="sm",
            name=options.pop("name", "r"),
            dir=str(tmp_path),
            backends=[],
            max_open_seconds=None,
            alert={
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
                ]
            },
            alert_rules=rules,
            **options,
        )
        created.append(instance)
        return instance, received

    yield factory
    if et.get_run() is not None:
        et.finish()


# ------------------------------------------------------------------ max_fires


def test_max_fires_caps_the_number_of_alerts(run):
    instance, received = run(
        [{"condition": "loss > 1", "mode": "level", "cooldown": None, "max_fires": 3}]
    )
    for _ in range(10):
        instance.log({"loss": 5.0})
    assert len(received) == 3


def test_max_fires_counts_across_episodes(run):
    instance, received = run([{"condition": "loss > 1", "max_fires": 2}])
    for value in (5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0):
        instance.log({"loss": value})
    assert len(received) == 2  # four episodes, only two alerts


def test_without_max_fires_every_episode_alerts(run):
    instance, received = run([{"condition": "loss > 1"}])
    for value in (5.0, 0.0, 5.0, 0.0, 5.0):
        instance.log({"loss": value})
    assert len(received) == 3


def test_max_fires_is_reported_in_the_stats(run):
    instance, received = run([{"condition": "loss > 1", "max_fires": 1}])
    instance.log({"loss": 5.0})
    instance.log({"loss": 0.0})
    instance.log({"loss": 5.0})
    stats = instance.info()["alerts"]["rules"]
    assert next(iter(stats.values()))["fires"] == 1
    assert len(received) == 1


# ------------------------------------------------------------------ recovery


def test_recovery_sends_a_second_message(run):
    instance, received = run(
        [{"condition": "loss > 1", "message": "high", "notify_recovery": True}]
    )
    instance.log({"loss": 5.0})
    assert len(received) == 1
    instance.log({"loss": 0.5})
    assert len(received) == 2


def test_a_recovery_message_is_informational(run):
    instance, received = run(
        [
            {
                "condition": "loss > 1",
                "level": "critical",
                "title": "Diverged",
                "notify_recovery": True,
            }
        ]
    )
    instance.log({"loss": 5.0})
    instance.log({"loss": 0.5})
    fired, recovered = received
    assert fired.level is AlertLevel.CRITICAL and fired.title == "Diverged"
    assert recovered.level is AlertLevel.INFO
    assert recovered.title == "[recovered] Diverged"
    assert "no longer holds" in recovered.text


def test_recovery_is_off_by_default(run):
    instance, received = run([{"condition": "loss > 1"}])
    instance.log({"loss": 5.0})
    instance.log({"loss": 0.5})
    assert len(received) == 1


def test_recovery_only_follows_an_actual_fire(run):
    instance, received = run([{"condition": "loss > 1", "notify_recovery": True}])
    for _ in range(5):
        instance.log({"loss": 0.5})  # never fired, so nothing to recover from
    assert received == []


def test_recovery_and_max_fires_interact(run):
    """The cap applies to fires; a recovery still closes an open episode."""
    instance, received = run(
        [{"condition": "loss > 1", "max_fires": 1, "notify_recovery": True}]
    )
    instance.log({"loss": 5.0})
    instance.log({"loss": 0.5})
    instance.log({"loss": 5.0})
    instance.log({"loss": 0.5})
    kinds = [m.title.startswith("[recovered]") for m in received]
    assert kinds == [False, True]


def test_recovery_carries_the_rule_tags(run):
    instance, received = run(
        [{"condition": "loss > 1", "tags": ["gpu"], "notify_recovery": True}]
    )
    instance.log({"loss": 5.0})
    instance.log({"loss": 0.5})
    assert all(m.tags == ["gpu"] for m in received)


def test_fire_and_recovery_have_distinct_dedup_keys(run):
    instance, received = run([{"condition": "loss > 1", "notify_recovery": True}])
    instance.log({"loss": 5.0})
    instance.log({"loss": 0.5})
    assert received[0].key() != received[1].key()


# ------------------------------------------------------------------ error cap


def test_a_rule_that_keeps_failing_is_disabled(run, monkeypatch):
    instance, received = run([{"condition": "loss > 1"}])
    compiled = next(iter(instance.alerts.rules.values()))
    monkeypatch.setattr(
        "expr_tracker.alerts.engine.evaluate",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    for _ in range(5):
        instance.log({"loss": 5.0})
    assert compiled.rule.enabled is False
    assert received == []
    assert len(instance.history_query(-1)) == 5  # logging is unaffected


def test_a_transient_failure_does_not_disable_a_rule(run, monkeypatch):
    from expr_tracker.alerts import engine as engine_module

    instance, received = run(
        [{"condition": "loss > 1", "mode": "level", "cooldown": None}]
    )
    compiled = next(iter(instance.alerts.rules.values()))
    real = engine_module.evaluate
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return real(*args, **kwargs)

    monkeypatch.setattr(engine_module, "evaluate", flaky)
    for _ in range(4):
        instance.log({"loss": 5.0})
    assert compiled.rule.enabled is True
    assert len(received) == 3


# ------------------------------------------------------------------ rule dicts


def test_prometheus_style_keys_are_accepted():
    rule = AlertRule.from_dict(
        {"alert": "HighLoss", "expr": "loss > 10", "for": 3, "level": "error"}
    )
    assert rule.name == "HighLoss"
    assert rule.condition == "loss > 10"
    assert rule.for_steps == 3
    assert rule.level is AlertLevel.ERROR


def test_native_keys_win_over_their_prometheus_aliases():
    rule = AlertRule.from_dict(
        {"alert": "ignored", "name": "kept", "expr": "a > 1", "condition": "b > 1"}
    )
    assert rule.name == "kept" and rule.condition == "b > 1"


def test_an_unknown_rule_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown alert rule options"):
        AlertRule.from_dict({"condition": "loss > 1", "typo": 1})


def test_an_unknown_rule_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown alert rule mode"):
        AlertRule(condition="loss > 1", mode="sideways")


def test_a_rule_without_a_name_gets_a_stable_one():
    first = AlertRule(condition="loss > 1")
    second = AlertRule(condition="loss > 1")
    third = AlertRule(condition="loss > 2")
    assert first.name == second.name != third.name
    assert first.name.startswith("rule_")


def test_for_steps_is_clamped_to_at_least_one():
    assert AlertRule(condition="a > 1", for_steps=0).for_steps == 1
    assert AlertRule(condition="a > 1", for_steps=-5).for_steps == 1


def test_a_prometheus_rule_works_end_to_end(run):
    instance, received = run([{"alert": "Spike", "expr": "loss > 10", "for": 2}])
    instance.log({"loss": 50.0})
    assert received == []  # one step is not enough
    instance.log({"loss": 50.0})
    assert len(received) == 1
    assert received[0].title == "Spike"


# ------------------------------------------------------------------ messages


def test_alert_accepts_a_subtitle(monkeypatch):
    from expr_tracker.alerts import configure_alert, reset_alert_config

    received: list = []
    configure_alert(
        channels=[
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
        ]
    )
    try:
        alert(title="T", text="body", subtitle="a subtitle", level="warning")
        assert received[0].subtitle == "a subtitle"
        assert received[0].to_dict()["subtitle"] == "a subtitle"
    finally:
        reset_alert_config()


def test_a_rule_alert_carries_the_run_name_as_its_subtitle(run):
    instance, received = run(["loss > 1 => warning: x"], name="subtitled")
    instance.log({"loss": 5.0})
    assert received[0].subtitle == "subtitled"


def test_a_message_serialises_every_field(run):
    instance, received = run(
        [{"condition": "loss > 1", "title": "T", "message": "m", "tags": ["a"]}]
    )
    instance.log({"loss": 5.0})
    payload = received[0].to_dict()
    assert json.dumps(payload)  # a webhook body must be serialisable
    assert payload["title"] == "T" and payload["text"] == "m"
    assert payload["level"] == "warning" and payload["tags"] == ["a"]
    assert payload["fields"]["condition"] == "loss > 1"


# ------------------------------------------------------------------ buffering


def test_max_pending_records_bounds_the_write_buffer(tmp_path, monkeypatch):
    from expr_tracker.history import HistoryStore

    store = HistoryStore()
    store.init(
        project="sm",
        name="pending",
        dir=str(tmp_path),
        max_open_seconds=None,
        buffer_size=1,
        max_pending_records=5,
    )
    try:
        monkeypatch.setattr(
            store.writer, "_write", lambda payload: (_ for _ in ()).throw(OSError("x"))
        )
        for step in range(50):
            store.log({"loss": float(step)})
        assert len(store.writer.buffer) <= 5
        assert store.writer.dropped >= 45

        monkeypatch.undo()
        store.flush(commit_open=True)
        steps = [r["_step"] for r in store.get(-1)]
        assert steps[-1] == 49  # the newest records are the ones kept
    finally:
        store.finish()


def test_max_pending_records_is_at_least_the_buffer_size(tmp_path):
    from expr_tracker.history.writer import JsonlWriter

    writer = JsonlWriter(tmp_path / "m.jsonl", buffer_size=100, max_pending_records=10)
    try:
        assert writer.max_pending_records == 100
    finally:
        writer.close()


def test_an_unknown_policy_option_is_rejected():
    with pytest.raises(ValueError, match="Unknown webhook policy options"):
        WebhookPolicy.from_dict({"timeout": 1, "nonsense": 2})


def test_an_unknown_history_option_is_rejected(tmp_path):
    from expr_tracker.history import HistoryStore

    with pytest.raises(TypeError, match="Unknown"):
        HistoryStore().init(project="sm", name="bad", dir=str(tmp_path), nonsense=1)


# ------------------------------------------------------------------ identity


def test_rules_differing_only_in_level_both_register(run):
    """ "Warn the team" and "page on-call" on one condition must coexist."""
    instance, received = run(
        ["loss > 1 => warning: notify", "loss > 1 => critical: page"]
    )
    assert len(et.list_alert_rules()) == 2
    instance.log({"loss": 5.0})
    assert sorted(m.text for m in received) == ["notify", "page"]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            {"condition": "a > 1", "message": "x"},
            {"condition": "a > 1", "message": "y"},
        ),
        (
            {"condition": "a > 1", "level": "warning"},
            {"condition": "a > 1", "level": "error"},
        ),
        ({"condition": "a > 1", "title": "T1"}, {"condition": "a > 1", "title": "T2"}),
        (
            {"condition": "a > 1", "tags": ["gpu"]},
            {"condition": "a > 1", "tags": ["cpu"]},
        ),
        (
            {"condition": "a > 1", "channels": ["x"]},
            {"condition": "a > 1", "channels": ["y"]},
        ),
        (
            {"condition": "a > 1", "mode": "edge"},
            {"condition": "a > 1", "mode": "level"},
        ),
        (
            {"condition": "a > 1", "for_steps": 1},
            {"condition": "a > 1", "for_steps": 2},
        ),
    ],
)
def test_any_distinguishing_field_yields_a_distinct_name(first, second):
    assert AlertRule.from_dict(first).name != AlertRule.from_dict(second).name


def test_identical_rules_still_share_a_name():
    a = AlertRule.from_dict({"condition": "a > 1", "message": "m", "level": "error"})
    b = AlertRule.from_dict({"condition": "a > 1", "message": "m", "level": "error"})
    assert a.name == b.name


def test_replacing_a_named_rule_is_reported(run):
    instance, _ = run([{"condition": "a > 1", "name": "mine", "message": "first"}])
    instance.alerts.add_rule(
        {"condition": "b > 1", "name": "mine", "message": "second"}
    )
    rules = et.list_alert_rules()
    assert len(rules) == 1 and rules[0].message == "second"


# ------------------------------------------------------------------ typos


def test_a_typo_in_a_metric_name_is_reported(run):
    instance, received = run(["los > 1 => error: typo", "loss > 1 => error: real"])
    for _ in range(5):
        instance.log({"loss": 5.0})
    stats = instance.info()["alerts"]["rules"]
    unresolved = {n: s["unresolved_metrics"] for n, s in stats.items()}
    assert sorted(unresolved.values()) == [[], ["los"]]
    assert len(received) == 1  # only the correctly spelled rule fired


def test_a_resolved_metric_is_never_reported_as_missing(run):
    instance, _ = run(["train.loss > 1 => error: dotted"])
    instance.log({"train/loss": 5.0})
    stats = next(iter(instance.info()["alerts"]["rules"].values()))
    assert stats["unresolved_metrics"] == []  # dots resolve to slashes


def test_nothing_is_reported_before_anything_is_logged(run):
    instance, _ = run(["los > 1 => error: typo"])
    engine = instance.alerts
    engine._report_unresolved()  # must not accuse a run that never logged
    stats = next(iter(instance.info()["alerts"]["rules"].values()))
    assert stats["unresolved_metrics"] == ["los"]


def test_a_metric_that_appears_late_is_not_a_typo(run):
    instance, _ = run(["eval.acc > 0.5 => warning: good"])
    for _ in range(5):
        instance.log({"loss": 1.0})
    assert next(iter(instance.info()["alerts"]["rules"].values()))["unresolved_metrics"]
    instance.log({"eval/acc": 0.9})
    assert not next(iter(instance.info()["alerts"]["rules"].values()))[
        "unresolved_metrics"
    ]
