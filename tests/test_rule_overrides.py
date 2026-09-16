"""Rule-object overrides must work like text/dict/builder overrides."""

import expr_tracker as et
from expr_tracker.alerts import AlertRule
from expr_tracker.alerts.expr import M, parse_rule
from expr_tracker.alerts.models import DEFAULT_RULE_MESSAGE


def test_rule_objects_are_preserved_when_no_override_is_requested():
    rule = AlertRule(condition="loss > 0")
    assert parse_rule(rule) is rule


def test_rule_object_overrides_are_validated_copies():
    original = AlertRule(
        condition="loss > 0", name="source", channels=["left"], tags=["original"]
    )
    copied = parse_rule(
        original, name="copy", level="error", enabled=False, channels=["right"]
    )
    assert copied is not original
    assert copied.name == "copy" and copied.level.value == "error"
    assert not copied.enabled and copied.channels == ["right"]
    copied.tags.append("copy")
    assert original.name == "source" and original.enabled
    assert original.channels == ["left"] and original.tags == ["original"]


def test_auto_named_copies_keep_or_update_identity_as_appropriate():
    original = AlertRule(condition="loss > 0")
    disabled = parse_rule(original, enabled=False)
    changed = parse_rule(original, level="error")
    assert disabled.auto_named and disabled.name == original.name
    assert changed.auto_named and changed.name != original.name
    assert original.level.value == "warning" and original.enabled


def test_explicit_default_message_has_the_same_auto_identity():
    implicit = AlertRule(condition="loss > 0")
    explicit = AlertRule(condition="loss > 0", message=DEFAULT_RULE_MESSAGE)
    assert implicit.name == explicit.name


def test_builder_condition_can_be_overridden_without_duplicate_arguments():
    rule = parse_rule(M.loss > 0, condition="accuracy > 0", level="error")
    assert rule.condition == "accuracy > 0"
    assert rule.level.value == "error"


def test_object_overrides_control_real_rule_enablement_and_routing(run):
    left, right = [], []
    run(alert={"channels": [
        {
            "type": "callable", "name": "left", "options": {"handler": left.append},
            "policy": {"async_send": False},
        },
        {
            "type": "callable", "name": "right", "options": {"handler": right.append},
            "policy": {"async_send": False},
        },
    ]})
    template = AlertRule(condition="loss > 0", channels=["left"])
    et.add_alert_rule(template, name="disabled", enabled=False)
    enabled = et.add_alert_rule(template, name="enabled", level="error", channels=["right"])
    et.log({"loss": 1})
    assert left == []
    assert len(right) == 1 and right[0].level.value == "error"
    assert enabled.channels == ["right"]
    assert template.enabled and template.channels == ["left"]
