"""Small public surfaces: explain(), AlertLevel ordering, artifact objects, policies."""

import itertools
import operator

import pytest

import expr_tracker as et
from expr_tracker.alerts import AlertLevel
from expr_tracker.alerts.expr import EvalContext, explain, parse
from expr_tracker.alerts.models import AlertRule, ChannelConfig, WebhookPolicy
from expr_tracker.artifacts import Artifact, ArtifactStore, coerce_artifact
from expr_tracker.history import MetricSeries

# ====================================================================== explain


def context(points: dict[str, list] | None = None, **kwargs):
    points = points or {}
    series = MetricSeries()
    length = max((len(v) for v in points.values()), default=0)
    for i in range(length):
        series.add(i, float(i), {k: v[i] for k, v in points.items() if i < len(v)})
    kwargs.setdefault("step", max(length - 1, 0))
    kwargs.setdefault(
        "record", {name: values[-1] for name, values in points.items() if values}
    )
    return EvalContext(series, **kwargs)


def test_explain_annotates_a_comparison():
    text = explain(parse("loss > 5"), context({"loss": [9.0]}))
    assert "9" in text and ">" in text and "5" in text


def test_explain_renders_a_not_expression():
    ctx = context({"loss": [9.0]})
    text = explain(parse("not (loss > 5)"), ctx)
    assert text.startswith("not ")
    assert "9" in text


def test_explain_renders_nested_not():
    text = explain(parse("not (not (loss > 5))"), context({"loss": [9.0]}))
    assert text.count("not ") == 2


def test_explain_renders_not_inside_a_boolean():
    ctx = context({"a": [1.0], "b": [2.0]})
    text = explain(parse("not (a > 5) and b > 1"), ctx)
    assert "not " in text and " and " in text


def test_explain_renders_boolean_chains():
    ctx = context({"a": [1.0], "b": [2.0], "c": [3.0]})
    assert explain(parse("a > 0 or b > 0 or c > 0"), ctx).count(" or ") == 2


def test_explain_annotates_function_calls():
    text = explain(parse("mean(loss[3]) > 1"), context({"loss": [1.0, 2.0, 3.0]}))
    assert "mean(loss[3])" in text and "2" in text


def test_explain_marks_unresolvable_values():
    assert "?" in explain(parse("not (missing > 1)"), context({"loss": [1.0]}))


def test_explain_of_a_bare_metric():
    assert "9" in explain(parse("loss"), context({"loss": [9.0]}))


def test_explain_is_used_in_the_default_rule_message(tmp_path):
    received: list = []
    run = et.init(
        project="x",
        name="explain",
        dir=str(tmp_path),
        backends=[],
        max_open_seconds=None,
        alert={
            "channels": [
                {
                    "type": "callable",
                    "name": "c",
                    "options": {"handler": received.append},
                    "policy": {"async_send": False, "dedup_window": 0},
                }
            ]
        },
        alert_rules=[{"condition": "not (loss < 5)"}],  # no message: uses {expr}
    )
    try:
        run.log({"loss": 9.0})
        assert "not " in received[0].text
    finally:
        et.finish()


# ====================================================================== levels


ORDER = ["debug", "info", "warning", "error", "critical"]


def test_levels_are_ordered():
    levels = [AlertLevel.parse(name) for name in ORDER]
    for lower, higher in itertools.pairwise(levels):
        assert lower < higher and higher > lower
        assert lower <= higher and higher >= lower


def test_a_level_compares_equal_to_itself():
    assert AlertLevel.ERROR >= AlertLevel.ERROR
    assert AlertLevel.ERROR <= AlertLevel.ERROR
    assert not AlertLevel.ERROR > AlertLevel.ERROR
    assert not AlertLevel.ERROR < AlertLevel.ERROR


@pytest.mark.parametrize("other", [3, None, 1.5, object()])
def test_comparing_with_a_foreign_type_is_a_type_error(other):
    """Returning NotImplemented lets Python raise instead of comparing nonsense."""
    for name in ("__lt__", "__le__", "__gt__", "__ge__"):
        assert getattr(AlertLevel.ERROR, name)(other) is NotImplemented
    with pytest.raises(TypeError):
        operator.lt(AlertLevel.ERROR, other)


@pytest.mark.parametrize(
    ("left", "op", "right", "expected"),
    [
        ("critical", "ge", "error", True),
        ("error", "lt", "critical", True),
        ("warning", "ge", "error", False),
        ("error", "ge", "warn", True),
        ("info", "le", "INFO", True),
    ],
)
def test_levels_compare_with_names_by_severity(left, op, right, expected):
    """A str enum would otherwise compare alphabetically: 'critical' < 'error'."""
    assert getattr(operator, op)(AlertLevel.parse(left), right) is expected


def test_comparing_with_an_unknown_level_name_is_an_error():
    with pytest.raises(ValueError, match="Unknown alert level"):
        _ = AlertLevel.ERROR < "catastrophic"


def test_severity_filtering_reads_naturally():
    """The idiom this protects: dropping anything below a named threshold."""
    levels = [AlertLevel.parse(name) for name in ORDER]
    assert [x.value for x in levels if x >= "error"] == ["error", "critical"]


def test_a_level_is_not_equal_to_its_own_name():
    assert AlertLevel.ERROR != 3
    assert AlertLevel.ERROR.value == "error"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("warn", AlertLevel.WARNING),
        ("WARNING", AlertLevel.WARNING),
        ("Error", AlertLevel.ERROR),
        ("fatal", AlertLevel.CRITICAL),
        (AlertLevel.INFO, AlertLevel.INFO),
    ],
)
def test_levels_are_parsed_leniently(given, expected):
    assert AlertLevel.parse(given) is expected


def test_an_unknown_level_lists_the_valid_ones():
    with pytest.raises(ValueError, match="Unknown alert level") as excinfo:
        AlertLevel.parse("catastrophic")
    assert "critical" in str(excinfo.value)


def test_rank_is_stable_and_distinct():
    ranks = [AlertLevel.parse(name).rank for name in ORDER]
    assert ranks == sorted(ranks) and len(set(ranks)) == len(ranks)


# ====================================================================== artifacts


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "store")


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"weights")
    return path


def test_coerce_accepts_a_path(source):
    artifact = coerce_artifact(str(source), name="m", type="model")
    assert artifact.name == "m" and artifact.type == "model"
    assert artifact.files() == ["model.bin"]


def test_coerce_accepts_a_path_object(source):
    assert coerce_artifact(source, name="m", type="model").files() == ["model.bin"]


def test_coerce_passes_an_artifact_through(source):
    original = Artifact("m", type="model").add_file(source)
    assert coerce_artifact(original) is original


def test_coerce_overrides_the_type_of_an_artifact(source):
    original = Artifact("m", type="model").add_file(source)
    assert coerce_artifact(original, type="dataset").type == "dataset"


def test_coerce_merges_metadata_into_an_artifact(source):
    original = Artifact("m", type="model", metadata={"a": 1}).add_file(source)
    merged = coerce_artifact(original, metadata={"b": 2})
    assert merged.metadata == {"a": 1, "b": 2}


def test_coerce_leaves_the_type_alone_when_not_given(source):
    original = Artifact("m", type="model").add_file(source)
    assert coerce_artifact(original).type == "model"


def test_use_artifact_accepts_an_artifact_object(tmp_path, source):
    run = et.init(project="x", name="art", dir=str(tmp_path), backends=[])
    try:
        logged = run.log_artifact(str(source), name="m", type="model")
        resolved = run.use_artifact(logged)
        assert resolved is logged
        lineage = (run.history.log_dir / "artifacts.jsonl").read_text().splitlines()
        assert len(lineage) == 2  # the use was still recorded
    finally:
        et.finish()


def test_download_returns_the_stored_directory(store, source):
    stored = store.log(Artifact("m", type="model").add_file(source))
    assert stored.download() == stored.dir
    assert (stored.download() / "model.bin").read_bytes() == b"weights"


def test_download_copies_into_a_target_directory(store, source, tmp_path):
    stored = store.log(Artifact("m", type="model").add_file(source))
    target = tmp_path / "out"
    assert stored.download(target) == target
    assert (target / "model.bin").read_bytes() == b"weights"


def test_downloading_an_unlogged_artifact_is_an_error(source):
    with pytest.raises(RuntimeError, match="has not been logged"):
        Artifact("m", type="model").add_file(source).download()


def test_get_path_of_an_unlogged_artifact_is_an_error(source):
    with pytest.raises(RuntimeError, match="has not been logged"):
        Artifact("m", type="model").add_file(source).get_path("model.bin")


def test_get_path_addresses_a_stored_file(store, source):
    stored = store.log(Artifact("m", type="model").add_file(source))
    assert stored.get_path("model.bin").read_bytes() == b"weights"


def test_a_draft_artifact_reports_a_draft_name(source):
    assert Artifact("m", type="model").add_file(source).qualified_name == "m:draft"


def test_the_repr_shows_the_version_and_file_count(store, source):
    stored = store.log(Artifact("m", type="model").add_file(source))
    text = repr(stored)
    assert "m:v0" in text and "files=1" in text and "model" in text


def test_reference_entries_are_not_downloaded(store, tmp_path):
    artifact = Artifact("remote", type="dataset").add_reference("s3://bucket/key")
    stored = store.log(artifact, mode="reference")
    target = tmp_path / "out"
    assert stored.download(target) == target
    assert list(target.iterdir()) == []


# ====================================================================== policies


def test_retry_on_status_accepts_a_list():
    policy = WebhookPolicy.from_dict({"retry_on_status": [500, 503]})
    assert policy.retry_on_status == (500, 503)
    assert isinstance(policy.retry_on_status, tuple)


def test_retry_on_status_keeps_its_default():
    assert 429 in WebhookPolicy().retry_on_status


def test_from_dict_of_nothing_is_the_default():
    assert WebhookPolicy.from_dict(None) == WebhookPolicy()
    assert WebhookPolicy.from_dict({}) == WebhookPolicy()


def test_from_dict_overrides_only_what_is_given():
    policy = WebhookPolicy.from_dict({"timeout": 1.5})
    assert policy.timeout == 1.5
    assert policy.max_retries == WebhookPolicy().max_retries


def test_merged_prefers_the_override():
    default = WebhookPolicy(timeout=10)
    override = WebhookPolicy(timeout=1)
    assert default.merged(override) is override
    assert default.merged(None) is default


def test_a_channel_falls_back_to_the_default_policy():
    from expr_tracker.alerts import AlertConfig, Dispatcher

    default = WebhookPolicy(timeout=99, async_send=False)
    dispatcher = Dispatcher(
        AlertConfig(
            default_policy=default,
            channels=[
                ChannelConfig(
                    type="callable", name="c", options={"handler": lambda m: None}
                )
            ],
        )
    )
    try:
        assert dispatcher.channels["c"].policy.timeout == 99
    finally:
        dispatcher.close()


def test_a_channel_policy_wins_over_the_default():
    from expr_tracker.alerts import AlertConfig, Dispatcher

    dispatcher = Dispatcher(
        AlertConfig(
            default_policy=WebhookPolicy(timeout=99),
            channels=[
                ChannelConfig(
                    type="callable",
                    name="c",
                    options={"handler": lambda m: None},
                    policy=WebhookPolicy(timeout=1, async_send=False),
                )
            ],
        )
    )
    try:
        assert dispatcher.channels["c"].policy.timeout == 1
    finally:
        dispatcher.close()


def test_a_channel_config_needs_a_type():
    from expr_tracker.alerts.models import _as_channel

    with pytest.raises(ValueError, match="requires a 'type' key"):
        _as_channel({"name": "c"})


def test_a_channel_can_be_declared_by_type_alone():
    from expr_tracker.alerts.models import _as_channel

    channel = _as_channel("slack")
    assert channel.type == "slack" and channel.name == "slack"


def test_a_rule_can_be_declared_as_a_string_or_a_dict():
    from expr_tracker.alerts.models import _as_rule

    assert _as_rule("loss > 1 => error: x").level is AlertLevel.ERROR
    assert _as_rule({"condition": "loss > 1"}).condition == "loss > 1"
    existing = AlertRule(condition="a > 1")
    assert _as_rule(existing) is existing
