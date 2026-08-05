"""Artifact storage modes, plus print_to_screen, lexer edges and TOML config.

These are the features the coverage report showed as untested rather than merely
under-covered.
"""

import json
import os
import sys

import pytest

from expr_tracker.alerts import AlertConfig, load_config
from expr_tracker.alerts.expr import ExprSyntaxError, parse, parse_duration, tokenize
from expr_tracker.artifacts import Artifact, ArtifactStore
from expr_tracker.history import HistoryStore

# ====================================================================== artifacts


@pytest.fixture
def artifacts(tmp_path):
    return ArtifactStore(tmp_path / "store")


@pytest.fixture
def payload(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "model.bin").write_bytes(b"weights")
    (source / "notes.txt").write_text("hello")
    return source


def logged(artifacts, payload, **kwargs):
    artifact = Artifact("model", type="model").add_dir(payload)
    return artifacts.log(artifact, **kwargs)


@pytest.mark.parametrize("mode", ["copy", "link"])
def test_copy_and_link_both_materialise_the_files(artifacts, payload, mode):
    stored = logged(artifacts, payload, mode=mode)
    assert stored.dir is not None
    assert (stored.dir / "model.bin").read_bytes() == b"weights"
    assert (stored.dir / "notes.txt").read_text() == "hello"


def test_copy_is_independent_of_the_source(artifacts, payload):
    stored = logged(artifacts, payload, mode="copy")
    (payload / "model.bin").write_bytes(b"rewritten in place")
    assert (stored.dir / "model.bin").read_bytes() == b"weights"


def test_link_shares_the_inode_with_the_source(artifacts, payload):
    stored = logged(artifacts, payload, mode="link")
    assert (stored.dir / "model.bin").stat().st_ino == (
        payload / "model.bin"
    ).stat().st_ino
    # this is exactly why copy is the default: an in-place rewrite is visible
    (payload / "model.bin").write_bytes(b"weights!")
    assert (stored.dir / "model.bin").read_bytes() == b"weights!"


def test_link_falls_back_to_copy_when_linking_fails(artifacts, payload, monkeypatch):
    def refuse(source, destination):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", refuse)
    stored = logged(artifacts, payload, mode="link")
    assert (stored.dir / "model.bin").read_bytes() == b"weights"
    assert (stored.dir / "model.bin").stat().st_ino != (
        payload / "model.bin"
    ).stat().st_ino


def test_reference_mode_records_without_copying(artifacts, payload):
    stored = logged(artifacts, payload, mode="reference")
    assert stored.version == 0
    assert stored.dir is None or not any(stored.dir.glob("*"))
    entry = next(iter(stored.entries))
    assert entry.digest  # the content was still hashed for dedup


def test_reference_and_copy_share_a_digest(artifacts, payload, tmp_path):
    referenced = logged(artifacts, payload, mode="reference")
    other = ArtifactStore(tmp_path / "other")
    copied = logged(other, payload, mode="copy")
    assert referenced.digest == copied.digest


def test_an_unknown_mode_is_rejected(artifacts, payload):
    with pytest.raises(ValueError, match="Unknown artifact mode"):
        logged(artifacts, payload, mode="teleport")


@pytest.mark.parametrize("mode", ["copy", "link", "reference"])
def test_identical_content_is_deduplicated_in_every_mode(artifacts, payload, mode):
    first = logged(artifacts, payload, mode=mode)
    second = logged(artifacts, payload, mode=mode)
    assert second.version == first.version == 0
    assert len([a for a in artifacts.entries() if a.name == "model"]) == 1


def test_changed_content_creates_a_new_version(artifacts, payload):
    first = logged(artifacts, payload)
    (payload / "model.bin").write_bytes(b"different")
    second = logged(artifacts, payload)
    assert (first.version, second.version) == (0, 1)
    assert (first.dir / "model.bin").read_bytes() == b"weights"
    assert (second.dir / "model.bin").read_bytes() == b"different"


def test_a_single_file_artifact(artifacts, tmp_path):
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"state")
    stored = artifacts.log(Artifact("ckpt", type="model").add_file(path))
    assert (stored.dir / "ckpt.pt").read_bytes() == b"state"


def test_a_file_can_be_stored_under_another_name(artifacts, tmp_path):
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"state")
    artifact = Artifact("ckpt", type="model").add_file(path, name="weights/final.pt")
    stored = artifacts.log(artifact)
    assert (stored.dir / "weights" / "final.pt").read_bytes() == b"state"


def test_nested_directories_keep_their_layout(artifacts, tmp_path):
    root = tmp_path / "tree"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "deep.txt").write_text("deep")
    (root / "top.txt").write_text("top")
    stored = artifacts.log(Artifact("tree", type="dataset").add_dir(root))
    assert (stored.dir / "a" / "b" / "deep.txt").read_text() == "deep"
    assert (stored.dir / "top.txt").read_text() == "top"


def test_materialising_twice_does_not_duplicate_work(artifacts, payload):
    stored = logged(artifacts, payload)
    marker = stored.dir / "model.bin"
    marker.write_bytes(b"tampered")
    artifacts._materialise(stored, "copy")  # existing files are left alone
    assert marker.read_bytes() == b"tampered"


# ====================================================================== screen


@pytest.fixture
def store(tmp_path):
    created = []

    def factory(**options):
        instance = HistoryStore()
        options.setdefault("max_open_seconds", None)
        instance.init(project="p", name="r", dir=str(tmp_path), **options)
        created.append(instance)
        return instance

    yield factory
    for instance in created:
        instance.finish()


def test_printing_is_off_by_default(store, capsys):
    instance = store()
    instance.log({"loss": 1.0})
    instance.flush(commit_open=True)
    assert capsys.readouterr().out == ""


def test_printing_can_be_switched_on(store, capsys):
    instance = store(print_to_screen=True)
    instance.log({"loss": 1.0})
    instance.flush(commit_open=True)
    output = capsys.readouterr().out
    assert "loss" in output and "_step" in output


def test_a_custom_print_handle_receives_every_committed_row(store):
    lines: list[str] = []
    instance = store(print_to_screen=True, print_handle=lines.append)
    for step in range(5):
        instance.log({"loss": float(step)})
    instance.flush(commit_open=True)
    assert len(lines) == 5
    assert "4.0" in lines[-1]


def test_only_committed_rows_are_printed(store):
    lines: list[str] = []
    instance = store(print_to_screen=True, print_handle=lines.append)
    instance.log({"a": 1}, step=0, commit=False)
    instance.log({"b": 2}, step=0, commit=False)
    assert lines == []
    instance.flush(commit_open=True)
    assert len(lines) == 1
    assert "'a': 1" in lines[0] and "'b': 2" in lines[0]


def test_a_failing_print_handle_does_not_lose_data(store):
    def explode(line):
        raise RuntimeError("terminal gone")

    instance = store(print_to_screen=True, print_handle=explode)
    for step in range(5):
        instance.log({"loss": float(step)})
    instance.flush(commit_open=True)
    assert [r["_step"] for r in instance.get(-1)] == list(range(5))


def test_the_print_handle_is_not_called_when_printing_is_off(store):
    lines: list[str] = []
    instance = store(print_to_screen=False, print_handle=lines.append)
    instance.log({"loss": 1.0})
    instance.flush(commit_open=True)
    assert lines == []


# ====================================================================== lexer


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1e-8", 1e-8),
        ("1E-8", 1e-8),
        ("1e+3", 1e3),
        ("1E+3", 1e3),
        ("2e5", 2e5),
        ("1.5e-3", 1.5e-3),
        (".5", 0.5),
        ("0.5", 0.5),
        ("42", 42),
        ("0", 0),
    ],
)
def test_numeric_literals(source, expected):
    node = parse(f"loss < {source}")
    assert node.right.value == pytest.approx(expected)


def test_scientific_notation_survives_a_round_trip():
    assert (
        parse("lr < 1e-8").to_source()
        == parse(parse("lr < 1e-8").to_source()).to_source()
    )


@pytest.mark.parametrize("source", ["1e", "1e+", "1.2.3", "1e-"])
def test_malformed_numbers_are_rejected(source):
    with pytest.raises((ExprSyntaxError, ValueError)):
        parse(f"loss < {source}")


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("30s", 30),
        ("5m", 300),
        ("2h", 7200),
        ("1d", 86400),
        ("1.5m", 90),
        ("45", 45),
        (" 30s ", 30),
    ],
)
def test_duration_literals(text, seconds):
    assert parse_duration(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["5x", "abc", "", "s", "m30", "5ss"])
def test_malformed_durations_are_rejected(text):
    with pytest.raises(ValueError, match="Invalid duration"):
        parse_duration(text)


def test_a_duration_window_uses_the_lexer(monkeypatch):
    node = parse("mean(loss[90s]) > 1")
    assert node.to_source() == "mean(loss[90s]) > 1"


@pytest.mark.parametrize("source", ["loss # 1", "loss @ 1", "loss $ 1", "loss ? 1"])
def test_the_lexer_rejects_characters_it_cannot_tokenise(source):
    with pytest.raises(ExprSyntaxError, match="Unexpected character"):
        tokenize(source)


def test_an_unterminated_quoted_identifier_is_reported():
    with pytest.raises(ExprSyntaxError, match="Unterminated"):
        tokenize("`train/loss > 1")


@pytest.mark.parametrize(
    "name", ["a-b c", "train/loss", "with space", "中文指标", "0start", "a.b.c"]
)
def test_quoted_identifiers_carry_exotic_metric_names(name):
    """Backticks are the escape hatch for names the bare grammar cannot express."""
    from expr_tracker.alerts.expr import EvalContext, evaluate
    from expr_tracker.history import MetricSeries

    series = MetricSeries()
    series.add(0, 0.0, {name: 5.0})
    node = parse(f"`{name}` > 1")
    assert node.metrics() == {name}
    assert evaluate(node, EvalContext(series, step=0, record={name: 5.0})) is True
    assert parse(node.to_source()).metrics() == {name}


def test_an_operator_shaped_typo_is_rejected_by_the_parser():
    """``~`` tokenises as an operator, so the parser is what refuses it."""
    assert [t.kind for t in tokenize("loss ~ 1")] == ["NAME", "OP", "NUMBER", "EOF"]
    with pytest.raises(ExprSyntaxError, match="Unexpected token"):
        parse("loss ~ 1")


@pytest.mark.parametrize("source", ["loss # 1", "loss ~ 1"])
def test_syntax_errors_point_at_the_column(source):
    with pytest.raises(ExprSyntaxError) as excinfo:
        parse(source)
    message = str(excinfo.value)
    assert "column 6" in message
    assert "^" in message  # the caret line makes the position obvious


# ====================================================================== config


CONFIG = {
    "alert": {
        "enabled": True,
        "channels": [
            {"type": "slack", "name": "team", "url": "https://hooks.example/x"}
        ],
        "rules": ["loss > 10 => error: high"],
    }
}


def write_config(tmp_path, suffix: str, text: str):
    path = tmp_path / f"alert{suffix}"
    path.write_text(text, encoding="utf-8")
    return path


def test_toml_config_is_loaded(tmp_path):
    if sys.version_info < (3, 11):
        pytest.importorskip("tomli")
    path = write_config(
        tmp_path,
        ".toml",
        """
        [alert]
        enabled = true
        rules = ["loss > 10 => error: high"]

        [[alert.channels]]
        type = "slack"
        name = "team"
        url = "https://hooks.example/x"
        """.replace("        ", ""),
    )
    config = load_config(path)
    assert config.enabled is True
    assert [c.type for c in config.channels] == ["slack"]
    assert config.channels[0].url == "https://hooks.example/x"
    assert len(config.rules) == 1


def test_json_yaml_and_toml_agree(tmp_path):
    yaml = pytest.importorskip("yaml")
    json_path = write_config(tmp_path, ".json", json.dumps(CONFIG))
    yaml_path = write_config(tmp_path, ".yaml", yaml.safe_dump(CONFIG))
    toml_path = write_config(
        tmp_path,
        ".toml",
        '[alert]\nenabled = true\nrules = ["loss > 10 => error: high"]\n\n'
        '[[alert.channels]]\ntype = "slack"\nname = "team"\n'
        'url = "https://hooks.example/x"\n',
    )

    configs = [load_config(p) for p in (json_path, yaml_path, toml_path)]
    for config in configs:
        assert isinstance(config, AlertConfig)
        assert [(c.type, c.name, c.url) for c in config.channels] == [
            ("slack", "team", "https://hooks.example/x")
        ]
        assert [r.condition for r in config.rules] == ["loss > 10"]
        assert config.enabled is True


def test_a_config_file_without_an_alert_section_is_used_as_is(tmp_path):
    path = write_config(
        tmp_path, ".json", json.dumps({"channels": [], "enabled": False})
    )
    assert load_config(path).enabled is False


def test_an_unknown_suffix_is_read_as_json(tmp_path):
    path = write_config(tmp_path, ".conf", json.dumps(CONFIG))
    assert [c.type for c in load_config(path).channels] == ["slack"]


def test_a_broken_config_file_is_an_error(tmp_path):
    path = write_config(tmp_path, ".json", "{ not json")
    with pytest.raises(json.JSONDecodeError):
        load_config(path)


def test_a_missing_config_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.json")


def test_an_unsupported_config_source_is_rejected():
    with pytest.raises(TypeError, match="Unsupported alert config"):
        load_config(12345)
