"""UTF-8 persistence preserves filesystem-surrogate strings using JSON escapes."""

import json
import os
from pathlib import Path

import pytest

import expr_tracker as et
from expr_tracker.alerts.backends import base as backend_base
from expr_tracker.alerts.expr import M
from expr_tracker.history import store as store_module
from expr_tracker.history.codec import encode_line
from expr_tracker.trace import write_trace


@pytest.mark.parametrize("text", ["file\udcff", "value\ud800", "\u4e2d\u6587", "\U0001f600"])
def test_jsonl_strings_and_keys_round_trip_through_valid_utf8(text):
    record = {"_step": 0, text: {"value": text}, "normal": 1.0}
    payload = encode_line(record)
    assert json.loads(payload.decode("utf-8")) == record
    assert payload.endswith(b"\n")
    if not any(0xD800 <= ord(character) <= 0xDFFF for character in text):
        assert text.encode("utf-8") in payload


def test_surrogate_values_do_not_drop_metrics_config_summary_or_spans(run):
    text = "file\udcff"
    instance = run(config={"path": text})
    et.log({"path": Path(text), "loss": 1.0})
    with et.span(f"work/{text}", path=Path(text)):
        pass
    et.finish()
    rows = instance.history_query(-1)
    assert rows[0]["path"] == text and rows[0]["loss"] == 1.0
    assert json.loads(instance.history.config_fp.read_text())["path"] == text
    assert json.loads(instance.summary.path.read_text())["path"] == text
    span = json.loads(instance.history.spans_fp.read_text())
    assert span["name"] == f"work/{text}"
    assert span["args"]["path"] == text
    output = instance.history.log_dir / "trace.json"
    assert write_trace(instance.dir, output) == 1
    events = json.loads(output.read_text())["traceEvents"]
    assert [event["name"] for event in events if event["ph"] == "X"] == [f"work/{text}"]


def test_failed_config_replacement_preserves_the_existing_file(store, monkeypatch):
    history = store(config={"original": 1})
    original = history.config_fp.read_bytes()

    def fail(value, stream, **kwargs):
        stream.write('{"partial":')
        raise OSError("simulated full disk")

    monkeypatch.setattr(store_module.json, "dump", fail)
    history._write_config({"replacement": 2})
    assert history.config_fp.read_bytes() == original
    assert not list(history.log_dir.glob(f".{history.config_fp.name}.*.tmp"))


@pytest.mark.skipif(os.name != "posix", reason="filesystem surrogateescape is POSIX-specific")
def test_artifact_index_and_lineage_preserve_surrogate_strings(run, tmp_path):
    instance = run()
    text = os.fsdecode(b"data-\xff")
    source = tmp_path / f"{text}.bin"
    source.write_bytes(b"contents")
    artifact = et.Artifact("bundle", type=text, description=text, metadata={"label": text})
    artifact.add_file(str(source))
    et.log_artifact(artifact)
    restored = et.use_artifact("bundle:latest", type=text)
    assert restored.description == text
    assert restored.metadata["label"] == text
    assert (restored.dir / restored.entries[0].path).read_bytes() == b"contents"
    lineage = [
        json.loads(line)
        for line in (instance.history.log_dir / "artifacts.jsonl").read_text().splitlines()
    ]
    assert all(record["type"] == text for record in lineage)


def test_webhook_json_can_encode_surrogate_diagnostics_without_network(monkeypatch):
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"ok"

    def open_request(request, timeout):
        captured.append(request)
        return Response()

    monkeypatch.setattr(backend_base.urllib.request, "urlopen", open_request)
    payload = {"text": "failed on file\udcff", "normal": "\u4e2d\u6587"}
    assert backend_base.post_json("https://example.invalid/hook", payload, 1.0) == "ok"
    assert json.loads(captured[0].data.decode("utf-8")) == payload


@pytest.mark.skipif(os.name != "posix", reason="filesystem surrogateescape is POSIX-specific")
def test_surrogate_artifact_paths_do_not_collide_with_literal_escape_names(run, tmp_path):
    run()
    first = tmp_path / os.fsdecode(b"data-\xff.bin")
    second = tmp_path / r"data-\udcff.bin"
    first.write_bytes(b"same contents")
    second.write_bytes(b"same contents")
    original = et.log_artifact(str(first), name="bundle")
    escaped = et.log_artifact(str(second), name="bundle")
    assert original.digest != escaped.digest
    assert original.version == 0 and escaped.version == 1


def test_surrogate_alert_text_and_metric_names_survive_identity_hashing(run, collector):
    channel, messages = collector
    run(alert={"channels": [channel()]})
    text = "file\udcff"
    et.alert(text, text=text)
    assert messages[0].title == text and messages[0].text == text
    et.add_alert_rule(M[text] > 0, message=text)
    et.log({text: 1.0})
    assert len(messages) == 2
    assert messages[1].text == text
