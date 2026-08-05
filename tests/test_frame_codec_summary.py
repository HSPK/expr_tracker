"""Unit tests for output conversion, value encoding and the run summary."""

import json
import math

import pytest

from expr_tracker.history.codec import (
    RecordCodec,
    encode_key,
    encode_line,
    fallback_repr,
)
from expr_tracker.history.frame import project, to_output
from expr_tracker.summary import Summary

ROWS = [
    {"_step": 0, "_time": 1.0, "loss": 1.0},
    {"_step": 1, "_time": 2.0, "acc": 0.5},
    {"_step": 2, "_time": 3.0, "loss": 0.5, "acc": 0.9},
]


# ---------------------------------------------------------------- projection


def test_project_without_selection_keeps_every_key():
    assert project(ROWS) == ROWS


def test_project_can_drop_metadata():
    rows = project(ROWS, include_meta=False)
    assert rows[0] == {"loss": 1.0}
    assert rows[1] == {"acc": 0.5}


def test_project_selects_columns_and_keeps_metadata_first():
    rows = project(ROWS, metrics=["loss"])
    assert list(rows[0]) == ["_step", "_time", "loss"]
    assert rows[1] == {"_step": 1, "_time": 2.0}  # metric missing for that step


def test_project_dropna_removes_rows_without_the_selection():
    rows = project(ROWS, metrics=["loss"], dropna=True)
    assert [r["_step"] for r in rows] == [0, 2]


def test_project_fill_missing_with_and_without_selection():
    selected = project(ROWS, metrics=["loss", "acc"], fill_missing=True)
    assert selected[0] == {"_step": 0, "_time": 1.0, "loss": 1.0, "acc": None}
    everything = project(ROWS, fill_missing=True)
    assert everything[0] == {"_step": 0, "_time": 1.0, "loss": 1.0, "acc": None}
    assert list(everything[1]) == ["_step", "_time", "loss", "acc"]


def test_project_on_empty_input():
    assert project([], metrics=["loss"], fill_missing=True) == []


# ---------------------------------------------------------------- output types


@pytest.mark.parametrize("name", ["dict", "dicts", "records", "list"])
def test_dict_aliases_return_the_rows_unchanged(name):
    assert to_output(ROWS, name) is ROWS


@pytest.mark.parametrize("name", ["pd", "pandas", "dataframe", "df"])
def test_pandas_aliases(name):
    pytest.importorskip("pandas")
    frame = to_output(ROWS, name)
    assert list(frame.columns) == ["_step", "_time", "loss", "acc"]
    assert len(frame) == 3
    assert math.isnan(frame["acc"][0])


@pytest.mark.parametrize("name", ["pl", "polars"])
def test_polars_aliases(name):
    pl = pytest.importorskip("polars")
    frame = to_output(ROWS, name)
    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == ["_step", "_time", "loss", "acc"]
    assert frame.height == 3


def test_empty_frames():
    pytest.importorskip("pandas")
    pytest.importorskip("polars")
    assert to_output([], "pandas").empty
    assert to_output([], "polars").height == 0


def test_unknown_output_type_names_the_valid_ones():
    with pytest.raises(ValueError, match="Unsupported output_type"):
        to_output(ROWS, "parquet")


# ---------------------------------------------------------------- codec


def test_codec_encodes_nested_and_exotic_values():
    codec = RecordCodec()
    encoded = codec.encode(
        {"a": {"b": [1, 2]}, "t": (1, 2), "s": {"x"}, "n": None, "f": 1.5}
    )
    assert encoded["a"] == {"b": [1, 2]}
    assert encoded["t"] == [1, 2]
    assert sorted(encoded["s"]) == ["x"]
    json.dumps(encoded)  # must be serialisable


def test_codec_falls_back_to_repr_and_warns_once(caplog):
    class Weird:
        __slots__ = ()

        def __repr__(self):
            return "<weird>"

    codec = RecordCodec()
    for _ in range(3):
        assert codec.encode({"bad": Weird()}) == {"bad": "<weird>"}
    assert sum("not JSON serializable" in r.message for r in caplog.records) <= 1


def test_codec_handles_non_mapping_input():
    assert RecordCodec().encode(5) == {"value": 5}
    assert RecordCodec().encode(None) == {}
    assert RecordCodec().encode({}) == {}


def test_codec_normalises_non_string_keys():
    encoded = RecordCodec().encode({1: "a", None: "b", (2, 3): "c", "ok": "d"})
    assert all(isinstance(key, str) for key in encoded)
    assert encoded["ok"] == "d"


def test_codec_reset_clears_the_warned_set():
    codec = RecordCodec()
    assert codec.warn_once("k", "first") is True
    assert codec.warn_once("k", "second") is False
    codec.reset()
    assert codec.warn_once("k", "third") is True


def test_encode_key_and_fallback_repr():
    assert encode_key("a") == "a"
    assert encode_key(1) == "1"
    assert encode_key(None) == "None"

    class Boom:
        def __repr__(self):
            raise RuntimeError("no repr")

    assert fallback_repr(Boom()).startswith("<unrepresentable")
    assert len(fallback_repr("x" * 5000)) < 600


def test_encode_line_round_trip():
    line = encode_line({"_step": 1, "名字": "值"})
    assert line.endswith(b"\n")
    assert json.loads(line)["名字"] == "值"


# ---------------------------------------------------------------- summary


def test_summary_without_a_path_is_usable():
    summary = Summary()
    summary.observe({"loss": 1.0, "_step": 3})
    assert dict(summary) == {"loss": 1.0}
    summary.save()  # must be a no-op, not an error


def test_summary_ignores_non_mapping_input():
    summary = Summary()
    summary.observe([1, 2, 3])
    assert dict(summary) == {}


def test_summary_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text("{not json}")
    assert dict(Summary(path)) == {}


def test_summary_ignores_a_non_object_file(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text("[1, 2]")
    assert dict(Summary(path)) == {}


def test_summary_encodes_exotic_values_on_save(tmp_path):
    path = tmp_path / "summary.json"
    summary = Summary(path)
    summary["set"] = {1}
    summary["obj"] = object()
    summary.save()
    reloaded = json.loads(path.read_text())
    assert reloaded["set"] == [1]
    assert isinstance(reloaded["obj"], str)


def test_summary_repr_and_deletion():
    summary = Summary()
    summary["a"] = 1
    assert "a" in repr(summary)
    del summary["a"]
    assert dict(summary) == {}
    with pytest.raises(KeyError):
        del summary["a"]
