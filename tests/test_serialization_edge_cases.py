"""Serialization contracts for scientific values, model options and sparse frames."""

import datetime
import json
import math
import re
import sys
from collections import UserDict, deque
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from types import MappingProxyType

import pytest
from loguru import logger
from pydantic import BaseModel, Field, RootModel, model_serializer
from pydantic_core import PydanticUndefined

from expr_tracker._compat import _model_dump
from expr_tracker.encoders import decimal_encoder, jsonable_encoder
from expr_tracker.history.codec import (
    MAX_FALLBACK_REPR_LENGTH,
    RESERVED_KEYS,
    RecordCodec,
    encode_key,
    encode_line,
    fallback_repr,
)
from expr_tracker.history.frame import project, to_output


class Unsupported:
    __slots__ = ()

    def __repr__(self):
        return "<unsupported>"


@pytest.fixture
def warnings():
    messages = []
    sink = logger.add(lambda message: messages.append(message.record), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink)


class ModelOptions(BaseModel):
    rate: float = Field(default=0.1, alias="learning_rate")
    count: int = 1
    note: str | None = None


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, {"learning_rate": 0.1, "count": 2, "note": None}),
        ({"by_alias": False}, {"rate": 0.1, "count": 2, "note": None}),
        ({"include": ["rate", "count"]}, {"learning_rate": 0.1, "count": 2}),
        ({"exclude": ["rate", "note"]}, {"count": 2}),
        ({"exclude_unset": True}, {"count": 2}),
        ({"exclude_defaults": True}, {"count": 2}),
        ({"exclude_none": True}, {"learning_rate": 0.1, "count": 2}),
    ],
)
def test_model_serialization_options(options, expected):
    assert jsonable_encoder(ModelOptions(count=2), **options) == expected


@pytest.mark.parametrize(
    "wrap",
    [lambda model: model, lambda model: [model], lambda model: {"config": model}],
    ids=["model", "list", "mapping"],
)
def test_exclude_defaults_reaches_nested_models(wrap):
    assert jsonable_encoder(wrap(ModelOptions(count=2)), exclude_defaults=True) == wrap(
        {"count": 2}
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (None, None),
        (False, False),
        ("__root__", "__root__"),
        (["__root__"], ["__root__"]),
        ({"__root__": 1, "other": 2}, {"__root__": 1, "other": 2}),
        (datetime.date(2026, 1, 2), "2026-01-02"),
    ],
)
def test_pydantic_root_models_keep_their_shape(value, expected):
    model = RootModel[type(value)](value)
    assert jsonable_encoder(model) == expected
    assert json.loads(encode_line(RecordCodec().encode({"model": model}))) == {
        "model": expected
    }


def test_pydantic_model_serializer_can_return_a_scalar():
    class Token(BaseModel):
        value: int

        @model_serializer
        def serialize(self) -> int:
            return self.value

    assert jsonable_encoder(Token(value=7)) == 7


def test_model_dump_preserves_python_mode_and_passes_options():
    class Dated(BaseModel):
        day: datetime.date = Field(alias="date")
        note: str | None = None

    model = Dated(date=datetime.date(2026, 1, 2))
    assert _model_dump(model, by_alias=True, exclude_none=True) == {
        "date": "2026-01-02"
    }
    assert _model_dump(model, mode="python", include={"day"}) == {
        "day": datetime.date(2026, 1, 2)
    }


def test_undefined_and_dataclass_options():
    @dataclass
    class Settings:
        path: PurePath
        count: int
        note: str | None = None

    value = Settings(PurePath("model.bin"), 2)
    assert jsonable_encoder(PydanticUndefined) is None
    assert jsonable_encoder(value, include=["path", "note"], exclude_none=True) == {
        "path": "model.bin"
    }
    assert jsonable_encoder(value, exclude=["path", "note"]) == {"count": 2}


@pytest.mark.parametrize("wrap", [dict, UserDict, MappingProxyType])
def test_mapping_filters_are_applied_without_mutating_input(wrap):
    original = {"keep": 1, "skip": 2, "none": None, "_sa_instance_state": 3}
    value = wrap(original)
    assert jsonable_encoder(value, include=["keep", "skip"], exclude=["skip"]) == {
        "keep": 1
    }
    assert jsonable_encoder(value, exclude_none=True) == {"keep": 1, "skip": 2}
    assert jsonable_encoder(value, sqlalchemy_safe=False) == original
    assert dict(value) == original


def test_mapping_keys_and_custom_encoder_precedence():
    class Metric:
        def __init__(self, value):
            self.value = value

    class DerivedMetric(Metric):
        pass

    encoders = {
        int: lambda value: f"integer:{value}",
        Metric: lambda value: f"base:{value.value}",
        DerivedMetric: lambda value: f"derived:{value.value}",
    }
    assert jsonable_encoder(DerivedMetric(2), custom_encoder=encoders) == "derived:2"
    assert (
        jsonable_encoder(
            DerivedMetric(2), custom_encoder={str: str.upper, Metric: lambda v: v.value}
        )
        == 2
    )
    assert jsonable_encoder(Metric(3), custom_encoder={str: str.upper}) == {"VALUE": 3}
    assert jsonable_encoder({1: Metric(3)}, custom_encoder=encoders) == {
        "integer:1": "base:3"
    }


@pytest.mark.parametrize(
    "wrap",
    [list, tuple, set, frozenset, deque, lambda values: (v for v in values)],
    ids=["list", "tuple", "set", "frozenset", "deque", "generator"],
)
def test_containers_recursively_encode_their_items(wrap):
    day = datetime.date(2026, 1, 2)
    assert jsonable_encoder(wrap([day])) == ["2026-01-02"]


def test_builtin_encoder_lookup_and_subclasses():
    class Bytes(bytes):
        pass

    assert jsonable_encoder(Bytes(b"model")) == "model"
    assert jsonable_encoder(re.compile("loss.*")) == "loss.*"
    assert decimal_encoder(Decimal(123456789012345678901234567890)) == (
        123456789012345678901234567890
    )
    assert type(decimal_encoder(Decimal("1.0"))) is float


def test_enum_values_are_recursively_json_encoded():
    class Value(Enum):
        PATH = PurePath("model.bin")
        DAY = datetime.date(2026, 1, 2)
        PAIR = (1, 2)

    record = RecordCodec().encode({item.name: item for item in Value})
    assert json.loads(encode_line(record)) == {
        "PATH": "model.bin",
        "DAY": "2026-01-02",
        "PAIR": [1, 2],
    }


def test_unsupported_values_raise_useful_encoder_errors():
    with pytest.raises(ValueError) as error:
        jsonable_encoder(Unsupported())
    assert len(error.value.args[0]) == 2
    assert all(isinstance(cause, TypeError) for cause in error.value.args[0])
    assert isinstance(error.value.__cause__, TypeError)


def test_codec_does_not_consume_good_values_twice_on_field_failure(warnings):
    codec = RecordCodec()
    record = codec.encode(
        {"samples": (value for value in [1, 2, 3]), "bad": Unsupported(), "loss": 0.5}
    )
    assert record == {"samples": [1, 2, 3], "bad": "<unsupported>", "loss": 0.5}
    assert len(warnings) == 1
    assert json.loads(encode_line(record)) == record


def test_codec_field_failure_does_not_disable_sqlalchemy_filtering():
    codec = RecordCodec()
    values = {"_sa_instance_state": object(), "loss": 0.5, "_step": 2, "_time": 3.0}
    assert codec.encode(values) == {"loss": 0.5, "_step": 2, "_time": 3.0}
    values["bad"] = Unsupported()
    assert codec.encode(values) == {
        "loss": 0.5,
        "_step": 2,
        "_time": 3.0,
        "bad": "<unsupported>",
    }
    assert frozenset({"_step", "_time"}) == RESERVED_KEYS
    assert codec.encode({b"_step": 2, b"_sa_label": 3}) == {
        "_step": 2,
        "_sa_label": 3,
    }


@pytest.mark.parametrize("value", [False, 0, 0.0, "", [], ()])
def test_codec_preserves_falsey_non_mapping_values(value):
    assert RecordCodec().encode(value) == {"value": jsonable_encoder(value)}


def test_codec_empty_and_non_dict_mapping_contract():
    codec = RecordCodec()
    assert codec.encode(None) == {}
    assert codec.encode({}) == {}
    assert codec.encode(UserDict({"loss": 1})) == {"value": {"loss": 1}}


def test_failed_key_encoding_uses_repr_not_the_none_key():
    key = Unsupported()
    assert encode_key(key) == "<unsupported>"
    assert RecordCodec().encode({key: 1, None: 2}) == {
        "<unsupported>": 1,
        "None": 2,
    }


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (None, "None"),
        (True, "True"),
        (1.5, "1.5"),
        (b"loss", "loss"),
        (PurePath("model.bin"), "model.bin"),
        ((1, 2), "(1, 2)"),
    ],
)
def test_key_encoding_keeps_existing_conventions(key, expected):
    assert encode_key(key) == expected


def test_codec_warnings_are_per_key_and_resettable(warnings):
    codec = RecordCodec()
    for _ in range(2):
        assert codec.encode({"bad": Unsupported()}, kind="config") == {
            "bad": "<unsupported>"
        }
    codec.encode({"other": Unsupported()})
    assert len(warnings) == 2
    assert "Config 'bad'" in warnings[0]["message"]
    codec.reset()
    codec.encode({"bad": Unsupported()})
    assert len(warnings) == 3


def test_repr_failure_and_truncation_are_bounded():
    class BrokenRepr:
        __slots__ = ()

        def __repr__(self):
            raise RuntimeError("repr unavailable")

    assert RecordCodec().encode({"bad": BrokenRepr()}) == {
        "bad": "<unrepresentable BrokenRepr>"
    }
    text = "x" * MAX_FALLBACK_REPR_LENGTH
    assert fallback_repr(text) == repr(text)[:MAX_FALLBACK_REPR_LENGTH] + "..."
    assert fallback_repr("short") == "'short'"


def test_numpy_temporal_and_object_arrays_preserve_nested_shape():
    np = pytest.importorskip("numpy")
    value = np.array(
        [[np.datetime64("2026-01-02"), np.datetime64("NaT")]], dtype=object
    )
    assert jsonable_encoder(value) == [["2026-01-02", None]]
    assert jsonable_encoder(np.timedelta64(2, "s")) == 2.0
    assert jsonable_encoder(np.array(7)) == 7


def test_codec_can_encode_a_top_level_numpy_array():
    np = pytest.importorskip("numpy")
    assert RecordCodec().encode(np.array([1, 2])) == {"value": [1, 2]}


def test_numpy_extended_scalar_conversion_fails_without_recursing(warnings):
    np = pytest.importorskip("numpy")
    value = np.clongdouble(1 + 2j)
    with pytest.raises(ValueError):
        jsonable_encoder(value)
    assert RecordCodec().encode({"complex": value}) == {"complex": repr(value)}
    assert len(warnings) == 1


def test_nonfinite_jsonl_semantics_are_unchanged():
    np = pytest.importorskip("numpy")
    codec = RecordCodec()
    encoded = codec.encode(
        {
            "nan": np.float32("nan"),
            "pos": np.float64("inf"),
            "neg": -math.inf,
            "none": None,
            "text": "line one\nline two",
        }
    )
    line = encode_line(encoded)
    assert line.count(b"\n") == 1
    assert b"NaN" in line and b"Infinity" in line and b"-Infinity" in line
    decoded = json.loads(line)
    assert math.isnan(decoded["nan"])
    assert decoded["pos"] == math.inf and decoded["neg"] == -math.inf
    assert decoded["none"] is None
    assert decoded["text"] == "line one\nline two"


@pytest.mark.parametrize("value", [10**1000, -(10**1000)])
def test_huge_integers_remain_exact_even_when_a_neighbor_needs_fallback(
    value, warnings
):
    record = RecordCodec().encode(
        {"huge": value, "bad": Unsupported(), "good": 0.5, "nested": [value]}
    )
    decoded = json.loads(encode_line(record))
    assert type(decoded["huge"]) is int
    assert decoded == {
        "huge": value,
        "bad": "<unsupported>",
        "good": 0.5,
        "nested": [value],
    }
    assert encode_key(value) == str(value)
    assert len(warnings) == 1 and "'bad'" in warnings[0]["message"]


@pytest.mark.parametrize("library", ["pandas", "polars"])
@pytest.mark.parametrize("value", [10**1000, -(10**1000)])
def test_frame_integer_overflow_is_explicit_and_does_not_mutate_records(library, value):
    pytest.importorskip(library)
    rows = [{"huge": value, "good": 0.5}]
    with pytest.raises(OverflowError, match="too large"):
        to_output(rows, library)
    assert to_output(rows, "dict") is rows
    assert json.loads(encode_line(rows[0])) == {"huge": value, "good": 0.5}


def test_pandas_nullable_and_temporal_values_become_json_values():
    pd = pytest.importorskip("pandas")
    values = {
        "missing": pd.NA,
        "time": pd.NaT,
        "series": pd.Series([1, None], dtype="Int64"),
        "timestamp": pd.Timestamp("2026-01-02T03:04:05"),
    }
    assert json.loads(encode_line(RecordCodec().encode(values))) == {
        "missing": None,
        "time": None,
        "series": [1, None],
        "timestamp": "2026-01-02T03:04:05",
    }


@pytest.mark.parametrize("library", ["pandas", "polars"])
@pytest.mark.parametrize("size", [0, 1, 2, 3])
def test_scientific_frames_and_series_keep_column_oriented_shape(library, size):
    module = pytest.importorskip(library)
    columns = {"loss": list(range(size)), "label": ["eval"] * size}
    frame = module.DataFrame(columns)
    assert jsonable_encoder(frame) == columns
    assert jsonable_encoder(frame["loss"]) == columns["loss"]
    assert jsonable_encoder(frame, include=["loss"]) == {"loss": columns["loss"]}


@pytest.mark.parametrize("fill_missing", [False, True])
def test_projection_dropna_means_absent_not_none_or_nonfinite(fill_missing):
    rows = [
        {"_step": 0, "_time": 1.0},
        {"_step": 1, "_time": 2.0, "loss": None},
        {"_step": 2, "_time": 3.0, "loss": math.nan},
        {"_step": 3, "_time": 4.0, "loss": math.inf},
    ]
    result = project(rows, metrics=["loss"], dropna=True, fill_missing=fill_missing)
    assert [row["_step"] for row in result] == [1, 2, 3]
    assert result[0]["loss"] is None
    assert math.isnan(result[1]["loss"])
    assert result[2]["loss"] == math.inf
    assert "loss" not in rows[0]


def test_projection_fill_order_and_explicit_metadata_selection():
    rows = [{"z": 1, "_time": 1.0}, {"a": 2, "_step": 3}]
    filled = project(rows, fill_missing=True)
    assert list(filled[0]) == ["_step", "_time", "z", "a"]
    assert filled[0] == {"_step": None, "_time": 1.0, "z": 1, "a": None}
    assert project(rows, metrics=["_step"], include_meta=False) == [{}, {"_step": 3}]
    assert project(rows, metrics=[], include_meta=False) == [{}, {}]
    assert project(rows, metrics=[], dropna=True) == []
    assert rows == [{"z": 1, "_time": 1.0}, {"a": 2, "_step": 3}]


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_sparse_frame_columns_follow_first_seen_order_after_metadata(library):
    pytest.importorskip(library)
    rows = [{"z": 1, "_time": 1.0}, {"a": 2, "_step": 3}]
    frame = to_output(rows, library)
    assert list(frame.columns) == ["_step", "_time", "z", "a"]
    assert frame.shape == (2, 4)
    assert frame["z"][0] == 1 and frame["a"][1] == 2


@pytest.mark.parametrize(
    ("prefix", "last", "expected"),
    [(None, 1, 1), (1, 1.5, 1.5), (None, "late", "late")],
)
def test_polars_infers_sparse_columns_using_all_rows(prefix, last, expected):
    pytest.importorskip("polars")
    rows = [{"_step": step, "metric": prefix} for step in range(100)]
    rows.append({"_step": 100, "metric": last})
    frame = to_output(rows, "polars")
    assert frame.shape == (101, 2)
    assert frame["metric"][-1] == expected


def test_polars_keeps_none_distinct_from_nonfinite_values():
    pytest.importorskip("polars")
    frame = to_output(
        [{"loss": None}, {"loss": math.nan}, {"loss": math.inf}, {"loss": -math.inf}],
        "polars",
    )
    assert frame["loss"].null_count() == 1
    assert math.isnan(frame["loss"][1])
    assert frame["loss"][2] == math.inf and frame["loss"][3] == -math.inf


def test_polars_incompatible_column_values_raise_instead_of_being_dropped():
    pl = pytest.importorskip("polars")
    with pytest.raises(pl.exceptions.SchemaError):
        to_output([{"value": [1]}, {"value": {"x": 2}}], "polars")


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_optional_frame_dependency_errors_are_actionable(monkeypatch, library):
    monkeypatch.setitem(sys.modules, library, None)
    with pytest.raises(ImportError, match=rf"expr_tracker\[{library}\]") as error:
        to_output([{"loss": 1}], library)
    assert isinstance(error.value.__cause__, ImportError)
    rows = [{"loss": 1}]
    assert to_output(rows, "dict") is rows
    assert RecordCodec().encode(rows[0]) == rows[0]
