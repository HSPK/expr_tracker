"""What survives the trip to disk, and what comes back out of a query.

The tracker is used from ML code, so the values it must handle are numpy scalars
and arrays, pydantic models, paths, enums and timestamps -- not just floats.
"""

import datetime
import decimal
import enum
import math
import pathlib

import pytest

import expr_tracker as et
from expr_tracker.history import HistoryStore, read_history

np = pytest.importorskip("numpy")


@pytest.fixture
def store(tmp_path):
    created = []

    def factory(**options):
        instance = HistoryStore()
        options.setdefault("max_open_seconds", None)
        instance.init(
            project="enc", name=options.pop("name", "r"), dir=str(tmp_path), **options
        )
        created.append(instance)
        return instance

    yield factory
    for instance in created:
        instance.finish()


def round_trip(store, value):
    """The value as the cache returns it and as the file returns it."""
    store.log({"v": value})
    store.flush(commit_open=True)
    from_memory = store.get(-1)[-1]["v"]
    from_disk = read_history(store.log_dir, -1)[-1]["v"]
    assert from_memory == from_disk or (
        isinstance(from_memory, float) and math.isnan(from_memory)
    )
    return from_memory


# ------------------------------------------------------------------ numpy


@pytest.mark.parametrize(
    ("value", "expected", "kind"),
    [
        (np.int8(7), 7, int),
        (np.int32(7), 7, int),
        (np.int64(7), 7, int),
        (np.uint8(255), 255, int),
        (np.float16(0.5), 0.5, float),
        (np.float32(1.5), 1.5, float),
        (np.float64(2.5), 2.5, float),
        (np.bool_(True), True, bool),
        (np.bool_(False), False, bool),
    ],
)
def test_numpy_scalars_become_python_scalars(store, value, expected, kind):
    got = round_trip(store(), value)
    assert got == expected
    assert isinstance(got, kind)


def test_a_numpy_array_becomes_a_list(store):
    assert round_trip(store(), np.array([1.0, 2.0, 3.0])) == [1.0, 2.0, 3.0]


def test_a_2d_numpy_array_keeps_its_shape(store):
    assert round_trip(store(), np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]


def test_an_empty_numpy_array(store):
    assert round_trip(store(), np.array([])) == []


def test_a_numpy_scalar_from_a_reduction(store):
    """``arr.mean()`` is the single most common thing a training loop logs."""
    assert round_trip(store(), np.array([1.0, 2.0, 3.0]).mean()) == 2.0


@pytest.mark.parametrize("value", [np.float32("nan"), np.float64("nan")])
def test_numpy_nan_stays_nan(store, value):
    assert math.isnan(round_trip(store(), value))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(np.float32("inf"), math.inf), (np.float64("-inf"), -math.inf)],
)
def test_numpy_infinities_survive(store, value, expected):
    assert round_trip(store(), value) == expected


def test_numpy_values_are_visible_to_alert_rules(tmp_path):
    received: list = []
    run = et.init(
        project="enc",
        name="numpy-alert",
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
        alert_rules=["loss > 10 => error: numpy value {loss:.1f}"],
    )
    try:
        run.log({"loss": np.float32(50.0)})
        assert len(received) == 1 and "50.0" in received[0].text
    finally:
        et.finish()


# ------------------------------------------------------------------ stdlib


def test_a_decimal_becomes_a_float(store):
    assert round_trip(store(), decimal.Decimal("1.25")) == 1.25


def test_a_datetime_becomes_an_iso_string(store):
    got = round_trip(store(), datetime.datetime(2024, 1, 2, 3, 4, 5))
    assert got == "2024-01-02T03:04:05"


def test_a_date_becomes_an_iso_string(store):
    assert round_trip(store(), datetime.date(2024, 1, 2)) == "2024-01-02"


def test_a_timedelta_becomes_seconds(store):
    assert round_trip(store(), datetime.timedelta(minutes=2)) == 120.0


def test_a_path_becomes_a_string(store):
    assert round_trip(store(), pathlib.Path("/tmp/model.bin")) == "/tmp/model.bin"


def test_an_enum_becomes_its_value(store):
    class Stage(enum.Enum):
        TRAIN = "train"

    assert round_trip(store(), Stage.TRAIN) == "train"


def test_an_int_enum_becomes_its_number(store):
    class Level(enum.IntEnum):
        HIGH = 3

    assert round_trip(store(), Level.HIGH) == 3


def test_bytes_become_text(store):
    assert round_trip(store(), b"raw") == "raw"


@pytest.mark.parametrize(
    ("value", "expected"),
    [({1, 2}, [1, 2]), (frozenset({3}), [3]), ((1, 2), [1, 2])],
)
def test_collections_become_lists(store, value, expected):
    assert sorted(round_trip(store(), value)) == expected


def test_a_complex_number_falls_back_to_repr(store):
    assert round_trip(store(), complex(1, 2)) == "(1+2j)"


def test_a_nested_structure_keeps_its_shape(store):
    value = {"a": [1, {"b": (2, 3)}], "c": {"d": np.float32(1.5)}}
    assert round_trip(store(), value) == {"a": [1, {"b": [2, 3]}], "c": {"d": 1.5}}


# ------------------------------------------------------------------ pydantic


class Config(pytest.importorskip("pydantic").BaseModel):
    lr: float = 0.1
    arch: str = "resnet"
    layers: list[int] = [1, 2]  # noqa: RUF012 - pydantic copies defaults


def test_a_pydantic_model_is_logged_as_a_dict(store):
    assert round_trip(store(), Config()) == {
        "lr": 0.1,
        "arch": "resnet",
        "layers": [1, 2],
    }


def test_a_nested_pydantic_model(store):
    from pydantic import BaseModel

    class Outer(BaseModel):
        inner: Config = Config()
        name: str = "outer"

    assert round_trip(store(), Outer())["inner"]["arch"] == "resnet"


def test_a_pydantic_model_works_as_the_run_config(tmp_path):
    import json

    run = et.init(
        project="enc",
        name="pyd",
        dir=str(tmp_path),
        backends=[],
        max_open_seconds=None,
        config=Config(lr=0.5),
    )
    try:
        run.log({"loss": 1.0})
        saved = json.loads((run.history.log_dir / "config.json").read_text())
        assert saved["lr"] == 0.5 and saved["arch"] == "resnet"
    finally:
        et.finish()


def test_a_pydantic_model_in_the_summary(tmp_path):
    import json

    run = et.init(
        project="enc",
        name="pydsum",
        dir=str(tmp_path),
        backends=[],
        max_open_seconds=False or None,
    )
    try:
        run.summary["config"] = Config()
        run.log({"loss": 1.0})
    finally:
        et.finish()
    saved = json.loads((tmp_path / "enc" / "pydsum" / "summary.json").read_text())
    assert saved["config"]["arch"] == "resnet"


# ------------------------------------------------------------------ keys


@pytest.mark.parametrize("key", [1, 2.5, None, (1, 2)])
def test_non_string_keys_are_stringified(store, key):
    instance = store()
    instance.log({key: 1.0})
    instance.flush(commit_open=True)
    row = instance.get(-1)[0]
    assert all(isinstance(k, str) for k in row)
    assert 1.0 in row.values()


def test_a_key_colliding_after_stringification_keeps_one_value(store):
    instance = store()
    instance.log({1: "int", "1": "str"})
    instance.flush(commit_open=True)
    row = instance.get(-1)[0]
    assert row["1"] in ("int", "str")


# ------------------------------------------------------------------ outputs


@pytest.fixture
def populated(tmp_path):
    run = et.init(
        project="enc", name="out", dir=str(tmp_path), backends=[], max_open_seconds=None
    )
    for step in range(10):
        run.log({"loss": float(step), "lr": 0.1})
    yield run
    if et.get_run() is not None:
        et.finish()


@pytest.mark.parametrize("name", ["pandas", "pd"])
def test_pandas_output_end_to_end(populated, name):
    pd = pytest.importorskip("pandas")
    frame = et.history(-1, output_type=name)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame["loss"]) == [float(i) for i in range(10)]
    assert list(frame.columns) == ["_step", "_time", "loss", "lr"]


@pytest.mark.parametrize("name", ["polars", "pl"])
def test_polars_output_end_to_end(populated, name):
    pl = pytest.importorskip("polars")
    frame = et.history(-1, output_type=name)
    assert isinstance(frame, pl.DataFrame)
    assert frame["loss"].to_list() == [float(i) for i in range(10)]
    assert frame.shape == (10, 4)


def test_a_frame_of_a_selected_metric(populated):
    pytest.importorskip("polars")
    frame = et.history(-1, output_type="polars", metrics=["loss"], include_meta=False)
    assert frame.columns == ["loss"]


def test_an_empty_result_still_produces_a_frame(tmp_path):
    pytest.importorskip("polars")
    et.init(project="enc", name="empty", dir=str(tmp_path), backends=[])
    try:
        frame = et.history(-1, output_type="polars")
        assert frame.shape[0] == 0
    finally:
        et.finish()


@pytest.mark.parametrize("name", ["numpy", "arrow", "DICT ", "pandas!", "list "])
def test_an_unknown_output_type_lists_the_valid_ones(populated, name):
    with pytest.raises(ValueError, match="Unsupported output_type") as excinfo:
        et.history(-1, output_type=name)
    assert "polars" in str(excinfo.value)


@pytest.mark.parametrize("name", ["", "dict", "dicts", "records", "list", "DICT"])
def test_falsy_and_dict_aliases_return_plain_rows(populated, name):
    rows = et.history(-1, output_type=name)
    assert isinstance(rows, list) and isinstance(rows[0], dict)


# ------------------------------------------------------------------ query bounds


def test_n_none_returns_everything(populated):
    assert len(et.history(None)) == 10


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((0, 10), list(range(10))),
        ((0, 1), [0]),
        ((9, 10), [9]),
        ((5, 5), []),
        ((8, 3), []),
        ((-5, 3), [0, 1, 2]),
        ((-10, -1), []),
        ((20, 30), []),
        ((7, 100), [7, 8, 9]),
        ((None, 2), [0, 1]),
        ((7, None), [7, 8, 9]),
        ((None, None), list(range(10))),
    ],
)
def test_step_range_bounds(populated, bounds, expected):
    assert [r["_step"] for r in et.history(-1, step_range=bounds)] == expected


def test_a_step_range_is_capped_by_n(populated):
    assert [r["_step"] for r in et.history(2, step_range=(0, 10))] == [8, 9]


@pytest.mark.parametrize("n", [-100, -1])
def test_any_negative_n_means_everything(populated, n):
    assert len(et.history(n)) == 10


def test_zero_returns_nothing(populated):
    assert et.history(0) == []
