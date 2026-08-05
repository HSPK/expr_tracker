"""Data correctness: what goes in comes out, whatever the path it took.

These are model-based checks -- a plain dict of expected state is maintained
alongside the store and compared against every query.
"""

import json
import math
import random

import pytest

from expr_tracker.history import HistoryStore, read_history

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


@pytest.fixture
def make(tmp_path):
    created = []

    def factory(name="run", **options):
        store = HistoryStore()
        options.setdefault("max_open_seconds", None)
        store.init(project="correct", name=name, dir=str(tmp_path), **options)
        created.append(store)
        return store

    yield factory
    for store in created:
        store.finish()


def expected_rows(model: dict[int, dict]) -> list[dict]:
    return [model[step] for step in sorted(model)]


def compare(store, model: dict[int, dict]):
    """Every access path must agree with the model."""
    rows = store.get(-1)
    assert [r["_step"] for r in rows] == sorted(model)
    for row in rows:
        for key, value in model[row["_step"]].items():
            assert row[key] == value, f"step {row['_step']} key {key}"
    return rows


# ------------------------------------------------------------------ round trip


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (-1, -1),
        (2**62, 2**62),
        (1.5, 1.5),
        (-0.0, -0.0),
        (1e308, 1e308),
        (5e-324, 5e-324),
        (True, True),
        (False, False),
        (None, None),
        ("", ""),
        ("unicode 中文 🎉", "unicode 中文 🎉"),
        ('quotes " and \\ backslash', 'quotes " and \\ backslash'),
        ("new\nline\ttab", "new\nline\ttab"),
        ([1, 2, 3], [1, 2, 3]),
        ({"a": {"b": [1, {"c": 2}]}}, {"a": {"b": [1, {"c": 2}]}}),
        ([], []),
        ({}, {}),
    ],
)
def test_values_survive_the_round_trip(make, value, expected):
    store = make()
    store.log({"v": value})
    store.flush(commit_open=True)

    for rows in (store.get(-1), read_history(store.log_dir, -1)):
        assert rows[0]["v"] == expected
        assert type(rows[0]["v"]) is type(expected)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_round_trip(make, value):
    store = make()
    store.log({"v": value})
    store.flush(commit_open=True)
    got = store.get(-1)[0]["v"]
    assert math.isnan(got) if math.isnan(value) else got == value
    assert read_history(store.log_dir, -1)[0]["v"] == pytest.approx(got, nan_ok=True)


def test_non_finite_floats_are_written_as_python_json_literals(make):
    """A deliberate deviation: bare NaN/Infinity keep the signal isnan() needs.

    Python and pandas read this; strict parsers such as jq do not.
    """
    store = make()
    store.log({"a": float("nan"), "b": float("inf"), "c": float("-inf")})
    store.flush(commit_open=True)
    text = store.log_fp.read_text()
    assert '"a": NaN' in text and '"b": Infinity' in text and '"c": -Infinity' in text


@pytest.mark.parametrize(
    "key",
    ["a", "train/loss", "a.b", "with space", "中文", "0", "_leading", "x" * 200, "a-b"],
)
def test_metric_names_survive_the_round_trip(make, key):
    store = make()
    store.log({key: 1.0})
    store.flush(commit_open=True)
    assert store.get(-1)[0][key] == 1.0
    assert read_history(store.log_dir, -1)[0][key] == 1.0


def test_non_string_keys_are_normalised_consistently(make):
    store = make()
    store.log({1: "int", 2.5: "float", None: "none"})  # True would collide with 1
    store.flush(commit_open=True)
    row = store.get(-1)[0]
    disk = read_history(store.log_dir, -1)[0]
    assert {k: v for k, v in row.items() if not k.startswith("_")} == {
        k: v for k, v in disk.items() if not k.startswith("_")
    }
    assert all(isinstance(k, str) for k in row)


def test_large_values_are_not_truncated(make):
    store = make()
    payload = {"blob": "x" * 100_000, "nested": {"deep": list(range(1000))}}
    store.log(payload)
    store.flush(commit_open=True)
    row = store.get(-1)[0]
    assert len(row["blob"]) == 100_000
    assert row["nested"]["deep"] == list(range(1000))


def test_reserved_keys_cannot_be_overwritten(make):
    store = make()
    store.log({"_step": 999, "_time": 0.0, "loss": 1.0}, step=3)
    store.flush(commit_open=True)
    row = store.get(-1)[0]
    assert row["_step"] == 3 and row["_time"] > 0 and row["loss"] == 1.0


# ------------------------------------------------------------------ model based


@pytest.mark.parametrize("seed", SEEDS)
def test_random_commit_patterns_preserve_every_value(make, seed):
    """Any mix of implicit/explicit steps and commits must lose nothing."""
    rng = random.Random(seed)
    store = make(f"r{seed}", cache_bytes=rng.choice([1, 512, 10_000, 1 << 20]))
    model: dict[int, dict] = {}

    for i in range(300):
        payload = {f"m{rng.randrange(4)}": float(i) for _ in range(rng.randint(1, 3))}
        mode = rng.random()
        if mode < 0.5:
            step = store.current_step
            store.log(payload)
        elif mode < 0.8:
            step = store.current_step
            store.log(payload, step=step)
        else:
            step = store.current_step
            store.log(payload, step=step, commit=True)
        model.setdefault(step, {}).update(payload)
        if rng.random() < 0.1:
            store.flush()
        if rng.random() < 0.1:
            compare(store, model)

    store.flush(commit_open=True)
    compare(store, model)
    disk = read_history(store.log_dir, -1)
    assert [r["_step"] for r in disk] == sorted(model)


@pytest.mark.parametrize("seed", SEEDS)
def test_random_queries_agree_with_the_full_history(make, seed):
    rng = random.Random(seed)
    store = make(f"q{seed}", cache_bytes=rng.choice([1, 400, 5_000, 1 << 20]))
    total = 200
    for step in range(total):
        store.log({"loss": float(step), "pad": "y" * rng.randint(1, 30)})
    store.flush(commit_open=True)

    everything = store.get(-1)
    by_step = {r["_step"]: r for r in everything}
    assert len(everything) == total

    for _ in range(40):
        n = rng.choice([1, 2, 5, 13, 50, 199, 200, 500, -1])
        rows = store.get(n)
        expected = everything if n < 0 else everything[-n:]
        assert rows == expected

        start = rng.randrange(-10, total + 10)
        end = start + rng.randrange(0, 40)
        ranged = store.get(-1, step_range=(start, end))
        assert ranged == [
            by_step[s] for s in range(max(0, start), min(end, total)) if s in by_step
        ]


@pytest.mark.parametrize("seed", SEEDS[:4])
def test_out_of_order_writes_merge_deterministically(make, seed):
    rng = random.Random(seed)
    store = make(f"o{seed}", step_policy="allow", cache_bytes=rng.choice([1, 2_000]))
    model: dict[int, dict] = {}

    steps = list(range(50))
    rng.shuffle(steps)
    for step in steps:
        for round_no in range(rng.randint(1, 3)):
            payload = {f"k{round_no}": step * 10 + round_no}
            store.log(payload, step=step, commit=True)
            model.setdefault(step, {}).update(payload)

    store.flush(commit_open=True)
    compare(store, model)
    assert [r["_step"] for r in read_history(store.log_dir, -1)] == sorted(model)


def test_last_write_wins_within_a_step(make):
    store = make()
    store.log({"loss": 1.0}, step=0)
    store.log({"loss": 2.0}, step=0)
    store.log({"loss": 3.0}, step=0, commit=True)
    store.flush(commit_open=True)
    assert store.get(-1) == [{**store.get(-1)[0], "loss": 3.0}]
    assert read_history(store.log_dir, -1)[0]["loss"] == 3.0


def test_a_patch_line_wins_over_the_original(make):
    store = make(step_policy="allow")
    store.log({"a": 1, "b": 1}, step=0, commit=True)
    store.log({"b": 2, "c": 3}, step=0, commit=True)
    store.flush(commit_open=True)
    row = store.get(-1)[0]
    assert (row["a"], row["b"], row["c"]) == (1, 2, 3)
    assert len(store.get(-1)) == 1
    # the file really does hold two physical rows for one step
    lines = store.log_fp.read_text().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["_step"] == 0 for line in lines)


# ------------------------------------------------------------------ ordering


@pytest.mark.parametrize("seed", SEEDS[:4])
def test_results_are_always_sorted_and_unique(make, seed):
    rng = random.Random(seed)
    store = make(f"s{seed}", step_policy="allow", cache_bytes=rng.choice([1, 900]))
    for _ in range(200):
        store.log({"v": 1}, step=rng.randrange(60), commit=rng.random() < 0.5)
    store.flush(commit_open=True)

    for n in (1, 5, 30, -1):
        steps = [r["_step"] for r in store.get(n)]
        assert steps == sorted(set(steps))
    assert [r["_step"] for r in store.get(-1)] == sorted(
        {r["_step"] for r in read_history(store.log_dir, -1)}
    )


def test_time_never_goes_backwards(make):
    store = make()
    for step in range(100):
        store.log({"loss": float(step)})
    store.flush(commit_open=True)
    times = [r["_time"] for r in store.get(-1)]
    assert times == sorted(times)


def test_steps_are_contiguous_for_a_plain_loop(make):
    store = make()
    for step in range(500):
        store.log({"loss": float(step)})
    store.flush(commit_open=True)
    steps = [r["_step"] for r in store.get(-1)]
    assert steps == list(range(500))


# ------------------------------------------------------------------ projection


def test_projection_never_invents_or_loses_values(make):
    store = make()
    for step in range(20):
        payload = {"a": step}
        if step % 3 == 0:
            payload["b"] = step * 2
        store.log(payload)
    store.flush(commit_open=True)

    full = store.get(-1)
    only_b = store.get(-1, metrics=["b"])
    assert len(only_b) == 20
    for row, source in zip(only_b, full, strict=True):
        assert ("b" in row) == ("b" in source)
        if "b" in row:
            assert row["b"] == source["b"]

    filled = store.get(-1, metrics=["b"], fill_missing=True)
    assert all("b" in r for r in filled)
    assert [r["b"] for r in filled if r["_step"] % 3] == [None] * 13

    dropped = store.get(-1, metrics=["b"], dropna=True)
    assert [r["_step"] for r in dropped] == [0, 3, 6, 9, 12, 15, 18]


def test_meta_can_be_excluded_without_touching_metrics(make):
    store = make()
    store.log({"a": 1, "b": 2})
    store.flush(commit_open=True)
    row = store.get(-1, include_meta=False)[0]
    assert row == {"a": 1, "b": 2}


def test_pandas_and_dict_outputs_carry_the_same_data(make):
    pd = pytest.importorskip("pandas")
    store = make()
    for step in range(30):
        store.log({"loss": float(step), "acc": step / 30})
    store.flush(commit_open=True)

    rows = store.get(-1)
    frame = store.get(-1, output_type="pandas")
    assert isinstance(frame, pd.DataFrame)
    assert list(frame["_step"]) == [r["_step"] for r in rows]
    assert list(frame["loss"]) == [r["loss"] for r in rows]
    assert frame.shape == (30, 4)


# ------------------------------------------------------------------ series


def test_the_metric_series_matches_the_history(make):
    store = make()
    for step in range(50):
        payload = {"dense": float(step)}
        if step % 5 == 0:
            payload["sparse"] = float(step)
        store.log(payload)
    store.flush(commit_open=True)

    rows = store.get(-1)
    dense = [(r["_step"], r["dense"]) for r in rows]
    sparse = [(r["_step"], r["sparse"]) for r in rows if "sparse" in r]
    assert [(p[0], p[2]) for p in store.series.points("dense")] == dense
    assert [(p[0], p[2]) for p in store.series.points("sparse")] == sparse


def test_the_series_only_holds_its_window(make):
    store = make(alert_window=10)
    for step in range(100):
        store.log({"loss": float(step)})
    store.flush(commit_open=True)
    points = store.series.points("loss")
    assert len(points) <= 10
    assert points[-1][2] == 99.0
    assert len(store.get(-1)) == 100  # the history is unaffected


# ------------------------------------------------------------------ durability


@pytest.mark.parametrize("seed", SEEDS[:4])
def test_the_file_always_matches_the_store(make, seed):
    rng = random.Random(seed)
    store = make(f"d{seed}", cache_bytes=rng.choice([1, 700, 1 << 20]))
    for step in range(150):
        store.log({"loss": float(step), "pad": "z" * rng.randint(1, 20)})
        if rng.random() < 0.15:
            store.flush()
    store.flush(commit_open=True)

    from_memory = store.get(-1)
    from_disk = read_history(store.log_dir, -1)
    assert from_memory == from_disk
    for line in store.log_fp.read_text().splitlines():
        assert isinstance(json.loads(line)["_step"], int)


def test_a_reopened_store_sees_exactly_what_was_written(make, tmp_path):
    store = make("reopen")
    for step in range(40):
        store.log({"loss": float(step)})
    store.flush(commit_open=True)
    before = store.get(-1)
    store.finish()

    reopened = HistoryStore()
    reopened.init(project="correct", name="reopen", dir=str(tmp_path))
    try:
        assert reopened.get(-1) == before
        assert reopened.current_step == 40
        reopened.log({"loss": 40.0})
        reopened.flush(commit_open=True)
        assert [r["_step"] for r in reopened.get(-1)] == list(range(41))
    finally:
        reopened.finish()
