"""Step identifiers normalize before serialization or state mutation."""

import numpy as np
import pytest


class StepIndex:
    def __index__(self):
        return 5


@pytest.mark.parametrize(
    ("given", "expected"),
    [(0, 0), (np.int64(3), 3), (np.uint32(4), 4), (StepIndex(), 5), (True, 1), (np.array(6), 6)],
)
def test_integer_index_steps_are_persisted_as_native_integers(store, given, expected):
    history = store()
    assert history.log({"value": 1}, step=given, commit=True) == expected
    row = history.get(1)[0]
    assert type(row["_step"]) is int
    assert row["_step"] == expected
    assert history.current_step == expected + 1
    assert history.log({"next": 2}) == expected + 1


@pytest.mark.parametrize(
    "step",
    [3.0, "3", b"3", [], np.float64(3), object(), float("nan"), np.array([1, 2])],
    ids=["float", "string", "bytes", "list", "numpy-float", "object", "nan", "array"],
)
def test_invalid_steps_do_not_poison_an_existing_open_row(store, step):
    history = store()
    history.log({"original": 1}, commit=False)
    before = history.open_record()
    with pytest.raises(TypeError, match="step"):
        history.log({"bad": 1}, step=step)
    assert history.open_record() == before
    assert history.current_step == 0
    history.log({"good": 2})
    row = history.get(1)[0]
    assert row["original"] == 1 and row["good"] == 2
    assert "bad" not in row


def test_encoded_ingress_also_validates_steps_before_opening_a_row(store):
    history = store()
    with pytest.raises(TypeError, match="step"):
        history.ingest({"bad": 1}, step="bad", commit=True)
    assert history.open_record() is None
    assert history.current_step == 0


def test_invalid_step_does_not_consume_a_generator_payload(store):
    history = store()
    consumed = []

    def values():
        consumed.append(True)
        yield 1

    payload = {"values": values()}
    with pytest.raises(TypeError, match="step"):
        history.log(payload, step="bad")
    assert consumed == []
    history.log(payload)
    assert consumed == [True]
    assert history.get(1)[0]["values"] == [1]


def test_finished_store_drops_without_consuming_payload_or_validating_step(store):
    history = store()
    history.finish()
    consumed = []

    def values():
        consumed.append(True)
        yield 1

    assert history.log({"values": values()}, step="bad") is None
    assert consumed == []
    assert history.ingest({"value": 1}, step="bad") is None
