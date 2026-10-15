"""Shared fixtures and helpers for the corresponding split regression suite."""

import json

import pytest

import expr_tracker as et
from expr_tracker.history.naming import spans_filename
from expr_tracker.spans import (
    COUNT_METRIC,
    TIME_METRIC,
    span_metric,
)


def timed(path):
    return span_metric(TIME_METRIC, path)


def counted(path):
    return span_metric(COUNT_METRIC, path)


def group_of(metric):
    """What a tracker UI files a metric under: the part before the first slash."""
    return metric.split("/")[0]


@pytest.fixture
def run(tmp_path):
    created = []

    def factory(**options):
        created.append(
            et.init(
                project="sp",
                name=options.pop("name", "r"),
                dir=str(tmp_path),
                backends=[],
                **options,
            )
        )
        return created[-1]

    yield factory
    if et.get_run() is not None:
        et.finish()


def metrics_of(instance):
    instance.history.flush(commit_open=True)
    row = instance.history_query(-1)[-1]
    return {k: v for k, v in row.items() if not k.startswith("_")}


def spans_of(tmp_path, name="r", stream=None):
    path = tmp_path / "sp" / name / spans_filename(stream, rank_aware=False)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]
