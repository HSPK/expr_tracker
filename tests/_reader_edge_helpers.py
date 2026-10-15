"""Shared fixtures and helpers for the corresponding split regression suite."""

import json

import pytest

from expr_tracker.history import reader as reader_module


def write_rows(path, rows):
    payload = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    path.write_bytes(payload)
    return payload


def write_meta(path, **overrides):
    meta = {
        "schema": 2,
        "size": path.stat().st_size,
        "sorted": True,
        "has_duplicate_steps": False,
        "index": [],
    }
    meta.update(overrides)
    path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def reader_warnings():
    records = []
    sink = reader_module.logger.add(
        lambda message: records.append(message.record), level="WARNING"
    )
    try:
        yield records
    finally:
        reader_module.logger.remove(sink)
