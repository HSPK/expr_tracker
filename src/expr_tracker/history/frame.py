"""Projection and output-format conversion for history query results."""

from __future__ import annotations

from collections.abc import Sequence

META_KEYS = ("_step", "_time")


def project(
    records: Sequence[dict],
    *,
    metrics: Sequence[str] | None = None,
    include_meta: bool = True,
    fill_missing: bool = False,
    dropna: bool = False,
) -> list[dict]:
    """Select columns, optionally filling gaps or dropping empty rows."""
    selected = list(metrics) if metrics is not None else None
    rows: list[dict] = []
    for record in records:
        if selected is None:
            row = {
                k: v for k, v in record.items() if include_meta or k not in META_KEYS
            }
        else:
            row = {k: record[k] for k in selected if k in record}
            if dropna and not row:
                continue
            if fill_missing:
                row = {k: record.get(k) for k in selected}
            if include_meta:
                meta = {k: record[k] for k in META_KEYS if k in record}
                row = {**meta, **row}
        rows.append(row)
    if selected is None and fill_missing:
        keys = _ordered_keys(rows)
        rows = [{k: row.get(k) for k in keys} for row in rows]
    return rows


def to_output(records: list[dict], output_type: str):
    kind = (output_type or "dict").lower()
    if kind in ("dict", "dicts", "records", "list"):
        return records
    if kind in ("pd", "pandas", "dataframe", "df"):
        return _to_pandas(records)
    if kind in ("pl", "polars"):
        return _to_polars(records)
    raise ValueError(
        f"Unsupported output_type {output_type!r}; expected one of "
        "'dict', 'pandas', 'polars'."
    )


def _ordered_keys(records: Sequence[dict]) -> list[str]:
    """``_step``/``_time`` first, then metrics in first-seen order."""
    keys: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    meta = [k for k in META_KEYS if k in seen]
    return meta + [k for k in keys if k not in META_KEYS]


def _to_pandas(records: list[dict]):
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            'output_type="pandas" requires pandas: pip install "expr_tracker[pandas]"'
        ) from e
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records, columns=_ordered_keys(records))


def _to_polars(records: list[dict]):
    try:
        import polars as pl
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            'output_type="polars" requires polars: pip install "expr_tracker[polars]"'
        ) from e
    if not records:
        return pl.DataFrame()
    keys = _ordered_keys(records)
    return pl.DataFrame([{k: row.get(k) for k in keys} for row in records])
