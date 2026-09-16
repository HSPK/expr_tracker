"""Turning arbitrary metric values into JSON lines, without ever losing a record."""

from __future__ import annotations

import json

from loguru import logger

from ..encoders import jsonable_encoder

MAX_FALLBACK_REPR_LENGTH = 512
RESERVED_KEYS = frozenset({"_step", "_time"})

_LINE_ENCODER = json.JSONEncoder(ensure_ascii=False)


class RecordCodec:
    """Encode metric mappings to JSON-safe values, warning about each bad key once."""

    def __init__(self):
        self._warned_keys: set[str] = set()

    def reset(self):
        self._warned_keys.clear()

    def warn_once(self, key: str, message: str) -> bool:
        """Log ``message`` the first time ``key`` misbehaves. Returns whether it did."""
        if key in self._warned_keys:
            return False
        self._warned_keys.add(key)
        logger.warning(message)
        return True

    def encode(self, metrics: dict | None, kind: str = "metric") -> dict:
        """Encode a mapping; one bad value must never drop the whole record."""
        if metrics is None:
            return {}
        if not isinstance(metrics, dict):
            return {"value": self.encode_value("value", metrics, kind)}
        # Encode each field once: retrying a failed mapping consumes generators
        # and other one-shot values a second time.
        encoded = {}
        for key, value in metrics.items():
            if isinstance(key, str) and key.startswith("_sa"):
                continue
            encoded_key = encode_key(key)
            encoded[encoded_key] = self.encode_value(encoded_key, value, kind)
        return encoded

    def encode_value(self, key: str, value, kind: str = "metric"):
        try:
            return jsonable_encoder(value)
        except Exception as e:
            self.warn_once(
                key,
                f"{kind.capitalize()} {key!r} of type {type(value).__name__} is "
                f"not JSON serializable ({e}); falling back to repr().",
            )
            return fallback_repr(value)


def encode_line(record: dict) -> bytes:
    """Serialise a record as one JSONL line."""
    return (_LINE_ENCODER.encode(record) + "\n").encode(
        "utf-8", errors="backslashreplace"
    )


def fallback_repr(value) -> str:
    try:
        text = repr(value)
    except Exception:
        return f"<unrepresentable {type(value).__name__}>"
    if len(text) > MAX_FALLBACK_REPR_LENGTH:
        return text[:MAX_FALLBACK_REPR_LENGTH] + "..."
    return text


def encode_key(key) -> str:
    if isinstance(key, str):
        return key
    try:
        encoded = jsonable_encoder(key)
    except Exception:
        return fallback_repr(key)
    if isinstance(encoded, str):
        return encoded
    if isinstance(encoded, int | float | bool) or encoded is None:
        return str(encoded)
    return fallback_repr(key)
