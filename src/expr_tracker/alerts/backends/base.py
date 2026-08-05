"""Alert backend base class and registry. HTTP uses stdlib ``urllib``, no new deps."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from ..models import AlertMessage, ChannelConfig


class SendError(Exception):
    """Delivery failure; ``retryable`` gates retries and ``retry_after`` is a hint."""

    def __init__(
        self, message: str, *, retryable: bool = True, retry_after: float | None = None
    ):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class AlertBackend:
    """One delivery channel. Subclasses implement :meth:`send`."""

    type: str = ""

    def __init__(self, config: ChannelConfig):
        self.config = config

    def validate(self):
        """Validate configuration at setup time; failures raise immediately."""

    def send(self, message: AlertMessage) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


_REGISTRY: dict[str, type[AlertBackend]] = {}


def register_backend(kind: str, cls: type[AlertBackend]):
    _REGISTRY[kind.lower()] = cls


def create_backend(config: ChannelConfig) -> AlertBackend:
    cls = _REGISTRY.get(config.type)
    if cls is None:
        raise ValueError(
            f"Unknown alert backend {config.type!r}; available: {sorted(_REGISTRY)}"
        )
    backend = cls(config)
    backend.validate()
    return backend


def post_json(
    url: str, payload: dict, timeout: float, headers: dict | None = None
) -> str:
    """POST a JSON body, raising :class:`SendError` on failure."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        retry_after = _parse_retry_after(
            e.headers.get("Retry-After") if e.headers else None
        )
        raise SendError(
            f"HTTP {e.code} from {_redact(url)}: {e.reason}",
            retryable=e.code in (408, 429, 500, 502, 503, 504),
            retry_after=retry_after,
        ) from e
    except urllib.error.URLError as e:
        raise SendError(f"Network error posting to {_redact(url)}: {e.reason}") from e
    except Exception as e:  # socket timeouts and similar
        raise SendError(f"Failed to post to {_redact(url)}: {e}") from e


def render_text(message: AlertMessage, include_fields: bool = True) -> str:
    """Render a message as plain text for channels without rich cards."""
    parts = [message.text]
    if include_fields and message.fields:
        parts.append("")
        parts.extend(f"{key}: {value}" for key, value in message.fields.items())
    if message.link:
        parts.append(f"link: {message.link}")
    if message.traceback:
        parts.append("")
        parts.append(message.traceback)
    return "\n".join(parts)


def _parse_retry_after(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _redact(url: str) -> str:
    """Keep only scheme://host: paths and query strings usually carry the token."""
    try:
        parts = urllib.parse.urlsplit(url)
    except Exception:
        return "<webhook>"
    if not parts.hostname:
        return "<webhook>"
    return f"{parts.scheme}://{parts.hostname}/…"
