"""Alert backend base class and registry. HTTP uses stdlib ``urllib``, no new deps."""

from __future__ import annotations

import html
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
    parts = [message.subtitle, message.text] if message.subtitle else [message.text]
    if include_fields and message.fields:
        parts.append("")
        parts.extend(f"{key}: {value}" for key, value in message.fields.items())
    if message.link:
        parts.append(f"link: {message.link}")
    if message.traceback:
        parts.append("")
        parts.append(message.traceback)
    return "\n".join(parts)


LEVEL_COLORS = {
    "debug": "#616161",
    "info": "#0288d1",
    "warning": "#ed6c02",
    "error": "#d32f2f",
    "critical": "#b71c1c",
}


def render_html(message: AlertMessage) -> str:
    """Render a message as self-contained HTML for email.

    Everything is inline-styled and table-based: mail clients strip stylesheets,
    and several of them ignore ``div`` layout entirely.
    """
    color = LEVEL_COLORS.get(message.level.value, "#616161")
    esc = html.escape
    rows = "".join(
        f'<tr><td style="padding:4px 12px 4px 0;color:#666;'
        f'white-space:nowrap;vertical-align:top">{esc(str(key))}</td>'
        f'<td style="padding:4px 0;color:#111">{esc(str(value))}</td></tr>'
        for key, value in message.fields.items()
    )
    blocks = [
        f'<tr><td style="background:{color};padding:14px 20px;color:#fff">'
        f'<div style="font-size:17px;font-weight:600">{esc(message.title)}</div>'
        f'<div style="font-size:12px;opacity:.85;text-transform:uppercase;'
        f'letter-spacing:.05em">{esc(message.level.value)}'
        + (f" &middot; {esc(message.subtitle)}" if message.subtitle else "")
        + "</div></td></tr>"
    ]
    if message.text:
        blocks.append(
            '<tr><td style="padding:18px 20px 0;font-size:14px;line-height:1.5;'
            f'color:#111">{esc(message.text)}</td></tr>'
        )
    if rows:
        blocks.append(
            '<tr><td style="padding:16px 20px 0"><table cellpadding="0" '
            f'cellspacing="0" style="font-size:13px">{rows}</table></td></tr>'
        )
    if message.traceback:
        blocks.append(
            '<tr><td style="padding:16px 20px 0"><pre style="margin:0;padding:12px;'
            "background:#f6f6f6;border-radius:4px;font-size:12px;overflow:auto;"
            f'white-space:pre-wrap">{esc(message.traceback)}</pre></td></tr>'
        )
    if message.link:
        link = esc(message.link)
        blocks.append(
            f'<tr><td style="padding:18px 20px 0"><a href="{link}" '
            f'style="display:inline-block;padding:8px 16px;background:{color};'
            'color:#fff;text-decoration:none;border-radius:4px;font-size:13px">'
            "Open run</a></td></tr>"
        )
    blocks.append('<tr><td style="padding:20px"></td></tr>')
    return (
        '<html><body style="margin:0;background:#f4f4f4;'
        'font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">'
        '<table cellpadding="0" cellspacing="0" width="100%" '
        'style="background:#f4f4f4;padding:24px 0"><tr><td align="center">'
        '<table cellpadding="0" cellspacing="0" width="520" '
        'style="background:#fff;border-radius:6px;overflow:hidden;max-width:520px">'
        + "".join(blocks)
        + "</table></td></tr></table></body></html>"
    )


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
