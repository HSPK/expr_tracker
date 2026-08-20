"""Lark (Feishu) interactive message cards.

The card layout is the one ``slark`` builds, reproduced here as plain dicts so
the Lark channel needs no third-party client: a webhook post is one JSON body
over stdlib HTTP, which is what every other channel already does.

Reference: https://open.feishu.cn/document/server-docs/im-v1/message-card
"""

from __future__ import annotations

import time
from typing import Any

GREEN = "green"
RED = "red"
YES_ICON = "yes_filled"
ERROR_ICON = "error_filled"


def _heading(content: str) -> dict[str, Any]:
    """A section heading. Lark has no heading element, hence the wrapping."""
    return {
        "tag": "column_set",
        "flex_mode": "none",
        "horizontal_spacing": "default",
        "horizontal_align": "left",
        "background_style": "default",
        "columns": [
            {
                "tag": "column",
                "background_style": "default",
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "plain_text",
                            "content": content,
                            "text_size": "heading",
                            "text_align": "left",
                            "text_color": "default",
                        },
                    }
                ],
                "width": "auto",
                "weight": 1,
                "vertical_align": "top",
                "vertical_spacing": "default",
            }
        ],
    }


def _code_block(content: str, language: str = "txt") -> dict[str, Any]:
    return {
        "tag": "markdown",
        "content": f"```{language}\n{content}\n```",
        "text_align": "left",
        "text_size": "normal",
    }


def _header(title: str, subtitle: str, template: str, icon: str) -> dict[str, Any]:
    return {
        "title": {"tag": "plain_text", "content": title},
        "subtitle": {"tag": "plain_text", "content": subtitle},
        "template": template,
        "ud_icon": {"token": icon},
    }


def build_card(
    message: str,
    title: str,
    subtitle: str | None = None,
    traceback: str | None = None,
    *,
    failed: bool = False,
) -> dict[str, Any]:
    """One card: a coloured header, the message, and a traceback if there is one."""
    elements: list[dict[str, Any]] = [
        _heading("Message"),
        _code_block(message),
    ]
    if traceback and traceback.strip():
        # Skipped when absent: most alerts are metric conditions, not crashes,
        # and an empty "Traceback" code block on every one of them is noise
        elements += [_heading("Traceback"), _code_block(traceback.strip(), "")]
    return {
        "header": _header(
            title,
            subtitle if subtitle is not None else time.strftime("%Y-%m-%d %H:%M:%S"),
            RED if failed else GREEN,
            ERROR_ICON if failed else YES_ICON,
        ),
        "elements": elements,
    }


def card_payload(card: dict[str, Any]) -> dict[str, Any]:
    """The webhook envelope a card has to travel in."""
    return {"msg_type": "interactive", "card": card}
