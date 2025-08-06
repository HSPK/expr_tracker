from typing import Literal


class LarkBackend:
    def __init__(self, **kwargs):
        from slark import Lark

        self.lark = Lark(**kwargs)

    def publish_alert(
        self,
        title: str,
        text: str,
        subtitle: str | None = None,
        level: Literal["info", "warning", "error"] = "info",
        traceback: str | None = None,
    ):
        if level == "info" or level == "warning":
            self.lark.webhook.post_success_card(
                msg=text, title=title, subtitle=subtitle
            )
        elif level == "error":
            self.lark.webhook.post_error_card(
                msg=text, traceback=traceback or "", title=title, subtitle=subtitle
            )
        else:
            raise ValueError(
                f"Unsupported level: {level}. Supported levels are 'info', 'warning', and 'error'."
            )


def alert(
    title: str,
    text: str,
    subtitle: str | None = None,
    traceback: str | None = None,
    level: Literal["info", "warning", "error"] = "info",
    backends: list[Literal["lark", "slack", "email"]] = ["lark"],
):
    """
    Send an alert message to specified backends.

    Args:
        title (str): The title of the alert.
        text (str): The main content of the alert.
        subtitle (str | None): Optional subtitle for the alert.
        level (Literal["info", "warning", "error"]): The severity level of the alert.
        backends (list[Literal["lark", "slack", "email"]]): List of backends to send the alert to.
    """
    for backend in backends:
        if backend == "lark":
            lark_backend = LarkBackend()
            lark_backend.publish_alert(title, text, subtitle, level, traceback)
        else:
            raise NotImplementedError(f"Backend '{backend}' is not implemented.")
