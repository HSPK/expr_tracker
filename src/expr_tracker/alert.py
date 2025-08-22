from typing import Literal
from loguru import logger


def ignore_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Error occurred in {func.__name__}: {e}")
            return None

    return wrapper


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


@ignore_exception
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
            if lark_backend.lark._webhook_url is None:
                logger.warning(
                    "Lark webhook URL is not set. Please set the WEBHOOK_URL environment variable."
                )
                return
            lark_backend.publish_alert(title, text, subtitle, level, traceback)
        else:
            raise NotImplementedError(f"Backend '{backend}' is not implemented.")
