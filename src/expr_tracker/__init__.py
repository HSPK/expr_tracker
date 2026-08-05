from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from . import tracker
from .alerts import (
    AlertConfig,
    AlertLevel,
    AlertMessage,
    AlertRule,
    ChannelConfig,
    M,
    WebhookPolicy,
    add_alert_rule,
    alert,
    configure_alert,
    list_alert_rules,
    register_backend,
    remove_alert_rule,
)
from .artifacts import Artifact
from .run import Run

try:
    __version__ = _version("expr_tracker")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"

init = tracker.init
finish = tracker.finish
log = tracker.log
info = tracker.info
history = tracker.history
summary = tracker.summary
get_run = tracker.get_run
define_metric = tracker.define_metric
log_artifact = tracker.log_artifact
use_artifact = tracker.use_artifact

__all__ = [
    "AlertConfig",
    "AlertLevel",
    "AlertMessage",
    "AlertRule",
    "Artifact",
    "ChannelConfig",
    "M",
    "Run",
    "WebhookPolicy",
    "__version__",
    "add_alert_rule",
    "alert",
    "configure_alert",
    "define_metric",
    "finish",
    "get_run",
    "history",
    "info",
    "init",
    "list_alert_rules",
    "log",
    "log_artifact",
    "register_backend",
    "remove_alert_rule",
    "summary",
    "use_artifact",
]
