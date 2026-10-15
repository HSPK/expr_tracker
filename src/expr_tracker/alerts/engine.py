"""Python SDK adapters for the Rust rule registry, state machine and watchdog.

AST/model objects and reentrant callback boundaries remain Python-compatible;
rule compilation, selection, evaluation, transitions and scheduling run natively.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from loguru import logger

from .._native import (
    AlertRuleRegistry,
    AlertRuleState,
    AlertTimer,
    alert_evaluate_rule,
    alert_fire,
    alert_rule_metadata,
    alert_rules_for_step,
    alert_template_values,
    alert_unresolved,
)
from .dispatch import Dispatcher
from .expr import EvalContext, Node, explain
from .expr.rule import parse_rule
from .models import DEFAULT_WATCHDOG_INTERVAL, AlertRule

MAX_RULE_ERRORS = 3
TIME_FUNCTIONS = {"no_data", "elapsed", "age"}


RuleState = AlertRuleState


@dataclass
class CompiledRule:
    rule: AlertRule
    node: Node
    metrics: set[str] = field(default_factory=set)
    time_based: bool = False
    window: int = 1
    state: RuleState = field(default_factory=RuleState)
    plan: object = None
    # on_step and the watchdog can hit one rule concurrently: serialise transitions
    lock: threading.RLock = field(default_factory=threading.RLock)


class AlertEngine:
    """Evaluate rules on every committed step and hand firing ones to the dispatcher."""

    def __init__(
        self,
        dispatcher: Dispatcher,
        context_factory: Callable[[dict | None], EvalContext],
        *,
        rules: Sequence = (),
        run_info: dict | None = None,
        watchdog_interval: float = DEFAULT_WATCHDOG_INTERVAL,
    ):
        self.dispatcher = dispatcher
        self.context_factory = context_factory
        self.run_info = dict(run_info or {})
        self.watchdog_interval = watchdog_interval
        self._registry = AlertRuleRegistry()
        self.rules = self._registry.rules
        self._lock = threading.RLock()
        self._watchdog = None
        self._closed = False
        for rule in rules:
            self.add_rule(rule, _defer_watchdog=True)
        self._start_watchdog_if_needed()

    # ------------------------------------------------------------------ rules

    def add_rule(
        self, rule, _defer_watchdog: bool = False, **overrides
    ) -> CompiledRule:
        parsed = parse_rule(rule, **overrides)
        with self._lock:
            compiled = self._registry.add(parsed, self.context_factory)
        if not _defer_watchdog:
            self._start_watchdog_if_needed()
        return compiled

    def _start_watchdog_if_needed(self):
        with self._lock:
            needed = self._registry.time_based()
        if needed:
            self._ensure_watchdog()

    def remove_rule(self, name: str) -> bool:
        with self._lock:
            return self._registry.remove(name)

    def list_rules(self) -> list[AlertRule]:
        with self._lock:
            return self._registry.list()

    # ------------------------------------------------------------------ evaluation

    def on_step(self, record: dict):
        self._registry.evaluate(self.context_factory, record, self._fire)

    _native_step = on_step

    def _tick(self):
        self._registry.evaluate(self.context_factory, None, self._fire)

    def _evaluate(self, compiled: CompiledRule, ctx: EvalContext, record: dict | None):
        with compiled.lock:
            self._evaluate_locked(compiled, ctx, record)

    def _evaluate_locked(
        self, compiled: CompiledRule, ctx: EvalContext, record: dict | None
    ):
        alert_evaluate_rule(compiled, ctx, record, self._fire)

    # ------------------------------------------------------------------ firing

    def _fire(
        self,
        compiled: CompiledRule,
        ctx: EvalContext,
        record: dict | None,
        recovered: bool = False,
    ) -> bool:
        return alert_fire(
            compiled,
            ctx,
            record,
            self.run_info,
            self.dispatcher,
            recovered,
        )

    def _template_values(
        self, compiled: CompiledRule, ctx: EvalContext, record: dict | None
    ):
        return alert_template_values(compiled, ctx, record, self.run_info)

    # ------------------------------------------------------------------ watchdog

    def _ensure_watchdog(self):
        if self.watchdog_interval <= 0:
            return
        with self._lock:
            if self._watchdog is not None and self._watchdog.is_alive():
                return
            self._watchdog = AlertTimer(self._tick, self.watchdog_interval)

    def _series(self):
        series = getattr(self.context_factory, "series", None)
        if series is None:
            series = getattr(self.context_factory(None), "series", None)
        return series

    def unresolved_metrics(self, compiled: CompiledRule) -> list[str]:
        """Metrics a rule references that were never logged, so it can never fire."""
        series = self._series()
        if series is None:
            return []
        return alert_unresolved(compiled, series)

    def _report_unresolved(self):
        """A typo'd metric name makes a rule silently dead; say so once, at the end."""
        self._registry.report_unresolved(self._series())

    def close(self):
        try:
            self._report_unresolved()
        except Exception as e:  # closing must never raise
            logger.warning(f"Could not check alert rule metrics: {e}")
        self._closed = True
        watchdog = self._watchdog
        if watchdog is not None and watchdog.is_alive():
            watchdog.close()
        self._watchdog = None
        self.dispatcher.close()

    def stats(self) -> dict:
        with self._lock:
            return self._registry.stats(self._series())


def _required_window(node: Node) -> int:
    """Largest point window the expression needs, so buffers can be sized for it."""
    return alert_rule_metadata(node)[2]


def _touches(compiled: CompiledRule, keys: set[str]) -> bool:
    """Whether this step updated a metric the rule references (dotted alias too)."""
    return bool(alert_rules_for_step({"rule": compiled}, dict.fromkeys(keys)))


def _safe_explain(node: Node, ctx: EvalContext) -> str:
    try:
        return explain(node, ctx)
    except Exception:
        return node.to_source()


class _SafeDict(dict):
    def __missing__(self, key):
        return _Placeholder(key)


class _Placeholder(str):
    def __format__(self, spec: str) -> str:
        return "{" + str(self) + (":" + spec if spec else "") + "}"


def _format(template: str, values: dict) -> str:
    return template.format_map(_SafeDict(values))
