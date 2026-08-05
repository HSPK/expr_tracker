"""Alert rule engine: compilation, state machine and watchdog."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from loguru import logger

from .dispatch import Dispatcher
from .expr import EvalContext, Node, compile_condition, evaluate, explain, validate
from .expr.eval import tristate
from .expr.functions import UNKNOWN
from .expr.nodes import RangeRef
from .expr.rule import parse_rule
from .models import DEFAULT_WATCHDOG_INTERVAL, AlertLevel, AlertMessage, AlertRule

MAX_RULE_ERRORS = 3
TIME_FUNCTIONS = {"no_data", "elapsed", "age"}


@dataclass
class RuleState:
    consecutive: int = 0
    firing: bool = False
    last_fire: float | None = None
    fires: int = 0
    errors: int = 0


@dataclass
class CompiledRule:
    rule: AlertRule
    node: Node
    metrics: set[str] = field(default_factory=set)
    time_based: bool = False
    window: int = 1
    state: RuleState = field(default_factory=RuleState)
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
        self.rules: dict[str, CompiledRule] = {}
        self._lock = threading.RLock()
        self._watchdog: threading.Thread | None = None
        self._stopping = threading.Event()
        self._closed = False
        for rule in rules:
            self.add_rule(rule, _defer_watchdog=True)
        self._start_watchdog_if_needed()

    # ------------------------------------------------------------------ rules

    def add_rule(
        self, rule, _defer_watchdog: bool = False, **overrides
    ) -> CompiledRule:
        parsed = parse_rule(rule, **overrides)
        node = compile_condition(parsed.condition)
        validate(node)
        compiled = CompiledRule(
            rule=parsed,
            node=node,
            metrics=node.metrics(),
            time_based=bool(node.functions() & TIME_FUNCTIONS),
            window=_required_window(node),
        )
        self._reserve_window(compiled)
        with self._lock:
            if parsed.name in self.rules:
                logger.warning(
                    f"Alert rule {parsed.name!r} already exists and was replaced."
                )
            self.rules[parsed.name] = compiled
        if not _defer_watchdog:
            self._start_watchdog_if_needed()
        return compiled

    def _reserve_window(self, compiled: CompiledRule):
        """Make sure the metric buffers keep enough points for this rule."""
        series = getattr(self.context_factory, "series", None)
        if series is None:
            context = self.context_factory(None)
            series = getattr(context, "series", None)
        ensure = getattr(series, "ensure_capacity", None)
        if ensure is not None:
            ensure(compiled.window)

    def _start_watchdog_if_needed(self):
        with self._lock:
            needed = any(c.time_based for c in self.rules.values())
        if needed:
            self._ensure_watchdog()

    def remove_rule(self, name: str) -> bool:
        with self._lock:
            return self.rules.pop(name, None) is not None

    def list_rules(self) -> list[AlertRule]:
        with self._lock:
            return [compiled.rule for compiled in self.rules.values()]

    # ------------------------------------------------------------------ evaluation

    def on_step(self, record: dict):
        with self._lock:
            compiled_rules = list(self.rules.values())
        if not compiled_rules:
            return
        keys = {key for key in record if not key.startswith("_")}
        ctx = self.context_factory(record)
        for compiled in compiled_rules:
            if not compiled.rule.enabled:
                continue
            if not compiled.time_based and not _touches(compiled, keys):
                continue
            self._evaluate(compiled, ctx, record)

    def _tick(self):
        with self._lock:
            compiled_rules = [
                c for c in self.rules.values() if c.time_based and c.rule.enabled
            ]
        if not compiled_rules:
            return
        ctx = self.context_factory(None)
        for compiled in compiled_rules:
            self._evaluate(compiled, ctx, None)

    def _evaluate(self, compiled: CompiledRule, ctx: EvalContext, record: dict | None):
        with compiled.lock:
            self._evaluate_locked(compiled, ctx, record)

    def _evaluate_locked(
        self, compiled: CompiledRule, ctx: EvalContext, record: dict | None
    ):
        rule, state = compiled.rule, compiled.state
        try:
            result = tristate(evaluate(compiled.node, ctx))
        except Exception as e:
            state.errors += 1
            logger.warning(f"Alert rule {rule.name!r} failed to evaluate: {e}")
            if state.errors >= MAX_RULE_ERRORS:
                rule.enabled = False
                logger.error(
                    f"Alert rule {rule.name!r} disabled after {state.errors} errors."
                )
            return
        state.errors = 0
        if result is UNKNOWN:
            return
        now = time.time()
        if result:
            state.consecutive += 1
            if state.consecutive < rule.for_steps:
                return
            if rule.mode == "edge" and state.firing:
                return
            if (
                rule.mode == "level"
                and state.last_fire is not None
                and rule.cooldown
                and now - state.last_fire < rule.cooldown
            ):
                return
            if rule.max_fires is not None and state.fires >= rule.max_fires:
                return
            if not self._fire(compiled, ctx, record):
                return  # rendering or dispatch failed: keep state so it can retry
            state.firing, state.last_fire = True, now
            state.fires += 1
        else:
            state.consecutive = 0
            if state.firing:
                state.firing = False
                if rule.notify_recovery:
                    self._fire(compiled, ctx, record, recovered=True)

    # ------------------------------------------------------------------ firing

    def _fire(
        self,
        compiled: CompiledRule,
        ctx: EvalContext,
        record: dict | None,
        recovered: bool = False,
    ) -> bool:
        rule = compiled.rule
        values = self._template_values(compiled, ctx, record)
        try:
            rendered = _format(rule.message, values)
        except Exception as e:  # pragma: no cover - format_map already tolerates gaps
            logger.warning(f"Failed to render message for rule {rule.name!r}: {e}")
            rendered = rule.condition
        title = _format(rule.title, values) if rule.title else rule.name
        level = AlertLevel.INFO if recovered else rule.level
        message = AlertMessage(
            title=f"[recovered] {title}" if recovered else title,
            text=f"Condition no longer holds: {values['expr']}"
            if recovered
            else rendered,
            subtitle=self.run_info.get("run"),
            level=level,
            fields={
                "project": self.run_info.get("project"),
                "run": self.run_info.get("run"),
                "step": values["step"],
                "condition": rule.condition,
            },
            tags=list(rule.tags),
            link=self.run_info.get("link"),
            source=f"rule:{rule.name}",
            dedup_key=(
                f"rule:{rule.name}:"
                f"{'recovered' if recovered else 'fire'}:{compiled.state.fires}"
            ),
        )
        try:
            self.dispatcher.send(message, rule.channels)
        except Exception as e:
            logger.warning(f"Failed to dispatch alert for rule {rule.name!r}: {e}")
            return False
        return True

    def _template_values(
        self, compiled: CompiledRule, ctx: EvalContext, record: dict | None
    ):
        values = dict(record or ctx.record)
        values.update(
            {
                "name": compiled.rule.name,
                "condition": compiled.rule.condition,
                "step": ctx.step if ctx.step is not None else "-",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "project": self.run_info.get("project", ""),
                "run": self.run_info.get("run", ""),
                "expr": _safe_explain(compiled.node, ctx),
            }
        )
        return values

    # ------------------------------------------------------------------ watchdog

    def _ensure_watchdog(self):
        if self.watchdog_interval <= 0:
            return
        with self._lock:
            if self._watchdog is not None and self._watchdog.is_alive():
                return
            self._stopping.clear()
            self._watchdog = threading.Thread(
                target=self._watch, name="et-alert-watchdog", daemon=True
            )
            self._watchdog.start()

    def _watch(self):
        while not self._stopping.wait(self.watchdog_interval):
            try:
                self._tick()
            except Exception as e:  # pragma: no cover - background thread guard
                logger.warning(f"Alert watchdog failed: {e}")

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
        return sorted(
            name
            for name in compiled.metrics
            if not series.has(name) and not series.has(name.replace(".", "/"))
        )

    def _report_unresolved(self):
        """A typo'd metric name makes a rule silently dead; say so once, at the end."""
        series = self._series()
        if series is None or not series.names():
            return  # nothing was ever logged: absence proves nothing
        for name, compiled in list(self.rules.items()):
            missing = self.unresolved_metrics(compiled)
            if not missing:
                continue
            hints = "".join(_suggest(series, metric) for metric in missing)
            logger.warning(
                f"Alert rule {name!r} never fired: metric(s) "
                f"{', '.join(repr(m) for m in missing)} were never logged.{hints}"
            )

    def close(self):
        try:
            self._report_unresolved()
        except Exception as e:  # closing must never raise
            logger.warning(f"Could not check alert rule metrics: {e}")
        self._closed = True
        self._stopping.set()
        watchdog = self._watchdog
        if watchdog is not None and watchdog.is_alive():
            watchdog.join(timeout=2.0)
        self._watchdog = None
        self.dispatcher.close()

    def stats(self) -> dict:
        with self._lock:
            return {
                name: {
                    "fires": compiled.state.fires,
                    "firing": compiled.state.firing,
                    "enabled": compiled.rule.enabled,
                    "unresolved_metrics": self.unresolved_metrics(compiled),
                }
                for name, compiled in self.rules.items()
            }


def _suggest(series, metric: str) -> str:
    from difflib import get_close_matches

    matches = get_close_matches(metric, sorted(series.names()), n=3, cutoff=0.6)
    return f" Did you mean: {', '.join(matches)}?" if matches else ""


def _required_window(node: Node) -> int:
    """Largest point window the expression needs, so buffers can be sized for it."""
    window = 1
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, RangeRef) and current.count:
            window = max(window, current.count)
        stack.extend(current.children())
    return window


def _touches(compiled: CompiledRule, keys: set[str]) -> bool:
    """Whether this step updated a metric the rule references (dotted alias too)."""
    if not compiled.metrics:
        return True
    return any(
        name in keys or name.replace(".", "/") in keys for name in compiled.metrics
    )


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
