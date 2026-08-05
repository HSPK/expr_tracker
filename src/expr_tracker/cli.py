"""``et`` command line: history queries, rule replay/explain, manual alerts."""

from __future__ import annotations

import json
import time

import click

from .alerts import alert as send_alert
from .alerts.dispatch import Dispatcher
from .alerts.engine import AlertEngine
from .alerts.expr import EvalContext, compile_condition, parse_rule, validate
from .alerts.models import AlertConfig, ChannelConfig, WebhookPolicy
from .history import MetricSeries, read_history


@click.group()
def main():
    """expr_tracker command line."""


@main.command()
@click.argument("msg")
@click.option("--title", default="Alert", help="Title of the alert")
@click.option("--level", default="info", help="info | warning | error | critical")
@click.option("--channel", "channels", multiple=True, help="Restrict to these channels")
def alert(msg: str, title: str, level: str, channels: tuple[str, ...]):
    """Send a manual alert."""
    send_alert(title=title, text=msg, level=level, channels=list(channels) or None)


@main.command()
@click.argument("run", type=click.Path(exists=True))
@click.option("-n", default=20, help="Number of steps (-1 for all)")
@click.option("--metrics", default=None, help="Comma separated metric names")
@click.option("--step-range", default=None, help="start:end (end exclusive)")
@click.option(
    "--format", "fmt", type=click.Choice(["table", "json", "csv"]), default="table"
)
def history(run: str, n: int, metrics: str | None, step_range: str | None, fmt: str):
    """Print the recorded history of a run."""
    rows = read_history(
        run,
        n,
        metrics=metrics.split(",") if metrics else None,
        step_range=_parse_range(step_range),
    )
    if fmt == "json":
        click.echo(json.dumps(rows, ensure_ascii=False, indent=2))
    elif fmt == "csv":
        click.echo(_to_csv(rows))
    else:
        click.echo(_to_table(rows))


@main.group()
def rules():
    """Alert rule tooling."""


@rules.command("explain")
@click.argument("expression")
def rules_explain(expression: str):
    """Print how an expression parses and which metrics it references."""
    rule = parse_rule(expression)
    node = compile_condition(rule.condition)
    validate(node)
    click.echo(f"condition : {node.to_source()}")
    click.echo(f"level     : {rule.level.value}")
    click.echo(f"message   : {rule.message}")
    click.echo(f"metrics   : {', '.join(sorted(node.metrics())) or '-'}")
    click.echo(f"functions : {', '.join(sorted(node.functions())) or '-'}")


@rules.command("test")
@click.argument("rule")
@click.option(
    "--run", required=True, type=click.Path(exists=True), help="Run dir or jsonl"
)
@click.option("-n", default=-1, help="Only replay the last n steps")
def rules_test(rule: str, run: str, n: int):
    """Replay a rule over recorded history and print every step it would fire on."""
    fired, count = replay(parse_rule(rule), read_history(run, n))
    click.echo(f"replayed {count} steps, {len(fired)} alert(s)")
    for message in fired:
        click.echo(
            f"  step={message.fields.get('step')}  {message.title}: {message.text}"
        )


def replay(rule, records: list[dict]):
    """Replay one rule over records; returns ``(fired messages, steps replayed)``."""
    fired: list = []
    context = _ReplayContext()
    engine = _replay_engine(rule, context, fired)
    count = 0
    for record in records:
        step = record.get("_step")
        if not isinstance(step, int):
            continue
        context.series.add(step, float(record.get("_time") or 0.0), record)
        engine.on_step(record)
        count += 1
    return fired, count


class _ReplayContext:
    """Evaluation-context factory for replay, backed by its own MetricSeries."""

    def __init__(self):
        self.series = MetricSeries()
        self.started_at = 0.0

    def __call__(self, record: dict | None) -> EvalContext:
        record = record or {}
        now = float(record.get("_time") or 0.0)
        if not self.started_at:
            self.started_at = now
        step = record.get("_step")
        return EvalContext(
            self.series,
            step=step if isinstance(step, int) else None,
            now=now,
            started_at=self.started_at,
            last_commit_time=now,
            record=record,
        )


def _replay_engine(rule, context, sink: list) -> AlertEngine:
    channel = ChannelConfig(
        type="callable",
        name="replay",
        options={"handler": sink.append},
        policy=WebhookPolicy(
            async_send=False, dedup_window=0, rate_limit_per_minute=None, max_retries=0
        ),
    )
    dispatcher = Dispatcher(AlertConfig(channels=[channel]))
    return AlertEngine(dispatcher, context, rules=[rule], watchdog_interval=0)


def _parse_range(value: str | None):
    if not value:
        return None
    start, _, end = value.partition(":")
    return (int(start) if start else None, int(end) if end else None)


def _columns(rows: list[dict]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _to_table(rows: list[dict]) -> str:
    if not rows:
        return "(no records)"
    columns = _columns(rows)
    cells = [[_cell(row.get(c), c) for c in columns] for row in rows]
    widths = [
        max([len(col)] + [len(row[i]) for row in cells])
        for i, col in enumerate(columns)
    ]
    lines = ["  ".join(col.ljust(widths[i]) for i, col in enumerate(columns))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(
        "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in cells
    )
    return "\n".join(lines)


def _to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    columns = _columns(rows)
    lines = [",".join(columns)]
    lines.extend(",".join(_cell(row.get(c), c) for c in columns) for row in rows)
    return "\n".join(lines)


def _cell(value, column: str = "") -> str:
    if value is None:
        return ""
    if column == "_time" and isinstance(value, (int, float)):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
