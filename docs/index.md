# Experiment Tracker

A local-first experiment tracker. Metrics land in a JSONL file you own, stay
queryable while the run is live, and can trigger alerts from an expression language.

```python
import expr_tracker as et

et.init(project="demo", name="run-1", alert_rules=["zscore(loss[50]) > 3 => error: spike"])
for step in range(1000):
    et.log({"loss": loss, "lr": lr})
et.finish()

et.history(50)                    # the last 50 steps, as dicts
et.history(-1, output_type="pd")  # everything, as a DataFrame
```

## Why

**The file is the source of truth.** One JSON object per step, appended to
`metrics.jsonl`. No server, no database, no vendor. `wandb` and `trackio` are
optional mirrors, not requirements.

**History is queryable during the run.** `et.history(n)` answers from an in-memory
cache and falls back to the file only for what it has evicted, so it stays cheap at
any scale — 227&nbsp;µs for `history(50)` whether the run has 1,000 steps or 100,000.

**Alerts are expressions, not callbacks.** `zscore(loss[50]) > 3 or isnan(loss)`
is parsed, validated and evaluated against a rolling window. Rules can be replayed
over a finished run to tune thresholds before you trust them.

**It stays out of the way.** `log()` costs ~26&nbsp;µs. A failed disk, a dead webhook
or an unserialisable value degrades with a warning; none of them can stop training.

## Install

```bash
uv add expr_tracker                 # local-first, three small dependencies
uv add "expr_tracker[all]"          # + wandb, trackio, lark, pandas, polars
```

## Next

- [Getting started](getting-started.md) — a complete run, end to end.
- [Logging metrics](guide/logging.md) — commit semantics and the step model.
- [Alerts](guide/alerts.md) — rules, channels and delivery policy.
- [Design](design.md) — the data model and the invariants behind it.
