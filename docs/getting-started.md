# Getting started

## Install

```bash
uv add expr_tracker
```

Only `click`, `loguru` and `pydantic` are required. Everything else is an extra:

| Extra | Adds |
| --- | --- |
| `wandb` | mirror metrics to Weights & Biases |
| `trackio` | mirror metrics to trackio |
| `lark` | Feishu/Lark alert channel |
| `pandas` / `polars` | `history(output_type=...)` frames |
| `all` | all of the above |

A missing extra is reported with the exact install command; it never crashes a run.

## A complete run

```python
import expr_tracker as et

et.init(
    project="mnist",
    name="baseline",
    config={"lr": 3e-4, "batch_size": 64},
    alert_rules=["isnan(loss) => critical: loss diverged"],
)

for step in range(1000):
    loss = train_one_step()
    et.log({"loss": loss, "lr": scheduler.get_last_lr()[0]})

    if step % 100 == 0:
        et.log({"eval/acc": evaluate()}, step=step)

et.summary()["best_acc"] = best
et.log_artifact("checkpoints/final.pt", name="model", type="model")
et.finish()
```

This writes:

```
./tracker/jsonl/mnist/baseline/
├── metrics.jsonl      # one JSON object per step
├── metrics.meta.json  # index sidecar, for fast resume and seeks
├── config.json
├── summary.json
└── artifacts.jsonl    # lineage: what this run produced and consumed
```

Change the location with `et.init(dir="/data/runs")`.

## Reading it back

While the run is live, or from another process afterwards:

```python
et.history(50)                             # last 50 steps
et.history(-1)                             # everything
et.history(-1, metrics=["loss"])           # one metric
et.history(-1, step_range=(100, 200))      # a slice, end exclusive
et.history(100, output_type="pandas")      # a DataFrame

et.history(50, run="tracker/jsonl/mnist/baseline")   # no init() needed
```

Rows are plain dicts with `_step` and `_time` plus whatever you logged:

```python
{"_step": 42, "_time": 1754323200.123, "loss": 0.31, "lr": 0.0003}
```

## Resuming

Re-running `init()` with the same project and name continues the same file:

```python
et.init(project="mnist", name="baseline")
et.get_run().step        # 1000 - the cursor picked up where it left off
len(et.history(-1))      # 1000 - the old rows are still there
```

A process that dies without `finish()` still leaves a complete, valid file: the open
row is committed and the summary saved by an exit hook, and a torn trailing line is
repaired on the next run.

## Next

- [Logging metrics](guide/logging.md) for commit semantics and out-of-order steps.
- [Alerts](guide/alerts.md) to get notified instead of watching curves.
- [CLI](guide/cli.md) to inspect runs without writing code.
