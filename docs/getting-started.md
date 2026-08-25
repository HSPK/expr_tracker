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
| `pandas` / `polars` | `history(output_type=...)` frames |
| `gpu` | `GpuStats` span plugin, via NVML |
| `all` | all of the above |

Alert channels need nothing: every one of them, Lark included, is built on the
standard library.

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
./tracker/jsonl/            # the root, from dir=
└── mnist/                  # the project
    ├── baseline/           # this run
    │   ├── metrics.jsonl      # one JSON object per step
    │   ├── metrics.meta.json  # index sidecar, for fast resume and seeks
    │   ├── config.json
    │   ├── summary.json
    │   └── artifacts.jsonl    # lineage: what this run produced and consumed
    └── artifacts/          # shared by every run of the project
```

`dir` is a **root**, so a run lands in `<dir>/<project>/<name>`. That is what
lets a project's runs share and deduplicate artifacts, and what lets a resume
find its files from the project and name alone. The path is logged at `init()`,
and `et.get_run().dir` returns it.

```python
et.init(project="mnist", name="baseline", dir="/data/runs")
# -> /data/runs/mnist/baseline

et.init(project="mnist", name="baseline", run_dir="/data/runs/exp-42")
# -> /data/runs/exp-42, exactly
```

Use `run_dir` when something else already chose the path — a scheduler's output
directory, say. Artifacts then live in `<run_dir>/artifacts` and are no longer
shared with the project's other runs, which is the price of the flat layout.
Passing both `dir` and `run_dir` is an error.

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

This is unconditional: `resume` is passed on to wandb and trackio, but the local
file always continues. Use a different name to start clean.

A process that dies without `finish()` still leaves a complete, valid file: the open
row is committed and the summary saved by an exit hook, and a torn trailing line is
repaired on the next run.

## Next

- [Examples](examples.md) — six runnable programs, all offline.
- [Logging metrics](guide/logging.md) for commit semantics and out-of-order steps.
- [Alerts](guide/alerts.md) to get notified instead of watching curves.
- [CLI](guide/cli.md) to inspect runs without writing code.
