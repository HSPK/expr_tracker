# Logging metrics

## One step, one line

`log()` merges metrics into an **open row** for the current step. The row is written
when it is committed, so several `log()` calls for one step become one line:

```python
et.log({"loss": 1.0}, step=5)
et.log({"acc": 0.9}, step=5)     # merged into the same row
et.log({"loss": 0.9}, step=6)    # the step advanced -> step 5 is committed
```

This matters because training and evaluation usually run at different points in the
loop but describe the same step.

## Commit semantics

`log(data, step=None, commit=None)` mirrors `wandb.log`.

| Call | `commit` default | Behaviour |
| --- | --- | --- |
| `log(d)` | `True` | Merge, commit, advance the step |
| `log(d, commit=False)` | — | Merge only |
| `log(d, step=N)` | `False` | Commit when the step advances |
| `log(d, step=N, commit=True)` | — | Commit immediately |

An open row is also committed by `finish()`, by the process exit hook, and by the
`max_open_seconds` timeout (off by default).

```python
et.log({"train/loss": 1.0}, commit=False)
et.log({"train/grad_norm": 2.0}, commit=False)
et.log({"lr": 3e-4})              # commits all three as one step
```

## Step ordering

By default steps must not go backwards. A backward step is dropped with a warning and
reaches no sink — not the file, not the summary, not the alert rules, not wandb:

```python
et.log({"loss": 1.0}, step=10)
et.log({"loss": 2.0}, step=3)     # dropped, with a warning
```

Pass `step_policy="allow"` to keep out-of-order writes. The file then contains
repeated steps, and the reader merges them on the way out.

!!! note
    With `step_policy="allow"`, `history(n)` returns the *n* most recently **written**
    steps rather than the *n* highest-numbered ones. `history(-1)` and `step_range`
    are always step-ordered.

## What you can log

Anything JSON-serialisable, plus the things ML code actually holds:

```python
et.log({
    "loss": np.float32(0.5),          # numpy scalars -> python scalars
    "grads": np.array([1.0, 2.0]),    # arrays -> lists
    "config": PydanticModel(),        # pydantic models -> dicts
    "when": datetime.now(),           # datetime -> ISO string
    "path": Path("ckpt.pt"),          # Path -> str
    "stage": Stage.TRAIN,             # Enum -> its value
})
```

`NaN` and `Infinity` are preserved as-is, because a NaN loss is a signal and
`isnan()` in an alert rule depends on it. Python and pandas read this back; strict
JSON parsers such as `jq` do not.

A value that cannot be encoded falls back to `repr()` with a one-time warning, per
field — one bad value never costs you the rest of the row.

Metric names are free-form. `/` is conventional for grouping (`train/loss`), and the
alert language accepts `train.loss` as a shorthand for it.

## Reserved keys

`_step` and `_time` are written by the tracker. If you log them they are ignored, with
a warning.

## The summary

Every metric's last value is tracked automatically. Explicit assignments win and are
never overwritten:

```python
et.log({"acc": 0.91})
et.summary()["best_acc"] = 0.93   # survives later log() calls
dict(et.summary())                # {"acc": 0.91, "best_acc": 0.93}
```

The summary is saved on `finish()` and by the exit hook if the process dies first.

## Printing to screen

```python
et.init(..., print_to_screen=True)                 # print every committed row
et.init(..., print_to_screen=True, print_handle=logger.info)
```

Only committed rows are printed, so you see one line per step. A failing handle is
logged and never interrupts training.

## Cost

`log()` costs about 26&nbsp;µs (p50) without rules and 74&nbsp;µs with five rules,
around 33,000 calls/s. The write buffer batches lines, and the p99 stays under
5&nbsp;ms so a flush cannot stall the loop.
