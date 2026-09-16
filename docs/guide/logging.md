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

An open row is also committed by `finish()`, by the process exit hook, and by
`run.history.flush(commit_open=True)`. It never commits just because time passes,
so a long training step stays on one row. A plain `run.history.flush()` only writes
already committed rows; it does not commit the open row.

`finish()` waits for accepted writes and their commit callbacks before closing
downstream sinks. Concurrent `run.finish()` callers wait for the same shutdown
completion; a failed shutdown is surfaced to waiters rather than reported as
success. Call it outside commit/print and asynchronous alert-delivery callbacks;
trying to finish or reinitialize from such a callback raises `RuntimeError` to
avoid reentrant shutdown.
Lifecycle calls are also rejected from writer diagnostic callbacks, history
writer workers, unfinished write transactions, or an already-running lifecycle
operation; use the caller thread after the logging operation returns.
Writer diagnostics are emitted after writer locks are released and accounting is
consistent, allowing error sinks to inspect history without locking out writers.

Commit callbacks run outside the write transaction lock. A slow backend or
metric-print callback therefore does not block another thread recording an
uncommitted span. Committing callers still wait for their own notifications,
which are delivered serially in commit order; recursive logging and flush are
supported. No asynchronous callback worker or unbounded work queue is created.

Delivery may occur on another committing caller's thread. Context variables
come from the originating commit, but callbacks must not assume thread identity.
`flush()` and `finish()` wait for outstanding notifications before returning.
Alert-series windows advance with notification delivery, so a newer queued
commit cannot leak into an earlier alert. Ordinary history queries can already
see those accepted rows. Recursive logging advances the live alert series; when
an outer callback resumes, it sees the nested delivered rows, not a frozen snapshot.

```python
et.log({"train/loss": 1.0}, commit=False)
et.log({"train/grad_norm": 2.0}, commit=False)
et.log({"lr": 3e-4})              # commits all three as one step
```

Once an explicit step is open, subsequent calls without `step=` (including
completed spans) reuse it:

```python
et.log({"train/loss": 1.0}, step=100)
et.log({"lr": 3e-4})              # commits step 100; the next step is 101
```

## Timestamps and liveness

Every row's `_time` is a Unix timestamp in seconds, captured when its open row
was first created. Later logs merged into that row do not replace the timestamp,
and committing or flushing it does not change it.

`no_data(duration)` uses the actual time the latest metric row was accepted,
not its `_time`, so completing a long training step does not immediately look
stale. Uncommitted updates, trace-only events and rejected rows do not refresh
this commit clock. `age(metric)` uses the sample timestamp from the row instead.
On resume, an invalid or non-finite timestamp is skipped for alert-window
backfill with a warning, rather than preventing later valid samples from being
restored. The original history rows remain available for inspection.

## Step ordering

Steps accept Python integers and integer-index values such as NumPy integer
scalars. They are normalized to native integers before encoding. Strings and
floating-point steps are rejected before payload consumption or open-row mutation,
so an invalid step cannot poison later logs.

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
    are always step-ordered. A positive `n` combined with `step_range` selects the
    highest matching steps. Live and offline queries use the same selection rules.

Patch lines are merged in full even when separated by other steps or split across
cache and disk. Reads pin an append-prefix boundary and do not flush unrelated
span records or wait for commit callbacks. A damaged or unreadable metadata
sidecar produces a warning and a scan of the readable data file.

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
JSON integers too large for floating-point alert buffers remain unchanged in
history. They register their metric name but are skipped by numeric series with
a warning; neighboring metrics and their alerts continue to work.
Filesystem-surrogate characters in strings and paths are written as JSON escapes
rather than invalid UTF-8, so they do not discard neighboring metrics. Ordinary
Unicode remains readable UTF-8. Config replacement is atomic: a failed write
preserves the previous config file.

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
