# Examples

Every example runs offline: no account, no network, no GPU required. They write
to `runs/` by default, which is gitignored — pass `--dir` to put them elsewhere.

```bash
uv run python examples/quickstart.py
uv run python examples/<name>.py --help      # every example takes arguments
```

## [`quickstart.py`](https://github.com/HSPK/expr_tracker/blob/main/examples/quickstart.py)

The sixty-second tour. Logs a training run, merges a sparse eval metric into the
training step with `commit=False`, queries the history while the run is still
open, and reads it back afterwards from the directory.

## [`alert_rules.py`](https://github.com/HSPK/expr_tracker/blob/main/examples/alert_rules.py)

Four rules against four faults: a loss spike caught by z-score, a non-finite
loss, a curve that goes flat, and an accuracy regression that has to persist for
three steps before it counts.

```bash
uv run python examples/alert_rules.py --fault spike   # or nan, stall, none
```

`--fault none` sends nothing. That is the interesting case: warm-up, missing
data and NaN all evaluate to UNKNOWN rather than False, so rules cannot cry wolf
before they have the evidence.

## [`profile_step.py`](https://github.com/HSPK/expr_tracker/blob/main/examples/profile_step.py)

Where a training step actually goes. Nested spans become metrics on the step's
row, so timings query and alert like any other metric; a plugin attaches CPU
cost; the tree exports to a Chrome Trace for Perfetto.

```bash
uv run python examples/profile_step.py --print-spans
```

## [`early_stopping.py`](https://github.com/HSPK/expr_tracker/blob/main/examples/early_stopping.py)

The loop reading its own history to decide what to do next: decay the learning
rate when the eval metric plateaus, and stop once decaying no longer pays. This
is what a local, queryable history buys you over shipping metrics out.

## [`checkpoints.py`](https://github.com/HSPK/expr_tracker/blob/main/examples/checkpoints.py)

Checkpoints as artifacts — versioned, deduplicated by content, aliased `best`,
and fetched by a later run that knows nothing about which run wrote them.

## [`multiprocess_pipeline.py`](https://github.com/HSPK/expr_tracker/blob/main/examples/multiprocess_pipeline.py)

Four data producers and four trainers as eight processes sharing one run, with a
queue that lets a producer run at most `--staleness` batches ahead. Each worker
writes its own stream and gets its own lane in the trace, and the blocking spans
show which side is the bottleneck.

```bash
# producers faster than trainers: they stall on a full queue
uv run python examples/multiprocess_pipeline.py --produce-ms 10 --train-ms 40

# trainers faster than producers: they starve waiting for batches
uv run python examples/multiprocess_pipeline.py --produce-ms 40 --train-ms 10
```
