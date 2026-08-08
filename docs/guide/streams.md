# Streams

A training loop and a data worker measure different things on different clocks:
training step 100 and data batch 100 are unrelated. A **stream** gives each
producer its own step cursor and its own file inside one run directory.

```python
# the training process
et.init(project="llm", name="sft-1")
et.log({"train/loss": loss})                    # steps 0, 1, 2, ...

# the data worker, a separate process
et.init(project="llm", name="sft-1", stream="data")
et.log({"data/produce_ms": 12.4})               # its own steps 0, 1, 2, ...
```

Without this, both producers share one cursor: their steps interleave, the merge
puts unrelated metrics on one row, and `step_policy="monotonic"` drops whichever
producer happens to fall behind.

## Layout

```
runs/llm/sft-1/
├── metrics.jsonl            # the default producer
├── metrics.data.jsonl       # stream "data"
├── metrics.meta.json
├── metrics.data.meta.json
├── summary.json             # per stream, so processes cannot clobber each other
├── summary.data.json
├── config.json
└── config.data.json
```

Streams compose with [rank shards](distributed.md): rank 1 of the data worker
writes `metrics.data.rank1.jsonl`.

Stream names become part of a filename, so they must be letters, digits, `_` or
`-`, starting with a letter or digit. `rank1` and friends are rejected because
they already mean a rank shard.

## Reading

```python
et.history(50)                       # whichever stream this process writes
et.history(50, stream=None)          # the default producer
et.history(50, stream="data")        # another stream
et.history(50, run=path, stream="data")   # offline
```

Omitting `stream` reads the running stream; passing `stream=None` explicitly means
the default, unnamed one. Another process's stream is read from its file, so you
see what it has flushed rather than what it has buffered.

```python
from expr_tracker.history import list_streams
list_streams("runs/llm/sft-1")       # [None, "data"]
```

## Alerts

Each process alerts on what it can see, which is its own stream. That is usually
what you want — the data worker is the thing that knows its pipeline stalled:

```python
et.init(
    project="llm", name="sft-1", stream="data",
    alert_rules=[
        "produce_ms > 1000        => warning: data pipeline slow",
        "no_data(5m)              => error: data worker stopped producing",
    ],
)
```

!!! note
    A rule cannot span streams that live in different processes, because neither
    process holds the other's metrics. If you need that, log both from one
    process, or evaluate the rule downstream against the files.

## Backends

A stream is forwarded as its own backend run, grouped under the run name:

```python
et.init(project="llm", name="sft-1", stream="data", backends=["wandb"])
# -> wandb.init(name="sft-1-data", group="sft-1", job_type="data")
```

Both wandb and trackio understand `group`, and neither can merge two step axes
into a single run. wandb's shared mode can, but it needs a live server and has no
trackio equivalent, so grouping is the default.

Override it per backend if you want something else:

```python
et.init(..., stream="data", backend_kwargs={"wandb": {"group": "my-group"}})
```

## When you do not need a stream

If the producers are in **one process at different cadences** — an eval loop
every 100 steps, say — you do not need a stream. Log with the training step and
the sparse metric simply appears on the steps where you logged it:

```python
et.log({"train/loss": loss})
if step % 100 == 0:
    et.log({"eval/acc": acc}, step=step)
```

Window functions already work on that, because `eval.acc[20]` counts points of
that metric, not rows. Reach for a stream when producers are genuinely concurrent
and their step numbers mean different things.
