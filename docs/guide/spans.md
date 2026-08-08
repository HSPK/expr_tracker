# Spans

A step is rarely one thing. `et.span` times the parts, and their parts, and turns
each duration into an ordinary metric — so `history()`, alert rules and plots all
work on it with no extra machinery.

```python
for batch in loader:
    with et.span("forward"):
        with et.span("attention"):
            attn = attention(batch)
        with et.span("mlp"):
            out = mlp(attn)
    with et.span("backward"):
        loss.backward()
    et.log({"loss": loss.item()})
```

The step's row then carries:

```jsonc
{"_step": 42, "_time": ...,
 "forward/duration_ms": 31.2,  "forward/count": 1,
 "forward/attention/duration_ms": 18.4, "forward/attention/count": 1,
 "forward/mlp/duration_ms": 9.1, "forward/mlp/count": 1,
 "backward/duration_ms": 44.7, "backward/count": 1,
 "loss": 0.31}
```

Nested names join with `/`, so `forward/norm` and `backward/norm` stay distinct,
and the alert language reads them directly.

## Forms

```python
with et.span("forward"): ...                # context manager

async with et.span("fetch"): ...            # async

@et.span("preprocess")                      # decorator, sync or async
def preprocess(batch): ...

span = et.start_span("epoch")               # manual, for crossing scopes
...
span.end()
```

## Repeated spans

The same span several times in one step sums, and counts:

```python
for layer in layers:            # 32 layers
    with et.span("layer"):
        x = layer(x)
```

```jsonc
{"layer/duration_ms": 412.8, "layer/count": 32}
```

The total is usually what you want; divide by the count for the mean.

## A span never commits a step

Durations join whatever `log()` commits, so a span costs no row of its own and
you can time things before you know what to log:

```python
with et.span("forward"):
    ...
# nothing written yet
et.log({"loss": loss})     # one row, with the metrics and the durations
```

## Attributes

Attributes describe an individual span. They reach `spans.jsonl`, not the metrics,
because they are usually not numbers:

```python
with et.span("load", batch_size=32) as span:
    rows = read()
    span.set(rows=len(rows))
```

## Alerting on durations

Because a duration is a metric, the [expression language](../reference/expressions.md)
already handles it:

```python
et.init(..., alert_rules=[
    "mean(data/load_ms[50]) > 200 => warning: data loading is slowing down",
    "forward/duration_ms > 3 * mean(forward/duration_ms[100]) => error: slow step",
])
```

## The span file

The full tree is written to `spans.jsonl` beside the metrics, one line per span:

```jsonc
{"_step": 42, "name": "forward/attention", "depth": 1,
 "start": 1754323200.123456, "dur_ms": 18.4, "args": {"batch_size": 32}}
```

Children appear before their parents, because a parent closes last. It follows
the [stream](streams.md): a data worker writes `spans.data.jsonl`.

Turn it off if you only want the metrics:

```python
et.init(..., spans=False)
```

## Viewing the timeline

`et trace` turns the span file into a Chrome Trace, which
[Perfetto](https://ui.perfetto.dev) and `chrome://tracing` open directly:

```bash
et trace runs/llm/sft-1 -o trace.json
et trace runs/llm/sft-1 --stream data --step-range 100:200
```

Each stream becomes a process and each thread a track, so a data worker and a
training loop sit on one timeline and the gap where one waited for the other is
visible. Spans keep their nesting, their step and their attributes.

Exporting a standard format rather than drawing our own view means the result
can be loaded beside a `torch.profiler` trace, which is usually where the real
question is: what were the GPUs doing while the loader stalled.

## Errors

An exception is recorded and re-raised — the span never swallows it:

```jsonc
{"_step": 42, "name": "risky", "dur_ms": 3.1, "error": "ValueError"}
```

A failure inside the recording itself is logged and ignored: measuring something
must not break it.

## Threads and tasks

The nesting stack is per thread and per asyncio task, so concurrent work does not
nest inside unrelated spans:

```python
async def work(name):
    async with et.span(name):        # "a" and "b", never "a/b"
        await asyncio.sleep(1)

await asyncio.gather(work("a"), work("b"))
```

## Cost

| | per span |
| --- | --- |
| default | ~13 µs |
| `spans=False` | ~6 µs |
| no active run | ~3 µs |

For comparison, `et.log()` is ~24 µs. Twenty spans on a 100 ms step is 0.26% of
the step. If your step is closer to a millisecond, set `spans=False` and keep the
metrics, or time fewer regions.
