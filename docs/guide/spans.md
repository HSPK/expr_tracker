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

## Printing a span as it runs

Pass `print_fn` and the span announces itself, indented one tab per level:

```python
with et.span("step", print_fn=print):
    with et.span("forward"):
        with et.span("attention"):
            ...
```

```
-> step  16:41:38
	-> forward  16:41:38
		-> attention  16:41:38
		<- attention  16:41:38  3.074ms
	<- forward  16:41:38  5.481ms
<- step  16:41:38  9.641ms
```

Children inherit the handler, so one argument on the outermost span prints the
whole tree. A child that passes its own `print_fn` takes over its subtree; pass
`print_fn=lambda line: None` to silence one. A failed span ends with `!ValueError`.

Indentation is relative to the span that started printing, not to absolute
nesting depth, so turning printing on deep inside a call stack still gives you a
tree rooted at the left margin. A new thread begins its own tree, because the
nesting stack does not cross threads. Concurrent asyncio tasks share one handler
and interleave their lines; each is still indented by its own depth, and the
`->` and `<-` markers pair them up.

Any callable taking one string works — `print`, `logger.info`, a list's `append`
in tests. It is called on the thread that ran the span, and an exception in it is
logged and swallowed.

Set it for the whole run instead of per span with `span_print_fn`:

```python
et.init(project="demo", span_print_fn=logger.info)
```

## Plugins

A plugin measures a resource across a span. It is any object with
`start(span)` and `end(span) -> dict`; a plain `fn(span) -> dict` is an
end-only plugin. Whatever `end` returns is merged into the span's metrics under
the span's own path:

```python
from expr_tracker.plugins import CpuTime, GpuStats, TorchMemory

with et.span("forward", plugins=[CpuTime(), TorchMemory()]):
    loss = model(batch)
```

```python
et.history(1)[0]
# {"forward/duration_ms": 41.2, "forward/count": 1,
#  "forward/cpu_percent": 101.2, "forward/cpu_time_ms": 41.4,
#  "forward/gpu_mem_peak_mb": 456.1, "forward/gpu_mem_delta_mb": 8.1, ...}
```

Like `print_fn`, plugins are inherited by children, can be overridden per span,
and can be set run-wide with `span_plugins`. They are opt-in because they are not
free. A plugin that raises is logged and skipped; it never breaks the span or the
code being measured. Plugin metrics also reach `spans.jsonl` and show up in the
`et trace` viewer under a span's *Arguments*.

### Built-ins

| plugin | metrics | needs |
| --- | --- | --- |
| `CpuTime()` | `cpu_time_ms`, `cpu_percent` | nothing |
| `TorchMemory(device=None)` | `gpu_mem_peak_mb`, `gpu_mem_delta_mb` | `torch` + CUDA |
| `GpuStats(index=0, interval=0.1)` | `gpu_percent`, `gpu_mem_used_mb` | `pip install expr-tracker[gpu]` |

`CpuTime` divides process CPU time by wall time, so `cpu_percent` above 100 means
the span used more than one core. It counts every thread in the process, so
unrelated concurrent work inflates it, and it is noise on spans under a
millisecond.

`TorchMemory` reports what your own allocator held, which is the number that
predicts an OOM. `gpu_mem_peak_mb` is absolute, so memory a tensor was already
holding when the span opened counts towards it; `gpu_mem_delta_mb` is what the
span kept. `torch.cuda.max_memory_allocated` is a single global high-water mark,
so a child span resetting it would erase its parent's history — the plugin keeps
its own stack and hands each parent back the peak it had reached before the child
opened.

`GpuStats` reads the whole device, including other processes. A background thread
samples NVML every `interval`, and a span reports the mean utilisation and the
peak memory over its own window, so the span itself pays almost nothing. Spans
shorter than `interval` contain no sample and fall back to one live reading,
which for utilisation is NVML's own rolling average rather than a measurement of
that span. Samplers are shared across plugins with the same device and interval.

### CUDA is asynchronous

A span measures the wall time of the Python block, and CUDA kernels are queued,
not awaited. A span around a launch can close in microseconds while the GPU is
still busy, and the wait then lands on whatever later line synchronises. If you
want a span to mean GPU time, synchronise inside it:

```python
with et.span("forward", plugins=[TorchMemory()]):
    out = model(batch)
    torch.cuda.synchronize()
```

`gpu_mem_peak_mb` is unaffected: allocator bookkeeping is recorded at launch.

## Cost

| | per span |
| --- | --- |
| default | ~15 µs |
| `spans=False` | ~6 µs |
| no active run | ~3 µs |
| `print_fn` | +6 µs |
| `CpuTime()` | +8 µs |
| `GpuStats()` | +4 µs, or +36 µs on a span shorter than `interval` |
| `TorchMemory()` | +62 µs |

For comparison, `et.log()` is ~24 µs. Twenty spans on a 100 ms step is 0.3% of
the step. If your step is closer to a millisecond, set `spans=False` and keep the
metrics, or time fewer regions. Put `TorchMemory` on the few spans whose memory
you actually care about rather than run-wide.
