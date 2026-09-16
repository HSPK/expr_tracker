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
 "time_ms/forward": 31.2,
 "time_ms/forward/attention": 18.4,
 "time_ms/forward/mlp": 9.1,
 "time_ms/backward": 44.7,
 "loss": 0.31}
```

## How a span metric is named

```
<what was measured>/<the span's path>
```

Nested span names join with `/`, so `forward/norm` and `backward/norm` stay
distinct, and the alert language reads them directly.

The measurement comes **first**, not last, because a tracker UI groups metrics
by the segment before the first slash. Naming them `forward/time_ms` would file
every timing under whatever it timed — a span called `train` would put its
duration in the same group as `train/loss`, and a deep tree would scatter
timings across the whole metric namespace. This way every timing shares one
`time_ms` group, every count a `count` group, and your own metric names stay
yours.

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

The same span several times in one step sums its durations:

```python
for layer in layers:            # 32 layers
    with et.span("layer"):
        x = layer(x)
```

```jsonc
{"time_ms/layer": 412.8}
```

Counts are disabled by default. If you need the number of calls or the mean
duration, enable them for the run:

```python
et.init(project="demo", span_count=True)
```

The row then also contains `"count/layer": 32`; divide `time_ms/layer` by it for
the mean. Disabling counts does not affect duration accumulation, plugin metrics
or individual records in the span file.

## A span never commits a step

Durations join whatever `log()` commits, so a span costs no row of its own and
you can time things before you know what to log. Open rows never time out, so long
training steps do not get split by a timer:

```python
with et.span("forward"):
    ...
# no metric row committed yet
et.log({"loss": loss})     # one row, with the metrics and the durations
```

A span's `_step` is assigned when it ends, not when it begins. Close all spans
for an iteration before calling `log()` to commit it; a span that ends after that
commit belongs to the next step. The name `"step"` is just a name, and
`et.span(..., step=N)` only adds `args.step` to its record.

To select a history step explicitly, open it before recording the spans:

```python
et.log({}, step=100)
with et.span("work"):
    process(batch)
et.log({})  # commits step 100, including the span metrics
```

### A process that only records spans

After `et.init()`, a new run starts at step 0; a resumed run starts one past its
highest recorded metric step. Without `log()` or another explicit commit, every
completed span stays on that step:

- `spans.jsonl` keeps one record per span, with its own timestamps and duration.
- The open metric row sums repeated paths. `finish()` or normal process exit
  commits that single aggregate row; metrics are not committed periodically.

For one metric row per iteration, commit after the spans close, even if there
are no other metrics:

```python
for batch in loader:
    with et.span("work"):
        process(batch)
    et.log({})  # commit the span metrics and advance the step
```

Each process/stream has its own step cursor; another process logging metrics does
not advance this one. Without an active run, spans still time and optionally print
but write no records.
After `fork`, create a new run and new plugins in the child; inherited client
objects are rejected before accessing their locks. See [Streams](streams.md).

If the process only needs a timeline, disable metric aggregation:

```python
et.init(project="demo", stream="worker", span_metrics=False, backends=[])
```

Every completed span and its plugin measurements still reach the span file, but
no `time_ms`, count or plugin metrics are retained in an open history row.
This avoids metric-state growth even when span names are unique per request.
Ordinary `et.log()` calls continue to work. With aggregation enabled, keep
request IDs in attributes rather than creating a new metric path for every ID;
`cache_bytes` does not cap the open row.

## Attributes

Attributes describe an individual span. They reach `spans.jsonl`, not the metrics,
because they are usually not numbers:

```python
with et.span("load", batch_size=32) as span:
    rows = read()
    span.set(rows=len(rows))
```

Attributes use the same value encoding as `et.log()`, including NumPy values,
paths and datetimes. The span object's attributes retain their original values;
the file receives an encoded copy.

## Alerting on durations

Because a duration is a metric, the [expression language](../reference/expressions.md)
already handles it:

```python
et.init(..., alert_rules=[
    "mean(data/load_ms[50]) > 200 => warning: data loading is slowing down",
    "time_ms/forward > 3 * mean(time_ms/forward[100]) => error: slow step",
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

The span file flushes independently of metric commits, on a reusable background
worker by default. The first completed span requests an immediate write;
subsequent records use `buffer_size`, `buffer_interval` and a
`max_buffer_seconds` deadline (5 seconds by default). Even if no more spans or
logs arrive, the worker flushes pending span records without advancing the step.
Set `max_buffer_seconds=None` to disable timed flushing, or
`span_async_write=False` to perform due writes on the producer thread.

`et.get_run().history.flush()` flushes completed span records without committing
the open metric row and waits for the write. `finish()` drains accepted writes
before closing history, summary, alerts and backends; normal process exit also
flushes the files. A span that has not ended has no complete record to flush yet.

Buffering is bounded by both record count and encoded bytes. By default, one
record is limited to 8 MiB and queued plus in-flight data to 64 MiB. Large
attributes may cause a span event to be rejected with an error and a dropped
counter; its valid timing metrics still aggregate. Configure `max_record_bytes`
and `max_pending_bytes` when larger records are intentional.

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
et trace runs/llm/sft-1 --time-range 1700000000:1700000060 -o minute.json
et trace runs/llm/sft-1 --temp-dir /scratch/trace-spool -o trace.json
```

`--time-range` uses absolute epoch seconds and selects spans overlapping the
half-open interval, without clipping them. Either bound can be omitted; spans
ending at the lower bound or starting at the upper bound are excluded. Empty or
reversed ranges select nothing. With a time filter, zero-duration spans are
excluded; without one they are retained. Time and step filters combine with AND.

Each stream becomes a process and each thread a track, so a data worker and a
training loop sit on one timeline and the gap where one waited for the other is
visible. Spans keep their nesting, their step and their attributes.

Exporting a standard format rather than drawing our own view means the result
can be loaded beside a `torch.profiler` trace, which is usually where the real
question is: what were the GPUs doing while the loader stalled.

The CLI streams output and uses a temporary SQLite spool to sort completion-order
records without loading the whole input into Python. Lane allocation is
O(n log n) and keeps unrelated top-level spans separate. Memory depends on the
largest record, stream/thread metadata and peak overlap, not all historical
events in a sequential worker; temporary disk usage grows with selected data.
Nonempty filters still scan the input. `build_trace()` returns an in-memory dict,
so use `write_trace()` or the CLI for large exports.

`--temp-dir` (Python: `temp_dir=`) selects an existing directory for the temporary
spool database. Use a disk-backed scratch volume when the system temporary
directory is small or memory-backed. The temporary output stays beside the final
destination for atomic replacement, so both volumes need enough free space.
SQLite's internal sort-file location remains environment/platform dependent;
this option does not change process-global SQLite settings. Temporary artifacts
are removed on failure, and an existing destination is preserved.
Non-finite attribute/plugin values are rendered as `"NaN"`, `"Infinity"` or
`"-Infinity"` strings in the exported trace, so it remains standard JSON while
retaining diagnostic information. The original span and metric files are unchanged.

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

Pass `print_fn` and the span announces itself, indented two spaces per level:

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
in tests. The indent is spaces rather than a tab so that it survives a log
prefix: a terminal measures tab stops from the start of the line, so behind
`... | INFO | ` the first level would collapse to whatever is left of the stop. It is called on the thread that ran the span, and an exception in it is
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
# {"time_ms/forward": 41.2,
#  "cpu_percent/forward": 101.2, "cpu_time_ms/forward": 41.4,
#  "gpu_mem_peak_mb/forward": 456.1, "gpu_mem_delta_mb/forward": 8.1, ...}
```

Like `print_fn`, plugins are inherited by children, can be overridden per span,
and can be set run-wide with `span_plugins`. They are opt-in because they are not
free. A plugin that raises is logged and skipped; it never breaks the span or the
code being measured, and a plugin that fails on every span is logged only once.
Plugin metrics also reach `spans.jsonl` and show up in the `et trace` viewer
under a span's *Arguments*.
With an active run, plugin results are normalized using the same per-field
encoding as `et.log()` before aggregation. NumPy scalars therefore aggregate
correctly, and an unsupported value falls back to a warned representation
instead of poisoning the whole metric row. Bare spans without attributes or
plugins still bypass this value encoder.

`time_ms` and `count` belong to the span. A plugin returning either has that key
dropped with a warning, so nothing can quietly redefine what a span measured. Any
other key becomes its own group: a plugin returning `gpu_mem_peak_mb` writes
`gpu_mem_peak_mb/<path>`.

### Repeated plugin measurements

Within one open step row, repeated measurements of the same path use these rules:

| Measurement | Aggregation |
| --- | --- |
| `time_ms`, `cpu_time_ms`, `gpu_mem_delta_mb`, optional `count` | sum |
| `gpu_mem_peak_mb`, `gpu_mem_used_mb` | maximum |
| `cpu_percent`, `gpu_percent` | mean weighted by each span's duration |
| other custom plugin metrics | sum, preserving the default plugin contract |

These rules follow the measurement names, so a custom plugin using one of the
named GPU/CPU metrics gets the same semantics. A missing measurement contributes
no weight. Zero-duration spans do not affect a positive-duration mean; if all
durations are zero, the latest value is retained.

Aggregations reset when the row commits. A regular `et.log()` overwrite also
resets the weighted aggregate for that metric. Individual span objects and
`spans.jsonl` retain each encoded measurement separately, not the aggregate.
A non-numeric replacement resets that metric's numeric aggregation state; omit
a key entirely when a plugin has no measurement for that span.

### Built-ins

| plugin | metrics | needs |
| --- | --- | --- |
| `CpuTime()` | `cpu_time_ms`, `cpu_percent` | nothing |
| `TorchMemory(device=None)` | `gpu_mem_peak_mb`, `gpu_mem_delta_mb` | `torch` + CUDA |
| `GpuStats(index=0, interval=0.1)` | `gpu_percent`, `gpu_mem_used_mb` | `pip install expr-tracker[gpu]` |

`CpuTime` divides process CPU time by wall time, so `cpu_percent` above 100 means
the span used more than one core. It counts every thread in the process, so
unrelated concurrent work inflates it, and it is noise on spans under a
millisecond. Start times are paired by span identity, so overlapping asyncio
tasks and spans ended out of order do not exchange their CPU measurements.

`TorchMemory` reports what your own allocator held, which is the number that
predicts an OOM. `gpu_mem_peak_mb` is absolute, so memory a tensor was already
holding when the span opened counts towards it; `gpu_mem_delta_mb` is what the
span kept. `torch.cuda.max_memory_allocated` is a single global high-water mark.
Before resetting it, the plugin preserves the previous peak for every active
span on that device, across plugin instances, threads and tasks. Each span keeps
the device it started on. Resets performed outside the plugin cannot be accounted
for.

`GpuStats` reads the whole device, including other processes. A background thread
samples NVML every `interval`, and a span reports the mean utilisation and the
peak memory over its own window, so the span itself pays almost nothing. Spans
shorter than `interval` contain no sample and fall back to one live reading,
which for utilisation is NVML's own rolling average rather than a measurement of
that span. Samplers are shared across plugins with the same device and interval.
Each active span retains a running sum, sample count and peak, so measurements
cover the entire span even when it lasts hours. Memory use depends on active
spans rather than elapsed time; the bounded recent-sample deque is diagnostic
only. Stopping or failing the sampler invalidates unfinished measurements rather
than reporting a partial window as complete.

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

Default span recording keeps timing, plugin hooks and serialization on the
producer, while a reusable worker handles disk writes. Measure producer latency
and throughput including the final flush separately: enqueue speed alone does
not measure disk throughput.

```bash
uv run pytest tests/test_benchmark.py -k "span_throughput or trace_only_unique" -s
```

The report includes latency percentiles, drained throughput, worker thread starts
and retained memory for 10,000 unique trace-only spans. Absolute timings depend
on the filesystem, CPU, attributes, printing and enabled plugins.

Use `span_metrics=False` for timeline-only workers, or `spans=False` when only
aggregate metrics are needed. Put `TorchMemory` on the regions whose memory you
actually need rather than run-wide. Each NVML sample updates active windows, so
its sampling cost scales with concurrent active spans, not how long they run.
