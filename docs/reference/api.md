# Python API

Everything below is exported from `expr_tracker`.

```python
et.__version__   # the installed version
```

## Lifecycle

### `init`

```python
et.init(
    project: str,
    name: str | None = None,
    entity: str | None = None,
    dir: str | None = None,
    run_dir: str | None = None,
    notes: str | None = None,
    tags: list[str] | None = None,
    resume: bool | str | None = "allow",
    config: dict | None = None,
    backends: Sequence[str] = ("wandb", "jsonl"),
    backend_kwargs: dict[str, dict] | None = None,
    print_to_screen: bool = False,
    stream: str | None = None,
    alert=None,
    alert_rules: Sequence = (),
    alert_on_rank: int | None = 0,
    backend_on_rank: int | None = 0,
    **history_options,
) -> Run
```

Starts a run and publishes it as the process-wide current run. `name` defaults to a
timestamp. Extra keyword arguments are
[history options](configuration.md#history-options).

`dir` is a root and defaults to `./tracker/jsonl`, so the run lands in
`<dir>/<project>/<name>` and artifacts in `<dir>/<project>/artifacts`. `run_dir`
names this run's directory outright, putting its artifacts in
`<run_dir>/artifacts`; passing both is an error. The resolved path is logged at
`init()` and available as `Run.dir`.

`resume` is forwarded to wandb and trackio. The local history always continues an
existing run directory, so re-running with the same project and name picks up the
step cursor whatever `resume` says; use a new name to start clean.

`alert_on_rank` and `backend_on_rank` pick which rank alerts and which opens the
remote backend run; `None` means every rank. See
[Distributed runs](../guide/distributed.md).

Calling `init()` twice without `finish()` raises.

### `finish`

```python
et.finish(exit_code: int | None = None, quiet: bool | None = None)
```

Commits the open row, saves the summary, stops the alert dispatcher and closes every
backend. Idempotent. `exit_code` and `quiet` exist for wandb compatibility.

### `get_run`

```python
et.get_run() -> Run | None
```

The current run, or `None`. Every other module-level function raises `RuntimeError`
if there is no run.

## Metrics

### `log`

```python
et.log(data: dict, step: int | None = None, commit: bool | None = None)
```

See [Logging metrics](../guide/logging.md).

### `history`

```python
et.history(
    n: int | None = 50,
    *,
    output_type: str = "dict",
    metrics: Sequence[str] | None = None,
    step_range: tuple[int | None, int | None] | None = None,
    include_meta: bool = True,
    include_open: bool = True,
    fill_missing: bool = False,
    dropna: bool = False,
    run: str | Path | None = None,
    stream: str | None = ...,
)
```

Omit `stream` to read the running one; pass `None` for the default producer. See
[Streams](../guide/streams.md).

`n=-1` or `None` returns everything. With `run=`, reads that run offline and no
`init()` is needed. See [Querying history](../guide/history.md).

### `summary`

```python
et.summary()   # a MutableMapping
```

Last value per metric, plus anything you assign. Explicit assignments win. Persisted
to `summary.json`.

### `define_metric`

```python
et.define_metric(name: str, **kwargs)
```

Forwarded to backends that support it; a no-op locally.

## Alerts

### `alert`

```python
et.alert(
    title: str,
    text: str = "",
    subtitle: str | None = None,
    traceback: str | None = None,
    level: str | AlertLevel = "info",
    channels: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    mentions: Sequence[str] | None = None,
    fields: dict | None = None,
    link: str | None = None,
    dedup_key: str | None = None,
)
```

Sends a one-off alert. Never raises; delivery failures are logged.

### Rules

```python
et.add_alert_rule(rule, **overrides) -> AlertRule
et.remove_alert_rule(name: str) -> bool
et.list_alert_rules() -> list[AlertRule]
```

`rule` is a string, a dict, an `AlertRule`, or an `M` builder expression.
Overrides apply to rule objects as well as text/dict/builder inputs. A rule
object with overrides is copied and validated, not mutated; without overrides
the original object is returned. Auto-generated names remain stable for
enablement-only changes and change when the alert's identity changes.

### `configure_alert`

```python
et.configure_alert(channels=..., policy=..., rules=..., enabled=True)
```

Sets process-wide alert configuration, used by runs that do not pass `alert=`.

### `register_backend`

```python
et.register_backend(kind: str, cls: type[AlertBackend])
```

Registers a custom channel type.

## Spans

```python
with et.span(name, print_fn=None, plugins=(), **attributes) as span: ...
span = et.start_span(name, print_fn=None, plugins=(), **attributes)
span.set(**attributes)
span.duration_ms
span.metrics                                    # what the plugins measured
```

`et.span` is also an async context manager and a decorator. A closed span adds
`time_ms/<path>` and one key per plugin metric to the open row,
and appends the full record to `spans.jsonl`. The measurement leads the name so
that a UI groups every timing together instead of filing each one under whatever
it timed; `Span.duration_ms` is the same number on the object.

Counts are disabled by default. Pass `span_count=True` to `et.init()` to also
accumulate `count/<path>`. This does not change step assignment or the span file.
Repeated CPU/GPU utilization measurements are duration-weighted means, and
memory peaks are maxima; durations, deltas and other custom metrics still sum.

Spans record their history step when they end, not when they begin, and never
advance it themselves. `et.span(..., step=N)` only stores an attribute; it does not
set the history step. See [Spans](../guide/spans.md) for span-only processes.
The span file has its own reusable background writer and flushes even without
metric commits. `span_async_write=False` selects synchronous span writes;
`span_metrics=False` records only span events without metric aggregation.

`print_fn(line)` announces the span's start and end, indented two spaces per level.
`plugins` measure a resource across the span; both are inherited by child spans
and default to the run's `span_print_fn` and `span_plugins`.

### Plugins

```python
from expr_tracker.plugins import CpuTime, GpuStats, TorchMemory
```

| plugin | metrics |
| --- | --- |
| `CpuTime()` | `cpu_time_ms`, `cpu_percent` |
| `TorchMemory(device=None)` | `gpu_mem_peak_mb`, `gpu_mem_delta_mb` |
| `GpuStats(index=0, interval=0.1)` | `gpu_percent`, `gpu_mem_used_mb` |

Any object with `start(span)` and `end(span) -> dict` is a plugin, as is a plain
`fn(span) -> dict`. See [Spans](../guide/spans.md).

### Trace export

```python
from expr_tracker.trace import build_trace, write_trace

write_trace(
    run, "trace.json", stream="*", step_range=None, time_range=None, temp_dir=None
)  # returns the span count
build_trace(
    run, stream="*", step_range=None, time_range=None, temp_dir=None
)  # the dict, unwritten
```

`time_range=(low, high)` uses absolute epoch seconds and selects positive-duration
spans overlapping `[low, high)` without clipping. Either bound can be `None`;
equal/reversed bounds select nothing. It combines with `step_range` using AND.
All streams are aligned to the earliest selected start. `write_trace()` streams
through temporary disk storage and atomically replaces the output; `build_trace()`
retains its returned event list in memory.
`temp_dir` selects an existing directory for the temporary spool database; the
default is the system temporary directory. Output staging remains on the
destination filesystem for atomic replacement.
Non-finite arguments are exported as diagnostic strings rather than invalid JSON
numeric constants; original span records are not changed.

## Artifacts

```python
et.log_artifact(
    artifact_or_path: Artifact | str,
    name: str | None = None,
    type: str | None = None,
    aliases: list[str] | None = None,
    metadata: dict | None = None,
    mode: str = "copy",
) -> Artifact

et.use_artifact(artifact_or_name: Artifact | str, type: str | None = None) -> Artifact
```

`Artifact` methods: `add_file(path, name=None)`, `add_dir(path)`,
`add_reference(uri)`, `download(root=None)`, `get_path(name)`, `files()`,
`qualified_name`. See [Artifacts](../guide/artifacts.md).

## Introspection

### `info`

```python
et.info() -> dict
```

```python
{
  "history":   {"log_dir", "metrics_file", "rows_on_disk", "rows_logged",
                "last_step", "cached_rows", "cached_bytes", "cache_limit_bytes",
                "evicted_rows", "disk_prefix", "queries", "disk_queries"},
  "artifacts": {"root"},
  "summary":   {...},
  "rank":      0,
  "alerts":    {"rules":    {"<rule>": {"fires", "firing", "enabled",
                                        "unresolved_metrics"}},
                "channels": {"<channel>": {"sent", "failed", "suppressed",
                                           "dropped", "pending", "alive"}}},
}
```

### `Run`

Attributes: `project`, `name`, `config`, `rank`, `backends`, `history`, `summary`,
`artifacts`, `alerts`.
Properties: `step`, `dir`, `url`.

## Offline reading

```python
from expr_tracker.history import read_history, resolve_run_path

read_history("tracker/jsonl/demo/run-1", -1)
```

Same keyword arguments as `et.history`, without needing a run.
