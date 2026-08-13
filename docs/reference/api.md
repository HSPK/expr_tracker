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
timestamp. `dir` defaults to `./tracker/jsonl`. Extra keyword arguments are
[history options](configuration.md#history-options).

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
`<path>/duration_ms`, `<path>/count` and one key per plugin metric to the open
row, and appends the full record to `spans.jsonl`.

`print_fn(line)` announces the span's start and end, indented one tab per level.
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

write_trace(run, "trace.json", stream="*", step_range=None)   # returns the span count
build_trace(run, stream="*")                                  # the dict, unwritten
```

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
