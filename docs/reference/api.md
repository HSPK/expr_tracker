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
    alert=None,
    alert_rules: Sequence = (),
    **history_options,
) -> Run
```

Starts a run and publishes it as the process-wide current run. `name` defaults to a
timestamp. `dir` defaults to `./tracker/jsonl`. Extra keyword arguments are
[history options](configuration.md#history-options).

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
)
```

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
