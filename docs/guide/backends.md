# Backends

Local JSONL history is always on. Remote backends are optional mirrors: they receive
the same metrics, and none of them can break the run if they fail.

```python
et.init(project="demo", backends=["wandb"])            # local + wandb
et.init(project="demo", backends=["wandb", "trackio"]) # local + both
et.init(project="demo", backends=[])                   # local only
```

`"jsonl"` is accepted for backward compatibility and ignored — local history is not
optional.

## What gets forwarded

| Call | Forwarded as |
| --- | --- |
| `et.log(data, step, commit)` | `backend.log(data, step=<resolved>, commit=<resolved>)` |
| `et.init(...)` | `backend.init(project, name, config, entity, tags, notes, resume, ...)` |
| `et.define_metric(name, **kw)` | `backend.define_metric` if it has one |
| `et.log_artifact(...)` | `backend.log_artifact` if it has one |
| `et.finish(exit_code)` | `backend.finish()` |

The **resolved** step matters: two `log()` calls for one step stay one step on the
backend too, and a step dropped by the local step policy is never forwarded. A
backend that does not accept `commit` (trackio) simply does not receive it.

## Failure handling

- A backend that cannot be imported is skipped with the exact install command.
- A backend whose `init()` fails is dropped; the run continues without it.
- A failing `log()`, `define_metric()`, `log_artifact()` or `finish()` is logged and
  ignored.

In every case the local history is complete.

## Per-backend options

```python
et.init(
    project="demo",
    backends=["wandb"],
    backend_kwargs={"wandb": {"group": "ablation", "job_type": "train"}},
)
```

## Custom backends

Anything with `init`, `log` and `finish` works:

```python
class MyBackend:
    def init(self, **kwargs): ...
    def log(self, data, step=None, commit=None): ...
    def finish(self, **kwargs): ...

et.init(project="demo", backends=[MyBackend()])
```

It is registered under its lowercased class name.

## wandb compatibility

Migrating an existing script is usually one line:

```python
# import wandb as et
import expr_tracker as et
```

| wandb | expr_tracker | Notes |
| --- | --- | --- |
| `init(project, name, config, tags, notes, entity, dir, resume)` | same | plus `alert`, `alert_rules`, history tuning |
| `log(data, step, commit)` | same | identical semantics |
| `finish(exit_code, quiet)` | same | arguments kept for drop-in use |
| `alert(title, text, level)` | `alert(..., channels=...)` | a superset |
| `log_artifact` / `use_artifact` / `Artifact` | same | local implementation |
| `define_metric(name, ...)` | same | forwarded; a no-op locally |
| `wandb.run` | `et.get_run()` | the current `Run` |
| `run.summary` | `et.summary()` | persisted as `summary.json` |
| `run.step` / `run.dir` / `run.url` | same | `url` needs the wandb backend |
| `wandb.Image` / `Table` / `Video` | not implemented | forwarded to wandb; stored locally as `repr` |
| `wandb.watch` / `sweep` / `save` | not implemented | out of scope |

!!! note
    Recent wandb API keys are 86 characters (`wandb_v1_...`) and need a current
    wandb release; older versions reject them and the backend is skipped with a
    warning.
