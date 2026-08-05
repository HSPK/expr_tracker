# Architecture

How the pieces fit together, and which module owns what. Pair this with
[`design.md`](design.md), which records *why* the data model looks the way it does.

## Module map

```
expr_tracker/
├── __init__.py        public names re-exported for `import expr_tracker as et`
├── tracker.py         the functional API (init/log/history/finish/artifacts/summary)
├── run.py             Run object + global singleton + backend fan-out
├── summary.py         run.summary mapping, persisted as summary.json
├── artifacts.py       Artifact + project-scoped ArtifactStore
├── encoders.py        JSON coercion (numpy/torch/pydantic/dataclasses/...)
├── cli.py             `et history` / `et rules` / `et alert`
├── history/
│   ├── store.py       HistoryStore: open-row assembly, cache, query planning
│   ├── writer.py      JsonlWriter: buffered append, sparse index, meta sidecar
│   ├── reader.py      JsonlReader + the offline read_history entry points
│   ├── codec.py       RecordCodec: metric values -> JSON lines, warn-once
│   ├── series.py      MetricSeries: per-metric numeric buffers
│   └── frame.py       projection + dict/pandas/polars output
└── alerts/
    ├── __init__.py    public alert API + config resolution + engine wiring
    ├── models.py      AlertLevel/Message/ChannelConfig/WebhookPolicy/AlertRule
    ├── engine.py      rule compilation, state machine, watchdog
    ├── dispatch.py    routing, rate limiting, dedup, retry, async worker
    ├── backends/      lark, slack, dingtalk, wecom, webhook, email, callable
    └── expr/          lexer, parser, AST nodes + builder, functions, evaluator
```

## Dependency direction

```
tracker.py ──> run.py ──> history/   (always)
                    └──> alerts/     (lazily, to avoid an import cycle)
                    └──> artifacts.py, summary.py
alerts/expr ──> history/series.py    (read-only: evaluation needs metric windows)
```

Rules:

* `history/` never imports from `alerts/`. The alert engine reads `MetricSeries`,
  which lives in `history/` because it is populated on the write path.
* `alerts/` never imports `run` at module scope; `alerts/__init__` imports it inside
  functions so `run.py` can import `alerts` lazily without a cycle.
* `expr/` is self-contained apart from `MetricSeries`, so the DSL can be parsed,
  validated and replayed without a live run (this is what `et rules test` uses).

## Write path

```
et.log(data, step, commit)
        │
        ▼
Run.log ──> Summary.observe(data)          # last value per metric
        └──> HistoryStore.log
                 │  encode once (jsonable_encoder)
                 │  merge into the open row for the current step
                 ▼  commit when the step advances / commit=True / finish / timeout
             HistoryStore._emit
                 ├── _store_row(record, line)         (holds the store lock)
                 │      ├── cache.append((row, step, line))  # ordinal allocated here
                 │      ├── MetricSeries.add(...)            # feeds alert evaluation
                 │      └── JsonlWriter.enqueue(step, row, line)
                 ├── writer.flush() / schedule_timer  (outside the lock)
                 ├── _evict()
                 └── _notify(record)                  # screen + alert engine
                 │
                 ▼
             on_commit ──> AlertEngine.on_step(record) ──> Dispatcher (async)
        └──> other backends (wandb, trackio, custom)
```

The ordinal allocation and the writer enqueue happen under one lock so that
row ordinal *N* is always physical line *N*; the query planner depends on it.

`HistoryStore.log` returns the `_step` the metrics landed on, or `None` when the call
was rejected. `Run.log` forwards that resolved step (and the resolved commit flag) to
the remote backends, so a backend's row layout matches the local history instead of
drifting on its own counter — two `log()` calls for one step stay one step everywhere.
Test the result with `is None`, since step 0 is falsy.

`log()` never branches on the open row directly: `_switch_open_row(step)` points it
at the requested step (or reuses it when no step is given) and hands back whatever row
that displaced. `JsonlWriter._track_line` is likewise the single place that folds a line
into the counters and index, shared by live appends and by resume rescans — so the
two can never drift apart.

## Read path

```
et.history(n, ...)
        │
        ▼
HistoryStore.get ──> _collect ──> _collect_tail | _collect_range
        │                 │              │
        │                 │              ├─ _view_*   one lock: cache rows + boundary
        │                 │              └─ _older_*  only when the view is incomplete
        │                 ├─ _open_rows   the uncommitted row, if wanted and in range
        │                 └─ _take_steps  merge by step, keep the newest n
        ▼
frame.project(...)  ──>  frame.to_output(dict | pandas | polars)
```

Both query kinds share one shape: take a `_CacheView` under a single lock, and touch
the disk only when the view reports it cannot answer on its own.

```python
view = self._view_tail(steps)          # or _view_range(step_range)
records = view.records()
if view.complete:
    return records
return self._older_tail(view, steps, records) + records
```

`_CacheView.complete` folds together the three reasons the disk can be skipped: the
cache holds an older step, nothing was ever evicted, or there is no writer. The
scans themselves (`_scan_tail`, `_scan_range`, `_nearer_front`, `_newest_steps`) are
plain functions over a sequence, so they hold no lock and are tested directly.

`et.history(run=...)` bypasses the store entirely and reads through `JsonlReader`
(`read_history`), which is what makes offline analysis and `et rules test` possible.

### Rows versus steps

A step normally occupies one physical row, but a `max_open_seconds` timeout followed
by more data for that step writes a *patch line*, so one step can span several rows.
The two are kept strictly separate in the read path:

* `JsonlReader.tail_rows(n)` returns **physical rows**; `JsonlReader.tail(n)` returns
  **merged steps** and widens its own read until it has one whole step to spare.
* `HistoryStore._collect_tail(steps)` works in rows, then `_take_steps` merges and
  trims by *step*, so `get(n)` never returns a half-merged oldest row.
* The cache stores each row's step next to its bytes, so `_view_tail` snapshots the
  exact rows covering *n* steps without parsing any JSON, and stops as soon as one
  older step proves nothing is truncated.
* `parse_rows` drops lines that are corrupt or carry no integer `_step`, so every
  record a reader returns can be ordered and merged by step.

### Query cost

Queries never scan the whole cache:

| query | cost |
| --- | --- |
| `history(n)` | O(rows returned), independent of cache size |
| `history(step_range=...)` | O(distance from the nearer end of the cache) |
| `history(-1)` | O(run length) — it has to materialise everything |

Disk fallback only happens once rows have been evicted or the run was resumed;
`stats()["disk_prefix"]` reports whether that is the case.

## Concurrency model

| Lock / thread | Owner | Protects |
| --- | --- | --- |
| `HistoryStore._lock` (RLock) | store | open row, cache, series, row ordinal, writer enqueue |
| `JsonlWriter._lock` (RLock) | writer | buffer, index, meta fields |
| `JsonlWriter._write_lock` | writer | batch swap + file append (ordering) |
| open-row `threading.Timer` | store | commits a stale open row; carries a generation token |
| buffer `threading.Timer` | writer | flushes records that sat in memory too long |
| `CompiledRule.lock` | engine | one rule's state machine transitions |
| dispatch worker thread | dispatcher | drains the send queue; drained at exit |
| watchdog thread | engine | evaluates time-based rules when no logs arrive |

Lock order is always `_write_lock` → `_lock`; nothing acquires them the other way.

## Extension points

| I want to add... | Where it goes |
| --- | --- |
| a notification channel | subclass `AlertBackend`, call `register_backend("type", cls)` |
| an expression function | add to `expr/functions.py` (`WINDOW_FUNCS` / `SCALAR_FUNCS` / `SPECIAL_FUNCS`) |
| an output format | add a branch in `history/frame.to_output` |
| a metrics backend | pass any object with `init/log/finish` in `backends=[...]` |
| an artifact storage mode | extend `ArtifactStore._materialise` (default is `copy`: `link` shares the caller's inode) |
| a value type to encode | extend `history/codec.py` (and `encoders.py` for the coercion) |
| a history tunable | add a field to `HistoryOptions`; it is validated and documented automatically |

## Configuration

Every history tunable lives on `HistoryOptions`, a frozen dataclass validated once in
`HistoryStore.init()`. Unknown names raise `TypeError` listing the valid ones, so a
typo like `cache_byte=` fails loudly instead of silently keeping the default. Run
state is initialised in exactly one place, `HistoryStore._reset()`, which both
`__init__` and `init()` call — re-initialising cannot leave a field behind.

## Invariants

See [`design.md` §G](design.md). The short version: metadata is written last,
undurable rows are never evicted, the cache/disk boundary is addressed by row
ordinal, alerts are evaluated once per committed step, and expression evaluation
degrades to `UNKNOWN` instead of raising.

## Naming

The write path reads as one sentence, so the stages are named after what they do to
the open row:

```
_accept_step → _switch_open_row → _update_open_row → _close_open_row → _emit
```

Two words are reserved and mean exactly one thing each:

| word | meaning |
| --- | --- |
| **merge** | combining rows that share a `_step` (`merge_steps`, `_needs_merge`). Never used for folding metrics into a row — that is `_update_open_row`. |
| **row** | one physical JSONL line. A step may span several rows, so `tail_rows`/`parse_rows` return rows while `tail`/`parse` return merged steps. |

`stats()` distinguishes `rows_on_disk` (lines in the file) from `rows_logged`
(ordinals this process handed out); they differ only after records are dropped.
