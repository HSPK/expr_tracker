# Design

The data model, the history subsystem and the alert subsystem, and why each is
shaped the way it is.

---

## A. Data model

**One line per step, appended in `_step` order.**

```jsonc
{"_step": 42, "_time": 1754323200.123, "train/loss": 0.31, "lr": 1e-4}
```

Non-finite floats are written the way Python's `json` writes them (`NaN`,
`Infinity`, `-Infinity`) rather than as `null`. A NaN loss is itself a signal, and
`isnan()` / `isinf()` in an alert rule depend on it. Python and pandas read this
back; strict parsers such as `jq` do not. That is a deliberate trade.

Three write stages:

```
log() ──merge──> open row (current step, in memory)
                   │ step advances / commit=True / flush / finish / max_open_seconds
                   ▼
                write buffer (adaptive: buffer_size / buffer_interval / max_buffer_seconds)
                   │
                   ▼
                metrics.jsonl  +  meta.json (sidecar)
```

### A.1 Commit semantics (aligned with wandb)

```python
log(metrics: dict, step: int | None = None, commit: bool | None = None)
```

| Call | `commit` default | Behaviour |
| --- | --- | --- |
| `log(d)` | `True` | merge into the open row, commit, `step += 1` |
| `log(d, commit=False)` | — | merge only |
| `log(d, step=N)` | **`False`** | `N > open`: commit the old row, open a new one. `N == open`: merge. `N < open`: per `step_policy` |

- `step_policy="monotonic"` (default) drops backward writes with a warning;
  `"allow"` accepts them and marks `meta.sorted = false`.
- Writing the same key twice in one step is last-wins, warned once per key per run.
- `max_open_seconds` (default 60s) commits an idle open row so a crash cannot lose
  it. If more data arrives for that step afterwards it is written as a **patch line**
  and `meta.has_duplicate_steps` is set; the reader merges by step, so correctness is
  unaffected.

**The writer guarantees one line per step in ascending order; the reader assumes
none of that** — it must cope with older files, stray writers and patch lines.

### A.2 Sidecar `meta.json`

```jsonc
{"schema": 2, "size": 1048576, "lines": 12000,
 "last_step": 11999, "max_step": 11999,
 "sorted": true, "has_duplicate_steps": false,
 "index": [[step, line, offset], ...],   // one anchor every index_every lines
 "run": {"project": "...", "name": "...", "started_at": ...}}
```

- Written at most every 2s during flushes, always on `close()`, via a temporary file
  and `os.replace` so it is atomic.
- The sidecar is only a **hint**: if `size` disagrees with the file, the writer
  rescans from the last anchor to EOF and repairs itself.

### A.3 Resume

The next step is the file's **maximum** step (`writer.max_step`), not the step on its
last line. Patch lines and `step_policy="allow"` both let a file end on a lower step,
and continuing from that would reuse an existing one. `max_step` is maintained
incrementally and persisted in the sidecar.

### A.4 Multiple processes

The supported model is **rank 0 tracks**; there is no cross-rank merging.

- With `RANK` / `LOCAL_RANK` above 0 the run writes `metrics.rank{N}.jsonl`
  (`rank_aware=True`, can be disabled), because concurrent appends would otherwise
  break step monotonicity.
- Non-zero ranks do not alert by default (`alert_on_rank=0`), so one incident is not
  delivered N times.
- `resolve_run_path()` picks `metrics.jsonl` from a directory and logs an info line
  when shards exist, rather than silently treating a shard as the main file. Address
  a shard by passing its path.

---

## B. History subsystem

```
history/writer.py   JsonlWriter    buffered append + sparse index + sidecar
history/reader.py   JsonlReader    tail / range / seek by step / merge mode
history/series.py   MetricSeries   fixed-size ring buffer per metric (alerts; never hits disk)
history/store.py    HistoryStore   open row + cache + query planning
history/frame.py    dict / pandas / polars output
```

### B.0 Rows versus steps

A step usually occupies one line, but an open row that timed out and then received
more data produces a **patch line**, so a step can span several. The read path keeps
the two strictly apart:

- `JsonlReader.tail_rows(n)` returns **physical rows**; `JsonlReader.tail(n)` returns
  **merged steps** and widens its own read until it has one whole spare step.
- `HistoryStore._collect_tail(steps)` works in rows, then `_take_steps` merges and
  trims by **step**. Trimming by row would cut the boundary step in half and return a
  record with missing fields.
- The cache stores each row's step next to its bytes, so `_view_tail` can snapshot
  exactly the rows covering *n* steps without parsing any JSON.

### B.1 Cache

- The cache holds **encoded line bytes**, not dicts: the memory budget is then exact,
  and encoding moves from flush time to log time, so nothing is encoded twice.
- Two limits: `cache_bytes` (default 1 GiB) and `cache_rows` (default 2,000,000).
- Invariant: **rows that are not yet on disk are never evicted** (over budget, the
  writer is flushed first).

### B.2 Disk addressing by physical row ordinal

The cache/disk boundary is addressed by **physical row ordinal** — one append
sequence number per record — not by `_step`. Patch lines make a step appear more than
once, and a step lookup would exclude that step's earlier row as well, losing data.
Out-of-order files (`step_policy="allow"`) have the same problem.

Index anchors still record `(step, line, offset)`, so both `offset_of_line()` and
`offset_of_step()` can binary-search. If writes have been failing and records were
dropped, line numbers shift, and addressing degrades to steps.

The eviction watermark is likewise a row ordinal (`flushed_row`), not a step: a step
watermark would treat that step's patch lines as already written.

On resume the cache is empty while the file is not, so the ordinal must continue from
the existing line count and the store must record that a disk prefix exists —
otherwise `history()` cannot see anything written before the restart.

With `step_policy="allow"`, `history(n)` returns the *n* most recently **written**
steps rather than the *n* highest-numbered ones: ordering by number would need a full
scan of an unsorted file, which contradicts the O(n) tail read. `history(-1)` and
`step_range` are exact by step in all cases.

Disk reads only happen after an eviction (or when a resume left a disk prefix);
otherwise a query does zero IO.

Whether the cache is earning its keep is observable: `stats()` reports `cached_rows`,
`cached_bytes`, `cache_limit_bytes`, `evicted_rows`, `queries` and `disk_queries`.
`disk_queries == 0` means every query was answered from memory;
`disk_queries / queries` is the miss rate, and a persistently high one means
`cache_bytes` is small relative to the window being queried.

### B.3 API

```python
et.history(
    n=50,                    # the newest n steps; -1/None = everything
    output_type="dict",      # dict | dicts | pandas | pd | polars
    metrics=None,            # column selection
    step_range=None,         # (start, end), end exclusive, binary-searched
    include_meta=True,       # keep _step / _time
    include_open=True,       # include the uncommitted current step
    fill_missing=False,      # absent keys become None
    dropna=False,            # drop rows where all selected metrics are missing
    run=None,                # read any run directory/file offline, without init()
)
```

`n` counts **steps**. With gaps it returns the newest n *recorded* steps; for a
literal step interval use `step_range`.

`output_type="pandas"` imports lazily and, when missing, names the extra to install.

### B.4 JSONL is always on

`Run` always builds a `HistoryStore`, whatever `backends` says. `"jsonl"` in
`backends` remains valid and is ignored.

---

## C. Alert expressions

### C.1 Grammar

```
rule    := expr "=>" [level] [":" message]   # split on the first top-level =>,
                                             # falling back to bracket-aware commas
or_expr := and_expr (("or"|"||"|"|") and_expr)*
and_expr:= not_expr (("and"|"&&"|"&") not_expr)*
not_expr:= ("not"|"!") not_expr | cmp
cmp     := arith (("<"|"<="|">"|">="|"=="|"!=") arith)*   # chained a<b<c ⇒ (a<b) and (b<c)
arith   := term (("+"|"-") term)*
term    := unary (("*"|"/"|"%") unary)*
unary   := ("-"|"+") unary | postfix
postfix := primary ("[" window "]")?
primary := NUMBER | DURATION | ident | ident "(" args ")" | "(" expr ")"
window  := INT (points) | DURATION (30s / 5m / 2h)
ident   := [A-Za-z_][A-Za-z0-9_.]* | `anything`
```

A hand-written lexer and Pratt parser, **not Python's `ast`**: Python reads
`diff(m1)>50 | m1>5` as `diff(m1) > (50|m1) > 5`. Precedence, low to high:
`or < and < not < comparison < +- < */% < unary < call/window`.

**Metric resolution**: exact name → `.` replaced by `/` → error with close-match
suggestions. So `eval.acc` finds `eval/acc`; use backticks for exotic names.

### C.2 Three-valued logic (Kleene)

Missing data, too few points, NaN and division by zero all yield `UNKNOWN`.
`UNKNOWN and False = False`, `UNKNOWN or True = True`, anything else containing
UNKNOWN is UNKNOWN. **Only an exact `True` fires; UNKNOWN neither fires nor advances
the state machine**, so warm-up is silent by construction.

### C.3 Function library

| Category | Functions |
| --- | --- |
| Aggregate | `mean std var median sum min max first last count (m[w])` |
| Change | `diff(m[w])` (last − first; `diff(m)` ≡ `diff(m[2])`), `rate(m[w])`, `pct_change(m[w])` |
| Trend | `slope(m[w])` `zscore(m[w])` `ema(m[w], alpha)` `increasing/decreasing(m[w])` |
| Predicate | `isnan(m)` `isinf(m)` `has(m)` `stalled(m[w], eps)` |
| Context | `step()` `elapsed()` `age(m)` `no_data(dur)` |
| Scalar | `abs log exp sqrt min max` |
| Bare `m` | the latest value |

`min(loss[20])` is a rolling minimum and `min(a, b)` is scalar; the node types
disambiguate, so there is no ambiguity.

### C.4 Three front ends, one AST

```python
# 1) DSL string
et.init(..., alert_rules=["diff(m1) > 50 or m1 > 5 => warn: m1 failure"])

# 2) Python builder (editor completion)
from expr_tracker import M
et.add_alert_rule(M["train/loss"][50].zscore() > 4, level="error", for_steps=3)

# 3) structured dict / config file
et.add_alert_rule({"name": "loss_spike", "condition": "zscore(loss[50]) > 4", "level": "error"})
```

### C.5 Rules and the state machine

```python
AlertRule(name, condition, level="warning", title=None,
          message="...", mode="edge"|"level", for_steps=1,
          cooldown=300.0, max_fires=None, notify_recovery=False,
          channels=None, tags=(), enabled=True)
```

- **Evaluated once per committed step**, not per `log()` call, so several logs for
  one step cannot fire twice.
- State machine `OK → PENDING → FIRING`; UNKNOWN changes nothing; `FIRING → OK` can
  send a recovery notice.
- Dependency short-circuit: if a step touched none of a rule's metrics and the rule
  is not time-based, evaluation is skipped.
- **Watchdog timer**: time-based rules (`no_data`, `age`, `elapsed`) are evaluated
  even when nothing is logged, which is what catches a hung run. It polls every 30s
  by default and is set with `alert={"watchdog_interval": N}`. The interval bounds
  detection latency, so a tight rule like `no_data(10s)` needs a smaller one. The
  thread only starts when a time-based rule exists.
- A rule that keeps raising is disabled after a threshold, with one alert.
- The auto-generated rule name covers everything that distinguishes two rules
  (condition, level, message, title, mode, `for_steps`, channels, tags), so two rules
  on the same condition with different levels or channels both register instead of
  one silently replacing the other.
- Message templates: `{step} {time} {run} {project}`, every metric on the latest
  step, and `{expr}` (the condition rendered with observed values). An unknown
  placeholder is left as written rather than raising.
- A rule referencing a metric that was never logged can never fire; it is reported in
  `stats()` as `unresolved_metrics` and warned about once at `finish()`, with a
  close-match suggestion.

---

## D. Alert delivery

- **Levels**: `debug < info < warning < error < critical`, with aliases
  `warn / err / fatal / crit`. Comparison against a level *name* is by severity, not
  string order, so `level >= "error"` correctly includes `critical`.
- **`AlertMessage`**: title, text, subtitle, level, traceback, fields, tags, mentions,
  link, source, dedup_key, timestamp. The dispatcher injects project/run/step context
  into `fields`.
- **`ChannelConfig`**: `type` + `name` (several webhooks of one type) + `url`/`url_env`
  + `min_level` / `levels` allowlist / `tags` routing + `options` + its own `policy`.
- **`WebhookPolicy`**: `timeout`, `max_retries`, `backoff_{initial,factor,max}`,
  `retry_on_status`, `respect_retry_after`, `rate_limit_per_minute`,
  `on_rate_limited`, `dedup_window`, `async_send`, `queue_size`, `on_queue_full`,
  `fail_silently`.
- **Dispatcher**: one daemon worker per channel with a bounded queue; token bucket →
  dedup (suppressed duplicates summarised as `(+N suppressed)` once the window
  expires) → exponential backoff with jitter. `finish()` and `atexit` drain with a
  timeout.
- **Only send failures are swallowed**; configuration errors raise at configuration
  time.
- **Backends**: `lark` (via slark, client reused), `slack`, `dingtalk`, `wecom` and
  `webhook` (a generic template) use stdlib `urllib`; `email` uses stdlib `smtplib` —
  **no new hard dependencies**. `register_backend()` extends the set.
- **Configuration precedence**: `init(alert=)` > `configure_alert()` >
  `ET_ALERT_CONFIG` file > environment (`ET_LARK_WEBHOOK_URL`, legacy `WEBHOOK_URL`)
  > defaults.

---

## E. CLI

```
et history <run> -n 50 [--metrics a,b] [--step-range a:b] [--format table|json|csv]
et rules test "<rule>" --run <run>    # replay over history, print every firing step
et rules explain "<expr>"             # print the parse and the metrics referenced
et alert "<msg>" --title --level --channel
```

---

## F. Compatibility

| Item | Change | Breaking |
| --- | --- | --- |
| `log(d)` without a step | unchanged | no |
| `log(d, step=N)` | write deferred by one call; several logs per step merge into one line | behaviour fix |
| `backends` | jsonl is always on | no |
| Record schema | added `_time` and `meta.json` | no, the reader tolerates both |
| `alert(backends=)` | renamed `channels=`, old name still accepted | no |
| `JsonlTracker` | removed in favour of `expr_tracker.history.HistoryStore` | **yes** |
| `_tracker` | `ContextVar` → global singleton with a lock (fixes worker threads) | no |
| New extras | `[pandas]`, `[polars]`, `[trackio]` | no |

---

## G. Key invariants

Read these before changing the modules they mention.

1. **Metadata is written last**: `{**open_row, "_step": ..., "_time": ...}`, and
   `_step` / `_time` are reserved metric names.
2. **Never evict rows that are not on disk**: judged by `flushed_row`. If the cache
   holds nothing else, flush first, then evict.
3. **The cache/disk boundary is a row ordinal, not a step** (see B.2). Allocating the
   ordinal and enqueueing the line must happen under one lock, and
   `JsonlWriter.flush()` must take its batch and write it under one lock too, or
   concurrent writes will let ordinals and physical line numbers diverge.
4. **Repair a torn last line before resuming**: an unterminated line left by a crash
   would otherwise have the next record glued onto it, and permanently inflate the
   line count.
5. **`history(n)` must over-read one step in merge mode**: the step on the window
   boundary may be only half read, so read one more and drop it.
6. **Alerts are evaluated at step commit**, so `finish()` must commit the open row
   before closing alert delivery.
7. **Expression evaluation never raises**: any error inside a function degrades to
   UNKNOWN. Syntax and semantic errors are raised when the rule is registered.
8. **Rule transitions are serialised**: `on_step` and the watchdog can run
   concurrently, so transitions hold `CompiledRule.lock`, and a failed render or
   dispatch does not advance the state.
9. **Delivery failures are only logged** (`fail_silently`); configuration errors
   raise at configuration time.
10. **Open-row timers carry a generation token**: once the step advances, an old
    timer must not commit the new open row.

---

## H. Known trade-offs

Raised in review and deliberately not done, recorded so they are not re-litigated.

| Item | Today | Why not yet |
| --- | --- | --- |
| Split out `RecordCodec` / `HistoryQuery` | `HistoryStore` owns encoding, the open row, the cache and query planning | The append path (open row → ordinal → cache → enqueue → evict) has to stay together to hold the ordinal invariant. The query side could be split, but that needs a snapshot/flush contract first and the payoff is too small |
| Move `MetricSeries` into `alerts/` | still in `history/`, filled by the write path | `ensure_capacity()` already solved the real problem (a rule window wider than the buffer being silently truncated); the move is pure restructuring |
| Media types (Image/Table/Video) | unsupported; forwarded to the wandb backend, stored locally as `repr` | Needs a metric codec registry plus asset storage — a separate feature. Artifacts already cover the "large file" case |
| Streaming export (Parquet, large CSV) | `history()` materialises a list | A very long run wants a dedicated streaming exporter, not another `output_type` |
| A tracking-backend protocol | `run.py` special-cases wandb and trackio in a few places | A backend only needs `init/log/finish`; the special cases are short and localised, so the abstraction would not pay for itself |
