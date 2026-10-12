# Configuration

## History options

Any of these can be passed straight to `et.init()`. An unknown name raises with the
list of valid options rather than being silently ignored.

| Option | Default | Meaning |
| --- | --- | --- |
| `cache_bytes` | 1 GiB | in-memory cache budget, in encoded bytes |
| `cache_rows` | 2,000,000 | second cache limit, in rows |
| `alert_window` | 4096 | points kept per metric for alert rules |
| `step_policy` | `"monotonic"` | `"monotonic"` drops backward steps, `"allow"` keeps them |
| `rank_aware` | `True` | non-zero ranks write their own shard |
| `stream` | `None` | an independent producer with its own file and step cursor |
| `spans` | `True` | write the span tree to `spans.jsonl` beside the metrics |
| `span_count` | `False` | emit `count/<path>` metrics for completed spans |
| `span_metrics` | `True` | aggregate span measurements into metric history; disable for trace-only workers |
| `span_async_write` | `True` | write span records on a reusable background worker instead of the producer thread |
| `span_print_fn` | `None` | default `print_fn` for every span in the run |
| `span_plugins` | `()` | default plugins for every span in the run |
| `print_to_screen` | `False` | print every committed row |
| `print_handle` | `print` | where those lines go |
| `buffer_size` | 50 | flush after this many buffered rows |
| `buffer_interval` | 1.0 | seconds; a gap this long means "low frequency, write now" |
| `max_buffer_seconds` | 5.0 | force a flush of anything held longer |
| `max_pending_records` | 100,000 | cap on queued plus in-flight records |
| `max_pending_bytes` | 64 MiB | cap on queued plus in-flight encoded bytes; `None` disables |
| `max_record_bytes` | 8 MiB | maximum encoded size of one record; `None` disables |
| `max_file_bytes` | 100 MiB | rotate metric and span files at record boundaries; `None` disables |

```python
et.init(project="p", cache_bytes=256 << 20, step_policy="allow")
```

### Write buffering

The flush strategy adapts to how often you log, so it suits both network mounts and
low-latency local disks:

- If more than `buffer_interval` has passed since the last write, treat logging as
  infrequent and write immediately.
- Otherwise batch, and flush once `buffer_size` rows have accumulated.
- A reusable background worker flushes anything held longer than `max_buffer_seconds`.
- A failed write rolls the batch back and requeues it; `max_pending_records` bounds
  the backlog, dropping the oldest rows if the disk stays unavailable.

When a healthy writer reaches either pending cap, producers wait for capacity
instead of losing records. Only failure/retry mode drops overflowing queued
records, with an error log and dropped-record/byte counters.

A record exceeding `max_record_bytes`, or unable to fit into
`max_pending_bytes` by itself, is rejected immediately with an error. Rejected
metric rows are not cached or forwarded to backends; their assigned step may be
skipped. Rejecting an oversized span event does not discard its otherwise valid
timing metrics.

The byte budget covers encoded queued and in-flight records, not caller-owned
objects or serialization temporaries. Batch assembly may temporarily allocate
another buffer of up to the pending-byte budget. Both byte limits must be positive
integers or `None`.

These settings flush buffered records to disk; they never commit an open metric
row or advance its step. Open rows have no time limit.

Span file writes are asynchronous by default. `span_async_write=False` preserves
synchronous writes when a batch is due. Explicit `history.flush()` and `finish()`
wait for accepted records to be flushed in either mode.
`et.info()["history"]` reports metric/span `pending_bytes`, `span_pending_bytes`,
`dropped_records` and `dropped_spans` for diagnosing overload and failed writes.

### Automatic rotation

Metrics and spans rotate by default before the next record would make a file
exceed 100 MiB. Change the threshold with `et.init(max_file_bytes=20 << 20)`, or
disable rotation with `et.init(max_file_bytes=None)`. The limit must be a positive
integer or `None`. A record larger than the threshold stays intact in its own file
(subject to the independent `max_record_bytes` limit).

The original filename remains in place. Subsequent segments live in its `.parts`
directory, for example `metrics.jsonl.parts/00000001.jsonl`. Stream and rank files
rotate independently. All segments are retained: rotation limits individual file
size, not total disk usage. Keep the `.parts` directory alongside the original file
when copying a run. Live and offline history queries, resume, and span trace export
read the complete logical history across segments, even when rotation is later
disabled. Segment files are implementation details, not separate streams.

## Alert configuration

Resolved in this order:

1. `et.init(alert=...)`
2. `et.configure_alert(...)`
3. the file pointed to by `ET_ALERT_CONFIG` (`.json`, `.toml`, `.yaml`)
4. environment variables — `ET_LARK_WEBHOOK_URL`, or the legacy `WEBHOOK_URL`

```yaml
# alerts.yaml
alert:
  enabled: true
  watchdog_interval: 30
  policy:
    rate_limit_per_minute: 20
    dedup_window: 300
    max_retries: 3
  channels:
    - type: lark
      name: oncall
      url_env: ET_LARK_WEBHOOK_URL
      min_level: warning
    - type: slack
      name: daily
      url: https://hooks.slack.com/...
      tags: [daily]
  rules:
    - "isnan(loss) => critical: non-finite loss"
```

### Channel options

| Option | Meaning |
| --- | --- |
| `type` | `lark`, `slack`, `dingtalk`, `wecom`, `webhook`, `email`, `callable` |
| `name` | how rules and `et.alert(channels=...)` address it; defaults to `type` |
| `url` / `url_env` | the webhook, literal or from the environment |
| `enabled` | switch it off without removing it |
| `min_level` | drop anything below this severity |
| `levels` | an exact allowlist, overriding `min_level` |
| `tags` | deliver only if the message tags intersect |
| `options` | backend-specific settings |
| `options.html` | email only: send an HTML part alongside the text (default `true`) |
| `policy` | overrides the default delivery policy |

### Delivery policy

| Option | Default | Meaning |
| --- | --- | --- |
| `async_send` | `True` | deliver on a worker thread |
| `timeout` | 10.0 | HTTP timeout, seconds |
| `max_retries` | 3 | retry attempts for retryable failures |
| `backoff_initial` / `backoff_factor` / `backoff_max` | 0.5 / 2.0 / 30.0 | exponential backoff |
| `retry_on_status` | 408, 429, 500, 502, 503, 504 | which HTTP codes are retryable |
| `respect_retry_after` | `True` | honour the `Retry-After` header |
| `rate_limit_per_minute` | 20 | token bucket; `None` disables |
| `on_rate_limited` | `"coalesce"` | `drop`, `queue` or `coalesce` |
| `dedup_window` | 300.0 | seconds; duplicates are summarised on the next send |
| `queue_size` | 1000 | per-channel queue depth |
| `on_queue_full` | `"drop_oldest"` | `drop_oldest`, `drop_new` or `block` |

### Rule options

| Option | Default | Meaning |
| --- | --- | --- |
| `condition` | — | the expression (alias: `expr`) |
| `name` | derived | identifies the rule (alias: `alert`) |
| `level` | `warning` | `debug`, `info`, `warning`, `error`, `critical` |
| `title` / `message` | derived | supports templates; the title defaults to `name` when you set one, else to the condition |
| `mode` | `"edge"` | `edge` fires on transition, `level` keeps firing |
| `for_steps` | 1 | must hold this many consecutive steps (alias: `for`) |
| `cooldown` | 300.0 | seconds between repeats in level mode |
| `max_fires` | `None` | stop after this many |
| `notify_recovery` | `False` | send an info when the condition clears |
| `channels` | all | restrict delivery |
| `tags` | `[]` | matched against channel tags |
| `enabled` | `True` | switch it off |

The auto-generated name covers everything that makes two rules distinct, so two
rules on the same condition with different levels or channels both register.

## Environment variables

| Variable | Used for |
| --- | --- |
| `RANK`, `LOCAL_RANK` | distributed rank detection |
| `ET_ALERT_CONFIG` | path to an alert config file |
| `ET_LARK_WEBHOOK_URL`, `WEBHOOK_URL` | default Lark channel |
| `WANDB_API_KEY`, `WANDB_HOST` | passed to `wandb.login` |
