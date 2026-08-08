# Configuration

## History options

Any of these can be passed straight to `et.init()`. An unknown name raises with the
list of valid options rather than being silently ignored.

| Option | Default | Meaning |
| --- | --- | --- |
| `cache_bytes` | 1 GiB | in-memory cache budget, in encoded bytes |
| `cache_rows` | 2,000,000 | second cache limit, in rows |
| `alert_window` | 4096 | points kept per metric for alert rules |
| `max_open_seconds` | 60.0 | commit an idle open row after this long; `None` disables |
| `step_policy` | `"monotonic"` | `"monotonic"` drops backward steps, `"allow"` keeps them |
| `rank_aware` | `True` | non-zero ranks write their own shard |
| `stream` | `None` | an independent producer with its own file and step cursor |
| `spans` | `True` | write the span tree to `spans.jsonl` beside the metrics |
| `print_to_screen` | `False` | print every committed row |
| `print_handle` | `print` | where those lines go |
| `buffer_size` | 50 | flush after this many buffered rows |
| `buffer_interval` | 1.0 | seconds; a gap this long means "low frequency, write now" |
| `max_buffer_seconds` | 5.0 | force a flush of anything held longer |
| `max_pending_records` | 100,000 | cap on the backlog when writes are failing |

```python
et.init(project="p", cache_bytes=256 << 20, max_open_seconds=None, step_policy="allow")
```

### Write buffering

The flush strategy adapts to how often you log, so it suits both network mounts and
low-latency local disks:

- If more than `buffer_interval` has passed since the last write, treat logging as
  infrequent and write immediately.
- Otherwise batch, and flush once `buffer_size` rows have accumulated.
- A background timer flushes anything held longer than `max_buffer_seconds`.
- A failed write rolls the batch back and requeues it; `max_pending_records` bounds
  the backlog, dropping the oldest rows if the disk stays unavailable.

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
