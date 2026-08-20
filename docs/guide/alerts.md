# Alerts

A rule is an expression plus a level and a message. It is evaluated once per
committed step against a rolling window of your metrics.

```python
et.init(project="demo", alert_rules=[
    "isnan(loss) or isinf(loss)   => critical: non-finite loss",
    "zscore(loss[50]) > 4         => error: loss spike {loss:.4f} @ step {step}",
    "stalled(loss[100])           => warning: loss flat for 100 steps",
    "no_data(10m)                 => error: training looks hung",
])

et.add_alert_rule("eval.acc < 0.5 => warning: accuracy too low")  # any time
```

## Rule syntax

```
<expression> => [level][: message]
```

Both the level and the message are optional. The older comma form
(`expr, warn, message`) still works.

Key points, covered fully in [Alert expressions](../reference/expressions.md):

- `metric[window]` selects a window: `loss[20]` is the last 20 points,
  `loss[5m]` the last 5 minutes.
- Precedence is `or < and < not < comparison < arithmetic` — **not** Python's. So
  `diff(m1) > 50 | m1 > 5` parses the way it reads, as `(diff(m1) > 50) or (m1 > 5)`.
- Metric names may contain `/` and `@` directly (`train/loss`, `val/m1/acc@16`), so
  division has to be spaced: `a/b` is one metric, `a / b` divides. Quote anything
  else with `"`, `'` or backticks: `"train loss"`.
- Missing data, too few points, NaN and division by zero all evaluate to UNKNOWN,
  which never fires and never changes state. Warm-up cannot produce false alarms.

### Python builder

Equivalent, with editor completion:

```python
from expr_tracker import M

et.add_alert_rule((M.m1.diff() > 50) | (M.m1 > 5), level="warning", message="m1 failure")
et.add_alert_rule(M["train/loss"][50].zscore() > 4, level="error", for_steps=3)
```

## Rule options

```python
et.add_alert_rule({
    "name": "loss_spike",
    "condition": "zscore(loss[50]) > 4",
    "level": "error",
    "mode": "edge",            # edge: only on False->True; level: keep firing
    "for_steps": 3,            # must hold for 3 consecutive steps
    "cooldown": 300,           # seconds between repeats in level mode
    "max_fires": 5,
    "notify_recovery": True,   # send an info when the condition clears
    "channels": ["oncall"],    # restrict delivery
    "tags": ["gpu"],
})
```

Prometheus-style keys are accepted too: `alert` for `name`, `expr` for `condition`,
`for` for `for_steps`.

### Message templates

`{step}`, `{time}`, `{run}`, `{project}`, any metric on the current step, and
`{expr}` — the condition rendered with observed values, e.g.
`diff(m1)=63.2 > 50 or m1=1.2 > 5`. Unknown placeholders are left as-is rather than
raising.

## Channels

```python
et.configure_alert(
    channels=[
        {"type": "lark",  "name": "oncall", "url_env": "ET_LARK_WEBHOOK_URL",
         "min_level": "warning"},
        {"type": "slack", "name": "daily", "url": "...", "tags": ["daily"]},
        {"type": "email", "options": {"host": "smtp.x", "to": ["me@x"], "tls": True}},
    ],
    policy={"rate_limit_per_minute": 20, "dedup_window": 300, "max_retries": 3},
)

et.alert(title="done", text="training finished", level="info", channels=["oncall"])
```

Built-in types: `lark`, `slack`, `dingtalk`, `wecom`, `webhook` (a generic JSON
template), `email`, `callable`. All of them use only the standard library, so no
channel needs an extra. Add your own with `register_backend()`.

### Email

Email needs an SMTP server to send *from*, even when the recipient is Gmail.

```python
et.init(
    project="demo",
    alert={
        "channels": [
            {
                "type": "email",
                "name": "inbox",
                "options": {
                    "host": "smtp.gmail.com",
                    "port": 587,
                    "tls": True,
                    "user": "you@gmail.com",
                    "password": os.environ["SMTP_PASSWORD"],
                    "sender": "you@gmail.com",
                    "to": ["you@gmail.com", "teammate@example.com"],
                },
                "min_level": "error",
            }
        ]
    },
    alert_rules=["isnan(loss) => critical: loss diverged"],
)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `host` | — | SMTP server, **required** |
| `to` | — | one address or a list, **required** |
| `port` | 465 with `ssl`, else 25 | |
| `tls` | `false` | STARTTLS on a plain connection (port 587) |
| `ssl` | `false` | implicit TLS from the start (port 465) |
| `user` / `password` | — | omit both for an unauthenticated relay |
| `sender` | `user`, else `expr-tracker` | the `From` address |
| `html` | `true` | send the HTML part as well as the text |

Use `tls` **or** `ssl`, not both: `tls` upgrades a plain connection, `ssl` starts
encrypted.

!!! warning "Keep the password out of your code"
    Read it from the environment, as above. Gmail additionally rejects account
    passwords for SMTP — turn on 2-step verification and create an
    [app password](https://myaccount.google.com/apppasswords), which you can
    revoke independently of your account.

Mail is sent as `multipart/alternative`: a severity-coloured HTML card with the
fields as a table, plus the plain text as a fallback for clients that will not
render HTML. Set `html: false` for text only.

Common servers:

| Provider | host | port | setting |
| --- | --- | --- | --- |
| Gmail | `smtp.gmail.com` | 587 | `tls: true` (app password required) |
| Outlook / Office 365 | `smtp.office365.com` | 587 | `tls: true` |
| QQ / 163 | `smtp.qq.com`, `smtp.163.com` | 465 | `ssl: true` (authorisation code) |
| SendGrid | `smtp.sendgrid.net` | 587 | `tls: true`, user `apikey` |
| Internal relay | your host | 25 | often no `user`/`password` |

### Routing

Each channel filters independently:

| Option | Effect |
| --- | --- |
| `min_level` | drop anything below this severity |
| `levels` | an exact allowlist, overriding `min_level` |
| `tags` | deliver only if the message tags intersect |
| `enabled` | switch the channel off |

A rule's `channels=[...]` restricts which channels it reaches at all.

### Delivery policy

Sending is asynchronous by default so it cannot block the training loop. Each channel
has a token-bucket rate limit, a dedup window (suppressed duplicates are summarised as
`(+N suppressed)` on the next send), and retries with exponential backoff that respect
`Retry-After`. Delivery failures are logged, never raised.

### Configuration precedence

`et.init(alert=...)` > `et.configure_alert(...)` > the json/toml/yaml file pointed to
by `ET_ALERT_CONFIG` > environment variables (`ET_LARK_WEBHOOK_URL`, and the legacy
`WEBHOOK_URL`).

## Time-based rules

`no_data`, `age` and `elapsed` are checked by a background watchdog, because a hung
run logs nothing and nothing else would trigger evaluation. It polls every 30 seconds
by default, which also bounds detection latency:

```python
et.init(..., alert={"watchdog_interval": 5},
        alert_rules=["no_data(10s) => error: hung"])
```

The thread only starts when a time-based rule exists, and stops on `finish()`.

## Testing rules before you trust them

Replay a rule over a finished run:

```bash
et rules test "zscore(loss[50]) > 4 => error: spike" --run tracker/jsonl/demo/run-1
# replayed 1000 steps, 2 alert(s)
#   step=417  loss_spike: ...
```

And check how an expression parses:

```bash
et rules explain "diff(m1)>50 | m1 > 5 => warn: x"
```

## Diagnosing silent rules

A rule that references a metric you never log can never fire. `info()` reports it,
and `finish()` warns with a suggestion:

```python
et.info()["alerts"]["rules"]
# {"rule_74ef8c4d": {"fires": 0, "firing": False, "enabled": True,
#                    "unresolved_metrics": ["los"]}}
```

```
Alert rule 'rule_74ef8c4d' never fired: metric(s) 'los' were never logged.
Did you mean: loss?
```

## Distributed runs

Only rank 0 alerts by default, so one incident is not reported N times. See
[Distributed runs](distributed.md).
