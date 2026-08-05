# CLI

Installed as `et`.

## `et history`

Print a run's recorded history.

```bash
et history tracker/jsonl/demo/run-1
et history tracker/jsonl/demo/run-1 -n 50 --metrics loss,lr
et history tracker/jsonl/demo/run-1 --step-range 100:200 --format json
et history tracker/jsonl/demo/run-1 -n -1 --format csv > run.csv
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-n` | 20 | number of steps, `-1` for all |
| `--metrics` | all | comma-separated names |
| `--step-range` | — | `start:end`, end exclusive; either side may be empty |
| `--format` | `table` | `table`, `json` or `csv` |

The argument is a run directory or a `metrics.jsonl` file.

## `et rules explain`

Show how an expression parses, and what it references. Useful when precedence is in
question:

```bash
et rules explain "diff(m1)>50 | m1 > 5 => warn: x"
```

```
condition : (diff(m1) > 50) or (m1 > 5)
level     : warning
message   : x
metrics   : m1
functions : diff
```

The rendered condition is fully parenthesised, so there is nothing left to misread.

## `et rules test`

Replay a rule over recorded history to tune a threshold before trusting it:

```bash
et rules test "zscore(loss[50]) > 4 => error: spike" --run tracker/jsonl/demo/run-1
```

```
replayed 1000 steps, 2 alert(s)
  step=417  loss_spike: spike at 417
  step=863  loss_spike: spike at 863
```

`-n` limits the replay to the last N steps.

## `et alert`

Send a one-off alert through the configured channels:

```bash
et alert "training finished" --title Done --level info
et alert "node 3 is on fire" --level critical --channel oncall
```

Channels come from the same configuration the library uses: `ET_ALERT_CONFIG` or the
`ET_LARK_WEBHOOK_URL` environment variable. See [Alerts](alerts.md).
