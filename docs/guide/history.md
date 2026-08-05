# Querying history

```python
et.history(n=50, *, output_type="dict", metrics=None, step_range=None,
           include_meta=True, include_open=True, fill_missing=False,
           dropna=False, run=None)
```

## Selecting rows

```python
et.history()                            # the last 50 steps (the default)
et.history(200)                         # the last 200
et.history(-1)                          # everything
et.history(-1, step_range=(100, 200))   # steps 100..199, end exclusive
et.history(-1, step_range=(None, 100))  # everything before step 100
```

`step_range` is clamped, so out-of-range bounds are empty rather than an error.

## Selecting columns

```python
et.history(-1, metrics=["loss", "lr"])              # only these, plus _step/_time
et.history(-1, metrics=["loss"], include_meta=False)  # only loss
et.history(-1, metrics=["loss"], fill_missing=True)   # absent -> None
et.history(-1, metrics=["eval/acc"], dropna=True)     # drop steps without it
```

`dropna` drops a row when **all** selected metrics are missing, which is what you
want for sparse eval metrics logged every N steps.

## Output types

```python
et.history(100)                            # list[dict] (default)
et.history(100, output_type="pandas")      # or "pd"
et.history(100, output_type="polars")      # or "pl"
```

`pandas` and `polars` are optional extras. Columns are ordered `_step`, `_time`, then
metrics in first-seen order.

## Reading another run

```python
et.history(50, run="tracker/jsonl/demo/run-1")            # a run directory
et.history(50, run="tracker/jsonl/demo/run-1/metrics.jsonl")  # or the file
```

No `init()` required, and it works while the other process is still writing — a
reader only ever sees complete lines, and never sees fewer rows than last time.

## The open row

The uncommitted row is included by default, so a query mid-step sees the metrics you
have logged so far:

```python
et.log({"loss": 1.0}, step=7)     # not committed yet
et.history(1)                     # [{"_step": 7, "loss": 1.0, ...}]
et.history(1, include_open=False) # the last committed step instead
```

## Caching

Recent rows are kept in memory as encoded bytes, bounded by two limits:

```python
et.init(..., cache_bytes=1 << 30, cache_rows=2_000_000)   # the defaults
```

Queries inside the cache do **zero** file IO. Once rows are evicted, only the part
below the boundary is read back from disk. Cost is proportional to the window you
ask for, not to the size of the run: `history(50)` takes about 227&nbsp;µs at 1,000
steps and at 100,000 steps alike.

Check whether the cache is doing its job:

```python
stats = et.info()["history"]
stats["queries"]          # total history() calls
stats["disk_queries"]     # how many had to touch the file
stats["cached_rows"], stats["cached_bytes"], stats["cache_limit_bytes"]
stats["evicted_rows"]
```

`disk_queries == 0` means everything was served from memory. A persistently high
`disk_queries / queries` means `cache_bytes` is small relative to the window you
query.

Rows are never evicted before they are durable, so shrinking the cache can cost IO
but never data.

## Metric series

Alert rules read from a separate rolling buffer sized by the widest window any rule
needs, not from the cache. Eviction therefore never starves a rule:

```python
et.init(..., alert_window=1000)      # points kept per metric
et.get_run().history.series.points("loss")   # [(step, time, value), ...]
```
