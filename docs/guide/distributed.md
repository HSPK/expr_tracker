# Distributed runs

The model is **rank 0 tracks, other ranks stay quiet** — with enough escape hatches
that you can change your mind.

## Sharded files

Rank is read from `RANK`, then `LOCAL_RANK`, defaulting to 0. Ranks above 0 write
their own file so concurrent appends cannot interleave and corrupt step order:

```
tracker/jsonl/demo/run-1/
├── metrics.jsonl          # rank 0
├── metrics.rank1.jsonl
├── metrics.rank2.jsonl
└── metrics.rank3.jsonl
```

Each shard has its own sidecar and resumes independently.

```python
et.init(..., rank_aware=False)   # all ranks share one file (only if you serialise writes)
```

## Reading shards

`et.history()` reads the current process's shard. Offline, a run directory resolves
to rank 0's file; address another shard by path:

```python
et.history(-1, run="tracker/jsonl/demo/run-1")                        # rank 0
et.history(-1, run="tracker/jsonl/demo/run-1/metrics.rank2.jsonl")    # rank 2
```

## Alerts

Non-zero ranks do not send alerts by default, otherwise one incident is delivered
once per GPU:

```python
et.init(...)                       # alerts on rank 0 only (default)
et.init(..., alert_on_rank=3)      # alert from rank 3 instead
et.init(..., alert_on_rank=None)   # every rank alerts
```

Suppression only affects delivery. Every rank still records its own history, keeps
its rules, and reports them through `info()`:

```python
et.info()["rank"]     # which rank this process was detected as
```

## Typical setup

```python
import expr_tracker as et

et.init(
    project="llm",
    name=f"sft-{run_id}",          # the same name on every rank
    dir="/shared/runs",
    config=cfg,
    alert_rules=["isnan(loss) => critical: diverged"],
)
```

Every rank writes to `/shared/runs/llm/sft-<id>/`, each into its own shard, and only
rank 0 pages you.
