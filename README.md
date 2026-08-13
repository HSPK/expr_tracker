# Experiment Tracker

[![PyPI](https://img.shields.io/pypi/v/expr-tracker)](https://pypi.org/project/expr-tracker/)
[![Python](https://img.shields.io/pypi/pyversions/expr-tracker)](https://pypi.org/project/expr-tracker/)
[![License](https://img.shields.io/pypi/l/expr-tracker)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://hspk.github.io/expr_tracker/)

A local-first experiment tracker. Metrics land in a JSONL file you own, stay
queryable while the run is live, and can trigger alerts from an expression language.
`wandb` and `trackio` are optional mirrors, not requirements.

📖 **Documentation: <https://hspk.github.io/expr_tracker/>**

```python
import expr_tracker as et

et.init(project="demo", name="run-1", alert_rules=["zscore(loss[50]) > 3 => error: spike"])
for step in range(1000):
    et.log({"loss": loss, "lr": lr})
et.finish()

et.history(50)                    # the last 50 steps, as dicts
et.history(-1, output_type="pd")  # everything, as a DataFrame
```

## Why

- **The file is the source of truth.** One JSON object per step, appended to
  `metrics.jsonl`. No server, no database, no vendor.
- **History is queryable during the run.** `et.history(n)` answers from an in-memory
  cache and touches the file only for what it evicted — 227&nbsp;µs for
  `history(50)` whether the run has 1,000 steps or 100,000.
- **Alerts are expressions, not callbacks.** `zscore(loss[50]) > 3 or isnan(loss)`
  is parsed, validated, and evaluated against a rolling window. Rules can be replayed
  over a finished run to tune thresholds before you trust them.
- **It stays out of the way.** `log()` costs ~26&nbsp;µs. A failed disk, a dead
  webhook or an unserialisable value degrades with a warning; none can stop training.

## Install

```bash
uv add expr_tracker                 # local-first: click, loguru, pydantic only
uv add "expr_tracker[wandb]"        # mirror to Weights & Biases
uv add "expr_tracker[trackio]"      # mirror to trackio
uv add "expr_tracker[lark]"         # Feishu/Lark alert channel
uv add "expr_tracker[pandas]"       # history(output_type="pandas")
uv add "expr_tracker[all]"          # everything
```

Only the JSONL history is built in. A missing extra is reported with the exact
install command; it never crashes a run.

## Features

| | |
| --- | --- |
| [Logging](https://hspk.github.io/expr_tracker/guide/logging/) | One line per step. Several `log()` calls for one step merge into one row, wandb-compatible `step`/`commit` semantics, numpy and pydantic values handled. |
| [History](https://hspk.github.io/expr_tracker/guide/history/) | `et.history(n)` during or after the run, offline reads of any run directory, dict/pandas/polars output, bounded in-memory cache with observable hit rate. |
| [Alerts](https://hspk.github.io/expr_tracker/guide/alerts/) | An expression DSL with rolling windows, three-valued logic (no false alarms during warm-up), a rule state machine, and a watchdog that catches a hung run. |
| [Channels](https://hspk.github.io/expr_tracker/guide/alerts/#channels) | Lark, Slack, DingTalk, WeCom, generic webhook, email — with rate limiting, dedup, retries and per-channel routing. |
| [Artifacts](https://hspk.github.io/expr_tracker/guide/artifacts/) | Versioned file sets, deduplicated by content, shared across a project's runs, with lineage. |
| [Spans](https://hspk.github.io/expr_tracker/guide/spans/) | Time the parts of a step, and their parts. Each duration becomes a metric, so alerts and queries work on it unchanged; `et trace` exports the timeline for Perfetto. `print_fn` prints the tree live, and plugins attach CPU and GPU cost to each region. |
| [Streams](https://hspk.github.io/expr_tracker/guide/streams/) | Independent producers — a data worker and a training loop — each with their own step cursor and file inside one run. |
| [Distributed](https://hspk.github.io/expr_tracker/guide/distributed/) | Per-rank shards so concurrent appends cannot corrupt step order; only rank 0 alerts by default. |
| [CLI](https://hspk.github.io/expr_tracker/guide/cli/) | `et history`, `et trace`, `et rules explain`, `et rules test`, `et alert`. |

## Examples

| | |
| --- | --- |
| [`multiprocess_pipeline.py`](examples/multiprocess_pipeline.py) | Four data producers and four trainers as eight processes in one run, with bounded staleness. Each worker gets its own stream and its own lane in the exported trace, and the blocking spans show which side is the bottleneck. |

```bash
uv run python examples/multiprocess_pipeline.py --produce-ms 10 --train-ms 40
```

## wandb compatibility

Migrating an existing script is usually one line:

```python
# import wandb as et
import expr_tracker as et
```

`init`, `log`, `finish`, `alert`, `log_artifact`, `use_artifact`, `Artifact`,
`define_metric`, `run.summary`, `run.step`, `run.dir` and `run.url` keep their wandb
names and signatures. See the
[compatibility table](https://hspk.github.io/expr_tracker/guide/backends/#wandb-compatibility).

## Development

```bash
uv sync --all-extras
uv run pytest                                   # everything
uv run pytest -m "not slow and not benchmark"   # the fast suite
uv run pytest -m benchmark -s                   # timing and memory report
uv run pytest --cov=expr_tracker                # coverage
uv run ruff check src tests
uv run ruff format src tests
```

### Test layout

| File | Covers |
| --- | --- |
| `test_history`, `test_expr_*`, `test_alert_*`, `test_writer_durability`, … | per-module unit tests |
| `test_correctness.py` | value and type round trips, randomised commit sequences, ordering invariants |
| `test_cache.py` | that the cache really serves reads: zero-IO assertions, eviction boundaries, warm/cold parity |
| `test_failure_modes.py` | degradation: write failures, read-only dirs, encoder blow-ups, dead alert backends |
| `test_e2e.py` | full runs, resume, crash recovery, offline reads, CLI |
| `test_scenarios.py` | live cross-process reads, alerts during eviction, out-of-order resume |
| `test_hot_paths.py` | contracts and defaults of `et.log` / `et.history` / summary / alerts |
| `test_value_encoding.py` | numpy, pydantic, datetime, Path, Enum round trips; output types; query bounds |
| `test_expr_properties.py` | DSL properties: render round-trip stability, precedence, the whole `M` builder |
| `test_trace.py` | Chrome Trace export: lane layout, stream and step selection, the CLI |
| `test_spans.py` | nesting, aggregation, decorator and async forms, errors, thread and task isolation |
| `test_examples.py` | the shipped examples run, and their backpressure claims hold |
| `test_span_plugins.py` | `print_fn` output and indentation, the plugin protocol, failure isolation, CPU/GPU built-ins |
| `test_streams.py` | stream naming and validation, isolation, resolution order, backend grouping, two-process runs |
| `test_distributed.py` | rank shards, `alert_on_rank`, real multi-process runs |
| `test_wandb.py` | real wandb in offline mode: parameter mapping, step alignment, artifacts |
| `test_trackio.py` | trackio contract, resume mapping, real end-to-end |
| `test_lark_live.py` | Lark channel; real delivery when `ET_LARK_TEST_WEBHOOK` is set |
| `test_stress.py` (`slow`) | 100k-row writes, concurrency, cache thrash, write-failure recovery |
| `test_benchmark.py` (`benchmark`) | throughput, tail latency, query cost, memory stability |

### Docs

```bash
uv run --group docs mkdocs serve    # preview at localhost:8000
uv run --group docs mkdocs build    # build into site/
```

Published to GitHub Pages by `.github/workflows/docs.yaml` on every push to `main`.
Internals: [`docs/design.md`](docs/design.md) (data model and key invariants) and
[`docs/architecture.md`](docs/architecture.md) (module map, read/write paths,
concurrency model).

## License

[MIT](LICENSE)
