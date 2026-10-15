# Architecture

How the pieces fit together, and which module owns what. Pair this with
[`design.md`](design.md), which records *why* the data model looks the way it does.

## Native core and SDK boundary

The package is a mixed Rust/Python distribution built with maturin. The
`expr_tracker._native` ABI3 extension owns the core algorithms; Python retains
the public SDK and integration boundaries. JSONL remains the source of truth.

| Surface | Rust ownership | Python ownership |
| --- | --- | --- |
| History | row assembly, cache/query planning, numeric series, segmented I/O, buffering, rotation, resume, index/meta persistence, ordered commit tickets | public options, normalization, output frames, lifecycle and callback adapters |
| Alerts | parsing, compiled evaluation, rule transitions, watchdog/retry/rate-limit/dedup/queue state | models, formatting, channel/backend callbacks and SDK coordination |
| Spans | timer/lifecycle state and reductions | ContextVar nesting, decorators and plugins |
| Artifacts and summary | manifests, publication, aliases, copy/digest/locking, lineage append and summary persistence | public objects and value conversion |
| Trace export | segmented span reads, lane assignment and trace construction | CLI/public entry points |
| Local viewer | HTTP server, discovery, query indexing and chart sampling | Click command that starts the server |

The frontend in `web/` is TypeScript/Vite/uPlot. Its built assets are included in
the Python wheel; Node.js is only needed when developing or building those assets.

Artifact copies restore permissions and nanosecond modification/access times;
copy publication restores them again after checksum verification reads the
destination. These are mutable filesystem timestamps, not immutable manifest
fields: subsequent reads can change access time. Exact-atime integration checks
require control over other readers, including readers outside the SDK process.
The exact-metadata fixture validates a read-stable test filesystem. On supported
Linux filesystems it sets the owner-controlled `noatime` inode flag only on its
private pytest directory and existing children; new files must inherit that
policy. This leaves explicit timestamp setting and nanosecond assertions intact
without changing production copies or system-wide mounts. Unsupported environments
fail with a request for a real `noatime` test filesystem, rather than retrying or
silently skipping precision checks.

## SDK module map

```
expr_tracker/
├── __init__.py        public names re-exported for `import expr_tracker as et`
├── tracker.py         the functional API (init/log/history/finish/artifacts/summary)
├── run.py             Run object + global singleton + backend fan-out
├── summary.py         run.summary mapping adapter
├── artifacts.py       Artifact + project-scoped ArtifactStore adapters
├── encoders.py        JSON coercion (numpy/torch/pydantic/dataclasses/...)
├── cli.py             `et history` / `et rules` / `et alert` / `et serve`
├── history/
│   ├── store.py       HistoryStore lifecycle and native state/callback adapter
│   ├── _store_types.py options and transaction/notification boundary types
│   ├── _store_writes.py state-sharing metric/span admission mixin
│   ├── writer.py      JsonlWriter adapter for NativeWriter
│   ├── reader.py      JsonlReader adapter + offline read_history entry points
│   ├── codec.py       RecordCodec: metric values -> JSON lines, warn-once
│   ├── series.py      MetricSeries adapter for NativeMetricSeries
│   └── frame.py       projection + dict/pandas/polars output
└── alerts/
    ├── __init__.py    public alert API + config resolution + engine wiring
    ├── models.py      AlertLevel/Message/ChannelConfig/WebhookPolicy/AlertRule
    ├── engine.py      native rule/watchdog adapters and callback coordination
    ├── dispatch.py    native dispatch-state adapter and backend invocation
    ├── backends/      lark, slack, dingtalk, wecom, webhook, email, callable
    └── expr/          public AST/builders and native parser/evaluator adapters
```

Native implementation modules live under `rust/`: `history/`, `alerts.rs`,
`alerts_parser.rs`, `alerts_plan.rs`, `spans.rs`, `artifacts.rs`, `summary.rs`,
`trace.rs`, and `viewer.rs`. `rust/lib.rs` registers their Python bindings.

Large native surfaces have private responsibility modules: `alerts/` separates
evaluation, rule transitions, delivery and watchdog lifecycle;
`history/reader/segments.rs` owns segmented snapshots/I/O while `reader.rs`
adapts Python calls; `history/state/operations.rs` groups native commit and query
operations; `history/writer/persistence.rs` owns flush, rollback and close methods;
`viewer/index.rs` owns temporary index refresh, `viewer/directories.rs` owns
server-path resolution and bounded directory browsing; and `trace/lanes.rs` owns lane
allocation. Writer/viewer unit tests live separately beside their implementations.
These are ordinary modules and inherent methods, not new forwarding APIs or
owners: locks, buffers and callback ordering stay on their existing objects.

`HistoryStore` shares its state with the private `HistoryWrites` mixin rather than
allocating a second adapter object. Compatibility serialization stays in
`store.py`, preserving the existing module-level `encode_line` hook. Public
options and helper imports remain available from their original module.

Handwritten source files, tests and benchmark scripts are limited to **800
physical lines**, including comments and blanks. Generated Web assets and
third-party dependencies are excluded. `tests/test_source_size.py` enforces the
limit for the source trees. Regression suites are grouped by responsibility
with shared fixtures/helpers; benchmark orchestration remains in
`benchmarks/compare_native.py`, with workload bodies/output checks in
`benchmarks/native_workloads.py`. Splitting files is a readability change, not
a claim of fewer algorithms, less state, or improved performance.

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

`HistoryStore` has no dedicated `Run` back-reference and does not inspect its
backends, alert registry, or Python summary implementation. Run owns that policy and
registers a capability provider alongside its commit callback. After flush and
cache-budget enforcement, history asks the provider for an eligible native
summary handle. An unavailable capability, changed callback, or store/series hook
selects ordered notification delivery. The provider is evaluated each time,
not cached across flushes, and registration is cleared on reinitialization.
Bound callbacks/providers can still retain their owner; this is a dependency
boundary change, not elimination of all object-reference cycles.

Observed Run commits build one independent shallow metrics snapshot shared by
the summary callback and backends; alerts retain the complete committed row.
Standard dictionaries with exact UTF-8 string keys are filtered in one native
pass, avoiding Python dispatch for every field. Dictionary/string subclasses
and surrogate keys retain the original Python projection, including their
`items`/`startswith` hooks and iteration errors. This does not eliminate the
snapshot, merge series/summary passes, cache projections across callbacks, or
change backend mutation visibility.

Some apparently repeated observer work belongs to distinct compatibility
boundaries. Summary observation is also a standalone mapping API and filters
hidden/non-string keys while respecting pinned keys; arbitrary key comparisons
can reenter the SDK. Series accepts standalone records and numeric subclasses,
so only unexposed validated native records may bypass its eligibility scan.
Removing those generic checks merely because ordinary Run input was normalized
would change supported behavior. Likewise, per-record notification completion
state is distinct from the producer's batch-tail ticket: recursive drains capture
the current queued tail, which can be an intermediate record from that batch.
Sharing one completion flag across the batch would lose that boundary. Further
fusion needs evidence beyond fewer loops or wrappers, without invoking arbitrary
Python callbacks while holding native state locks.

## Write path

```
et.log(data, step, commit)
        │
        ▼
Run.log ──> HistoryStore.log
                 │  exact builtin scalars: typed native ingestion
                 │  custom/nested values: SDK normalization
                 │  NativeHistoryState merges into the open row
                 ▼  commit when the step advances / commit=True / finish / flush(commit_open=True)
             native admission + cache/ordinal allocation
                 ├── NativeWriter queues and persists the encoded row
                 ├── unobserved local commits: native series + summary
                 └── observed commits: ordered NativeCommitQueue ticket
                 │
                 ▼
             SDK ticket delivery outside native state locks
                 ├── native series + summary + rule evaluation
                 └── Dispatcher / other backends (wandb, trackio, custom)
```

The ordinal allocation and the writer enqueue happen under one lock so that
row ordinal *N* is always physical line *N*; the query planner depends on it.

`HistoryStore.log` returns the `_step` the metrics landed on, or `None` when the call
was rejected. Run's commit callback forwards the complete committed metrics and
their actual step to remote backends, so a backend's row layout matches local
history instead of drifting on its own counter. Test the history result with
`is None`, since step 0 is falsy.

Open-row transitions, row admission, cache eviction and writer accounting are
native operations. Live appends and resume use the same native accounting rules,
so metadata and sparse indexes do not drift between those paths.

Ordinary scalar logs bypass Python JSON encoding. Fresh committed rows serialize
directly into one owned `Vec<u8>`, using a bounded cache of escaped metric keys;
they do not construct per-value JSON strings or intermediate row tables. Each
fresh scalar metric probes that cache once: hits borrow encoded keys, misses
populate it only below the 4,096-key bound, and uncached keys beyond that bound
serialize directly into the output vector without another key buffer. Partial
rows, reductions and compatibility inputs retain the indexed native object
representation. A shallow dictionary snapshot freezes inputs before releasing
the GIL. Exact loaded NumPy `float64` scalars are converted to builtin floats in
that snapshot without importing NumPy or changing the caller's dictionary.

Fresh scalar records retain their validation result and whether numeric
publication can proceed without overflow fallback. Inline publication reuses
that result rather than rescanning the row for publication/series eligibility.
General records keep ordinary validation. The internal Python-accessible
`NativeRecord.dictionary()` accessor invalidates the result before exposing its
mutable snapshot; the native publisher uses a separate unexposed accessor.
Huge integers still select ordered fallback before any observer is updated.

The encoded vector is moved into `Arc<Vec<u8>>` without copying its payload.
The record, cache and writer queue share this allocation. Normal flushes pass
borrowed `IoSlice` descriptors to vectored file writes, grouped by rotation
segment; no concatenated batch buffer is constructed. Partial writes and
interruptions advance those descriptors, with errors going through the existing
rollback/retry path. Python writer hooks retain compatibility batching; external
multiline/split payloads also use it when rotation needs newline-aware grouping.

This is **zero additional user-space payload copies after serialization** in
the ordinary native write path, not end-to-end zero-copy. Python references are
snapshotted, JSON bytes must be encoded, and ordinary file I/O copies bytes into
the kernel. Series still store numeric values, while summary retains Python
object references. Per-metric `VecDeque` windows avoid moving retained samples
at trimming boundaries and share each record's step/time metadata; point storage
is 16 rather than 32 bytes on 64-bit targets. Trimming before insertion avoids
the old temporary capacity doubling. Sparse metrics and later capacity growth
retain the previous last-N/hidden-retention semantics.

Validated series updates probe the bounded ASCII-key cache once on a hit and
borrow its existing string. Cacheable misses use an entry insertion without an
extra string clone; after 4,096 names, misses still use a temporary encoding.
Insertion remains before numeric conversion, preserving cache and prior-point
state when conversion raises. Clearing points/warnings keeps the bounded key
cache, as before. This removes hash probes, not snapshots or payload copies.

Typed admission and inline publication have separate eligibility checks.
Backends, active alert rules and custom callbacks can use native scalar admission
without bypassing ordered notification delivery or publishing series/summary
early. The fused local publisher remains enabled only while the notification
queue is idle and the standard series/summary observers are unchanged. Eligibility
is checked again after flushing, so observers installed during a flush receive
that same commit. Overridden admission hooks and unsupported values retain their
compatibility path. Empty alert registries return before extracting metric keys.
Neither path disables series, summary, rotation or cache backpressure.

`_log_local` is the local commit coordinator. The typed native prepare result is
unpacked once into explicit step, accepted-record, flush-due, cache-budget,
diagnostic and warning variables without allocating another result wrapper.
The coordinator reports the outcome, schedules persistence/eviction, then
attempts inline publication or stages notifications. Both native and
compatibility admissions share `_schedule_commit` and `_stage_notification`.
These are explicit phases of the existing write transaction, not a new lock or
worker: scheduling can release the GIL, so publication capability is resolved
afterward; ordered delivery still happens outside the transaction/state locks.
The low-level native ABI tuple remains unchanged.

The [100-metric investigation](benchmarks/native-migration.md#hundred-metric-data-path-investigation)
describes the earlier transport's copies and profiles its observed/NumPy paths;
the transport follow-up measures these changes against that exact native build.

## Read path

```
et.history(n, ...)
        │
        ▼
HistoryStore.get ──> NativeHistoryState cache snapshot/query planning
        │                 ├─ NativeReader disk fallback when cache is incomplete
        │                 ├─ include the open row, if requested and in range
        │                 └─ native patch merge and step selection
        ▼
batched JSON decode + owned projection ──> frame.to_output(dict | pandas | polars)
```

Both query kinds capture cache rows and their disk boundary together, and only
touch disk when the native snapshot cannot answer completely. Tail/range scans,
step selection and patch merging execute in Rust.

Unmerged disk queries validate the complete JSON grammar and step before
forwarding original row bytes; they do not parse and re-encode every value.
Python materialization decodes bounded batches of up to 1,024 rows and projects
freshly owned snapshots without copying each dictionary again. Corrupt batches
fall back to individual decoding with explicit warnings; extended numbers,
duplicate keys, surrogate escapes and query boundaries retain their semantics.

`et.history(run=...)` bypasses the store entirely and reads through `JsonlReader`
(`read_history`), which is what makes offline analysis and `et rules test` possible.

### Rotation segments

Metric and span writers default to `max_file_bytes=100 << 20`; `None` disables
rotation. The original file is never renamed. Additional, permanently retained
segments live in `<filename>.parts/00000001.jsonl`, incrementing numerically.
Writes split batches only at record boundaries, and oversized records stay intact.
On failure, rollback truncates the logical byte stream to its pre-batch size and
removes newly written suffix segments before retrying.

The sparse index and metadata use cumulative byte offsets and row counts across
all segments. Readers capture the segment list and each file's byte boundary per
query, then scan or reverse-tail them as one logical file. This preserves cache
boundaries, cross-segment patch merging, and snapshots while new segments appear.
Resume repairs the active segment's torn tail and restores the whole run's state.
Span trace export uses the same segmented raw reader.

### Rows versus steps

A step normally occupies one physical row, but an explicit commit followed by
more data for that step writes a *patch line*, so one step can span several rows.
The two are kept strictly separate in the read path:

* `JsonlReader.tail_rows(n)` returns **physical rows**; `JsonlReader.tail(n)` selects
  the most recently written **steps** and merges all their patches. Unique-step
  files use reverse reads; arbitrary patch histories require scanning older data.
* `HistoryStore._collect_tail(steps)` works in rows, then `_take_steps` merges and
  trims by *step* through native helpers, so `get(n)` never returns a half-merged
  oldest row.
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
| SDK lifecycle/write coordination | Python adapters | reinitialization, shutdown, admission and callback ordering |
| Native history/writer state locks | Rust | open row, cache, ordinal, buffer, index/meta and file append |
| Native commit queue/tickets | Rust | ordered delivery ownership and completion |
| SDK rule callback coordination | Python adapters | serialization across native transitions and reentrant callbacks |
| Native alert/channel state | Rust | deadlines, queue/rate limits, retries, dedup and rule state |
| Background worker lifecycle | native core + SDK adapters | scheduled flush/watchdog/dispatch work and shutdown draining |

Store writes enter the transaction lock before the store-state lock. Shutdown
rejects new writes, waits for accepted metric/span transactions, then flushes and
closes the writers before closing downstream sinks. Writer workers never acquire
store locks. Open rows have no timeout and no background commit thread.
Admission waits outside the store-state lock, so queue backpressure does not
block history reads. Rejected records do not consume cache row ordinals or reach
commit callbacks.
Read snapshots acquire the metric writer lock before the store-state lock, the
same order used when writer error handlers inspect history.

Commit notifications are staged in write order, then delivered after releasing
transaction/state locks. Committing callers wait for their own tickets; span-only
writes do not wait for unrelated callbacks. Delivery is serial and supports
recursive logging/flush, while lifecycle operations are rejected inside callbacks.
Reinitialization drains the old run's notifications before replacing its metadata.
Alert-series updates occur during ordered delivery, not admission, preventing
queued future rows from changing an earlier notification's scalar/window results.
Native state locks are released before invoking Python user/backend callbacks.

Notification coordination has one owner: `NativeCommitQueue` owns its condition
variable, delivery owner/depth and per-row completion. The SDK stages each batch
with its options, callback and ContextVar context, then receives only the last
row's completion ticket. It does not allocate a Python ticket for every row or
maintain a second notification condition/completion flag. The internal queue
protocol is `enqueue(batch) -> last_ticket | None` and
`drain(owner, callback, target=None)`. Recursive drains capture their queued tail
inside the native queue; they do not wait for an active callback ancestor or
consume later admissions past that boundary. Native completion still advances
when a callback raises. Tests observe native queue state without adding SDK
synchronization solely for instrumentation.

Run/store/span/plugin objects are process-owned. Fork resets process-global
locks and the active run/stack in the child; inherited objects are rejected before
acquiring their locks, and inherited exit hooks do not flush parent data.

## Extension points

| I want to add... | Where it goes |
| --- | --- |
| a notification channel | subclass `AlertBackend`, call `register_backend("type", cls)` |
| an expression function | update the public DSL metadata and matching native evaluator/compiled plan |
| an output format | add a branch in `history/frame.to_output` |
| a metrics backend | pass any object with `init/log/finish` in `backends=[...]` |
| an artifact storage mode | extend native artifact materialization and its SDK adapter (`copy` is default; `link` shares the caller's inode) |
| a value type to encode | extend `history/codec.py` (and `encoders.py` for the coercion) |
| a history tunable | add a field to `HistoryOptions`; it is validated and documented automatically |

## Local viewer

`et serve` scans the selected root and serves only discovered metric/span sources.
The CLI root is the initial location. The directory picker can browse any
server-readable directory; canonical selected roots travel with each request,
so one page never changes another page's root. Run IDs identify absolute
directories. The discovery cache retains only the most recent scan; requests
that miss it use their own discovery result, not a concurrently replaced cache.
Log sources and rotation segments remain contained within the request's root.
It reads the same retained JSONL segments as the SDK. A bounded cache of temporary
SQLite indexes merges metric patches and supports pagination/downsampling;
these indexes are disposable and never replace or modify experiment logs.

The registry is bounded to 32 indexes. A shared registry guard pins their lifetime
during queries, while each log has its own refresh/query mutex: already cached
independent logs can progress concurrently, but queries of the same log remain
exclusive. Creating, pruning or evicting an index requires an exclusive registry
guard and waits for active queries; queued registry writers can also delay new
readers. A refresh failure immediately drops the invalid index, and the next
registry write prunes its empty slot before applying the capacity policy.
Query errors retain healthy indexes. No index leases or background pool are
needed, but concurrent query working sets can increase memory use.

Chart downsampling bounds returned points, not work: selected rows still need
scanning/parsing. This ownership change does not make large chart queries
constant-time or parallelize cold index creation.

Only complete records are visible, so buffered/open training rows appear after
they are flushed. Large integers and Python's legacy nonfinite values are encoded
explicitly at the browser boundary instead of silently losing precision. The
server binds only to loopback and checks Host/Origin; remote access uses SSH
forwarding. See [Web viewer](guide/web-viewer.md) and
[native development](native-development.md).

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
