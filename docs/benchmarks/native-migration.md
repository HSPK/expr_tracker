# Native migration performance comparison

## Exact artifact access-time isolation

The earlier exact-atime integration failure is now addressed at its filesystem
boundary, without changing artifact production code or the native binary. The
strict fixture probes read behavior and, when needed on Linux, sets the
owner-controlled `FS_NOATIME_FL` only on its private pytest tree. New nested
files must inherit the policy. No privileged mount, FUSE service or dependency is
needed on this ext4 host. Unsupported/forbidden flags require an actual
`noatime` test filesystem and cause an explicit setup error, not a skip/retry.

The original publication/download permission and nanosecond mtime/atime
comparisons are retained in a shared assertion; atime must additionally match
the exact originally supplied historical nanoseconds. Explicit timestamp updates
still work. New regressions exercise competing reads, detect deliberately omitted
or second-truncated atime in copied outputs at both stages, reject ineffective
isolation/inheritance and restrict flag changes to the private tree.

Two complete SDK runs each pass **3,214 tests with five existing skips**.
Independent extracted-wheel artifact/native/rotation checks pass 89 tests;
31 Rust tests also pass. A separate traced diagnostic performs 50 real SDK
publication/download cycles with a 1 ms observation interval and 679 competing
reads: all 100 exact metadata checks pass with zero mismatches. These are
correctness diagnostics, not performance measurements. An initial extracted-wheel
runner used stdin and broke multiprocessing spawn; a file-based, main-guarded
runner fixes that harness issue and preserves artifact lock coverage.

Sanitized evidence and identities are retained in
`benchmarks/results/artifact-atime-isolation.json`. The unidentified external
reader from earlier investigations has not been identified or disabled; normal
production filesystem access times remain mutable. The resolved item is
deterministic exact-time test isolation, not a guarantee that arbitrary readers
cannot change experiment-file metadata outside this test fixture.

## Responsibility-based file splitting

All 13 handwritten source/test/benchmark files previously over 800 physical
lines are split by responsibility. The current largest authored file is 764
lines, and `tests/test_source_size.py` enforces the 800-line limit, including
comments and blanks but excluding generated assets/dependencies. This is source
organization, not a new algorithm, buffer, lock owner or speedup claim.

Moved Python function/decorator ASTs are exactly unchanged. All 3,210 original
parametrized test identities are preserved across the new filenames; one
source-size regression is added. Native exported names, object kinds, modules
and text signatures match the frozen previous wheel. Both wheels use the same
uv/maturin profile and unchanged manifests. New Rust modules use inherent
methods/private dependencies, while the Python write mixin shares the original
HistoryStore instance. Legacy `store.encode_line` interception is retained.

Validation passes 31 Rust tests and 437 independent extracted-wheel
native/viewer/alert/admission/rotation checks. The complete SDK run reports
**3,205 passes, one artifact exact-atime download failure, and five skips**.
Artifact source and the original precision assertion are byte-identical to the
frozen baseline; the previously demonstrated external-access interference
remains unresolved. No test is retried, skipped or relaxed to hide that failure.

An 18-trial fresh-process benchmark CLI smoke uses three alternating pairs on
CPU 23 for default hundred-metric logging, hundred-metric backend logging and
synchronous spans. It validates every value in 37,200 physical hundred-metric
rows / 3,720,000 metric values, including warmup. No builds/tests overlap those
trials. Short-run median means move roughly +1%; this smoke is not a statistically
conclusive performance campaign and establishes neither a causal slowdown nor
a universal no-impact guarantee. All timings, source/wheel identities, line
counts and actual validation results are retained in
`benchmarks/results/file-splitting.json`.

## Bounded per-log viewer ownership

The viewer previously held one global cache mutex across refresh and the entire
query, serializing unrelated experiments. It now uses a shared registry guard
and a per-log mutex. The registry guard pins index lifetime without an Arc lease
or pool; its exclusive guard alone creates/prunes/evicts indexes, preserving the
32-index bound. Refresh failure drops the invalid index immediately; empty slots
are pruned before eviction so they do not displace healthy indexes. Same-log
queries remain exclusive, and action errors retain healthy indexes.

The final production source and native binary exactly match the isolated,
measured prototype. The control binary exactly matches the preceding production
build; Python SDK, APIs, schemas and dependencies are unchanged. Five direct
Rust tests cover independent cached progress, same-log exclusion, active lifetime
and capacity, refresh-error rebuild/action-error retention, and invalid-slot
pruning. The controlled independent-progress test fails on the old runtime,
releasing and joining threads safely before asserting.

Two reversed-order five-pair HTTP campaigns use fresh viewer processes on CPUs
22/23, with the client on CPU 23 and no overlapping builds/tests. Four independent
logs share a verified 20,000-row, 100-metric fixture. Each chart checks totals,
step bounds, sample limits and all returned metric values. A fast ten-row log is
queried only after the large query's process CPU has advanced at least two ticks.
HTTP completion is marked after JSON reading and before client validation; a
deterministic native gate separately proves independent critical-section
progress. All 20 server trials, 420 chart responses and 200 fast responses are
retained in `benchmarks/results/viewer-lock-scope.json`.

| Viewer workload | First campaign, ms: control → candidate | Reversed-order confirmation, ms: control → candidate |
| --- | ---: | ---: |
| Cached small query during large query | 557.05 → 1.76 | 562.47 → 2.20 |
| Four independent cached chart queries | 2304.12 → 1190.42 | 2311.13 → 1184.93 |
| Cold index creation | 1093.55 → 1117.16 | 1088.68 → 1117.15 |
| Solo cached chart | 576.15 → 575.37 | 578.02 → 573.31 |

Cached four-log throughput improves about 1.9x on two server CPUs, while the
small cached request's interference latency falls about 256–317x. Candidate
fast responses finish before the large HTTP response in 50/50 overlaps, versus
2/50 for control; response ordering alone is not proof of lock independence.
Cold requests still wait roughly 550–570 ms during a large query: registry
writes need active readers to finish, and queued writers can delay new readers.
Cold creation regresses about 2%–3%. Post-parallel median RSS rises
91.07 → 105.42 MiB and 91.04 → 103.77 MiB; before parallel batches both versions
are around 73.6 MiB ready and 82.5 MiB after solo/interference queries. The index
bound is unchanged, but concurrent query working sets are not free. Chart
sampling still scans/parses selected rows; response limits do not bound query cost.

Two additional seven-workload SDK guardrail campaigns retain 140 trials and
verify 2,010,000 hundred-metric physical rows / 201,000,000 values, including
warmup. The table normalizes the reversed campaign back to control → candidate.

| SDK workload | First campaign, us/op: control → candidate | Reversed-order confirmation, us/op: control → candidate |
| --- | ---: | ---: |
| 100 metrics, SDK defaults | 46.09 → 46.01 (-0.18%) | 47.43 → 46.05 (-2.92%) |
| 100 changing values | 48.43 → 48.89 (+0.95%) | 48.80 → 48.67 (-0.25%) |
| 100 NumPy float64 values | 50.39 → 50.60 (+0.43%) | 51.12 → 51.44 (+0.62%) |
| 100 metrics, counting no-op backend | 59.75 → 60.53 (+1.30%) | 60.72 → 60.30 (-0.69%) |
| 100 metrics, ten alert rules | 129.23 → 128.69 (-0.42%) | 130.62 → 128.92 (-1.30%) |
| Five-metric buffered Run.log | 12.78 → 12.33 (-3.53%) | 12.21 → 12.26 (+0.38%) |
| Synchronous trace-only span | 15.85 → 15.57 (-1.75%) | 16.19 → 16.10 (-0.56%) |

These small/mixed SDK differences do not establish a universal speedup or
zero impact. NumPy mean regresses in both campaigns; backend p99 also worsens,
95.448 → 99.756 and 95.888 → 104.754 us. Other tails are mixed. The ownership
change is retained for the large, independently verified viewer responsiveness
gain with these explicit cold, memory and SDK tradeoffs.

The merged tree passes 31 Rust tests, 3,205 SDK tests with five skips, and 62
independent extracted-wheel viewer/native/queue/projection/rotation checks.
The original artifact atime assertion remains unchanged; its passing result
here does not resolve the previously demonstrated host-access race. Preliminary
measurements of the rejected invalid-slot variant or the earlier completion
marker are excluded from the final evidence. The HTTP harness uses a child-local
Python SIGINT handler compatible with Rust graceful shutdown and requires clean
server termination; no production signal behavior was changed.

## Single-probe series key caching

Validated series ingestion now performs one warmed ASCII-key cache lookup
instead of `contains_key` followed by `get`. A cacheable miss uses `entry` to
insert and borrow the existing encoding without a new string clone; a miss
after 4,096 names still uses a temporary encoding. Numeric conversion, partial
error state, overflow warnings, clear behavior, sample sharing and trimming are
unchanged. There is no new API, dependency, buffer, lock, snapshot or wrapper.

The isolated uv/maturin control wheel matches the preceding production native
binary exactly. Production was changed only after two seven-workload campaigns,
two short name-churn campaigns, and additional backend/long name-churn checks.
The confirmation reverses candidate/control roles to balance which runs first.
All use CPU 23, fresh subprocesses and alternating pairs; no builds/tests overlap
measurement. The merged source and wheel binary exactly match the prototype,
and the Python SDK is unchanged. All 190 trials are retained in
`benchmarks/results/series-cache-probe.json`, including verification of
2,635,000 hundred-metric physical rows / 263,500,000 metric values.

The table reports medians of per-process means, not pooled samples. Standard
workloads measure 20,000 calls plus 100 warmup rows per trial; the short
name-churn workload measures 1,000 calls and cycles a 1,024-payload ring of
distinct 100-metric name sets, exercising saturated caches and new series.

| Workload | First campaign, us/op: control → candidate | Reversed-order confirmation, us/op: control → candidate |
| --- | ---: | ---: |
| 100 metrics, SDK defaults | 47.75 → 46.77 (-2.05%) | 48.64 → 46.20 (-5.03%) |
| 100 changing values | 50.91 → 48.43 (-4.87%) | 50.40 → 47.71 (-5.34%) |
| 100 NumPy float64 values | 53.21 → 51.45 (-3.32%) | 52.67 → 50.52 (-4.09%) |
| 100 metrics, counting no-op backend | 62.98 → 61.23 (-2.78%) | 62.61 → 60.57 (-3.26%) |
| 100 metrics, ten alert rules | 132.53 → 130.26 (-1.71%) | 131.13 → 129.38 (-1.33%) |
| Five-metric buffered Run.log | 12.48 → 12.30 (-1.50%) | 12.31 → 12.46 (+1.17%) |
| Synchronous trace-only span | 15.73 → 15.77 (+0.27%) | 16.37 → 15.77 (-3.69%) |
| Short 100-metric name churn | 104.36 → 105.10 (+0.71%) | 100.81 → 103.65 (+2.82%) |

All five common hundred-metric mean gains repeat, but small-record/span results
are mixed and the short name-churn mean regresses in both campaigns. Backend
p99 worsens in both initial campaigns (99.03 → 103.14 and 95.16 → 108.70 us);
a further ten paired rounds improve mean 61.78 → 59.71 us (-3.35%) and p99
96.82 → 93.56 us. Longer name-churn trials, at 20,000 measured calls, improve
mean 79.72 → 77.00 us (-3.41%) and p99 128.72 → 120.50 us. These additional
checks do not erase the short-run regressions or establish uniformly better
tails. NumPy and rule p99 also reverse direction between the main campaigns.
The source of this tail variation is not established. The lookup simplification
is retained for repeated common-workload mean gains with these explicit
tradeoffs, not a claim of universal improvement.

The merged tree passes 26 Rust tests, 3,205 SDK tests with five skips, and 62
independent extracted-wheel scalar/queue/projection/rotation/viewer tests.
The prototype also passes 235 targeted SDK tests. New direct native tests check
the exact cache bound, ASCII encoding against Python JSON, cached allocation
reuse after saturation, overflow warnings, clear behavior and conversion-error
partial state. The original artifact access-time assertion is retained:
its passing result in this full run does **not** resolve the host-access
interference documented below, which remains an open isolation issue.

## Artifact access-time investigation: no runtime change

This section records the earlier, unisolated investigation. The later inode
policy fixture above resolves its test-isolation blocker on supported
filesystems; the diagnostic reader identity remains unknown.

The outstanding exact-atime assertion has been traced independently of
publication and checksum verification. A direct native download-copy probe
reproduces three mismatches in ten copies. For a failing copy, `utimensat`
receives the exact source nanosecond timestamp and succeeds, but a later stat
reports a newer destination access time. Watching only the probe destination
directory reveals additional opens/accesses absent from the copying process's
syscall trace; the other reader has not been identified.

On the host's ext4/relatime filesystem, traced copies followed by a diagnostic
1 ms observation interval produce 15/50 native mismatches and 50/50 Python
`shutil.copy2` mismatches. Uninstrumented immediate checks produce zero failures
in 100 copies of each implementation. These are interference diagnostics, not
performance measurements or evidence of a native-only regression.

A future-atime fixture was also tested, including mutations that drop access
time or truncate nanoseconds. Although that fixture passes the complete SDK
suite (3,205 passes, five skips), a separate traced native probe still fails
1/50 copies; the equivalent Python probe has zero failures in 50. The fixture
change is therefore **rejected**, not treated as a reliable isolation strategy.
The original test and runtime implementation remain intact. A private mount
namespace probe is denied on this host.

That investigation's final run with the original assertions reports **3,204 passes, one
exact-atime download failure, and five skips**. No retry, relaxed assertion,
skip, delay, or extra copy is added to production. The native binary and artifact
source hashes remain unchanged. Sanitized counts and the rejected experiment
are recorded in `benchmarks/results/artifact-metadata-investigation.json`;
reliable isolation of this integration check remains blocked.

## Single-probe escaped-key serialization

Fresh scalar serialization now probes the bounded escaped-key cache once.
Previously every warmed metric used `contains_key` followed by `get`; a cold
cacheable key was inserted and looked up again before writing. Hits now borrow
the cached bytes directly, misses serialize/write/insert once, and misses after
the 4,096-key limit still serialize directly into the output vector. There is
no new buffer, abstraction, dependency, lock, schema or ABI change.

An isolated control wheel rebuilt through uv/maturin is byte-identical to the
previous observer-projection production binary. The candidate uses the same
build settings and was measured before production changes. Two independent
seven-workload campaigns each use five alternating paired rounds: 140 trials,
validating 2,010,000 hundred-metric rows / 201,000,000 values.
`benchmarks/results/serializer-cache-probe.json` preserves all trials and
source/binary identities. The merged production serializer and wheel binary
match the measured candidate; the added Rust tests are test-only.

| Workload | First campaign, us/op: control → candidate | Independent confirmation, us/op: control → candidate |
| --- | ---: | ---: |
| 100 metrics, SDK defaults | 52.79 → 49.21 (-6.80%) | 51.22 → 48.18 (-5.93%) |
| 100 changing values | 54.48 → 50.70 (-6.94%) | 53.83 → 50.74 (-5.74%) |
| 100 NumPy float64 values | 56.74 → 54.57 (-3.82%) | 55.73 → 53.71 (-3.64%) |
| 100 metrics, counting no-op backend | 66.40 → 63.83 (-3.86%) | 65.84 → 61.75 (-6.22%) |
| 100 metrics, ten alert rules | 136.82 → 131.52 (-3.88%) | 135.70 → 129.91 (-4.26%) |
| Five-metric buffered Run.log | 12.41 → 12.62 (+1.72%) | 12.67 → 12.58 (-0.75%) |
| Synchronous trace-only span | 16.19 → 16.47 (+1.73%) | 16.05 → 16.33 (+1.75%) |

Hundred-metric mean gains repeat across all five cases. Small-record changes
reverse direction, while span means are about 0.28 us worse in both campaigns.
No span improvement or universally faster SDK is claimed. Default p99 improves
in both (159.16 → 154.24 and 156.21 → 151.60 us); backend p99 also improves
in both (125.28 → 115.51 and 108.53 → 94.58 us). Changing-value p99 improves
initially but worsens in confirmation (86.86 → 90.28 us). NumPy p99 is slightly
worse in both (101.80 → 104.37 and 93.64 → 93.89 us); rule tails are mixed.

Native validation passes 24 Rust tests, including new direct tests of the
exact 4,096-key cap, cached allocation reuse after saturation, uncached escaped
keys, reserved metadata warnings, encoded values and fallback without cache
population. Independently extracted production-wheel SDK/rotation/queue/viewer
checks pass 104 tests.

**At that stage, the full SDK suite was not clean:** both runs have 3,204 passes, five skips and
one artifact exact-access-time assertion failure. Artifact source is unchanged;
the preceding production control reproduces that assertion in five of five
isolated runs, and the candidate in two of five. This establishes a baseline
failure, not a serializer-cache regression; the subsequent investigation above
documents filesystem-access interference without claiming a clean original
suite. No retry/skip or artifact code
change is included in this serializer refactor.

## Validated observer metric projection

The standard observed path now builds its required metrics snapshot in one
native dictionary pass, replacing per-field Python `isinstance`/`startswith`
dispatch. Summary callbacks and backends still share that independent shallow
snapshot; alerts still receive the complete committed row. Key/value identity
and insertion order remain unchanged. Dictionary/string subclasses and
surrogate keys fall back to the original Python comprehension, preserving hooks
and mutation errors. This does **not** remove a dictionary copy, cache a mutable
projection, fuse series/summary processing, or add a new record wrapper/lock.

The prototype was measured in isolated source/wheel copies before production
edits. Both variants used identical uv/maturin production release settings;
the isolated control binary is byte-identical to the preceding notification
refactor. Seven workloads in five alternating paired rounds give 70 trials.
An independent six-workload confirmation adds 60 trials. The merged sources
and production-wheel native binary exactly match the measured prototype.
Full evidence is in `benchmarks/results/observer-projection.json`.

| Workload | First campaign, us/op: control → prototype | Independent confirmation, us/op: control → prototype |
| --- | ---: | ---: |
| 100 metrics, counting no-op backend | 73.59 → 66.43 (-9.72%) | 73.83 → 66.10 (-10.47%) |
| 100 metrics, ten alert rules | 145.26 → 134.83 (-7.18%) | 146.56 → 135.79 (-7.35%) |
| 100 metrics, SDK defaults | 51.98 → 52.49 (+0.98%) | 50.77 → 50.23 (-1.08%) |
| 100 changing values | 53.96 → 53.78 (-0.34%) | Not repeated |
| 100 NumPy float64 values | 56.26 → 56.44 (+0.33%) | 57.29 → 56.34 (-1.66%) |
| Five-metric buffered Run.log | 12.46 → 12.59 (+1.12%) | 12.51 → 12.68 (+1.38%) |
| Synchronous trace-only span | 16.02 → 15.73 (-1.80%) | 16.66 → 16.01 (-3.89%) |

Observer mean gains repeat. There is no demonstrated local hundred-metric
speedup, and the five-metric mean is slightly worse in both campaigns
(about 0.14-0.17 us/op). Span code does not use this projection and its
batch-mean variation is high, so its apparent gain is not attributed to the
change. The change is retained for measured observer-path gains, not a claim
that all paths become faster.

Backend p99 worsens in both campaigns: 111.70 → 117.12 and
110.25 → 114.48 us. Rule p99 improves in both: 216.08 → 200.58 and
219.05 → 205.70 us. NumPy p99 worsens 89.86 → 100.62 us initially,
but improves 103.37 → 95.96 us in confirmation. These are medians of
per-process quantiles; tail latency is not uniformly improved.

The qualifying campaigns validate 1,809,000 hundred-metric rows /
180,900,000 values. The merged tree passes 3,205 Python tests with five skips,
22 Rust tests, and 117 independently extracted production-wheel SDK/viewer
tests. Seven new regressions cover projection order/reference identity,
metadata-only rows, dictionary and string hooks, lossless surrogate keys,
hook-induced iteration errors, and summary/backend snapshot sharing without
changing the alerts' original row.

## Notification protocol simplification

This follow-up compares against the SDK **after** the publication-ownership
refactor below. Notification synchronization now belongs entirely to
`NativeCommitQueue`: the SDK's unused completion flag and notification Condition
are removed. That Condition had no production waiters; it existed for test
instrumentation but still incurred enqueue/drain wakeups on observed commits.
Tests now observe actual native queue state without adding production locks.

The internal delivery interface shrinks from five operations to two:
`enqueue(batch) -> last_ticket | None` and
`drain(owner, callback, target=None)`. Manual `next`, `complete`, and separate
`target` entry points have no callers and are removed. A nonempty batch returns
one Python completion ticket instead of a list of tickets for every row.
Per-row native completion flags remain necessary for recursive delivery.
Recursive drain boundaries are captured natively without creating another
Python ticket. Callback errors still complete their row and release delivery
ownership. The private queue ABI changes with its SDK adapter in the same wheel;
public SDK behavior, JSONL and the history prepare-result ABI are unchanged.

Both SDKs use the same interpreter/dependencies and production uv/maturin
release-wheel settings. Eight workloads in five alternating paired rounds yield
80 trials; an independent five-round confirmation adds 50 trials. Binary and
source fingerprints, every trial and validation are preserved in
`benchmarks/results/notification-simplification.json`.

| Workload | Full paired campaign, us/op: before → after | Independent confirmation, us/op: before → after |
| --- | ---: | ---: |
| 100 metrics, SDK defaults | 50.49 → 50.57 (+0.15%) | Not repeated |
| 100 changing values | 54.28 → 53.72 (-1.03%) | Not repeated |
| 100 NumPy float64 values | 54.05 → 54.76 (+1.32%) | Not repeated |
| 100 metrics, counting no-op backend | 77.61 → 73.28 (-5.58%) | 75.29 → 72.84 (-3.25%) |
| 100 metrics, ten alert rules | 147.27 → 142.23 (-3.42%) | 144.67 → 142.47 (-1.52%) |
| Five-metric buffered Run.log | 12.75 → 12.90 (+1.18%) | 12.60 → 12.60 (+0.04%) |
| Synchronous trace-only span | 16.13 → 16.47 (+2.10%) | 16.36 → 16.33 (-0.18%) |
| Writer under four readers | 169.31 → 158.52 (-6.38%) | 149.89 → 153.02 (+2.09%) |

Observer-path mean improvements repeat, with smaller gains in confirmation.
The local path is not the optimization target and has no demonstrated speedup.
Small-record/span mean regressions in the first campaign do not repeat at the
same magnitude. Contention changes direction with substantial batch-mean
variation; no contention improvement is claimed.

Tails remain mixed. Backend p99 improves 128.89 → 120.17 us initially but
worsens 109.96 → 121.11 us in confirmation. Ten-rule p99 worsens in both:
220.98 → 225.09 and 220.36 → 224.03 us. Changing-value p99 is
95.63 → 101.27 us. No uniform tail-latency improvement is claimed.

The two campaigns validate 1,407,000 hundred-metric rows / 140,700,000 values.
The final tree passes 3,198 Python tests with five environmental skips,
22 Rust tests, and 91 independently extracted-wheel SDK/queue/viewer tests.
Seven new native-queue cases cover empty batches, last-ticket/FIFO boundaries,
already-completed targets, recursive bounded drains, empty recursive drains,
and ordinary/interruption callback failures. Existing concurrency, reentrancy,
ContextVar, lifecycle, fork and retention checks remain in place.
The initial full run had one artifact access-time assertion failure in an
unchanged subsystem; five isolated baseline runs and five current runs passed,
and the subsequent complete suite passed. No artifact code was changed.

## Publication ownership and commit coordination

This Python-only change removes history's dedicated Run back-reference and its
knowledge of backends, alert rules and the Python Summary implementation. Run
supplies a dynamically resolved publication capability; history owns admission,
persistence scheduling and notification delivery. Bound callbacks/providers still
retain their owner: this simplifies policy dependencies, not all reference cycles.
`_log_local` coordinates the typed native result directly, while native and
compatibility paths share scheduling and notification staging. There is no new
lock, worker, JSONL schema or native ABI change.

An intermediate result-wrapper allocation and then a forwarding-only coordinator
were removed rather than retained as architectural abstraction overhead. The final
implementation is compared against the frozen pre-change SDK with the **same
native binary**, using eight workloads in five alternating paired rounds
(80 fresh-process trials). An independent five-round campaign repeats small
records and reader contention (20 additional trials). Python source fingerprints,
all trials and validation are in
`benchmarks/results/architecture-simplification.json`.

| Workload | Before, us/op | After, us/op | Mean change |
| --- | ---: | ---: | ---: |
| 100 metrics, SDK defaults | 50.75 | 50.37 | -0.77% |
| 100 changing values | 53.90 | 53.70 | -0.38% |
| 100 NumPy float64 values | 56.44 | 57.02 | +1.03% |
| 100 metrics, counting no-op backend | 75.61 | 76.74 | +1.50% |
| 100 metrics, ten alert rules | 142.64 | 145.43 | +1.96% |
| Five-metric buffered Run.log | 12.24 | 12.48 | +1.97% |
| Synchronous trace-only span | 15.64 | 15.87 | +1.49% |
| Writer under four readers | 156.14 | 162.67 | +4.18% |

The hundred-metric means remain close, but this is **not a speedup claim**.
Five-metric overhead repeats in confirmation: 11.94 → 12.27 us (+2.77%),
approximately 0.33 us/log. Reader contention's +4.18% does not repeat:
confirmation is 156.84 → 154.89 us (-1.24%). Its after-mean coefficient of
variation is 12.90% in the full campaign and 7.51% in confirmation, so no
reader-contention improvement or precise regression is established.

Tails are not uniformly preserved. NumPy p99 worsens 89.00 → 103.57 us;
default p99 is 142.98 → 143.30 us and backend p99 is 124.98 → 127.72 us.
Changing-value p99 improves 100.73 → 85.89 us, and rule p99 is
207.89 → 206.87 us. Small-record p99 is 32.74 → 32.67 us in the full
campaign but 32.59 → 33.10 us in confirmation. These are medians of
per-process quantiles, not pooled percentiles.

The final full campaign validates all 1,005,000 hundred-metric rows and
100,500,000 metric values after sampling. The final tree passes 3,191 Python
tests with five environmental skips. New regressions cover capability lookup
after flush, summary-hook and summary-object replacement, commit-callback
replacement during flush, and clearing registration on reinitialization.
Rust and the viewer are unchanged in this round. The remaining cross-language
commit stages, compatibility hooks and Python SDK are not claimed to have been
removed or fully consolidated.

## Validated follow-up: reuse scalar validation

This change was prototyped in an isolated source copy **before changing the
production implementation**. It caches fresh scalar validation and the numeric
overflow decision, allowing inline publication to skip its full-row preflight
and the series' repeated type-validation pass. It does not add a second numeric
buffer, share metric IDs, change observer delivery or remove the input snapshot.
General records continue through the existing checks. Exposing the native
record's mutable dictionary invalidates the cached validation first.

Both control and prototype were built with the same uv/maturin release-wheel
configuration. The rebuilt control's binary is byte-identical to the current
transport baseline. An initial direct-Cargo exploratory build produced a
different binary and inconsistent results; it was not used to approve the change.
The matched campaign covers nine workloads in five alternating paired rounds;
an independent five-round confirmation repeats the main scalar cases and span/
small-record checks. Full evidence is in `benchmarks/results/validation-reuse.json`.

| Workload | First paired campaign, us: baseline → prototype | Independent confirmation, us: baseline → prototype |
| --- | ---: | ---: |
| 100 metrics, SDK defaults | 56.77 → 52.05 (-8.3%) | 56.68 → 51.64 (-8.9%) |
| 100 changing values | 59.40 → 54.21 (-8.7%) | 59.09 → 53.97 (-8.7%) |
| 100 NumPy float64 values | 61.08 → 58.12 (-4.9%) | 60.75 → 56.65 (-6.7%) |
| Five-metric buffered Run.log | 13.07 → 12.44 (-4.8%) | 12.93 → 12.50 (-3.3%) |
| Synchronous trace-only span | 15.99 → 16.78 (+5.0%) | 16.02 → 16.20 (+1.1%) |

The main builtin-scalar mean gains reproduce across both campaigns. The first
NumPy prototype batch-mean variation is 16.0%, so its first exact ratio is weak;
the confirmation variation is 1.9%. Backend and ten-rule cases change only
-0.8% and -1.2% in the first campaign; they do not use the trusted inline
publisher, so **no meaningful observer-path gain is claimed**. Reader contention
changes -0.5%. Full-read mean changes -2.3%, but that unchanged reader path is not
the optimization target and is not attributed to validation reuse.

Tail latency is not uniformly improved: changing-value p99 is 93.82 → 107.95 us
in the first campaign but 109.43 → 91.85 us in confirmation. SDK-default p99
improves in both (152.05 → 147.14 and 150.40 → 144.41 us). Span means remain
slightly worse in confirmation; there is no claim of a span speedup. Memory is
approximately unchanged. Independently measured native publication falls from
the preceding campaign's 22.92 to 16.94 us; these are isolated, successive stage
measurements, not additive SDK costs or a paired stage experiment.

Only this demonstrated scalar-validation reuse was merged. The merged Rust
sources and native binary are **byte-identical to the measured matched-wheel
prototype**. The two qualifying campaigns contain 140 fresh-process trials and
validate 1,608,000 hundred-metric rows / 160,800,000 values. The prototype passes
the previous full suite; the merged tree passes **3,186 Python tests, five
environmental skips and 22 Rust tests**, including new exposed-snapshot
invalidation/reentrant-subclass tests and huge-integer fallback without partial
observer updates.

**Not yet validated or implemented:** shared metric descriptors/IDs, standard
observer projection reuse, scheduler coalescing, or a new native commit
coordinator. The measured gain is about 8-9% for ordinary hundred-metric logs,
not evidence that a broader architectural rewrite will achieve the same gain.

## Transport simplification

The fresh scalar path now uses one frozen input snapshot and one direct JSON
serializer. Escaped keys are cached, exact loaded NumPy `float64` inputs are
normalized in the snapshot, and observed scalar commits no longer detour through
Python per-field/JSON encoding. Empty alert registries skip metric-key extraction.
The series keeps per-metric deques and shares record step/time metadata.

Encoded rows retain their original vector allocation in `Arc<Vec<u8>>`. Normal
flushes borrow those vectors directly: a single buffer uses `write_all`, multiple
buffers use vectored writes in groups of at most 128 descriptors. Rotation,
partial writes, interruptions, rollback/retry and hook instrumentation remain
supported. Python writer hooks retain their old concatenated compatibility path;
external multiline/split byte payloads also use it when rotation is enabled.

**Zero additional user-space payload copies after serialization is not
end-to-end zero-copy.** Dictionary references are still snapshotted, JSON output
must be constructed, series store numeric samples, and ordinary file writes copy
into the kernel. External Python-byte enqueue and compatibility normalization
are not claimed to be copy-free.

### Paired results against the previous native build

These are **native/native**, not Python/native, comparisons. The baseline is the
preserved first-optimized wheel, whose binary hash matches
`logging-100-metrics.json`. The final campaign measures 20 workloads in five
alternating paired rounds: **200 fresh-process SDK trials**, with the same locked
Python/dependencies, CPU pinning and corpus. No measurements overlap builds,
tests or profiling. Full sanitized evidence and both binary identities are in
`benchmarks/results/transport-optimized.json`; historical campaigns are unchanged.

| Workload | Unit/op | Previous Rust | New transport | Speedup |
| --- | --- | ---: | ---: | ---: |
| Run.log, 100 metrics, SDK defaults | us | 88.77 | 58.57 | 1.52x |
| Run.log, 100 metrics, buffer 200 | us | 87.69 | 55.83 | 1.57x |
| Run.log, 100 longer names/decimal values | us | 92.69 | 59.56 | 1.56x |
| Run.log, 100 changing values | us | 92.30 | 60.45 | 1.53x |
| Run.log, 100 metrics, immediate flush | us | 126.33 | 93.39 | 1.35x |
| Run.log, 100 metrics, counting no-op backend | us | 155.62 | 76.73 | 2.03x |
| Run.log, 100 metrics, ten alert rules | us | 214.33 | 147.48 | 1.45x |
| Run.log, 100 NumPy float64 metrics | us | 314.31 | 61.15 | 5.14x |
| Buffered Run.log, five metrics | us | 16.44 | 13.28 | 1.24x |
| Buffered HistoryStore, five metrics | us | 14.38 | 11.65 | 1.23x |
| Run.log, five metrics, immediate flush | us | 45.67 | 41.66 | 1.10x |
| Writer under four continuous readers | us | 160.41 | 155.52 | 1.03x |
| Run.log, 64 KiB cache eviction | us | 16.32 | 12.93 | 1.26x |
| Real default 100 MiB rotation, 512 KiB records | ms | 1.278 | 0.607 | 2.10x |
| Synchronous trace-only span | us | 15.81 | 16.14 | 0.98x |
| Asynchronous trace-only span | us | 13.91 | 14.01 | 0.99x |
| Span with metric reduction | us | 17.03 | 17.12 | 0.99x |
| Offline full history, 100k steps | ms | 320.93 | 324.29 | 0.99x |
| Patched tail, 100 of 20k steps / 40k rows | ms | 71.19 | 74.36 | 0.96x |
| Run.log, 100k writes / 4 MiB cache | us | 16.85 | 13.18 | 1.28x |

Each hundred-metric trial checks every field of every physical row, sequential
steps, final summary, latest series points and zero drops; backend counts are
also checked. The final matrix validates **1,438,000 rows / 143,800,000 metric
values**. These checks run after timing and RSS sampling.

The hundred-metric batch-mean variation is below 3.6% before and 2.5% after.
Absolute immediate-write timings changed noticeably between development
campaigns on this shared host, so use the final paired ratios rather than mixing
its new native times with historical baseline times. Span means are 0.5-2.1%
slower, full-read mean is 1.0% slower and patched-tail mean is 4.5% slower in the
final campaign. This is a write-path improvement, not a universal speedup.

### Tails, memory and remaining costs

| Hundred-metric case | Previous p99, us | New p99, us | Previous peak RSS, MiB | New peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: |
| SDK defaults | 239.8 | 162.0 | 108.7 | 103.3 |
| Buffer 200 | 142.3 | 100.2 | 115.1 | 103.2 |
| Longer names/decimals | 159.4 | 98.6 | 143.9 | 130.9 |
| Changing values | 161.1 | 105.9 | 148.9 | 156.8 |
| No-op backend | 228.4 | 130.7 | 115.4 | 103.2 |
| Ten alert rules | 314.9 | 219.7 | 115.3 | 103.6 |
| NumPy float64 | 472.9 | 101.8 | 135.9 | 115.9 |

Per-point storage is now 16 rather than 32 bytes on this 64-bit target, and
pre-insertion trimming avoids the old temporary vector-capacity doubling.
This does **not** imply that total RSS halves. Owned vectors retain spare
serialization capacity instead of the old tightly sized slice allocations;
cache/writer byte counters still measure encoded length, not allocator capacity.
The changing-value case's peak RSS increases about 5.3%, while most other wide
cases decrease. Further allocation-capacity tuning must preserve input freezing
and must not silently reintroduce full-row transport copies.

Both default-rotation implementations physically persist 256 approximately
512 KiB rows, about 128 MiB total, in two retained segments, neither exceeding
100 MiB. Continuous-reader trials verify actual reader query counts; the small
writer gain is not obtained by disabling readers.

Uninstrumented isolated native ingestion falls from the investigation's
40.64 to **16.90 us**, while native series/summary publication falls from 25.71
to **22.92 us**. Those are successive isolated-stage campaigns, not paired SDK
timings and not additive pipeline costs. The dictionary snapshot remains about
0.40 us. Separately instrumented native-call counts remain about three per local
log, fall from nine to seven for the backend path, and from twelve to three for
NumPy. Series/summary hashing, numeric conversion, Python observer work and
cross-boundary coordination remain; there is no claim that all serialization
or per-metric work has disappeared.

Final validation: **3,181 Python tests passed, 5 environmental skips; 22 Rust
tests; three frontend unit tests, TypeScript/production build and a real Chromium
viewer check**. New regressions cover native admission with observers, NumPy
snapshot isolation, escaped/reserved fields, ingestion hooks, retained sparse
series/capacity growth, scalar/vectored partial-write handling, native rotation
failure rollback/retry and external multiline/split payload boundaries.
An independently installed final ABI3 wheel passes hundred-metric input-mutation,
rotation, summary/series and resume checks plus all six HTTP viewer tests. Its
native binary hash exactly matches the benchmarked build. The refreshed source
archive includes the transport implementation, benchmark tools, docs and evidence.

Reproduce against an extracted copy of the preserved optimized wheel:

```bash
uv run --no-sync python benchmarks/compare_native.py \
  --baseline-source /tmp/et-previous-native --baseline-native --native-source src \
  --data /tmp/et-benchmark-data --output /tmp/et-transport.json \
  --calls 20000 --repeats 5 --cpu 23 \
  --cases run_metrics_100_default run_metrics_100 \
    run_metrics_100_realistic run_metrics_100_changing \
    run_metrics_100_immediate run_metrics_100_backend \
    run_metrics_100_rules10 run_metrics_100_numpy \
    run_buffered store_buffered run_immediate store_concurrent_readers4 \
    run_eviction rotation_default_100mib span_sync span_async span_metrics \
    disk_full patch_tail_100 memory_cache_4mib

uv run --no-sync python benchmarks/profile_logging.py \
  --source src --metrics 100 --calls 10000 --cpu 23 \
  --output /tmp/et-transport-profile.json
```

## Optimized result

This is the first hot-path optimization campaign. The transport follow-up below
compares its exact native wheel with the subsequent implementation; numbers here
are preserved historical Python/native measurements.

The hot-path rewrite now outperforms the published Python baseline on the
**batch-mean median of all 28 measured workloads**. This does not mean every
individual call, tail percentile or memory footprint improves.
The same host/interpreter, CPU pinning, baseline source and checksummed corpus
were used for **five alternating paired rounds: 280 fresh-process SDK trials**,
plus five fresh-server viewer rounds. No benchmarks ran alongside builds/tests.
The original migration evidence below remains unchanged; new sanitized evidence
is in `benchmarks/results/native-optimized.json`.

Wide-record cases use 1,000 measured calls for 50 metrics and 200 calls for
500/2,000 metrics, each after 100 warmup calls. This bounds total series/cache
allocation; both implementations use identical counts. Every logging case
checks the final metric fields in addition to physical row counts and buffering
constraints.

| Workload | Unit/op | Python 0.4.1 | Optimized Rust | Speedup |
| --- | --- | ---: | ---: | ---: |
| Buffered HistoryStore, five metrics | us | 30.12 | 14.47 | 2.08x |
| Writer under four continuous readers | us | 1846.11 | 155.51 | 11.87x |
| Buffered Run.log, five metrics | us | 34.69 | 16.01 | 2.17x |
| Run.log, immediate flush | us | 63.33 | 33.43 | 1.89x |
| Run.log, 64 KiB cache with eviction | us | 35.28 | 16.41 | 2.15x |
| Run.log, 64 KiB segments | us | 36.09 | 16.63 | 2.17x |
| Real default 100 MiB rotation, 512 KiB records | ms | 2.172 | 1.276 | 1.70x |
| Run.log with ten alert rules | us | 131.55 | 83.21 | 1.58x |
| Run.log, 50 scalar metrics | us | 123.07 | 47.06 | 2.62x |
| Run.log, 500 scalar metrics | us | 968.26 | 410.96 | 2.36x |
| Run.log, 2,000 scalar metrics | ms | 3.730 | 1.601 | 2.33x |
| Synchronous trace-only span | us | 16.58 | 15.80 | 1.05x |
| Asynchronous trace-only span | us | 14.10 | 13.59 | 1.04x |
| Synchronous span with metric reduction | us | 20.93 | 17.90 | 1.17x |
| Cached tail, 100 steps from 20k | us | 390.27 | 168.77 | 2.31x |
| Cached middle range, 100 steps from 20k | us | 1814.47 | 342.51 | 5.30x |
| Offline tail, 100 steps from 100k | us | 981.84 | 718.50 | 1.37x |
| Offline indexed middle range, 100 steps | us | 1156.92 | 504.18 | 2.29x |
| Offline full history, 100k steps | ms | 455.73 | 310.94 | 1.47x |
| Patched tail, 100 of 20k steps / 40k rows | ms | 129.87 | 69.47 | 1.87x |
| Parse representative alert expression | us | 35.60 | 13.02 | 2.73x |
| Evaluate two-window expression | us | 19.23 | 10.64 | 1.81x |
| SHA256 digest, 16 MiB | ms | 12.942 | 11.991 | 1.08x |
| Publish artifact, 128 files | ms | 31.091 | 18.429 | 1.69x |
| Deduplicate artifact, 128 files | ms | 11.595 | 6.161 | 1.88x |
| Filter 100 spans from 32k | ms | 142.528 | 109.885 | 1.30x |
| Export Chrome trace, 32k spans | ms | 824.86 | 609.75 | 1.35x |
| Run.log, 100k writes / 4 MiB cache | us | 33.81 | 15.69 | 2.16x |

Compared with the initial migrated build's historical medians, ordinary
`Run.log` moved from 47.75 to 16.01 us, full history from 592.98 to 310.94 ms,
and large default-rotation records from 4.941 to 1.276 ms. Those are successive
benchmark campaigns, not paired old-native/new-native trials. Python/native
speedups in the table use only the new campaign's paired measurements.
Not every native workload improved over the initial build: patched tail moved
from 62.15 to 69.47 ms, and artifact deduplication from 5.494 to 6.161 ms.
Deduplication's new native batch-mean variation is 16.3%; other new native cases
are below 6%. Preserve this uncertainty rather than interpreting the medians as
hardware-independent guarantees.

### What changed

* Indexed, insertion-ordered native objects replace linear field lookups,
  removing quadratic wide-record construction/merge behavior.
* Exact builtin scalar dictionaries enter Rust directly. Their scalar snapshot
  is frozen before releasing the GIL; encoded bytes are shared by writer/cache.
  Unobserved local commits publish native series and summary together.
* Numeric series ingestion avoids Python name/value lists and caches compatible
  ASCII JSON names. Shared step strings reduce per-point allocation.
* Validated unmerged JSON rows pass through without per-value re-encoding.
  Python decodes bounded 1,024-row batches and avoids copying freshly owned
  unprojected snapshots.
* Ordinary spans use native duration accumulation and fixed-schema event
  encoding. SDK lifecycle avoids redundant construction, copying and stack
  filtering; injected clocks/plugins retain the compatibility path.
* Portable release builds use ThinLTO and one codegen unit, not
  `target-cpu=native`.

The optimizations do **not** disable summaries, series, alerts, rotation,
backpressure, callbacks or output materialization. Active observers/overridden
hooks retain ordered notifications. Observers added during flushing receive
that same commit; unsupported/subclass/surrogate values retain normalization.
Instrumented attribution now finds roughly three native calls per ordinary
local record, versus roughly thirteen in the regressing build. Profile timings
are attribution only, not the table's performance measurements.

### Tail latency and limits

Quantiles remain medians of per-process quantiles, excluding final drain.
Batch means, throughput and CPU include drain.

| Workload | Implementation | p50 / p95 / p99, us | Calls/s | Batch CPU, s |
| --- | --- | ---: | ---: | ---: |
| Buffered Run.log, 20k calls | Python | 30.91 / 49.02 / 77.92 | 28,824 | 0.689 |
| Buffered Run.log, 20k calls | Rust | 14.72 / 18.57 / 38.33 | 62,446 | 0.320 |
| Immediate Run.log, 3k calls | Python | 60.07 / 83.73 / 93.35 | 15,791 | 0.190 |
| Immediate Run.log, 3k calls | Rust | 31.24 / 42.17 / 62.27 | 29,912 | 0.100 |
| Ten-rule Run.log, 20k calls | Python | 124.10 / 161.56 / 215.48 | 7,601 | 2.631 |
| Ten-rule Run.log, 20k calls | Rust | 78.92 / 106.41 / 139.58 | 12,018 | 1.663 |
| Synchronous trace-only span | Python | 13.52 / 25.15 / 113.37 | 60,305 | 0.166 |
| Synchronous trace-only span | Rust | 13.36 / 32.08 / 63.14 | 63,311 | 0.158 |
| Asynchronous trace-only span | Python | 12.66 / 15.78 / 32.22 | 70,903 | 0.141 |
| Asynchronous trace-only span | Rust | 12.39 / 16.61 / 35.09 | 73,563 | 0.136 |
| Run.log, 2,000 metrics | Python | 3660.01 / 3909.05 / 5123.82 | 268 | 0.746 |
| Run.log, 2,000 metrics | Rust | 1535.91 / 1613.29 / 6377.35 | 625 | 0.320 |
| Writer under four readers | Python | 1854.82 / 1952.06 / 3108.62 | 542 | 10.257 |
| Writer under four readers | Rust | 12.93 / 19.27 / 45.84 | 6,431 | 3.148 |

The four-reader case still caps writes at ten seconds or 20k calls. Native
completed all 20k calls; Python completed fewer within ten seconds. Median
concurrent reader rates were 2,156 versus 5,260 queries/s: the writer gain did
not come from stopping readers. CPU totals include all threads and are not
equal-work CPU comparisons.

**The aspirational <=15 us ordinary Run.log and <=10 us lightweight-span
batch-mean targets have not been reached.** Standalone HistoryStore is 14.47 us,
but Run.log is 16.01 us; lightweight spans remain 13.59-15.80 us. Synchronous
span p95, asynchronous span p95/p99, and wide-record p99 remain worse than
Python. ContextVars, SDK lifecycle/eligibility checks, flushing and observer
work still cost time. The result is an end-to-end improvement, not a claim of
universally minimal latency.

### Rotation, memory and viewer

Each default-rotation trial physically wrote all 256 records, about
128.017 MiB, into two retained segments. For the first native trial the
segments were 104,347,472 and 29,888,504 bytes, both within the 104,857,600-byte
threshold. Row counts, final row contents and all segment limits are checked
in every trial. Median throughput rose from about 230 to 392 MiB/s.

| Workload | Python peak RSS, MiB | Optimized Rust peak RSS, MiB |
| --- | ---: | ---: |
| Buffered Run.log, 20k writes | 47.6 | 46.7 |
| Full 100k history, repeated queries | 232.0 | 169.2 |
| Run.log, 2,000 metrics | 98.8 | 88.2 |
| 100k writes with 4 MiB cache | 55.3 | 56.1 |
| Synchronous trace-only spans | 38.3 | 40.9 |
| Real 100 MiB rotation | 39.4 | 41.2 |
| Writer under four readers | 42.2 | 47.4 |

Full-read peak RSS is down about 27% versus Python. Native import, span and
contention memory are not universally lower.

Five real-server viewer rounds measured discovery 1.12 ms, cold 100k-row
indexing 689.06 ms, warm full-range charts 209.49 ms, 1,000-step-range charts
3.65 ms, latest-100 records 1.71 ms, and a newly appended 100-row segment
refresh 2.54 ms. This is a native-only viewer comparison with the historical
viewer numbers below, not a comparison against a nonexistent Python server.
Full-range charts still scan source rows; no chart pyramid/column index was
added. Return-point limits do not imply constant query cost.

Validation after optimization: **3,171 Python tests passed, 5 environmental
skips**, **18 Rust tests**, frontend unit/TypeScript/build/Chromium checks, and
an independently installed ABI3 wheel's SDK/rotation/history/summary/span/
artifact/HTTP checks. New tests cover wide scalar records, frozen input across
GIL handoff/I/O, observer changes during flush, reentrant numeric conversion,
foreign pinned summary keys, batched decoding and independent query snapshots.
Native parser tests compare step-only validation with full lossless parsing
over escaped/duplicate keys, malformed syntax and extended values.

## Hundred-metric data-path investigation

This investigation describes the pre-transport implementation, not the current
write path. Its copy accounting and priorities are retained as the motivation
for the transport follow-up.

Additional measurements focus on 100 metrics per log, not extrapolation from
five metrics. Eight workloads each use five paired rounds (**80 fresh-process
trials**), CPU 23, the unchanged optimized release extension, and the same
published Python baseline. Each process makes 100 warmup calls followed by
20,000 measured calls, except immediate-flush trials, which make 3,000.
Initialization/input-fixture generation are excluded; final drain is included.
Default-setting trials use the SDK's 50-row buffer, 1 GiB cache and enabled
100 MiB rotation. Other trials use a 200-row buffer and 64 MiB cache, unless
immediate flushing sets the buffer to one.

| 100-metric workload | Python, us/log | Rust, us/log | Speedup | Rust p95 / p99, us |
| --- | ---: | ---: | ---: | ---: |
| Builtin floats, SDK defaults | 215.09 | 81.36 | 2.64x | 99.84 / 217.55 |
| Builtin floats, buffered 200 rows | 219.14 | 83.42 | 2.63x | 102.29 / 156.31 |
| Longer names and nontrivial decimals | 248.21 | 87.94 | 2.82x | 105.74 / 151.64 |
| Values change on every log | 241.31 | 86.75 | 2.78x | 102.88 / 157.28 |
| Flush every call | 250.49 | 106.13 | 2.36x | 130.69 / 144.86 |
| One counting no-op backend | 220.46 | 152.73 | 1.44x | 174.30 / 256.37 |
| Ten alert rules | 328.80 | 207.53 | 1.58x | 235.63 / 315.26 |
| `numpy.float64` values | 225.47 | 303.86 | **0.74x** | 335.96 / 442.52 |

The NumPy row is a **34.8% regression**, specific to `numpy.float64` in this
matrix, not a claim about every NumPy dtype. The no-op backend adds about
69 us versus the local buffered path even without networking. Merely
activating an observer changes the SDK path substantially.

Changing-value trials cycle 1,024 precomputed dictionaries of distinct finite
floats; generation is outside timing and the fixture is included in RSS. This
avoids generating values inside the measured call and avoids pretending that
unchanging `0.5` values represent every training workload. Paired calls include
the caller's `dict(payload)` copy; attribution below reuses its input dictionary.
Long-name trials actually evict around 4,000 rows under the 64 MiB cache cap.
All rows still reach disk.

Every trial validates **every metric of every physical row**, sequential steps,
the final summary and latest series points, and absence of drops. Backend
trials also verify callback counts. In total, 1,438,000 rows and 143,800,000
metric values were checked. Validation runs after latency/CPU/RSS sampling.
Evidence, including profiles and microbenchmarks, is preserved in
`benchmarks/results/logging-100-metrics.json`. No SDK/Rust implementation changes
were made during this investigation.

### Local path and copy accounting

For a plain builtin-float dictionary with no open partial row or external
observer, the path is:

```text
Run.log -> HistoryStore.log -> _log_local
  -> NativeHistoryState.prepare_local
     -> dict snapshot
     -> scalar_object: owned keys + JSON value fragments
     -> ingest_object: encoded object -> open-row table
     -> close: open-row table -> committed-row table
     -> NativeRecord: assemble JSON -> shared encoded bytes
     -> native writer/cache admission
  -> SDK flush/timer/diagnostics boundaries
  -> NativeHistoryState.publish_local
     -> numeric series
     -> summary
```

Source anchors: `rust/history/state.rs` (`ingest_values`, `ingest_object`,
`State::close`, `prepare_local`, `publish_local`, `enqueue_record`),
`rust/history/record.rs` (`scalar_object`, `NativeRecord::new`),
`rust/history/json.rs` (`JsonObject::encode`), and
`rust/history/writer.rs` (`flush`, `append_data`).

| Operation | Actual ownership/copy behavior |
| --- | --- |
| Input snapshot | One shallow Python dictionary copy. Keys/values retain references; numeric objects are not deep-copied. |
| Scalar conversion | Each key gets an owned Rust string, and each float gets a JSON-token string. |
| Three native object tables | Fields are reinserted/rehashed. String handles move; this is not three deep copies of all metric strings, but still repeats table allocation/work. |
| Final JSON construction | Each key is serialized into a temporary quoted string; raw value fragments are appended into the final string. For 100 floats there are roughly 300 per-field string-construction sites before metadata/output, derived from source rather than an allocator counter. |
| `String::into_bytes` | Ownership transfer, no full-row byte copy. |
| `Vec<u8>` to `Arc<[u8]>` | A new slice-Arc allocation and full-row byte copy in the standard conversion. |
| Writer/cache `Arc` clones | Reference-count changes, no full-row byte copy. |
| Flush batch concatenation | Each row is copied once into a contiguous batch vector. |
| File write | Ordinary buffered kernel I/O; not zero-copy or fsync durability. |
| Series / summary | Series copy numeric samples; summary updates Python key/value references, not deep copies. |

Thus the local path has one dictionary snapshot and two full-row **user-space
copies after final JSON construction** (slice-Arc conversion and batching).
Fragment serialization/construction and kernel copying are separate costs.
The input dictionary snapshot must survive GIL release during I/O or admission;
removing it without another immutable snapshot would reintroduce the previously
tested input-mutation race.

### Where the time goes

Uninstrumented isolated stage medians, five rounds of 10,000 operations:

| Isolated operation | us/op |
| --- | ---: |
| Shallow copy of the 100-field dictionary | 0.40 |
| Python JSON encoding of the scalar payload | 28.12 |
| Native ingestion + row construction, without a writer | 40.64 |
| Native series + summary publishing, without a writer | 25.71 |

These are **not additive pipeline measurements**. The isolated native stages
use fixed timestamps/scalar objects and do not perform disk I/O; Python JSON
encoding does not include equivalent state/observer work. They show that the
0.4 us dictionary copy is not the dominant target.

The separately instrumented local profile attributes approximately 43.4 us/log
to `prepare_local` and 29.1 us/log to `publish_local`. These two own-time buckets
account for roughly 76% of that profiled workload. Profiler timings are not
substitutes for the paired table. Native-call counts are approximately 3 per
local log, 9 with the backend, and 12 with `numpy.float64`.

The builtin observed path still performs SDK per-field normalization, another
dictionary snapshot, decoded-record construction and metric filtering before
callback delivery. It generally creates four SDK/native Python dictionaries
(normalized fields, frozen snapshot, full decoded record, filtered metrics),
rather than the local path's one snapshot. Callback/notification objects mostly
hold references, not additional deep copies.

`numpy.float64` is a float subclass: the exact-builtin scalar fast path rejects
it, while `jsonable_encoder`'s primitive check returns it unchanged. The second
typed-ingestion attempt therefore rejects it again, leading to:

```text
failed typed attempt -> Python normalization
  -> failed typed attempt -> Python JSON string/bytes/string
  -> Rust String + parse + native object tables + encode
  -> Python string + JSON decoding
  -> UTF-8 bytes + separate native writer/cache buffers
  -> ordered SDK observers
```

This path includes two unsuccessful dictionary snapshots, the normalized
dictionary, JSON-decoded record and filtered metrics dictionary. It also
materializes complete strings/bytes several times, and unlike native-record
admission its writer/cache buffers are not one shared allocation. This is
structural overhead, not a reason to drop NumPy support or silently cast
arbitrary subclasses.

### Simplification priorities

1. **Stop avoidable compatibility detours.** Handle supported NumPy scalars
   explicitly and losslessly at the normalization boundary, and separate
   eligibility for typed admission from eligibility for inline publication.
   A backend/rule should require ordered observer delivery, not automatically
   force builtin metrics through per-field Python encoding. Keep hook fallback,
   notification order, reentrancy and observers installed during flush.
2. **Avoid work for zero rules.** `alert_rules_for_step` in `rust/alerts.rs`
   currently extracts/copies every record key before discovering that its rules
   dictionary is empty. The no-op-backend profile still makes one registry
   evaluation per record (about 11.6 profiled us/log). An empty-rule early return
   can avoid selection/key construction without removing registered observer
   hooks. The real uninstrumented gain must be measured after implementation.
3. **Use one typed snapshot and one serializer for fresh committed rows.**
   Capture validated scalar values once; reuse cached escaped key descriptors
   and numeric conversions for serialization/series/summary. Serialize directly
   into one final buffer instead of creating per-value JSON strings and passing
   fields through three object tables. Keep the general merge/patch/reduction
   path for partial rows and complex values. Dynamic keys, reserved names,
   nonfinite values, arbitrary integers, surrogates and callback materialization
   remain compatibility requirements.
4. **Replace series batch shifting with bounded per-metric storage.**
   `StoredPoint` currently contains an `Arc<str>` and two floats (32 bytes on this
   64-bit target). At the default 4,096-point window, vectors allow about 8,192
   live entries, then drain roughly 4,097 entries at once. The remaining 4,096
   entries move: about 12.5 MiB across 100 metrics at one drain boundary, plus
   reference drops. Under ordinary vector doubling, capacity can grow to 16,384
   before that drain, about 50 MiB of point capacity across 100 metrics; this
   capacity figure is an implementation estimate, not an allocator measurement.
   A bounded ring avoids shifting and overgrowth. Shared step/time metadata
   could further reduce point width, but sparse metrics must retain their own
   last-N semantics; a naive global last-N row ring is not equivalent.
5. **Optimize transport copies after the representation changes.**
   Reusing owned buffers or vectored writes may remove the slice-Arc/batch
   copies. Removing per-field strings, hashing, repeated conversions and series
   moves is better supported by current attribution than deleting the 0.4 us
   snapshot or optimizing memcpy alone. Partial writes, rollback, rotation,
   retries and portable wheels must remain correct.

Reproduce the additional matrix and attribution independently:

```bash
uv run --no-sync python benchmarks/compare_native.py \
  --baseline-source /tmp/et-python-baseline/src --native-source src \
  --data /tmp/et-benchmark-data --output /tmp/et-100-metrics.json \
  --calls 20000 --repeats 5 --cpu 23 \
  --cases run_metrics_100_default run_metrics_100 \
    run_metrics_100_realistic run_metrics_100_changing \
    run_metrics_100_immediate run_metrics_100_backend \
    run_metrics_100_rules10 run_metrics_100_numpy

uv run --no-sync python benchmarks/profile_logging.py \
  --source src --metrics 100 --calls 10000 --cpu 23 \
  --output /tmp/et-100-metrics-profile.json
```

## Initial migration result (historical)

Measured on 2026-10-12 with Python 3.12.13, Linux x86_64/glibc 2.39 and an
AMD EPYC 7V13 VM exposing 24 single-threaded cores. Ordinary tests used CPU 23;
the contention case additionally used reader CPUs 20-22.

**This migration is not a blanket performance win.** Cached range queries,
patched history, expressions, artifact publication/deduplication and trace export
improve. Ordinary scalar logging, spans, large records and full materialized
history reads remain slower than the Python baseline.

All figures below are medians from five paired rounds. Speedup is Python time
divided by native time: below 1 means a regression.
Full evidence is saved in `benchmarks/results/native-migration.json`: 250 isolated
SDK trials, five viewer trials, checksums and environment metadata.

| Workload | Unit/op | Python 0.4.1 | Initial Rust | Speedup | Latency change |
| --- | --- | ---: | ---: | ---: | ---: |
| Buffered HistoryStore, five metrics | us | 30.70 | 39.69 | 0.77x | +29.3% |
| Writer under four continuous readers | us | 1788.40 | 262.71 | 6.81x | -85.3% |
| Buffered Run.log, five metrics | us | 35.09 | 47.75 | 0.73x | +36.1% |
| Run.log, immediate flush | us | 64.36 | 76.61 | 0.84x | +19.0% |
| Run.log, 64 KiB cache with eviction | us | 35.51 | 48.63 | 0.73x | +36.9% |
| Run.log, 64 KiB segments | us | 35.37 | 48.93 | 0.72x | +38.4% |
| Real default 100 MiB rotation, 512 KiB records | ms | 2.217 | 4.941 | 0.45x | +122.9% |
| Run.log with ten alert rules | us | 133.70 | 108.04 | 1.24x | -19.2% |
| Synchronous trace-only span | us | 16.72 | 24.16 | 0.69x | +44.5% |
| Asynchronous trace-only span | us | 15.36 | 21.96 | 0.70x | +43.0% |
| Synchronous span with metric reduction | us | 20.72 | 33.19 | 0.62x | +60.2% |
| Cached tail, 100 steps from 20k | us | 402.33 | 325.53 | 1.24x | -19.1% |
| Cached middle range, 100 steps from 20k | us | 1906.59 | 570.29 | 3.34x | -70.1% |
| Offline tail, 100 steps from 100k | us | 973.90 | 990.56 | 0.98x | +1.7% |
| Offline indexed middle range, 100 steps | us | 1164.48 | 883.12 | 1.32x | -24.2% |
| Offline full history, 100k steps | ms | 458.22 | 592.98 | 0.77x | +29.4% |
| Patched tail, 100 of 20k steps / 40k rows | ms | 135.71 | 62.15 | 2.18x | -54.2% |
| Parse representative alert expression | us | 37.46 | 14.12 | 2.65x | -62.3% |
| Evaluate two-window expression | us | 20.37 | 11.76 | 1.73x | -42.3% |
| SHA256 digest, 16 MiB | ms | 13.111 | 12.016 | 1.09x | -8.4% |
| Publish artifact, 128 files | ms | 31.088 | 18.734 | 1.66x | -39.7% |
| Deduplicate artifact, 128 files | ms | 12.074 | 5.494 | 2.20x | -54.5% |
| Filter 100 spans from 32k | ms | 150.001 | 110.404 | 1.36x | -26.4% |
| Export Chrome trace, 32k spans | ms | 850.79 | 643.12 | 1.32x | -24.4% |
| Run.log, 100k writes / 4 MiB cache | us | 35.36 | 49.28 | 0.72x | +39.4% |

The native indexed-range and artifact-publication batches had variability of
16.4% and 11.7% respectively, including one slow trial each. Their medians show
improvement, but those exact ratios should not be treated as universal constants.
Other native batch-mean coefficients of variation were below 9%.

### Tail latency, throughput and CPU

Per-call quantiles exclude the final drain; throughput and batch CPU include it.
Quantiles are medians of the five per-process quantiles, not pooled percentiles.

| Workload | Implementation | p50 / p95 / p99, us | Calls/s | Batch CPU, s |
| --- | --- | ---: | ---: | ---: |
| Buffered Run.log, 20k calls | Python | 31.00 / 51.85 / 85.66 | 28,502 | 0.702 |
| Buffered Run.log, 20k calls | Rust | 43.74 / 67.24 / 84.64 | 20,944 | 0.954 |
| Immediate Run.log, 3k calls | Python | 60.21 / 85.85 / 97.47 | 15,538 | 0.192 |
| Immediate Run.log, 3k calls | Rust | 71.30 / 103.55 / 117.55 | 13,053 | 0.230 |
| Ten-rule Run.log, 20k calls | Python | 124.20 / 168.01 / 260.58 | 7,480 | 2.674 |
| Ten-rule Run.log, 20k calls | Rust | 100.33 / 134.09 / 210.47 | 9,256 | 2.156 |
| Writer under four readers | Python | 1853.33 / 1966.73 / 3597.75 | 559 | 10.261 |
| Writer under four readers | Rust | 36.73 / 63.57 / 1064.36 | 3,807 | 5.300 |

The contention case is capped at ten seconds or 20k writes. Python completed
roughly 5.2k-5.7k writes in ten seconds; Rust completed all 20k in roughly
5.1-6.3 seconds. Concurrent read throughput was also higher: approximately
2,149 versus 2,524 queries/s. This is not a speedup obtained by disabling readers.
Its batch CPU sums all threads and the two implementations complete different
numbers of operations; do not interpret that column as equal-work CPU savings.

### Real 100 MiB rotation

Both implementations wrote about 128.017 MiB in two retained segments. Python
segment sizes were 104,347,485 and 29,888,506 bytes; native sizes differed by one
byte. All 256 rows were flushed, and neither segment exceeded the
104,857,600-byte threshold. Data-rate medians were approximately **226 MiB/s**
for Python and **101 MiB/s** for Rust.

The threshold is therefore exercised physically, not just through tiny test
limits. The large-record workload still regresses substantially.

### Memory

| Workload | Python peak RSS, MiB | Rust peak RSS, MiB |
| --- | ---: | ---: |
| Buffered Run.log, 20k writes | 47.6 | 47.1 |
| Full 100k history, repeated queries | 232.0 | 239.7 |
| 100k writes with 4 MiB cache | 54.9 | 55.7 |
| Synchronous trace-only spans | 38.3 | 40.9 |
| Real 100 MiB rotation | 39.4 | 45.2 |
| 32k-span trace export | 42.1 | 43.7 |

Native memory is not uniformly lower. Removing redundant full-read
representations reduced the initial Rust full-read peak from approximately
317.4 MiB to 239.7 MiB, but it remains about 3.3% above Python.

### Native Web viewer

These are five fresh-server rounds over the same 100k-step corpus, with two
selected chart metrics and a maximum of 1,000 returned points.

| Operation | Median |
| --- | ---: |
| Discover the run | 1.01 ms |
| First index of all 100k steps | 699.66 ms |
| Warm full-range chart | 229.49 ms |
| Warm 1,000-step range chart | 3.79 ms |
| Latest 100 records | 1.76 ms |
| Index 100 newly appended rows in a new segment | 2.51 ms |
| Server RSS before indexing | about 38.7 MiB |
| Server RSS after indexing | about 50.5 MiB |

Unbounded record pagination originally caused SQLite to choose the step-range
index and sort all rows. Removing the unnecessary full-range predicate reduced
that measured operation from about 113 ms to 1.76 ms. Warm full-range chart
sampling still scans selected source rows; downsampling limits response size,
not scan cost.

### Investigation and remaining cost

The first migration build regressed more severely. Profiling led to fixes for
redundant JSON conversion, per-append timer-worker wakeups, generic evaluator
window conversion, full-read copies, artifact/trace I/O and serialization, and
unnecessary GIL handoffs. A related writer-state/GIL lock inversion during Python
exception formatting was also corrected and covered by reentrant-error tests.

A dedicated four-reader reproducer initially failed to finish 1,000 native writes
within 120 seconds. After fixing starvation, the independent reproducer completed
in 0.231 seconds; the final paired benchmark above additionally exercises
20,000 writes and verifies active reader throughput.

The remaining regressions chiefly occur where small operations repeatedly cross
the SDK/native boundary or large JSON payloads need conversion and copying.
Rust ownership does not remove Python normalization, callback and output costs.
The next useful optimization target is reducing those conversions and call
boundaries, not claiming that a language change alone guarantees higher speed.

Final validation on this host: **3,159 Python tests passed, 5 skipped**
(external webhook/CUDA prerequisites), **16 Rust tests**, **3 frontend tests**,
TypeScript/production build, real Chromium smoke and an independently installed
wheel's SDK/rotation/artifact/viewer checks. The native wheel is approximately
2.83 MiB; the cleaned source archive is about 640 KiB. Other OS/architecture
wheels require their release-matrix CI; these are Linux-local measurements.

## Method

The baseline is the exact published `v0.4.1` Python implementation, commit
`cafa4be8aaf501522bab8aaee42024ea76f4b8cd`. Both implementations run with the
same Python interpreter and installed dependency versions. The native extension
is built with maturin's **release** profile, not Cargo's debug profile.

Each workload runs in a fresh subprocess. Five paired rounds alternate the order
of Python and Rust to reduce ordering bias. Results report the median of the
five batch means; per-call p50/p95/p99 and CPU time are retained separately.
GC remains enabled and is collected before each measured batch.

Single-threaded measurements pin the process to one available CPU. The
four-reader case pins the writer to that CPU and reader threads to three other
available CPUs. It reports writer latency/throughput and reader query counts,
not merely a native function's duration. That case stops after ten seconds or
the requested call count, whichever happens first.

Logging tests include the final buffer drain, but exclude run initialization
and final summary persistence. Read tests use a warm OS page cache; "disk" means
the SDK's file-backed path rather than a cold physical disk. Neither backend
adds an `fsync`, so throughput is **not** a power-loss durability measurement.

The shared, checksummed corpus contains:

* 100,000 ordered steps with five scalar metrics, retained 1 MiB rotation segments,
  and schema-compatible index/meta sidecars produced by the baseline writer.
* 20,000 patched steps / 40,000 physical rows.
* 32,000 reverse-completion-order spans with 256-byte payloads.
* A 16 MiB checkpoint and 128 files of 16 KiB each.

The default-rotation workload actually writes 256 records containing 512 KiB
payloads, exceeding **100 MiB**. It checks multiple real segment files, every
segment's size limit, total bytes and the number of flushed rows.

Memory results use fresh-process Linux RSS/high-water RSS, including Rust and
SQLite allocations. `tracemalloc` alone would omit native allocations. Full-read
batches make three queries; assignment retains the previous result until its
replacement is constructed, identically on both implementations. Their peak RSS
is therefore a repeated-query peak, not the footprint of one isolated result.

Native viewer measurements start a real HTTP server for each round, verify
100,000 indexed rows and bounded chart output, query ranges and record pages,
append a new retained segment, and check incremental refresh. They include
loopback HTTP transport and client JSON decoding. The Python baseline has no
equivalent Web UI, so these results are not a before/after server comparison.

## Reproduce

Use Linux and Python 3.11+ for the benchmark tooling. The SDK still supports
Python 3.10+. Export the baseline into a separate artifact directory; this does
not change the current worktree:

```bash
mkdir -p /tmp/et-python-baseline
git archive v0.4.1 | tar -x -C /tmp/et-python-baseline
npm --prefix web ci
npm --prefix web run build
uv sync --locked --all-extras

uv run python benchmarks/compare_native.py \
  --baseline-source /tmp/et-python-baseline/src \
  --native-source src \
  --data /tmp/et-benchmark-data \
  --output /tmp/et-benchmark.json \
  --rows 100000 --calls 20000 --repeats 5 --cpu 23

uv run python benchmarks/viewer.py \
  --data /tmp/et-benchmark-data \
  --output /tmp/et-viewer-benchmark.json \
  --repeats 5 --cpu 23
```

Choose a CPU present in `os.sched_getaffinity(0)`, rather than assuming CPU 23 is
available. Do not run the comparison alongside builds, tests or other benchmarks.
Raw JSON retains fixture checksums, native-binary hash, dependencies, individual
trials, quantiles, throughput, CPU time, RSS and variability. Verify workload
output rather than interpreting a faster but incomplete query as a speedup.
