# Native development and distribution

The Rust crate in `Cargo.toml` builds `expr_tracker._native` through PyO3 and
maturin. Python remains the public training SDK, arbitrary-object normalization
boundary, and adapter for Python plugins, callbacks, remote tracking SDKs, and
pandas/polars output. The local HTTP viewer uses the same segmented native storage
reader as the SDK; it does not invoke a Python HTTP server or require RVX.

The viewer's TypeScript/Vite/uPlot workspace follows RVX's charts-first direction
but reads this project's step-oriented JSONL model, not RVX snapshot/Source APIs.
There is no dependency on the sibling checkout.

## Build and check

```bash
npm --prefix web ci
npm --prefix web run check
uv sync --locked --all-extras
cargo test --locked --lib
uv run ruff check src tests benchmarks
uv run pytest
uv build
```

Rebuild the editable native extension with `uv sync --locked --all-extras` after
Rust changes. Native
and Python sources participate in uv's local build cache keys. A source build
needs a Rust toolchain. Rebuild Web assets before building a distribution when
frontend sources change.

### Exact artifact timestamp tests

The artifact metadata test checks permissions and exact nanosecond modification
and access times for both publication and download. Ordinary reads can change
access time on `relatime`/`strictatime` filesystems, including reads outside the SDK.
Its fixture first probes actual read behavior. If needed on Linux, it sets
`FS_NOATIME_FL` on only its private pytest tree, preserving existing inode flags,
then verifies that new nested files inherit read-stable access times. This needs
no privileged mount or extra dependency on supported filesystems such as ext4.
Explicit `os.utime`/native timestamp updates still work; the assertion is not
relaxed and now also checks the originally supplied access-time nanoseconds.

If inode flags are unsupported or forbidden, use a dedicated filesystem mounted
with `noatime` and select a disposable subdirectory with
`pytest --basetemp=/isolated-test-fs/pytest`. The fixture reports a setup error if
reads still change access time; it does not skip/retry the test or use future
timestamps. These settings apply to tests only, not experiment logs.

The release profile enables portable ThinLTO with one codegen unit. Initial
release rebuilds take longer; do not use `target-cpu=native` for distributed
wheels. Benchmark the installed release extension, never a debug build.

The extension targets the stable CPython ABI starting at Python 3.10. This reduces
the number of Python-version-specific wheels, but does not make a wheel portable
across operating systems or CPU architectures.

## Release contract

Unlike the previous pure-Python/tag-derived build, maturin reads the package
version from `pyproject.toml` and `Cargo.toml`. Update both versions together before
the next release. A pushed release tag must exactly match the package version.
Do not reuse the already published `v0.4.1` tag for the native distribution.

CI checks Python 3.10–3.13, native tests, Python lint/tests, and Web tests/build.
Release builds package Web assets and native wheels for Linux x86_64/aarch64,
macOS x86_64/aarch64, and Windows x86_64, plus a source distribution. Linux wheels
target manylinux 2.28. Every platform must build and pass an installed-wheel
smoke check before publishing. Runtime dependencies and the viewer assets must
be present when testing outside the source checkout.

The existing JSONL filenames, retained rotation directories, and schema-2 sidecar
contract remain compatibility surfaces. Migration tests must include ordinary
values, missing metrics, non-finite numbers, arbitrary-sized integers, filesystem
surrogates, patch histories, resume, write failures, and lifecycle/concurrency.
The browser data boundary is strict JSON and may tag values that JavaScript
cannot represent exactly; this must not silently rewrite the stored experiment.

The [native migration benchmark](benchmarks/native-migration.md) documents the
paired Python/Rust workloads and reproduction commands. Measure release builds,
check output correctness, and report regressions as well as speedups.
