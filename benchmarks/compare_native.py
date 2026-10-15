"""Reproducible, fresh-process comparison of Python and native SDK implementations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from native_workloads import CASES, PAYLOAD, configure_source, worker


def prepare(options):
    from expr_tracker.history import HistoryStore

    root = Path(options.data)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "rows": options.rows,
        "patch_steps": min(options.rows, 20_000),
        "trace_rows": min(options.rows, 32_000),
    }
    for name, count in (
        ("ordered", metadata["rows"]),
        ("patched", metadata["patch_steps"]),
    ):
        store = HistoryStore()
        store.init(
            project="benchmark",
            run_dir=str(root / name),
            cache_bytes=1,
            buffer_size=1000,
            max_file_bytes=1 << 20,
        )
        try:
            for index in range(count):
                store.log({**PAYLOAD, "loss": index / count}, step=index, commit=True)
                if name == "patched":
                    store.log({"loss": index / count + 0.01}, step=index, commit=True)
            store.finish()
            assert store.stats()["rows_on_disk"] == count * (
                2 if name == "patched" else 1
            )
        finally:
            store.finish()
    with (root / "spans.jsonl").open("w", encoding="utf-8") as target:
        for index in reversed(range(metadata["trace_rows"])):
            target.write(
                json.dumps(
                    {
                        "name": "work",
                        "start": 1_700_000_000 + index,
                        "dur_ms": 1,
                        "depth": 0,
                        "_step": index,
                        "args": {"payload": "x" * 256},
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    block = bytes(range(256)) * 4096
    digest = hashlib.sha256()
    with (root / "checkpoint.bin").open("wb") as target:
        for _ in range(16):
            target.write(block)
            digest.update(block)
    metadata["checkpoint_digest"] = digest.hexdigest()
    tree = root / "bundle"
    tree.mkdir()
    for index in range(128):
        (tree / f"file-{index:03}.bin").write_bytes(block[: 16 << 10])
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as source:
                checksum = hashlib.file_digest(source, "sha256").hexdigest()
            files[str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": checksum,
            }
    metadata["files"] = files
    (root / "fixture.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


def run_child(options, source: str, extra: list[str]):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source",
        source,
        "--data",
        options.data,
        "--rows",
        str(options.rows),
        "--calls",
        str(options.calls),
        *extra,
    ]
    if options.cpu is not None:
        command += ["--cpu", str(options.cpu)]
    process = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=300
    )
    if process.returncode:
        raise RuntimeError(
            f"Benchmark worker failed ({process.returncode}): {' '.join(extra)}\n"
            f"{process.stderr}"
        )
    return json.loads(process.stdout)


def compare(options):
    root = Path(options.data)
    if not (root / "fixture.json").exists():
        run_child(options, options.baseline_source, ["--prepare"])
    fixture = json.loads((root / "fixture.json").read_text())
    binary = Path(options.native_source) / "expr_tracker" / "_native.abi3.so"
    with binary.open("rb") as source:
        binary_digest = hashlib.file_digest(source, "sha256").hexdigest()
    results = []
    report = {
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "pinned_cpu": options.cpu,
            "repeats": options.repeats,
            "clock": "perf_counter_ns / process_time_ns",
            "gc": "enabled; collected before each measured batch",
            "io": "warm OS page cache; no fsync; final buffer drain included",
            "memory": "fresh-process Linux RSS, including native allocations",
            "baseline_source": str(Path(options.baseline_source).resolve()),
            "native_source": str(Path(options.native_source).resolve()),
            "native_binary_sha256": binary_digest,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in (
                    "click",
                    "filelock",
                    "loguru",
                    "pydantic",
                    "python-dotenv",
                    "numpy",
                    "pandas",
                    "polars",
                )
            },
        },
        "fixture": fixture,
        "trials": results,
    }
    if options.baseline_native:
        baseline_binary = (
            Path(options.baseline_source) / "expr_tracker" / "_native.abi3.so"
        )
        with baseline_binary.open("rb") as source:
            report["environment"]["baseline_native_binary_sha256"] = (
                hashlib.file_digest(source, "sha256").hexdigest()
            )
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for case in options.cases:
        for repeat in range(options.repeats):
            order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
            for implementation in order:
                source = (
                    options.baseline_source
                    if implementation == "before"
                    else options.native_source
                )
                trial = run_child(options, source, ["--worker", case])
                assert trial["native"] == (
                    implementation == "after" or options.baseline_native
                )
                trial.update(implementation=implementation, repeat=repeat)
                results.append(trial)
                output.write_text(json.dumps(report, indent=2))
                print(
                    f"{case:30} {implementation:6} {repeat + 1}/{options.repeats} "
                    f"{trial['mean_us']:10.2f} us/op "
                    f"RSS {trial['peak_rss_bytes'] / 2**20:.1f} MiB",
                    flush=True,
                )
    summary = []
    for case in options.cases:
        row = {"case": case}
        for implementation in ("before", "after"):
            trials = [
                trial
                for trial in results
                if trial["case"] == case and trial["implementation"] == implementation
            ]
            row[implementation] = {
                key: statistics.median(trial[key] for trial in trials)
                for key in (
                    "mean_us",
                    "p50_us",
                    "p95_us",
                    "p99_us",
                    "calls_per_second",
                    "wall_seconds",
                    "cpu_seconds",
                    "import_rss_bytes",
                    "ready_rss_bytes",
                    "final_rss_bytes",
                    "peak_rss_bytes",
                )
            }
            means = [trial["mean_us"] for trial in trials]
            row[implementation]["variation_pct"] = (
                statistics.stdev(means) / statistics.fmean(means) * 100
                if len(means) > 1
                else 0
            )
        row["speedup"] = row["before"]["mean_us"] / row["after"]["mean_us"]
        row["latency_change_pct"] = (
            row["after"]["mean_us"] / row["before"]["mean_us"] - 1
        ) * 100
        summary.append(row)
    report["summary"] = summary
    output.write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source")
    parser.add_argument(
        "--baseline-native",
        action="store_true",
        help="Compare against an earlier native build instead of the Python baseline.",
    )
    parser.add_argument("--native-source", default="src")
    parser.add_argument("--source")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="benchmark-results.json")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--calls", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--prepare", action="store_true")
    options = parser.parse_args()
    if options.rows < 1000 or options.calls < 1000 or options.repeats < 1:
        parser.error("rows/calls must be >=1000 and repeats must be >=1")
    if options.worker or options.prepare:
        if not options.source:
            parser.error("--source is required for a worker")
        options.available_cpus = configure_source(options.source, options.cpu)
        if options.prepare:
            prepare(options)
        else:
            worker(options)
    else:
        if not options.baseline_source:
            parser.error("--baseline-source is required")
        compare(options)


if __name__ == "__main__":
    main()
