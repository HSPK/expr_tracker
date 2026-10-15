"""Attribute logging overhead; stage timings are not end-to-end SDK benchmarks."""

from __future__ import annotations

import argparse
import cProfile
import functools
import gc
import json
import os
import pstats
import statistics
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="src")
    parser.add_argument("--metrics", type=int, default=100)
    parser.add_argument("--calls", type=int, default=10_000)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    if options.metrics < 1 or options.calls < 1000:
        parser.error("metrics must be positive and calls must be >=1000")
    sys.path.insert(0, str(Path(options.source).resolve()))
    if options.cpu is not None:
        os.sched_setaffinity(0, {options.cpu})

    import numpy as np
    from expr_tracker._native import (
        NativeHistoryState,
        NativeMetricSeries,
        NativeSummary,
    )
    from loguru import logger

    import expr_tracker as et

    logger.remove()
    payload = {f"metric_{index}": 0.5 for index in range(options.metrics)}
    results = []
    for case in ("local", "backend", "numpy"):
        values = (
            {key: np.float64(value) for key, value in payload.items()}
            if case == "numpy"
            else payload
        )
        with tempfile.TemporaryDirectory(prefix="et-profile-") as directory:
            run = et.init(
                project="profile",
                run_dir=directory,
                backends=[],
                buffer_size=200,
                cache_bytes=64 << 20,
                max_file_bytes=None,
            )
            if case == "backend":

                class Backend:
                    def log(self, metrics, step=None):
                        pass

                    def finish(self):
                        pass

                run.backends["profile"] = Backend()
            try:
                for _ in range(100):
                    run.log(values)
                gc.collect()
                profile = cProfile.Profile()
                profile.enable()
                for _ in range(options.calls):
                    run.log(values)
                run.history.flush(commit_open=True)
                profile.disable()
                stats = pstats.Stats(profile)
                functions = []
                for (file, line, function), (
                    _,
                    calls,
                    own,
                    cumulative,
                    _,
                ) in stats.stats.items():
                    functions.append(
                        {
                            "file": Path(file).name,
                            "line": line,
                            "function": function,
                            "calls": calls,
                            "own_seconds": own,
                            "cumulative_seconds": cumulative,
                        }
                    )
                functions.sort(key=lambda row: row["own_seconds"], reverse=True)
                assert run.history.stats()["rows_on_disk"] == options.calls + 100
                results.append(
                    {
                        "case": case,
                        "calls": options.calls,
                        "profile_seconds": stats.total_tt,
                        "native_calls_per_log": sum(
                            row["calls"]
                            for row in functions
                            if "Native" in row["function"]
                        )
                        / options.calls,
                        "functions": functions,
                    }
                )
            finally:
                et.finish()

    stages = {}
    for name in ("dict_snapshot", "python_json", "native_ingest", "native_publish"):
        timings = []
        for _ in range(5):
            state = NativeHistoryState(64 << 20, 100_000, False)
            series = NativeMetricSeries()
            summary = NativeSummary()
            record = state.ingest_values(payload, None, True, False, {}, 1.0, 1.0)[1]
            if name == "dict_snapshot":
                call = payload.copy
            elif name == "python_json":
                call = functools.partial(json.dumps, payload, ensure_ascii=False)
            elif name == "native_ingest":
                call = functools.partial(
                    state.ingest_values, payload, None, True, False, {}, 1.0, 1.0
                )
            else:
                call = functools.partial(state.publish_local, record, series, summary)

            for _ in range(100):
                call()
            gc.collect()
            start = time.perf_counter_ns()
            for _ in range(options.calls):
                call()
            timings.append((time.perf_counter_ns() - start) / options.calls / 1000)
        stages[name] = {
            "trials_us": timings,
            "median_us": statistics.median(timings),
        }
    report = {
        "metrics": options.metrics,
        "cpu": options.cpu,
        "calls": options.calls,
        "profile_caveat": "Instrumented timings are attribution, not SDK latency.",
        "stage_caveat": (
            "Isolated operations, not additive pipeline timings; fixed timestamp "
            "and scalar objects; no writer in native_ingest/native_publish."
        ),
        "profiles": results,
        "stages": stages,
    }
    Path(options.output).write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        print(
            result["case"],
            "native calls/log:",
            round(result["native_calls_per_log"], 2),
        )
        for row in result["functions"][:12]:
            print(
                f"  {row['own_seconds']:.3f}s own / "
                f"{row['cumulative_seconds']:.3f}s cumulative "
                f"{row['function']} ({row['calls']} calls)"
            )
    print(
        "Isolated stages, us:",
        {name: round(row["median_us"], 3) for name, row in stages.items()},
    )


if __name__ == "__main__":
    main()
