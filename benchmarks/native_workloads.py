"""Fresh-process benchmark workloads and their physical-output oracles."""

from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import json
import math
import os
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path

PAYLOAD = {
    "loss": 0.5,
    "acc": 0.9,
    "lr": 1e-4,
    "grad_norm": 1.2,
    "step_time": 0.03,
}
RULES = [
    "diff(loss) > 50 => warning: spike",
    "zscore(loss[50]) > 4 => error: z",
    "mean(loss[20]) > 100 => warning: mean",
    "stalled(loss[100]) => warning: stalled",
    "isnan(loss) or isinf(loss) => critical: nan",
    "slope(loss[30]) > 1 => warning: slope",
    "acc < 0.1 => warning: acc",
    "lr < 1e-8 => warning: lr",
    "grad_norm > 1e4 => error: grad",
    "count(loss[10]) < 1 => warning: gap",
]
CASES = (
    "store_buffered",
    "store_concurrent_readers4",
    "run_buffered",
    "run_immediate",
    "run_eviction",
    "run_rotation",
    "rotation_default_100mib",
    "run_rules_10",
    "run_metrics_50",
    "run_metrics_100",
    "run_metrics_100_default",
    "run_metrics_100_immediate",
    "run_metrics_100_realistic",
    "run_metrics_100_changing",
    "run_metrics_100_numpy",
    "run_metrics_100_backend",
    "run_metrics_100_rules10",
    "run_metrics_500",
    "run_metrics_2000",
    "span_sync",
    "span_async",
    "span_metrics",
    "cache_tail_100",
    "cache_range_100",
    "disk_tail_100",
    "disk_range_100",
    "disk_full",
    "patch_tail_100",
    "alert_parse",
    "alert_evaluate",
    "artifact_digest_16mib",
    "artifact_publish_128files",
    "artifact_dedup_128files",
    "trace_filtered_100",
    "trace_export",
    "memory_cache_4mib",
)


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))] / 1e3


def rss_bytes() -> int:
    with open("/proc/self/status", encoding="ascii") as source:
        for line in source:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("This RSS comparison requires Linux /proc")


def configure_source(source: str, cpu: int | None):
    available = sorted(os.sched_getaffinity(0))
    sys.path.insert(0, str(Path(source).resolve()))
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    from loguru import logger

    logger.remove()
    return available


def worker(options):
    import expr_tracker as et
    from expr_tracker.history import HistoryStore

    root = Path(options.data)
    fixture = json.loads((root / "fixture.json").read_text())
    case = options.worker
    module = str(Path(et.__file__).resolve())
    native = importlib.util.find_spec("expr_tracker._native") is not None
    imported_rss = rss_bytes()
    count = options.calls
    warmup = 100

    def cleanup():
        return None

    def drain():
        return None

    def validate(result):
        return None

    def begin():
        return None

    details = {}
    with tempfile.TemporaryDirectory(prefix="trial-", dir=root) as directory:
        if case.startswith(("store_", "run_", "memory_", "rotation_")):
            arguments = {
                "project": "benchmark",
                "run_dir": directory,
                "buffer_size": 200,
                "cache_bytes": 64 << 20,
                "max_file_bytes": None,
            }
            if case in ("run_immediate", "run_metrics_100_immediate"):
                arguments["buffer_size"] = 1
                count = min(count, 3000)
            if case == "run_metrics_100_default":
                for key in ("buffer_size", "cache_bytes", "max_file_bytes"):
                    arguments.pop(key)
            if case == "run_eviction":
                arguments["cache_bytes"] = 64 << 10
            if case == "run_rotation":
                arguments["max_file_bytes"] = 64 << 10
            if case == "memory_cache_4mib":
                arguments["cache_bytes"] = 4 << 20
                count = max(fixture["rows"], 50_000)
            payload = PAYLOAD
            payload_ring = None
            hundred_metrics = case.startswith("run_metrics_100")
            if case.startswith("run_metrics_"):
                width = 100 if hundred_metrics else int(case.rsplit("_", 1)[1])
                payload = {f"metric_{index}": 0.5 for index in range(width)}
                if not hundred_metrics:
                    count = min(count, max(200, 50_000 // width))
                if case == "run_metrics_100_realistic":
                    payload = {
                        f"{'train' if index % 2 else 'validation'}/metric_{index:03}": (
                            index + 1
                        )
                        * 0.123456789012345
                        for index in range(width)
                    }
                elif case == "run_metrics_100_rules10":
                    payload = {
                        **PAYLOAD,
                        **{
                            f"metric_{index}": 0.5
                            for index in range(width - len(PAYLOAD))
                        },
                    }
                elif case == "run_metrics_100_changing":
                    payload_ring = [
                        {
                            key: (sample + 1) * (index + 1) * 0.000123456789123
                            for index, key in enumerate(payload)
                        }
                        for sample in range(1024)
                    ]
                elif case == "run_metrics_100_numpy":
                    import numpy as np

                    payload = {key: np.float64(value) for key, value in payload.items()}
            if case == "rotation_default_100mib":
                arguments.pop("max_file_bytes")
                arguments["cache_bytes"] = 1
                arguments["buffer_size"] = 16
                payload = {"loss": 0.5, "payload": "x" * (512 << 10)}
                count, warmup = 256, 0
            if case in ("store_buffered", "store_concurrent_readers4"):
                store = HistoryStore()
                store.init(**arguments)
                log = store.log
                cleanup = store.finish
            else:
                run = et.init(
                    **arguments,
                    backends=[],
                    alert_rules=(
                        RULES
                        if case in ("run_rules_10", "run_metrics_100_rules10")
                        else ()
                    ),
                )
                store, log, cleanup = run.history, run.log, et.finish
                if case == "run_metrics_100_backend":

                    class CountingBackend:
                        def __init__(self):
                            self.calls = 0

                        def log(self, metrics, step=None):
                            self.calls += 1

                        def finish(self):
                            pass

                    backend = CountingBackend()
                    run.backends["benchmark"] = backend

            def call(index):
                values = (
                    payload
                    if payload_ring is None
                    else payload_ring[index % len(payload_ring)]
                )
                return log(dict(values))

            def drain():
                return store.flush(commit_open=True)

            if case == "store_concurrent_readers4":
                stop = threading.Event()
                errors = []
                read_counts = [0] * 4
                reader_cpus = []
                available = [
                    cpu for cpu in options.available_cpus if cpu != options.cpu
                ]

                def read(index):
                    try:
                        if options.cpu is not None and available:
                            cpu = available[-1 - index % min(3, len(available))]
                            os.sched_setaffinity(threading.get_native_id(), {cpu})
                            reader_cpus.append(cpu)
                        while not stop.is_set():
                            records = store.get(100)
                            assert len(records) == 100
                            read_counts[index] += 1
                    except Exception as error:
                        errors.append(error)
                        stop.set()

                readers = [
                    threading.Thread(target=read, args=(index,)) for index in range(4)
                ]

                def begin():
                    stop.clear()
                    for reader in readers:
                        reader.start()

                original_drain = drain

                def drain():
                    original_drain()
                    stop.set()
                    for reader in readers:
                        if reader.ident is not None:
                            reader.join(timeout=30)
                            if reader.is_alive():
                                raise RuntimeError("Concurrent reader did not finish")
                    if errors:
                        raise RuntimeError("Concurrent read failed") from errors[0]
                    details["reader_queries"] = sum(read_counts)
                    details["reader_cpus"] = reader_cpus
                    if all(reader.ident is not None for reader in readers):
                        assert sum(read_counts) > 0

            def validate(result):
                stats = store.stats()
                assert stats["rows_on_disk"] == count + warmup
                details.update(stats)
                last = store.get(1)[0]
                expected = (
                    payload
                    if payload_ring is None
                    else payload_ring[(count - 1) % len(payload_ring)]
                )
                assert all(last[key] == value for key, value in expected.items())
                details["validated_metric_fields"] = len(payload)
                details["series_window"] = store.series.window
                details["writer_buffer_size"] = store.writer.buffer_size
                if hundred_metrics:
                    with (Path(directory) / "metrics.jsonl").open(
                        encoding="utf-8"
                    ) as source:
                        validated = 0
                        for row_index, line in enumerate(source):
                            row = json.loads(line)
                            index = row_index - warmup
                            values = (
                                payload
                                if payload_ring is None
                                else payload_ring[index % len(payload_ring)]
                            )
                            assert row["_step"] == row_index
                            assert len(row) == len(values) + 2
                            assert all(
                                row[key] == value for key, value in values.items()
                            ), f"{case}: invalid metrics at row {row_index}"
                            validated += 1
                    assert validated == count + warmup
                    details["validated_rows"] = validated
                    assert all(
                        run.summary[key] == value for key, value in expected.items()
                    )
                    assert all(
                        store.series.latest(key)
                        == (last["_step"], last["_time"], float(value))
                        for key, value in expected.items()
                    )
                if case == "run_metrics_100_backend":
                    assert backend.calls == count + warmup
                    details["backend_calls"] = backend.calls
                if case in ("run_eviction", "memory_cache_4mib"):
                    assert stats["evicted_rows"] > 0
                if case.startswith(("run_rotation", "rotation_default")):
                    segments = [Path(directory) / "metrics.jsonl"]
                    segments += sorted(
                        (Path(directory) / "metrics.jsonl.parts").glob("*.jsonl")
                    )
                    limit = 100 << 20 if case == "rotation_default_100mib" else 64 << 10
                    assert len(segments) > 1
                    assert all(path.stat().st_size <= limit for path in segments)
                    sizes = [path.stat().st_size for path in segments]
                    details["segment_bytes"] = sizes
                    details["total_file_bytes"] = sum(sizes)
                    if case == "rotation_default_100mib":
                        assert sum(sizes) > 100 << 20

        elif case.startswith("span_"):
            run = et.init(
                project="benchmark",
                run_dir=directory,
                backends=[],
                span_async_write=case == "span_async",
                span_metrics=case == "span_metrics",
                max_file_bytes=None,
            )
            cleanup, drain = et.finish, run.history.flush
            count = min(count, 10_000)

            def call(index):
                with et.span("work"):
                    pass

            def validate(result):
                assert run.history.span_writer.lines == count + warmup
                assert run.step == 0
                details["spans"] = run.history.span_writer.lines

        elif case.startswith("cache_") or case == "alert_evaluate":
            store = HistoryStore()
            store.init(
                project="benchmark",
                run_dir=directory,
                cache_bytes=64 << 20,
                buffer_size=1000,
                max_file_bytes=None,
            )
            cleanup = store.finish
            rows = min(fixture["rows"], 20_000)
            for _ in range(rows):
                store.log(dict(PAYLOAD))
            store.flush(commit_open=True)
            assert store.stats()["evicted_rows"] == 0
            if case == "alert_evaluate":
                from expr_tracker.alerts.expr import EvalContext, evaluate, parse

                node = parse("mean(loss[50]) > 0.4 and slope(loss[30]) > 1")
                context = EvalContext(
                    store.series,
                    step=rows,
                    started_at=store.started_at,
                    record=PAYLOAD,
                )

                def call(index):
                    return evaluate(node, context)

                def validate(result):
                    assert result is False

            else:
                count = 500
                if case == "cache_tail_100":

                    def call(index):
                        return store.get(100)

                    first = rows - 100
                else:
                    first = rows // 2

                    def call(index):
                        return store.get(-1, step_range=(first, first + 100))

                def validate(result):
                    assert len(result) == 100
                    assert result[0]["_step"] == first
                    assert store.stats()["disk_queries"] == 0

        elif case.startswith(("disk_", "patch_")):
            patched = case == "patch_tail_100"
            source = root / ("patched" if patched else "ordered")
            rows = fixture["patch_steps"] if patched else fixture["rows"]
            warmup, count = 1, 30
            if case == "disk_full":
                count = 3

                def call(index):
                    return et.history(-1, run=str(source))

                expected, first = rows, 0
            elif case == "disk_range_100":
                first, expected = rows // 2, 100

                def call(index):
                    return et.history(
                        -1, run=str(source), step_range=(first, first + 100)
                    )
            else:
                count = 10 if patched else count
                first, expected = rows - 100, 100

                def call(index):
                    return et.history(100, run=str(source))

            def validate(result):
                assert len(result) == expected
                assert result[0]["_step"] == first
                assert result[-1]["_step"] == first + expected - 1
                if patched:
                    assert math.isclose(result[-1]["loss"], (rows - 1) / rows + 0.01)

        elif case == "alert_parse":
            from expr_tracker.alerts.expr import parse

            count = 2000
            expression = "mean(loss[50]) > 0.4 and slope(loss[30]) > 1"

            def call(index):
                return parse(expression)

            def validate(result):
                assert result is not None

        elif case.startswith("artifact_"):
            from expr_tracker.artifacts import Artifact, file_digest

            warmup = 1
            if case == "artifact_digest_16mib":
                count = 8

                def call(index):
                    return file_digest(root / "checkpoint.bin")

                def validate(result):
                    assert result.endswith(fixture["checkpoint_digest"])

            else:
                run = et.init(project="benchmark", run_dir=directory, backends=[])
                cleanup = et.finish
                dedup = case == "artifact_dedup_128files"
                count = 10 if dedup else 3

                def call(index):
                    name = "bundle" if dedup else f"bundle-{index}"
                    artifact = Artifact(name, type="checkpoint")
                    artifact.add_dir(root / "bundle")
                    return run.log_artifact(artifact, aliases=["best"])

                def validate(result):
                    assert result.version == 0
                    assert len(result.files()) == 128
                    assert (result.dir / "file-000.bin").stat().st_size == 16 << 10

        elif case.startswith("trace_"):
            from expr_tracker.trace import read_spans, write_trace

            count, warmup = 3, 1
            source = root / "spans.jsonl"
            rows = fixture["trace_rows"]
            if case == "trace_filtered_100":

                def call(index):
                    return read_spans(
                        source, time_range=(1_700_000_000 + rows - 100, None)
                    )

                def validate(result):
                    assert len(result) == 100

            else:
                output = Path(directory) / "trace.json"

                def call(index):
                    return write_trace(source, output)

                def validate(result):
                    assert result == rows
                    events = json.loads(output.read_text())["traceEvents"]
                    assert sum(event.get("ph") == "X" for event in events) == rows

        else:
            raise ValueError(f"Unknown benchmark: {case}")

        for index in range(-warmup, 0):
            call(index)
        drain()
        gc.collect()
        ready_rss = rss_bytes()
        samples = []
        begin()
        cpu_started = time.process_time_ns()
        started = time.perf_counter_ns()
        for index in range(count):
            tick = time.perf_counter_ns()
            result = call(index)
            samples.append(time.perf_counter_ns() - tick)
            if (
                case == "store_concurrent_readers4"
                and time.perf_counter_ns() - started >= 10_000_000_000
            ):
                break
        count = len(samples)
        drain()
        wall = (time.perf_counter_ns() - started) / 1e9
        cpu = (time.process_time_ns() - cpu_started) / 1e9
        final_rss = rss_bytes()
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        validate(result)
        cleanup()
    print(
        json.dumps(
            {
                "case": case,
                "native": native,
                "module": module,
                "count": count,
                "wall_seconds": wall,
                "cpu_seconds": cpu,
                "mean_us": wall / count * 1e6,
                "p50_us": percentile(samples, 0.50),
                "p95_us": percentile(samples, 0.95),
                "p99_us": percentile(samples, 0.99),
                "calls_per_second": count / wall,
                "import_rss_bytes": imported_rss,
                "ready_rss_bytes": ready_rss,
                "final_rss_bytes": final_rss,
                "peak_rss_bytes": peak_rss,
                "details": details,
            }
        )
    )
