"""Measure real native HTTP indexing, chart queries and incremental refresh."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


def process_memory(pid: int) -> dict[str, int]:
    fields = {}
    with open(f"/proc/{pid}/status", encoding="ascii") as source:
        for line in source:
            key = line.split(":", 1)[0]
            if key in ("VmRSS", "VmHWM"):
                fields[key] = int(line.split()[1]) * 1024
    return fields


def request(base: str, endpoint: str, **parameters):
    started = time.perf_counter_ns()
    url = base + endpoint
    if parameters:
        url += "?" + urlencode(parameters)
    with urlopen(url, timeout=120) as response:
        value = json.load(response)
    return value, (time.perf_counter_ns() - started) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--source", default="src")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu", type=int)
    options = parser.parse_args()
    if options.repeats < 1:
        parser.error("repeats must be >=1")
    if options.cpu is not None:
        os.sched_setaffinity(0, {options.cpu})
    fixture = json.loads((Path(options.data) / "fixture.json").read_text())
    rows = fixture["rows"]
    trials = []
    for repeat in range(options.repeats):
        with tempfile.TemporaryDirectory(prefix="viewer-", dir=options.data) as scratch:
            source = Path(scratch) / "run"
            shutil.copytree(Path(options.data) / "ordered", source)
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            environment = {
                **os.environ,
                "PYTHONPATH": str(Path(options.source).resolve()),
            }
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from expr_tracker.cli import main; main()",
                    "serve",
                    "--root",
                    str(source),
                    "--port",
                    str(port),
                ],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            base = f"http://127.0.0.1:{port}"
            try:
                deadline = time.monotonic() + 20
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(server.stderr.read())
                    try:
                        runs, discovery = request(base, "/api/runs")
                        break
                    except URLError:
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "Viewer did not become responsive"
                            ) from None
                        time.sleep(0.05)
                assert len(runs) == 1
                run = runs[0]["id"]
                ready = process_memory(server.pid)
                detail, cold = request(base, "/api/run", run=run)
                assert detail["rows"] == rows
                chart_times, range_times, record_times = [], [], []
                for _ in range(10):
                    chart, elapsed = request(
                        base,
                        "/api/history",
                        run=run,
                        metrics=json.dumps(["loss", "acc"]),
                        max_points=1000,
                    )
                    assert chart["total"] == rows
                    assert 0 < len(chart["x"]) <= 1000
                    assert chart["x"][0] == 0 and chart["x"][-1] == rows - 1
                    chart_times.append(elapsed)
                    ranged, elapsed = request(
                        base,
                        "/api/history",
                        run=run,
                        metrics=json.dumps(["loss"]),
                        max_points=1000,
                        from_step=rows // 2,
                        to_step=rows // 2 + 1000,
                    )
                    assert ranged["total"] == 1000
                    range_times.append(elapsed)
                    records, elapsed = request(base, "/api/records", run=run, limit=100)
                    assert len(records["records"]) == 100
                    record_times.append(elapsed)
                parts = source / "metrics.jsonl.parts"
                parts.mkdir(exist_ok=True)
                number = (
                    max((int(path.stem) for path in parts.glob("*.jsonl")), default=0)
                    + 1
                )
                with (parts / f"{number:08}.jsonl").open("w") as target:
                    for step in range(rows, rows + 100):
                        target.write(
                            json.dumps(
                                {
                                    "_step": step,
                                    "_time": 1_700_000_000 + step,
                                    "loss": 0.5,
                                    "acc": 0.9,
                                }
                            )
                            + "\n"
                        )
                refreshed, incremental = request(base, "/api/run", run=run)
                assert refreshed["rows"] == rows + 100
                result = {
                    "repeat": repeat,
                    "rows": rows,
                    "discovery_ms": discovery,
                    "cold_index_ms": cold,
                    "chart_p50_ms": statistics.median(chart_times),
                    "chart_p95_ms": sorted(chart_times)[9],
                    "range1000_p50_ms": statistics.median(range_times),
                    "records100_p50_ms": statistics.median(record_times),
                    "increment100_ms": incremental,
                    "ready_memory": ready,
                    "indexed_memory": process_memory(server.pid),
                }
            finally:
                if server.poll() is None:
                    server.send_signal(signal.SIGINT)
                    try:
                        server.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.communicate()
                        raise RuntimeError(
                            "Native server failed graceful shutdown"
                        ) from None
            trials.append(result)
            Path(options.output).write_text(json.dumps(trials, indent=2))
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
