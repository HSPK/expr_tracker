"""Where a training step actually spends its time.

Spans nest, and every duration becomes a metric on the step's row, so the same
alerting and history queries work on timings as on losses. Plugins attach CPU
and GPU cost to a region, and the whole tree exports to a Chrome Trace.

    uv run python examples/profile_step.py
    uv run python examples/profile_step.py --print-spans
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import expr_tracker as et
from expr_tracker.plugins import CpuTime
from expr_tracker.trace import write_trace


def busy(milliseconds: float) -> None:
    """Spin, so CpuTime has something real to measure."""
    deadline = time.perf_counter() + milliseconds / 1000.0
    while time.perf_counter() < deadline:
        pass


@et.span("data")  # a decorator works as well as a context manager
def load_batch(args) -> None:
    with et.span("read"):
        time.sleep(args.io_ms / 1000.0)  # waiting, not working
    with et.span("collate"):
        busy(args.io_ms / 4)


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--io-ms", type=float, default=8.0)
    parser.add_argument("--compute-ms", type=float, default=12.0)
    parser.add_argument("--print-spans", action="store_true", help="print the tree")
    parser.add_argument("--dir", default="runs")
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)
    args.name = args.name or time.strftime("profile-%H%M%S")

    run = et.init(
        project="profile",
        name=args.name,
        dir=args.dir,
        backends=[],
        # Defaults for every span in this run; a span can still override them
        span_print_fn=print if args.print_spans else None,
        span_plugins=[CpuTime()],
    )

    for step in range(args.steps):
        with et.span("step"):
            load_batch(args)
            with et.span("forward"):
                busy(args.compute_ms * 0.4)
            with et.span("backward"):
                busy(args.compute_ms * 0.6)
        et.log({"train/loss": 1.0 / (step + 1)})

    # Timings are ordinary metrics, so they query like any other
    rows = et.history(-1, metrics=["step/duration_ms", "step/data/read/duration_ms"])
    total = sum(row["step/duration_ms"] for row in rows) / len(rows)
    waiting = sum(row["step/data/read/duration_ms"] for row in rows) / len(rows)
    print(f"\nmean step {total:.2f}ms, of which {waiting:.2f}ms waiting on IO")
    print(f"the data loader is {100 * waiting / total:.0f}% of the step")

    last = et.history(1)[0]
    print("\nwhat one step recorded:")
    for key in sorted(k for k in last if k.endswith(("duration_ms", "cpu_percent"))):
        print(f"  {key:<38} {last[key]:.2f}")

    et.finish()
    output = Path(run.dir) / "trace.json"
    spans = write_trace(run.dir, output)
    print(f"\n{spans} spans -> {output}")
    print("Open it at https://ui.perfetto.dev")
    print("Timings are metrics, so they alert too:")
    print(
        '  et.init(..., alert_rules=["step/duration_ms > 500 => warning: slow step"])'
    )
    return output


if __name__ == "__main__":
    main()
