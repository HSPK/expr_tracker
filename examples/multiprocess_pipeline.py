"""Four data producers feeding four trainers, with bounded staleness.

Eight processes share one run. Each writes its own stream, so the producers'
batch counter and the trainers' step counter never collide, and each becomes its
own process lane in the exported trace.

The queue is the backpressure: it holds at most ``--staleness`` batches per
trainer, so a producer can run at most that many steps ahead. Whichever side is
slower shows up as a wide blocking span --- ``enqueue`` when the producers are
held back, ``wait_for_batch`` when the trainers are starved.

    # producers faster than trainers: they stall on a full queue
    uv run python examples/multiprocess_pipeline.py --produce-ms 10 --train-ms 40

    # trainers faster than producers: they starve waiting for batches
    uv run python examples/multiprocess_pipeline.py --produce-ms 40 --train-ms 10

Open the trace it writes at https://ui.perfetto.dev.
"""

from __future__ import annotations

import argparse
import contextlib
import multiprocessing as mp
import queue as queuelib
import random
import time
from pathlib import Path

import expr_tracker as et
from expr_tracker.history.naming import parse_stream
from expr_tracker.trace import read_spans, span_files, write_trace

PROJECT = "pipeline"
BLOCK_TIMEOUT = 60.0  # waiting this long means the other side is gone


def spend(milliseconds: float, jitter: float, rng: random.Random) -> None:
    """Burn wall time the way a real stage would, give or take some jitter."""
    seconds = milliseconds / 1000.0
    if jitter:
        seconds *= 1.0 + rng.uniform(-jitter, jitter)
    if seconds > 0:
        time.sleep(seconds)


def produce(worker: int, args, batches, claimed) -> None:
    """Fill the queue until every batch the trainers need has been claimed."""
    rng = random.Random(1000 + worker)
    et.init(
        project=PROJECT,
        name=args.name,
        dir=args.dir,
        backends=[],
        stream=f"producer{worker}",
    )
    wanted = args.steps * args.trainers
    made = 0
    while True:
        with claimed.get_lock():  # one shared counter, so no batch is made twice
            index = claimed.value
            if index >= wanted:
                break
            claimed.value = index + 1

        with et.span("produce", batch=index):
            with et.span("read"):
                spend(args.produce_ms * 0.6, args.jitter, rng)
            with et.span("decode"):
                spend(args.produce_ms * 0.4, args.jitter, rng)
            # Blocks once the queue is full: this span *is* the backpressure
            with et.span("enqueue"):
                batches.put((index, worker, time.time()), timeout=BLOCK_TIMEOUT)
        made += 1
        et.log({"produce/batch": index, "produce/made": made})
    et.finish()


def train(worker: int, args, batches) -> None:
    """Consume exactly ``--steps`` batches, whoever produced them."""
    rng = random.Random(2000 + worker)
    et.init(
        project=PROJECT,
        name=args.name,
        dir=args.dir,
        backends=[],
        stream=f"trainer{worker}",
    )
    for step in range(args.steps):
        with et.span("step"):
            # Blocks while the queue is empty: this span *is* the starvation
            with et.span("wait_for_batch"):
                index, source, queued_at = batches.get(timeout=BLOCK_TIMEOUT)
            age_ms = (time.time() - queued_at) * 1000.0
            with et.span("forward"):
                spend(args.train_ms * 0.4, args.jitter, rng)
            with et.span("backward"):
                spend(args.train_ms * 0.6, args.jitter, rng)
        et.log(
            {
                "train/loss": 1.0 / (step + 1),
                "train/batch": index,
                "train/from_producer": source,
                "train/queue_age_ms": age_ms,
            }
        )
    et.finish()


def summarise(run_dir: Path) -> None:
    """Where each side's time actually went, straight out of the span files."""
    blocking = {"enqueue": "<- backpressure", "wait_for_batch": "<- starvation"}
    totals: dict[str, dict[str, float]] = {}
    for path in span_files(run_dir):
        stream = parse_stream(path.name) or "default"
        side = "producers" if stream.startswith("producer") else "trainers"
        bucket = totals.setdefault(side, {})
        for record in read_spans(path):
            leaf = record["name"].rsplit("/", 1)[-1]
            bucket[leaf] = bucket.get(leaf, 0.0) + float(record["dur_ms"])

    print("\nWhere the time went, summed over all workers:")
    for side, root in (("producers", "produce"), ("trainers", "step")):
        stages = totals.get(side, {})
        whole = stages.get(root, 0.0)
        print(f"\n  {side}  {whole / 1000:.2f}s in {root}")
        for name, spent in sorted(stages.items(), key=lambda kv: -kv[1]):
            if name == root:
                continue
            share = 100 * spent / whole if whole else 0.0
            note = blocking.get(name, "")
            print(f"    {name:<15} {spent / 1000:6.2f}s {share:5.1f}%  {note}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--producers", type=int, default=4)
    parser.add_argument("--trainers", type=int, default=4)
    parser.add_argument(
        "--staleness",
        type=int,
        default=3,
        help="batches a producer may run ahead, per trainer (default: 3)",
    )
    parser.add_argument("--steps", type=int, default=20, help="steps per trainer")
    parser.add_argument("--produce-ms", type=float, default=40.0)
    parser.add_argument("--train-ms", type=float, default=20.0)
    parser.add_argument("--jitter", type=float, default=0.25, help="0 for a metronome")
    parser.add_argument("--dir", default="runs")
    parser.add_argument("--name", default=None)
    parser.add_argument("--trace", default=None, help="where to write the trace")
    args = parser.parse_args(argv)
    if args.name is None:
        args.name = time.strftime("pipeline-%H%M%S")
    return args


def main(argv=None) -> Path:
    args = parse_args(argv)
    ctx = mp.get_context("spawn")
    batches = ctx.Queue(maxsize=args.staleness * args.trainers)
    claimed = ctx.Value("i", 0)

    workers = [
        ctx.Process(target=produce, args=(i, args, batches, claimed), daemon=True)
        for i in range(args.producers)
    ] + [
        ctx.Process(target=train, args=(i, args, batches), daemon=True)
        for i in range(args.trainers)
    ]

    print(
        f"{args.producers} producers @ {args.produce_ms:g}ms, "
        f"{args.trainers} trainers @ {args.train_ms:g}ms, "
        f"queue holds {args.staleness * args.trainers} batches "
        f"({args.staleness} per trainer)"
    )
    started = time.time()
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    elapsed = time.time() - started

    failed = {w.name: w.exitcode for w in workers if w.exitcode}
    if failed:
        raise SystemExit(f"workers failed: {failed}")

    with contextlib.suppress(queuelib.Empty, OSError, ValueError):
        while True:  # let the queue's feeder thread shut down cleanly
            batches.get_nowait()

    run_dir = Path(args.dir) / PROJECT / args.name
    output = Path(args.trace) if args.trace else run_dir / "trace.json"
    spans = write_trace(run_dir, output)
    summarise(run_dir)
    print(f"\n{args.steps * args.trainers} batches in {elapsed:.2f}s")
    print(f"{spans} spans -> {output}")
    print("Open it at https://ui.perfetto.dev")
    return output


if __name__ == "__main__":
    main()
