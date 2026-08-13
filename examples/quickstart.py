"""The sixty-second tour: log a run, watch it, read it back.

Everything here is local. Add ``"wandb"`` to ``backends`` and the same calls
mirror to wandb as well.

    uv run python examples/quickstart.py
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import expr_tracker as et


def train(args) -> Path:
    run = et.init(
        project="quickstart",
        name=args.name,
        dir=args.dir,
        backends=[],  # local only; add "wandb" or "trackio" to mirror
        config={"lr": args.lr, "batch_size": 32, "seed": args.seed},
    )
    rng = random.Random(args.seed)
    best = 0.0

    for step in range(args.steps):
        loss = math.exp(-step / 40) + rng.uniform(0, 0.05)

        if step % 10 == 0:
            accuracy = 1 - loss / 2 + rng.uniform(-0.02, 0.02)
            # commit=False leaves the row open, so this joins the training step
            # below instead of starting a step of its own
            et.log({"eval/accuracy": accuracy}, commit=False)
            if accuracy > best:
                best = accuracy
                et.summary()["best_accuracy"] = best
                et.summary()["best_step"] = step

        et.log({"train/loss": loss, "train/lr": args.lr * (0.99**step)})

    print(f"run directory: {run.dir}")
    print(f"summary:       {dict(et.summary())}")

    # History is available while the run is open, not just afterwards
    recent = et.history(5, metrics=["train/loss"])
    print(f"last 5 losses: {[round(row['train/loss'], 4) for row in recent]}")

    # Sparse metrics: keep only the steps that actually have one
    evals = et.history(-1, metrics=["eval/accuracy"], dropna=True)
    print(f"{len(evals)} eval points across {args.steps} steps")

    et.finish()
    return Path(run.dir)


def read_back(run_dir: Path) -> None:
    """The same query works on a finished run, from any process."""
    rows = et.history(3, run=run_dir, metrics=["train/loss", "eval/accuracy"])
    print(f"\nreopened {run_dir.name}:")
    for row in rows:
        accuracy = row.get("eval/accuracy")
        shown = f"{accuracy:.4f}" if accuracy is not None else "-"
        print(f"  step {row['_step']:>3}  loss {row['train/loss']:.4f}  acc {shown}")
    print(f"\nInspect it with: et history {run_dir}")


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dir", default="runs")
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)
    # A fresh name each time: the local history always continues an existing run
    # directory, so a fixed name would keep counting up from the last step
    args.name = args.name or time.strftime("quickstart-%H%M%S")

    run_dir = train(args)
    read_back(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
