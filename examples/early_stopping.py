"""Reading your own history mid-run to decide what to do next.

Most trackers only ship metrics out. Because history is local and queryable
while the run is open, the loop can ask what has been happening and act on it:
decay the learning rate on a plateau, stop early, skip a bad batch.

    uv run python examples/early_stopping.py
    uv run python examples/early_stopping.py --patience 3 --plateau 0.002
"""

from __future__ import annotations

import argparse
import math
import random
import time

import expr_tracker as et


def plateaued(window: int, threshold: float) -> bool:
    """Has the eval metric stopped improving over the last `window` evals?

    `n` counts rows and `dropna` filters them afterwards, so asking for the last
    4 rows of a metric logged every 10 steps finds nothing. Take the eval points
    first, then the last few.
    """
    points = et.history(-1, metrics=["eval/loss"], dropna=True)[-window:]
    if len(points) < window:
        return False  # not enough evidence yet, so do nothing
    values = [row["eval/loss"] for row in points]
    return max(values) - min(values) < threshold


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4, help="evals before acting")
    parser.add_argument("--plateau", type=float, default=0.004, help="no-progress band")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dir", default="runs")
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)
    args.name = args.name or time.strftime("earlystop-%H%M%S")

    et.init(
        project="control",
        name=args.name,
        dir=args.dir,
        backends=[],
        config={"lr": args.lr, "patience": args.patience},
    )
    rng = random.Random(args.seed)
    lr, decays, stopped_at, since = args.lr, 0, None, 0

    for step in range(args.steps):
        # Improvement restarts after each decay and then flattens again, so
        # there is a real plateau to find. Noise stays under --plateau.
        floor = 0.25 * (0.6**decays)
        loss = floor + 0.8 * math.exp(-(step - since) / 30) + rng.uniform(0, 0.001)

        if step % args.eval_every == 0 and step:
            et.log({"eval/loss": loss * 0.98}, commit=False)
            if plateaued(args.patience, args.plateau):
                if lr > args.min_lr:
                    lr, decays, since = lr / 10, decays + 1, step
                    print(f"step {step:>3}: plateau -> lr {lr:.1e}")
                    et.log({"event/lr_decay": decays}, commit=False)
                else:
                    stopped_at = step
                    print(f"step {step:>3}: plateau at the minimum lr, stopping")
                    et.summary()["stopped_early"] = True

        et.log({"train/loss": loss, "train/lr": lr})
        if stopped_at is not None:
            break

    et.summary()["decays"] = decays
    et.summary()["steps_run"] = et.get_run().step

    rows = et.history(-1, metrics=["eval/loss"], dropna=True)
    print(f"\n{len(rows)} evals, {decays} decay(s)")
    print(f"first eval {rows[0]['eval/loss']:.4f} -> last {rows[-1]['eval/loss']:.4f}")
    if stopped_at is None:
        print(f"ran all {args.steps} steps without stopping")
    else:
        saved = args.steps - stopped_at
        print(f"stopped at step {stopped_at}, saving {saved} steps of compute")

    info = dict(et.summary())
    et.finish()
    return info


if __name__ == "__main__":
    main()
