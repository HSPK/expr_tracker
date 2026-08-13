"""Alert rules that catch the four ways a run goes wrong.

Rules are expressions over a rolling window of your metrics, evaluated once per
committed step. This sends to a local handler so it runs offline; swap the
channel for lark/slack/email and nothing else changes.

    uv run python examples/alert_rules.py
    uv run python examples/alert_rules.py --fault stall

Faults: spike, nan, stall, none.
"""

from __future__ import annotations

import argparse
import math
import random
import time

import expr_tracker as et

RULES = [
    # Never fires during warm-up: too few points evaluates to UNKNOWN, not False
    "zscore(train/loss[30]) > 4    => error: loss spike {train/loss:.4f} @ step {step}",
    "isnan(train/loss)             => critical: loss went non-finite",
    "stalled(train/loss[20])       => warning: loss flat for 20 steps",
    # Must hold for 3 consecutive steps, so one noisy eval does not page anyone
    {
        "name": "accuracy_regression",
        "condition": "eval/accuracy < 0.5",
        "level": "warning",
        "for_steps": 3,
        "notify_recovery": True,
    },
]


def channel(sink):
    """A channel that appends to a list. Synchronous, so output stays ordered."""
    return {
        "channels": [
            {
                "type": "callable",
                "name": "local",
                "options": {"handler": sink.append},
                "policy": {"async_send": False, "dedup_window": 0},
            }
        ]
    }


def losses(args):
    """A loss curve with the requested fault injected into it."""
    rng = random.Random(args.seed)
    for step in range(args.steps):
        loss = math.exp(-step / 60) + rng.uniform(0, 0.02)
        if args.fault == "spike" and step == args.at:
            loss *= 12  # a sudden jump the z-score will notice
        elif args.fault == "nan" and step >= args.at:
            loss = float("nan")
        elif args.fault == "stall" and step >= args.at:
            loss = 0.5  # exactly flat, which stalled() is looking for
        yield step, loss


def main(argv=None) -> list:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fault", default="spike", choices=["spike", "nan", "stall", "none"]
    )
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--at", type=int, default=50, help="step to inject the fault")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dir", default="runs")
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)
    # A fresh run each time: resuming would leave the previous run's losses in
    # the rolling window, and the jump back to a high loss reads as a spike
    args.name = args.name or f"{args.fault}-{time.strftime('%H%M%S')}"

    fired: list = []
    et.init(
        project="alerts",
        name=args.name,
        dir=args.dir,
        backends=[],
        alert=channel(fired),
        alert_rules=RULES,
    )

    for step, loss in losses(args):
        accuracy = 0.4 if args.fault == "stall" and step > args.at else 0.9
        et.log({"eval/accuracy": accuracy}, commit=False)
        et.log({"train/loss": loss})

    print(f"fault: {args.fault}, {args.steps} steps")
    print(f"\n{len(fired)} message(s):")
    for message in fired:
        print(f"  [{message.level.value}] {message.title}")
        print(f"      {message.text.splitlines()[0]}")

    state = et.info()["alerts"]["rules"]
    print("\nrule state:")
    for rule in et.list_alert_rules():
        # An unnamed rule is known by its condition, which is what you read
        label = rule.condition if rule.auto_named else rule.name
        counts = state[rule.name]
        print(f"  {label:<34} fires={counts['fires']} firing={counts['firing']}")

    et.finish()
    return fired


if __name__ == "__main__":
    main()
