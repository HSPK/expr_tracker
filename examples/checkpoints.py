"""Checkpoints as artifacts: versioned, deduplicated, and found again later.

Artifacts are stored per project and shared by its runs, so a later run can ask
for ``model:best`` without knowing which run produced it.

    uv run python examples/checkpoints.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import expr_tracker as et


def write_checkpoint(directory: Path, step: int, accuracy: float) -> Path:
    """Stand-in for torch.save."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.pt"
    path.write_text(f"weights@{step}:{accuracy:.3f}", encoding="utf-8")
    return path


def train(args, workdir: Path) -> str:
    run = et.init(
        project="checkpoints",
        name=args.name,
        dir=args.dir,
        backends=[],
        config={"steps": args.steps},
    )
    best = -1.0

    for step in range(0, args.steps, args.every):
        accuracy = 0.7 + 0.1 * (step / args.steps)
        et.log({"eval/accuracy": accuracy}, commit=False)
        et.log({"train/loss": 1.0 / (step + 1)})

        path = write_checkpoint(workdir, step, accuracy)
        aliases = ["best"] if accuracy > best else []
        artifact = et.log_artifact(path, name="model", type="model", aliases=aliases)
        if accuracy > best:
            best = accuracy
            et.summary()["best_accuracy"] = accuracy
        print(f"step {step:>3}: acc {accuracy:.3f} -> model:v{artifact.version}")

    # The same bytes again: deduplicated, so no new version is created
    repeat = et.log_artifact(workdir / "model.pt", name="model", type="model")
    print(f"logging the same file again -> v{repeat.version} (deduplicated)")

    et.finish()
    return run.name


def restore(args, into: Path) -> None:
    """A different run, later, asking for the best checkpoint."""
    et.init(project="checkpoints", name=f"{args.name}-eval", dir=args.dir, backends=[])
    artifact = et.use_artifact("model:best")
    path = Path(artifact.download(str(into)))
    payload = (path / "model.pt").read_text(encoding="utf-8")
    print(f"\nrestored model:best (v{artifact.version}) -> {payload}")
    print(f"downloaded to {path}")
    et.finish()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--every", type=int, default=10, help="checkpoint interval")
    parser.add_argument("--dir", default="runs")
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)
    args.name = args.name or time.strftime("ckpt-%H%M%S")

    workdir = Path(args.dir) / "scratch" / args.name
    train(args, workdir)
    restore(args, workdir / "restored")


if __name__ == "__main__":
    main()
