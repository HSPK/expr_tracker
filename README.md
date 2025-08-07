# Experiment Tracker

A simple experiment tracker that supports `wandb`, `trackio` and local `jsonl` storage. Features include:

## Features
1. **Multi-Backend Support**: Compatible with `wandb`, `trackio`, and local `jsonl` storage.
2. **Alert System**: Send alerts via Lark, with email and other platforms to be added.
3. **Resume Functionality**: Allows resuming experiments using project and name as unique identifiers.
4. **Simple API**: Provides a straightforward API similar to `wandb`, making it easy to integrate into existing workflows.

## Usage

Add dependency to your project:

```bash
uv add expr_tracker
```

Simple usage example:
```python
import expr_tracker as et

et.init(project="my_project", name="my_experiment", backends=["wandb", "jsonl"])
et.log({"accuracy": 0.95, "loss": 0.05})
et.alert("Experiment completed!", text="Your experiment has finished successfully.", subtitle="Experiment Status")
et.finish()
```