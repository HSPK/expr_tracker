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

### JSONL buffering

The `jsonl` backend adapts its buffering to how often you call `log()`, so that
high-frequency logging doesn't hammer the disk (helpful on network mounts such as
BlobFuse) while low-frequency logging still lands on disk immediately:

- If the gap since the previous `log()` call is `>= buffer_interval` (default `1.0s`),
  the call is considered low-frequency and is written straight through.
- Otherwise records are batched in memory and flushed once `buffer_size` (default `50`)
  records accumulate.
- A background timer flushes records that have been buffered for more than
  `max_buffer_seconds` (default `5.0s`), so a burst that suddenly stops is never stranded
  in memory. `finish()` and the `atexit` hook flush any remainder.

Set `buffer_interval=None` to disable the frequency check, or `max_buffer_seconds=None`
to disable the background timer. Tune them via `backend_kwargs`:

```python
et.init(
    project="my_project",
    name="my_experiment",
    backends=["jsonl"],
    backend_kwargs={
        "jsonl": {"buffer_size": 200, "buffer_interval": 0.5, "max_buffer_seconds": 10},
    },
)
```