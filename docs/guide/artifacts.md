# Artifacts

Versioned file sets, stored per project and shared between its runs. The API matches
`wandb`.

```python
et.log_artifact("ckpt.pt", name="model", type="model", aliases=["best"])

# or build one explicitly
art = et.Artifact("bundle", type="config", metadata={"seed": 1})
art.add_file("train.yaml").add_dir("configs/").add_reference("s3://bucket/data.tar")
et.log_artifact(art)

# retrieve from any run in the same project
model = et.use_artifact("model:latest")   # or "model:v3" / "model:best"
path = model.download()                   # the materialised directory
model.download("/tmp/restore")            # or copy it somewhere
```

## Versions and aliases

Versions are `v0`, `v1`, … in log order. `latest` always points at the highest.
Custom aliases are recorded against the version they were logged with.

Artifacts are **deduplicated by content**: logging the same files again reuses the
existing version rather than creating a new one, while any new aliases are still
recorded against it.

```python
a = et.log_artifact("ckpt.pt", name="model")   # v0
b = et.log_artifact("ckpt.pt", name="model", aliases=["best"])
assert b.version == 0                          # same bytes, same version
et.use_artifact("model:best").version == 0
```

## Storage modes

```python
et.log_artifact("ckpt.pt", name="model", mode="copy")   # default
```

| Mode | Behaviour |
| --- | --- |
| `copy` | Copy the files into the store. Independent of later edits. |
| `link` | Hard-link them. No extra disk usage. |
| `reference` | Record path and digest only; nothing is materialised. |

`copy` is the default deliberately. `link` shares an inode with your file, and
`torch.save` rewrites in place — with `link`, saving a new checkpoint over the same
path would silently change an already-logged version. Use `link` only for files you
will not overwrite. If linking fails (a cross-device path, for instance) it falls
back to copying.

## Lineage

Each run appends to its own `artifacts.jsonl`, recording which versions it produced
and consumed:

```json
{"_time": 1754323200.1, "action": "log", "name": "model", "version": 0, "type": "model", "step": 100}
{"_time": 1754323300.5, "action": "use", "name": "dataset", "version": 2, "type": "dataset", "step": 0}
```

## Layout

```
tracker/jsonl/<project>/
├── artifacts/
│   ├── index.jsonl        # every version, with digests and aliases
│   └── <name>/v<N>/...    # the files themselves
└── <run>/
    └── artifacts.jsonl    # this run's lineage
```

The index is append-only and tolerates corrupt lines: a damaged entry is skipped with
a warning rather than losing the rest.

## With wandb

When the `wandb` backend is enabled, `log_artifact` is mirrored to
`wandb.log_artifact` as well. A failure there is logged and never affects the local
copy.
