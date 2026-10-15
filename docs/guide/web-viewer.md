# Local experiment viewer

The native development build includes a read-only Web workspace for existing
JSONL experiments. Training jobs do not need to connect to the viewer, and the
viewer does not modify their logs.

```bash
et serve --root ./tracker/jsonl
```

Open `http://127.0.0.1:9110`. The root may be a single run directory or a directory
containing project/run directories. Use `--port` to choose another local port.
Select one run to inspect it or up to six runs to compare. Each stream and rank
is an explicit source selection; they are never silently merged.

Click **Choose directory** to browse directories on the server, move up to a
parent directory, or enter a server path and press **Go**. Press **Use this
directory** to load its experiments. The current directory is shown above the
experiment list. This is not a browser upload picker: with SSH forwarding, paths
refer to the remote training host, not your laptop. Relative paths are resolved
against the startup `--root`; absolute paths can select any directory accessible
to the server process, including directories outside the startup root.

Directory selection is local to each page and saved in its URL. Different tabs
can inspect different roots without changing each other's selection. Switching
directories clears selected runs, metrics, and step filters and cancels stale
requests. Files are never uploaded, modified, or deleted. Browsing directories
with more than 10,000 entries fails explicitly; enter a child directory path
directly instead.

The workspace includes metric charts, step/time axes, step-range filtering,
record pagination, config/summary inspection, and recorded span tables. Chart
queries limit returned points and preserve per-bucket extrema and missing values.
Non-finite metrics appear as gaps rather than zeroes. Record inspection preserves
large integers through explicit exact-value tags instead of JavaScript rounding.
The Records download contains the displayed API representation, including these
tags; it is not a byte-for-byte download of the source JSONL.
Record pages and each config/summary metadata group have a 16 MiB response budget.
Requests exceeding it fail explicitly; narrow record requests rather than loading
large tensor-like values into the browser.

Live refresh sees only complete rows already written to disk. It cannot display
an open metric row or records still held in the training process's write buffer.
Original files and retained `.parts` directories are read together. Patches for
the same step are merged; span events remain separate even when they share a step.

The server lazily builds a temporary, rebuildable SQLite query index. JSONL remains
the source of truth. Unchanged files reuse the index; append-only changes extend it,
and detected truncation/replacement rebuilds it. The index bounds in-memory data
and disappears when the viewer exits. Initial indexing of large runs can take time.
Malformed records are skipped with a visible count; file/query failures are
reported rather than shown as an empty successful result.

Up to 32 indexes are cached. Queries of different already indexed logs can run
concurrently; each individual log's refresh and query remain serialized.
Creating/rebuilding an index or evicting a cached index can still wait for active
queries, and a queued cache update can delay new queries. Concurrent comparisons
can use more memory even though the index count stays bounded. Limiting chart
points limits response size, not the number of source rows scanned.

The viewer displays the last file modification time, not a guessed run outcome:
absence of new rows does not establish that training succeeded, failed, or ended.

## Remote training hosts

Only loopback binding is supported. Start the viewer on the training host and
forward its port:

```bash
ssh -L 9110:127.0.0.1:9110 training-host
```

Open the same local URL in your browser. Do not expose the viewer through a public
reverse proxy: this local-only version has no network authentication. The startup
`--root` is an initial directory, **not a filesystem access sandbox**. The viewer
can browse directories and read experiment logs anywhere the server account has
access; run it with an appropriately restricted account when needed. Log sources
and rotation segments must stay within each request's selected root. Source
selection, Host validation, and same-origin checks apply to all native HTTP
endpoints, including directory browsing.

## Build the development UI

Source development requires Rust, Python 3.10+, uv, and Node.js 22.12+:

```bash
npm --prefix web ci
npm --prefix web run build
uv sync --all-extras
uv run et serve --root ./tracker/jsonl
```

For frontend development, run the server on port 9110 and then
`npm --prefix web run dev`. Vite proxies `/api` to the native local server.
Production builds are included in native wheel/source distributions; installed
wheels do not require Rust or Node at runtime.
