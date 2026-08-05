"""Regressions from code review; each test pins one fixed bug."""

import itertools
import json
import sys
import threading
import time

import pytest

import expr_tracker as et
from expr_tracker.alerts.backends import create_backend
from expr_tracker.alerts.backends.base import SendError, _redact
from expr_tracker.alerts.expr import EvalContext, ExprError, evaluate, parse, validate
from expr_tracker.alerts.models import AlertMessage, ChannelConfig
from expr_tracker.history import HistoryStore, MetricSeries


def store(tmp_path, name, **kwargs):
    options = {
        "project": "p",
        "name": name,
        "dir": str(tmp_path),
        "max_open_seconds": None,
    }
    options.update(kwargs)
    return HistoryStore().init(**options)


# ---------------------------------------------------------------- history reads


def test_resume_keeps_existing_history_visible(tmp_path):
    """After a resume the cache is empty, so queries must fall back to disk."""
    first = store(tmp_path, "resume")
    for i in range(5):
        first.log({"v": i})
    first.finish()

    second = store(tmp_path, "resume")
    assert [r["_step"] for r in second.get(-1)] == [0, 1, 2, 3, 4]
    assert [r["_step"] for r in second.get(3)] == [2, 3, 4]
    assert [r["_step"] for r in second.get(-1, step_range=(1, 4))] == [1, 2, 3]
    second.log({"v": 5})
    assert [r["_step"] for r in second.get(-1)] == [0, 1, 2, 3, 4, 5]
    second.finish()


def test_patch_line_not_lost_at_cache_boundary(tmp_path):
    """When one step spans the cache boundary, a step lookup drops its earlier row."""
    s = store(tmp_path, "patch_boundary", cache_rows=1, buffer_size=1)
    s.log({"a": 1}, step=3)
    s.flush(commit_open=True)
    s.log({"b": 2}, step=3)
    s.flush(commit_open=True)
    record = s.get(-1)[0]
    assert record["a"] == 1 and record["b"] == 2
    s.finish()


def test_get_n_returns_n_steps_despite_patches(tmp_path):
    """Many patch lines must not make get(n) return fewer steps."""
    s = store(tmp_path, "patch_count", cache_rows=1, buffer_size=1)
    s.log({"x": 0}, step=0)
    s.flush(commit_open=True)
    for i in range(10):
        s.log({f"p{i}": i}, step=1)
        s.flush(commit_open=True)
    assert [r["_step"] for r in s.get(2)] == [0, 1]
    s.finish()


def test_eviction_uses_row_watermark_not_step(tmp_path):
    """The durability watermark must be a row ordinal, not a step."""
    s = store(
        tmp_path,
        "watermark",
        cache_bytes=1,
        buffer_size=1000,
        buffer_interval=None,
        max_buffer_seconds=None,
    )
    s.log({"a": 1}, step=0)
    s.flush(commit_open=True)  # first row becomes durable and is evicted
    assert len(s._cache) == 0
    assert s.writer.flushed_row == 0

    original_flush, s.writer.flush = (
        s.writer.flush,
        lambda: None,
    )  # freeze the watermark
    try:
        s.log({"b": 2}, step=0, commit=True)  # another step 0 row, not yet durable
        # A step watermark would call step 0 durable and evict it; a row one does not
        assert len(s._cache) == 1
        assert s._cache[0][0] == 1 and s._cache[0][1] == 0  # row=1, step=0
    finally:
        s.writer.flush = original_flush

    s.finish()
    record = s.get(-1)[0]
    assert record["a"] == 1 and record["b"] == 2


def test_reader_detects_unsorted_file_without_meta(tmp_path):
    """Without meta, scanning must still detect that merging by step is required."""
    from expr_tracker.history import JsonlReader

    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"_step": 2, "a": 1}\n{"_step": 1, "b": 2}\n{"_step": 2, "c": 3}\n',
        encoding="utf-8",
    )
    reader = JsonlReader(path)
    assert reader.merge is True
    assert [r["_step"] for r in reader.read_all()] == [1, 2]
    assert reader.read_all()[1] == {"_step": 2, "a": 1, "c": 3}


def test_reserved_metric_names_are_ignored(tmp_path):
    """Logging `_step`/`_time` as metrics must not corrupt record identity."""
    s = store(tmp_path, "reserved")
    s.log({"loss": 1.0, "_step": 999, "_time": 0})
    s.finish()
    record = s.get(-1)[0]
    assert record["_step"] == 0 and record["_time"] > 0 and record["loss"] == 1.0


# ---------------------------------------------------------------- lifecycle


def test_open_row_timer_does_not_commit_the_next_step(tmp_path):
    """After the step advances, the old timer must not commit the new open row."""
    s = store(tmp_path, "timer_gen", max_open_seconds=0.15, buffer_size=1)
    s.log({"a": 1}, step=0)
    time.sleep(0.12)
    s.log({"b": 2}, step=1)  # commits step 0 and re-arms the timer for step 1
    time.sleep(0.08)  # the old timer would expire around now
    assert s.open_record() is not None and s.open_record()["_step"] == 1
    s.finish()


def test_log_after_finish_is_rejected(tmp_path):
    s = store(tmp_path, "closed")
    s.log({"v": 1})
    s.finish()
    s.log({"v": 2})
    assert [r["_step"] for r in s.get(-1)] == [0]


def test_concurrent_init_only_creates_one_run(tmp_path):
    from expr_tracker.run import current_run, set_run

    results: list = []

    def worker():
        try:
            results.append(
                (
                    "ok",
                    et.init(project="p", name="race", dir=str(tmp_path), backends=[]),
                )
            )
        except RuntimeError:
            results.append(("rejected", None))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert [kind for kind, _ in results].count("ok") == 1
        assert current_run() is not None
    finally:
        et.finish() if current_run() is not None else set_run(None)


def test_finish_commits_open_row_before_closing_alerts(tmp_path, collector):
    """An alert fired by the final open row must still be delivered."""
    channel, messages = collector
    et.init(
        project="p",
        name="final",
        dir=str(tmp_path),
        backends=[],
        alert={"channels": [channel()]},
        alert_rules=["loss > 5 => error: final alert"],
    )
    et.log({"loss": 10}, step=0)  # open row, not committed yet
    assert messages == []
    et.finish()
    assert [m.text for m in messages] == ["final alert"]


def test_invalid_rule_does_not_leave_a_watchdog_thread(tmp_path):
    """A time-based rule followed by an invalid one must not leak a watchdog."""
    before = {t.name for t in threading.enumerate()}
    with pytest.raises(ExprError):
        et.init(
            project="p",
            name="bad",
            dir=str(tmp_path),
            backends=[],
            alert_rules=["no_data(1s) => error: hung", "mean(loss) => warn: bad"],
        )
    time.sleep(0.1)
    leaked = {t.name for t in threading.enumerate()} - before
    assert not any(name.startswith("et-alert-watchdog") for name in leaked)


# ---------------------------------------------------------------- alert eval


def test_no_data_uses_last_commit_time(tmp_path, collector):
    """A committed step leaves no open row; no_data() must not read that as no data."""
    channel, messages = collector
    run = et.init(
        project="p",
        name="nodata",
        dir=str(tmp_path),
        backends=[],
        alert={"channels": [channel()]},
    )
    et.log({"loss": 1.0})
    engine = run.alerts
    engine.add_rule("no_data(60s) => error: hung")
    engine._tick()
    assert messages == []  # data was committed a moment ago
    et.finish()


def test_scalar_min_max_require_two_arguments():
    with pytest.raises(ExprError, match="at least two scalar arguments"):
        validate(parse("min(1)"))
    validate(parse("min(1, 2)"))
    validate(parse("min(loss[20])"))


def test_function_errors_degrade_to_unknown(monkeypatch):
    from expr_tracker.alerts.expr import functions

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(functions.WINDOW_FUNCS, "mean", (explode, None, 1, 1))
    series = MetricSeries()
    series.add(0, 0.0, {"loss": 1.0})
    assert evaluate(parse("mean(loss[5]) > 1"), EvalContext(series, step=0)) is not True


# ---------------------------------------------------------------- delivery


def test_webhook_url_is_redacted_in_errors():
    redacted = _redact("https://open.feishu.cn/hook/v2/abc-secret-token?sign=xyz")
    assert "abc-secret-token" not in redacted and "xyz" not in redacted
    assert redacted.startswith("https://open.feishu.cn")


@pytest.mark.parametrize("kind", ["dingtalk", "wecom"])
def test_provider_errcode_is_treated_as_failure(monkeypatch, kind):
    from expr_tracker.alerts import backends as backend_module

    monkeypatch.setattr(
        backend_module,
        "post_json",
        lambda *a, **k: '{"errcode": 310000, "errmsg": "nope"}',
    )
    backend = create_backend(ChannelConfig(type=kind, url="http://hook"))
    with pytest.raises(SendError, match="errcode=310000"):
        backend.send(AlertMessage(title="t", text="x"))


def test_dispatch_worker_survives_backend_errors(tmp_path):
    from expr_tracker.alerts.dispatch import Dispatcher
    from expr_tracker.alerts.models import AlertConfig, WebhookPolicy

    delivered: list = []
    calls = {"n": 0}

    def handler(message):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("backend blew up")
        delivered.append(message)

    channel = ChannelConfig(
        type="callable",
        name="c",
        options={"handler": handler},
        policy=WebhookPolicy(
            async_send=True, dedup_window=0, rate_limit_per_minute=None, max_retries=0
        ),
    )
    dispatcher = Dispatcher(AlertConfig(channels=[channel]))
    dispatcher.send(AlertMessage(title="a", text="1"))
    dispatcher.send(AlertMessage(title="b", text="2"))
    dispatcher.close(timeout=3.0)
    assert [m.title for m in delivered] == ["b"]  # the first failure did not kill it


def test_dedup_state_is_bounded():
    from expr_tracker.alerts.dispatch import MAX_DEDUP_KEYS, Deduper

    deduper = Deduper(0.001)
    for i in range(MAX_DEDUP_KEYS + 500):
        deduper.check(f"k{i}", now=i)
    assert len(deduper._seen) <= MAX_DEDUP_KEYS + 1


# ---------------------------------------------------------------- own review pass


def test_offline_read_with_n_zero_and_step_range(tmp_path):
    """`records[-0:]` returns everything, so n=0 must short-circuit."""
    from expr_tracker.history import read_history

    path = tmp_path / "metrics.jsonl"
    path.write_text('{"_step":0,"a":1}\n{"_step":1,"a":2}\n', encoding="utf-8")
    assert read_history(path, 0) == []
    assert read_history(path, 0, step_range=(0, None)) == []
    assert [r["_step"] for r in read_history(path, 1, step_range=(0, None))] == [1]


def test_reinit_flushes_the_previous_writer(tmp_path):
    """Re-initialising a store must not strand the previous run's buffer."""
    s = HistoryStore()
    s.init(
        project="p",
        name="first",
        dir=str(tmp_path),
        max_open_seconds=None,
        buffer_size=1000,
        buffer_interval=None,
        max_buffer_seconds=None,
    )
    s.log({"v": 1})
    first_file = s.log_fp
    s.init(project="p", name="second", dir=str(tmp_path), max_open_seconds=None)
    try:
        assert first_file.exists() and '"v": 1' in first_file.read_text()
    finally:
        s.finish()


def test_history_n_with_many_patch_rows_per_step(tmp_path):
    """The widening loop must read from disk even when the cache already holds n rows."""
    s = store(tmp_path, "widen", max_open_seconds=0.02, cache_rows=4, buffer_size=1)
    for step in range(12):
        for k in range(3):  # three physical rows per step via timeout patches
            s.log({f"k{k}": step * 10 + k}, step=step)
            time.sleep(0.03)
    s.flush(commit_open=True)

    rows = s.get(6)
    assert [r["_step"] for r in rows] == list(range(6, 12))
    for row in rows:  # the oldest returned step must not be half-merged
        assert all(f"k{k}" in row for k in range(3)), row
    assert len(s.get(-1)) == 12
    s.finish()


def test_concurrent_logging_keeps_row_order_and_history(tmp_path):
    """Row ordinals must stay aligned with physical lines under concurrent logging."""
    switch = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    s = store(
        tmp_path,
        "rowrace",
        cache_rows=2,
        buffer_size=3,
        buffer_interval=None,
        max_buffer_seconds=None,
    )
    try:

        def worker(tid):
            for i in range(150):
                s.log({"t": tid, "i": i})

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        total = 6 * 150
        assert [r["_step"] for r in s.get(30)] == list(range(total - 30, total))
        s.finish()
        assert [r["_step"] for r in s.get(-1)] == list(range(total))
        with open(s.log_fp, "rb") as f:
            steps = [json.loads(line)["_step"] for line in f if line.strip()]
        assert len(steps) == total
        # An out-of-order file is only safe because merge mode is switched on
        assert all(b >= a for a, b in itertools.pairwise(steps)) or s._needs_merge
    finally:
        sys.setswitchinterval(switch)


def test_torn_trailing_line_is_repaired_on_resume(tmp_path):
    """A half-written final line must be dropped, not appended onto."""
    first = store(tmp_path, "torn", buffer_size=1)
    for i in range(3):
        first.log({"v": i})
    first.finish()
    with open(first.log_fp, "ab") as f:
        f.write(b'{"_step": 3, "v": 3, "pa')  # crash mid-line

    second = store(tmp_path, "torn", cache_rows=2, buffer_size=1)
    assert second.writer.lines == 3
    assert second.current_step == 3
    for i in range(3, 10):
        second.log({"v": i})
    assert [r["_step"] for r in second.get(-1)] == list(range(10))
    second.finish()
    with open(second.log_fp, "rb") as f:
        physical = [json.loads(line)["_step"] for line in f if line.strip()]
    assert physical == list(range(10))


# ---------------------------------------------------------------- whole-code review


def test_non_string_metric_keys_do_not_break_logging(tmp_path):
    """A non-str key used to raise mid-_store_row, skewing row ordinals forever."""
    import enum

    class Level(enum.IntEnum):
        A = 1

    s = store(tmp_path, "keys")
    s.log({"loss": 0.0})
    for key in (1, 2.5, True, None, Level.A, (1, 2), b"k"):
        s.log({key: 1, "loss": 9.0})
    for i in range(3):
        s.log({"loss": float(i)})
    s.flush(commit_open=True)

    rows = s.get(-1)
    assert [r["_step"] for r in rows] == list(range(len(rows)))
    stats = s.stats()
    assert stats["rows_logged"] == stats["rows_on_disk"]  # no ordinal was skipped
    assert all(isinstance(k, str) for row in rows for k in row)
    s.finish()


def test_open_row_repeating_a_committed_step_is_merged(tmp_path):
    """A timeout commit followed by more metrics for that step is still one row."""
    s = store(tmp_path, "reopen", max_open_seconds=0.05, buffer_size=1)
    s.log({"train/loss": 1.0}, step=7)
    time.sleep(0.2)  # the open row times out and is committed
    s.log({"eval/acc": 0.9}, step=7)  # same step, re-opened

    rows = s.get(1)
    assert len(rows) == 1
    assert rows[0]["train/loss"] == 1.0 and rows[0]["eval/acc"] == 0.9
    s.finish()


def test_alias_added_on_a_dedup_reuse_is_persisted(tmp_path):
    """The reuse index line was skipped on replay, losing the alias."""
    source = tmp_path / "f.bin"
    source.write_bytes(b"same-content")
    et.init(project="p", name="a", dir=str(tmp_path / "runs"), backends=[])
    try:
        first = et.log_artifact(str(source), name="model", type="model")
        again = et.log_artifact(
            str(source), name="model", type="model", aliases=["best"]
        )
        assert first.version == again.version  # deduped, no new version
        assert et.use_artifact("model:best").qualified_name == "model:v0"
    finally:
        et.finish()


def test_summary_ignores_non_string_keys(tmp_path):
    et.init(project="p", name="s", dir=str(tmp_path), backends=[])
    try:
        et.log({(1, 2): 3, b"k": 4, "loss": 5})
        assert dict(et.summary()) == {"loss": 5}
    finally:
        et.finish()


def test_logged_artifact_survives_an_in_place_overwrite(tmp_path):
    """Hard-linking made torch.save-style rewrites mutate already logged versions."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"EPOCH-1")
    et.init(project="p", name="m", dir=str(tmp_path / "runs"), backends=[])
    try:
        v0 = et.log_artifact(str(ckpt), name="model")
        with open(ckpt, "wb") as f:  # in-place truncate + rewrite
            f.write(b"EPOCH-2")
        v1 = et.log_artifact(str(ckpt), name="model")
        assert (v0.dir / "ckpt.pt").read_bytes() == b"EPOCH-1"
        assert (v1.dir / "ckpt.pt").read_bytes() == b"EPOCH-2"
    finally:
        et.finish()
