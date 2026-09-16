"""Sample timestamps and commit liveness must not be confused for long steps."""

import threading
import time

import pytest

import expr_tracker as et
from expr_tracker.alerts.expr import EvalContext, evaluate, parse
from expr_tracker.history import MetricSeries


@pytest.fixture
def clock(monkeypatch):
    current = [1000.0]
    monkeypatch.setattr(time, "time", lambda: current[0])
    return current


def alert_options(collector):
    channel, messages = collector
    return {
        "alert": {"watchdog_interval": 0, "channels": [channel()]},
        "alert_rules": ["no_data(60s) => warning: no committed data"],
    }, messages


@pytest.mark.parametrize("opener", ["log", "span"])
def test_long_step_keeps_sample_time_but_refreshes_commit_liveness(
    run, clock, collector, opener
):
    options, messages = alert_options(collector)
    instance = run(**options)
    if opener == "log":
        et.log({"first": 1}, commit=False)
    else:
        with et.span("first"):
            pass
    assert instance.history.last_commit_time is None
    clock[0] += 300
    et.log({"loss": 0.5})
    assert et.history(1)[0]["_time"] == 1000.0
    assert instance.history.last_commit_time == 1300.0
    assert messages == []
    instance.alerts._tick()
    assert messages == []
    clock[0] += 61
    instance.alerts._tick()
    assert [message.text for message in messages] == ["no committed data"]
    et.log({"loss": 0.25})
    assert instance.history.last_commit_time == 1361.0
    assert len(messages) == 1


def test_uncommitted_updates_and_trace_only_events_do_not_refresh_liveness(
    run, clock, collector
):
    options, messages = alert_options(collector)
    instance = run(span_metrics=False, **options)
    et.log({"loss": 1.0})
    clock[0] += 61
    et.log({"pending": 1}, commit=False)
    with et.span("trace-only"):
        pass
    assert instance.history.last_commit_time == 1000.0
    instance.alerts._tick()
    assert len(messages) == 1


@pytest.mark.parametrize("rejected", [False, True])
def test_lagging_callback_uses_latest_accepted_commit_time(
    run, clock, collector, rejected
):
    entered, release = threading.Event(), threading.Event()

    class Backend:
        def init(self, **kwargs):
            pass

        def log(self, metrics, step=None):
            if step == 0:
                entered.set()
                assert release.wait(5)

        def finish(self):
            pass

    options, messages = alert_options(collector)
    instance = run(backends=[Backend()], max_record_bytes=512, **options)
    errors = []

    def log(data):
        try:
            instance.log(data)
        except Exception as error:
            errors.append(error)

    first = threading.Thread(target=log, args=({"loss": 0},))
    second = threading.Thread(
        target=log,
        args=({"loss": "x" * 1024} if rejected else {"loss": 1},),
    )
    first.start()
    try:
        assert entered.wait(5)
        clock[0] += 61
        second.start()
        if rejected:
            second.join(5)
            assert not second.is_alive()
        else:
            with instance.history._notification_condition:
                assert instance.history._notification_condition.wait_for(
                    lambda: len(instance.history._notifications) == 1, timeout=5
                )
        assert instance.history.last_commit_time == (1000.0 if rejected else 1061.0)
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not errors
    assert not first.is_alive() and not second.is_alive()
    assert len(messages) == (1 if rejected else 0)


def test_no_data_treats_epoch_zero_as_a_real_commit_timestamp():
    context = EvalContext(
        MetricSeries(), now=10.0, started_at=9.0, last_commit_time=0.0
    )
    assert evaluate(parse("no_data(5s)"), context) is True
