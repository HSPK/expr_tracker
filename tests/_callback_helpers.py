"""Shared fixtures and helpers for the corresponding split regression suite."""

import threading
import time

import pytest


@pytest.fixture
def threads():
    started, errors = [], []

    def start(call):
        done = threading.Event()

        def invoke():
            try:
                call()
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=invoke, daemon=True)
        started.append(thread)
        thread.start()
        return done

    yield start
    for thread in started:
        thread.join(5)
    assert not any(thread.is_alive() for thread in started), "thread deadlocked"
    assert not errors


def wait_pending(history, count):
    deadline = time.monotonic() + 5
    while len(history._notifications) != count:
        assert time.monotonic() < deadline, (
            "commits did not reach notification delivery"
        )
        time.sleep(0.001)


def assert_drained(history):
    assert not history._notifications
    assert history._notification_owner is None
    assert history._notification_depth == 0
    assert not getattr(history._callback_state, "active", False)
    assert not hasattr(history._callback_state, "pending")


class Backend:
    def __init__(self, callback):
        self.callback = callback
        self.finished = threading.Event()

    def init(self, **kwargs):
        pass

    def log(self, data, step=None, commit=None):
        self.callback()

    def finish(self):
        self.finished.set()
