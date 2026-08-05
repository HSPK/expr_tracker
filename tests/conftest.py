import pytest

from expr_tracker.history import HistoryStore


@pytest.fixture
def store(tmp_path):
    """A HistoryStore without the open-row timeout, for deterministic tests."""
    created = []

    def factory(**kwargs):
        instance = HistoryStore()
        options = {
            "project": "p",
            "name": kwargs.pop("name", "run"),
            "dir": str(tmp_path),
            "max_open_seconds": None,
        }
        options.update(kwargs)
        instance.init(**options)
        created.append(instance)
        return instance

    yield factory
    for instance in created:
        instance.finish()


@pytest.fixture
def run(tmp_path):
    """Initialise a run and clean up the global singleton afterwards."""
    import expr_tracker as et
    from expr_tracker.run import current_run, set_run

    def factory(**kwargs):
        options = {
            "project": "p",
            "name": "run",
            "dir": str(tmp_path),
            "backends": [],
            "max_open_seconds": None,
        }
        options.update(kwargs)
        return et.init(**options)

    yield factory
    if current_run() is not None:
        try:
            et.finish()
        except Exception:
            set_run(None)


@pytest.fixture
def collector():
    """Return ``(channel_config, messages)`` capturing every delivered alert."""
    messages: list = []

    def channel(**overrides):
        config = {
            "type": "callable",
            "name": "test",
            "options": {"handler": messages.append},
            "policy": {
                "async_send": False,
                "dedup_window": 0,
                "rate_limit_per_minute": None,
                "max_retries": 0,
            },
        }
        config.update(overrides)
        return config

    return channel, messages
