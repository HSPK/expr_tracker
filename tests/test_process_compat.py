"""Supported source-checkout and non-fork platform behavior."""

import importlib
import importlib.metadata
import os

import expr_tracker as et
from expr_tracker._process import register_fork_reset


def test_fork_registration_is_a_noop_when_the_platform_has_no_hook(monkeypatch):
    monkeypatch.delattr(os, "register_at_fork", raising=False)
    called = []
    register_fork_reset(lambda: called.append(True))
    assert called == []


def test_source_checkout_has_a_version_without_distribution_metadata(monkeypatch):
    original_version = et.__version__
    lookup = importlib.metadata.version

    def missing(name):
        if name == "expr_tracker":
            raise importlib.metadata.PackageNotFoundError(name)
        return lookup(name)

    try:
        with monkeypatch.context() as context:
            context.setattr(importlib.metadata, "version", missing)
            importlib.reload(et)
            assert et.__version__ == "0.0.0.dev0"
    finally:
        importlib.reload(et)
    assert et.__version__ == original_version
