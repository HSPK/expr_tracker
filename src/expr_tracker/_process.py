"""Process ownership checks that do not add a syscall to every recorded span."""

from __future__ import annotations

import os
from collections.abc import Callable

_PID = os.getpid()


def process_id() -> int:
    return _PID


def check_process(owner_pid: int) -> None:
    if owner_pid != _PID:
        raise RuntimeError(
            "Tracker objects cannot be reused after fork; "
            "call et.init() and create plugins in the child process."
        )


def register_fork_reset(callback: Callable[[], None]) -> None:
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=callback)


def _reset_pid():
    global _PID
    _PID = os.getpid()


register_fork_reset(_reset_pid)
