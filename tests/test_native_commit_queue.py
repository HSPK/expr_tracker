"""The native queue owns delivery boundaries and completion."""

import threading

import pytest
from expr_tracker._native import NativeCommitQueue, NativeCommitTicket


def assert_idle(queue):
    assert queue.count == 0
    assert queue.owner is None
    assert queue.depth == 0


def test_empty_batch_has_no_completion_ticket():
    queue = NativeCommitQueue()
    assert queue.enqueue([]) is None
    queue.drain(threading.get_ident(), lambda row: pytest.fail("unexpected callback"))
    assert_idle(queue)


def test_batch_returns_only_its_last_ticket_and_preserves_fifo():
    queue = NativeCommitQueue()
    first = queue.enqueue([1, 2])
    second = queue.enqueue([3])
    assert isinstance(first, NativeCommitTicket)
    assert isinstance(second, NativeCommitTicket)
    observed = []
    owner = threading.get_ident()

    queue.drain(owner, observed.append, first)
    assert observed == [1, 2]
    assert queue.count == 1
    assert queue.owner is None
    assert queue.depth == 0

    queue.drain(owner, observed.append, second)
    assert observed == [1, 2, 3]
    assert_idle(queue)


def test_completed_target_does_not_consume_a_later_batch():
    queue = NativeCommitQueue()
    first = queue.enqueue([1])
    owner = threading.get_ident()
    observed = []
    queue.drain(owner, observed.append, first)

    second = queue.enqueue([2])
    queue.drain(owner, observed.append, first)
    assert observed == [1]
    assert queue.count == 1
    queue.drain(owner, observed.append, second)
    assert observed == [1, 2]
    assert_idle(queue)


def test_recursive_drain_captures_its_tail_without_consuming_later_rows():
    queue = NativeCommitQueue()
    target = queue.enqueue(["outer", "pending"])
    owner = threading.get_ident()
    observed = []

    def notify(row):
        observed.append(row)
        if row == "outer":
            assert queue.depth == 1
            queue.enqueue(["nested"])
            queue.drain(owner, notify)
            observed.append("outer-return")
            assert queue.count == 1
            assert queue.owner == owner
            assert queue.depth == 1
        elif row == "pending":
            assert queue.depth == 2
            queue.enqueue(["later"])

    queue.drain(owner, notify, target)
    assert observed == ["outer", "pending", "nested", "outer-return"]
    assert queue.count == 1
    queue.drain(owner, notify)
    assert observed == ["outer", "pending", "nested", "outer-return", "later"]
    assert_idle(queue)


def test_recursive_drain_with_no_pending_rows_does_not_wait_for_its_ancestor():
    queue = NativeCommitQueue()
    target = queue.enqueue([1])
    owner = threading.get_ident()
    observed = []

    def notify(row):
        queue.drain(owner, observed.append)
        assert queue.owner == owner
        assert queue.depth == 1
        observed.append(row)

    queue.drain(owner, notify, target)
    assert observed == [1]
    assert_idle(queue)


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_callback_failure_completes_the_row_and_releases_ownership(error):
    queue = NativeCommitQueue()
    target = queue.enqueue([1, 2])
    owner = threading.get_ident()

    def fail(row):
        assert row == 1
        raise error("injected")

    with pytest.raises(error, match="injected"):
        queue.drain(owner, fail, target)
    assert queue.count == 1
    assert queue.owner is None
    assert queue.depth == 0
    observed = []
    queue.drain(owner, observed.append, target)
    assert observed == [2]
    assert_idle(queue)
