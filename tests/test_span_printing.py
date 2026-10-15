"""Span printing and indentation."""

import asyncio
import threading
import time

import pytest
from _span_plugin_helpers import (
    MB as MB,
)
from _span_plugin_helpers import (
    FakeCuda as FakeCuda,
)
from _span_plugin_helpers import (
    FakeDevice as FakeDevice,
)
from _span_plugin_helpers import (
    FakeNvml as FakeNvml,
)
from _span_plugin_helpers import (
    FastFakeCuda as FastFakeCuda,
)
from _span_plugin_helpers import (
    Recorder as Recorder,
)
from _span_plugin_helpers import (
    _visual_indent as _visual_indent,
)
from _span_plugin_helpers import (
    depth_of as depth_of,
)
from _span_plugin_helpers import (
    fake_nvml as fake_nvml,
)
from _span_plugin_helpers import (
    fake_torch as fake_torch,
)
from _span_plugin_helpers import (
    fast_torch as fast_torch,
)
from _span_plugin_helpers import (
    has_cuda as has_cuda,
)
from _span_plugin_helpers import (
    has_nvml as has_nvml,
)
from _span_plugin_helpers import (
    lines as lines,
)
from _span_plugin_helpers import (
    measured_span as measured_span,
)
from _span_plugin_helpers import (
    needs_cuda as needs_cuda,
)
from _span_plugin_helpers import (
    needs_nvml as needs_nvml,
)
from _span_plugin_helpers import (
    resource_span as resource_span,
)
from _span_plugin_helpers import (
    run as run,
)
from _span_plugin_helpers import (
    span_records as span_records,
)

import expr_tracker as et
from expr_tracker.spans import INDENT


def test_prints_a_start_and_an_end_line(run, lines):
    run()
    with et.span("forward", print_fn=lines.append):
        pass
    assert len(lines) == 2
    assert lines[0].startswith("-> forward  ")
    assert lines[1].startswith("<- forward  ")
    assert lines[1].endswith("ms")


def test_context_manager_can_be_reentered_without_leaking_outer_span(run):
    instance = run()
    region = et.span("work")
    with region as outer:
        with region as inner:
            assert inner.path == "work/work"
        assert et.spans.current_span() is outer
    assert et.spans.current_span() is None
    assert [row["name"] for row in span_records(instance)] == ["work/work", "work"]


def test_span_cannot_be_begun_twice(run):
    run()
    current = et.start_span("work")
    try:
        with pytest.raises(RuntimeError, match="begun more than once"):
            current.begin()
    finally:
        current.end()


def test_indents_one_tab_per_level(run, lines):
    run()
    with et.span("a", print_fn=lines.append), et.span("b"), et.span("c"):
        pass
    assert [depth_of(line) for line in lines] == [0, 1, 2, 2, 1, 0]
    assert lines[2].startswith(INDENT * 2 + "-> c")


def test_indentation_is_relative_to_where_printing_started(run, lines):
    """A silent parent must not leave a stray leading tab on its child."""
    run()
    with (
        et.span("outer"),
        et.span("mid"),
        et.span("inner", print_fn=lines.append),
        et.span("leaf"),
    ):
        pass
    assert [depth_of(line) for line in lines] == [0, 1, 1, 0]
    assert lines[0] == lines[0].lstrip(" ")


def test_an_overriding_child_roots_its_own_indentation(run, lines):
    other = []
    run()
    with (
        et.span("a", print_fn=lines.append),
        et.span("b"),
        et.span("c", print_fn=other.append),
        et.span("d"),
    ):
        pass
    assert [depth_of(line) for line in other] == [0, 1, 1, 0]


def test_the_run_default_still_indents_the_whole_tree(run, lines):
    """The default is one shared handler, so nesting must survive it."""
    run(span_print_fn=lines.append)
    with et.span("a"), et.span("b"), et.span("c"):
        pass
    assert [depth_of(line) for line in lines] == [0, 1, 2, 2, 1, 0]


def test_indentation_keeps_up_with_deep_nesting(run, lines):
    run()
    depth = 12
    spans = [et.span("s0", print_fn=lines.append).__enter__()]
    spans += [et.span(f"s{i}").__enter__() for i in range(1, depth)]
    for sp in reversed(spans):
        sp.end()
    assert [depth_of(line) for line in lines[:depth]] == list(range(depth))


def test_a_new_thread_starts_a_fresh_tree(run, lines):
    """Threads do not inherit the stack, so a worker must not be indented."""
    run()
    inner = []

    def work():
        with et.span("worker", print_fn=inner.append):
            pass

    with et.span("main", print_fn=lines.append):
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
    assert [depth_of(line) for line in inner] == [0, 0]


def test_concurrent_tasks_are_each_indented_by_their_own_depth(run, lines):
    run()

    async def leg(name):
        async with et.span(name), et.span("sub"):
            await asyncio.sleep(0.01)

    async def main():
        async with et.span("gather", print_fn=lines.append):
            await asyncio.gather(leg("a"), leg("b"))

    asyncio.run(main())
    by_name = {}
    for line in lines:
        parts = line.split()
        by_name.setdefault(parts[1], []).append(depth_of(line))
    assert by_name["gather"] == [0, 0]
    assert by_name["a"] == [1, 1] and by_name["b"] == [1, 1]
    assert by_name["sub"] == [2, 2, 2, 2]


def test_a_recursive_span_indents_by_recursion_depth(run, lines):
    run()

    @et.span("fib", print_fn=lines.append)
    def fib(n):
        return n if n < 2 else fib(n - 1) + fib(n - 2)

    fib(2)
    assert [depth_of(line) for line in lines] == [0, 1, 1, 1, 1, 0]


def test_a_span_ended_out_of_order_does_not_resurrect_itself(run):
    """Resetting the ContextVar token would restore the whole stack with it."""
    from expr_tracker.spans import _STACK

    run()
    outer = et.start_span("outer")
    inner = et.start_span("inner")
    outer.end()
    assert [sp.name for sp in _STACK.get()] == ["inner"]
    inner.end()
    assert _STACK.get() == ()
    after = et.start_span("after")
    assert after.path == "after"  # not "outer/after"
    after.end()


def test_out_of_order_ends_keep_the_indentation_honest(run, lines):
    run()
    outer = et.start_span("outer", print_fn=lines.append)
    inner = et.start_span("inner")
    outer.end()
    inner.end()
    later = et.start_span("later", print_fn=lines.append)
    later.end()
    assert [depth_of(line) for line in lines] == [0, 1, 0, 1, 0, 0]


def test_a_slashed_name_does_not_fake_a_level(run, lines):
    """`depth` is nesting, not a count of slashes in the path."""
    r = run()
    with et.span("data/load", print_fn=lines.append), et.span("decode"):
        pass
    et.log({"loss": 1.0})
    et.finish()
    assert [depth_of(line) for line in lines] == [0, 1, 1, 0]
    records = {rec["name"]: rec["depth"] for rec in span_records(r)}
    assert records == {"data/load": 0, "data/load/decode": 1}


def test_the_indent_survives_a_log_prefix(run, lines):
    """A tab is measured from the start of the line, so behind a logger prefix
    the first level collapses to whatever is left of the tab stop. Spaces do
    not care where the line begins."""
    run()
    with et.span("a", print_fn=lines.append), et.span("b"), et.span("c"):
        pass
    assert "\t" not in "".join(lines)
    for prefix in ("", "x" * 11, "x" * 71, "x" * 72):
        columns = [_visual_indent(prefix + line) for line in lines]
        relative = [column - len(prefix) for column in columns]
        assert relative == [0, 2, 4, 4, 2, 0], prefix


def test_children_inherit_the_handler(run, lines):
    run()
    with et.span("outer", print_fn=lines.append), et.span("inner"):
        pass
    assert any("inner" in line for line in lines)


def test_a_child_can_override_the_handler(run, lines):
    other = []
    run()
    with (
        et.span("outer", print_fn=lines.append),
        et.span("inner", print_fn=other.append),
        et.span("deepest"),
    ):
        pass
    assert not any("inner" in line for line in lines)
    assert [line.split()[1] for line in other] == [
        "inner",
        "deepest",
        "deepest",
        "inner",
    ]


def test_a_silent_child_stays_silent_under_a_printing_parent(run, lines):
    """A parent handler is inherited, so opting a subtree out needs a no-op."""
    run()
    with (
        et.span("outer", print_fn=lines.append),
        et.span("inner", print_fn=lambda _: None),
    ):
        pass
    assert [line.split()[1] for line in lines] == ["outer", "outer"]


def test_reports_the_duration_it_measured(run, lines):
    run()
    with et.span("slow", print_fn=lines.append):
        time.sleep(0.02)
    measured = float(lines[1].rsplit("  ", 1)[1].removesuffix("ms"))
    assert 15 < measured < 200


def test_marks_a_failed_span(run, lines):
    run()
    with pytest.raises(ValueError), et.span("boom", print_fn=lines.append):
        raise ValueError("x")
    assert lines[1].endswith("!ValueError")


def test_shows_plugin_metrics_on_the_end_line(run, lines):
    run()
    with et.span("m", print_fn=lines.append, plugins=[lambda s: {"hits": 3}]):
        pass
    assert "hits=3" in lines[1]


def test_trims_trailing_zeros_from_floats(run, lines):
    run()
    with et.span(
        "m", print_fn=lines.append, plugins=[lambda s: {"a": 1.5000, "b": 2.0}]
    ):
        pass
    assert "a=1.5" in lines[1] and "b=2 " in lines[1] + " "


def test_a_broken_handler_does_not_break_the_span(run):
    run()

    def explode(line):
        raise RuntimeError("printer is down")

    with et.span("safe", print_fn=explode) as sp:
        pass
    assert sp.duration_ms > 0
    assert span_records(et.get_run())[0]["name"] == "safe"


def test_prints_without_a_run(lines):
    with et.span("orphan", print_fn=lines.append) as sp:
        pass
    assert len(lines) == 2
    assert sp.duration_ms >= 0


def test_the_decorator_takes_a_handler(run, lines):
    run()

    @et.span("work", print_fn=lines.append)
    def work():
        return 7

    assert work() == 7
    assert len(lines) == 2


def test_async_spans_print(run, lines):
    run()

    async def main():
        async with et.span("io", print_fn=lines.append):
            await asyncio.sleep(0)

    asyncio.run(main())
    assert len(lines) == 2


def test_start_span_takes_a_handler(run, lines):
    run()
    sp = et.start_span("manual", print_fn=lines.append)
    sp.end()
    assert len(lines) == 2


def test_the_run_supplies_a_default_handler(run, lines):
    run(span_print_fn=lines.append)
    with et.span("a"):
        pass
    assert len(lines) == 2


def test_a_span_argument_beats_the_run_default(run, lines):
    mine = []
    run(span_print_fn=lines.append)
    with et.span("a", print_fn=mine.append):
        pass
    assert not lines and len(mine) == 2


def test_printing_is_off_by_default(run, capsys):
    run()
    with et.span("quiet"):
        pass
    assert capsys.readouterr().out == ""


def test_sibling_spans_restart_the_indentation(run, lines):
    run()
    with et.span("root", print_fn=lines.append):
        with et.span("one"):
            pass
        with et.span("two"):
            pass
    assert [depth_of(line) for line in lines] == [0, 1, 1, 1, 1, 0]
