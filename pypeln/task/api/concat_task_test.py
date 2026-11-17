import time
import typing as tp

import hypothesis as hp
from flaky import flaky
from hypothesis import strategies as st

import pypeln as pl

MAX_EXAMPLES = 10
T = tp.TypeVar("T")


@flaky(max_runs=10, min_passes=1)
@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_concat_basic(nums: tp.List[int]):
    time.sleep(0.1)
    nums_py = list(map(lambda x: x + 1, nums))
    nums_py1 = list(map(lambda x: x**2, nums_py))
    nums_py2 = list(map(lambda x: -x, nums_py))
    nums_py = nums_py1 + nums_py2

    nums_pl = pl.task.map(lambda x: x + 1, nums)
    nums_pl1 = pl.task.map(lambda x: x**2, nums_pl)
    nums_pl2 = pl.task.map(lambda x: -x, nums_pl)
    nums_pl = pl.task.concat([nums_pl1, nums_pl2])

    time.sleep(0.1)

    sorted_pl = sorted(nums_pl)
    sorted_py = sorted(nums_py)

    time.sleep(0.1)

    assert sorted_pl == sorted_py


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
@pl.task.utils.run_test_async
async def test_concat_basic_2(nums: tp.List[int]):
    nums_py = list(map(lambda x: x + 1, nums))
    nums_py1 = list(map(lambda x: x**2, nums_py))
    nums_py2 = list(map(lambda x: -x, nums_py))
    nums_py = nums_py1 + nums_py2

    nums_pl = pl.task.map(lambda x: x + 1, nums)
    nums_pl1 = pl.task.map(lambda x: x**2, nums_pl)
    nums_pl2 = pl.task.map(lambda x: -x, nums_pl)
    nums_pl = await pl.task.concat([nums_pl1, nums_pl2])

    assert sorted(nums_pl) == sorted(nums_py)


# @hp.given(nums=st.lists(st.integers()))
# @hp.settings(max_examples=MAX_EXAMPLES)
def test_concat_multiple(nums: tp.List[int] = [1, 2, 3]):
    nums_py = [x + 1 for x in nums]
    nums_py1 = nums_py + nums_py
    nums_py2 = nums_py1 + nums_py

    nums_pl = pl.task.map(lambda x: x + 1, nums)
    nums_pl1 = pl.task.concat([nums_pl, nums_pl])
    nums_pl2 = pl.task.concat([nums_pl1, nums_pl])

    # assert sorted(nums_py1) == sorted(list(nums_pl1))
    assert sorted(nums_py2) == sorted(list(nums_pl2))


@flaky(max_runs=3, min_passes=1)
@pl.task.utils.run_test_async
async def test_concat_multiple_2(nums: tp.List[int] = [1, 2, 3]):
    nums_py = [x + 1 for x in nums]
    nums_py1 = nums_py + nums_py
    nums_py2 = nums_py1 + nums_py

    nums_pl = pl.task.map(lambda x: x + 1, nums)
    nums_pl1 = pl.task.concat([nums_pl, nums_pl])
    nums_pl2 = await pl.task.concat([nums_pl1, nums_pl])

    # assert sorted(nums_py1) == sorted(list(nums_pl1))
    assert sorted(nums_py2) == sorted(list(nums_pl2))


# ----------------------------------------------------------------
# Regression tests: one source finishing must not end the others
# ----------------------------------------------------------------

import asyncio
import threading

import pytest


def _run_in_thread(fn, timeout=30):
    box = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as e:
            box["exception"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)

    assert not thread.is_alive(), "pipeline hung"

    if "exception" in box:
        raise box["exception"]

    return box.get("result")


async def _slow_identity(x):
    await asyncio.sleep(0.02)
    return x


def test_concat_first_source_finishes_early():
    """
    When the first source drains immediately the concat worker must keep
    consuming the remaining sources until they are done too.
    """
    fast = pl.task.map(lambda x: x, [100])
    slow_a = pl.task.map(_slow_identity, list(range(20)))
    slow_b = pl.task.map(_slow_identity, list(range(20, 40)))

    result = _run_in_thread(
        lambda: sorted(list(pl.task.concat([fast, slow_a, slow_b])))
    )

    expected = sorted([100] + list(range(20)) + list(range(20, 40)))

    assert result == expected


def test_concat_first_source_finishes_early_async():
    async def main():
        fast = pl.task.map(lambda x: x, [100])
        slow_a = pl.task.map(_slow_identity, list(range(20)))
        slow_b = pl.task.map(_slow_identity, list(range(20, 40)))

        return sorted([x async for x in pl.task.concat([fast, slow_a, slow_b])])

    result = _run_in_thread(lambda: asyncio.run(main()))
    expected = sorted([100] + list(range(20)) + list(range(20, 40)))

    assert result == expected


def test_concat_empty_source():
    empty = pl.task.map(lambda x: x, [])
    other = pl.task.map(lambda x: x, list(range(10)))

    result = _run_in_thread(lambda: sorted(list(pl.task.concat([empty, other]))))

    assert result == list(range(10))


def test_concat_failure_surfaces():
    class Error(Exception):
        pass

    def boom(x):
        raise Error()

    failing = pl.task.map(boom, [1])
    healthy = pl.task.map(_slow_identity, list(range(10)))

    with pytest.raises(Error):
        _run_in_thread(lambda: list(pl.task.concat([failing, healthy])))


def test_run_multiple_stages():
    """
    `run` merges its stages through `concat`, so a fast stage finishing first
    must not truncate the slow one.
    """
    seen = []
    lock = threading.Lock()

    async def slow(x):
        await asyncio.sleep(0.02)
        with lock:
            seen.append(x)

    fast_1 = pl.task.each(lambda x: None, [1, 2, 3])
    slow_2 = pl.task.each(slow, [4, 5, 6])

    _run_in_thread(lambda: pl.task.run(fast_1, slow_2, maxsize=1))

    assert sorted(seen) == [4, 5, 6]
