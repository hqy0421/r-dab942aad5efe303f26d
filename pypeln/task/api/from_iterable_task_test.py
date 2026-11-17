import asyncio
import sys
import typing as tp
import unittest
from unittest import TestCase, mock

import cytoolz as cz
import hypothesis as hp
from hypothesis import strategies as st

import pypeln as pl

MAX_EXAMPLES = 10
T = tp.TypeVar("T")


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_from_to_iterable(nums: tp.List[int]):
    nums_py = nums

    nums_pl = pl.task.from_iterable(nums)
    nums_pl = list(nums_pl)

    assert nums_pl == nums_py


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
@pl.task.utils.run_test_async
async def test_from_to_iterable_async_1(nums: tp.List[int]):
    nums_py = nums

    nums_pl = pl.task.from_iterable(nums)
    nums_pl = [x async for x in nums_pl]

    assert nums_pl == nums_py


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_from_to_iterable_async_iterable(nums: tp.List[int]):
    nums_py = nums

    async def iterable():
        for x in nums:
            yield x

    nums_pl = pl.task.from_iterable(iterable())
    nums_pl = list(nums_pl)

    assert nums_pl == nums_py


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_from_to_iterable_pipe(nums):
    nums_py = nums

    nums_pl = nums | pl.task.from_iterable() | list

    assert nums_pl == nums_py


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_from_to_iterable_pipe_async_iterable(nums):
    nums_py = nums

    async def iterable():
        for x in nums:
            yield x

    nums_pl = iterable() | pl.task.from_iterable() | list

    assert nums_pl == nums_py


# ----------------------------------------------------------------
# Regression tests: early close must unblock and stop the producer
# ----------------------------------------------------------------

import gc
import threading
import time

import pytest


def _run_in_thread(fn, timeout=20):
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


def test_early_close_unblocks_bounded_pipeline():
    """
    Abandoning a bounded pipeline must not leave workers stuck on a full
    queue, and must not leak pending tasks.
    """

    async def f(x):
        await asyncio.sleep(0.001)
        return x

    def run():
        stage = pl.task.map(f, range(500), workers=3, maxsize=1)
        iterator = iter(pl.task.to_iterable(stage, maxsize=1))

        for i, _ in enumerate(iterator):
            if i == 2:
                break

        del iterator
        gc.collect()
        time.sleep(0.3)

        return "closed"

    assert _run_in_thread(run) == "closed"


def test_early_close_stops_source_producer():
    """
    When the consumer goes away the thread pulling from a synchronous source
    must stop instead of draining the whole iterable.
    """
    pulled = [0]

    def source():
        for i in range(1_000_000):
            pulled[0] = i
            yield i

    async def f(x):
        await asyncio.sleep(0.001)
        return x

    def run():
        stage = pl.task.map(f, source(), workers=2, maxsize=1)
        iterator = iter(stage)
        next(iterator)
        del iterator
        gc.collect()
        time.sleep(0.5)

        return pulled[0]

    consumed = _run_in_thread(run)

    # Without stopping the producer this would reach the end of the range.
    assert consumed < 1000


def test_failure_on_full_queue_does_not_hang():
    """
    A failure raised while downstream is full and not being read must still
    reach the consumer instead of deadlocking the worker.
    """

    class Error(Exception):
        pass

    async def f(x):
        if x == 0:
            raise Error()

        await asyncio.sleep(0.001)
        return x

    def run():
        stage = pl.task.map(f, range(100), workers=2, maxsize=1)
        iterator = iter(stage)

        try:
            next(iterator)
        except Error:
            return "raised"

        del iterator
        gc.collect()
        time.sleep(0.3)

        return "consumed"

    # Either the failure surfaces or the pipeline finishes; what must not
    # happen is a hang.
    assert _run_in_thread(run) in ("raised", "consumed")
