import asyncio
import sys
import time
import typing as tp
import unittest
from dataclasses import dataclass
from unittest import TestCase, mock

import cytoolz as cz
import hypothesis as hp
import pytest
from hypothesis import strategies as st

import pypeln as pl
from pypeln import utils as pypeln_utils
from pypeln.task.utils import run_test_async

MAX_EXAMPLES = 10
T = tp.TypeVar("T")
# ----------------------------------------------------------------
# queue
# ----------------------------------------------------------------


class MyException(Exception):
    pass


class TestQueue(TestCase):
    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    def test_done(self, nums):
        queue = pl.task.IterableQueue()

        async def worker():
            for i in nums:
                await queue.put(i)

            await queue.worker_done()

        processes = pl.task.start_workers(worker)

        nums_pl = list(queue)

        assert len(processes) == 1
        assert nums_pl == nums

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    @run_test_async
    async def test_done_async(self, nums):
        queue = pl.task.IterableQueue()

        async def worker():
            for i in nums:
                await queue.put(i)

            await queue.worker_done()

        processes = pl.task.start_workers(worker)

        nums_pl = [x async for x in queue]

        assert len(processes) == 1
        assert nums_pl == nums

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    @run_test_async
    async def test_get(self, nums):
        queue = pl.task.IterableQueue()

        async def worker():
            for i in nums:
                await queue.put(i)

            await queue.worker_done()

        processes = pl.task.start_workers(worker)

        if len(nums) > 0:
            x = await queue.get()
            assert x == nums[0]

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    @run_test_async
    async def test_get_2(self, nums):
        queue = pl.task.IterableQueue()

        async def worker():
            for i in nums:
                await queue.put(i)

            await queue.worker_done()

        processes = pl.task.start_workers(worker)

        await asyncio.sleep(0.01)

        if len(nums) > 0:
            x = queue._get_nowait()
            assert x == nums[0]

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    def test_done_nowait(self, nums):
        queue = pl.task.IterableQueue()

        async def worker():
            for i in nums:
                await queue.put(i)

            queue.worker_done_nowait()

        processes = pl.task.start_workers(worker)

        nums_pl = list(queue)

        assert len(processes) == 1
        assert nums_pl == nums

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    def test_done_many(self, nums):
        n_workers = 3

        queue = pl.task.IterableQueue(total_sources=n_workers)

        async def worker():
            for i in nums:
                await queue.put(i)

            await queue.worker_done()

        processes = pl.task.start_workers(worker, n_workers=n_workers)

        nums_pl = list(queue)

        assert len(processes) == n_workers
        assert len(nums_pl) == (len(nums) * 3)

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    @run_test_async
    async def test_done_many_async(self, nums):
        n_workers = 3

        queue = pl.task.IterableQueue(total_sources=n_workers)

        async def worker():
            for i in nums:
                await queue.put(i)

            await queue.worker_done()

        processes = pl.task.start_workers(worker, n_workers=n_workers)

        nums_pl = [x async for x in queue]

        assert len(processes) == n_workers
        assert len(nums_pl) == (len(nums) * 3)

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    def test_stop(self, nums):
        queue = pl.task.IterableQueue()

        async def worker():
            for i in nums:
                await queue.put(i)

            await queue.stop()

        processes = pl.task.start_workers(worker)

        nums_pl = list(queue)

        assert len(processes) == 1
        assert nums_pl == nums

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    @run_test_async
    async def test_stop_async(self, nums):
        queue = pl.task.IterableQueue()

        async def worker():
            for i in nums:
                await queue.put(i)

            await queue.stop()

        processes = pl.task.start_workers(worker)

        nums_pl = [x async for x in queue]

        assert len(processes) == 1
        assert nums_pl == nums

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    def test_stop_nowait(self, nums):
        queue = pl.task.IterableQueue()

        async def worker():
            for i in nums:
                await queue.put(i)

            queue.stop_nowait()

        processes = pl.task.start_workers(worker)

        nums_pl = list(queue)

        assert len(processes) == 1
        assert nums_pl == nums

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    def test_kill(self, nums):
        queue = pl.task.IterableQueue()

        async def worker():
            for i in nums:
                await queue.put(i)

            await queue.kill()

        processes = pl.task.start_workers(worker)

        nums_pl = list(queue)

        assert len(processes) == 1
        assert len(nums_pl) <= len(nums)

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    @run_test_async
    async def test_kill_async(self, nums):
        queue = pl.task.IterableQueue()

        async def worker():
            for i in nums:
                await queue.put(i)

            await queue.kill()

        processes = pl.task.start_workers(worker)

        nums_pl = [x async for x in queue]

        assert len(processes) == 1
        assert len(nums_pl) <= len(nums)

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    def test_kill_nowait(self, nums):
        queue = pl.task.IterableQueue()

        async def worker():
            for i in nums:
                await queue.put(i)

            queue.kill_nowait()

        processes = pl.task.start_workers(worker)

        nums_pl = list(queue)

        assert len(processes) == 1
        assert len(nums_pl) <= len(nums)

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    def test_raise(self, nums):
        queue = pl.task.IterableQueue()

        async def worker():
            try:
                raise MyException()
            except BaseException as e:
                await queue.raise_exception(e)

        processes = pl.task.start_workers(worker)

        with pytest.raises(MyException):
            nums_pl = list(queue)

        assert len(processes) == 1

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    @run_test_async
    async def test_raise_async(self, nums):
        queue = pl.task.IterableQueue()

        async def worker():
            try:
                raise MyException()
            except BaseException as e:
                await queue.raise_exception(e)

        processes = pl.task.start_workers(worker)

        with pytest.raises(MyException):
            nums_pl = [x async for x in queue]

        assert len(processes) == 1

    @hp.given(nums=st.lists(st.integers()))
    @hp.settings(max_examples=MAX_EXAMPLES)
    def test_raise_nowait(self, nums):
        queue = pl.task.IterableQueue()

        async def worker():
            try:
                raise MyException()
            except BaseException as e:
                queue.raise_exception_nowait(e)

        processes = pl.task.start_workers(worker)

        with pytest.raises(MyException):
            nums_pl = list(queue)

        assert len(processes) == 1


class TestOutputQueues(TestCase):
    def test_basic_nowait(self):
        queues: pl.task.OutputQueues[int] = pl.task.OutputQueues()
        queue: pl.task.IterableQueue[int] = pl.task.IterableQueue()

        queues.append(queue)

        queues.put_nowait(3)

        x = queue._get_nowait()

        assert isinstance(queues, list)
        assert x == 3

    @run_test_async
    async def test_basic(self):
        queues: pl.task.OutputQueues[int] = pl.task.OutputQueues()
        queue: pl.task.IterableQueue[int] = pl.task.IterableQueue()

        queues.append(queue)

        await queues.put(3)

        x = await queue.get()

        assert isinstance(queues, list)
        assert x == 3

    @run_test_async
    async def test_done(self):
        queues: pl.task.OutputQueues[int] = pl.task.OutputQueues()
        queue = pl.task.IterableQueue()

        queues.append(queue)

        await queues.worker_done()

        x = await queue.get()

        assert isinstance(x, pl.utils.Done)

    def test_done_nowait(self):
        queues: pl.task.OutputQueues[int] = pl.task.OutputQueues()
        queue = pl.task.IterableQueue()

        queues.append(queue)

        queues.worker_done_nowait()

        with pytest.raises(asyncio.QueueEmpty):
            x = queue._get_nowait()

    @run_test_async
    async def test_stop(self):
        queues: pl.task.OutputQueues[int] = pl.task.OutputQueues()
        queue = pl.task.IterableQueue()

        queues.append(queue)

        assert queue.namespace.remaining == 1

        await queues.stop()

        assert queue.namespace.remaining == 0

    def test_stop_nowait(self):
        queues: pl.task.OutputQueues[int] = pl.task.OutputQueues()
        queue = pl.task.IterableQueue()

        queues.append(queue)

        assert queue.namespace.remaining == 1

        queues.stop_nowait()

        assert queue.namespace.remaining == 0

    @run_test_async
    async def test_kill(self):
        queues: pl.task.OutputQueues[int] = pl.task.OutputQueues()
        queue = pl.task.IterableQueue()

        queues.append(queue)

        assert queue.namespace.force_stop == False

        await queues.kill()

        assert queue.namespace.remaining == True

    def test_kill_nowait(self):
        queues: pl.task.OutputQueues[int] = pl.task.OutputQueues()
        queue = pl.task.IterableQueue()

        queues.append(queue)

        assert queue.namespace.force_stop == False

        queues.kill_nowait()

        assert queue.namespace.remaining == True


class TestTaskPool(unittest.TestCase):
    @run_test_async
    async def test_basic(self):
        namespace = pl.task.Namespace(x=0)

        async def task():
            await asyncio.sleep(0.1)
            namespace.x = 1

        tasks = pl.task.TaskPool.create(workers=0)

        await tasks.put(task)

        assert namespace.x == 0

        await tasks.join()

        assert namespace.x == 1

    @run_test_async
    async def test_context(self):
        namespace = pl.task.Namespace(x=0)

        async def task():
            await asyncio.sleep(0.1)
            namespace.x = 1

        async with pl.task.TaskPool.create(workers=0) as tasks:
            await tasks.put(task)

        assert namespace.x == 1

    @run_test_async
    async def test_put_wait(self):
        timeout = 0.1
        namespace = pl.task.Namespace(x=0)

        async def task():
            await asyncio.sleep(timeout)
            namespace.x = 1

        async def no_task():
            pass

        async with pl.task.TaskPool.create(workers=1) as tasks:
            await tasks.put(task)
            assert len(tasks.tasks) == 1

            t0 = time.time()
            await tasks.put(no_task)

            assert time.time() - t0 > timeout

        assert namespace.x == 1

    @run_test_async
    async def test_put_no_wait(self):
        timeout = 0.1
        namespace = pl.task.Namespace(x=0)

        async def task():
            await asyncio.sleep(timeout)
            namespace.x = 1

        async def no_task():
            pass

        async with pl.task.TaskPool.create(workers=2) as tasks:
            await tasks.put(task)
            assert len(tasks.tasks) == 1

            t0 = time.time()
            await tasks.put(no_task)

            assert len(tasks.tasks) == 2
            assert time.time() - t0 < timeout

        assert namespace.x == 1


# ----------------------------------------------------------------
# Regression tests: failure signalling and end-of-stream semantics
# ----------------------------------------------------------------


class TestQueueFailureSemantics(TestCase):
    def test_exception_is_not_lost_when_queue_is_full(self):
        """
        Reporting a failure must not block on a full queue: with no consumer
        awake, a blocking signal would hang the producer forever.
        """
        queue = pl.task.IterableQueue(maxsize=1)
        queue.put_nowait("occupied")

        async def worker():
            try:
                raise MyException()
            except BaseException as e:
                await asyncio.wait_for(queue.raise_exception(e), timeout=1)

        processes = pl.task.start_workers(worker)
        processes[0].result(timeout=5)

        with pytest.raises(MyException):
            list(queue)

    def test_exception_is_visible_to_later_consumers(self):
        queue = pl.task.IterableQueue()

        async def worker():
            try:
                raise MyException()
            except BaseException as e:
                await queue.raise_exception(e)

        pl.task.start_workers(worker)

        with pytest.raises(MyException):
            list(queue)

        # The failure is stored, not consumed: a second reader sees it too.
        with pytest.raises(MyException):
            list(queue)

    @run_test_async
    async def test_multiple_sources_all_must_finish(self):
        """
        A queue fed by N sources must only be done after every source emitted
        its `Done`; one source finishing early cannot end the iteration.
        """
        n_sources = 3
        queue = pl.task.IterableQueue(total_sources=n_sources)

        async def worker(i):
            await asyncio.sleep(0.01 * (n_sources - i))
            await queue.put(i)
            await queue.worker_done()

        for i in range(n_sources):
            pl.task.start_workers(lambda i=i: worker(i), n_workers=1)

        result = [x async for x in queue]

        assert sorted(result) == list(range(n_sources))

    def test_consumer_wakes_cross_thread_producer(self):
        """
        A synchronous consumer draining the queue from another thread must
        wake a producer blocked on `put`, otherwise the pipeline deadlocks.

        This is the shape of `IterableQueue.__iter__`: the loop lives in a
        background thread while the consumer calls `_get_nowait` from the
        thread that created the pipeline.
        """
        queue = pl.task.IterableQueue(maxsize=1)

        async def producer():
            for i in range(5):
                await queue.put(i)

            await queue.worker_done()

        processes = pl.task.start_workers(producer)

        result = list(queue)

        assert result == list(range(5))
        processes[0].result(timeout=5)

    def test_is_done_requires_all_sources(self):
        queue = pl.task.IterableQueue(total_sources=2)
        queue.put_nowait(1)
        assert queue._get_nowait() == 1

        # One source finishes, the other still has to report.
        queue.put_nowait(pl.utils.DONE)

        with pytest.raises(asyncio.QueueEmpty):
            queue._get_nowait()

        assert not queue.is_done()

        queue.put_nowait(pl.utils.DONE)

        with pytest.raises(asyncio.QueueEmpty):
            queue._get_nowait()

        assert queue.is_done()


class TestAsyncSourceCompletion(TestCase):
    @run_test_async
    async def test_one_source_done_does_not_end_async_iteration(self):
        """
        `Done` used to be accounted twice (once in `get`, once in `__aiter__`),
        so with N sources the first `Done` could end the iteration and every
        other source's results were lost.
        """
        queue = pl.task.IterableQueue(total_sources=2)

        async def consume():
            return [x async for x in queue]

        consumer = asyncio.ensure_future(consume())

        # First source finishes.
        await queue.put(pl.utils.DONE)
        await asyncio.sleep(0.05)

        assert not consumer.done(), "iteration ended before all sources finished"

        # Second source still has data.
        await queue.put("late")
        await queue.put(pl.utils.DONE)

        result = await asyncio.wait_for(consumer, timeout=5)

        assert result == ["late"]

    @run_test_async
    async def test_remaining_counts_each_source_once(self):
        queue = pl.task.IterableQueue(total_sources=2)

        await queue.put(pl.utils.DONE)
        assert await queue.get() is pl.utils.DONE
        assert queue.namespace.remaining == 1

        await queue.put(pl.utils.DONE)
        assert await queue.get() is pl.utils.DONE
        assert queue.namespace.remaining == 0
