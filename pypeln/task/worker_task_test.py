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

MAX_EXAMPLES = 10
T = tp.TypeVar("T")


@dataclass
class CustomProcess:
    f: tp.Callable[..., tp.Awaitable]

    async def __call__(self, worker: pl.task.Worker, **kwargs):
        await self.f(worker, **kwargs)


class MyException(Exception):
    pass


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_basic(nums):
    input_queue = pl.task.IterableQueue()
    output_queue = pl.task.IterableQueue()
    output_queues = pl.task.OutputQueues([output_queue])

    async def f(worker: pl.task.Worker):
        for x in nums:
            await worker.stage_params.output_queues.put(x)

    stage_params: pl.task.StageParams = mock.Mock(
        input_queue=input_queue,
        output_queues=output_queues,
        total_workers=1,
    )

    worker = pl.task.Worker.create(
        process_fn=CustomProcess(f),
        stage_params=stage_params,
        main_queue=output_queue,
        timeout=0,
        on_start=None,
        on_done=None,
        f_args=[],
        tasks=pl.task.TaskPool.create(workers=1),
    )

    worker.start()
    assert not worker.process.cancelled()

    nums_pl = list(output_queue)

    assert nums_pl == nums

    time.sleep(0.01)
    assert worker.is_done


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
@pl.task.utils.run_test_async
async def test_basic_async(nums):
    input_queue = pl.task.IterableQueue()
    output_queue = pl.task.IterableQueue()
    output_queues = pl.task.OutputQueues([output_queue])

    async def f(worker: pl.task.Worker):
        for x in nums:
            await worker.stage_params.output_queues.put(x)

    stage_params: pl.task.StageParams = mock.Mock(
        input_queue=input_queue,
        output_queues=output_queues,
        total_workers=1,
    )

    worker = pl.task.Worker.create(
        process_fn=CustomProcess(f),
        stage_params=stage_params,
        main_queue=output_queue,
        timeout=0,
        on_start=None,
        on_done=None,
        f_args=[],
        tasks=pl.task.TaskPool.create(workers=1),
    )

    worker.start()

    nums_pl = [x async for x in output_queue]

    assert nums_pl == nums

    await asyncio.sleep(0.01)
    assert worker.is_done


def test_raises():
    input_queue = pl.task.IterableQueue()
    output_queue = pl.task.IterableQueue()
    output_queues = pl.task.OutputQueues([output_queue])

    async def f(worker: pl.task.Worker):
        raise MyException()

    stage_params: pl.task.StageParams = mock.Mock(
        input_queue=input_queue,
        output_queues=output_queues,
        total_workers=1,
    )

    worker = pl.task.Worker.create(
        process_fn=CustomProcess(f),
        stage_params=stage_params,
        main_queue=output_queue,
        timeout=0,
        on_start=None,
        on_done=None,
        f_args=[],
        tasks=pl.task.TaskPool.create(workers=1),
    )

    worker.start()

    with pytest.raises(MyException):
        nums_pl = list(output_queue)

    time.sleep(0.01)
    assert worker.is_done


@pl.task.utils.run_test_async
async def test_raises_async():
    input_queue = pl.task.IterableQueue()
    output_queue = pl.task.IterableQueue()
    output_queues = pl.task.OutputQueues([output_queue])

    async def f(worker: pl.task.Worker):
        raise MyException()

    stage_params: pl.task.StageParams = mock.Mock(
        input_queue=input_queue,
        output_queues=output_queues,
        total_workers=1,
    )

    worker = pl.task.Worker(
        process_fn=CustomProcess(f),
        stage_params=stage_params,
        main_queue=output_queue,
        timeout=0,
        on_start=None,
        on_done=None,
        f_args=[],
        tasks=pl.task.TaskPool.create(workers=1),
    )

    worker.start()

    with pytest.raises(MyException):
        nums_pl = [x async for x in output_queue]

    await asyncio.sleep(0.01)
    assert worker.is_done


def test_timeout_base():
    input_queue = pl.task.IterableQueue()
    output_queue = pl.task.IterableQueue()
    output_queues = pl.task.OutputQueues([output_queue])
    namespace = pl.task.Namespace(x=0)

    async def f(worker: pl.task.Worker):
        async def task():
            for i in range(1000):
                await asyncio.sleep(0.01)
            namespace.x = 1

        await worker.tasks.put(task)

    stage_params: pl.task.StageParams = mock.Mock(
        input_queue=input_queue,
        output_queues=output_queues,
        total_workers=1,
    )

    timeout = 0.001

    worker = pl.task.Worker.create(
        process_fn=CustomProcess(f),
        stage_params=stage_params,
        main_queue=output_queue,
        timeout=timeout,
        on_start=None,
        on_done=None,
        f_args=[],
        tasks=pl.task.TaskPool.create(workers=1, timeout=timeout),
    )

    worker.start()

    # wait for worker to start
    time.sleep(0.02)

    while len(worker.tasks.tasks) > 0:
        time.sleep(0.001)

    assert namespace.x == 0

    time.sleep(0.1)

    assert worker.is_done


@pl.task.utils.run_test_async
async def test_timeout_async():
    input_queue = pl.task.IterableQueue()
    output_queue = pl.task.IterableQueue()
    output_queues = pl.task.OutputQueues([output_queue])
    namespace = pl.task.Namespace(x=0)

    async def f(worker: pl.task.Worker):
        async def task():
            await asyncio.sleep(0.01)
            namespace.x = 1

        await worker.tasks.put(task)

    stage_params: pl.task.StageParams = mock.Mock(
        input_queue=input_queue,
        output_queues=output_queues,
        total_workers=1,
    )

    timeout = 0.001

    worker = pl.task.Worker.create(
        process_fn=CustomProcess(f),
        stage_params=stage_params,
        main_queue=output_queue,
        timeout=timeout,
        on_start=None,
        on_done=None,
        f_args=[],
        tasks=pl.task.TaskPool.create(
            workers=1,
            timeout=timeout,
        ),
    )

    # wait for worker to start
    await asyncio.sleep(0.02)

    worker.start()

    while len(worker.tasks.tasks) > 0:
        await asyncio.sleep(0.01)

    assert namespace.x == 0

    await asyncio.sleep(0.01)

    assert worker.is_done


def test_no_timeout():
    input_queue = pl.task.IterableQueue()
    output_queue = pl.task.IterableQueue()
    output_queues = pl.task.OutputQueues([output_queue])
    namespace = pl.task.Namespace(x=0)

    async def f(worker: pl.task.Worker):
        async def task():
            await asyncio.sleep(0.01)
            namespace.x = 1

        await worker.tasks.put(task)

    stage_params: pl.task.StageParams = mock.Mock(
        input_queue=input_queue,
        output_queues=output_queues,
        total_workers=1,
    )

    timeout = 0.0

    worker = pl.task.Worker.create(
        process_fn=CustomProcess(f),
        stage_params=stage_params,
        main_queue=output_queue,
        timeout=timeout,
        on_start=None,
        on_done=None,
        f_args=[],
        tasks=pl.task.TaskPool.create(workers=0, timeout=timeout),
    )
    worker.start()

    # wait for worker to start
    time.sleep(0.02)

    while len(worker.tasks.tasks) > 0:
        time.sleep(0.001)

    assert namespace.x == 1

    assert worker.is_done


@pl.task.utils.run_test_async
async def test_no_timeout_async():
    input_queue = pl.task.IterableQueue()
    output_queue = pl.task.IterableQueue()
    output_queues = pl.task.OutputQueues([output_queue])
    namespace = pl.task.Namespace(x=0)

    async def f(worker: pl.task.Worker):
        async def task():
            await asyncio.sleep(0.01)
            namespace.x = 1

        await worker.tasks.put(task)

    timeout = 0.02

    stage_params: pl.task.StageParams = mock.Mock(
        input_queue=input_queue,
        output_queues=output_queues,
        total_workers=1,
    )
    worker = pl.task.Worker.create(
        process_fn=CustomProcess(f),
        stage_params=stage_params,
        main_queue=output_queue,
        timeout=timeout,
        on_start=None,
        on_done=None,
        f_args=[],
        tasks=pl.task.TaskPool.create(workers=0, timeout=timeout),
    )
    worker.start()

    # wait for worker to start
    await asyncio.sleep(0.02)

    await worker.tasks.join()

    assert namespace.x == 1

    assert worker.is_done


def test_del1():
    input_queue = pl.task.IterableQueue()
    output_queue = pl.task.IterableQueue()
    output_queues = pl.task.OutputQueues([output_queue])

    async def f(worker: pl.task.Worker):
        for _ in range(1000):
            await asyncio.sleep(0.01)

    stage_params: pl.task.StageParams = mock.Mock(
        input_queue=input_queue,
        output_queues=output_queues,
        total_workers=1,
    )

    worker = pl.task.Worker.create(
        process_fn=CustomProcess(f),
        stage_params=stage_params,
        main_queue=output_queue,
        timeout=0,
        on_start=None,
        on_done=None,
        f_args=[],
        tasks=pl.task.TaskPool.create(workers=1),
    )

    worker.start()
    task = worker.process

    time.sleep(0.01)

    worker.stop()
    time.sleep(0.2)

    assert task.cancelled()
    assert worker.is_done


@pl.task.utils.run_test_async
async def test_del1_async():
    input_queue = pl.task.IterableQueue()
    output_queue = pl.task.IterableQueue()
    output_queues = pl.task.OutputQueues([output_queue])

    async def f(worker: pl.task.Worker):
        for _ in range(1000):
            await asyncio.sleep(0.01)

    stage_params: pl.task.StageParams = mock.Mock(
        input_queue=input_queue,
        output_queues=output_queues,
        total_workers=1,
    )

    worker = pl.task.Worker.create(
        process_fn=CustomProcess(f),
        stage_params=stage_params,
        main_queue=output_queue,
        timeout=0,
        on_start=None,
        on_done=None,
        f_args=[],
        tasks=pl.task.TaskPool.create(workers=0),
    )

    worker.start()
    task = worker.process

    await asyncio.sleep(0.01)

    worker.stop()
    await asyncio.sleep(0.2)

    assert task.cancelled()
    assert worker.is_done


def test_del3():
    def start_worker():
        input_queue = pl.task.IterableQueue()
        output_queue = pl.task.IterableQueue()
        output_queues = pl.task.OutputQueues([output_queue])

        async def f(worker: pl.task.Worker):
            for _ in range(1000):
                await asyncio.sleep(0.01)

        stage_params: pl.task.StageParams = mock.Mock(
            input_queue=input_queue,
            output_queues=output_queues,
            total_workers=1,
        )

        stage_params: pl.task.StageParams = mock.Mock(
            input_queue=input_queue,
            output_queues=output_queues,
            total_workers=1,
        )

        worker = pl.task.Worker.create(
            process_fn=CustomProcess(f),
            stage_params=stage_params,
            main_queue=output_queue,
            timeout=0,
            on_start=None,
            on_done=None,
            f_args=[],
            tasks=pl.task.TaskPool.create(workers=1),
        )
        worker.start()

        time.sleep(0.01)

        assert not worker.process.cancelled()

        return worker, worker.process

    worker, task = start_worker()

    assert not task.cancelled()
    assert not worker.is_done


@pl.task.utils.run_test_async
async def test_del3_async():
    async def start_worker():
        input_queue = pl.task.IterableQueue()
        output_queue = pl.task.IterableQueue()
        output_queues = pl.task.OutputQueues([output_queue])

        async def f(worker: pl.task.Worker):
            for _ in range(1000):
                await asyncio.sleep(0.01)

        stage_params: pl.task.StageParams = mock.Mock(
            input_queue=input_queue,
            output_queues=output_queues,
            total_workers=1,
        )

        worker = pl.task.Worker.create(
            process_fn=CustomProcess(f),
            stage_params=stage_params,
            main_queue=output_queue,
            timeout=0,
            on_start=None,
            on_done=None,
            f_args=[],
            tasks=pl.task.TaskPool.create(workers=1),
        )
        worker.start()

        await asyncio.sleep(0.01)

        assert not worker.process.cancelled()

        return worker, worker.process

    worker, task = await start_worker()

    await asyncio.sleep(0.01)

    assert not task.cancelled()
    assert not worker.is_done


# ----------------------------------------------------------------
# Regression tests: failures must never be silently dropped
# ----------------------------------------------------------------


class SilentFailure(Exception):
    pass


async def _fail_on_zero(x):
    if x == 0:
        raise SilentFailure()

    await asyncio.sleep(0.1)
    return x


def _run_in_thread(fn, timeout=20):
    """
    Run `fn` in a daemon thread so a hang fails the test instead of blocking
    the whole test session.
    """
    box = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as e:
            box["exception"] = e

    import threading

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)

    assert not thread.is_alive(), "pipeline hung"

    if "exception" in box:
        raise box["exception"]

    return box.get("result")


def test_task_failure_not_swallowed_sync():
    """
    A task that fails early (completing before every other task) must still
    surface its exception to the consumer.
    """
    with pytest.raises(SilentFailure):
        _run_in_thread(lambda: list(pl.task.map(_fail_on_zero, range(30), workers=2)))


def test_task_failure_not_swallowed_async():
    async def main():
        return await pl.task.map(_fail_on_zero, range(30), workers=2)

    with pytest.raises(SilentFailure):
        _run_in_thread(lambda: asyncio.run(main()))


@pytest.mark.parametrize("api", ["map", "filter", "flat_map", "each"])
def test_task_failure_not_swallowed_all_apis(api):
    async def flat_map_fn(x):
        if x == 0:
            raise SilentFailure()

        await asyncio.sleep(0.1)
        return [x]

    async def call():
        if api == "map":
            return await pl.task.map(_fail_on_zero, range(30), workers=4)
        elif api == "filter":
            return await pl.task.filter(_fail_on_zero, range(30), workers=4)
        elif api == "flat_map":
            return await pl.task.flat_map(flat_map_fn, range(30), workers=4)
        else:
            return await pl.task.each(_fail_on_zero, range(30), workers=4)

    with pytest.raises(SilentFailure):
        _run_in_thread(lambda: asyncio.run(call()))


def test_task_failure_not_swallowed_run():
    with pytest.raises(SilentFailure):
        _run_in_thread(
            lambda: pl.task.run(
                pl.task.each(_fail_on_zero, range(30), workers=2)
            )
        )


def test_task_timeout_still_skips():
    """
    Timeouts are not failures: the element is skipped and the run succeeds.
    """

    async def f(x):
        if x == 2:
            await asyncio.sleep(10)

        return x

    result = _run_in_thread(
        lambda: sorted(list(pl.task.map(f, range(5), workers=1, timeout=0.05)))
    )

    assert result == [0, 1, 3, 4]
