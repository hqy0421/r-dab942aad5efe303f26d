import abc
import asyncio
import inspect
import time
import typing as tp
from concurrent.futures import Future
from copy import copy
from dataclasses import dataclass, field

from pypeln import utils as pypeln_utils

from . import utils
from .queue import IterableQueue, OutputQueues

WorkerConstructor = tp.Callable[["StageParams", IterableQueue], "Worker"]
Kwargs = tp.Dict[str, tp.Any]
T = tp.TypeVar("T")


class ProcessFn(pypeln_utils.Protocol):
    async def __call__(self, worker: "Worker", **kwargs):
        ...


class StageParams(tp.NamedTuple):
    input_queue: IterableQueue
    output_queues: OutputQueues
    namespace: utils.Namespace

    @classmethod
    def create(
        cls, input_queue: IterableQueue, output_queues: OutputQueues
    ) -> "StageParams":
        return cls(
            namespace=utils.Namespace(active_workers=1),
            input_queue=input_queue,
            output_queues=output_queues,
        )

    def worker_done(self):
        self.namespace.active_workers -= 1


class WorkerInfo(tp.NamedTuple):
    index: int


@dataclass
class Worker(tp.Generic[T]):
    process_fn: ProcessFn
    timeout: float
    stage_params: StageParams
    main_queue: IterableQueue
    on_start: tp.Optional[tp.Callable[..., tp.Union[Kwargs, tp.Awaitable[Kwargs]]]]
    on_done: tp.Optional[tp.Callable[..., tp.Union[Kwargs, tp.Awaitable[Kwargs]]]]
    f_args: tp.List[str]
    tasks: "TaskPool"
    process: tp.Optional[Future] = None
    is_done: bool = False
    closed_event: tp.Optional[asyncio.Event] = None

    @classmethod
    def create(
        cls,
        process_fn: ProcessFn,
        timeout: float,
        stage_params: StageParams,
        main_queue: IterableQueue,
        on_start: tp.Optional[tp.Callable[..., tp.Union[Kwargs, tp.Awaitable[Kwargs]]]],
        on_done: tp.Optional[tp.Callable[..., tp.Union[Kwargs, tp.Awaitable[Kwargs]]]],
        f_args: tp.List[str],
        tasks: "TaskPool",
        process: tp.Optional[Future] = None,
        is_done: bool = False,
    ) -> "Worker":
        worker = cls(
            process_fn=process_fn,
            timeout=timeout,
            stage_params=stage_params,
            main_queue=main_queue,
            on_start=on_start,
            on_done=on_done,
            f_args=f_args,
            process=process,
            tasks=tasks,
        )

        assert worker.timeout == tasks.timeout

        return worker

    async def __call__(self):
        worker_info = WorkerInfo(index=0)

        # Signals any helper (e.g. the source producer thread) that this
        # worker is gone and no more work should be produced.
        if self.closed_event is None:
            self.closed_event = asyncio.Event()

        # Any failure of a real task (timeouts excluded) must reach the final
        # consumer even if the task finished before the consumer asked for it.
        self.tasks.exception_handler = self.main_queue.raise_exception_nowait

        on_start_args: tp.List[str] = (
            pypeln_utils.function_args(self.on_start) if self.on_start else []
        )
        on_done_args: tp.List[str] = (
            pypeln_utils.function_args(self.on_done) if self.on_done else []
        )

        try:
            if self.on_start is not None:
                on_start_kwargs = dict(worker_info=worker_info)
                kwargs = self.on_start(
                    **{
                        key: value
                        for key, value in on_start_kwargs.items()
                        if key in on_start_args
                    }
                )
                if isinstance(kwargs, tp.Awaitable):
                    kwargs = await kwargs
            else:
                kwargs = {}

            if kwargs is None:
                kwargs = {}

            kwargs.setdefault("worker_info", worker_info)

            async with self.tasks:
                await self.process_fn(
                    self,
                    **{
                        key: value
                        for key, value in kwargs.items()
                        if key in self.f_args
                    },
                )

            self.stage_params.worker_done()

            if self.on_done is not None:
                kwargs.setdefault(
                    "stage_status",
                    StageStatus(),
                )

                coro = self.on_done(
                    **{
                        key: value
                        for key, value in kwargs.items()
                        if key in on_done_args
                    }
                )

                if isinstance(coro, tp.Awaitable):
                    await coro

            await self.stage_params.output_queues.worker_done()

        except asyncio.CancelledError:
            pass
        except TaskPoolClosed:
            # The pipeline was closed (failure or early stop) while this
            # worker was submitting elements: exit quietly, the original
            # cause is already being reported.
            pass
        except BaseException as e:
            # Use the non-blocking variant: a full queue with no consumer must
            # never prevent the worker from shutting down.
            self.main_queue.raise_exception_nowait(e)
        finally:
            self.is_done = True

            if self.closed_event is not None:
                self.closed_event.set()

            await self.tasks.shutdown()

    def start(self):
        [self.process] = start_workers(self)

    def stop(self):
        if self.process is None:
            return

        def cancel():
            self.process.cancel()
            # self.is_done = True

        self.tasks.stop()

        if self.closed_event is not None:
            utils.run_function_in_loop(self.closed_event.set)

        utils.run_function_in_loop(cancel)


class Applicable(pypeln_utils.Protocol):
    def apply(self, worker: "Worker", elem: tp.Any, **kwargs):
        ...


class ApplyProcess(ProcessFn, Applicable):
    async def __call__(self, worker: Worker, **kwargs):
        async for elem in worker.stage_params.input_queue:
            await worker.tasks.put(
                pypeln_utils.get_callable(self.apply, worker, elem, **kwargs)
            )


class StageStatus(tp.NamedTuple):
    """
    Object passed to various `on_done` callbacks. It contains information about the stage in case book keeping is needed.
    """

    @property
    def done(self) -> bool:
        """
        `bool` : `True` if all workers finished.
        """
        return True

    @property
    def active_workers(self):
        """
        `int` : Number of active workers.
        """
        return 0

    def __str__(self):
        return (
            f"StageStatus(done = {self.done}, active_workers = {self.active_workers})"
        )


class TaskPoolClosed(RuntimeError):
    """
    Raised when an element is submitted to a `TaskPool` that has already been
    closed, e.g. because the pipeline failed or was closed early.
    """


@dataclass
class TaskPool:
    semaphore: tp.Optional[asyncio.Semaphore]
    tasks: tp.Set[asyncio.Task]
    timeout: float
    closed: bool
    exception_handler: tp.Optional[tp.Callable[[BaseException], tp.Any]] = None
    waiting: int = 0

    @classmethod
    def create(cls, workers: int, timeout: float = 0):
        return cls(
            semaphore=asyncio.Semaphore(workers) if workers else None,
            tasks=set(),
            timeout=timeout,
            closed=False,
        )

    async def put(self, coro_f: tp.Callable[[], tp.Awaitable]):
        if self.closed:
            raise TaskPoolClosed("Trying put items into a closed TaskPool")

        if self.semaphore:
            acquired = await self._acquire()

            if not acquired:
                raise TaskPoolClosed("Trying put items into a closed TaskPool")

            if self.closed:
                self.semaphore.release()
                raise TaskPoolClosed("Trying put items into a closed TaskPool")

        if self.closed:
            if self.semaphore:
                self.semaphore.release()

            raise TaskPoolClosed("Trying put items into a closed TaskPool")

        task = asyncio.ensure_future(self.get_task(coro_f))

        self.tasks.add(task)

        task.add_done_callback(self.on_task_done)

    async def _acquire(self) -> bool:
        """
        Acquire a slot, or give up as soon as the pool is closed. Waiting
        forever would leak a pending `put` when the pipeline fails or is closed
        early.
        """
        if self.closed:
            return False

        self.waiting += 1

        try:
            await self.semaphore.acquire()
        finally:
            self.waiting -= 1

        return True

    async def get_task(self, coro_f: tp.Callable[[], tp.Awaitable]):
        coro = coro_f()

        if self.timeout:
            coro = asyncio.wait_for(coro, timeout=self.timeout)

        try:
            await coro
        except asyncio.TimeoutError:
            # Timeouts skip the element, they are not pipeline failures.
            pass

    def on_task_done(self, task: asyncio.Task):
        self.tasks.discard(task)

        if self.semaphore:
            self.semaphore.release()

        self.report_exception(task)

    def report_exception(self, task: asyncio.Task):
        """
        Forward failures even when the task finished before anyone waited for
        it, otherwise a failed element could silently vanish from the output.
        """
        if self.exception_handler is None or task.cancelled():
            return

        exception = task.exception()

        if exception is None or isinstance(exception, asyncio.CancelledError):
            return

        self.exception_handler(exception)

    def _wake_waiters(self):
        if not self.semaphore:
            return

        for _ in range(self.waiting):
            try:
                self.semaphore.release()
            except ValueError:
                pass

    def cancel(self):
        """
        Cancel every task owned by the pool and stop accepting new ones.
        Safe to call from any thread.
        """
        self.closed = True

        utils.run_function_in_loop(self._wake_waiters)

        for task in list(self.tasks):
            utils.run_function_in_loop(task.cancel)

    # Kept for backwards compatibility.
    def stop(self):
        self.cancel()

    async def join(self):
        """
        Wait until every task spawned so far has really finished.
        """
        try:
            while self.tasks:
                tasks = list(self.tasks)
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for task, result in zip(tasks, results):
                    if isinstance(result, BaseException):
                        self.report_exception(task)
        finally:
            self.closed = True

    async def shutdown(self):
        """
        Cancel and collect every task owned by the pool. Never blocks on a
        full queue because tasks are cancelled, not awaited to completion.
        """
        self.closed = True

        self._wake_waiters()

        tasks = list(self.tasks)

        for task in tasks:
            if not task.done():
                task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.tasks.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.join()

    def __len__(self) -> int:
        return len(self.tasks)


# ----------------------------------------------------------------
# create_daemon_workers
# ----------------------------------------------------------------


def start_workers(
    target: tp.Callable,
    n_workers: int = 1,
    args: tp.Tuple[tp.Any, ...] = tuple(),
    kwargs: tp.Optional[tp.Dict[tp.Any, tp.Any]] = None,
    use_threads: bool = False,
) -> tp.List[Future]:
    if kwargs is None:
        kwargs = {}

    workers: tp.List[Future] = []

    for _ in range(n_workers):
        t = utils.run_coroutine_in_loop(
            pypeln_utils.get_callable(target, *args, **kwargs)
        )
        workers.append(t)

    return workers
