import asyncio
import sys
import time
import traceback
import typing as tp

from pypeln import utils as pypeln_utils

from . import utils

T = tp.TypeVar("T")


PipelineException = pypeln_utils.PipelineException


class IterableQueue(asyncio.Queue, tp.Generic[T], tp.Iterable[T]):
    """
    An `asyncio.Queue` that knows when all of its sources are done and that can
    carry a failure to its consumers.

    A failure is stored (not consumed) on the `namespace` so every consumer,
    current or future, sees it. Signalling consumers never blocks, even when
    the queue is full: blocking here would deadlock a producer that is trying
    to report an error while the consumer is gone.
    """

    def __init__(self, maxsize: int = 0, total_sources: int = 1):
        super().__init__(maxsize=maxsize)

        self.namespace = utils.Namespace(
            remaining=total_sources, exception=False, force_stop=False
        )
        # Kept for backwards compatibility, it holds the first exception.
        self.exception_queue: asyncio.Queue[PipelineException] = asyncio.Queue()

    # ----------------------------------------------------------------
    # exception handling
    # ----------------------------------------------------------------

    def _get_pipeline_exception(self) -> tp.Optional[PipelineException]:
        exception = self.namespace.exception

        if isinstance(exception, PipelineException):
            return exception

        try:
            return self.exception_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def _raise_pipeline_exception(self):
        pipeline_exception = self._get_pipeline_exception()

        if pipeline_exception is None:
            return

        exception, trace = pipeline_exception

        try:
            exception = exception(f"\n\n{trace}")
        except BaseException:
            exception = Exception(f"\n\nOriginal: {exception}\n\n{trace}")

        raise exception

    def _signal(self, value: tp.Any):
        """
        Wake up blocked consumers without ever blocking the caller.
        """
        waiters = list(self._getters)

        try:
            self.put_nowait(value)
        except asyncio.QueueFull:
            # A full queue means no consumer is blocked waiting for an item,
            # so there is nobody to wake up.
            pass

        if waiters:
            self._wake_loops(waiters)

    @staticmethod
    def _wake_loops(waiters: tp.Iterable[tp.Any]):
        """
        A synchronous consumer / producer may free or fill a slot from a thread
        that is not the one running the event loop. `asyncio.Queue` schedules
        the woken waiter's callback on that loop but does not wake the
        selector, so a loop with nothing else to do can sleep forever. Nudge
        every loop that has a waiter.
        """
        loops = set()

        for waiter in waiters:
            loop = getattr(waiter, "get_loop", None)

            if loop is not None:
                loop = loop()
            else:
                loop = getattr(waiter, "_loop", None)

            if loop is not None:
                loops.add(loop)

        for loop in loops:
            if loop.is_closed():
                continue

            try:
                loop.call_soon_threadsafe(lambda: None)
            except RuntimeError:
                pass

    def _get_nowait_unlocked(self) -> tp.Any:
        """
        Like `get_nowait` but wakes the loop of the producer it unblocks.
        """
        putters = list(self._putters)

        x = super().get_nowait()

        if putters:
            self._wake_loops(putters)

        return x

    # ----------------------------------------------------------------
    # get / iteration
    # ----------------------------------------------------------------

    async def get(self) -> tp.Awaitable[T]:
        while True:
            if self.namespace.exception:
                self._raise_pipeline_exception()

            x = await super().get()

            if self.namespace.exception:
                self._raise_pipeline_exception()

            if isinstance(x, pypeln_utils.Done):
                self.namespace.remaining -= 1

            return x

    def _get_nowait(self) -> T:
        while True:
            if self.namespace.exception:
                self._raise_pipeline_exception()

            x = self._get_nowait_unlocked()

            if isinstance(x, pypeln_utils.Continue):
                continue
            elif isinstance(x, pypeln_utils.Done):
                self.namespace.remaining -= 1
                continue

            return x

    def __iter__(self) -> tp.Iterator[T]:
        while True:
            if self.namespace.exception:
                self._raise_pipeline_exception()

            if self.is_done():
                if self.namespace.exception:
                    self._raise_pipeline_exception()
                return

            try:
                x = self._get_nowait()
            except asyncio.QueueEmpty:
                time.sleep(pypeln_utils.TIMEOUT)
                continue

            yield x

    async def __aiter__(self) -> tp.AsyncIterator[T]:
        while True:
            if self.namespace.exception:
                self._raise_pipeline_exception()

            if self.is_done():
                if self.namespace.exception:
                    self._raise_pipeline_exception()
                return

            x = await self.get()

            if isinstance(x, pypeln_utils.Continue):
                continue
            elif isinstance(x, pypeln_utils.Done):
                # `get` already accounted for this source.
                continue

            yield x

    def is_done(self):
        return self.namespace.force_stop or (
            self.namespace.remaining <= 0 and self.empty()
        )

    # ----------------------------------------------------------------
    # lifecycle
    # ----------------------------------------------------------------

    async def worker_done(self):
        await self.put(pypeln_utils.DONE)

    def worker_done_nowait(self):
        self.put_nowait(pypeln_utils.DONE)

    async def stop(self):
        self.namespace.remaining = 0
        self._signal(pypeln_utils.CONTINUE)

    def stop_nowait(self):
        self.namespace.remaining = 0
        self._signal(pypeln_utils.CONTINUE)

    async def kill(self):
        self.namespace.force_stop = True
        self._signal(pypeln_utils.CONTINUE)

    def kill_nowait(self):
        self.namespace.force_stop = True
        self._signal(pypeln_utils.CONTINUE)

    async def raise_exception(self, exception: BaseException):
        self.raise_exception_nowait(exception)

    def raise_exception_nowait(self, exception: BaseException):
        pipeline_exception = self.get_pipeline_exception(exception)

        if not self.namespace.exception:
            self.namespace.exception = pipeline_exception

            try:
                self.exception_queue.put_nowait(pipeline_exception)
            except asyncio.QueueFull:
                pass

        self._signal(pypeln_utils.CONTINUE)

    def get_pipeline_exception(self, exception: BaseException) -> PipelineException:
        if isinstance(exception, PipelineException):
            return exception

        if isinstance(exception, BaseException):
            exception_type = type(exception)
            traceback_object = exception.__traceback__
            args = getattr(exception, "args", ())
        else:
            exception_type, _exception, traceback_object = sys.exc_info()
            exception_type = exception_type or type(exception)
            args = getattr(exception, "args", ())

        trace = "".join(
            traceback.format_exception(exception_type, exception, traceback_object)
        )

        trace = f"{args}\n\n{trace}"

        return PipelineException(exception_type, trace)


class OutputQueues(tp.List[IterableQueue[T]], tp.Generic[T]):
    async def put(self, x: T):
        for queue in self:
            await queue.put(x)

    def put_nowait(self, x: T):
        for queue in self:
            queue.put_nowait(x)

    async def worker_done(self):
        for queue in self:
            await queue.worker_done()

    def worker_done_nowait(self):
        for queue in self:
            queue.worker_done_nowait()

    async def stop(self):
        for queue in self:
            await queue.stop()

    def stop_nowait(self):
        for queue in self:
            queue.stop_nowait()

    async def kill(self):
        for queue in self:
            await queue.kill()

    def kill_nowait(self):
        for queue in self:
            queue.kill_nowait()
