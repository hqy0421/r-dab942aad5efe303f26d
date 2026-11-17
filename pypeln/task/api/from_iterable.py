import asyncio
import typing as tp

from pypeln import utils as pypeln_utils
from pypeln.utils import A, B, T

from .. import utils
from ..queue import IterableQueue
from ..stage import Stage
from ..worker import ProcessFn, Worker


async def _wait_closed(worker: Worker):
    event = worker.closed_event

    if event is None:
        # Worker has not started yet (should not happen in practice), fall back
        # to polling its `is_done` flag.
        while not worker.is_done:
            await asyncio.sleep(_PUT_POLL_INTERVAL)
        return

    await event.wait()


# How long to wait before re-checking a worker that has not created its
# `closed_event` yet.
_PUT_POLL_INTERVAL = 0.01


class FromIterable(tp.NamedTuple):
    iterable: tp.Union[tp.Iterable, tp.AsyncIterable]
    maxsize: int

    async def __call__(self, worker: Worker, **kwargs):
        iterable: tp.AsyncIterable

        if isinstance(self.iterable, tp.AsyncIterable):
            iterable = self.iterable
        else:
            sync_iterable: tp.Iterable

            if isinstance(self.iterable, pypeln_utils.BaseStage):
                sync_iterable = self.iterable.to_iterable(
                    maxsize=self.maxsize, return_index=True
                )
            else:
                sync_iterable = self.iterable

            queue = IterableQueue(
                maxsize=max(q.maxsize for q in worker.stage_params.output_queues)
            )
            loop = utils.get_running_loop()

            loop.run_in_executor(
                None,
                lambda: FromIterable.consume_iterable(
                    worker=worker, iterable=sync_iterable, queue=queue, loop=loop
                ),
            )

            iterable = queue

        i = 0
        async for x in iterable:
            if not isinstance(x, pypeln_utils.Element):
                x = pypeln_utils.Element(index=(i,), value=x)

            await worker.stage_params.output_queues.put(x)

            i += 1

    @staticmethod
    def consume_iterable(
        worker: Worker,
        iterable: tp.Iterable,
        queue: IterableQueue,
        loop: asyncio.AbstractEventLoop,
    ):
        try:
            for i, x in enumerate(iterable):
                if not isinstance(x, pypeln_utils.Element):
                    x = pypeln_utils.Element(index=(i,), value=x)

                if not FromIterable.put(worker, queue, x, loop):
                    return

            FromIterable.put(worker, queue, pypeln_utils.DONE, loop)

        except BaseException as e:
            try:
                asyncio.run_coroutine_threadsafe(
                    queue.raise_exception(e), loop
                ).result()
            except BaseException:
                pass

    @staticmethod
    def put(
        worker: Worker,
        queue: IterableQueue,
        x: tp.Any,
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        """
        Put `x` into the internal queue and only return once it is really
        there, so the producer respects `maxsize` and never leaves a pending
        coroutine behind. Returns `False` when the worker is gone (failure,
        timeout or early close) and production should stop.
        """

        async def put():
            while not worker.is_done:
                if worker.closed_event is not None and worker.closed_event.is_set():
                    return False

                put_task = asyncio.ensure_future(queue.put(x))
                closed_task = asyncio.ensure_future(_wait_closed(worker))

                try:
                    await asyncio.wait(
                        [put_task, closed_task], return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    if not put_task.done():
                        put_task.cancel()

                    if not closed_task.done():
                        closed_task.cancel()

                    await asyncio.gather(
                        put_task, closed_task, return_exceptions=True
                    )

                if put_task.done() and not put_task.cancelled():
                    exception = put_task.exception()

                    if exception is None:
                        return True

                    return False

                # The worker closed before the item could be delivered.
                if worker.is_done:
                    return False

            return False

        try:
            return asyncio.run_coroutine_threadsafe(put(), loop).result()
        except BaseException:
            # The loop is gone, there is nobody left to consume.
            return False


@tp.overload
def from_iterable(
    iterable: tp.Union[tp.Iterable[T], tp.AsyncIterable[T]],
    use_thread: bool = True,
    maxsize: int = 0,
) -> Stage[T]:
    ...


@tp.overload
def from_iterable(
    use_thread: bool = True, maxsize: int = 0
) -> pypeln_utils.Partial[Stage[T]]:
    ...


def from_iterable(
    iterable: tp.Union[
        tp.Iterable[T], tp.AsyncIterable[T], pypeln_utils.Undefined
    ] = pypeln_utils.UNDEFINED,
    use_thread: bool = True,
    maxsize: int = 0,
) -> tp.Union[Stage[T], pypeln_utils.Partial[Stage[T]]]:
    """
    Creates a stage from an iterable. This function gives you more control of the iterable is consumed.
    Arguments:
        iterable: A source Iterable or AsyncIterable.
        use_thread: This parameter is not used and only kept for API compatibility with the other modules.

    Returns:
        Returns a `Stage` if the `iterable` parameters is given, else it returns a `Partial`.
    """

    if isinstance(iterable, pypeln_utils.Undefined):
        return pypeln_utils.Partial(
            lambda iterable: from_iterable(iterable, use_thread=use_thread)
        )

    return Stage(
        process_fn=FromIterable(iterable, maxsize=maxsize),
        workers=1,
        maxsize=maxsize,
        total_sources=1,
        timeout=0,
        dependencies=[],
        on_start=None,
        on_done=None,
        f_args=[],
    )
