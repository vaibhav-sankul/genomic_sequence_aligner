"""
Multi-Threaded Parallel Processor and Batch Pipeline Coordinator
Overlaps host I/O, sequence preprocessing, and GPU kernel execution using ThreadPoolExecutor.
"""

import queue
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Generator, List, Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)


class BatchParallelExecutor:
    """
    Asynchronous pipeline executor implementing a bounded producer-consumer queue.
    Prefetches and preprocesses upcoming chunks while the compute engine (CUDA or CPU)
    processes the current batch.
    """

    def __init__(self, num_workers: int = 4, max_prefetch: int = 2):
        self.num_workers = max(1, num_workers)
        self.max_prefetch = max_prefetch
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="aligner_worker")
        self.queue: queue.Queue = queue.Queue(maxsize=self.max_prefetch)
        self._stop_event = threading.Event()

    def process_stream(
        self,
        chunk_generator: Generator[List[Dict[str, Any]], None, None],
        align_func: Callable[[List[Dict[str, Any]]], Any]
    ) -> Generator[Any, None, None]:
        """
        Consumes chunks from chunk_generator, dispatching alignment asynchronously.
        Yields alignment results in stream order.
        """
        futures: List[Future] = []

        try:
            for chunk in chunk_generator:
                if self._stop_event.is_set():
                    break

                # Submit chunk to worker pool
                future = self.executor.submit(align_func, chunk)
                futures.append(future)

                # Keep active window within max_prefetch
                while len(futures) >= self.max_prefetch:
                    completed_future = futures.pop(0)
                    yield completed_future.result()

            # Drain remaining active futures
            while futures:
                completed_future = futures.pop(0)
                yield completed_future.result()

        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully shuts down executor threads."""
        self._stop_event.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
