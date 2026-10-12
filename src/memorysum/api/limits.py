"""Bounded Add/Search admission queues."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from memorysum.core.errors import ServiceError


class RequestGate:
    def __init__(self, concurrency: int, max_waiters: int) -> None:
        self._concurrency = concurrency
        self._max_waiters = max_waiters
        self._active = 0
        self._waiting = 0
        self._condition = threading.Condition()

    @contextmanager
    def slot(self) -> Iterator[None]:
        with self._condition:
            if self._active >= self._concurrency:
                if self._waiting >= self._max_waiters:
                    raise ServiceError(
                        "service_queue_full",
                        "The service is busy; retry this request later.",
                        status_code=429,
                        retry_after=5,
                    )
                self._waiting += 1
                try:
                    while self._active >= self._concurrency:
                        self._condition.wait()
                finally:
                    self._waiting -= 1
            self._active += 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify()
