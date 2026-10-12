"""Process-local bounded TTL/LRU caches. Scope keys always include the user hash."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class _Entry[T]:
    expires_at: float
    value: T


class TTLCache[T]:
    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, _Entry[T]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry.value

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._entries[key] = _Entry(time.monotonic() + self._ttl_seconds, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear_scope(self, scope_key: str) -> None:
        marker = scope_key + ":"
        with self._lock:
            for key in [key for key in self._entries if key.startswith(marker)]:
                del self._entries[key]
