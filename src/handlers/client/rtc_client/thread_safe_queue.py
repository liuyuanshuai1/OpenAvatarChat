"""Synchronous storage with coalesced asyncio wakeups for RTC worker threads."""
import asyncio
from collections import deque
import threading
import time


class ThreadSafeAsyncQueue:
    def __init__(self, duration=None):
        self._items = deque()
        self._lock = threading.Lock()
        self._loop = None
        self._available = None
        self._wake_pending = False
        self._duration = duration or (lambda _: 0.0)
        self._seconds = 0.0
        self._peak_seconds = 0.0
        self._peak_items = 0
        self._cleared = 0

    @property
    def maxsize(self):
        return 0

    def qsize(self):
        with self._lock:
            return len(self._items)

    def empty(self):
        return self.qsize() == 0

    def put_nowait(self, item):
        with self._lock:
            seconds = max(0.0, float(self._duration(item)))
            self._items.append((item, seconds, time.monotonic()))
            self._seconds += seconds
            self._peak_seconds = max(self._peak_seconds, self._seconds)
            self._peak_items = max(self._peak_items, len(self._items))
            if self._loop is not None and not self._wake_pending:
                self._wake_pending = True
                try:
                    self._loop.call_soon_threadsafe(self._wake)
                except RuntimeError:
                    self._wake_pending = False

    def _wake(self):
        with self._lock:
            self._wake_pending = False
            if self._items:
                self._available.set()

    def get_nowait(self):
        with self._lock:
            if not self._items:
                raise asyncio.QueueEmpty
            return self._pop()

    def _pop(self):
        item, seconds, _ = self._items.popleft()
        self._seconds = max(0.0, self._seconds - seconds)
        return item

    async def get(self):
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._loop is None:
                self._loop = loop
                self._available = asyncio.Event()
            elif self._loop is not loop:
                raise RuntimeError('RTC output queue belongs to a different event loop')
        while True:
            with self._lock:
                if self._items:
                    return self._pop()
                self._available.clear()
            await self._available.wait()

    def clear(self):
        with self._lock:
            self._cleared += len(self._items)
            self._items.clear()
            self._seconds = 0.0

    def snapshot(self):
        with self._lock:
            return dict(items=len(self._items), seconds=round(self._seconds, 4),
                        oldest_age_ms=round((time.monotonic() - self._items[0][2]) * 1000, 2)
                        if self._items else 0.0,
                        peak_seconds=round(self._peak_seconds, 4),
                        peak_items=self._peak_items, cleared_items=self._cleared)
