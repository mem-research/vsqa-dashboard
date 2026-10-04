"""Serve the pre-built `/api/overview` payload and rebuild it off the request path.

Building the overview re-stats every run directory on the shared filesystem (2-4 s warm,
~15 s cold), and each open page polls it every two minutes. The payload is therefore built
by a background thread and kept as JSON plus gzip bytes:

- age <= `ttl`: served as is;
- `ttl` < age <= `max_stale`: served as is while one background rebuild starts;
- older than `max_stale`, or invalidated by a write: the request waits for a fresh build,
  so a queued/cancelled datapoint is never overwritten by a pre-write snapshot.
"""
from __future__ import annotations

import gzip
import json
import logging
import threading
import time
from typing import Callable

log = logging.getLogger("dashboard.overview_cache")


class OverviewCache:
    def __init__(self, build: Callable[[], dict], ttl: float = 60, max_stale: float = 600):
        self._build = build
        self.ttl, self.max_stale = ttl, max_stale
        self._cv = threading.Condition()
        self._payload: tuple[bytes, bytes] | None = None  # (json, gzip)
        self._built_at = 0.0
        self._gen = 0          # bumped by every invalidation
        self._built_gen = -1   # generation the current payload was built for
        self._building = False
        self._error: Exception | None = None

    def _start(self) -> None:  # caller holds the lock
        if not self._building:
            self._building = True
            threading.Thread(target=self._run, name="overview-build", daemon=True).start()

    def _run(self) -> None:
        with self._cv:
            gen = self._gen
        t0 = time.monotonic()
        try:
            body = json.dumps(self._build(), ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=str).encode()
            payload, error = (body, gzip.compress(body, compresslevel=6)), None
        except Exception as e:  # keep serving the previous payload; surface the error if there is none
            log.exception("overview build failed")
            payload, error = None, e
        with self._cv:
            if payload:
                self._payload, self._built_at = payload, time.time()
            self._built_gen, self._error, self._building = gen, error, False
            if self._gen != gen:  # invalidated while building: that build may predate the write
                self._start()
            self._cv.notify_all()
        log.info("overview built in %.2fs (%s bytes)", time.monotonic() - t0, len(payload[0]) if payload else "failed")

    def warm(self) -> None:
        with self._cv:
            self._start()

    def invalidate(self) -> None:
        with self._cv:
            self._gen += 1
            self._start()

    def get(self) -> tuple[bytes, bytes, float]:
        """Return (json, gzip, age_seconds), waiting only when no acceptable payload exists."""
        with self._cv:
            while True:
                age = time.time() - self._built_at
                current = self._payload is not None and self._built_gen == self._gen
                if current and age <= self.max_stale:
                    if age > self.ttl:
                        self._start()
                    return (*self._payload, age)
                if not self._building and self._built_gen == self._gen and self._error is not None:
                    # Report the failure once; the next request starts a new attempt.
                    error, self._error = self._error, None
                    if self._payload is None:
                        raise error
                    return (*self._payload, age)  # rebuild failed: stale beats nothing
                self._start()
                self._cv.wait()
