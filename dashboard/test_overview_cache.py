"""The cached overview must never hide a write, and a failed build must not wedge the endpoint."""
import gzip
import json
import threading
import time
import unittest
from unittest.mock import patch

from dashboard import overview_cache
from dashboard.overview_cache import OverviewCache


class OverviewCacheTest(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        clock = patch.object(overview_cache.time, "time", lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.builds = 0
        self.broken = False

    def build(self):
        self.builds += 1
        if self.broken:
            raise RuntimeError("boom")
        return {"n": self.builds}

    @staticmethod
    def decode(result):
        body, gz, _ = result
        assert gzip.decompress(gz) == body
        return json.loads(body)["n"]

    def settle(self, cache):
        for _ in range(200):
            with cache._cv:
                if not cache._building:
                    return
            time.sleep(0.01)
        self.fail("build did not finish")

    def test_fresh_payload_is_reused(self):
        cache = OverviewCache(self.build, ttl=60, max_stale=600)
        self.assertEqual(self.decode(cache.get()), 1)
        self.now += 30
        self.assertEqual(self.decode(cache.get()), 1)
        self.assertEqual(self.builds, 1)

    def test_stale_payload_is_served_while_rebuilding(self):
        cache = OverviewCache(self.build, ttl=60, max_stale=600)
        cache.get()
        self.now += 120
        self.assertEqual(self.decode(cache.get()), 1)  # stale answer, no waiting
        self.settle(cache)
        self.assertEqual(self.decode(cache.get()), 2)

    def test_too_old_payload_waits_for_a_build(self):
        cache = OverviewCache(self.build, ttl=60, max_stale=600)
        cache.get()
        self.now += 601
        self.assertEqual(self.decode(cache.get()), 2)

    def test_invalidation_waits_for_a_post_write_build(self):
        release = threading.Event()
        started = threading.Event()

        def slow_build():
            started.set()
            release.wait(5)
            return self.build()

        cache = OverviewCache(slow_build, ttl=60, max_stale=600)
        release.set()
        cache.get()
        release.clear(); started.clear()
        cache.invalidate()  # a write happened
        started.wait(5)
        cache.invalidate()  # another write while that build runs: its result predates this write
        got = []
        reader = threading.Thread(target=lambda: got.append(self.decode(cache.get())))
        reader.start()
        release.set()
        reader.join(5)
        self.assertEqual(got, [3])  # neither the pre-write payload nor the mid-write build

    def test_failed_build_serves_stale_then_retries(self):
        cache = OverviewCache(self.build, ttl=60, max_stale=600)
        cache.get()
        self.broken = True
        self.now += 601
        self.assertEqual(self.decode(cache.get()), 1)  # rebuild failed: stale beats nothing
        self.broken = False
        self.assertEqual(self.decode(cache.get()), 3)  # next request retries

    def test_failed_first_build_raises_and_retries(self):
        cache = OverviewCache(self.build, ttl=60, max_stale=600)
        self.broken = True
        with self.assertRaises(RuntimeError):
            cache.get()
        self.broken = False
        self.assertEqual(self.decode(cache.get()), 2)


if __name__ == "__main__":
    unittest.main()
