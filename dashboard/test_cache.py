"""Cache reuse must not hide new Runs, completed jobs or changed results."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dashboard.sources import results, runs


class RunInventoryCacheTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.now = 100.0
        for context in (
            patch.object(runs, "PROJECT_ROOT", self.root),
            patch.object(runs, "LOGS_DIR", self.logs),
            patch.object(runs, "_cache", {}),
            patch.object(runs, "_list_cache", None),
            patch.object(runs.time, "monotonic", side_effect=lambda: self.now),
        ):
            context.start()
            self.addCleanup(context.stop)

    def make_run(self, name):
        path = self.logs / name
        path.mkdir()
        (path / "execution_state.txt").write_text("RUNNING 2026-09-15T00:00:00Z")
        return path

    def test_new_and_removed_runs_appear_after_inventory_expiry(self):
        first = self.make_run("first")
        self.assertEqual([r["id"] for r in runs.list_all()], ["first"])
        self.make_run("second")
        (first / "execution_state.txt").unlink()
        first.rmdir()
        self.now += runs._LIST_CACHE_SECONDS - 0.1
        self.assertEqual([r["id"] for r in runs.list_all()], ["first"])
        self.now += 0.1
        self.assertEqual([r["id"] for r in runs.list_all()], ["second"])
        self.assertIsNone(runs.scan("first"))

    def test_direct_scan_and_explicit_invalidation_do_not_wait_for_expiry(self):
        first = self.make_run("first")
        self.assertEqual(runs.list_all()[0]["execution"]["state"], "RUNNING")
        (first / "execution_state.txt").write_text("FINISHED 2026-09-15T00:01:00Z")
        self.assertEqual(runs.scan("first")["execution"]["state"], "FINISHED")
        self.make_run("second")
        runs.invalidate("logs/first")
        snapshot = {r["id"]: r for r in runs.list_all()}
        self.assertEqual(set(snapshot), {"first", "second"})
        self.assertEqual(snapshot["first"]["execution"]["state"], "FINISHED")


class ResultsCacheTest(unittest.TestCase):
    def test_same_size_rewrite_with_nanosecond_timestamp_is_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "results.yaml"
            path.write_text("variants: [{id: V0001, status: RUNNING}]\n")
            ns = 1700000000000000000
            os.utime(path, ns=(ns, ns))
            with patch.dict(results.EXP_DIR, {"cache-test": root}), patch.object(results, "_cache", {}):
                self.assertEqual(results.get("cache-test", "V0001")["status"], "RUNNING")
                path.write_text("variants: [{id: V0001, status: PENDING}]\n")
                os.utime(path, ns=(ns + 1, ns + 1))
                self.assertEqual(results.get("cache-test", "V0001")["status"], "PENDING")


if __name__ == "__main__":
    unittest.main()
