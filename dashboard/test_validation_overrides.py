"""Corrected batches replace native media without losing historical files."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dashboard.sources import runs


class ValidationOverridesTest(unittest.TestCase):
    def test_complete_publication_replaces_cached_scan_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "logs" / "training"
            ckpt = run / "checkpoints"
            ckpt.mkdir(parents=True)
            old = ckpt / "validation_step_100_inference_steps_50_rank_0_video_0.mp4"
            old.write_bytes(b"historical")
            corrected = run / "validation_shift3" / "100.mp4"
            corrected.parent.mkdir()
            corrected.write_bytes(b"corrected")
            manifest = {"schema_version": 1, "steps": {"100": {
                "expected": 1, "videos": [{"caption_idx": 0, "rank": 0, "idx": 0,
                    "inference_steps": 50, "file": "validation_shift3/100.mp4"}]}}}
            with patch.object(runs, "PROJECT_ROOT", root), patch.object(runs, "LOGS_DIR", root / "logs"), patch.object(runs, "_cache", {}):
                self.assertEqual(runs.scan("training")["validation_videos"]["100"][0]["file"], "checkpoints/" + old.name)
                path = run / "validation_overrides.json"
                path.write_text(json.dumps(manifest))
                self.assertEqual(runs.scan("training")["validation_videos"]["100"][0]["file"], "validation_shift3/100.mp4")
                self.assertEqual(old.read_bytes(), b"historical")
                pending = ckpt / "validation_step_200_inference_steps_50_rank_0_video_0.mp4"
                pending.write_bytes(b"pending correction")
                manifest["replace_all"] = True
                path.write_text(json.dumps(manifest))
                self.assertEqual(runs.scan("training")["validation_steps"], [100])
                self.assertEqual(pending.read_bytes(), b"pending correction")
                manifest.pop("replace_all")
                # An incomplete replacement cannot hide the original batch.
                manifest["steps"]["100"]["expected"] = 2
                path.write_text(json.dumps(manifest))
                self.assertEqual(runs.scan("training")["validation_videos"]["100"][0]["file"], "checkpoints/" + old.name)
                # A symlink outside logs must never be published.
                manifest["steps"]["100"]["expected"] = 1
                outside = root / "outside.mp4"
                outside.write_bytes(b"outside")
                corrected.unlink()
                corrected.symlink_to(outside)
                path.write_text(json.dumps(manifest))
                self.assertEqual(runs._validation_overrides(run), {})


if __name__ == "__main__":
    unittest.main()
