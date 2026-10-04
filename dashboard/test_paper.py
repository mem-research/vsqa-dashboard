"""Paper state follows file times; a half-written PDF never replaces the last complete one."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dashboard import paper


class PaperStatusTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        for context in (patch.object(paper, "PAPER_DIR", self.dir), patch.object(paper, "_snapshot", None)):
            context.start()
            self.addCleanup(context.stop)
        (self.dir / "tex").mkdir()
        (self.dir / "figure").mkdir()

    def write(self, rel, data, t):
        path = self.dir / rel
        path.write_bytes(data)
        os.utime(path, (t, t))

    def test_states(self):
        self.write("tex/1_intro.tex", b"x", 100)
        self.write("main.pdf", b"%PDF...%%EOF\n", 200)
        self.write("main.aux", b"generated", 300)  # build products never count as source edits
        s = paper.status()
        self.assertEqual((s["state"], s["source_file"], s["pdf_mtime"]), ("current", "tex/1_intro.tex", 200))
        self.write("figure/teaser.png", b"img", 250)
        s = paper.status()
        self.assertEqual((s["state"], s["source_file"]), ("compiling", "figure/teaser.png"))

    def test_truncated_pdf_keeps_previous_snapshot(self):
        self.write("main.tex", b"x", 100)
        self.write("main.pdf", b"%PDF complete %%EOF\n", 200)
        self.assertEqual(paper.snapshot()[0], 200)
        self.write("main.tex", b"y", 300)
        self.write("main.pdf", b"%PDF half writt", 310)  # pdflatex mid-write
        mtime, data = paper.snapshot()
        self.assertEqual((mtime, data), (200, b"%PDF complete %%EOF\n"))
        self.assertEqual(paper.status()["state"], "compiling")
        self.write("main.pdf", b"%PDF new %%EOF\n", 320)
        self.assertEqual(paper.status()["state"], "current")

    def test_missing_checkout(self):
        with patch.object(paper, "PAPER_DIR", self.dir / "absent"):
            self.assertFalse(paper.status()["available"])


if __name__ == "__main__":
    unittest.main()
