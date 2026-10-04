"""Build state of the live-compiled paper.

The paper lives in its own repository (an Overleaf git remote), by default the sibling checkout
`<project root>/../vsqa-paper`; `VSQA_PAPER_DIR` overrides it. Another session keeps the PDF
rebuilt; the dashboard only reads file times and never writes there.

`compiling` means a source file changed after the last complete PDF; `current` means that PDF
is newer than every source. Whether the reader's copy is stale is decided in the browser, against
the PDF time it loaded.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from .config import PROJECT_ROOT

PAPER_DIR = Path(os.environ.get("VSQA_PAPER_DIR") or PROJECT_ROOT.parent / "vsqa-paper").resolve()
MAIN = "main"
SOURCE_SUFFIXES = {".tex", ".bib", ".sty", ".cls", ".bst", ".bbx", ".cbx"}
FIGURE_DIRS = ("figure", "figures")


def pdf_path() -> Path:
    return PAPER_DIR / f"{MAIN}.pdf"


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _sources():
    """Every author-edited input: TeX/bib/style files anywhere, plus anything under figure/.
    Build products (the PDF, .aux, .bbl, ...) never match, so they cannot mask an edit."""
    for root, dirs, files in os.walk(PAPER_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        parts = Path(root).relative_to(PAPER_DIR).parts
        in_figures = bool(parts) and parts[0] in FIGURE_DIRS
        for name in files:
            if not name.startswith(".") and (in_figures or Path(name).suffix in SOURCE_SUFFIXES):
                yield Path(root) / name


def status() -> dict:
    out = {"paper_dir": str(PAPER_DIR), "now": time.time()}
    if not PAPER_DIR.is_dir():
        return {**out, "available": False, "error": f"paper checkout not found: {PAPER_DIR} (set VSQA_PAPER_DIR)"}
    newest, newest_path = None, None
    for path in _sources():
        m = _mtime(path)
        if m is not None and (newest is None or m > newest):
            newest, newest_path = m, path
    snap = snapshot()  # the newest *complete* PDF: a half-written file must not read as current
    pdf = snap[0] if snap else None
    return {
        **out, "available": pdf is not None,
        "state": "current" if pdf is not None and (newest is None or pdf >= newest) else "compiling",
        "source_mtime": newest,
        "source_file": str(newest_path.relative_to(PAPER_DIR)) if newest_path else None,
        "pdf_mtime": pdf,
        "pdf_size": len(snap[1]) if snap else None,
    }


_snapshot: tuple[float, bytes] | None = None


def snapshot() -> tuple[float, bytes] | None:
    """The newest complete PDF as (mtime, bytes). pdflatex rewrites the file in place, so a read
    during a build can be truncated; such a read (no trailing %%EOF) keeps the previous snapshot."""
    global _snapshot
    path = pdf_path()
    m = _mtime(path)
    if m is None:
        return _snapshot
    if _snapshot and _snapshot[0] == m:
        return _snapshot
    try:
        data = path.read_bytes()
    except OSError:
        return _snapshot
    if b"%%EOF" in data[-1024:] and _mtime(path) == m:
        _snapshot = (m, data)
    return _snapshot
