"""Scan `logs/<run>/` directories for execution facts, artifacts and media.

Everything here is read-only. Per-run snapshots are reused while their file
signatures match. The whole-directory inventory has a five-second TTL to avoid
repeating thousands of stat calls for each Variant; explicit invalidation
clears both caches, and direct scan() calls always recheck the Run's signature.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from ..config import LOGS_DIR, PROJECT_ROOT

_lock = threading.RLock()
_cache: dict[str, tuple[tuple, dict]] = {}
# Reuse the expensive whole-directory inventory for at most five seconds.
# Individual scan() calls still validate file signatures immediately.
_LIST_CACHE_SECONDS = 5.0
_list_cache: tuple[float, list[dict]] | None = None

_STATE_RE = re.compile(r"^(?P<state>[A-Z]+)(?:\s+(?P<ts>\S+))?(?P<rest>.*)$")
_KV_RE = re.compile(r"(\w+)=(\S+)")
_TQDM_RE = re.compile(r"Steps:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s+\[(\S+)<(\S+),\s*([\d.]+)s/it\]")
_VIDEO_RE = re.compile(r"^validation_step_(\d+)_inference_steps_(\d+)(?:_rank_(\d+))?_video_(\d+)(_overlay)?\.mp4$")
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")
_EXPORT_RE = re.compile(r"^export-(?:dense-)?step(\d+)(-ema)?$")


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return ""


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _validation_overrides(dirpath: Path) -> dict[int, list[dict]]:
    """Replace only complete, explicitly published validation batches.

    Corrected media use new URLs, preserving historical files and bypassing
    browser media caches. Symlinks may point to another replay under logs/.
    """
    manifest = _json(dirpath / "validation_overrides.json") or {}
    if manifest.get("schema_version") != 1:
        return {}
    batches = {}
    for step, batch in manifest.get("steps", {}).items():
        try:
            entries = batch["videos"]
            expected = int(batch["expected"])
            if expected <= 0 or len(entries) != expected:
                continue
            if sorted(v["caption_idx"] for v in entries) != list(range(expected)):
                continue
            for v in entries:
                relative = Path(v["file"])
                target = (dirpath / relative).resolve()
                if (relative.is_absolute() or ".." in relative.parts
                        or not target.is_relative_to(LOGS_DIR.resolve())
                        or target.suffix != ".mp4" or not target.is_file()):
                    raise ValueError("invalid corrected validation path")
            batches[int(step)] = sorted(entries, key=lambda v: v["caption_idx"])
        except (KeyError, TypeError, ValueError, OSError):
            continue
    return batches


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def run_dir(run_ref: str) -> Path:
    p = Path(run_ref)
    if p.is_absolute():
        return p
    if run_ref.startswith("logs/"):
        return PROJECT_ROOT / run_ref
    return LOGS_DIR / run_ref


def _parse_state(text: str) -> dict:
    out = {"state": None, "ts": None, "job": None, "slurm_step": None, "host": None, "mode": None}
    if not text:
        return out
    m = _STATE_RE.match(text.splitlines()[0])
    if not m:
        return out
    out["state"] = m.group("state")
    out["ts"] = m.group("ts")
    for k, v in _KV_RE.findall(m.group("rest") or ""):
        # Launchers write SLURM_STEP_ID here, never the optimizer step.
        if k == "step":
            k = "slurm_step"
        if k in out:
            out[k] = int(v) if k in ("job", "slurm_step") and v.isdigit() else v
    return out


def _tail(path: Path, nbytes: int = 4096) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _progress(run_log: Path) -> dict | None:
    # Use cumulative elapsed-time differences, not tqdm's smoothed s/it.
    # The first observed step may be a resumed global step.
    last = None
    previous = None
    seconds = 0
    steps = 0
    for m in _TQDM_RE.finditer(_read(run_log)):
        parts = m.group(3).split(":")
        if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
            continue
        elapsed = 0
        for part in parts:
            elapsed = elapsed * 60 + int(part)
        current = (int(m.group(1)), elapsed)
        if previous is not None:
            ds, dt = current[0] - previous[0], current[1] - previous[1]
            if ds > 0 and dt >= 0:
                steps += ds
                seconds += dt
        previous = current
        last = m
    if not last:
        return None
    return {"step": int(last.group(1)), "total": int(last.group(2)), "elapsed": last.group(3),
            "eta": last.group(4), "sec_per_it": float(last.group(5)),
            "avg_sec_per_step": seconds / steps if steps and seconds > 0 else None,
            "timed_steps": steps, "timed_seconds": seconds}


def _frontmatter(readme: Path) -> dict:
    text = _read(readme)
    if not text.startswith("---"):
        return {}
    parts = text.split("\n---", 2)
    if len(parts) < 2:
        return {}
    out = {}
    for line in parts[0].splitlines()[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"')
    return out


def _gpu_rows(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            rows.append({"gpu": int(parts[0]), "used_mib": int(parts[1].split()[0]),
                         "total_mib": int(parts[2].split()[0])})
    return rows


def _gpu_latest(path: Path) -> list[dict]:
    rows = _gpu_rows(_tail(path, 2048))
    if not rows:
        return []
    n = max(r["gpu"] for r in rows) + 1
    return rows[-n:]


def _gpu_peak(path: Path) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(r"peak_used_MiB=(\d+)", _read(path))]


def _scan(dirpath: Path) -> dict:
    name = dirpath.name
    info: dict = {"id": name, "path": rel(dirpath), "kind": "eval" if name.startswith("e0030-") else "train"}
    st = _parse_state(_read(dirpath / "execution_state.txt"))
    info["execution"] = st
    info["exit_code"] = _read(dirpath / "exit_code") or None
    info["trainer_exit_code"] = _read(dirpath / "trainer_exit_code") or None
    fm = _frontmatter(dirpath / "README.md")
    info["readme"] = {"status": fm.get("status"), "name": fm.get("name"), "finished_at": fm.get("finished_at"),
                      "host": fm.get("host"), "command": fm.get("command"), "entry": fm.get("entry")} if fm else None
    info["code_head"] = _read(dirpath / "code.head")[:40] or None
    run_log = dirpath / "run.log"
    try:
        info["log_mtime"] = run_log.stat().st_mtime
    except OSError:
        info["log_mtime"] = None
    info["progress"] = _progress(run_log)
    info["wandb"] = _json(dirpath / "wandb.json")
    info["gpu_peak_mib"] = _gpu_peak(dirpath / "gpu_peak.txt")
    info["gpu_latest"] = _gpu_latest(dirpath / "gpu_mem.csv")

    # training artifacts
    ckpt_dir = dirpath / "checkpoints"
    steps, videos = [], {}
    if ckpt_dir.is_dir():
        try:
            entries = list(ckpt_dir.iterdir())
        except OSError:
            entries = []
        for e in entries:
            m = _CKPT_RE.match(e.name)
            if m:
                steps.append(int(m.group(1)))
                continue
            m = _VIDEO_RE.match(e.name)
            if m and not m.group(5):
                step, k, rank, idx = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0), int(m.group(4))
                videos.setdefault(step, []).append({"rank": rank, "idx": idx, "inference_steps": k,
                                                    "file": f"checkpoints/{e.name}"})
    captions = []
    vj = _json(dirpath / "validation.json")
    if isinstance(vj, dict):
        captions = [d.get("caption", "") for d in vj.get("data", [])]
    ranks = sorted({v["rank"] for vs in videos.values() for v in vs}) or [0]
    per_rank = max(1, len(captions) // max(1, len(ranks))) if captions else None
    for vs in videos.values():
        vs.sort(key=lambda v: (v["rank"], v["idx"]))
        for v in vs:
            ci = v["rank"] * per_rank + v["idx"] if per_rank else v["idx"]
            v["caption_idx"] = ci
            v["caption"] = captions[ci] if ci < len(captions) else None
    # Captions added to the validation set after a Run finished are rendered afterwards into
    # `val_extra/<step>/<caption_idx:02d>.mp4`. They are addressed by caption index, never by the
    # trainer's rank/idx sharding, so they extend a step's video list without touching the mapping
    # of the videos the trainer itself wrote.
    extra_captions: list[str] = []
    xd = dirpath / "val_extra"
    if xd.is_dir():
        for sd in sorted(xd.iterdir()):
            if not sd.name.isdigit() or not sd.is_dir():
                continue
            man = _json(sd / "manifest.json") or {}
            caps = man.get("captions") or captions
            if len(caps) > len(extra_captions):
                extra_captions = caps
            for name in sorted(p.name for p in sd.glob("[0-9][0-9].mp4")):
                ci = int(name[:2])
                videos.setdefault(int(sd.name), []).append({
                    "rank": None, "idx": ci, "caption_idx": ci,
                    "caption": caps[ci] if ci < len(caps) else None, "extra": True,
                    "backend": man.get("attention_backend"),
                    "file": f"val_extra/{sd.name}/{name}"})
    if extra_captions:
        captions = extra_captions
    for vs in videos.values():
        vs.sort(key=lambda v: v["caption_idx"])
    # Renders of the same captions forced onto a different sampling-step count. They are a
    # separate comparison, not part of the Run's validation set, so they stay in their own
    # structure and get their own dashboard lane.
    forced: dict[str, dict] = {}
    for fd in sorted(dirpath.glob("val_forced*")):
        if not fd.is_dir():
            continue
        for sd in sorted(fd.iterdir()):
            if not sd.name.isdigit() or not sd.is_dir():
                continue
            man = _json(sd / "manifest.json") or {}
            caps = man.get("captions") or captions
            files = sorted(p.name for p in sd.glob("[0-9][0-9].mp4"))
            if not files:
                continue
            forced.setdefault(str(int(sd.name)), {
                "sampling_steps": man.get("forced_sampling_steps"),
                "declared_sampling_steps": man.get("declared_sampling_steps"),
                "backend": man.get("attention_backend"), "expected": len(man.get("rendered_indices") or caps),
                "count": 0, "videos": []})
            entry = forced[str(int(sd.name))]
            entry["count"] = len(files)
            entry["videos"] = [{"idx": int(f[:2]), "caption_idx": int(f[:2]),
                                "caption": caps[int(f[:2])] if int(f[:2]) < len(caps) else None,
                                "file": f"{fd.name}/{sd.name}/{f}"} for f in files]
    info["forced_val"] = forced
    exports = []
    try:
        for e in dirpath.iterdir():
            m = _EXPORT_RE.match(e.name)
            if m and e.is_dir():
                exports.append({"step": int(m.group(1)), "ema": bool(m.group(2)), "path": rel(e),
                                "has_metadata": (e / "metadata.json").is_file()})
    except OSError:
        pass
    inf_val = {}
    iv = dirpath / "inf_val"
    if iv.is_dir():
        for sd in iv.iterdir():
            if not sd.name.isdigit() or not sd.is_dir():
                continue
            man = _json(sd / "manifest.json") or {}
            caps = man.get("captions") or captions
            files = sorted(p.name for p in sd.glob("[0-9][0-9].mp4"))
            inf_val[str(int(sd.name))] = {
                "state": _read(sd / "state") or None, "exit_code": _read(sd / "exit_code") or None,
                "expected": len(caps), "count": len(files), "provider": man.get("provider"),
                "backend": man.get("attention_backend"), "started_at": man.get("started_at"),
                "videos": [{"idx": int(f[:2]), "caption_idx": int(f[:2]), "caption": caps[int(f[:2])] if int(f[:2]) < len(caps) else None,
                            "file": f"inf_val/{sd.name}/{f}"} for f in files]}
    info["inf_val"] = inf_val
    info["checkpoint_steps"] = sorted(steps)
    info["exports"] = sorted(exports, key=lambda x: (x["step"], x["ema"]))
    overrides = _validation_overrides(dirpath)
    if overrides and (_json(dirpath / "validation_overrides.json") or {}).get("replace_all"):
        videos = overrides
    else:
        videos.update(overrides)
    info["validation_steps"] = sorted(videos)
    info["validation_videos"] = {str(k): v for k, v in sorted(videos.items())}
    info["captions"] = captions

    # evaluation artifacts
    if info["kind"] == "eval":
        info["result"] = _json(dirpath / "result.json")
        info["checkpoint"] = _json(dirpath / "checkpoint.json")
        info["launch"] = _read(dirpath / "launch.txt") or None
        vd = dirpath / "videos"
        try:
            info["video_count"] = sum(1 for p in vd.iterdir() if p.suffix == ".mp4") if vd.is_dir() else 0
        except OSError:
            info["video_count"] = 0
        info["has_per_video"] = (dirpath / "scores_per_video.json").is_file()
    return info


def _signature(d: Path) -> tuple:
    sig = []
    for name in ("", "checkpoints", "run.log", "execution_state.txt", "exit_code", "README.md", "result.json",
                 "gpu_mem.csv", "wandb.json", "inf_val", "val_extra", "validation_overrides.json"):
        try:
            st = (d / name).stat() if name else d.stat()
            sig.append((st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append(None)
    try:
        for p in sorted((d / "inf_val").glob("*/state")):
            st = p.stat(); sig.append((p.parent.name, st.st_mtime_ns))
    except OSError:
        pass
    try:
        for p in sorted((d / "val_extra").glob("*/manifest.json")):
            st = p.stat(); sig.append((p.parent.name, st.st_mtime_ns, st.st_size))
        for p in sorted((d / "val_extra").glob("*")):
            if p.is_dir():
                st = p.stat(); sig.append((p.name, st.st_mtime_ns))
    except OSError:
        pass
    try:
        for p in sorted(d.glob("val_forced*/*/manifest.json")):
            st = p.stat(); sig.append((p.parent.parent.name, p.parent.name, st.st_mtime_ns, st.st_size))
        for p in sorted(d.glob("val_forced*/*")):
            if p.is_dir():
                st = p.stat(); sig.append((p.parent.name, p.name, st.st_mtime_ns))
    except OSError:
        pass
    except OSError:
        pass
    return tuple(sig)


def scan(run_ref: str) -> dict | None:
    """Snapshot of one run directory; recomputed only when a tracked file/dir stat changes."""
    d = run_dir(run_ref)
    if not d.is_dir():
        return None
    key = str(d)
    sig = _signature(d)
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == sig:
            return hit[1]
    info = _scan(d)
    with _lock:
        _cache[key] = (sig, info)
    return info


def invalidate(path: str) -> None:
    global _list_cache
    with _lock:
        _cache.pop(str(run_dir(path)), None)
        _list_cache = None


def list_all() -> list[dict]:
    """Share one inventory across callers; newly added/removed Runs appear within 5s."""
    global _list_cache
    with _lock:
        if _list_cache is not None and time.monotonic() - _list_cache[0] < _LIST_CACHE_SECONDS:
            return _list_cache[1]
        try:
            dirs = [p for p in LOGS_DIR.iterdir() if p.is_dir()]
        except OSError:
            _list_cache = None
            return []
        out = []
        for d in sorted(dirs, key=lambda p: p.name):
            info = scan(str(d))
            if info:
                out.append(info)
        present = {str(d) for d in dirs}
        for key in _cache.keys() - present:
            del _cache[key]
        _list_cache = (time.monotonic(), out)
        return out


def effective_status(info: dict) -> str:
    """Single status string combining execution_state, exit_code and README."""
    st = (info.get("execution") or {}).get("state")
    if st:
        return st
    rd = info.get("readme") or {}
    if rd.get("status"):
        return rd["status"]
    ec = info.get("exit_code")
    if ec is not None:
        return "FINISHED" if ec == "0" else "FAILED"
    return "UNKNOWN"


def per_video_scores(run_ref: str) -> dict | None:
    return _json(run_dir(run_ref) / "scores_per_video.json")
