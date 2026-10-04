"""results.yaml reader plus byte-preserving writers for Variant blocks.

Reads are cached by file mtime. Writes never reserialize the whole document:
a new Variant is appended as text after the last entry of the trailing
`variants:` list, and field patches replace only the targeted 2-space keys of
one `- id: V<NNNN>` block. Every write is validated by reloading the YAML and
checking that no other Variant changed.
"""
from __future__ import annotations

import copy
import re
import threading
import tempfile
import fcntl
from contextlib import contextmanager
from pathlib import Path

import yaml

from ..config import EXP_DIR, EVAL_EXP, PROJECT_ROOT

_lock = threading.Lock()
_cache: dict[str, tuple[tuple[int, int], dict]] = {}
_SafeLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

_VARIANT_RE = re.compile(r"^- id: (V\d{4})\s*$", re.M)
_SUBKEY_RE = re.compile(r"^  ([A-Za-z_][A-Za-z0-9_]*):", re.M)


def results_path(exp: str) -> Path:
    return EXP_DIR[exp] / "results.yaml"


def load(exp: str) -> dict:
    path = results_path(exp)
    stat = path.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    with _lock:
        hit = _cache.get(exp)
        if hit and hit[0] == signature:
            return hit[1]
        data = yaml.load(path.read_text(), Loader=_SafeLoader) or {}
        data.setdefault("variants", [])
        _cache[exp] = (signature, data)
        return data


def variants(exp: str) -> list[dict]:
    return load(exp)["variants"]


def columns(exp: str) -> list[dict]:
    return load(exp).get("columns", [])


def get(exp: str, vid: str) -> dict | None:
    return next((v for v in variants(exp) if v.get("id") == vid), None)


def next_id(exp: str) -> str:
    ids = [int(v["id"][1:]) for v in variants(exp) if re.fullmatch(r"V\d{4}", str(v.get("id", "")))]
    return f"V{(max(ids) + 1 if ids else 1):04d}"


# ----------------------------------------------------------------------------- text
def _block_span(text: str, vid: str) -> tuple[int, int]:
    """[start, end) byte span of the `- id: <vid>` list entry."""
    starts = list(_VARIANT_RE.finditer(text))
    for i, m in enumerate(starts):
        if m.group(1) == vid:
            end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
            return m.start(), end
    raise KeyError(f"{vid} not found")


def _subkey_span(block: str, key: str) -> tuple[int, int] | None:
    keys = list(_SUBKEY_RE.finditer(block))
    for i, m in enumerate(keys):
        if m.group(1) == key:
            end = keys[i + 1].start() if i + 1 < len(keys) else len(block)
            return m.start(), end
    return None


def _dump_subkey(key: str, value) -> str:
    body = yaml.safe_dump({key: value}, sort_keys=False, allow_unicode=True, width=120, default_flow_style=False)
    return "".join("  " + line + "\n" if line else "\n" for line in body.rstrip("\n").split("\n"))


def _dump_variant(variant: dict) -> str:
    body = yaml.safe_dump([variant], sort_keys=False, allow_unicode=True, width=120, default_flow_style=False)
    return body if body.endswith("\n") else body + "\n"


def _assert_only_changed(exp: str, before: dict, after_text: str, vid: str, allow_new: bool) -> dict:
    after = yaml.safe_load(after_text)
    b = {v["id"]: v for v in before["variants"]}
    a = {v["id"]: v for v in after.get("variants", [])}
    if allow_new:
        if vid not in a or vid in b:
            raise RuntimeError(f"append check failed for {vid}")
    if set(b) - set(a) or (set(a) - set(b)) - ({vid} if allow_new else set()):
        raise RuntimeError("variant set changed unexpectedly")
    for k in b:
        if k != vid and a[k] != b[k]:
            raise RuntimeError(f"unrelated variant {k} changed")
    for k in ("columns", "column_annotations", "schema_version"):
        if before.get(k) != after.get(k):
            raise RuntimeError(f"top-level {k} changed")
    return after


def _write(exp: str, text: str, expected: str) -> None:
    path = results_path(exp)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".results-", suffix=".tmp", delete=False) as stream:
        stream.write(text)
        tmp = Path(stream.name)
    if path.read_text() != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("results.yaml changed during edit; refresh and retry")
    tmp.replace(path)
    with _lock:
        _cache.pop(exp, None)


@contextmanager
def evaluation_lock():
    """Shared with E0030 Run aggregation and binding, including concurrent appends."""
    path = PROJECT_ROOT / "dashboard/state/e0030-binding.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def append_variant(exp: str, variant: dict) -> str:
    """Append a new declaration without racing E0030 Run aggregation."""
    from contextlib import nullcontext
    with evaluation_lock() if exp == EVAL_EXP else nullcontext():
        if exp == EVAL_EXP and (variant.get("runs") or variant.get("attempts")
                               or variant.get("status", "PLANNED") != "PLANNED"
                               or any(value is not None for value in (variant.get("metrics") or {}).values())):
            raise ValueError("New E0030 declarations must be unstarted; import execution data through sync_results.py")
        path = results_path(exp)
        text = path.read_text()
        before = copy.deepcopy(load(exp))
        if list(before)[-1] != "variants":
            raise RuntimeError("variants is not the trailing top-level key; refusing textual append")
        vid = variant.get("id") or next_id(exp)
        if vid in {v["id"] for v in before["variants"]}:
            raise RuntimeError(f"{vid} already exists")
        variant = {"id": vid, **{k: v for k, v in variant.items() if k != "id"}}
        new_text = text + ("" if text.endswith("\n") else "\n") + _dump_variant(variant)
        _assert_only_changed(exp, before, new_text, vid, allow_new=True)
        _write(exp, new_text, text)
        return vid


def patch_variant(exp: str, vid: str, updates: dict) -> None:
    """Replace the given 2-space sub-keys (status, runs, metrics, ...) of one Variant."""
    if exp == EVAL_EXP:
        raise ValueError("E0030 execution rows are Run-derived; use vbench_eval/sync_results.py")
    path = results_path(exp)
    text = path.read_text()
    before = copy.deepcopy(load(exp))
    s, e = _block_span(text, vid)
    block = text[s:e]
    for key, value in updates.items():
        rendered = _dump_subkey(key, value)
        span = _subkey_span(block, key)
        if span:
            block = block[:span[0]] + rendered + block[span[1]:]
        else:
            # insert before provenance when present, else at block end
            prov = _subkey_span(block, "provenance")
            at = prov[0] if prov else len(block)
            block = block[:at] + rendered + block[at:]
    new_text = text[:s] + block + text[e:]
    _assert_only_changed(exp, before, new_text, vid, allow_new=False)
    _write(exp, new_text, text)


def merge_metrics(exp: str, vid: str, metrics: dict) -> None:
    cur = dict((get(exp, vid) or {}).get("metrics") or {})
    cur.update(metrics)
    patch_variant(exp, vid, {"metrics": cur})


def add_run(exp: str, vid: str, run_rel: str) -> None:
    cur = list((get(exp, vid) or {}).get("runs") or [])
    if run_rel not in cur:
        cur.append(run_rel)
    patch_variant(exp, vid, {"runs": cur})
