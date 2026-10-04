"""Checkpoint-addressed VBench results written by `vbench_eval/checkpoint_eval.py`.

That evaluator deliberately produces no memon Run and no E0030 Variant row: it addresses a
checkpoint directly and drops its scores next to the videos under
`vbench_eval/outputs/E<exp>-V<id>__<step>/[<sub>/]result*.json`. The dashboard used to build
every VBench datapoint from declared E0030 rows only, so those scores were invisible.

Everything here is read-only and cached on a stat signature of the result/contract files, like
`sources.runs`: an unchanged outputs tree costs one stat per evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

from ..config import PROJECT_ROOT

OUTPUTS = PROJECT_ROOT / "vbench_eval/outputs"
_KEY_RE = re.compile(r"^(E\d{4})-(V\d{4})__(\d+)$")
_lock = threading.Lock()
_cache: dict[str, tuple] = {}
_datasets: dict[str, str] | None = None


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _dataset_names() -> dict[str, str]:
    """sha256 -> label for every pinned dataset manifest, so an evaluated prompt set is named by
    its bytes rather than by whatever string a caller happened to pass."""
    global _datasets
    if _datasets is None:
        found = {}
        for path in sorted((PROJECT_ROOT / "vbench_eval/datasets").glob("*.json")):
            m = re.fullmatch(r"([\w-]+)\.v(\d+)\.json", path.name)
            try:
                found[hashlib.sha256(path.read_bytes()).hexdigest()] = (f"{m.group(1)}-v{m.group(2)}" if m
                                                                        else path.stem)
            except OSError:
                continue
        _datasets = found
    return _datasets


def _prompt_set(contract: dict, directory: Path) -> str:
    """Label of the evaluated prompt set, taken from the dataset the evaluation pinned."""
    prompt = contract.get("prompt_set")
    if isinstance(prompt, dict):  # a validation-caption set carries its own hash, not the VBench corpus
        return f"validation{prompt.get('count')}"
    dataset = directory / "dataset.json"
    try:
        digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    except OSError:
        return "vbench"
    return _dataset_names().get(digest, "vbench-unpinned")


def _evaluation(result_path: Path, contract: dict) -> dict | None:
    result = _json(result_path)
    if not result or result.get("quality_score") is None:
        return None
    directory = result_path.parent
    sampling = contract.get("sampling") or {}
    return {
        "path": str(result_path.relative_to(PROJECT_ROOT)),
        "out_dir": str(directory.relative_to(PROJECT_ROOT)),
        "prompt_set": _prompt_set(contract, directory),
        "quality_score": result["quality_score"],
        "dimension_means": result.get("dimension_means") or {},
        "dimension_counts": result.get("dimension_counts") or {},
        "videos_scored": result.get("videos_scored"),
        "videos_missing": result.get("videos_missing") or [],
        "seeds": result.get("seeds") or [],
        "per_seed": result.get("per_seed") or {},
        "seed_dispersion": result.get("seed_dispersion") or {},
        "attention_backend": contract.get("backend"),
        "cube_shape": contract.get("cube_shape"),
        "sparsity": contract.get("sparsity"),
        "linear_quantization": contract.get("linear_quantization"),
        "sampling_steps": sampling.get("steps"),
        "cfg": sampling.get("cfg"),
        "timesteps": sampling.get("timesteps"),
        "geometry": f"{contract.get('frames')}x{contract.get('height')}x{contract.get('width')}"
                    f"@{contract.get('fps')}fps",
        "source": contract.get("source"),
    }


def _result_dirs(directory: Path) -> list[Path]:
    """Directories that can hold a contract/result. Seed directories (`0`, `1`, ...) hold only the
    rendered MP4s - hundreds to thousands of them - so walking into one costs far more than the
    whole rest of the scan and can never find a result."""
    found, stack = [], [directory]
    while stack:
        current = stack.pop()
        found.append(current)
        try:
            for entry in os.scandir(current):
                if entry.is_dir(follow_symlinks=False) and not entry.name.isdigit():
                    stack.append(Path(entry.path))
        except OSError:
            continue
    return found


def _scan_key(directory: Path) -> list[dict]:
    """Every scored evaluation under one E<exp>-V<id>__<step> directory (nested variants included)."""
    out = []
    for candidate in _result_dirs(directory):
        contract = _json(candidate / "cache_contract.json")
        if not contract:
            continue
        for result_path in sorted(candidate.glob("result*.json")):
            evaluation = _evaluation(result_path, contract)
            if evaluation:
                out.append(evaluation)
    return out


def _signature(directory: Path) -> tuple:
    sig: list = []
    for candidate in _result_dirs(directory):
        try:
            for entry in os.scandir(candidate):
                if entry.is_file(follow_symlinks=False) and (
                        entry.name.startswith("result") or entry.name == "cache_contract.json"):
                    st = entry.stat()
                    sig.append((entry.path, st.st_mtime_ns, st.st_size))
        except OSError:
            continue
    return tuple(sorted(sig))


def scan() -> dict[tuple[str, str, int], list[dict]]:
    """Scored evaluations keyed by (experiment, variant, step). Read-only; never touches a Run."""
    out: dict[tuple[str, str, int], list[dict]] = {}
    try:
        entries = sorted(OUTPUTS.iterdir())
    except OSError:
        return out
    for directory in entries:
        m = _KEY_RE.match(directory.name)
        if not (m and directory.is_dir()):
            continue
        sig = _signature(directory)
        if not sig:
            continue
        with _lock:
            hit = _cache.get(directory.name)
            if hit and hit[0] == sig:
                evaluations = hit[1]
            else:
                evaluations = _scan_key(directory)
                _cache[directory.name] = (sig, evaluations)
        if evaluations:
            out[(m.group(1), m.group(2), int(m.group(3)))] = evaluations
    return out


def for_variant(vid: str, exp: str = "E0029") -> dict[int, list[dict]]:
    """Scored evaluations of one training Variant, keyed by training step."""
    return {step: evaluations for (e, v, step), evaluations in scan().items() if v == vid and e == exp}
