"""Build PLANNED Variant rows for E0029 (training) and E0030 (evaluation) and
run the memon bookkeeping (`doc lint`, `journal submit`) after a write."""
from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import CODE_REPO, CODE_ROOT, EVAL_EXP, EVAL_LAUNCHER, MEMON, PROJECT_ROOT, TRAIN_EXP, VBENCH_QUALITY_DIMS
from .sources import results

log = logging.getLogger("dashboard.variants")
_creation_lock = threading.Lock()

VBENCH_INFO = (CODE_ROOT / "fastvideo/third_party/eval/vbench/vbench/VBench_full_info.json")
REF_RE = re.compile(r"^E0029/(V\d{4})@(\d+)$")
FT_SAMPLING = {"steps": 50, "cfg": 5.0, "flow_shift": 3.0}
DMD_SAMPLING = {"steps": 3, "cfg": 1.0, "flow_shift": 8.0}
BACKEND_LABEL = {
    "": "TORCH_SDPA (fastvideo default resolution; flash_attn not installed)",
    "VSQA": "VSQA (FVFA4-v3 NVFP4-QK/FP8-PV sparse, cube from checkpoint metadata)",
}
SCORER = "fastvideo.eval vbench.* @ vbench 45e79ec + E0030 static_filter port"
NEG_PROMPT = "fastvideo Wan preset default (Chinese-English Wan negative prompt)"


def memon(*args: str, timeout: int = 120) -> tuple[int, str]:
    argv = [MEMON, "--project-root", str(PROJECT_ROOT), "--format", "json", *args]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=PROJECT_ROOT)
        return done.returncode, (done.stdout + done.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def lint_and_submit(exp: str) -> dict:
    rc, out = memon("experiment", "doc", "lint", exp)
    rc2, out2 = memon("journal", "submit", "--files", f"docs/experiments/{exp}/results.yaml")
    return {"lint_rc": rc, "lint": out[-2000:], "journal_rc": rc2, "journal": out2[-1000:]}


# ----------------------------------------------------------------------- prompts
def prompt_counts(dimensions: list[str]) -> dict:
    rows = json.loads(VBENCH_INFO.read_text())
    dims = [d for d in dimensions if d in VBENCH_QUALITY_DIMS]
    per = {d: 0 for d in dims}
    union = 0
    official = 0
    for r in rows:
        hit = [d for d in r["dimension"] if d in dims]
        if not hit:
            continue
        union += 1
        official += 25 if "temporal_flickering" in r["dimension"] else 5
        for d in hit:
            per[d] += 1
    return {"dimensions": dims, "per_dimension": per, "union": union, "official_videos": official}


# ------------------------------------------------------------------- eval variant
def checkpoint_facts(source: str) -> dict:
    """Attention facts for the E0030 row from the E0029 Variant (or HF defaults)."""
    m = REF_RE.match(source)
    if not m:
        return {"checkpoint": f"{source} (HF/diffusers)", "attention_kind": "dense", "quantization": "none (BF16)",
                "cube_shape": "none", "sparsity_default": 0.0, "objective": "fine_tune", "e0029": None}
    vid, step = m.group(1), int(m.group(2))
    v = results.get(TRAIN_EXP, vid)
    if not v:
        raise ValueError(f"{vid} not in E0029 results.yaml")
    p = v.get("parameters") or {}
    objective = str(p.get("objective") or "fine_tune")
    return {"checkpoint": f"E0029 {vid} {v.get('name', '')} step-{step}".strip(),
            "attention_kind": p.get("attention_kind") or "dense",
            "quantization": p.get("quantization") or "unknown",
            "cube_shape": str(p.get("cube_shape") or "none"),
            "sparsity_default": float(p.get("sparsity") or 0.0), "objective": objective, "e0029": {"variant": vid, "step": step}}


def build_eval_variant(form: dict) -> tuple[dict, dict]:
    """Return (variant_row, launcher_env) for an E0030 evaluation request."""
    source = str(form["checkpoint_source"]).strip()
    backend = str(form.get("attention_backend") or "").strip()
    if backend not in BACKEND_LABEL:
        raise ValueError(f"unsupported backend {backend!r}")
    facts = checkpoint_facts(source)
    dims = [d for d in (form.get("dimensions") or VBENCH_QUALITY_DIMS) if d in VBENCH_QUALITY_DIMS]
    if not dims:
        raise ValueError("no VBench dimensions selected")
    counts = prompt_counts(dims)
    sample_mode = form.get("sample_mode") or "fixed"
    spp = int(form.get("samples_per_prompt") or 1)
    seed_base = int(form.get("seed_base") or 0)
    limit = int(form["limit"]) if form.get("limit") else None
    frames = int(form.get("frames") or 81); height = int(form.get("height") or 480); width = int(form.get("width") or 832)
    fps = int(form.get("fps") or 16)
    default_sampling = DMD_SAMPLING if facts["objective"].lower().startswith("dmd") else FT_SAMPLING
    steps = int(form["sampling_steps"]) if form.get("sampling_steps") else default_sampling["steps"]
    cfg = float(form["cfg"]) if form.get("cfg") else default_sampling["cfg"]
    flow_shift = float(form["flow_shift"]) if form.get("flow_shift") else default_sampling["flow_shift"]
    sparsity = float(form["sparsity"]) if form.get("sparsity") not in (None, "") else facts["sparsity_default"]
    if backend == "VSQA" and str(facts["attention_kind"]).lower() != "vsa":
        raise ValueError("VSQA requires a sparse (vsa) checkpoint")
    prompts = min(counts["union"], limit) if limit else counts["union"]
    videos = prompts * spp if sample_mode == "fixed" else (counts["official_videos"] if not limit else None)
    dims_label = "quality-7" if len(dims) == 7 else "quality subset: " + ",".join(dims)
    policy = f"fixed {spp}" if sample_mode == "fixed" else "official (5 per prompt, 25 for temporal_flickering)"
    short_backend = "dense BF16" if backend == "" else "VSQA"
    name = f"{source} under {short_backend} forward, {dims_label}, {policy}" + (f", limit {limit}" if limit else "")
    row = {
        "name": name,
        "status": "PLANNED",
        "description": (f"Queued from the VSQA dashboard on {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
                        "Execution and metrics are written back automatically by the dashboard scheduler; "
                        "no quality interpretation is implied."),
        "parameters": {
            "checkpoint": facts["checkpoint"], "checkpoint_source": source,
            "generator": f"{CODE_REPO} VideoGenerator",
            "attention_backend": BACKEND_LABEL[backend],
            "attention_kind": facts["attention_kind"] if backend == "VSQA" else "dense",
            "quantization": facts["quantization"] if backend == "VSQA" else "none (BF16)",
            "cube_shape": facts["cube_shape"] if backend == "VSQA" else "none",
            "sparsity": sparsity if backend == "VSQA" else 0.0,
            "dimensions": dims_label, "prompt_set": f"VBench_full_info.json {dims_label} union ({counts['union']} prompts)"
                          + (f", first {limit}" if limit else ""),
            "prompts": prompts, "sample_mode": policy, "videos": videos, "seed_base": seed_base,
            "frames": frames, "height": height, "width": width, "fps": fps,
            "sampling_steps": steps, "cfg": cfg, "flow_shift": flow_shift,
            "negative_prompt": NEG_PROMPT, "scorer": SCORER,
        },
        "metrics": {}, "runs": [], "attempts": [],
        "provenance": {"repo": CODE_REPO, "entry": EVAL_LAUNCHER, "env": {}},
    }
    env = {"CHECKPOINT": source, "SAMPLE_MODE": sample_mode, "SAMPLES_PER_PROMPT": str(spp), "SEED_BASE": str(seed_base),
           "NUM_FRAMES": str(frames), "HEIGHT": str(height), "WIDTH": str(width), "FPS": str(fps),
           "STEPS": str(steps), "CFG": str(cfg), "FLOW_SHIFT": str(flow_shift), "DIMENSIONS": ",".join(dims)}
    if backend:
        env["ATTENTION_BACKEND"] = backend
        env["VSA_SPARSITY"] = str(sparsity)
    if limit:
        env["LIMIT"] = str(limit)
    row["provenance"]["env"] = dict(env)
    return row, env


def eval_metrics_from_result(result: dict) -> dict:
    out = {}
    for d, v in (result.get("dimension_means") or {}).items():
        if isinstance(v, (int, float)):
            out[d] = float(v)
    if isinstance(result.get("quality_score"), (int, float)):
        out["quality_score"] = float(result["quality_score"])
    if result.get("videos_scored") is not None:
        out["videos_scored"] = int(result["videos_scored"])
    sf = result.get("temporal_flickering_static_filter") or {}
    if sf.get("kept") is not None:
        out["static_kept"] = int(sf["kept"])
    return out


# --------------------------------------------------------------- training variant
def build_train_variant(form: dict) -> dict:
    """A PLANNED E0029 row from a cloned template plus user edits (no launch)."""
    base = results.get(TRAIN_EXP, form["clone_from"]) if form.get("clone_from") else None
    params = dict((base or {}).get("parameters") or {})
    params.update({k: _coerce(v) for k, v in (form.get("parameters") or {}).items()})
    params = {k: v for k, v in params.items() if v not in (None, "")}
    prov = dict((base or {}).get("provenance") or {})
    env = dict(prov.get("env") or {})
    env.update({k: str(v) for k, v in (form.get("env") or {}).items() if v not in (None, "")})
    for k in list(env):
        if form.get("env", {}).get(k, "keep") is None:
            env.pop(k)
    prov = {"repo": prov.get("repo") or CODE_REPO, "entry": form.get("entry") or prov.get("entry"),
            "env": env, "notes": form.get("notes") or f"Predeclared from the VSQA dashboard on "
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}; not launched."}
    if form.get("commit"):
        prov["commit"] = form["commit"]
    if not form.get("name"):
        raise ValueError("name is required")
    return {"name": form["name"], "status": "PLANNED", "description": form.get("description") or prov["notes"],
            "parameters": params, "metrics": {}, "runs": [], "attempts": [], "provenance": prov}


def _coerce(v):
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        try:
            return int(s) if re.fullmatch(r"-?\d+", s) else float(s) if re.fullmatch(r"-?\d*\.\d+(e-?\d+)?|-?\d+e-?\d+", s, re.I) else s
        except ValueError:
            return s
    return v


def create_variant(exp: str, row: dict) -> dict:
    with _creation_lock:
        vid = results.append_variant(exp, row)
        checks = lint_and_submit(exp)
    log.info("created %s %s: %s", exp, vid, checks)
    return {"id": vid, **checks}
