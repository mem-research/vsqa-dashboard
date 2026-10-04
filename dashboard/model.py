"""Join results.yaml, run directories, the W&B cache and E0030 scores into the
shapes the frontend consumes."""
from __future__ import annotations

import re

from . import families, scheduler
from .config import EVAL_EXP, TRAIN_EXP, VBENCH_QUALITY_DIMS
from .sources import results, runs, vbench, wandb_cache

_REF_RE = re.compile(r"^E0029/(V\d{4})@(\d+)$")
# A run-addressed initializer names the step either through the artifact directory
# (`export-step1000`, `checkpoints/checkpoint-1000`) or through the `<run>@<step>` form
# used by the warm-start launchers; without a step the parent step is unknown, not 0.
_INIT_RE = re.compile(r"^(?:logs/)?([A-Za-z0-9][\w.-]*-\d{6}-\d{6})"
                      r"(?:[ /](?:export-(?:dense-)?step(\d+)(-ema)?|checkpoints/checkpoint-(\d+))|@(\d+))?")
# `V0241@1000` / `V0226/checkpoint-250`: the initializer addresses a sibling Variant's
# checkpoint directly instead of naming its Run directory.
_VARIANT_INIT_RE = re.compile(r"^(V\d{4})(?:@|/checkpoint-)(\d+)")

OVERVIEW_PARAMS = ["scope", "objective", "backend", "cube_shape", "attention_kind", "quantization", "steps",
                   "effective_batch", "batch_per_rank", "grad_accum", "generator_update_interval", "dmd_steps",
                   "student_lr", "critic_lr", "sparsity", "frames", "height", "width", "model", "init_kind",
                   "init_step", "dense_qat_step", "sparse_ft_step", "training_path", "ema_decay", "seed",
                   "validation_every", "sampling_schedule", "dataset"]


def _wandb_url_index() -> dict[str, str]:
    idx = {}
    for info in runs.list_all():
        wb = info.get("wandb") or {}
        if wb.get("url"):
            idx[wb["url"]] = info["id"]
    return idx


def latest_run(v: dict, url_index: dict[str, str] | None = None) -> dict | None:
    for ref in reversed(v.get("runs") or []):
        info = runs.scan(ref)
        if info:
            return info
    url = (v.get("metrics") or {}).get("wandb")
    if url and url_index and url in url_index:
        return runs.scan(url_index[url])
    return None


def owner_of_run(run_ref: str) -> tuple[str | None, str | None]:
    """(variant id, matched_by) for a run directory: results.yaml runs[] first, then W&B URL, then a
    unique run-less Variant sharing the run's launcher entry. None when nothing binds it."""
    for other in results.variants(TRAIN_EXP):
        if run_ref in (other.get("runs") or []):
            return other["id"], "runs"
        if run_ref in (other.get("attempts") or []):
            return other["id"], "attempts"
    info = runs.scan(run_ref)
    if not info:
        return None, None
    url = (info.get("wandb") or {}).get("url")
    if url:
        for other in results.variants(TRAIN_EXP):
            if (other.get("metrics") or {}).get("wandb") == url:
                return other["id"], "wandb_url"
    entry = ((info.get("readme") or {}).get("entry") or "")
    if entry:
        cands = [o for o in results.variants(TRAIN_EXP)
                 if not o.get("runs") and (o.get("provenance") or {}).get("entry") == entry and not is_smoke(o)]
        if len(cands) == 1:
            return cands[0]["id"], "entry"
    return None, None


def parent_of(v: dict) -> dict | None:
    checkpoint = (v.get("provenance") or {}).get("checkpoint_parent")
    if checkpoint:
        return {"variant": checkpoint["variant"], "step": checkpoint["step"],
                "ema": checkpoint.get("ema", False), "run": checkpoint.get("run"),
                "matched_by": "checkpoint_parent"}
    init = str((v.get("parameters") or {}).get("initializer") or "")
    vm = _VARIANT_INIT_RE.match(init)
    if vm:
        owner = results.get(TRAIN_EXP, vm.group(1))
        if owner:
            run = next((r for r in reversed(owner.get("runs") or []) if runs.scan(r)), None)
            return {"variant": vm.group(1), "step": int(vm.group(2)), "ema": False,
                    "run": run, "matched_by": "variant_checkpoint"}
    m = _INIT_RE.match(init)
    if not m:
        return {"external": init} if init else None
    run_ref = "logs/" + m.group(1)
    step_text = m.group(2) or m.group(4) or m.group(5)
    step = int(step_text) if step_text else None
    owner, how = owner_of_run(run_ref)
    if owner:
        return {"variant": owner, "run": run_ref, "step": step, "ema": bool(m.group(3)), "matched_by": how}
    return {"run": run_ref, "step": step, "ema": bool(m.group(3))}


def eval_variants() -> list[dict]:
    """E0030 rows excluding pipeline smokes (named smoke/probe or ≤ 4 prompts)."""
    out = []
    for e in results.variants(EVAL_EXP):
        p = e.get("parameters") or {}
        prompts = p.get("prompts")
        if _SMOKE_RE.search(str(e.get("name", ""))) or (isinstance(prompts, (int, float)) and prompts <= 4):
            continue
        out.append(e)
    return out


def eval_rows_for(vid: str) -> list[dict]:
    out = []
    for e in eval_variants():
        p = e.get("parameters") or {}
        src = str(p.get("checkpoint_source") or "")
        m = _REF_RE.match(src)
        if m and m.group(1) == vid:
            kind, err = families.classify_eval(p)
            out.append({"id": e["id"], "step": int(m.group(2)), "status": e.get("status"), "kind": kind, "kind_error": err,
                        "backend": _backend_tag(e), "attention_backend": p.get("attention_backend"),
                        "quality_score": (e.get("metrics") or {}).get("quality_score"),
                        "metrics": e.get("metrics") or {}, "prompts": p.get("prompts"), "sample_mode": p.get("sample_mode"),
                        "runs": e.get("runs") or []})
    return out


def timeline(v: dict, run_rows: list[dict], evals: list[dict], pending: dict | None = None,
             analysis: dict[int, list[dict]] | None = None) -> dict:
    """Per-row timeline: run segments, checkpoints, exports and datapoints on the training-step axis."""
    p = v.get("parameters") or {}
    target = families._num(p.get("steps")) or 0
    segments, ckpts, exports, dps = [], set(), {}, []
    cursor = families._num(p.get("resume_step")) or 0
    for r in run_rows:
        end = max([(r["progress"] or {}).get("step") or 0,
                   *(r["checkpoint_steps"] or []), *(r["validation_steps"] or [])] + [cursor])
        segments.append({"run": r["id"], "state": r["state"], "start": cursor, "end": end, "attempt": r.get("attempt", False)})
        cursor = end
        ckpts.update(r["checkpoint_steps"] or [])
        for ex in r["exports"] or []:
            exports.setdefault(ex["step"], {"step": ex["step"], "run": r["id"], "ema": False, "plain": False})
            exports[ex["step"]]["ema" if ex["ema"] else "plain"] = True
        for s in r["validation_steps"] or []:
            dps.append({"step": s, "kind": "train-kernel×val12", "run": r["id"], "status": "COMPLETED"})
        for s, iv in (runs.scan(r["path"]) or {}).get("inf_val", {}).items():
            status = "COMPLETED" if iv["count"] >= iv["expected"] and iv["expected"] else ("RUNNING" if iv["state"] == "RUNNING" else "FAILED" if iv["state"] == "FAILED" else "PARTIAL")
            dps.append({"step": int(s), "kind": "inf-kernel×val12", "run": r["id"], "status": status, "count": iv["count"],
                        "expected": iv["expected"], "provider": iv["provider"], "ref": f"E0029/{v['id']}@{s}",
                        "out_dir": f"{r['path']}/inf_val/{s}"})
        for s, fv in (runs.scan(r["path"]) or {}).get("forced_val", {}).items():
            steps_forced = fv.get("sampling_steps")
            dps.append({"step": int(s), "kind": f"train-kernel×val12@{steps_forced}step", "run": r["id"],
                        "status": "COMPLETED" if fv["count"] >= fv["expected"] else "PARTIAL",
                        "count": fv["count"], "expected": fv["expected"],
                        "forward": fv.get("backend"),
                        "provider": f"forced {steps_forced} sampling steps "
                                    f"(this Run validated at {fv.get('declared_sampling_steps')})"})
    for e in evals:
        # One row per evaluation: same checkpoint, same dataset, different forward stays two datapoints.
        dps.append({"step": e["step"], "kind": e["kind"], "error": e["kind_error"], "eval": e["id"], "status": e["status"],
                    "quality_score": e["quality_score"], "metrics": e["metrics"], "prompts": e["prompts"],
                    "backend": e["backend"], "attention_backend": e["attention_backend"],
                    "run": e["runs"][-1] if e["runs"] else None, "ref": f"E0029/{v['id']}@{e['step']}"})
    # Checkpoint-addressed evaluations (vbench_eval/outputs/): the evaluator writes scores next to
    # the videos and deliberately creates no E0030 row, so they are read straight from disk. Same
    # dataset lane as a declared row on the same prompt set, with the forward kept per datapoint.
    # `analysis` is scanned once per table build; scanning it per row cost ~0.4 s each.
    for step, evaluations in (analysis if analysis is not None else vbench.for_variant(v["id"])).items():
        for e in evaluations:
            seeds = e["seeds"] or []
            dps.append({"step": step, "kind": families.vbench_kind(e["prompt_set"]), "status": "COMPLETED",
                        "quality_score": e["quality_score"],
                        "metrics": {**e["dimension_means"], "quality_score": e["quality_score"],
                                    "videos_scored": e["videos_scored"]},
                        "count": e["videos_scored"], "expected": e["videos_scored"],
                        "prompts": e["videos_scored"] // len(seeds) if seeds else e["videos_scored"],
                        "seeds": seeds, "analysis": True, "out_dir": e["out_dir"], "result": e["path"],
                        "attention_backend": e["attention_backend"],
                        "backend": families.forward_tag({"attention_backend": e["attention_backend"],
                                                         "attention_kind": "vsa" if e["cube_shape"] else "dense"}),
                        "provider": f"checkpoint-addressed · {len(seeds)} seed{'s' if len(seeds) != 1 else ''}"
                                    f" · {e['sampling_steps']} steps · {e['geometry']}",
                        "ref": f"E0029/{v['id']}@{step}"})
    # Placeholders: every export step is evaluable under each registered kind the checkpoint fits.
    # A VBench dataset lane names no forward, so a placeholder never picks one: it only reports the
    # explicit source context (a sparse checkpoint's default inference kernel, or the kernel this
    # checkpoint was trained with) and the planning dialog stays where the forward is chosen.
    sparse = str(p.get("attention_kind") or "").lower() == "vsa"
    cube = str(p.get("cube_shape") or "")
    vsqa_ok = sparse and cube.split(" ")[0] in ("4x8x8", "8x4x8") and families.MODEL_SET.get(str(p.get("model") or "").lower()) == "1.3B"
    trained_with = families.forward_tag({"attention_backend": p.get("backend"), "attention_kind": p.get("attention_kind")})
    forward = ("VSQA inference kernel (default for this sparse checkpoint)" if vsqa_ok
               else f"pick one in the planning dialog; this checkpoint was trained with {trained_with}")
    have = {(d["step"], d["kind"]) for d in dps}
    pending = pending or {k: scheduler.pending_for(k) for k in ("inf_val", "vbench")}
    for s in sorted(set(exports) | ckpts):
        ref = f"E0029/{v['id']}@{s}"
        for kind, spec in families.DATAPOINT_KINDS.items():
            source = spec.get("source")
            # `forced_val` renders are produced offline by dashboard/val_extra.sh, not by the
            # queue, so that lane shows only what actually exists and offers nothing to click.
            if source in ("wandb", "forced_val") or (s, kind) in have:
                continue
            if source == "inf_val" and not vsqa_ok:  # rendered by the VSQA inference kernel only
                continue
            if source == "e0030" and not (vsqa_ok or not sparse):  # no forward fits this checkpoint
                continue
            q = pending["inf_val" if source == "inf_val" else "vbench"].get(ref)
            dps.append({"step": s, "kind": kind, "status": "QUEUED" if q and q["status"] == "queued" else "RUNNING" if q else "EVALUABLE",
                        "ref": ref, "queue_id": q["queue_id"] if q else None, "needs_export": s not in exports,
                        "forward": forward})
    for d in dps:  # overwrite re-renders queued on an existing dp
        if d["kind"] == "inf-kernel×val12" and d.get("ref") and d["status"] in ("COMPLETED", "PARTIAL", "FAILED"):
            q = pending["inf_val"].get(d["ref"])
            if q:
                d["requeued"] = q["status"]; d["queue_id"] = q["queue_id"]
    return {"target": target, "segments": segments, "checkpoints": sorted(ckpts),
            "exports": [exports[s] for s in sorted(exports)], "datapoints": sorted(dps, key=lambda d: (d["step"], d["kind"] or ""))}


def resume_source(v: dict) -> dict:
    """Where the Variant's weights come from: a Variant checkpoint, base Wan, or an external artifact."""
    par = parent_of(v)
    p = v.get("parameters") or {}
    if par is None:
        return {"kind": "base", "label": f"base {families.MODEL_SET.get(str(p.get('model') or '').lower(), 'Wan')}"}
    if par.get("variant"):
        step = par.get("step")
        return {"kind": "variant", "variant": par["variant"], "step": step, "ema": par.get("ema"), "run": par.get("run"),
                "label": f"{par['variant']}@{step if step is not None else '?'}{' ema' if par.get('ema') else ''}"}
    ext = par.get("external") or par.get("run") or ""
    if re.search(r"Wan-AI|Wan2\.1-T2V", ext):
        return {"kind": "base", "label": f"base {families.MODEL_SET.get(str(p.get('model') or '').lower(), 'Wan')}"}
    return {"kind": "external", "label": ext, "step": par.get("step")}


def _backend_tag(e: dict) -> str:
    return families.forward_tag(e.get("parameters") or {})


_SMOKE_RE = re.compile(r"\b(smoke|probe)\b", re.I)


def is_smoke(v: dict) -> bool:
    """Smoke/probe Variants are excluded from the panel: target ≤ 2 updates or named as such."""
    p = v.get("parameters") or {}
    steps = p.get("steps")
    if isinstance(steps, (int, float)) and steps <= 2:
        return True
    return bool(_SMOKE_RE.search(f"{v.get('name', '')} {p.get('scope', '')}"))


def train_variants() -> list[dict]:
    return [v for v in results.variants(TRAIN_EXP) if not is_smoke(v)]


def _run_row(info: dict) -> dict:
    ex = info.get("execution") or {}
    return {"id": info["id"], "path": info["path"], "state": runs.effective_status(info), "exec": ex,
            "exit_code": info.get("exit_code"), "trainer_exit_code": info.get("trainer_exit_code"),
            "progress": info.get("progress") or {}, "log_mtime": info.get("log_mtime"),
            "gpu_peak_mib": info.get("gpu_peak_mib"), "gpu_latest": info.get("gpu_latest"),
            "checkpoint_steps": info.get("checkpoint_steps"), "exports": info.get("exports"),
            "validation_steps": info.get("validation_steps"), "code_head": info.get("code_head"),
            "readme_status": (info.get("readme") or {}).get("status"),
            "wandb_url": (info.get("wandb") or {}).get("url")}


def variant_warnings(v: dict, run_rows: list[dict]) -> list[str]:
    """Contract checks: one Variant <-> one W&B run; every listed Run directory must exist."""
    warns = []
    declared = (v.get("metrics") or {}).get("wandb")
    urls = {r["wandb_url"] for r in run_rows if r.get("wandb_url")}
    if declared:
        urls.add(declared)
    if len(urls) > 1:
        warns.append("multiple W&B runs for one Variant: " + ", ".join(sorted(u.rsplit("/", 1)[-1] for u in urls)))
    for r in run_rows:
        if r.get("wandb_url") and declared and r["wandb_url"] != declared:
            warns.append(f"run {r['id']} logs to {r['wandb_url'].rsplit('/', 1)[-1]} but results.yaml declares {declared.rsplit('/', 1)[-1]}")
    listed = set(v.get("runs") or []) | set(v.get("attempts") or [])
    found = {r["path"] for r in run_rows}
    for ref in sorted(listed - found):
        warns.append(f"listed run directory missing: {ref}")
    par = parent_of(v)
    if par and par.get("run") and not par.get("variant"):
        warns.append(f"initializer run {par['run'].split('/')[-1]} is not bound to any Variant in results.yaml")
    for r in run_rows:
        if r.get("matched_by") == "wandb_url":
            warns.append(f"results.yaml runs[] is empty; {r['id']} matched through the W&B URL")
        elif r.get("matched_by") == "entry":
            warns.append(f"results.yaml runs[] is empty; {r['id']} matched only by launcher entry (bind it)")
    return warns


def variant_runs(v: dict, url_index: dict[str, str] | None = None) -> list[dict]:
    """Every Run directory of a Variant in results.yaml order (resumes/continuations included)."""
    out = []
    seen = set()
    # `runs[]` are the accepted Run directories; `attempts[]` are stopped/superseded executions of the
    # same Variant (memon keeps them for provenance). Both carry real checkpoints and validation videos.
    for ref, attempt in [(r, False) for r in v.get("runs") or []] + [(r, True) for r in v.get("attempts") or []]:
        info = runs.scan(ref)
        if info and info["id"] not in seen:
            seen.add(info["id"])
            out.append({**_run_row(info), "attempt": attempt})
    if not out:
        url = (v.get("metrics") or {}).get("wandb")
        if url and url_index and url in url_index:
            info = runs.scan(url_index[url])
            if info:
                out.append({**_run_row(info), "matched_by": "wandb_url"})
    if not out and not is_smoke(v):
        entry = (v.get("provenance") or {}).get("entry")
        for info in runs.list_all():
            if entry and (info.get("readme") or {}).get("entry") == entry and owner_of_run(info["path"]) == (v["id"], "entry"):
                out.append({**_run_row(info), "matched_by": "entry"})
    return out


def overview() -> list[dict]:
    url_index = _wandb_url_index()
    for e in eval_variants():  # register every VBench subset kind before placeholders are computed
        families.classify_eval(e.get("parameters") or {})
    pending = {k: scheduler.pending_for(k) for k in ("inf_val", "vbench")}
    wandb_meta = wandb_cache.all_meta()
    # One scan of vbench_eval/outputs for the whole table, grouped by training Variant.
    analysis: dict[str, dict[int, list[dict]]] = {}
    for (exp, vid, step), evaluations in vbench.scan().items():
        if exp == TRAIN_EXP.split("-")[0]:
            analysis.setdefault(vid, {})[step] = evaluations
    rows = []
    for v in train_variants():
        p = v.get("parameters") or {}
        m = v.get("metrics") or {}
        run_rows = variant_runs(v, url_index)
        live = next((r for r in run_rows if r["state"] == "RUNNING"), None) or (run_rows[-1] if run_rows else None)
        wb_url = m.get("wandb")
        wb_id = wandb_cache.run_id_from_url(wb_url)
        wb = wandb_meta.get(wb_id)
        max_step = max([r["progress"].get("step", 0) or 0 for r in run_rows] +
                       [s for r in run_rows for s in (r["validation_steps"] or [])] +
                       [s for r in run_rows for s in (r["checkpoint_steps"] or [])] + [0])
        evals = eval_rows_for(v["id"])
        rows.append({
            "id": v["id"], "name": v.get("name"), "status": v.get("status"), "description": v.get("description"),
            "params": {k: p.get(k) for k in OVERVIEW_PARAMS if k in p},
            "families": families.compute_families(p),
            "filters": families.row_filters(p),
            "metrics": {k: m[k] for k in m if k != "wandb"},
            "wandb": {"url": wb_url, "id": wb_id, "state": (wb or {}).get("state"),
                      "summary_step": (wb or {}).get("summary_step"), "cached_step": (wb or {}).get("last_step"),
                      "summary": (wb or {}).get("summary") or {}, "error": (wb or {}).get("error")},
            "run": live,
            "run_rows": run_rows,
            "max_step": max_step,
            "runs": v.get("runs") or [],
            "warnings": variant_warnings(v, run_rows),
            "parent": parent_of(v),
            "resume": resume_source(v),
            "evals": evals,
            "timeline": timeline(v, run_rows, evals, pending, analysis.get(v["id"], {})),
        })
    children: dict[str, list[dict]] = {}
    for r in rows:
        if r["resume"]["kind"] == "variant":
            children.setdefault(r["resume"]["variant"], []).append({"id": r["id"], "step": r["resume"]["step"]})
    for r in rows:
        r["children"] = children.get(r["id"], [])
    return rows


def lineage() -> dict:
    from .recipes import graph
    return graph()


def checkpoints() -> list[dict]:
    """Every E0029 (variant, step) with a DCP checkpoint or export, for the eval form and matrix."""
    out = []
    url_index = _wandb_url_index()
    evals = {}
    for e in results.variants(EVAL_EXP):
        src = str((e.get("parameters") or {}).get("checkpoint_source") or "")
        m = _REF_RE.match(src)
        if m:
            evals.setdefault((m.group(1), int(m.group(2))), []).append(
                {"id": e["id"], "status": e.get("status"), "backend": _backend_tag(e),
                 "quality_score": (e.get("metrics") or {}).get("quality_score"),
                 "prompts": (e.get("parameters") or {}).get("prompts"),
                 "sample_mode": (e.get("parameters") or {}).get("sample_mode")})
    for v in train_variants():
        p = v.get("parameters") or {}
        steps: dict[int, dict] = {}
        for ref in list(v.get("runs") or []) + list(v.get("attempts") or []):
            info = runs.scan(ref)
            if not info:
                continue
            for s in info.get("checkpoint_steps") or []:
                steps.setdefault(s, {"checkpoint": False, "export": False, "ema_export": False, "run": info["id"]})["checkpoint"] = True
            for ex in info.get("exports") or []:
                d = steps.setdefault(ex["step"], {"checkpoint": False, "export": False, "ema_export": False, "run": info["id"]})
                d["ema_export" if ex["ema"] else "export"] = True
        if not steps:
            run = latest_run(v, url_index)
            if run:
                for s in run.get("checkpoint_steps") or []:
                    steps.setdefault(s, {"checkpoint": True, "export": False, "ema_export": False, "run": run["id"]})
                for ex in run.get("exports") or []:
                    d = steps.setdefault(ex["step"], {"checkpoint": False, "export": False, "ema_export": False, "run": run["id"]})
                    d["ema_export" if ex["ema"] else "export"] = True
        for s in sorted(steps):
            out.append({"variant": v["id"], "name": v.get("name"), "variant_status": v.get("status"), "step": s,
                        "ref": f"E0029/{v['id']}@{s}", "objective": p.get("objective"), "cube_shape": p.get("cube_shape"),
                        "attention_kind": p.get("attention_kind"), "backend": p.get("backend"),
                        "frames": p.get("frames"), "height": p.get("height"), "width": p.get("width"),
                        **steps[s], "evals": evals.get((v["id"], s), [])})
    return out


def eval_table() -> list[dict]:
    rows = []
    for e in results.variants(EVAL_EXP):
        p = e.get("parameters") or {}
        m = e.get("metrics") or {}
        run = latest_run(e)
        src = str(p.get("checkpoint_source") or "")
        ref = _REF_RE.match(src)
        rows.append({"id": e["id"], "name": e.get("name"), "status": e.get("status"), "checkpoint_source": src,
                     "train_variant": ref.group(1) if ref else None, "step": int(ref.group(2)) if ref else None,
                     "backend": _backend_tag(e), "attention_backend": p.get("attention_backend"),
                     "cube_shape": p.get("cube_shape"), "sparsity": p.get("sparsity"),
                     "dimensions": p.get("dimensions"), "prompts": p.get("prompts"), "sample_mode": p.get("sample_mode"),
                     "videos": p.get("videos"), "geometry": f"{p.get('frames')}x{p.get('height')}x{p.get('width')}",
                     "sampling": f"{p.get('sampling_steps')} steps x CFG {p.get('cfg')} / shift {p.get('flow_shift')}",
                     "metrics": {k: m.get(k) for k in [*VBENCH_QUALITY_DIMS, "quality_score", "videos_scored", "static_kept",
                                                        "gen_seconds_per_video"] if k in m},
                     "runs": e.get("runs") or [],
                     "run": None if not run else {"id": run["id"], "state": runs.effective_status(run),
                                                  "video_count": run.get("video_count"), "has_result": bool(run.get("result")),
                                                  "has_per_video": run.get("has_per_video"), "exec": run.get("execution")},
                     "provenance_env": (e.get("provenance") or {}).get("env") or {}})
    return rows


def variant_videos(vid: str) -> dict | None:
    """Validation videos of every Run of the Variant, keyed by step; a later Run wins on a duplicate step."""
    v = results.get(TRAIN_EXP, vid)
    if not v:
        return None
    rows = variant_runs(v, _wandb_url_index())
    videos: dict[str, list] = {}
    inf_val: dict[str, list] = {}
    forced_val: dict[str, list] = {}
    captions: list[str] = []
    for r in rows:
        info = runs.scan(r["path"])
        if not info:
            continue
        captions = info.get("captions") or captions
        for step, vids in (info.get("validation_videos") or {}).items():
            videos[step] = [{**x, "run": info["id"]} for x in vids]
        for step, iv in (info.get("inf_val") or {}).items():
            inf_val[step] = [{**x, "run": info["id"]} for x in iv["videos"]]
        for step, fv in (info.get("forced_val") or {}).items():
            forced_val[step] = [{**x, "run": info["id"], "backend": fv.get("backend")} for x in fv["videos"]]

    return {"id": vid, "runs": [r["id"] for r in rows], "run": rows[-1]["id"] if rows else None,
            "captions": captions, "steps": sorted(int(s) for s in videos), "videos": videos,
            "inf_val": inf_val, "forced_val": forced_val}


def cluster() -> dict:
    from .sources import slurm
    allocs = slurm.allocations()
    by_job: dict[int, list[dict]] = {}
    for info in runs.list_all():
        ex = info.get("execution") or {}
        if ex.get("state") == "RUNNING":
            by_job.setdefault(ex.get("job") or -1, []).append({
                "id": info["id"], "kind": info["kind"], "host": ex.get("host"), "step": ex.get("step"),
                "progress": info.get("progress"), "gpu_latest": info.get("gpu_latest"), "log_mtime": info.get("log_mtime")})
    variant_by_run = {}
    for exp in (TRAIN_EXP, EVAL_EXP):
        for v in results.variants(exp):
            for r in v.get("runs") or []:
                variant_by_run[r.split("/")[-1]] = v["id"]
    for lst in by_job.values():
        for r in lst:
            r["variant"] = variant_by_run.get(r["id"])
    for a in allocs:
        a["runs"] = by_job.get(a["job"], [])
    return {"allocations": allocs, "unassigned": by_job.get(-1, [])}


