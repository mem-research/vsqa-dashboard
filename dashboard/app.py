"""VSQA E0029/E0030 dashboard: FastAPI backend serving JSON plus static assets.

Run: `bash dashboard/serve.sh` (or `python -m dashboard.app`).
"""
from __future__ import annotations

import logging
import hashlib
import json
import subprocess
import tempfile
import threading
import time
import mimetypes
from pathlib import Path

import uvicorn
import imageio_ffmpeg
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vbench_eval.e0030_common import CORPUS, QUALITY_DIMS, digest, load_rows

from . import families, model, paper, scheduler, tunnel, variants
from .overview_cache import OverviewCache
from .config import EVAL_EXP, HOST, LOGS_DIR, PORT, PROJECT_ROOT, STATE_DIR, STATIC_DIR, TRAIN_EXP, VBENCH_QUALITY_DIMS, VBENCH_SUBMISSION, WANDB_SCALAR_KEYS
from .sources import results, runs, wandb_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("dashboard")
app = FastAPI(title="VSQA E0029/E0030 dashboard")


def _wandb_targets():
    live_states = {"RUNNING", "PLANNED"}
    for v in results.variants(TRAIN_EXP):
        url = (v.get("metrics") or {}).get("wandb")
        rid = wandb_cache.run_id_from_url(url)
        if rid:
            yield rid, url, v.get("status") in live_states


class _CompressText:
    """gzip everything except video/thumbnail payloads: the tunnel otherwise carries the 6 MB
    overview uncompressed (Caddy only compresses after it), while MP4 range requests must stay
    byte-exact."""

    def __init__(self, app):
        self.app = app
        self.gzip = GZipMiddleware(app, minimum_size=1024, compresslevel=6)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not scope["path"].startswith(("/media/", "/thumbnail/", "/paper.pdf", "/public/")):
            return await self.gzip(scope, receive, send)
        return await self.app(scope, receive, send)


app.add_middleware(_CompressText)


@app.middleware("http")
async def _invalidate_on_write(request, call_next):
    """Any API write may change the overview (queue rows, planned Variants, settings), so the
    next overview request waits for a build that postdates it instead of a cached snapshot."""
    response = await call_next(request)
    if request.method not in ("GET", "HEAD") and request.url.path.startswith("/api/"):
        overview_cache.invalidate()
    return response


@app.middleware("http")
async def _cache_policy(request, call_next):
    """Without an explicit policy browsers heuristically cache index.html/app.js and keep
    serving an old frontend after a deploy. ETag revalidation is cheap, so everything is
    `no-cache` except the immutable vendored plot bundle and the media/thumbnail payloads."""
    response = await call_next(request)
    path = request.url.path
    if path.endswith("plotly.min.js"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif path.startswith(("/media/", "/thumbnail/")):
        response.headers["Cache-Control"] = "public, max-age=3600"
    elif "cache-control" not in response.headers:  # routes such as the paper PDF set their own
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.on_event("startup")
def _startup() -> None:
    app.state.refresher = wandb_cache.Refresher(_wandb_targets)
    app.state.refresher.start()
    app.state.scheduler = scheduler.Loop()
    app.state.scheduler.start()
    app.state.tunnel = tunnel.Tunnel()
    app.state.tunnel.start()
    overview_cache.warm()
    scheduler.event("info", f"dashboard started; tunnel {tunnel.TUNNEL or 'disabled'}")


@app.on_event("shutdown")
def _shutdown() -> None:
    app.state.tunnel.close()


@app.get("/api/tunnel")
def api_tunnel():
    return app.state.tunnel.status


# --------------------------------------------------------------------- read API
def _build_overview() -> dict:
    variants_ = model.overview()  # registers VBench subset kinds, so it runs before DATAPOINT_KINDS is read
    return {"columns": results.columns(TRAIN_EXP), "families": families.family_spec(),
            "filter_groups": families.FILTER_GROUPS,
            "datapoint_kinds": families.DATAPOINT_KINDS, "vbench_submission": VBENCH_SUBMISSION,
            "variants": variants_}


overview_cache = OverviewCache(_build_overview, ttl=60, max_stale=600)


@app.get("/api/overview")
def api_overview(request: Request):
    body, gz, age = overview_cache.get()
    headers = {"X-Overview-Age": f"{age:.1f}", "Vary": "Accept-Encoding"}
    if "gzip" in request.headers.get("accept-encoding", ""):
        return Response(gz, media_type="application/json", headers={**headers, "Content-Encoding": "gzip"})
    return Response(body, media_type="application/json", headers=headers)


@app.get("/api/analysis/{vid}")
def api_analysis(vid: str):
    """Checkpoint-addressed VBench evaluations of one training Variant, keyed by training step.
    These carry no E0030 row, so the dashboard reads them straight from vbench_eval/outputs/."""
    return {"variant": vid, "steps": {str(step): evaluations
                                      for step, evaluations in model.vbench.for_variant(vid).items()}}


@app.get("/api/variant/{vid}")
def api_variant(vid: str):
    v = results.get(TRAIN_EXP, vid) or results.get(EVAL_EXP, vid)
    if not v:
        raise HTTPException(404, vid)
    run = model.latest_run(v, model._wandb_url_index())
    return {"variant": v, "run": run, "parent": model.parent_of(v)}


@app.get("/api/scalars")
def api_scalars(ids: str = Query(...), keys: str | None = None, refresh: bool = False):
    want = keys.split(",") if keys else WANDB_SCALAR_KEYS
    out = {}
    for vid in ids.split(","):
        v = results.get(TRAIN_EXP, vid)
        if not v:
            continue
        url = (v.get("metrics") or {}).get("wandb")
        rid = wandb_cache.run_id_from_url(url)
        if not rid:
            out[vid] = {"error": "no wandb run", "series": {}}
            continue
        if refresh or wandb_cache.meta(rid) is None:
            wandb_cache.refresh(rid, url)
        out[vid] = {"run_id": rid, "meta": wandb_cache.meta(rid), "series": wandb_cache.scalars(rid, want)}
    return out


@app.post("/api/wandb/refresh")
def api_wandb_refresh(ids: str = Query(...)):
    out = {}
    for vid in ids.split(","):
        v = results.get(TRAIN_EXP, vid)
        url = (v.get("metrics") or {}).get("wandb") if v else None
        rid = wandb_cache.run_id_from_url(url)
        out[vid] = wandb_cache.refresh(rid, url) if rid else {"error": "no wandb run"}
    return out


@app.get("/api/videos/{vid}")
def api_videos(vid: str):
    out = model.variant_videos(vid)
    if out is None:
        raise HTTPException(404, vid)
    return out


@app.get("/api/lineage")
def api_lineage():
    return model.lineage()


@app.get("/api/checkpoints")
def api_checkpoints():
    return model.checkpoints()


@app.get("/api/eval")
def api_eval():
    return {"columns": results.columns(EVAL_EXP), "variants": model.eval_table(), "dimensions": VBENCH_QUALITY_DIMS}


@app.get("/api/eval/per_video/{vid}")
def api_eval_per_video(vid: str):
    v = results.get(EVAL_EXP, vid)
    if not v:
        raise HTTPException(404, vid)
    run = model.latest_run(v)
    if not run:
        return {"id": vid, "run": None, "scores": {}}
    return {"id": vid, "run": run["id"], "run_path": run["path"], "scores": runs.per_video_scores(run["path"]) or {}}


@app.get("/api/eval/prompt_counts")
def api_prompt_counts(dims: str = ""):
    return variants.prompt_counts(dims.split(",") if dims else list(VBENCH_QUALITY_DIMS))


@app.get("/api/vbench/prompts")
def api_vbench_prompts():
    """Expose original corpus IDs and the exact evaluation dataset selections."""
    dataset = PROJECT_ROOT / "vbench_eval/datasets/q7tiny.v1.json"
    try:
        corpus = json.loads(CORPUS.read_text())
        definition = json.loads(dataset.read_text())
        q7 = load_rows()
        tiny = load_rows(dataset=dataset)
        return {
            "corpus_path": str(CORPUS.relative_to(PROJECT_ROOT)),
            "corpus_sha256": digest(CORPUS),
            "dataset_path": str(dataset.relative_to(PROJECT_ROOT)),
            "dataset_sha256": digest(dataset),
            "quality_dimensions": list(QUALITY_DIMS),
            "datasets": {
                "all": {"label": "全部", "ids": list(range(len(corpus)))},
                "q7": {"label": "q7", "ids": [r["prompt_idx"] for r in q7]},
                "q7tiny": {"label": f"q7tiny-v{definition['dataset_version']}",
                           "ids": [r["prompt_idx"] for r in tiny]},
            },
            "prompts": [
                {"global_id": idx, "prompt": row["prompt_en"], "dimensions": row["dimension"]}
                for idx, row in enumerate(corpus)
            ],
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.exception("VBench prompt corpus unavailable or inconsistent")
        raise HTTPException(503, "VBench prompt corpus unavailable or inconsistent") from exc


@app.get("/vbench")
@app.get("/vbench/")
def vbench_page():
    return FileResponse(STATIC_DIR / "vbench.html", media_type="text/html")


@app.get("/paper")
@app.get("/paper/")
def paper_page():
    return FileResponse(STATIC_DIR / "paper.html", media_type="text/html")


@app.get("/api/paper")
def api_paper():
    return paper.status()


def _paper_pdf(disposition: str) -> Response:
    snap = paper.snapshot()
    if not snap:
        raise HTTPException(404, f"no complete PDF in {paper.PAPER_DIR}")
    mtime, data = snap
    stamp = time.strftime("%Y%m%d-%H%M", time.gmtime(mtime))
    return Response(data, media_type="application/pdf", headers={
        "Content-Disposition": f'{disposition}; filename="vsqa-paper-{stamp}Z.pdf"',
        "Cache-Control": "no-store", "X-Paper-Mtime": f"{mtime:.3f}"})


@app.api_route("/paper.pdf", methods=["GET", "HEAD"])
def paper_pdf():
    """Inline copy for the /paper viewer (behind the site login like every other route)."""
    return _paper_pdf("inline")


@app.api_route("/public/paper.pdf", methods=["GET", "HEAD"])
def paper_download():
    """Download without the site login: Caddy exempts exactly GET/HEAD on this path from
    basic_auth. Nothing else may be served under /public/."""
    return _paper_pdf("attachment")


@app.get("/api/cluster")
def api_cluster():
    return model.cluster()


@app.get("/api/runs")
def api_runs():
    return [{k: v for k, v in info.items() if k != "validation_videos"} for info in runs.list_all()]


@app.get("/media/{path:path}")
def media(path: str):
    target = (LOGS_DIR / path).resolve()
    if LOGS_DIR.resolve() not in target.parents or not target.is_file():
        raise HTTPException(404, path)
    if target.suffix.lower() not in (".mp4", ".png", ".jpg", ".jpeg", ".json", ".log", ".txt", ".csv", ".yaml"):
        raise HTTPException(403, "unsupported media type")
    mt = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return FileResponse(target, media_type=mt)


_thumbnail_slots = threading.BoundedSemaphore(2)


@app.get("/thumbnail/{path:path}")
def thumbnail(path: str):
    """Cache a small first-frame JPEG without loading MP4s in idle browsers."""
    target = (LOGS_DIR / path).resolve()
    if LOGS_DIR.resolve() not in target.parents or not target.is_file() or target.suffix.lower() != ".mp4":
        raise HTTPException(404, path)
    stat = target.stat()
    key = hashlib.sha256(f"{target}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
    cache = STATE_DIR / "thumbnails"
    image = cache / f"{key}.jpg"
    with _thumbnail_slots:
        if not image.is_file():
            cache.mkdir(parents=True, exist_ok=True)
            try:
                frame = subprocess.run(
                    [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error",
                     "-threads", "1", "-i", str(target), "-frames:v", "1",
                     "-vf", "scale=384:-2", "-threads", "1",
                     "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
                    capture_output=True, check=True, timeout=30,
                ).stdout
                if not frame:
                    raise ValueError("video contains no decodable frame")
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                log.warning("thumbnail extraction failed for %s: %s", target, exc)
                raise HTTPException(422, "could not extract video preview") from exc
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=cache, suffix=".jpg", delete=False) as out:
                    temporary = Path(out.name)
                    out.write(frame)
                temporary.replace(image)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return FileResponse(image, media_type="image/jpeg")


# -------------------------------------------------------------------- write API
class TrainVariantForm(BaseModel):
    clone_from: str | None = None
    name: str
    description: str | None = None
    parameters: dict = {}
    env: dict = {}
    entry: str | None = None
    commit: str | None = None
    notes: str | None = None
    dry_run: bool = False


@app.get("/api/train/template")
def api_train_template(clone_from: str | None = None):
    v = results.get(TRAIN_EXP, clone_from) if clone_from else None
    return {"columns": results.columns(TRAIN_EXP), "next_id": results.next_id(TRAIN_EXP), "template": v,
            "entries": sorted({(x.get("provenance") or {}).get("entry") for x in results.variants(TRAIN_EXP)
                               if (x.get("provenance") or {}).get("entry")})}


@app.post("/api/variants/train")
def api_create_train(form: TrainVariantForm):
    try:
        row = variants.build_train_variant(form.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    if form.dry_run:
        return {"preview": {"id": results.next_id(TRAIN_EXP), **row}}
    return variants.create_variant(TRAIN_EXP, row)


class BranchSource(BaseModel):
    variant: str
    step: int
    ema: bool = False
    run: str | None = None


class BranchForm(TrainVariantForm):
    source: BranchSource


@app.post("/api/recipes/branch")
def api_recipe_branch(form: BranchForm):
    from .recipes import build_branch
    try:
        row = build_branch(form.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    if form.dry_run:
        return {"preview": {"id": results.next_id(TRAIN_EXP), **row}}
    return variants.create_variant(TRAIN_EXP, row)


class EvalVariantForm(BaseModel):
    checkpoint_source: str
    attention_backend: str = ""
    sparsity: float | None = None
    dimensions: list[str] = list(VBENCH_QUALITY_DIMS)
    sample_mode: str = "fixed"
    samples_per_prompt: int = 1
    seed_base: int = 0
    limit: int | None = None
    frames: int = 81
    height: int = 480
    width: int = 832
    fps: int = 16
    sampling_steps: int | None = None
    cfg: float | None = None
    flow_shift: float | None = None
    note: str | None = None
    enqueue: bool = True
    dry_run: bool = False


@app.post("/api/variants/eval")
def api_create_eval(form: EvalVariantForm):
    if not VBENCH_SUBMISSION:
        raise HTTPException(403, scheduler.VBENCH_DISABLED)
    try:
        row, env = variants.build_eval_variant(form.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    if form.dry_run:
        return {"preview": {"id": results.next_id(EVAL_EXP), **row}, "env": env}
    created = variants.create_variant(EVAL_EXP, row)
    if form.enqueue:
        created["queue_id"] = scheduler.enqueue(created["id"], env, form.note)
    return created


class InfValForm(BaseModel):
    ref: str
    overwrite: bool = False
    note: str | None = None


@app.post("/api/inf_val/enqueue")
def api_inf_val_enqueue(form: InfValForm):
    """Queue a VSQA inference-kernel render of one E0029 checkpoint's validation captions."""
    m = model._REF_RE.match(form.ref)
    if not m:
        raise HTTPException(400, "ref must be E0029/V<NNNN>@<step>")
    v = results.get(TRAIN_EXP, m.group(1))
    if not v:
        raise HTTPException(404, m.group(1))
    step = int(m.group(2))
    run_rows = model.variant_runs(v, model._wandb_url_index())
    owner = next((r for r in reversed(run_rows) if any(e["step"] == step for e in (r.get("exports") or []))), None) \
        or next((r for r in reversed(run_rows) if step in (r.get("checkpoint_steps") or [])), None)
    if not owner:
        raise HTTPException(400, f"{form.ref}: no export or checkpoint at step {step} in {[r['id'] for r in run_rows]}")
    env = {"CHECKPOINT": form.ref, "OUT_DIR": f"{owner['path']}/inf_val/{step}", "OVERWRITE": "1" if form.overwrite else "0"}
    return {"queue_id": scheduler.enqueue(m.group(1), env, form.note, kind="inf_val"), "out_dir": env["OUT_DIR"]}


@app.get("/api/queue")
def api_queue():
    return {"settings": scheduler.settings(), "queue": scheduler.queue()}


class SettingsForm(BaseModel):
    eval_allocation_id: int | None = None
    paused: bool | None = None
    num_gpus: int | None = None


@app.post("/api/settings")
def api_settings(form: SettingsForm):
    if form.eval_allocation_id is not None:
        scheduler.set_setting("eval_allocation_id", form.eval_allocation_id or None)
    if form.paused is not None:
        scheduler.set_setting("paused", "1" if form.paused else "0")
    if form.num_gpus is not None:
        scheduler.set_setting("num_gpus", form.num_gpus)
    scheduler.event("info", f"settings updated: {form.model_dump(exclude_none=True)}")
    return scheduler.settings()


@app.post("/api/queue/{qid}/{action}")
def api_queue_action(qid: int, action: str):
    ok = {"cancel": scheduler.cancel, "up": lambda q: scheduler.move(q, "up"), "down": lambda q: scheduler.move(q, "down"),
          "requeue": scheduler.requeue, "kill": scheduler.kill}.get(action)
    if not ok:
        raise HTTPException(400, action)
    try:
        return {"ok": bool(ok(qid))}
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.post("/api/scheduler/tick")
def api_tick():
    return scheduler.tick()


@app.post("/api/queue/enqueue/{vid}")
def api_enqueue_existing(vid: str):
    if not VBENCH_SUBMISSION:
        raise HTTPException(403, scheduler.VBENCH_DISABLED)
    v = results.get(EVAL_EXP, vid)
    if not v:
        raise HTTPException(404, vid)
    env = dict(((v.get("provenance") or {}).get("env") or {}))
    if not env.get("CHECKPOINT"):
        raise HTTPException(400, f"{vid} provenance.env has no CHECKPOINT; cannot launch mechanically")
    return {"queue_id": scheduler.enqueue(vid, env, "enqueued from existing PLANNED row")}


@app.get("/api/events")
def api_events(limit: int = 100):
    con = scheduler._db()
    try:
        return [dict(r) for r in con.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))]
    finally:
        con.close()


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main() -> None:
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
