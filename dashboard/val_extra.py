#!/usr/bin/env python3
"""Render validation captions that were added after a Run finished.

The validation set grew from 12 to 16 captions. A finished Run's own videos stay
untouched: the trainer wrote them rank-sharded, and the dashboard derives their
caption index from `rank * (captions // ranks) + idx`, so rewriting that Run's
`validation.json` to 16 captions would silently re-label every existing video.
This renderer therefore writes only the *new* captions, addressed by caption
index, into `<run>/val_extra/<step>/<caption_idx:02d>.mp4`, which the dashboard
merges into the same step's validation strip.

Everything that must match the trainer's own validation - seed, geometry,
sampling steps / DMD ladder, guidance, flow shift, sparsity - is read from the
Run's `recipe.yaml` by `inf_val.run_facts`. The attention forward defaults to
the Run's own student training backend, so a supplementary video is comparable
with the trainer's videos rather than with an inference-kernel render.

    python -m dashboard.val_extra --checkpoint E0029/V0182@1000 \
        --captions-file logs/<run-with-16>/validation.json --indices 12-15 --shard 0/4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

from .inf_val import PROJECT_ROOT, resolve, run_facts

KIND = "train-kernel×val12"
FPS = 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="E0029/V<NNNN>@<step>")
    p.add_argument("--checkpoints", help="comma-separated E0029/V<NNNN>@<step> list rendered by one "
                                         "process, so torch import and CUDA setup are paid once")
    p.add_argument("--captions-file", type=Path, required=True,
                   help="validation.json holding the caption list to render")
    p.add_argument("--indices", default="12-15", help="caption indices to render, e.g. 12-15 or 0-15")
    p.add_argument("--out-dir", type=Path, help="one checkpoint's output directory")
    p.add_argument("--out-root", help="per-checkpoint output is <run>/<out-root>/<step>; "
                                      "default val_extra or val_forced<N>")
    p.add_argument("--attention-backend", help="default: the Run's student training backend")
    p.add_argument("--force-sampling-steps", type=int,
                   help="render at this sampling-step count instead of the Run's own validation "
                        "setting; the result is a separate comparison, not part of that set")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if not (args.checkpoint or args.checkpoints):
        p.error("one of --checkpoint / --checkpoints is required")
    return args


def parse_indices(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    if not out or len(set(out)) != len(out) or min(out) < 0:
        raise SystemExit(f"invalid --indices {text!r}")
    return sorted(out)


def student_backend(run_dir: Path) -> str:
    recipe = yaml.safe_load((run_dir / "recipe.yaml").read_text())
    backend = (((recipe.get("models") or {}).get("student") or {}).get("attention_backend"))
    if not backend:
        raise SystemExit(f"{run_dir}/recipe.yaml declares no models.student.attention_backend")
    return str(backend)


def strict_alignment(run_dir: Path, facts: dict) -> None:
    """Fail loudly unless the render reproduces this Run's validation settings exactly.

    The trainer's ValidationCallback takes its seed from `training.data.seed` and its output
    geometry from `training.data.num_{frames,height,width}`, while the callback block declares
    the same numbers; a render is only comparable with the Run's own videos when both agree and
    nothing had to be defaulted.
    """
    recipe = yaml.safe_load((run_dir / "recipe.yaml").read_text())
    val = (recipe.get("callbacks") or {}).get("validation") or {}
    data = ((recipe.get("training") or {}).get("data") or {})
    missing = [k for k in ("num_frames", "height", "width", "guidance_scale", "sampling_steps") if val.get(k) is None]
    if missing:
        raise SystemExit(f"{run_dir}/recipe.yaml validation block leaves {missing} unset; refusing to guess them")
    if (recipe.get("pipeline") or {}).get("flow_shift") is None:
        raise SystemExit(f"{run_dir}/recipe.yaml declares no pipeline.flow_shift; refusing to guess it")
    trainer_geometry = (data.get("num_frames"), data.get("num_height"), data.get("num_width"))
    if trainer_geometry != (facts["num_frames"], facts["height"], facts["width"]):
        raise SystemExit(f"validation geometry {facts['num_frames']}x{facts['height']}x{facts['width']} differs from the "
                         f"training data geometry {trainer_geometry} the trainer sampled at")
    if int(data.get("seed")) != facts["seed"]:
        raise SystemExit("seed disagrees with training.data.seed, the trainer's validation seed")


def check_existing(out_dir: Path, facts: dict, backend: str, captions: list[str]) -> None:
    """An earlier supplement in the same directory must have used the same settings."""
    previous = out_dir / "manifest.json"
    if not previous.is_file():
        return
    old = json.loads(previous.read_text())
    drift = {k: (old.get("validation", {}).get(k), v) for k, v in facts.items()
             if k != "captions" and old.get("validation", {}).get(k) != v}
    if old.get("attention_backend") != backend:
        drift["attention_backend"] = (old.get("attention_backend"), backend)
    if old.get("captions") and old["captions"] != captions:
        drift["captions"] = ("different caption list", "current caption list")
    if drift:
        raise SystemExit(f"{previous} was written with different settings {drift}; "
                         "re-render the whole step deliberately instead of mixing two contracts")




def plan_checkpoint(ref: str, args: argparse.Namespace, indices: list[int]) -> dict | None:
    """Everything needed to render one checkpoint, or None when it is already complete."""
    ckpt = resolve(ref)
    if not ckpt.get("model_path"):
        raise SystemExit(f"{ref} has no export; export the checkpoint first: {ckpt}")
    run_dir = PROJECT_ROOT / ckpt["run"]
    facts = run_facts(run_dir)
    captions = [d["caption"] for d in json.loads(args.captions_file.read_text())["data"]]
    trained = facts["captions"]
    forced = args.force_sampling_steps
    if forced is not None and forced == facts["steps"]:
        raise SystemExit(f"--force-sampling-steps {forced} equals this Run's own validation setting; "
                         "that would duplicate the Run's videos instead of adding a comparison")
    # The caption list must extend the Run's own in order, or an index would name a different
    # caption than the one the Run already rendered under that index.
    if captions[:len(trained)] != trained:
        raise SystemExit(f"{args.captions_file} does not extend this Run's {len(trained)} captions in order")
    if max(indices) >= len(captions):
        raise SystemExit(f"index {max(indices)} beyond the {len(captions)} captions in {args.captions_file}")
    if forced is None:
        # Same sampling settings as the trainer: only captions the trainer never rendered.
        overlap = [i for i in indices if i < len(trained)]
        if overlap:
            raise SystemExit(f"indices {overlap} were already rendered by the trainer; refusing to shadow them")
    backend = args.attention_backend or student_backend(run_dir)
    root = args.out_root or ("val_extra" if forced is None else f"val_forced{forced}")
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / root / str(ckpt["step"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    strict_alignment(run_dir, facts)
    render = dict(facts)
    if forced is not None:
        render["steps"] = forced
        # A trained DMD ladder pins its own step count, so forcing a different one drops it.
        render["ladder"] = None
    check_existing(out_dir, {k: v for k, v in render.items() if k != "captions"}, backend, captions)
    (out_dir / "manifest.json").write_text(json.dumps({
        "kind": KIND if forced is None else f"{KIND}@{forced}step",
        "supplement_of": ("trainer validation videos" if forced is None else
                          "separate comparison; not part of the Run's validation set"),
        "checkpoint": ref, "resolved": ckpt, "attention_backend": backend,
        "captions_file": str(args.captions_file), "rendered_indices": indices,
        "trained_captions": len(trained), "forced_sampling_steps": forced,
        "declared_sampling_steps": facts["steps"],
        "validation": {k: v for k, v in render.items() if k != "captions"}, "captions": captions,
        "files": [f"{i:02d}.mp4" for i in indices],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=1) + "\n")
    todo = [i for i in indices if args.overwrite or not (out_dir / f"{i:02d}.mp4").is_file()]
    if not todo:
        return None
    return {"ref": ref, "model_path": ckpt["model_path"], "out_dir": out_dir, "todo": todo,
            "captions": captions, "render": render, "backend": backend, "declared": facts["steps"]}


def render_checkpoint(job: dict, generator_factory) -> None:
    render, out_dir = job["render"], job["out_dir"]
    gen = generator_factory(job)
    try:
        for n, i in enumerate(job["todo"], 1):
            final = out_dir / f"{i:02d}.mp4"
            tmp = out_dir / f"tmp-{i:02d}.mp4"
            tmp.unlink(missing_ok=True)
            t0 = time.perf_counter()
            gen.generate_video(prompt=job["captions"][i], output_path=str(tmp), save_video=True,
                               seed=render["seed"], num_inference_steps=render["steps"],
                               guidance_scale=render["cfg"], num_frames=render["num_frames"],
                               height=render["height"], width=render["width"], fps=render["fps"])
            if not tmp.is_file():
                raise RuntimeError(f"generator did not write {tmp}")
            os.replace(tmp, final)
            print(f"[val_extra] {job['ref']}: {n}/{len(job['todo'])} {final.name} "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
    finally:
        gen.shutdown()


def main() -> None:
    args = parse_args()
    indices = parse_indices(args.indices)
    refs = [r.strip() for r in (args.checkpoints or args.checkpoint).split(",") if r.strip()]
    if args.out_dir and len(refs) > 1:
        raise SystemExit("--out-dir names a single checkpoint's directory; drop it for a multi-checkpoint run")
    # One process for every checkpoint this worker owns: the python/torch import and the CUDA
    # context are paid once instead of once per checkpoint.
    jobs = []
    for ref in refs:
        job = plan_checkpoint(ref, args, indices)
        if job is None:
            print(f"[val_extra] {ref}: already complete; skipping", flush=True)
            continue
        jobs.append(job)
        note = "" if job["render"]["steps"] == job["declared"] else f" (forced; Run validated at {job['declared']})"
        print(f"[val_extra] {ref}: {len(job['todo'])} captions to render; backend={job['backend']} "
              f"steps={job['render']['steps']}{note} ladder={job['render']['ladder']} "
              f"cfg={job['render']['cfg']} shift={job['render']['flow_shift']} seed={job['render']['seed']} "
              f"{job['render']['num_frames']}x{job['render']['height']}x{job['render']['width']} "
              f"sparsity={job['render']['sparsity']} -> {job['out_dir']}", flush=True)
    if not jobs:
        return

    backends = {j["backend"] for j in jobs}
    if len(backends) > 1:
        raise SystemExit(f"one process cannot mix attention backends {backends}; split the checkpoint list")
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = jobs[0]["backend"]
    from fastvideo import VideoGenerator

    def factory(job: dict):
        render = job["render"]
        kw = dict(num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False, vae_cpu_offload=False,
                  text_encoder_cpu_offload=True, pin_cpu_memory=True, flow_shift=render["flow_shift"],
                  VSA_sparsity=render["sparsity"])
        if render["ladder"]:
            kw["dmd_denoising_steps"] = render["ladder"]
        return VideoGenerator.from_pretrained(job["model_path"], **kw)

    failures = []
    for job in jobs:
        try:
            render_checkpoint(job, factory)
        except Exception as exc:  # one bad checkpoint must not drop the rest of the worker's list
            failures.append((job["ref"], f"{type(exc).__name__}: {exc}"))
            print(f"[val_extra] {job['ref']}: FAILED {type(exc).__name__}: {exc}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)}/{len(jobs)} checkpoints failed: {failures}")


if __name__ == "__main__":
    main()
