#!/usr/bin/env python3
"""Render a Variant checkpoint's training validation captions under the real
inference kernel (fastvideo `VSQA`, FVFA4-v3 NVFP4-QK/FP8-PV) and store them in
the Variant's own run directory: `<run>/inf_val/<step>/<idx:02d>.mp4`.

Generation parameters are taken from the run's `recipe.yaml` so the videos are
paired with the trainer's own validation videos: captions (`validation.json`),
seed (`training.data.seed`, same seed for every caption like the trainer),
geometry (`callbacks.validation.{num_frames,height,width}`), steps / DMD ladder
(`callbacks.validation.sampling_steps` + `sampling_timesteps` or the trained
`method.dmd_denoising_steps`), guidance (`callbacks.validation.guidance_scale`),
flow shift (`pipeline.flow_shift`), sparsity (`training.vsa.sparsity`), fps 16
(the Wan preset the trainer's SamplingParam uses).

Runs on a GPU node; the dashboard launches it through `inf_val.sh` inside the
evaluation allocation. `--shard i/n` splits captions across processes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
E0030 = PROJECT_ROOT / "vbench_eval"
CODE_ROOT = Path(os.environ.get("VSQA_CODE_ROOT", PROJECT_ROOT / "fastvideo"))
KIND = "inf-kernel×val12"
FPS = 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="E0029/V<NNNN>@<step>")
    p.add_argument("--out-dir", type=Path, required=True, help="<run>/inf_val/<step>")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--overwrite", action="store_true", help="re-render captions whose mp4 already exists")
    p.add_argument("--attention-backend", default="VSQA")
    return p.parse_args()


def resolve(ref: str) -> dict:
    out = subprocess.run([sys.executable, str(E0030 / "resolve_checkpoint.py"), ref], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"resolve failed ({out.returncode}): {out.stdout}\n{out.stderr}")
    return json.loads(out.stdout)


def run_facts(run_dir: Path) -> dict:
    recipe = yaml.safe_load((run_dir / "recipe.yaml").read_text())
    val = recipe["callbacks"]["validation"]
    # Batch-launched Runs keep no `validation.json` copy; their captions live where the
    # validation callback pointed. Either source must exist, and nothing is guessed.
    caption_file = run_dir / "validation.json"
    if not caption_file.is_file():
        declared = val.get("dataset_file")
        if not declared:
            raise SystemExit(f"{run_dir} has no validation.json and its recipe names no dataset_file")
        caption_file = Path(declared)
        if not caption_file.is_file():
            raise SystemExit(f"{run_dir} validation captions missing: {caption_file} does not exist")
    captions = [d["caption"] for d in json.loads(caption_file.read_text())["data"]]
    method = recipe.get("method") or {}
    ladder = val.get("sampling_timesteps") or method.get("dmd_denoising_steps")
    steps = int((val.get("sampling_steps") or [len(ladder) if ladder else 50])[0])
    if ladder and len(ladder) != steps:
        raise SystemExit(f"validation.sampling_steps {steps} disagrees with ladder {ladder}")
    seed = (recipe.get("training") or {}).get("data", {}).get("seed")
    if seed is None:
        raise SystemExit("training.data.seed missing; cannot pair with the trainer's validation seed")
    return {
        "captions": captions, "num_frames": int(val["num_frames"]), "height": int(val["height"]), "width": int(val["width"]),
        "steps": steps, "ladder": [int(t) for t in ladder] if ladder else None,
        "cfg": float(val.get("guidance_scale") if val.get("guidance_scale") is not None else (1.0 if ladder else 5.0)),
        "flow_shift": float((recipe.get("pipeline") or {}).get("flow_shift") or (8.0 if ladder else 3.0)),
        "sparsity": float(((recipe.get("training") or {}).get("vsa") or {}).get("sparsity", 0.0)),
        "seed": int(seed), "fps": FPS,
    }


def main() -> None:
    args = parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    ckpt = resolve(args.checkpoint)
    if not ckpt.get("model_path"):
        raise SystemExit(f"{args.checkpoint} has no export: {ckpt}")
    if str(ckpt.get("attention_kind") or "").lower() != "vsa":
        raise SystemExit(f"{args.checkpoint} is not a sparse checkpoint (attention_kind={ckpt.get('attention_kind')}); VSQA needs C256/C256T8")
    run_dir = PROJECT_ROOT / ckpt["run"]
    facts = run_facts(run_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": KIND, "checkpoint": args.checkpoint, "resolved": ckpt, "attention_backend": args.attention_backend,
        "provider": "tma_fused_qk_pool_quant_v_pool_fp8_compact_tiled_v1 (fastvideo VSQA, E0026 V0024)",
        "validation": {k: v for k, v in facts.items() if k != "captions"}, "captions": facts["captions"],
        "files": [f"{i:02d}.mp4" for i in range(len(facts["captions"]))],
        "fastvideo_head": subprocess.run(["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"],
                                         capture_output=True, text=True).stdout.strip(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if shard_i == 0:
        (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    mine = [(i, c) for i, c in enumerate(facts["captions"]) if i % shard_n == shard_i]
    todo = [(i, c) for i, c in mine if args.overwrite or not (args.out_dir / f"{i:02d}.mp4").is_file()]
    print(f"[inf_val] shard {shard_i}/{shard_n}: {len(mine)} captions, {len(todo)} to render; "
          f"steps={facts['steps']} ladder={facts['ladder']} cfg={facts['cfg']} shift={facts['flow_shift']} "
          f"seed={facts['seed']} {facts['num_frames']}x{facts['height']}x{facts['width']} sparsity={facts['sparsity']}", flush=True)
    if not todo:
        return

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = args.attention_backend
    if args.attention_backend == "VSQA":
        os.environ.setdefault("CUTE_DSL_ENABLE_TVM_FFI", "1")
    from fastvideo import VideoGenerator

    kw = dict(num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False, vae_cpu_offload=False,
              text_encoder_cpu_offload=True, pin_cpu_memory=True, flow_shift=facts["flow_shift"],
              VSA_sparsity=facts["sparsity"])
    if facts["ladder"]:
        kw["dmd_denoising_steps"] = facts["ladder"]
    gen = VideoGenerator.from_pretrained(ckpt["model_path"], **kw)
    try:
        for n, (i, caption) in enumerate(todo, 1):
            final = args.out_dir / f"{i:02d}.mp4"
            tmp = args.out_dir / f"tmp-{shard_i}-{i:02d}.mp4"
            if tmp.exists():
                tmp.unlink()
            t0 = time.perf_counter()
            gen.generate_video(prompt=caption, output_path=str(tmp), save_video=True, seed=facts["seed"],
                               num_inference_steps=facts["steps"], guidance_scale=facts["cfg"],
                               num_frames=facts["num_frames"], height=facts["height"], width=facts["width"], fps=facts["fps"])
            if not tmp.is_file():
                raise RuntimeError(f"generator did not write {tmp}")
            os.replace(tmp, final)
            print(f"[inf_val] shard {shard_i}: {n}/{len(todo)} {final.name} {time.perf_counter() - t0:.1f}s", flush=True)
    finally:
        gen.shutdown()


if __name__ == "__main__":
    main()
