"""Parameter families for the Variant table and the datapoint-kind registry.

A family maps raw `results.yaml` parameters to an expanded column set and one
collapsed display value. Two rule types exist:
  * set mapping  - raw value must be in a registry; otherwise the cell is an error;
  * format       - a string built from the expanded values, missing pieces omitted.
Everything user-visible is defined here so the frontend stays declarative.
"""
from __future__ import annotations

import re

RANKS_PER_NODE = 4

# ------------------------------------------------------------ set registries
MODEL_SET = {
    "wan-t2v-1.3b": "1.3B", "wan2.1-t2v-1.3b": "1.3B", "wan2.1-t2v-1.3b-diffusers": "1.3B",
    "wan2.1-t2v-14b": "14B", "wan-t2v-14b": "14B",
}
DATASET_RULES = [  # (regex on the raw dataset string, canonical name); first match wins
    # The ODE-init and VD caches carry "mixkit"/"mixed" in their names, so they must be
    # matched before the plain mixkit rules or they are silently labelled `mixkit`.
    (re.compile(r"vd-32k-61f", re.I), "vd-32k-61f"),
    (re.compile(r"ode-init-mixed-1\.5k", re.I), "ode-mixed-1.5k-61f"),
    (re.compile(r"ode-init-mixkit-4k", re.I), "ode-mixkit-4k-61f"),
    (re.compile(r"ode-init-mixkit-1k", re.I), "ode-mixkit-1k-61f"),
    (re.compile(r"vidprom-self-forcing", re.I), "vidprom-sf"),
    (re.compile(r"(?:^|/)mixed(?:$|/)", re.I), "mixed"),
    (re.compile(r"mixkit[^/]*61f", re.I), "mixkit-61f"),
    (re.compile(r"mixkit[^/]*93f", re.I), "mixkit-93f"),
    (re.compile(r"768p", re.I), "mixkit768p"),
    (re.compile(r"^mixkit$|mixkit-t2v(?!-(61f|93f))|mixkit(?![-_]?(61f|93f|768p))", re.I), "mixkit"),
]
# What the training data itself supplies. Expanded-only: it never shortens the collapsed label.
DATA_ROLE_SET = {
    "text_conditioning_and_batch_geometry": "prompts only",
    "ground_truth_video": "GT video",
    "offline_real_video_teacher_velocity": "teacher velocity cache",
    "cached_teacher_ode_endpoint_pairs": "ODE endpoint pairs",
    "cached_teacher_ode_start_states": "ODE start states",
}
CUBE_SET = {"4x4x4": ("C64", 64, 4, 4, 4), "4x4x8": ("C128", 128, 4, 4, 8),
            "4x8x8": ("C256", 256, 4, 8, 8), "8x4x8": ("C256T8", 256, 8, 4, 8),
            "none": ("dense", None, None, None, None), "dense": ("dense", None, None, None, None)}
MM_BF16_FAKE, MM_NATIVE, MM_BF16 = "BF16 fake", "native NVFP4·FP8", "BF16"
# backend -> (attention kernel family, qk quant, pv quant, matrix products, pv scale trick default).
# The cube is a separate family and the PV precision is appended by the collapsed label, so the
# family name carries neither a C64/C256/C256T8 nor an FP8PV suffix.
KERNEL_SET = {
    "ATTN_QAT_FP8_PV_TRAIN": ("ATTN-QAT", "nvfp4", "fp8", MM_BF16_FAKE, True),
    "ATTN_QAT_TRAIN": ("ATTN-QAT", "nvfp4", "nvfp4", MM_BF16_FAKE, False),
    "VSA_QAT_FP8_PV_TRAIN": ("VSA-QAT", "nvfp4", "fp8", MM_BF16_FAKE, True),
    "VSA_QAT_FP8_PV_TRAIN_C256": ("VSA-QAT", "nvfp4", "fp8", MM_BF16_FAKE, True),
    "VSA_QAT_FP8_PV_TRAIN_C256T8": ("VSA-QAT", "nvfp4", "fp8", MM_BF16_FAKE, True),
    "VSA_QAT_FP8_PV_TRAIN_C256_T8": ("VSA-QAT", "nvfp4", "fp8", MM_BF16_FAKE, True),
    "VSA_QAT_FP8_PV_INFER_FAKE_TRAIN": ("VSA-QAT", "nvfp4", "fp8", MM_BF16_FAKE, False),
    "VSA_QAT_FP8_PV_INFER_FAKE_TRAIN_C256T8": ("VSA-QAT", "nvfp4", "fp8", MM_BF16_FAKE, False),
    "VSA_QAT_FP8_PV_INFER_NATIVE_TRAIN_C256T8": ("VSA-QAT", "nvfp4", "fp8", MM_NATIVE, False),
    "VSA_QAT_TRAIN": ("VSA-QAT", "nvfp4", "nvfp4", MM_BF16_FAKE, False),
    "VSA_QAT_TRAIN_C256T8": ("VSA-QAT", "nvfp4", "nvfp4", MM_BF16_FAKE, False),
    "VSA_QAT_TRAIN_C128": ("VSA-QAT", "nvfp4", "nvfp4", MM_BF16_FAKE, False),
    "VSA_TRITON_TRAIN_C256T8": ("VSA-BF16", "bf16", "bf16", MM_BF16, False),
}
OBJECTIVE_SET = {"dense_qat": "Dense QAT", "fine_tune": "Sparse QAT", "dmd": "DMD", "dmd2": "DMD",
                 "teacher_velocity_mse": "VD", "ode_init_endpoint_regression": "ODE-init", "cd_init": "CD-init",
                 "fixed_teacher_attention_reconstruction": "Attn-recon",
                 "fixed_teacher_layer_reconstruction": "Layer-recon"}
# Linear-layer quantization, independent of the attention kernel. Unset means BF16 linears:
# before FastVideo commit 712ba502 no run could start with a linear quant config.
LINEAR_SET = {"nvfp4_qat_train": "NVFP4 W4A4"}
# An explicit declaration that the linears stay BF16; same meaning as leaving the field unset.
LINEAR_NONE = {"none", "no", "bf16", "n/a"}


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        m = re.match(r"\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)", v, re.I)
        if m:
            f = float(m.group(1))
            return int(f) if f.is_integer() and "e" not in m.group(1).lower() and "." not in m.group(1) else f
    return None


def _lr(v):
    n = _num(v)
    return None if n is None else f"{n:.0e}".replace("e-0", "e-")


# -------------------------------------------------------------- families
def fam_model(p: dict) -> dict:
    raw_model = str(p.get("model") or "").strip()
    model = MODEL_SET.get(raw_model.lower())
    raw_ds = str(p.get("dataset") or "")
    dataset = next((name for rx, name in DATASET_RULES if rx.search(raw_ds)), None)
    err = []
    if raw_model and model is None:
        err.append(f"model {raw_model!r} not in MODEL_SET")
    if raw_ds and dataset is None:
        err.append(f"dataset {raw_ds!r} matches no DATASET_RULES")
    f, h, w = _num(p.get("frames")), _num(p.get("height")), _num(p.get("width"))
    geo = f"{f}x{h}x{w}" if None not in (f, h, w) else None
    role_raw = str(p.get("data_role") or "").strip()
    return {"expanded": {"model": model or raw_model or None, "dataset": dataset or raw_ds or None,
                         "data_role": DATA_ROLE_SET.get(role_raw, role_raw or None),
                         "frames": f, "height": h, "width": w},
            "collapsed": " · ".join(x for x in (model, dataset) if x) or None,
            "error": "; ".join(err) or None, "sort": (model or "", dataset or "", geo or "")}


def fam_objective(p: dict) -> dict:
    raw = str(p.get("objective") or "")
    base = OBJECTIVE_SET.get(raw)
    sp = _num(p.get("sparsity"))
    k = _num(p.get("dmd_steps"))
    err = None
    if raw and base is None:
        err = f"objective {raw!r} not in OBJECTIVE_SET"
    label = base
    if base == "DMD":
        label = f"DMD-K{k}" if k is not None else "DMD-K?"
    elif base == "Sparse QAT" and sp is not None and sp == 0:
        label = "Dense QAT"
    pct = f"{round(sp * 100)}%" if sp is not None else None
    return {"expanded": {"objective": label, "K": k, "sparsity": sp},
            "collapsed": " · ".join(x for x in (label, pct) if x) or None, "error": err,
            "sort": (label or "", sp if sp is not None else -1)}


def fam_cube(p: dict) -> dict:
    raw = str(p.get("cube_shape") or "none").strip().lower()
    key = raw.split(" ")[0]
    hit = CUBE_SET.get(key)
    if hit is None:
        return {"expanded": {"size": None, "T": None, "H": None, "W": None}, "collapsed": raw,
                "error": f"cube_shape {raw!r} not in CUBE_SET", "sort": (raw,)}
    name, size, t, h, w = hit
    return {"expanded": {"size": size, "T": t, "H": h, "W": w}, "collapsed": name, "error": None,
            "sort": (size or 0, t or 0, h or 0, w or 0)}


# The backend name only carries the kernel's *default* PV-scale behaviour; a row whose own precision
# field states the behaviour (e.g. the current dense QAT kernel: "… no PV scaling") is authoritative,
# so a kernel that stopped scaling PV is displayed from metadata instead of the historical default.
_NO_PV_SCALE = re.compile(r"\bno\s+PV[\s-]?scal", re.I)
_PV_SCALE = re.compile(r"\bPV[\s-]?scal(?:e|ing)\s+trick\b|\bwith\s+PV[\s-]?scal", re.I)


def fam_kernel(p: dict) -> dict:
    raw = str(p.get("backend") or "").strip()
    family = str(p.get("training_kernel_family") or "").strip() or None
    hit = KERNEL_SET.get(raw)
    if hit is None:
        return {"expanded": {"attn": None, "qk": None, "pv": None, "mm": None, "pv_scale": None, "family": family},
                "collapsed": raw or None,
                "error": f"backend {raw!r} not in KERNEL_SET" if raw else "backend missing", "sort": (raw,)}
    attn, qk, pv, mm, scale = hit
    quant = str(p.get("quantization") or "")
    scaling = str(p.get("pv_scaling") or "").strip().lower()
    if scaling:
        scale = scaling != "none"
    elif _NO_PV_SCALE.search(quant):
        scale = False
    elif _PV_SCALE.search(quant):
        scale = True
    # The PV precision only qualifies a quantized kernel; an unquantized BF16 kernel needs no suffix.
    label = f"{attn} ({pv} pv)" if pv not in (None, "bf16") else attn
    return {"expanded": {"attn": attn, "qk": qk, "pv": pv, "mm": mm, "pv_scale": scale, "family": family},
            "collapsed": label, "error": None, "sort": (attn, pv, raw)}


def fam_linear(p: dict) -> dict:
    """Linear-layer quantization: BF16 unless the Variant declares a linear quant config."""
    raw = str(p.get("linear_quantization") or "").strip()
    if not raw or raw.lower() in LINEAR_NONE:
        return {"expanded": {"linear": "BF16"}, "collapsed": "BF16", "error": None, "sort": ("",)}
    label = LINEAR_SET.get(raw)
    return {"expanded": {"linear": label or raw}, "collapsed": label or raw,
            "error": None if label else f"linear_quantization {raw!r} not in LINEAR_SET", "sort": (raw,)}


def fam_batch(p: dict) -> dict:
    bpr, acc = _num(p.get("batch_per_rank")), _num(p.get("grad_accum"))
    eb_raw = p.get("effective_batch")
    eb = _num(eb_raw)
    gi = _num(p.get("generator_update_interval"))
    ema, ema_start = _num(p.get("ema_decay")), _num(p.get("ema_start_iter"))
    is_dmd = str(p.get("objective") or "").startswith("dmd")
    warn = []
    if bpr is not None and acc is not None:
        calc = RANKS_PER_NODE * bpr * acc
        if eb is None:
            eb = calc
        elif eb != calc:
            warn.append(f"effective_batch {eb_raw!r} != {RANKS_PER_NODE} ranks x {bpr} x {acc} = {calc}")
    if isinstance(eb_raw, str):
        warn.append(f"effective_batch recorded as text: {eb_raw!r}")
    if not is_dmd:
        ema, ema_start, gi = None, None, None
    ema_label = None
    if is_dmd and ema not in (None, 0):
        ema_label = f"EMA{ema}" + (f"@{ema_start}" if ema_start else "")
    parts = [f"EB{eb}" if eb is not None else None, f"GI{gi}" if gi is not None else None, ema_label]
    return {"expanded": {"batch_per_rank": bpr, "grad_accum": acc, "effective_batch": eb, "GI": gi,
                         "ema_decay": ema if is_dmd else None, "ema_start": ema_start if is_dmd else None},
            "collapsed": " · ".join(x for x in parts if x) or None, "error": None,
            "warning": "; ".join(warn) or None, "sort": (eb or 0, gi or 0, ema or 0, ema_start or 0)}


def fam_lr(p: dict) -> dict:
    s, c = _lr(p.get("student_lr")), _lr(p.get("critic_lr"))
    return {"expanded": {"student_lr": _num(p.get("student_lr")), "critic_lr": _num(p.get("critic_lr"))},
            "collapsed": "/".join(x for x in (s, c) if x) or None, "error": None,
            "sort": (_num(p.get("student_lr")) or 0, _num(p.get("critic_lr")) or 0)}


def fam_schedule(p: dict) -> dict:
    steps, val, keep, seed = _num(p.get("steps")), _num(p.get("validation_every")), _num(p.get("checkpoint_keep")), _num(p.get("seed"))
    parts = [str(steps) if steps is not None else None, f"val{val}" if val is not None else None]
    return {"expanded": {"steps": steps, "val_every": val, "ckpt_keep": keep, "seed": seed},
            "collapsed": " · ".join(x for x in parts if x) or None, "error": None,
            "sort": (steps or 0, val or 0)}


FAMILIES = [
    {"key": "model", "label": "Model & Dataset", "fn": fam_model,
     "columns": [("model", "Model"), ("dataset", "Dataset"), ("data_role", "Data role"),
                 ("frames", "F"), ("height", "H"), ("width", "W")]},
    {"key": "objective", "label": "Objective", "fn": fam_objective,
     "columns": [("objective", "Objective"), ("K", "K"), ("sparsity", "Sparsity")]},
    {"key": "cube", "label": "Cube", "fn": fam_cube,
     "columns": [("size", "Cube"), ("T", "T"), ("H", "H"), ("W", "W")]},
    {"key": "kernel", "label": "Kernel", "fn": fam_kernel,
     "columns": [("attn", "Attn kernel"), ("qk", "QK"), ("pv", "PV"), ("mm", "MM"),
                 ("pv_scale", "PV scale trick"), ("family", "Kernel family")]},
    {"key": "linear", "label": "Linear", "fn": fam_linear, "columns": [("linear", "Linear quant")]},
    {"key": "batch", "label": "Batch", "fn": fam_batch,
     "columns": [("batch_per_rank", "B/rank"), ("grad_accum", "Accum"), ("effective_batch", "Eff. batch"),
                 ("GI", "GI"), ("ema_decay", "EMA decay"), ("ema_start", "EMA start")]},
    {"key": "lr", "label": "LR", "fn": fam_lr, "columns": [("student_lr", "Student LR"), ("critic_lr", "Critic LR")]},
    {"key": "schedule", "label": "Schedule", "fn": fam_schedule,
     "columns": [("steps", "Steps"), ("val_every", "Val every"), ("ckpt_keep", "Ckpt keep"), ("seed", "Seed")]},
]


def family_spec() -> list[dict]:
    return [{"key": f["key"], "label": f["label"], "columns": [{"key": k, "label": lbl} for k, lbl in f["columns"]]} for f in FAMILIES]


def compute_families(params: dict) -> dict:
    out = {}
    for f in FAMILIES:
        try:
            out[f["key"]] = f["fn"](params)
        except Exception as e:  # a family must never take the row down
            out[f["key"]] = {"expanded": {}, "collapsed": None, "error": f"{type(e).__name__}: {e}", "sort": ()}
    return out


# ------------------------------------------------------- row filter groups
# Each group selects at most one option; a selection keeps only the matching rows and
# combines with every other group and with the hide switches. Every Variant belongs to
# exactly one option of every group, so no row can be lost by the classification itself.
FILTER_GROUPS = [
    {"key": "stage", "label": "Stage",
     "options": [{"value": "dense ft", "label": "dense ft"}, {"value": "sparse ft", "label": "sparse ft"},
                 {"value": "dense dmd", "label": "dense dmd"}, {"value": "sparse dmd", "label": "sparse dmd"}]},
    {"key": "linear", "label": "Linear",
     "options": [{"value": "bf16", "label": "linear bf16"}, {"value": "nvfp4", "label": "linear nvfp4"}]},
    # Cube size of the sparse attention kernel. A dense row has no cube, so it belongs to no option
    # here and any cube selection excludes it - that is the intended reading of "show C128 only".
    {"key": "cube", "label": "Cube",
     "options": [{"value": "C64", "label": "C64"}, {"value": "C128", "label": "C128"},
                 {"value": "C256T8", "label": "C256T8"}, {"value": "C256", "label": "C256"}]},
]


def row_filters(p: dict) -> dict:
    """The filter-group membership of one Variant.

    `stage` is DMD when the objective is a DMD variant and fine-tuning otherwise (ODE-init and
    reconstruction objectives are fine-tuning), crossed with sparse when the declared sparsity is
    positive. `linear` follows the linear-quantization contract, not the attention kernel.
    """
    dmd = str(p.get("objective") or "").lower().startswith("dmd")
    sparsity = _num(p.get("sparsity")) or 0
    linear = str(p.get("linear_quantization") or "").strip().lower()
    quantized = bool(linear) and linear not in LINEAR_NONE
    cube = CUBE_SET.get(str(p.get("cube_shape") or "none").strip().split(" ")[0])
    return {"stage": f"{'sparse' if sparsity > 0 else 'dense'} {'dmd' if dmd else 'ft'}",
            "linear": "nvfp4" if quantized else "bf16",
            "cube": cube[0] if cube and cube[0] != "dense" else None}


# ------------------------------------------------------- datapoint kinds
# A datapoint is one evaluation of a checkpoint at a training step. VBench datapoints are
# grouped by the evaluation dataset alone: one kind per prompt-set subset, whatever attention
# forward produced the videos. The forward is never folded into the group identity — it stays
# per datapoint as provenance (`forward_tag`), so two evaluations of the same checkpoint on the
# same dataset under different kernels are two distinct datapoints inside one dataset lane.
# The locally rendered validation-caption lanes stay kernel-tagged: they carry no VBench scores
# and `inf_val` renders them under exactly one kernel.
KERNEL_KINDS = [  # (regex on the E0030 attention_backend string, kernel tag); val12 lanes only
    (re.compile(r"^VSQA", re.I), "vsqa"),
    (re.compile(r"TORCH_SDPA|dense", re.I), "dense-bf16"),
]
VERSIONED_PROMPT_SET = re.compile(r"\b([\w-]+)\.v(\d+)\.json\b", re.I)
PROMPT_KINDS = [  # (regex on prompt_set, prompt-set tag)
    (re.compile(r"validation\.json|validation captions", re.I), "val12"),
    (re.compile(r"VBench_full_info\.json", re.I), "vbench"),
    (VERSIONED_PROMPT_SET, "vbench"),
]
DIM_SHORT = {"imaging_quality": "imaging", "aesthetic_quality": "aesthetic", "subject_consistency": "subject",
             "background_consistency": "background", "temporal_flickering": "flicker", "motion_smoothness": "smooth",
             "dynamic_degree": "dynamic"}
# Static kinds. VBench kinds are `vbench[<subset>]` and are discovered from E0030 rows
# (one evaluation dataset = one kind); the marker/color is assigned per subset in `vbench_kind`.
DATAPOINT_KINDS = {
    "train-kernel×val12": {"label": "training kernel × 12 validation captions (W&B)", "marker": "circle", "color": "#7fc4ff",
                           "source": "wandb"},
    "inf-kernel×val12": {"label": "VSQA inference kernel × 12 validation captions (inf_val/)", "marker": "diamond",
                         "color": "#f0cf7a", "source": "inf_val"},
    "train-kernel×val12@4step": {"label": "training kernel × validation captions forced to 4 sampling steps "
                                          "(val_forced4/; not part of the Run's own validation set)",
                                 "marker": "triangle", "color": "#3fbf7f", "source": "forced_val"},
}
_SUBSET_PALETTE = [("square", "#3fbf7f"), ("triangle", "#e5646a"), ("diamond", "#b48cff"), ("circle", "#ff9f43"),
                   ("square", "#4dd0e1"), ("triangle", "#f06292")]
_subset_slots: dict[str, int] = {}


def subset_label(params: dict) -> str:
    """Short label of the VBench prompt subset an E0030 row used: 'q7', 'q7-lim4', 'imaging+aesthetic'."""
    dims = str(params.get("dimensions") or "")
    m = re.search(r"quality subset:\s*(.+)", dims)
    if m:
        label = "+".join(DIM_SHORT.get(d.strip(), d.strip()) for d in m.group(1).split(","))
    else:
        label = "q7"
    ps = str(params.get("prompt_set") or "")
    manifest = VERSIONED_PROMPT_SET.search(ps)
    if manifest:
        label = f"{manifest.group(1)}-v{manifest.group(2)}"
    lim = re.search(r"first (\d+)", ps)
    if lim:
        label += f"-lim{lim.group(1)}"
    mode = str(params.get("sample_mode") or "")
    if mode.startswith("official"):
        label += "-official"
    return label


def vbench_kind(subset: str) -> str:
    """Display group of one VBench evaluation dataset; every attention forward shares the group."""
    kind = f"vbench[{subset}]"
    if kind not in DATAPOINT_KINDS:
        slot = _subset_slots.setdefault(subset, len(_subset_slots))
        marker, color = _SUBSET_PALETTE[slot % len(_SUBSET_PALETTE)]
        DATAPOINT_KINDS[kind] = {"label": f"VBench {subset} — any attention forward (the forward stays on each datapoint)",
                                 "marker": marker, "color": color, "source": "e0030", "subset": subset}
    return kind


def forward_tag(params: dict) -> str:
    """Short provenance tag of the forward an E0030 row ran, for display beside a score: 'dense BF16',
    'VSQA·vsa', 'NVFP4/FP8·dense'. The stored attention_backend string stays the authority and is shown
    verbatim in the datapoint detail; this tag never decides which display group a row belongs to."""
    backend = str(params.get("attention_backend") or "").strip()
    kind = str(params.get("attention_kind") or "").strip().lower()
    if not backend or re.match(r"TORCH_SDPA", backend, re.I):
        return "dense BF16"
    head = re.split(r"[\s(]", backend, maxsplit=1)[0]
    quant = KERNEL_SET.get(head.upper())
    tag = f"{quant[0].upper()}/{quant[1].upper()}" if quant else head[:20]
    return f"{tag}·{kind}" if kind else tag


def classify_eval(params: dict) -> tuple[str | None, str | None]:
    """(kind, error) for an E0030 row. VBench rows are classified by their evaluation dataset
    alone, so a new (but semantically suitable) attention forward needs no registration; an
    unregistered prompt set is still surfaced as an explicit error datapoint."""
    prompt_set = str(params.get("prompt_set") or "")
    prompts = next((tag for rx, tag in PROMPT_KINDS if rx.search(prompt_set)), None)
    if prompts is None:
        return None, f"unregistered evaluation dataset: prompt_set={prompt_set[:60]!r}"
    if prompts == "val12":
        backend = str(params.get("attention_backend") or "")
        kernel = next((tag for rx, tag in KERNEL_KINDS if rx.search(backend)), None)
        if kernel != "vsqa":
            return None, f"validation captions are registered for the VSQA inference kernel only, not backend={backend[:40]!r}"
        return "inf-kernel×val12", None
    return vbench_kind(subset_label(params)), None
