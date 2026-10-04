"""Paths and constants for the VSQA E0029/E0030 monitoring dashboard."""
from __future__ import annotations

import os
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_DIR = Path(os.environ.get("VSQA_DASHBOARD_STATE", DASHBOARD_DIR / "state"))
STATE_DB = STATE_DIR / "state.sqlite"
STATIC_DIR = DASHBOARD_DIR / "static"

TRAIN_EXP = "E0029-fastvideo-mixkit-training-migration"
EVAL_EXP = "E0030-vbench-quality-evaluation"
EXP_DIR = {TRAIN_EXP: PROJECT_ROOT / "docs" / "experiments" / TRAIN_EXP,
           EVAL_EXP: PROJECT_ROOT / "docs" / "experiments" / EVAL_EXP}
EVAL_LAUNCHER = "vbench_eval/run_vbench.sh"
CODE_ROOT = Path(os.environ.get("VSQA_CODE_ROOT", PROJECT_ROOT / "fastvideo"))  # FastVideo checkout used for runs
CODE_REPO = os.environ.get("VSQA_CODE_REPO", CODE_ROOT.name)  # provenance label of that checkout
# VBench evaluations are no longer submitted from the dashboard (owner, 2026-09-26): the training
# session runs post-training evaluation. Scores are still read and displayed.
VBENCH_SUBMISSION = False

# Deployment-specific settings are read from the environment.
WANDB_ENTITY_PROJECT = os.environ.get("VSQA_WANDB_PROJECT", "")  # "<entity>/<project>"; empty disables W&B refresh
WANDB_SCALAR_KEYS = ["generator_loss", "fake_score_loss", "total_loss", "grad_norm/student",
                     "grad_norm/critic", "step_time_sec", "loss", "grad_norm", "lr"]
WANDB_REFRESH_SEC = 120

HOST = os.environ.get("VSQA_DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("VSQA_DASHBOARD_PORT", "8029"))
SCHEDULER_TICK_SEC = 60
IDLE_GPU_MIB = 1024

MEMON = os.environ.get("MEMON_BIN", str(Path.home() / ".local/bin/memon"))

VBENCH_QUALITY_DIMS = ["imaging_quality", "aesthetic_quality", "subject_consistency",
                       "background_consistency", "temporal_flickering", "motion_smoothness",
                       "dynamic_degree"]
