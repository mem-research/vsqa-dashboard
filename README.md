# VSQA dashboard

FastAPI dashboard for monitoring VSQA training (E0029) and VBench evaluation (E0030) runs.
The `dashboard/` package expects to live inside the VSQA project root (it reads `docs/experiments/`,
`logs/`, `vbench_eval/` and `outputs/` relative to its parent directory) and is started from that
root with `dashboard/serve.sh` (`python -m dashboard.app`).

## Environment

Deployment-specific values are read from the environment:

| Variable | Required | Meaning |
| --- | --- | --- |
| `CONDA_SH` | yes (shell launchers) | Path to the cluster's `conda.sh` |
| `CONDA_ENV` | yes (shell launchers) | Conda environment to activate |
| `VSQA_CODE_ROOT` | no | FastVideo checkout used for runs (default `<project>/fastvideo`) |
| `VSQA_CODE_REPO` | no | Provenance label for that checkout (default: its directory name) |
| `VSQA_WANDB_PROJECT` | no | W&B `<entity>/<project>`; unset disables W&B refresh |
| `VSQA_TUNNEL` | no | `<ssh-host>:<remote-port>` reverse tunnel; unset disables it |
| `VSQA_DASHBOARD_HOST` / `VSQA_DASHBOARD_PORT` | no | Bind address (default `127.0.0.1:8029`) |
| `VSQA_DASHBOARD_STATE` | no | State directory (default `dashboard/state`) |
| `MEMON_BIN` | no | Path to the `memon` CLI (default `~/.local/bin/memon`) |

Export `CONDA_SH`/`CONDA_ENV` before `serve.sh` so the scheduler's per-job launchers inherit them.
