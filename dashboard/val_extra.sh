#!/usr/bin/env bash
# Render the validation captions added after a Run finished, for every checkpoint of one
# Variant, inside an existing Slurm allocation. One worker per GPU; each worker exports the
# DCP checkpoint it owns, renders its captions with the Run's own frozen training source and
# student attention backend. The export is a durable, reusable artifact: it is kept so that no
# checkpoint is ever exported twice (DROP_EXPORT=1 restores the old delete-after-render mode).
#
# Usage: ALLOCATION_ID=<job> VARIANT=V0249 CAPTIONS_FILE=<validation.json with the new captions> \
#        bash dashboard/val_extra.sh
#
# Writes <run>/val_extra/<step>/{manifest.json,<idx>.mp4,worker*.log} and a sweep log under
# <run>/val_extra/. Never touches the Run's own validation videos, recipe or validation.json.
set -euo pipefail
: "${ALLOCATION_ID:?Slurm job id of an allocation that already holds the node}"
# Bash reads a script incrementally, so editing this file while a sweep runs corrupts that
# sweep mid-flight. Execute an immutable snapshot instead, exactly like vbench_eval/run_vbench.sh.
if [ -z "${VAL_EXTRA_SNAPSHOT:-}" ]; then
  _self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  _snapdir="$(cd "$(dirname "$_self")/.." && pwd)/outputs/val-extra-snapshots"
  mkdir -p "$_snapdir"
  _snap="$(mktemp "$_snapdir/val_extra-XXXXXX.sh")"
  cp -- "$_self" "$_snap"
  chmod 400 "$_snap"
  echo "[val_extra] snapshot=$_snap"
  VAL_EXTRA_SNAPSHOT="$_self" exec bash "$_snap" "$@"
fi

: "${VARIANT:?E0029 Variant id, e.g. V0249}"
: "${CAPTIONS_FILE:?validation.json holding the extended caption list}"
# Resolve before anything else: a relative path breaks once a worker changes directory, and a
# path inside somebody else's Run directory can disappear mid-sweep.
CAPTIONS_FILE="$(realpath "$CAPTIONS_FILE")"
[ -f "$CAPTIONS_FILE" ] || { echo "CAPTIONS_FILE does not exist: $CAPTIONS_FILE" >&2; exit 2; }
INDICES="${INDICES:-12-15}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
STEPS="${STEPS:-}"                 # optional explicit comma-separated subset
# Exports are kept by default. `resolve_checkpoint` finds `<run>/export-step<N>` first, so every
# later render, VBench eval or manual inspection reuses this one instead of re-exporting.
DROP_EXPORT="${DROP_EXPORT:-0}"
# Strict reload verification rebuilds the 1.49B model and reads the export back (~16-20 s per
# checkpoint). A kept export is a durable artifact, so it stays on by default.
EXPORT_VERIFY="${EXPORT_VERIFY:-1}"
VERIFY_ARG=""
if [ "$EXPORT_VERIFY" = 1 ]; then VERIFY_ARG="--verify"; fi
# Two sweeps may share one node with disjoint GPU_IDS; each srun step then has to see every GPU
# so a worker can pin its absolute index. Default: exactly the GPUs this sweep uses.
SRUN_GRES="${SRUN_GRES:-gpu:$(echo "$GPU_IDS" | tr ',' '\n' | grep -c .)}"
SRUN_TIME="${SRUN_TIME:-12:00:00}"
# Checkpoints per render process. One process amortizes the torch import and CUDA context over
# its whole chunk; a small chunk still delivers the first videos early.
CHUNK="${CHUNK:-4}"
# FORCE_STEPS renders the captions at a different sampling-step count; the result is a separate
# comparison in `val_forced<N>/` with its own dashboard lane, never part of the Run's own set.
FORCE_STEPS="${FORCE_STEPS:-}"
OUT_ROOT="val_extra"
FORCE_ARGS=()
if [ -n "$FORCE_STEPS" ]; then
  OUT_ROOT="val_forced$FORCE_STEPS"
  FORCE_ARGS=(--force-sampling-steps "$FORCE_STEPS")
fi
# A Run launched from a batch directory records no code.frozen.json; name its code root and
# manifest digest explicitly rather than rendering an old checkpoint with newer code.
FROZEN_ROOT_OVERRIDE="${FROZEN_ROOT_OVERRIDE:-}"
FROZEN_SHA_OVERRIDE="${FROZEN_SHA_OVERRIDE:-}"
# A snapshot lives outside dashboard/, so paths come from the original script location.
DASH_DIR="$(cd "$(dirname "${VAL_EXTRA_SNAPSHOT:-${BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(cd "$DASH_DIR/.." && pwd)"
E0029="$PROJECT_ROOT/docs/experiments/E0029-fastvideo-mixkit-training-migration"
E0030="$PROJECT_ROOT/vbench_eval"

: "${CONDA_SH:?set CONDA_SH to the conda.sh of this cluster}" "${CONDA_ENV:?set CONDA_ENV to the conda environment name}"
# shellcheck disable=SC1090
source "$CONDA_SH"; conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

# ---- Run identity, frozen source and checkpoint list from the Variant declaration --------
read -r RUN_REL FROZEN_ROOT FROZEN_SHA FROZEN_COMMIT STEP_LIST < <(
  PYTHONPATH="$PROJECT_ROOT" python - "$VARIANT" "$STEPS" <<'PY'
import json, sys
from pathlib import Path
from dashboard.sources import results, runs
from dashboard.config import TRAIN_EXP
variant, wanted = sys.argv[1], [int(s) for s in sys.argv[2].split(',') if s.strip()]
v = results.get(TRAIN_EXP, variant) or sys.exit(f'{variant} not in E0029 results.yaml')
run = next((r for r in reversed(v.get('runs') or []) if runs.scan(r)), None) or sys.exit(f'{variant} has no scannable run')
info = runs.scan(run)
frozen_path = Path(run) / 'code.frozen.json'
frozen = json.loads(frozen_path.read_text()) if frozen_path.is_file() else {}
archive = frozen.get('source_archive') or {}
steps = [s for s in info['checkpoint_steps'] if not wanted or s in wanted]
if not steps:
    sys.exit(f'{variant} has no matching checkpoints')
print(run, archive.get('root') or frozen.get('code_root') or '-', archive.get('manifest_sha256') or '-',
      frozen.get('commit') or (Path(run) / 'code.head').read_text().strip(), ','.join(str(s) for s in steps))
PY
)
RUN_DIR="$PROJECT_ROOT/$RUN_REL"
SWEEP_LOG="$RUN_DIR/$OUT_ROOT/sweep.log"
mkdir -p "$RUN_DIR/$OUT_ROOT"
# One sweep per Run directory: two concurrent sweeps would export and render the same
# checkpoint at once and interleave their worker logs.
exec 9> "$RUN_DIR/$OUT_ROOT/.sweep.lock"
if ! flock -n 9; then
  echo "[val_extra] another sweep already owns $RUN_REL; refusing to run a second one" >&2
  exit 4
fi
# A previous sweep may have used more workers than this one; its leftover worker logs would
# otherwise read as live progress. Only logs are removed here, never a rendered video.
rm -f "$RUN_DIR/$OUT_ROOT"/worker[0-9].log "$RUN_DIR/$OUT_ROOT"/render-slot[0-9].log
exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "[val_extra] $(date -Iseconds) variant=$VARIANT allocation=$ALLOCATION_ID run=$RUN_REL"
echo "[val_extra] indices=$INDICES captions=$CAPTIONS_FILE gpus=$GPU_IDS" \
     "drop_export=$DROP_EXPORT export_verify=$EXPORT_VERIFY"
echo "[val_extra] out_root=$OUT_ROOT force_steps=${FORCE_STEPS:-none}"
echo "[val_extra] steps=$STEP_LIST"

# The checkpoints were written by a frozen source, so they are exported and rendered by that
# same source: it is the code that defines this Run's kernel and metadata contract.
if [ -n "$FROZEN_ROOT_OVERRIDE" ]; then
  FROZEN_ROOT="$(realpath "$FROZEN_ROOT_OVERRIDE")"
  FROZEN_SHA="${FROZEN_SHA_OVERRIDE:-$FROZEN_SHA}"
  echo "[val_extra] code root override: $FROZEN_ROOT (manifest ${FROZEN_SHA})"
fi
if [ "$FROZEN_SHA" = "-" ]; then
  echo "[val_extra] this Run records no frozen-source manifest; pass FROZEN_ROOT_OVERRIDE and" >&2
  echo "            FROZEN_SHA_OVERRIDE so the checkpoint is rendered by the code that wrote it" >&2
  exit 5
fi
if [ "$FROZEN_SHA" != "-" ]; then
  python "$E0029/frozen_source.py" verify --code-root "$FROZEN_ROOT" \
    --manifest-sha256 "$FROZEN_SHA" --commit "$FROZEN_COMMIT"
fi
export PYTHONPATH="$E0030/pyshims:$PROJECT_ROOT/outputs/e0029-python-overlay:$FROZEN_ROOT"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$PROJECT_ROOT/outputs/e0029-pycache}"
export GIT_CEILING_DIRECTORIES="$(dirname "$FROZEN_ROOT")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false

IFS=, read -r -a GPU_ARR <<< "$GPU_IDS"
IFS=, read -r -a STEP_ARR <<< "$STEP_LIST"
WORKERS=${#GPU_ARR[@]}

# Expand the requested caption indices once, so a worker can decide "already done" without
# starting python, an export or a model load.
IDX_LIST="$(PYTHONPATH="$PROJECT_ROOT" python -c "
from dashboard.val_extra import parse_indices
print(' '.join(str(i) for i in parse_indices('$INDICES')))")"
SWEEP_PID=$$

echo "[val_extra] caption indices: $IDX_LIST"

cat > "$RUN_DIR/$OUT_ROOT/worker.sh" <<EOF
#!/usr/bin/env bash
# One worker: own GPU, own subset of checkpoint steps, processed in chunks of $CHUNK. Each chunk
# is exported and then rendered by ONE python process, so the torch import and CUDA context are
# paid once per chunk instead of once per checkpoint. Chunking (rather than exporting the whole
# subset first) keeps the first videos early and bounds how many exports are in flight.
set -uo pipefail
gpu="\$1"; slot="\$2"
# The port must not collide with another export on this node: $SWEEP_PID separates sweeps,
# the slot separates this sweep's own workers.
export MASTER_ADDR=127.0.0.1 NUM_GPUS=1 NNODES=1
export MASTER_PORT=\$(( 20000 + ($SWEEP_PID % 3000) * 8 + slot ))
rc=0
chunk_n=0
render_chunk() {
  [ -n "\$refs" ] || return 0
  echo "[worker \$slot] gpu \$gpu render \$refs"
  chunk_n=\$(( chunk_n + 1 ))
  python -m dashboard.val_extra --checkpoints "\$refs" ${FORCE_ARGS[*]:-} \\
    --captions-file "$CAPTIONS_FILE" --indices "$INDICES" --out-root "$OUT_ROOT" \\
    >> "$RUN_DIR/$OUT_ROOT/render-slot\$slot.log" 2>&1 \\
    || { echo "[worker \$slot] render failed for \$refs"; rc=1; }
  if [ "$DROP_EXPORT" = 1 ]; then
    for s in \$(echo "\$refs" | tr ',' ' ' | sed 's/[^ ]*@//g'); do
      # resolve_checkpoint builds an "<export>.dmdview" symlink view for a DMD checkpoint;
      # it dangles once the export is gone, so both go together.
      rm -rf "$RUN_DIR/export-step\$s" "$RUN_DIR/export-step\$s.dmdview"
    done
  fi
  refs=""; n=0
}
refs=""; n=0
# Round-robin by position in the step list. Distributing by "step modulo workers" would strand
# workers whenever every checkpoint step shares a factor with the worker count: with steps
# every 50 and 4 workers, only slots 0 and 2 ever matched and half the GPUs idled.
position=0
for step in ${STEP_ARR[*]}; do
  slot_for_step=\$(( position % $WORKERS ))
  position=\$(( position + 1 ))
  [ "\$slot_for_step" -eq "\$slot" ] || continue
  out="$RUN_DIR/$OUT_ROOT/\$step"
  mkdir -p "\$out"
  missing=0
  for i in $IDX_LIST; do
    [ -f "\$out/\$(printf '%02d' "\$i").mp4" ] || missing=1
  done
  if [ "\$missing" = 0 ]; then
    echo "[worker \$slot] step \$step already complete; skipping"
    continue
  fi
  export_dir="$RUN_DIR/export-step\$step"
  # A finished export is the reusable artifact: this checkpoint is never exported twice, by this
  # sweep or by any later one.
  if [ -f "\$export_dir/metadata.json" ]; then
    echo "[worker \$slot] step \$step reusing existing export"
  else
    echo "[worker \$slot] gpu \$gpu export step \$step"
    rm -rf "\$export_dir"
    ( cd "$FROZEN_ROOT" && python -m fastvideo.train.entrypoint.dcp_to_diffusers \\
        --checkpoint "$RUN_DIR/checkpoints/checkpoint-\$step" --output-dir "\$export_dir" \\
        --weights student $VERIFY_ARG --overwrite ) > "\$out/export.log" 2>&1 \\
      || { echo "[worker \$slot] export failed at \$step"; rc=1; continue; }
  fi
  refs="\${refs:+\$refs,}E0029/$VARIANT@\$step"
  n=\$(( n + 1 ))
  [ "\$n" -ge $CHUNK ] && render_chunk
done
render_chunk
exit \$rc
EOF
chmod +x "$RUN_DIR/$OUT_ROOT/worker.sh"

cat > "$RUN_DIR/$OUT_ROOT/inner.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd "$PROJECT_ROOT"
pids=(); slot=0
for g in ${GPU_IDS//,/ }; do
  bash "$RUN_DIR/$OUT_ROOT/worker.sh" "\$g" "\$slot" > "$RUN_DIR/$OUT_ROOT/worker\$slot.log" 2>&1 &
  pids+=(\$!); slot=\$((slot+1))
done
rc=0
for p in "\${pids[@]}"; do wait "\$p" || rc=1; done
exit \$rc
EOF
set +e
srun --jobid="$ALLOCATION_ID" --overlap -N1 -n1 -c32 --gres="$SRUN_GRES" --time="$SRUN_TIME" \
  env PYTHONPATH="$PYTHONPATH" PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
      GIT_CEILING_DIRECTORIES="$GIT_CEILING_DIRECTORIES" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      WANDB_MODE=offline TOKENIZERS_PARALLELISM=false \
  bash "$RUN_DIR/$OUT_ROOT/inner.sh"
ec=$?
set -e
rendered=$(find "$RUN_DIR/$OUT_ROOT" -maxdepth 2 -name '[0-9][0-9].mp4' | wc -l)
echo "[val_extra] exit_code=$ec rendered_videos=$rendered run=$RUN_REL"
exit "$ec"
