#!/usr/bin/env bash
# Render one checkpoint's validation captions under the VSQA inference kernel inside a
# Slurm allocation (NUM_GPUS shards, one caption process per GPU). Output goes to
# OUT_DIR (= <run>/inf_val/<step>); exit_code and state files are written there.
set -euo pipefail
: "${ALLOCATION_ID:?Slurm job id of the evaluation allocation}"
: "${CHECKPOINT:?E0029/V<NNNN>@<step>}"
: "${OUT_DIR:?<run>/inf_val/<step>}"
GPU_IDS="${GPU_IDS:-0}"                       # comma-separated physical GPU indices on the node; one shard each
IFS=, read -r -a GPU_ARR <<< "$GPU_IDS"; NUM_GPUS=${#GPU_ARR[@]}
OVERWRITE="${OVERWRITE:-0}"
SRUN_TIME="${SRUN_TIME:-06:00:00}"
DASH_DIR="${DASH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"   # the scheduler runs a per-job copy of this script
PROJECT_ROOT="$(cd "$DASH_DIR/.." && pwd)"
E0029="$PROJECT_ROOT/docs/experiments/E0029-fastvideo-mixkit-training-migration"
E0030="$PROJECT_ROOT/vbench_eval"

: "${CONDA_SH:?set CONDA_SH to the conda.sh of this cluster}" "${CONDA_ENV:?set CONDA_ENV to the conda environment name}"
# shellcheck disable=SC1090
source "$CONDA_SH"; conda activate "$CONDA_ENV"
CODE_ROOT="${VSQA_CODE_ROOT:-$PROJECT_ROOT/fastvideo}"
export PYTHONPATH="$E0030/pyshims:$PROJECT_ROOT/outputs/e0029-python-overlay:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUTE_DSL_ENABLE_TVM_FFI=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$OUT_DIR"
echo RUNNING > "$OUT_DIR/state"
rm -f "$OUT_DIR/exit_code"
cd "$PROJECT_ROOT"
# Any early failure (resolve/export) still leaves exit_code + state for the scheduler.
trap 'ec=$?; if [ ! -f "$OUT_DIR/exit_code" ]; then echo "$ec" > "$OUT_DIR/exit_code"; echo FAILED > "$OUT_DIR/state"; fi' EXIT
# Concurrent exports on one node must not share the torch.distributed rendezvous port.
export MASTER_PORT="${MASTER_PORT:-$((29900 + ${GPU_ARR[0]} * 7 + RANDOM % 7))}"

# Export the DCP checkpoint first when only checkpoints/checkpoint-<step> exists (resolver exit 3).
set +e
python "$E0030/resolve_checkpoint.py" "$CHECKPOINT" > "$OUT_DIR/checkpoint.json"; rc=$?
set -e
if [ "$rc" -eq 3 ]; then
  CKPT_DIR="$(jq -r .checkpoint_dir "$OUT_DIR/checkpoint.json")"; EXPORT_DIR="$(jq -r .export_dir "$OUT_DIR/checkpoint.json")"
  echo "[inf_val] exporting $CKPT_DIR -> $EXPORT_DIR on gpu ${GPU_ARR[0]}"
  # Same export as E0029/export_checkpoint.sh, but pinned to this job's GPU (that script lets Slurm pick one).
  git -C "$CODE_ROOT" diff --quiet HEAD || { echo "[inf_val] training code is dirty; refusing to export"; exit 2; }
  [ -f "$CKPT_DIR/metadata.json" ] || { echo "[inf_val] not a complete checkpoint: $CKPT_DIR"; exit 2; }
  rm -rf "$EXPORT_DIR"
  srun --jobid="$ALLOCATION_ID" --overlap -N1 -n1 -c8 --gres=gpu:4 --time=00:40:00 \
    env CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" WANDB_MODE=offline TOKENIZERS_PARALLELISM=false NUM_GPUS=1 NNODES=1 \
        MASTER_ADDR=127.0.0.1 MASTER_PORT="$MASTER_PORT" PYTHONPATH="$PYTHONPATH" \
    bash -c "cd '$CODE_ROOT' && python -m fastvideo.train.entrypoint.dcp_to_diffusers \
        --checkpoint '$CKPT_DIR' --output-dir '$EXPORT_DIR' --weights student --verify --overwrite" 2>&1 | tee "$EXPORT_DIR.export.log"
  for _ in $(seq 12); do [ -f "$EXPORT_DIR/metadata.json" ] && [ -f "$EXPORT_DIR/artifact_manifest.json" ] && break; sleep 5; done
  [ -f "$EXPORT_DIR/metadata.json" ] && [ -f "$EXPORT_DIR/artifact_manifest.json" ] || { echo "[inf_val] export incomplete"; exit 3; }
  python "$E0030/resolve_checkpoint.py" "$CHECKPOINT" > "$OUT_DIR/checkpoint.json"
elif [ "$rc" -ne 0 ]; then
  echo "[inf_val] checkpoint resolution failed for $CHECKPOINT"; echo 2 > "$OUT_DIR/exit_code"; echo FAILED > "$OUT_DIR/state"; exit 2
fi

OW=(); [ "$OVERWRITE" = "1" ] && OW=(--overwrite)
cat > "$OUT_DIR/inner.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"
pids=(); i=0
for g in ${GPU_IDS//,/ }; do
  CUDA_VISIBLE_DEVICES=\$g python "$DASH_DIR/inf_val.py" --checkpoint "$CHECKPOINT" --out-dir "$OUT_DIR" --shard "\$i/$NUM_GPUS" ${OW[*]:-} \\
    > "$OUT_DIR/shard\$i.log" 2>&1 &
  pids+=(\$!); i=\$((i+1))
done
rc=0
for p in "\${pids[@]}"; do wait "\$p" || rc=1; done
exit \$rc
EOF
chmod +x "$OUT_DIR/inner.sh"

set +e
srun --jobid="$ALLOCATION_ID" --overlap -N1 -n1 -c16 --gres=gpu:4 --time="$SRUN_TIME" \
  bash "$OUT_DIR/inner.sh" 2>&1 | tee -a "$OUT_DIR/run.log"
ec=${PIPESTATUS[0]}
set -e
echo "$ec" > "$OUT_DIR/exit_code"
if [ "$ec" -eq 0 ]; then echo FINISHED > "$OUT_DIR/state"; else echo FAILED > "$OUT_DIR/state"; fi
echo "[inf_val] exit_code=$ec OUT_DIR=$OUT_DIR" | tee -a "$OUT_DIR/run.log"
exit "$ec"
