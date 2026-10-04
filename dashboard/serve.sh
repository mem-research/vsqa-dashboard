#!/usr/bin/env bash
# Start (or restart) the VSQA dashboard on the login node, detached from the shell.
# Access from a workstation: ssh -L 8029:127.0.0.1:8029 <login-host>  ->  http://127.0.0.1:8029
set -euo pipefail
DASH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DASH_DIR/.." && pwd)"
PORT="${VSQA_DASHBOARD_PORT:-8029}"
STATE_DIR="${VSQA_DASHBOARD_STATE:-$DASH_DIR/state}"
mkdir -p "$STATE_DIR"

: "${CONDA_SH:?set CONDA_SH to the conda.sh of this cluster}" "${CONDA_ENV:?set CONDA_ENV to the conda environment name}"
# shellcheck disable=SC1090
source "$CONDA_SH"; conda activate "$CONDA_ENV"

PIDFILE="$STATE_DIR/server.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  case "${1:-}" in
    restart) echo "stopping $(cat "$PIDFILE")"; kill "$(cat "$PIDFILE")"; sleep 2 ;;
    stop) echo "stopping $(cat "$PIDFILE")"; kill "$(cat "$PIDFILE")"; exit 0 ;;
    *) echo "already running (pid $(cat "$PIDFILE")) on port $PORT; use '$0 restart'"; exit 0 ;;
  esac
elif [ "${1:-}" = stop ]; then
  echo "not running"; exit 0
fi

cd "$PROJECT_ROOT"
setsid nohup env VSQA_DASHBOARD_PORT="$PORT" VSQA_DASHBOARD_STATE="$STATE_DIR" \
  python -m dashboard.app >> "$STATE_DIR/server.log" 2>&1 &
echo $! > "$PIDFILE"
sleep 3
if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "dashboard pid $(cat "$PIDFILE") on http://127.0.0.1:$PORT (log: $STATE_DIR/server.log)"
else
  echo "failed to start; see $STATE_DIR/server.log"; tail -n 30 "$STATE_DIR/server.log"; exit 1
fi
