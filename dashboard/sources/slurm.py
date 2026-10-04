"""Slurm allocation listing and an on-node GPU probe via `srun --overlap`."""
from __future__ import annotations

import subprocess
import threading
import time

_lock = threading.Lock()
_cache: dict[str, tuple[float, object]] = {}


def _run(argv: list[str], timeout: int) -> tuple[int, str]:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout + done.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def allocations(ttl: float = 20.0) -> list[dict]:
    now = time.time()
    with _lock:
        hit = _cache.get("squeue")
        if hit and now - hit[0] < ttl:
            return hit[1]  # type: ignore[return-value]
    rc, out = _run(["squeue", "--me", "-h", "-o", "%i|%j|%T|%N|%b|%L|%M|%C"], 20)
    rows = []
    if rc == 0:
        for line in out.splitlines():
            p = line.split("|")
            if len(p) < 8:
                continue
            rows.append({"job": int(p[0]) if p[0].isdigit() else p[0], "name": p[1], "state": p[2],
                         "node": p[3], "gres": p[4], "time_left": p[5], "elapsed": p[6], "cpus": p[7]})
    with _lock:
        _cache["squeue"] = (now, rows)
    return rows


def allocation(job: int) -> dict | None:
    return next((a for a in allocations() if a["job"] == job), None)


def probe_gpus(job: int, num_gpus: int = 4, timeout: int = 90) -> dict:
    """Run nvidia-smi inside allocation `job`. Returns {'ok', 'gpus': [...], 'error'}."""
    rc, out = _run(["srun", f"--jobid={job}", "--overlap", "-N1", "-n1", f"--gres=gpu:{num_gpus}",
                    "--time=00:05:00", "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"], timeout)
    if rc != 0:
        return {"ok": False, "gpus": [], "error": out.strip()[-500:]}
    gpus = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4 and p[0].isdigit():
            gpus.append({"gpu": int(p[0]), "used_mib": int(p[1]), "total_mib": int(p[2]), "util": int(p[3])})
    return {"ok": bool(gpus), "gpus": gpus, "error": None if gpus else out.strip()[-500:]}
