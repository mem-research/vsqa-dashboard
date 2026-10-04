"""Persistent evaluation queue driving `run_vbench.sh` on one dedicated allocation.

Queue rows live in sqlite (survive restarts). Every tick: reconcile running
rows against their run directories, then, if the queue is not paused, the
allocation exists, no dashboard job is running, no training run reports that
allocation as its job, and an on-node nvidia-smi probe shows every GPU idle,
launch the next queued row and record it with memon.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from . import db
from .config import EVAL_EXP, EVAL_LAUNCHER, IDLE_GPU_MIB, LOGS_DIR, PROJECT_ROOT, SCHEDULER_TICK_SEC, VBENCH_SUBMISSION
from .sources import results, runs, slurm
from .variants import lint_and_submit, memon
from vbench_eval.sync_results import Store

log = logging.getLogger("dashboard.scheduler")
VBENCH_DISABLED = "VBench submission from the dashboard is disabled; the training session runs evaluations"
_lock = threading.RLock()


def _db():
    return db.connect()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def event(level: str, message: str) -> None:
    log.log(logging.WARNING if level in ("warn", "error") else logging.INFO, message)
    con = _db()
    with con:
        con.execute("INSERT INTO events(ts,level,message) VALUES(?,?,?)", (_now(), level, message))
        con.execute("DELETE FROM events WHERE id < (SELECT MAX(id) FROM events) - 500")


# --------------------------------------------------------------------- settings
def get_setting(key: str, default=None):
    row = _db().execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_setting(key: str, value) -> None:
    con = _db()
    with con:
        con.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (key, None if value is None else str(value)))


def settings() -> dict:
    alloc = get_setting("eval_allocation_id")
    return {"eval_allocation_id": int(alloc) if alloc and str(alloc).isdigit() else None,
            "paused": get_setting("paused", "1") == "1",
            "num_gpus": int(get_setting("num_gpus", "4")),
            "last_tick": get_setting("last_tick"), "last_probe": json.loads(get_setting("last_probe") or "null"),
            "last_decision": get_setting("last_decision")}


# ------------------------------------------------------------------------ queue
def _rows(con, where: str = "", args: tuple = ()) -> list[dict]:
    out = []
    for r in con.execute(f"SELECT * FROM eval_queue {where} ORDER BY position, id", args):
        d = dict(r)
        d["env"] = json.loads(d["env"])
        out.append(d)
    return out


def queue() -> list[dict]:
    return _rows(_db())


def enqueue(variant_id: str, env: dict, note: str | None = None, kind: str = "vbench") -> int:
    """Queue a job. kind 'vbench' runs run_vbench.sh for an E0030 Variant; kind 'inf_val' renders an
    E0029 checkpoint's validation captions under the VSQA inference kernel into its run directory."""
    if kind not in ("vbench", "inf_val"):
        raise ValueError(f"unknown job kind {kind!r}")
    if kind == "vbench" and not VBENCH_SUBMISSION:
        raise PermissionError(VBENCH_DISABLED)
    con = _db()
    with con:
        dup = con.execute("SELECT id FROM eval_queue WHERE kind=? AND variant_id=? AND env=? AND status IN ('queued','running')",
                          (kind, variant_id, json.dumps(env))).fetchone()
        if dup:
            return int(dup[0])
        pos = con.execute("SELECT COALESCE(MAX(position),0)+1 FROM eval_queue").fetchone()[0]
        cur = con.execute("INSERT INTO eval_queue(kind,variant_id,env,status,position,created_at,note) VALUES(?,?,?,?,?,?,?)",
                          (kind, variant_id, json.dumps(env), "queued", pos, _now(), note))
        qid = cur.lastrowid
    event("info", f"queued #{qid} {kind} {variant_id} {env.get('CHECKPOINT', '')}")
    return qid


def pending_for(kind: str) -> dict[str, dict]:
    """Queued/running jobs by CHECKPOINT ref, for the timeline placeholders."""
    out = {}
    for r in _rows(_db(), "WHERE kind=? AND status IN ('queued','running')", (kind,)):
        out[r["env"].get("CHECKPOINT", "")] = {"queue_id": r["id"], "status": r["status"], "overwrite": r["env"].get("OVERWRITE") == "1"}
    return out


def cancel(qid: int) -> bool:
    con = _db()
    with con:
        n = con.execute("UPDATE eval_queue SET status='cancelled', finished_at=? WHERE id=? AND status='queued'",
                        (_now(), qid)).rowcount
    if n:
        event("info", f"cancelled #{qid}")
    return bool(n)


def requeue(qid: int) -> bool:
    con = _db()
    row = con.execute("SELECT kind FROM eval_queue WHERE id=?", (qid,)).fetchone()
    if row and (row[0] or "vbench") == "vbench" and not VBENCH_SUBMISSION:
        raise PermissionError(VBENCH_DISABLED)
    with con:
        pos = con.execute("SELECT COALESCE(MAX(position),0)+1 FROM eval_queue").fetchone()[0]
        n = con.execute("""UPDATE eval_queue SET status='queued', position=?, run_dir=NULL, pid=NULL, gpus=NULL, error=NULL,
                           started_at=NULL, finished_at=NULL WHERE id=? AND status IN ('failed','cancelled')""",
                        (pos, qid)).rowcount
    return bool(n)


def move(qid: int, direction: str) -> bool:
    con = _db()
    with con:
        rows = _rows(con, "WHERE status='queued'")
        idx = next((i for i, r in enumerate(rows) if r["id"] == qid), None)
        if idx is None:
            return False
        j = idx - 1 if direction == "up" else idx + 1
        if j < 0 or j >= len(rows):
            return False
        a, b = rows[idx], rows[j]
        con.execute("UPDATE eval_queue SET position=? WHERE id=?", (b["position"], a["id"]))
        con.execute("UPDATE eval_queue SET position=? WHERE id=?", (a["position"], b["id"]))
    return True


# ----------------------------------------------------------------------- launch
def _slug(variant_id: str, env: dict) -> str:
    src = env.get("CHECKPOINT", "hf")
    src = src.replace("E0029/", "").replace("@", "s").replace("/", "-").lower()
    backend = "vsqa" if env.get("ATTENTION_BACKEND") == "VSQA" else "bf16"
    mode = "official" if env.get("SAMPLE_MODE") == "official" else f"s{env.get('SAMPLES_PER_PROMPT', '1')}"
    dims = env.get("DIMENSIONS", "")
    dims_tag = "q7" if dims.count(",") == 6 or not dims else f"d{dims.count(',') + 1}"
    lim = f"-l{env['LIMIT']}" if env.get("LIMIT") else ""
    return f"e0030-{variant_id.lower()}-{src}-{backend}-{dims_tag}-{mode}{lim}"[:80]


def _launch(row: dict, alloc: int, gpus: list[int], node: str | None) -> None:
    if row.get("kind") == "inf_val":
        _launch_inf_val(row, alloc, gpus)
        return
    num_gpus = len(gpus)
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    run_name = _slug(row["variant_id"], row["env"])
    run_dir = Path(Store().bind(row["variant_id"]))
    env = {**os.environ, **row["env"], "ALLOCATION_ID": str(alloc), "RUN_NAME": run_name,
           "VARIANT": row["variant_id"], "NUM_GPUS": str(num_gpus),
           "GPU_IDS": ",".join(map(str, gpus)), "PROJECT_ROOT": str(PROJECT_ROOT)}
    env.pop("VIRTUAL_ENV", None)
    launch_dir = PROJECT_ROOT / "vbench_eval/outputs/queues"
    launch_dir.mkdir(parents=True, exist_ok=True)
    launch_prefix = launch_dir / f"dashboard-{row['id']}-{stamp}"
    launcher_log = launch_prefix.with_suffix(".log").open("ab")
    proc = subprocess.Popen(["bash", str(PROJECT_ROOT / EVAL_LAUNCHER)], cwd=PROJECT_ROOT, env=env,
                            stdout=launcher_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    launch_prefix.with_suffix(".json").write_text(json.dumps(
        {"queue_id": row["id"], "variant": row["variant_id"], "pid": proc.pid, "allocation": alloc,
         "env": row["env"], "started_at": _now()}, indent=2) + "\n")
    rel = runs.rel(run_dir)
    con = _db()
    with con:
        con.execute("UPDATE eval_queue SET status='running', run_dir=?, pid=?, allocation=?, gpus=?, started_at=? WHERE id=?",
                    (rel, proc.pid, alloc, ",".join(map(str, gpus)), _now(), row["id"]))
    cmd = " ".join(f"{k}={v}" for k, v in sorted(row["env"].items())) + f" ALLOCATION_ID={alloc} bash {EVAL_LAUNCHER}"
    v = results.get(EVAL_EXP, row["variant_id"]) or {}
    rc, out = memon("run", "record", rel, "--status", "RUNNING", "--name", f"E0030 {row['variant_id']} {v.get('name', '')}"[:120],
                    "--entry", EVAL_LAUNCHER, "--command", cmd, "--pid", str(proc.pid),
                    "--gpus", ",".join(map(str, gpus)), *(["--host", node] if node else []))
    if rc != 0:
        event("warn", f"memon run record failed for {rel}: {out[-300:]}")
    rc, out = memon("experiment", "link", EVAL_EXP, rel)
    if rc != 0:
        event("warn", f"memon experiment link failed for {rel}: {out[-300:]}")
    event("info", f"launched #{row['id']} {row['variant_id']} -> {rel} (pid {proc.pid}, allocation {alloc})")


def _launch_inf_val(row: dict, alloc: int, gpus: list[int]) -> None:
    out_dir = PROJECT_ROOT / row["env"]["OUT_DIR"]
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **row["env"], "ALLOCATION_ID": str(alloc), "GPU_IDS": ",".join(map(str, gpus)),
           "PROJECT_ROOT": str(PROJECT_ROOT)}
    env.pop("VIRTUAL_ENV", None)
    log = (out_dir / "launcher.log").open("ab")
    # Run a snapshot of the launcher: bash reads scripts incrementally, so editing dashboard/inf_val.sh
    # while a job runs would otherwise corrupt that job.
    dash_dir = Path(__file__).resolve().parent
    launcher = out_dir / "inf_val.launcher.sh"
    launcher.write_bytes((dash_dir / "inf_val.sh").read_bytes())
    env["DASH_DIR"] = str(dash_dir)
    proc = subprocess.Popen(["bash", str(launcher)], cwd=PROJECT_ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    (out_dir / "dashboard_launch.json").write_text(json.dumps(
        {"queue_id": row["id"], "variant": row["variant_id"], "pid": proc.pid, "allocation": alloc, "env": row["env"],
         "started_at": _now()}, indent=2) + "\n")
    con = _db()
    with con:
        con.execute("UPDATE eval_queue SET status='running', run_dir=?, pid=?, allocation=?, gpus=?, started_at=? WHERE id=?",
                    (row["env"]["OUT_DIR"], proc.pid, alloc, ",".join(map(str, gpus)), _now(), row["id"]))
    event("info", f"launched inf_val #{row['id']} {row['env'].get('CHECKPOINT')} -> {row['env']['OUT_DIR']} (pid {proc.pid}, allocation {alloc}, gpus {gpus})")


def _finalize_inf_val(row: dict, ok: bool, error: str | None) -> None:
    con = _db()
    with con:
        con.execute("UPDATE eval_queue SET status=?, finished_at=?, error=? WHERE id=?",
                    ("finished" if ok else "failed", _now(), error, row["id"]))
    d = PROJECT_ROOT / row["run_dir"]
    n = len(list(d.glob("[0-9][0-9].mp4"))) if d.is_dir() else 0
    runs.invalidate(str(d.parent.parent))
    event("info" if ok else "error", f"inf_val #{row['id']} {row['env'].get('CHECKPOINT')} {'finished' if ok else 'failed'}: {error or f'{n} videos'}")


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _finalize(row: dict, ok: bool, error: str | None) -> None:
    rel = row["run_dir"]
    try:
        store = Store()
        record = store.read_record(store.row(row["variant_id"]))
        if not ok and record["status"] not in ("FAILED", "COMPLETED"):
            execution = record.get("execution") or {}
            invocation = execution.get("invocation") if execution.get("phase") == "RUNNING" else f"dashboard-{row['id']}"
            store.event(row["variant_id"], "INTERRUPTED", invocation,
                        execution.get("output_dir"), error=error or "launcher lost")
        else:
            store.sync()
        checks = lint_and_submit(EVAL_EXP)
        if checks["lint_rc"] != 0:
            event("warn", f"lint after {row['variant_id']} writeback: {checks['lint'][-300:]}")
    except Exception as e:
        event("error", f"writeback failed for {row['variant_id']}: {e}")
        return
    con = _db()
    with con:
        con.execute("UPDATE eval_queue SET status=?, finished_at=?, error=? WHERE id=?",
                    ("finished" if ok else "failed", _now(), error, row["id"]))
    rc, out = memon("run", "status", "set", Path(rel).name, "--to", "FINISHED" if ok else "FAILED")
    if rc != 0:
        event("warn", f"memon run status set failed for {rel}: {out[-300:]}")
    event("info" if ok else "error", f"#{row['id']} {row['variant_id']} {'finished' if ok else 'failed'}: {error or 'Run records synchronized'}")


def _reconcile(con) -> None:
    for row in _rows(con, "WHERE status='running'"):
        if row.get("kind") == "inf_val":
            d = PROJECT_ROOT / row["run_dir"] if row["run_dir"] else None
            if not d or not d.is_dir():
                _finalize_inf_val(row, False, "output directory missing"); continue
            ec = (d / "exit_code").read_text().strip() if (d / "exit_code").is_file() else None
            if ec is not None:
                _finalize_inf_val(row, ec == "0", None if ec == "0" else f"exit_code {ec}"); continue
            if not _pid_alive(row["pid"]):
                _finalize_inf_val(row, False, "launcher process lost before completion (dashboard restart?)")
            continue
        d = runs.run_dir(row["run_dir"]) if row["run_dir"] else None
        if not d or not d.is_dir():
            _finalize(row, False, "run directory missing")
            continue
        try:
            store = Store()
            record = store.read_record(store.row(row["variant_id"]))
            execution = record.get("execution") or {}
            if execution.get("phase") == "RUNNING" and _pid_alive(row["pid"]):
                continue
            if execution.get("phase") == "FINISHED":
                _finalize(row, True, None)
            elif execution.get("phase") in ("FAILED", "INTERRUPTED"):
                _finalize(row, False, execution.get("error", "evaluation failed"))
            elif not _pid_alive(row["pid"]):
                _finalize(row, False, "launcher process lost before completion")
        except (ValueError, OSError) as error:
            event("error", f"Run evidence for {row['variant_id']} cannot be reconciled: {error}")


def kill(qid: int) -> bool:
    con = _db()
    row = next(iter(_rows(con, "WHERE id=? AND status='running'", (qid,))), None)
    if not row or not row["pid"]:
        return False
    try:
        os.killpg(row["pid"], signal.SIGTERM)
    except OSError:
        pass
    time.sleep(2)
    (_finalize_inf_val if row.get("kind") == "inf_val" else _finalize)(row, False, "killed from dashboard")
    return True


# ------------------------------------------------------------------------- tick
LIVE_LOG_SEC = 1800


def _training_on_allocation(alloc: int) -> list[str]:
    """Training run directories that report this allocation and still write logs (stale RUNNING markers ignored)."""
    hits = []
    now = time.time()
    for info in runs.list_all():
        ex = info.get("execution") or {}
        if ex.get("state") == "RUNNING" and ex.get("job") == alloc and info["kind"] == "train" \
                and info.get("log_mtime") and now - info["log_mtime"] < LIVE_LOG_SEC:
            hits.append(info["id"])
    return hits


def _gpus_needed(row: dict, num_gpus: int) -> int:
    """Every job takes the whole node: one job at a time, inf_val shards its 12 captions over the GPUs."""
    return num_gpus


def tick() -> dict:
    """One scheduling pass: launch the next queued job when every GPU of the node is free
    (not held by a running dashboard job and idle in nvidia-smi)."""
    with _lock:
        con = _db()
        _reconcile(con)
        st = settings()
        running = _rows(con, "WHERE status='running'")
        queued = _rows(con, "WHERE status='queued'")
        set_setting("last_tick", _now())
        held = {int(g) for r in running for g in (r.get("gpus") or "").split(",") if g != ""}
        decision = None
        if st["paused"]:
            decision = "paused" + (f" ({len(running)} running)" if running else "")
        elif not queued:
            decision = "queue empty" + (f" ({len(running)} running on gpus {sorted(held)})" if running else "")
        elif not st["eval_allocation_id"]:
            decision = "no eval allocation configured"
        else:
            alloc = st["eval_allocation_id"]
            a = slurm.allocation(alloc)
            if not a or a["state"] != "RUNNING":
                decision = f"allocation {alloc} not RUNNING in squeue"
            elif (busy := _training_on_allocation(alloc)):
                decision = f"training run on allocation {alloc}: {busy}"
            else:
                probe = slurm.probe_gpus(alloc, st["num_gpus"])
                set_setting("last_probe", json.dumps({"ts": _now(), **probe}))
                if not probe["ok"]:
                    decision = f"probe failed: {probe['error']}"
                else:
                    free = [g["gpu"] for g in probe["gpus"] if g["gpu"] not in held and g["used_mib"] <= IDLE_GPU_MIB]
                    foreign = [g["gpu"] for g in probe["gpus"] if g["gpu"] not in held and g["used_mib"] > IDLE_GPU_MIB]
                    launched = []
                    for row in queued:
                        need = _gpus_needed(row, st["num_gpus"])
                        if need > len(free):
                            continue
                        gpus = free[:need]
                        try:
                            _launch(row, alloc, gpus, a.get("node"))
                            launched.append(f"#{row['id']}→gpu{','.join(map(str, gpus))}")
                            free = free[need:]
                        except Exception as e:
                            event("error", f"launch #{row['id']} failed: {e}")
                    decision = (f"launched {' '.join(launched)}" if launched else
                                f"waiting: free gpus {free}, held {sorted(held)}" + (f", busy by others {foreign}" if foreign else ""))
        set_setting("last_decision", decision)
        return {"decision": decision}


class Loop(threading.Thread):
    def __init__(self):
        super().__init__(name="eval-scheduler", daemon=True)
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                tick()
            except Exception as e:
                log.exception("scheduler tick failed: %s", e)
            self.stop.wait(SCHEDULER_TICK_SEC)
