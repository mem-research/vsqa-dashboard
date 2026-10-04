"""Incremental W&B scalar cache in sqlite.

`refresh(run_id)` pulls rows with `_step > last cached step` through
`Run.scan_history(min_step=...)`, keeps only the configured scalar keys and
records run state/summary. A background thread refreshes every run that is
not yet `finished`, or whose cached step trails its summary `_step`.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from urllib.parse import urlparse

from .. import db
from ..config import WANDB_ENTITY_PROJECT, WANDB_REFRESH_SEC, WANDB_SCALAR_KEYS

log = logging.getLogger("dashboard.wandb")
_lock = threading.Lock()
_api = None


def _db():
    return db.connect()


def run_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parts = [p for p in urlparse(url).path.split("/") if p]
    if "runs" in parts:
        return parts[parts.index("runs") + 1]
    return parts[-1] if parts else None


def _get_api():
    global _api
    if _api is None:
        import wandb
        _api = wandb.Api(timeout=60)
    return _api


def refresh(run_id: str, url: str | None = None) -> dict:
    con = _db()
    row = con.execute("SELECT last_step, state FROM wb_runs WHERE run_id=?", (run_id,)).fetchone()
    last_step = row[0] if row and row[0] is not None else -1
    try:
        if not WANDB_ENTITY_PROJECT:
            raise RuntimeError("VSQA_WANDB_PROJECT is not set")
        run = _get_api().run(f"{WANDB_ENTITY_PROJECT}/{run_id}")
        summary = {k: v for k, v in run.summary.items()
                   if not k.startswith("_") and isinstance(v, (int, float, str, bool))}
        summary_step = int(run.summary.get("_step") or 0)
        rows = []
        for h in run.scan_history(min_step=last_step + 1, page_size=500):
            step = h.get("_step")
            if step is None:
                continue
            for k in WANDB_SCALAR_KEYS:
                v = h.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    rows.append((run_id, k, int(step), float(v)))
            last_step = max(last_step, int(step))
        with con:
            con.executemany("INSERT OR REPLACE INTO wb_scalars VALUES(?,?,?,?)", rows)
            con.execute("INSERT OR REPLACE INTO wb_runs VALUES(?,?,?,?,?,?,?,?,NULL)",
                        (run_id, url or run.url, run.name, run.state, summary_step, last_step,
                         json.dumps(summary), time.time()))
        return {"run_id": run_id, "state": run.state, "summary_step": summary_step,
                "last_step": last_step, "new_rows": len(rows)}
    except Exception as e:  # network / auth / missing run
        with con:
            con.execute("""INSERT INTO wb_runs(run_id,url,fetched_at,error) VALUES(?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET fetched_at=excluded.fetched_at, error=excluded.error""",
                        (run_id, url, time.time(), str(e)[:500]))
        log.warning("wandb refresh %s failed: %s", run_id, e)
        return {"run_id": run_id, "error": str(e)}


def _meta_row(row) -> dict:
    return {"run_id": row[0], "url": row[1], "name": row[2], "state": row[3], "summary_step": row[4],
            "last_step": row[5], "summary": json.loads(row[6]) if row[6] else {}, "fetched_at": row[7],
            "error": row[8]}


def meta(run_id: str) -> dict | None:
    row = _db().execute("SELECT run_id,url,name,state,summary_step,last_step,summary,fetched_at,error "
                        "FROM wb_runs WHERE run_id=?", (run_id,)).fetchone()
    return _meta_row(row) if row else None


def all_meta() -> dict[str, dict]:
    """Read one current database snapshot instead of one query per Variant."""
    rows = _db().execute("SELECT run_id,url,name,state,summary_step,last_step,summary,fetched_at,error FROM wb_runs")
    return {row[0]: _meta_row(row) for row in rows}


def scalars(run_id: str, keys: list[str] | None = None) -> dict[str, list[list[float]]]:
    q = "SELECT key, step, value FROM wb_scalars WHERE run_id=?"
    args: list = [run_id]
    if keys:
        q += f" AND key IN ({','.join('?' * len(keys))})"
        args += keys
    q += " ORDER BY key, step"
    out: dict[str, list[list[float]]] = {}
    for k, s, v in _db().execute(q, args):
        out.setdefault(k, []).append([s, v])
    return out

def needs_refresh(run_id: str, live: bool) -> bool:
    m = meta(run_id)
    if m is None:
        return True
    if m.get("error") and (time.time() - (m.get("fetched_at") or 0)) > WANDB_REFRESH_SEC:
        return True
    if m.get("state") != "finished" or live:
        return (time.time() - (m.get("fetched_at") or 0)) > WANDB_REFRESH_SEC
    return (m.get("last_step") or -1) < (m.get("summary_step") or 0) - 1


class Refresher(threading.Thread):
    """Periodically refresh every tracked run; `targets()` yields (run_id, url, live)."""

    def __init__(self, targets):
        super().__init__(name="wandb-refresher", daemon=True)
        self.targets = targets
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                for run_id, url, live in list(self.targets()):
                    if self.stop.is_set():
                        break
                    if needs_refresh(run_id, live):
                        with _lock:
                            refresh(run_id, url)
            except Exception as e:
                log.exception("refresher loop failed: %s", e)
            self.stop.wait(WANDB_REFRESH_SEC)
