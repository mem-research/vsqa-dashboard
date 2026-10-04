"""One sqlite state file, one connection per thread, schema created once.

The home directory is a network filesystem, so WAL mode is avoided (rollback
journal is the NFS-safe choice) and connections are reused: opening a fresh
connection per query cost ~100 ms there.
"""
from __future__ import annotations

import sqlite3
import threading

from .config import STATE_DB, STATE_DIR

_local = threading.local()
_schema_lock = threading.Lock()
_schema_done = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS wb_runs(
    run_id TEXT PRIMARY KEY, url TEXT, name TEXT, state TEXT, summary_step INTEGER,
    last_step INTEGER, summary TEXT, fetched_at REAL, error TEXT);
CREATE TABLE IF NOT EXISTS wb_scalars(
    run_id TEXT, key TEXT, step INTEGER, value REAL, PRIMARY KEY(run_id, key, step));
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS eval_queue(
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL DEFAULT 'vbench', variant_id TEXT NOT NULL, env TEXT NOT NULL,
    status TEXT NOT NULL, position REAL NOT NULL, run_dir TEXT, pid INTEGER, allocation INTEGER, gpus TEXT,
    created_at TEXT, started_at TEXT, finished_at TEXT, error TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, level TEXT, message TEXT);
"""


def connect() -> sqlite3.Connection:
    global _schema_done
    con = getattr(_local, "con", None)
    if con is None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(STATE_DB, timeout=30)
        con.execute("PRAGMA journal_mode=DELETE")
        con.execute("PRAGMA synchronous=NORMAL")
        con.row_factory = sqlite3.Row
        _local.con = con
    if not _schema_done:
        with _schema_lock:
            if not _schema_done:
                con.executescript(SCHEMA)
                cols = {r[1] for r in con.execute("PRAGMA table_info(eval_queue)")}
                if "kind" not in cols:
                    con.execute("ALTER TABLE eval_queue ADD COLUMN kind TEXT NOT NULL DEFAULT 'vbench'")
                if "gpus" not in cols:
                    con.execute("ALTER TABLE eval_queue ADD COLUMN gpus TEXT")
                _schema_done = True
    return con
