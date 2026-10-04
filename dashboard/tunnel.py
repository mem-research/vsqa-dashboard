"""Reverse SSH tunnel that publishes the local dashboard port on the ingress host.

Started with the FastAPI app: `ssh -N -R 127.0.0.1:<remote_port>:127.0.0.1:<local_port> <host>`
is kept alive in a supervising thread (restart with backoff on exit). Caddy on the
ingress host reverse-proxies the domain to that remote port. Configure with
`VSQA_TUNNEL=<ssh-host>:<remote-port>` (empty disables).
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

from .config import PORT

log = logging.getLogger("dashboard.tunnel")
TUNNEL = os.environ.get("VSQA_TUNNEL", "")


class Tunnel(threading.Thread):
    def __init__(self, spec: str = TUNNEL, local_port: int = PORT):
        super().__init__(name="ssh-tunnel", daemon=True)
        self.host, _, port = spec.partition(":")
        self.remote_port = int(port or 0)
        self.local_port = local_port
        self.stop = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.status = {"host": self.host, "remote_port": self.remote_port, "state": "disabled" if not self.host else "starting",
                       "restarts": 0, "last_exit": None, "last_error": None, "since": None}

    def argv(self) -> list[str]:
        return ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=3", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new",
                "-R", f"127.0.0.1:{self.remote_port}:127.0.0.1:{self.local_port}", self.host]

    def run(self) -> None:
        if not self.host or not self.remote_port:
            return
        backoff = 5
        while not self.stop.is_set():
            try:
                self.proc = subprocess.Popen(self.argv(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.PIPE, text=True, start_new_session=True)
            except OSError as e:
                self.status.update(state="error", last_error=str(e))
                self.stop.wait(backoff)
                continue
            self.status.update(state="up", since=time.time(), last_error=None)
            log.info("tunnel up: %s -> %s:%s", self.local_port, self.host, self.remote_port)
            started = time.time()
            self.proc.wait()
            err = (self.proc.stderr.read() if self.proc.stderr else "").strip()[-300:]
            self.status.update(state="down", last_exit=self.proc.returncode, last_error=err or None,
                               restarts=self.status["restarts"] + 1)
            log.warning("tunnel exited rc=%s: %s", self.proc.returncode, err)
            backoff = 5 if time.time() - started > 60 else min(120, backoff * 2)
            self.stop.wait(backoff)

    def close(self) -> None:
        self.stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
