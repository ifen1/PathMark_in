"""Lightweight, dependency-free logging used by train.py.

We intentionally avoid wandb/tensorboard so the repo runs offline without
extra installs. Replace with your favorite tracker if needed.
"""
import sys
import time
from typing import Dict


class Logger:
    """Prepends a wall-clock timestamp and run tag to every log line."""

    def __init__(self, tag: str = "pathmark", stream=sys.stdout):
        self.tag = tag
        self.stream = stream
        self.t0 = time.time()

    def log(self, msg: str) -> None:
        dt = time.time() - self.t0
        h, rem = divmod(dt, 3600)
        m, s = divmod(rem, 60)
        self.stream.write(f"[{int(h):02d}:{int(m):02d}:{s:05.2f}] {self.tag}: {msg}\n")
        self.stream.flush()

    def log_kvs(self, **kvs: Dict[str, object]) -> None:
        """Log a flat dict of key=value pairs."""
        self.log("  ".join(f"{k}={v}" for k, v in kvs.items()))
