"""
VelocityEngine entry point.

Run:
    .venv/bin/python auto_trader.py
"""

import os
import sys
import atexit
import fcntl

sys.path.insert(0, os.path.dirname(__file__))

from src.config import INSTANCE_LOCK_FILE
from src.engine import VelocityEngine


_LOCK_HANDLE = None


def _acquire_instance_lock():
    """Prevent two live engines from trading the same account/client id."""
    global _LOCK_HANDLE
    # O_CREAT|O_RDWR without O_TRUNC: opening must not destroy the current
    # holder's visible PID before flock() has actually confirmed exclusivity.
    # open(path, "w") truncates immediately on open, so a second instance
    # wiped the first instance's recorded PID from the lock file even though
    # its own flock() below then correctly failed and exited -- misleading
    # any monitoring tooling that reads the lock file's contents.
    fd = os.open(INSTANCE_LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
    _LOCK_HANDLE = os.fdopen(fd, "r+")
    try:
        fcntl.flock(_LOCK_HANDLE, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(
            f"Another VelocityEngine instance is already running "
            f"(lock: {INSTANCE_LOCK_FILE})."
        )
    _LOCK_HANDLE.seek(0)
    _LOCK_HANDLE.truncate()
    _LOCK_HANDLE.write(str(os.getpid()))
    _LOCK_HANDLE.flush()


def _release_instance_lock():
    if _LOCK_HANDLE is not None:
        fcntl.flock(_LOCK_HANDLE, fcntl.LOCK_UN)
        _LOCK_HANDLE.close()


if __name__ == "__main__":
    _acquire_instance_lock()
    atexit.register(_release_instance_lock)
    engine = VelocityEngine()
    engine.run()
