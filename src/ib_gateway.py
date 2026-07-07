"""Optional IB Gateway / IBC launcher and readiness checks.

This module deliberately does not know broker usernames, passwords, or 2FA
secrets.  It can supervise an external launcher command such as an IBC startup
script and wait until the configured IB API port is reachable.
"""

from __future__ import annotations

import atexit
import os
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pytz

from src.config import (
    IB_GATEWAY_AUTO_START,
    IB_GATEWAY_LOG_FILE,
    IB_GATEWAY_START_CMD,
    IB_GATEWAY_START_POLL_SEC,
    IB_GATEWAY_START_TIMEOUT_SEC,
    IB_GATEWAY_STOP_ON_EXIT,
    IB_HOST,
    IB_PORT,
    LOG_DIR,
    TZ_ET,
)


_TZ_NY = TZ_ET


class IBGatewayStartupError(RuntimeError):
    """Raised when the configured IB Gateway launcher cannot make the API ready."""


@dataclass(frozen=True)
class IBGatewayAutoStartConfig:
    enabled: bool = IB_GATEWAY_AUTO_START
    command: str = IB_GATEWAY_START_CMD
    host: str = IB_HOST
    port: int = IB_PORT
    timeout_sec: float = IB_GATEWAY_START_TIMEOUT_SEC
    poll_sec: float = IB_GATEWAY_START_POLL_SEC
    stop_on_exit: bool = IB_GATEWAY_STOP_ON_EXIT
    log_file: str = IB_GATEWAY_LOG_FILE


_gateway_process: Optional[subprocess.Popen] = None
_atexit_registered = False


def _et_timestamp() -> str:
    return datetime.now(_TZ_NY).strftime("%Y-%m-%d %H:%M:%S %Z")


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _launcher_label(command: str) -> str:
    try:
        argv = shlex.split(command)
    except ValueError:
        return "<invalid command>"
    if not argv:
        return "<empty command>"
    return os.path.basename(argv[0]) or argv[0]


def _stop_gateway_process() -> None:
    global _gateway_process
    proc = _gateway_process
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _start_gateway_process(config: IBGatewayAutoStartConfig) -> subprocess.Popen:
    global _atexit_registered, _gateway_process

    try:
        argv = shlex.split(config.command)
    except ValueError as exc:
        raise IBGatewayStartupError(f"Invalid IB Gateway start command: {exc}") from exc

    if not argv:
        raise IBGatewayStartupError(
            "IB Gateway auto-start is enabled but VELOCITY_IB_GATEWAY_START_CMD is empty."
        )

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.log_file) or ".", exist_ok=True)
    log_handle = open(config.log_file, "a", buffering=1)
    log_handle.write(
        f"\n[{_et_timestamp()}] starting IB Gateway launcher: "
        f"{_launcher_label(config.command)}\n"
    )

    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        log_handle.close()
        raise IBGatewayStartupError(f"Could not start IB Gateway launcher: {exc}") from exc

    _gateway_process = proc
    if config.stop_on_exit and not _atexit_registered:
        atexit.register(_stop_gateway_process)
        _atexit_registered = True
    return proc


def ensure_ib_gateway_ready(config: Optional[IBGatewayAutoStartConfig] = None) -> bool:
    """Ensure the configured IB API socket is reachable.

    Returns True when the port is already open or becomes open after launching
    the configured command.  Returns False when auto-start is disabled and the
    port is not open.  Raises IBGatewayStartupError when auto-start is enabled
    but startup fails.
    """

    config = config or IBGatewayAutoStartConfig()
    if _port_open(config.host, config.port):
        return True

    if not config.enabled:
        return False

    global _gateway_process
    if _gateway_process is None or _gateway_process.poll() is not None:
        _gateway_process = _start_gateway_process(config)

    deadline = time.monotonic() + max(1.0, float(config.timeout_sec))
    poll_sec = max(0.25, float(config.poll_sec))
    while time.monotonic() < deadline:
        if _port_open(config.host, config.port):
            return True
        if _gateway_process.poll() is not None:
            raise IBGatewayStartupError(
                f"IB Gateway launcher exited before API port {config.host}:{config.port} became ready."
            )
        time.sleep(poll_sec)

    raise IBGatewayStartupError(
        f"IB Gateway API port {config.host}:{config.port} was not ready within "
        f"{config.timeout_sec:.0f}s after starting {_launcher_label(config.command)}."
    )
