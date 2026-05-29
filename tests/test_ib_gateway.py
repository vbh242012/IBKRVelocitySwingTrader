"""Tests for optional IB Gateway / IBC auto-start supervision."""

from unittest.mock import MagicMock

import pytest

from src.ib_gateway import (
    IBGatewayAutoStartConfig,
    IBGatewayStartupError,
    ensure_ib_gateway_ready,
)


def _cfg(**overrides):
    values = {
        "enabled": False,
        "command": "",
        "host": "127.0.0.1",
        "port": 4002,
        "timeout_sec": 2.0,
        "poll_sec": 0.25,
        "stop_on_exit": False,
        "log_file": "/tmp/velocity-ib-gateway-test.log",
    }
    values.update(overrides)
    return IBGatewayAutoStartConfig(**values)


def test_ready_port_returns_true_without_starting(monkeypatch):
    import src.ib_gateway as gw

    monkeypatch.setattr(gw, "_port_open", lambda *_args, **_kwargs: True)
    start = MagicMock()
    monkeypatch.setattr(gw, "_start_gateway_process", start)

    assert ensure_ib_gateway_ready(_cfg(enabled=True, command="/bin/echo gateway")) is True
    start.assert_not_called()


def test_disabled_autostart_returns_false_when_port_closed(monkeypatch):
    import src.ib_gateway as gw

    monkeypatch.setattr(gw, "_port_open", lambda *_args, **_kwargs: False)

    assert ensure_ib_gateway_ready(_cfg(enabled=False)) is False


def test_enabled_autostart_requires_command(monkeypatch):
    import src.ib_gateway as gw

    monkeypatch.setattr(gw, "_port_open", lambda *_args, **_kwargs: False)

    with pytest.raises(IBGatewayStartupError, match="START_CMD is empty"):
        ensure_ib_gateway_ready(_cfg(enabled=True, command=""))


def test_enabled_autostart_waits_until_port_opens(monkeypatch):
    import src.ib_gateway as gw

    calls = iter([False, True])
    monkeypatch.setattr(gw, "_port_open", lambda *_args, **_kwargs: next(calls))
    proc = MagicMock()
    proc.poll.return_value = None
    start = MagicMock(return_value=proc)
    monkeypatch.setattr(gw, "_start_gateway_process", start)
    monkeypatch.setattr(gw.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gw, "_gateway_process", None)

    assert ensure_ib_gateway_ready(_cfg(enabled=True, command="/bin/echo gateway")) is True
    start.assert_called_once()


def test_enabled_autostart_raises_when_launcher_exits(monkeypatch):
    import src.ib_gateway as gw

    monkeypatch.setattr(gw, "_port_open", lambda *_args, **_kwargs: False)
    proc = MagicMock()
    proc.poll.return_value = 1
    monkeypatch.setattr(gw, "_start_gateway_process", MagicMock(return_value=proc))
    monkeypatch.setattr(gw, "_gateway_process", None)

    with pytest.raises(IBGatewayStartupError, match="launcher exited"):
        ensure_ib_gateway_ready(_cfg(enabled=True, command="/bin/echo gateway"))


def test_gateway_launcher_timestamp_is_eastern():
    import src.ib_gateway as gw

    stamp = gw._et_timestamp()

    assert stamp.endswith((" EST", " EDT"))
