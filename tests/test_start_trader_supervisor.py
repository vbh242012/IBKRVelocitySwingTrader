import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START_TRADER = ROOT / "scripts" / "start_trader.sh"


def test_start_trader_shell_syntax_is_valid():
    subprocess.run(["bash", "-n", str(START_TRADER)], check=True)


def test_start_trader_has_stale_dashboard_watchdog():
    script = START_TRADER.read_text()

    assert "VELOCITY_TRADER_WATCHDOG_ENABLED" in script
    assert "VELOCITY_TRADER_HEARTBEAT_FILE" in script
    assert "dashboard_data.json" in script
    assert "WATCHDOG_STALE_SEC" in script
    assert "WATCHDOG_STARTUP_GRACE_SEC" in script
    assert "stop_stale_child_if_needed" in script
    assert "stale for ${age_sec}s; terminating trader pid" in script
