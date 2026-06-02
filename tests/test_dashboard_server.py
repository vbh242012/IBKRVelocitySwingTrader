import json
from datetime import datetime

import pytz
import pytest

import dashboard_server as dashboard
from src.config import SETTLED_CASH_DEPLOYMENT_PCT, EOD_EXIT_TIME


def _write_json(path, data):
    path.write_text(json.dumps(data))


def _payload(response):
    return json.loads(response.body.decode("utf-8"))


@pytest.fixture
def dashboard_files(tmp_path, monkeypatch):
    state_file = tmp_path / "engine_state.json"
    dash_file = tmp_path / "dashboard_data.json"
    hist_file = tmp_path / "equity_history.json"
    monkeypatch.setattr(dashboard, "STATE_FILE", str(state_file))
    monkeypatch.setattr(dashboard, "DASHBOARD_FILE", str(dash_file))
    monkeypatch.setattr(dashboard, "EQUITY_HIST_FILE", str(hist_file))
    return state_file, dash_file, hist_file


def test_dashboard_bucket_uses_deployable_settled_cash(dashboard_files):
    state_file, dash_file, hist_file = dashboard_files
    now = pytz.timezone("US/Eastern").localize(datetime(2024, 6, 5, 10, 30))
    _write_json(state_file, {
        "AAPL": {
            "fill_price": 100.0,
            "price": 100.0,
            "current_price": 101.0,
            "qty": 1.0,
            "stop_loss": 90.0,
            "effective_stop": 95.0,
            "time": now.isoformat(),
        }
    })
    _write_json(dash_file, {
        "equity": 1400.0,
        "settled_cash": 1000.0,
        "connected": True,
    })
    _write_json(hist_file, [])

    data = _payload(dashboard.get_state())

    assert data["cash"] == pytest.approx(1000.0)
    assert data["deployable_cash"] == pytest.approx(1000.0 * SETTLED_CASH_DEPLOYMENT_PCT)
    assert data["bucket_size"] == pytest.approx(950.0)
    assert data["eod_exit_time"] == f"{EOD_EXIT_TIME[0]:02d}:{EOD_EXIT_TIME[1]:02d}"


def test_dashboard_blocks_bucket_when_buffer_pushes_cash_below_floor(dashboard_files):
    state_file, dash_file, hist_file = dashboard_files
    _write_json(state_file, {})
    _write_json(dash_file, {
        "equity": 1000.0,
        "settled_cash": 525.0,
        "connected": True,
    })
    _write_json(hist_file, [])

    data = _payload(dashboard.get_state())

    assert data["deployable_cash"] == pytest.approx(525.0 * SETTLED_CASH_DEPLOYMENT_PCT)
    assert data["bucket_size"] == pytest.approx(0.0)


def test_dashboard_equity_chart_uses_intraday_time_labels():
    html = dashboard._HTML

    assert 'id="eq-window"' in html
    assert "INTRADAY TODAY" in html
    assert "toLocaleTimeString" in html
    assert "toLocaleDateString" in html
    assert "3:50 PM ET" in html
    assert "EOD Profit Cleanup" in html
    assert "Velocity Exit" not in html
