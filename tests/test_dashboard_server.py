import json
import re
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
            "entry_risk_per_share": 5.0,
            "time": now.isoformat(),
            "strategy_profile": "indicator_swing",
            "relative_strength_63d": 0.067,
            "relative_strength_126d": 0.041,
            "return_13w": 0.12,
            "return_26w": 0.21,
            "weekly_uptrend": True,
            "price_vs_52w_high": 0.93,
            "analyst_rating_score": 0.4,
            "analyst_rating_total": "12",
            "analyst_rating_strong_buy": "3",
            "analyst_rating_buy": "4",
            "analyst_rating_hold": "4",
            "analyst_rating_sell": "1",
            "analyst_rating_strong_sell": "0",
            "analyst_rating_source": "csv",
            "analyst_rating_period": "2026-06-01",
            "protection_status": "confirmed",
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
    assert data["strategy"]["profile"] == "indicator_swing"
    assert data["strategy"]["scoring_model"] == "indicator_swing"
    assert data["strategy"]["eod_quality_cleanup"] is False
    assert data["strategy"]["allow_bear_phase_entries"] is False
    assert set(data["strategy"]["indicator_sleeves"]) == {"ma_cross"}
    assert data["positions"][0]["relative_strength_63d"] == pytest.approx(0.067)
    assert data["positions"][0]["r_multiple"] == pytest.approx(0.2)
    assert data["positions"][0]["analyst_rating_total"] == 12
    assert data["positions"][0]["analyst_rating_strong_buy"] == 3
    assert data["positions"][0]["analyst_rating_buy"] == 4
    assert data["positions"][0]["analyst_rating_hold"] == 4
    assert data["positions"][0]["analyst_rating_sell"] == 1
    assert data["positions"][0]["analyst_rating_strong_sell"] == 0
    assert data["positions"][0]["protection_status"] == "confirmed"


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


def test_dashboard_pnl_uses_et_calendar_period_baselines(dashboard_files, monkeypatch):
    _, _, hist_file = dashboard_files
    tz_ny = pytz.timezone("US/Eastern")

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            current = cls(2026, 6, 4, 10, 0)
            return tz.localize(current) if tz else current

    monkeypatch.setattr(dashboard, "datetime", FixedDateTime)
    _write_json(hist_file, [
        {"ts": tz_ny.localize(datetime(2026, 5, 29, 12, 0)).isoformat(), "eq": 900.0},
        {"ts": tz_ny.localize(datetime(2026, 6, 1, 0, 10)).isoformat(), "eq": 950.0},
        {"ts": tz_ny.localize(datetime(2026, 6, 3, 9, 59)).isoformat(), "eq": 1000.0},
        {"ts": tz_ny.localize(datetime(2026, 6, 4, 0, 5)).isoformat(), "eq": 1010.0},
        {"ts": tz_ny.localize(datetime(2026, 6, 4, 9, 45)).isoformat(), "eq": 1020.0},
    ])

    pnl = dashboard._pnl(1030.0)

    assert pnl["daily"] == {"amount": 20.0, "pct": 1.98}
    assert pnl["weekly"] == {"amount": 80.0, "pct": 8.42}
    assert pnl["monthly"] == {"amount": 80.0, "pct": 8.42}
    assert pnl["overall"] == {"amount": 130.0, "pct": 14.44}


def test_dashboard_equity_chart_uses_intraday_time_labels():
    html = dashboard._HTML

    assert 'id="eq-window"' in html
    assert "INTRADAY TODAY" in html
    assert "toLocaleTimeString" in html
    assert "toLocaleDateString" in html
    assert "Multi-Indicator Swing" in html
    assert "normal positions are not churned out" in html
    assert "EMA20 > SMA50" in html
    assert "Bollinger Sleeve" in html
    assert "PSAR Sleeve" in html
    assert "No EOD Churn" in html
    assert "Swing Time Stop" in html
    assert "Analyst Downgrade" in html
    assert "ANALYST SCORE" in html
    assert "ANALYST VOTES" in html
    assert "<th>STOP (TRAIL)</th>" in html
    # R (r_multiple) and RS 63D columns were removed from the dashboard in
    # commit 8588cbf; enforce that they stay gone.
    assert "<th>R</th>" not in html
    assert "<th>RS 63D</th>" not in html
    # STRATEGY column removed 2026-08-05: with only one strategy taking new
    # entries (bollinger_reversion_standalone), per-row strategy attribution
    # no longer earns a dedicated column.
    assert "<th>STRATEGY</th>" not in html
    # SCORE column removed 2026-08-05: bollinger_reversion_standalone (the
    # only strategy taking new entries) never populates p.score, so the
    # column would render '—' for every row once ma_cross positions close.
    assert "<th>SCORE</th>" not in html
    assert "B:${buyVotes" in html
    assert "__SWING_" not in html
    assert "Velocity Exit" not in html

    # Every colspan on the empty-state placeholder rows must match the actual
    # number of <th> columns in the open-positions table, or the "waiting for
    # data" / "no open positions" rows render with a broken/misaligned width.
    th_count = html.count("<th>")
    for colspan_html in re.findall(r'colspan="(\d+)"', html):
        assert int(colspan_html) == th_count, (
            f"colspan={colspan_html} does not match actual <th> count={th_count}"
        )
