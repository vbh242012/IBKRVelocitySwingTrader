#!/usr/bin/env python
"""
VelocityEngine Web Dashboard
─────────────────────────────
Standalone FastAPI server — completely independent of the AutoTrader.

Start:   python3 dashboard_server.py
Open:    http://localhost:8080

The server only reads JSON files written by the engine:
  • engine_state.json    — open positions
  • dashboard_data.json  — equity, VIX, connection status, scan times
  • equity_history.json  — rolling 60-day equity snapshots for P&L

Closing/restarting this server never affects the running AutoTrader.
"""

import json
import math
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

import pytz
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from src.config import (
    STATE_FILE,
    DASHBOARD_FILE,
    EQUITY_HIST_FILE,
    LOG_FILE,
    MAX_POSITIONS_CAP,
    MIN_BUCKET_SIZE,
    SETTLED_CASH_DEPLOYMENT_PCT,
    STRATEGY_PROFILE,
    VIX_THRESHOLD,
    EOD_EXIT_TIME,
    EOD_HOLD_MIN_PROFIT_PCT,
    EOD_HOLD_DAY_RANGE_LOCATION_MIN,
    EOD_HOLD_RELATIVE_STRENGTH_MIN,
    EOD_HOLD_REQUIRE_STOP_CONFIRMED,
    TRAIL_PCT,
    SWING_RS_MIN_63D,
    SWING_RS_MIN_126D,
    SWING_MIN_13W_RETURN,
    SWING_MIN_26W_RETURN,
    SWING_MIN_PRICE_VS_52W_HIGH,
    SWING_MAX_MA20_EXTENSION,
    SWING_MIN_VOLUME_PACE,
    SWING_TIME_STOP_BARS,
    INDICATOR_SWING_MIN_SCORE,
    ANALYST_RATINGS_ENABLED,
    ANALYST_RATINGS_FREE_SOURCE,
    ANALYST_RATING_SCORE_WEIGHT,
    ANALYST_RATING_SELL_THRESHOLD,
    ANALYST_RATING_EXIT_ENABLED,
    DASHBOARD_ALLOWED_ORIGINS,
    DASHBOARD_TOKEN,
    TZ_ET,
)
from src.strategy_profiles import get_strategy_profile

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="VelocityEngine Dashboard", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=DASHBOARD_ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["Authorization"],
)


_LOCAL_CLIENT_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _request_authorized(request: Request) -> bool:
    """Optional bearer/query-token auth for non-local dashboard deployments."""
    if not DASHBOARD_TOKEN:
        # No token configured is only safe for a request actually arriving
        # over the loopback interface. The __main__ CLI guard below refuses
        # to bind to a non-localhost host without a token, but that check
        # only runs on that one launch path -- this request-level check
        # protects the app itself even if served by a different ASGI
        # runner (a bare `uvicorn dashboard_server:app --host 0.0.0.0`)
        # that bypasses __main__ entirely.
        client_host = getattr(request.client, "host", None)
        return client_host in _LOCAL_CLIENT_HOSTS
    auth = request.headers.get("authorization", "")
    expected = f"Bearer {DASHBOARD_TOKEN}"
    if secrets.compare_digest(auth, expected):
        return True
    query_token = request.query_params.get("token", "")
    return secrets.compare_digest(query_token, DASHBOARD_TOKEN)


@app.middleware("http")
async def require_dashboard_token(request: Request, call_next):
    if request.url.path.startswith("/api/") and not _request_authorized(request):
        return JSONResponse({"detail": "Dashboard token required"}, status_code=401)
    return await call_next(request)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _read_json(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _read_history() -> list:
    try:
        if os.path.exists(EQUITY_HIST_FILE):
            with open(EQUITY_HIST_FILE) as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _finite_float_or_none(value) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _int_or_none(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pnl(equity_now: float) -> dict:
    """Compute calendar daily / weekly / monthly / overall P&L in ET."""
    history = _read_history()
    tz_ny   = TZ_ET
    now     = datetime.now(tz_ny)

    def _parse_ts(ts: str) -> Optional[datetime]:
        try:
            dt = datetime.fromisoformat(ts)
        except (TypeError, ValueError):
            return None
        if dt.tzinfo is None:
            return tz_ny.localize(dt)
        return dt.astimezone(tz_ny)

    parsed_history = []
    for entry in history:
        ts = _parse_ts(entry.get("ts"))
        if ts is None:
            continue
        try:
            parsed_history.append((ts, float(entry["eq"])))
        except (KeyError, TypeError, ValueError):
            continue
    parsed_history.sort(key=lambda item: item[0])

    def _period_start(year: int, month: int, day: int) -> datetime:
        return tz_ny.localize(datetime(year, month, day))

    today_start = _period_start(now.year, now.month, now.day)
    week_date   = now.date() - timedelta(days=now.weekday())
    week_start  = _period_start(week_date.year, week_date.month, week_date.day)
    month_start = _period_start(now.year, now.month, 1)

    def _find_calendar_base(period_start: datetime) -> Optional[float]:
        current_period = [eq for ts, eq in parsed_history if ts >= period_start]
        if current_period:
            return current_period[0]
        # If the engine has not written a snapshot yet today, use the most
        # recent prior value as the closest available start-of-period baseline.
        prior = [eq for ts, eq in parsed_history if ts < period_start]
        if prior:
            return prior[-1]
        return None

    def _entry(base: Optional[float]) -> dict:
        if base is None or base == 0:
            return {"amount": None, "pct": None}
        amount = round(equity_now - base, 2)
        pct    = round(amount / base * 100, 2)
        return {"amount": amount, "pct": pct}

    # Overall: oldest valid snapshot in history (first real IBKR reading)
    overall_base = parsed_history[0][1] if parsed_history else None

    return {
        "daily":   _entry(_find_calendar_base(today_start)),
        "weekly":  _entry(_find_calendar_base(week_start)),
        "monthly": _entry(_find_calendar_base(month_start)),
        "overall": _entry(overall_base),
    }


def _market_open() -> bool:
    tz_ny  = TZ_ET
    now_ny = datetime.now(tz_ny)
    if now_ny.weekday() >= 5:
        return False
    return (9, 30) <= (now_ny.hour, now_ny.minute) <= (16, 0)


# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/api/state")
def get_state():
    state     = _read_json(STATE_FILE)
    dash_data = _read_json(DASHBOARD_FILE)
    profile   = get_strategy_profile(STRATEGY_PROFILE)
    scoring_model = profile.scoring_model

    equity         = float(dash_data.get("equity") or 0)
    # settled_cash is written by the engine from IBKR accountSummary().  It is
    # the cash used for position sizing and does not drift with unrealized P&L.
    raw_settled = dash_data.get("settled_cash")
    position_value = sum(
        float(d.get("current_price", d.get("price", 0))) * float(d.get("qty", 0))
        for d in state.values()
        if not d.get("pending")
    )
    settled_cash = float(raw_settled) if raw_settled is not None else 0.0

    tz_ny = TZ_ET
    now   = datetime.now(tz_ny)
    positions        = []
    total_unrealized = 0.0
    for sym, d in state.items():
        ep       = float(d.get("fill_price") or d.get("price", 0))
        qty      = float(d.get("qty",         0))
        if d.get("pending") or qty <= 0:
            continue
        raw_commission = d.get("commission")    # None until IB commission report arrives
        unit_price = (
            round(ep + float(raw_commission) / qty, 4)
            if raw_commission is not None and qty > 0
            else (round(ep, 4) if ep > 0 else None)
        )
        cur      = float(d.get("current_price", ep))   # live price written by engine
        sl           = float(d.get("stop_loss",     0))
        effective_sl = float(d.get("effective_stop", sl))  # IB trail watermark if tracked
        vol      = float(d.get("volume",      0))
        entry_ts = d.get("time") or now.isoformat()
        try:
            # TypeError as well as ValueError: entry_ts can be present but
            # non-string (e.g. a stray null/number in state), which
            # fromisoformat() rejects with TypeError, not ValueError. This
            # endpoint has no per-position exception isolation, so leaving
            # TypeError uncaught would take down the entire dashboard
            # (every position, equity, cash) over one malformed record.
            entry_dt = datetime.fromisoformat(entry_ts)
            if entry_dt.tzinfo is None:
                entry_dt = tz_ny.localize(entry_dt)
            hold_h = (now - entry_dt).total_seconds() / 3600
        except (ValueError, TypeError):
            hold_h = 0.0
        unreal     = round((cur - ep) * qty, 2)
        unreal_pct = round((cur - ep) / ep * 100, 2) if ep else 0.0
        risk_per_share = (
            _finite_float_or_none(d.get("entry_risk_per_share"))
            or _finite_float_or_none(d.get("stop_dist"))
        )
        r_multiple = (
            round((cur - ep) / risk_per_share, 2)
            if ep > 0 and risk_per_share and risk_per_share > 0 else None
        )
        total_unrealized += unreal
        positions.append({
            "symbol":          sym,
            "strategy_profile": d.get("strategy_profile") or profile.name,
            "entry_strategy":  d.get("entry_strategy"),
            "entry_strategy_label": d.get("entry_strategy_label") or d.get("entry_strategy"),
            "regime":          d.get("regime"),
            "entry_price":     ep,
            "unit_price":      unit_price,
            "current_price":   cur,
            "qty":             qty,
            "entry_qty":       _finite_float_or_none(d.get("entry_qty")),
            "total_amount":    round(ep * qty, 2),
            "unrealized":      unreal,
            "unrealized_pct":  unreal_pct,
            "r_multiple":      r_multiple,
            "stop_loss":       sl,
            "effective_stop":  effective_sl,
            "volume":          vol,
            "hold_hours":      round(hold_h, 2),
            "entry_time":      entry_ts,
            "score":           d.get("score"),
            "relative_strength_63d":  _finite_float_or_none(d.get("relative_strength_63d")),
            "relative_strength_126d": _finite_float_or_none(d.get("relative_strength_126d")),
            "return_13w":             _finite_float_or_none(d.get("return_13w")),
            "return_26w":             _finite_float_or_none(d.get("return_26w")),
            "weekly_uptrend":         bool(d.get("weekly_uptrend", False)),
            "price_vs_52w_high":      _finite_float_or_none(d.get("price_vs_52w_high")),
            "analyst_rating_score":   _finite_float_or_none(d.get("analyst_rating_score")),
            "analyst_rating_total":   _int_or_none(d.get("analyst_rating_total")),
            "analyst_rating_strong_buy":  _int_or_none(d.get("analyst_rating_strong_buy")),
            "analyst_rating_buy":         _int_or_none(d.get("analyst_rating_buy")),
            "analyst_rating_hold":        _int_or_none(d.get("analyst_rating_hold")),
            "analyst_rating_sell":        _int_or_none(d.get("analyst_rating_sell")),
            "analyst_rating_strong_sell": _int_or_none(d.get("analyst_rating_strong_sell")),
            "analyst_rating_source":  d.get("analyst_rating_source"),
            "analyst_rating_period":  d.get("analyst_rating_period"),
            "protection_status":      d.get("protection_status"),
            "protection_reason":      d.get("protection_reason"),
        })

    max_positions = (
        min(int(equity / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)
        if equity >= MIN_BUCKET_SIZE else 0
    )
    capacity_slots = max(0, max_positions - len(positions))
    deployable_cash = settled_cash * min(max(float(SETTLED_CASH_DEPLOYMENT_PCT), 0.0), 1.0)
    cash_slots = (
        int(deployable_cash / MIN_BUCKET_SIZE)
        if deployable_cash >= MIN_BUCKET_SIZE else 0
    )
    entry_slots = min(capacity_slots, cash_slots)
    bucket_size = round(deployable_cash / entry_slots, 2) if entry_slots > 0 else 0.0

    return JSONResponse({
        "equity":            equity,
        "mkt_value":         round(position_value, 2),
        "cash":              settled_cash,
        "deployable_cash":   round(deployable_cash, 2),
        "allocation_pct":    round((position_value / equity * 100) if equity else 0, 1),
        "bucket_size":       bucket_size,
        "position_count":    len(positions),
        "max_positions":     max_positions,
        "positions":         positions,
        "total_unrealized":  round(total_unrealized, 2),
        "pnl":               _pnl(equity),
        "connected":         bool(dash_data.get("connected", False)),
        "market_open":       _market_open(),
        "vix":               dash_data.get("vix"),
        "vix_threshold":     VIX_THRESHOLD,
        "hold_trading_bars": profile.time_stop_bars,
        "eod_exit_time":     f"{EOD_EXIT_TIME[0]:02d}:{EOD_EXIT_TIME[1]:02d}",
        "strategy_profile":  profile.name,
        "strategy":          {
            "profile":                       profile.name,
            "label":                         profile.label,
            "description":                   profile.description,
            "scoring_model":                 scoring_model,
            "min_rs_63d":                    profile.min_rs_63d,
            "min_rs_126d":                   profile.min_rs_126d,
            "min_13w_return":                profile.min_13w_return,
            "min_26w_return":                profile.min_26w_return,
            "min_price_vs_52w_high":         profile.min_price_vs_52w_high,
            "min_volume_pace":               profile.min_volume_pace,
            "max_ma20_extension":            profile.max_ma20_extension,
            "min_score":                     profile.min_score,
            "eod_quality_cleanup":           profile.eod_quality_cleanup,
            "friday_close_enabled":          profile.friday_close_enabled,
            "time_stop_bars":                profile.time_stop_bars,
            "allow_bear_phase_entries":      profile.allow_bear_phase_entries,
            "indicator_sleeves":             list(getattr(profile, "indicator_sleeves", ()) or ()),
            "max_atr_pct":                   profile.max_atr_pct,
            "max_spread_pct":                profile.max_spread_pct,
            "analyst_ratings_enabled":       ANALYST_RATINGS_ENABLED,
            "analyst_ratings_free_source":   ANALYST_RATINGS_FREE_SOURCE,
            "analyst_rating_score_weight":   ANALYST_RATING_SCORE_WEIGHT,
            "analyst_rating_sell_threshold": ANALYST_RATING_SELL_THRESHOLD,
            "analyst_rating_exit_enabled":   ANALYST_RATING_EXIT_ENABLED,
        },
        "last_scan":         dash_data.get("last_scan"),
        "next_scan":         dash_data.get("next_scan"),
        "scanner_source":    dash_data.get("scanner_source"),
        "scanner_universe_size": dash_data.get("scanner_universe_size"),
        "scanner_universe_offset": dash_data.get("scanner_universe_offset"),
        "scanner_universe_batch_size": dash_data.get("scanner_universe_batch_size"),
        "scanner_prefilter_date": dash_data.get("scanner_prefilter_date"),
        "scanner_prefilter_status": dash_data.get("scanner_prefilter_status"),
        "scanner_prefilter_candidates": dash_data.get("scanner_prefilter_candidates"),
        "scanner_prefilter_stats": dash_data.get("scanner_prefilter_stats", {}),
        "last_updated":      dash_data.get("last_updated"),
        "blocked_today":     dash_data.get("blocked_today", []),
    })


@app.get("/api/equity_history")
def get_equity_history():
    return JSONResponse(_read_history())


@app.get("/api/logs")
def get_logs(n: int = 100):
    """Return the last n lines from the trading engine log file."""
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
        return JSONResponse({"lines": [ln.rstrip() for ln in lines[-n:]]})
    except FileNotFoundError:
        return JSONResponse({"lines": [], "error": "Log file not found"})
    except OSError as e:
        return JSONResponse({"lines": [], "error": str(e)})


# ── Dashboard HTML ────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>⚡ Velocity Engine</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg:        #07090f;
  --bg2:       #0d1220;
  --bg3:       #111827;
  --bg4:       #162035;
  --border:    #1c2d45;
  --border2:   #243d5c;
  --text:      #d4dde8;
  --dim:       #4e6070;
  --green:     #00d68f;
  --green-bg:  rgba(0,214,143,.08);
  --red:       #ff4d6d;
  --red-bg:    rgba(255,77,109,.08);
  --yellow:    #ffc530;
  --yellow-bg: rgba(255,197,48,.08);
  --cyan:      #00b4d8;
  --blue:      #4361ee;
  --purple:    #7b5ea7;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{
  background:var(--bg);color:var(--text);
  font-family:'Cascadia Code','Fira Code','Courier New',monospace;
  font-size:13px;line-height:1.6;padding:14px;
  background-image: radial-gradient(ellipse at top, #0d1a2e 0%, #07090f 70%);
}

/* ── TOPBAR ── */
#topbar{
  position:fixed;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--blue),var(--cyan),var(--green));
  z-index:999;opacity:.7;
}
#progress{height:100%;width:0%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.8));
  transition:width .1s ease;}

/* ── HEADER ── */
.header{
  border:1px solid var(--border2);border-radius:10px;
  background:linear-gradient(135deg,#0d1a2e,#111827 60%,#0d1a2e);
  padding:18px 24px;margin-bottom:14px;text-align:center;
  position:relative;overflow:hidden;
}
.header::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent 0%,var(--blue) 25%,var(--cyan) 50%,var(--green) 75%,transparent 100%);
}
.header::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--border2),transparent);
}
.header h1{
  font-size:17px;font-weight:700;letter-spacing:5px;
  background:linear-gradient(90deg,var(--cyan),var(--green));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}
.header .sub{font-size:10px;color:var(--dim);letter-spacing:4px;margin-top:5px;}
.header .badge{
  display:inline-block;font-size:9px;letter-spacing:2px;
  padding:2px 8px;border-radius:3px;margin-top:6px;
  border:1px solid var(--border2);color:var(--dim);
}

/* ── PANELS ── */
.panel{
  background:var(--bg3);border:1px solid var(--border);
  border-radius:8px;padding:16px 18px;margin-bottom:14px;
}
.ptitle{
  font-size:10px;font-weight:700;letter-spacing:3px;
  padding-bottom:10px;margin-bottom:12px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:8px;
}
.ptitle .icon{font-size:14px;}

/* ── ENTRY / EXIT CONDITIONS ── */
.entry-title{color:var(--green);}
.exit-title{color:var(--red);}
.cond-grid{display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;}
@media(max-width:900px){.cond-grid{grid-template-columns:1fr;}}
.cond{display:flex;align-items:baseline;gap:0;padding:6px 8px;border-radius:5px;transition:background .15s;}
.cond:hover{background:var(--bg4);}
.cn{color:var(--yellow);font-weight:700;font-size:10px;min-width:26px;opacity:.8;}
.cname{font-size:11px;font-weight:600;min-width:160px;padding-right:10px;}
.cname.en{color:var(--green);}
.cname.ex{color:var(--red);}
.cdesc{color:#7a92a8;font-size:11px;}

/* ── MIDDLE ROW ── */
.mid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;}
@media(max-width:760px){.mid{grid-template-columns:1fr;}}

/* ── CAPITAL CARDS ── */
.cap-title{color:var(--green);}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.card{
  background:var(--bg4);border:1px solid var(--border);
  border-radius:7px;padding:10px 14px;transition:border-color .2s;
}
.card:hover{border-color:var(--border2);}
.card.wide{grid-column:1/-1;}
.clabel{font-size:9px;color:var(--dim);letter-spacing:2px;margin-bottom:4px;}
.cval{font-size:22px;font-weight:700;color:var(--green);line-height:1.2;}
.cval.c2{color:var(--cyan);}
.cval.cy{color:var(--yellow);}
.cval.sm{font-size:16px;}
.card-row{display:flex;justify-content:space-between;align-items:center;gap:12px;}

/* ── STATUS ── */
.stat-title{color:var(--cyan);}
.slist{display:flex;flex-direction:column;gap:7px;}
.srow{
  display:flex;justify-content:space-between;align-items:center;
  padding:7px 12px;background:var(--bg4);border:1px solid var(--border);
  border-radius:6px;
}
.slabel{font-size:10px;color:var(--dim);letter-spacing:1px;}
.sval{font-weight:700;font-size:13px;}
.g{color:var(--green);} .r{color:var(--red);} .y{color:var(--yellow);} .c{color:var(--cyan);} .d{color:var(--dim);}
.dot{
  display:inline-block;width:7px;height:7px;
  border-radius:50%;margin-right:6px;vertical-align:middle;
}
.dg{background:var(--green);box-shadow:0 0 6px var(--green);animation:pulse 1.8s infinite;}
.dr{background:var(--red);}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.3;}}

/* ── P&L PANEL ── */
.pnl-title{color:var(--yellow);}
.pnl-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;}
@media(max-width:900px){.pnl-grid{grid-template-columns:repeat(3,1fr);}}
@media(max-width:560px){.pnl-grid{grid-template-columns:1fr 1fr;}}
@media(max-width:380px){.pnl-grid{grid-template-columns:1fr;}}
.pnl-card{
  background:var(--bg4);border:1px solid var(--border);
  border-radius:8px;padding:14px 16px;text-align:center;
  transition:border-color .2s;
}
.pnl-card:hover{border-color:var(--border2);}
.pnl-label{font-size:9px;color:var(--dim);letter-spacing:3px;margin-bottom:8px;}
.pnl-amt{font-size:20px;font-weight:700;line-height:1.2;}
.pnl-pct{font-size:11px;margin-top:4px;opacity:.85;}
.pnl-pos{color:var(--green);}
.pnl-neg{color:var(--red);}
.pnl-neu{color:var(--dim);}

/* ── PORTFOLIO TABLE ── */
.port-title{color:#9b7fe8;}
.tbl-wrap{overflow-x:hidden;border-radius:6px;}
table{width:100%;border-collapse:collapse;font-size:12px;}
thead tr{background:var(--bg4);}
th{
  padding:8px 8px;text-align:right;font-size:9px;
  letter-spacing:1px;color:var(--dim);font-weight:600;
  border-bottom:2px solid var(--border2);white-space:nowrap;
}
th:first-child{text-align:center;}
tbody tr{border-bottom:1px solid var(--border);transition:background .12s;}
tbody tr:hover{background:var(--bg4);}
tbody tr:last-child{border-bottom:none;}
td{padding:8px 8px;text-align:right;white-space:nowrap;}
td:first-child{text-align:center;font-weight:700;color:var(--cyan);font-size:13px;}
.sl{color:var(--red);font-weight:600;}
.hw{color:var(--yellow);font-weight:600;}
.hn{color:var(--dim);}
.up{color:var(--green);font-weight:600;}
.un{color:var(--red);font-weight:600;}
.uz{color:var(--dim);}
.empty td{text-align:center;color:var(--dim);font-style:italic;padding:28px;font-size:12px;}

/* ── EQUITY CHART ── */
.chart-title{color:var(--cyan);}
.chart-wrap{position:relative;height:180px;}

/* ── FOOTER ── */
footer{
  text-align:center;color:var(--dim);font-size:9px;
  letter-spacing:2px;padding:10px 0 4px;
}
footer a{color:var(--dim);text-decoration:none;}
</style>
</head>
<body>

<div id="topbar"><div id="progress"></div></div>

<!-- HEADER -->
<div class="header">
  <h1>⚡ &nbsp; V E L O C I T Y &nbsp; E N G I N E &nbsp; · &nbsp; L I V E &nbsp; T R A D I N G &nbsp; D A S H B O A R D &nbsp; ⚡</h1>
  <div class="sub">INTERACTIVE BROKERS &nbsp;·&nbsp; RELATIVE-STRENGTH SWING MOMENTUM &nbsp;·&nbsp; REAL-TIME</div>
  <div class="badge"><span id="strategy-badge">__PROFILE_LABEL__ &nbsp;·&nbsp; __SCORING_MODEL__ score</span> &nbsp;·&nbsp; auto-refresh 5 s</div>
</div>

<!-- MIDDLE ROW -->
<div class="mid">

  <!-- CAPITAL -->
  <div class="panel">
    <div class="ptitle cap-title"><span class="icon">💰</span> CAPITAL &amp; SIZING</div>
    <div class="cards">
      <div class="card">
        <div class="clabel">TOTAL EQUITY</div>
        <div class="cval" id="equity">—</div>
      </div>
      <div class="card">
        <div class="clabel">CASH AVAILABLE</div>
        <div class="cval" id="cash">—</div>
      </div>
      <div class="card">
        <div class="clabel">MKT VALUE</div>
        <div class="cval c2" id="mkt-value">—</div>
      </div>
      <div class="card">
        <div class="clabel">BUCKET SIZE</div>
        <div class="cval c2 sm" id="bucket">—</div>
      </div>
      <div class="card wide">
        <div class="clabel">ALLOCATION &nbsp;/&nbsp; OPEN POSITIONS</div>
        <div class="card-row">
          <div class="cval cy sm" id="alloc">—</div>
          <div class="cval c2 sm" id="poscount">— / 3</div>
        </div>
      </div>
    </div>
  </div>

  <!-- STATUS -->
  <div class="panel">
    <div class="ptitle stat-title"><span class="icon">📡</span> MARKET STATUS</div>
    <div class="slist">
      <div class="srow"><span class="slabel">IB GATEWAY</span>  <span class="sval" id="gw">—</span></div>
      <div class="srow"><span class="slabel">MARKET</span>       <span class="sval" id="mkt">—</span></div>
      <div class="srow"><span class="slabel">TIME&nbsp;(ET)</span>  <span class="sval c" id="clock">—</span></div>
      <div class="srow"><span class="slabel">VIX</span>          <span class="sval" id="vix">—</span></div>
      <div class="srow"><span class="slabel">STRATEGY</span>     <span class="sval c" id="strategy">—</span></div>
      <div class="srow"><span class="slabel">ANALYST DATA</span> <span class="sval" id="analyst-state">—</span></div>
      <div class="srow"><span class="slabel">LAST&nbsp;SCAN</span>  <span class="sval d" id="lscan">—</span></div>
      <div class="srow"><span class="slabel">NEXT&nbsp;SCAN&nbsp;IN</span><span class="sval" id="nscan">—</span></div>
    </div>
  </div>

</div><!-- /mid -->

<!-- P&L SUMMARY -->
<div class="panel">
  <div class="ptitle pnl-title"><span class="icon">📈</span> PROFIT &amp; LOSS SUMMARY</div>
  <div class="pnl-grid">
    <div class="pnl-card">
      <div class="pnl-label">DAILY</div>
      <div class="pnl-amt pnl-neu" id="pnl-daily-amt">—</div>
      <div class="pnl-pct pnl-neu" id="pnl-daily-pct">—</div>
    </div>
    <div class="pnl-card">
      <div class="pnl-label">WEEKLY</div>
      <div class="pnl-amt pnl-neu" id="pnl-weekly-amt">—</div>
      <div class="pnl-pct pnl-neu" id="pnl-weekly-pct">—</div>
    </div>
    <div class="pnl-card">
      <div class="pnl-label">MONTHLY</div>
      <div class="pnl-amt pnl-neu" id="pnl-monthly-amt">—</div>
      <div class="pnl-pct pnl-neu" id="pnl-monthly-pct">—</div>
    </div>
    <div class="pnl-card">
      <div class="pnl-label">OVERALL</div>
      <div class="pnl-amt pnl-neu" id="pnl-overall-amt">—</div>
      <div class="pnl-pct pnl-neu" id="pnl-overall-pct">—</div>
    </div>
    <div class="pnl-card">
      <div class="pnl-label">UNREALIZED</div>
      <div class="pnl-amt pnl-neu" id="pnl-unreal-amt">—</div>
      <div class="pnl-pct pnl-neu" id="pnl-unreal-sub">open positions</div>
    </div>
  </div>
</div>

<!-- PORTFOLIO -->
<div class="panel">
  <div class="ptitle port-title"><span class="icon">📊</span> OPEN PORTFOLIO</div>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>SYMBOL</th>
          <th>ENTRY PRICE</th>
          <th>UNIT PRICE</th>
          <th>CURRENT PRICE</th>
          <th>QTY</th>
          <th>TOTAL COST</th>
          <th>UNREALIZED P&amp;L</th>
          <th>STOP (TRAIL)</th>
          <th>ANALYST SCORE</th>
          <th>ANALYST VOTES</th>
          <th>PROTECTION</th>
          <th>HOLD TIME</th>
        </tr>
      </thead>
      <tbody id="tbody">
        <tr class="empty"><td colspan="12">Waiting for data…</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- EQUITY CURVE -->
<div class="panel">
  <div class="ptitle chart-title"><span class="icon">📉</span> EQUITY CURVE &nbsp;—&nbsp; <span id="eq-window">60-DAY ROLLING</span></div>
  <div class="chart-wrap"><canvas id="eqChart"></canvas></div>
</div>

<!-- ENTRY CONDITIONS -->
<div class="panel">
  <div class="ptitle entry-title"><span class="icon">✅</span> ENTRY RULES &nbsp;—&nbsp; SETUP GATES MUST PASS BEFORE SCORE, RANK, AND LIVE RECHECK</div>
  <div class="cond-grid" id="entry-conds"></div>
</div>

<!-- EXIT CONDITIONS -->
<div class="panel">
  <div class="ptitle exit-title"><span class="icon">🚪</span> EXIT / RISK CONTROLS &nbsp;—&nbsp; TRUE EXITS CAN CLOSE; HALTS BLOCK FRESH BUYING</div>
  <div class="cond-grid" id="exit-conds"></div>
</div>

<footer>
  VELOCITY ENGINE &nbsp;·&nbsp; LAST UPDATED: <span id="lu">—</span>
  &nbsp;·&nbsp; <a href="/api/state" target="_blank">API JSON</a>
</footer>

<script>
// ── Entry / Exit conditions ─────────────────────────────────────────────────
const ENTRY_CONDITIONS = [
  ["1",  "RS Trend Gate",     "en", "Weekly uptrend required; RS 63d ≥ __SWING_RS_63D__, RS 126d ≥ __SWING_RS_126D__, 13w return ≥ __SWING_RET_13W__, 26w return ≥ __SWING_RET_26W__, price ≥ __SWING_52W__ of 52-week high; MA50 > MA200, rising SMA200"],
  ["2",  "MA Timing",         "en", "Default sleeve: EMA20 > SMA50 trend required; entry timed by fresh EMA/SMA cross, MA20/MA50 reclaim, or prior-high break; price must not exceed __SWING_MA20_EXT__ above MA20"],
  ["3",  "Bollinger Sleeve",  "en", "Optional sleeve: Bollinger lower-band reclaim signals a mean-reversion entry; enabled only when VELOCITY_INDICATOR_SWING_STRATEGIES includes bollinger_reversion"],
  ["4",  "PSAR Sleeve",       "en", "Optional sleeve: three-dot PSAR bull state can time entry and adds a confirmation point; enabled only when VELOCITY_INDICATOR_SWING_STRATEGIES includes psar_flip"],
  ["5",  "Momentum Confirm",  "en", "RSI ≥ 50 (or recovering from oversold for Bollinger sleeve); at least 2 confirmations must agree from: MACD delta, OBV uptrend, PSAR, stochastic bull-exit, volume pace"],
  ["6",  "Volume Confirm",    "en", "Volume pace must be ≥ __SWING_VOL_PACE__ of average; OBV trend and dollar volume ≥ $75M/day; weak volume overrides strong price signals"],
  ["7",  "Analyst Weight",    "en", "Analyst consensus adjusts score by up to ±__ANALYST_WEIGHT__ pts; cannot create a buy on its own — all structural gates must pass first"],
  ["8",  "Risk / Liquidity",  "en", "ATR% ≤ 12%, bid/ask spread ≤ 1%, price ≥ $10, volume ≥ 1M shares, 20-day dollar volume ≥ $75M; any failure skips the candidate"],
  ["9",  "Market Regime",     "en", "VIX must be available and ≤ threshold; SPY must be above SMA50 and SMA200 with rising SMA200; bear regime blocks all fresh entries"],
  ["10", "Portfolio Fit",     "en", "Settled-cash bucket required; max 2 names per sector; daily-return correlation ≤ 0.70 vs. open positions; no duplicate symbols"],
  ["11", "Score Threshold",   "en", "Candidate must score ≥ __ACTIVE_MIN_SCORE__ on the indicator-swing model after all gate checks pass"],
  ["12", "Ranked Execution",  "en", "Passing candidates ranked highest-score-first; each rechecked at live price immediately before order placement"],
];
const EXIT_CONDITIONS = [
  ["1", "Percent Trail", "ex", "TRAIL SELL at __TRAIL_PCT__% below peak price (IBKR broker-side); stop ratchets up as price climbs — never down"],
  ["2", "Hard Stop",        "ex", "Software exit: 7% drawdown from fill price triggers immediate Market SELL regardless of ATR distance"],
  ["3", "Strategy Exit",    "ex", "Positions exit on the matching sleeve rule that opened them: MA bearish cross for the default profile, or the standalone research profile's own reversal rule"],
  ["4", "Swing Time Stop",  "ex", "__SWING_TIME_STOP_RULE__"],
  ["5", "Analyst Downgrade","ex", "__ANALYST_EXIT_RULE__"],
  ["6", "No EOD Churn",     "ex", "__EOD_PROFIT_CLEANUP_RULE__"],
  ["7", "Entry Halts",      "ex", "VIX risk-off, SPY bear regime, and 3% daily equity drawdown halt fresh swing buys; open positions still exit through stops"],
  ["8", "Manual Controls",  "ex", "HALT_TRADING blocks new entries; FORCE_EXIT_ALL requests a full market-exit pass"],
];
function renderConds(arr, containerId) {
  document.getElementById(containerId).innerHTML = arr.map(([n,name,cls,desc]) =>
    `<div class="cond">
      <span class="cn">${n}.</span>
      <span class="cname ${cls}">${name}</span>
      <span class="cdesc">${desc}</span>
    </div>`
  ).join('');
}
renderConds(ENTRY_CONDITIONS, 'entry-conds');
renderConds(EXIT_CONDITIONS,  'exit-conds');

// ── Live clock ──────────────────────────────────────────────────────────────
function tick() {
  const t = new Date().toLocaleTimeString('en-US',
    {hour:'2-digit',minute:'2-digit',second:'2-digit',
     hour12:false, timeZone:'America/New_York'});
  document.getElementById('clock').textContent = t + ' ET';
}
setInterval(tick, 1000); tick();

// ── Countdown ───────────────────────────────────────────────────────────────
let nextMs = null;
function countdown() {
  if (!nextMs) return;
  const s = Math.max(0, Math.floor((nextMs - Date.now()) / 1000));
  const m = String(Math.floor(s/60)).padStart(2,'0');
  const sc = String(s%60).padStart(2,'0');
  const el = document.getElementById('nscan');
  el.textContent = `${m}:${sc}`;
  el.className = 'sval ' + (s < 60 ? 'r' : s < 180 ? 'y' : 'g');
}
setInterval(countdown, 1000);

// ── Formatters ───────────────────────────────────────────────────────────────
const $f = v => '$' + (+v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const vol = v => v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?(v/1e3).toFixed(0)+'K':String(v|0);
const pct = (v, digits=1, signed=false) => {
  if (v == null || !isFinite(+v)) return '—';
  const val = +v * 100;
  return (signed && val > 0 ? '+' : '') + val.toFixed(digits) + '%';
};
const holdLabel = h => {
  if (h == null || !isFinite(+h)) return '—';
  const hours = +h;
  return hours >= 24 ? (hours / 24).toFixed(1) + 'd' : hours.toFixed(1) + 'h';
};
const cleanLabel = v => (v || '—').toString().replace(/_/g, ' ').toUpperCase();

// ── Progress bar ─────────────────────────────────────────────────────────────
function flash() {
  const p = document.getElementById('progress');
  p.style.transition = 'none'; p.style.width = '0%';
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      p.style.transition = 'width 4.8s linear';
      p.style.width = '100%';
    });
  });
}

// ── Render ───────────────────────────────────────────────────────────────────
function render(d) {
  // Capital
  document.getElementById('equity').textContent    = $f(d.equity||0);
  document.getElementById('cash').textContent      = $f(d.cash||0);
  document.getElementById('mkt-value').textContent = $f(d.mkt_value||0);
  document.getElementById('bucket').textContent    = $f(d.bucket_size||0);
  document.getElementById('alloc').textContent    = (d.allocation_pct||0).toFixed(1)+'%';
  document.getElementById('poscount').textContent = `${d.position_count??0} / ${d.max_positions??0}`;

  const strat = d.strategy || {};
  const strategyBadge = document.getElementById('strategy-badge');
  const scoring = (strat.scoring_model || 'indicator_swing').toUpperCase();
  const label = strat.label || 'Multi-Indicator Swing';
  if (strategyBadge) strategyBadge.textContent = `${label} · ${scoring} score`;
  document.getElementById('strategy').textContent = label;
  const analystState = document.getElementById('analyst-state');
  const analystOn = !!strat.analyst_ratings_enabled;
  const analystSource = (strat.analyst_ratings_free_source || 'configured').toString().toUpperCase();
  analystState.textContent = analystOn
    ? `ON · ${analystSource} · ±${(+strat.analyst_rating_score_weight || 0).toFixed(0)} pts`
    : 'OFF';
  analystState.className = 'sval ' + (analystOn ? 'g' : 'd');

  // Gateway
  const gw = document.getElementById('gw');
  gw.innerHTML = d.connected
    ? '<span class="dot dg"></span><span class="g">CONNECTED</span>'
    : '<span class="dot dr"></span><span class="r">DISCONNECTED</span>';

  // Market
  const mk = document.getElementById('mkt');
  mk.textContent = d.market_open ? 'OPEN' : 'CLOSED';
  mk.className   = 'sval ' + (d.market_open ? 'g' : 'r');

  // VIX
  const ve = document.getElementById('vix');
  if (d.vix != null) {
    ve.textContent = (+d.vix).toFixed(2);
    const vixThreshold = d.vix_threshold ?? 35;
    ve.className   = 'sval ' + (d.vix>vixThreshold?'r':d.vix>(vixThreshold*0.714)?'y':'g');
  } else { ve.textContent='—'; ve.className='sval d'; }

  // Scan times
  document.getElementById('lscan').textContent = d.last_scan || '—';
  if (d.next_scan) {
    nextMs = new Date(d.next_scan).getTime();
    countdown();
  }

  // Last updated
  if (d.last_updated) {
    document.getElementById('lu').textContent =
      new Date(d.last_updated).toLocaleTimeString('en-US',{timeZone:'America/New_York'});
  }

  // P&L
  function renderPnl(period, data) {
    const amt = document.getElementById(`pnl-${period}-amt`);
    const pct = document.getElementById(`pnl-${period}-pct`);
    if (!data || data.amount == null) {
      amt.textContent = '—'; amt.className = 'pnl-amt pnl-neu';
      pct.textContent = '—'; pct.className = 'pnl-pct pnl-neu';
      return;
    }
    const pos = data.amount >= 0;
    const cls = pos ? 'pnl-pos' : 'pnl-neg';
    const sign = pos ? '+' : '';
    amt.textContent = sign + $f(data.amount);
    amt.className   = `pnl-amt ${cls}`;
    pct.textContent = sign + data.pct.toFixed(2) + '%';
    pct.className   = `pnl-pct ${cls}`;
  }
  if (d.pnl) {
    renderPnl('daily',   d.pnl.daily);
    renderPnl('weekly',  d.pnl.weekly);
    renderPnl('monthly', d.pnl.monthly);
    renderPnl('overall', d.pnl.overall);
  }
  // Unrealized P&L card
  const ua = document.getElementById('pnl-unreal-amt');
  const us = document.getElementById('pnl-unreal-sub');
  const tu = d.total_unrealized ?? null;
  if (tu === null || d.position_count === 0) {
    ua.textContent = '—'; ua.className = 'pnl-amt pnl-neu';
    us.textContent = 'no open positions'; us.className = 'pnl-pct pnl-neu';
  } else {
    const pos = tu >= 0;
    ua.textContent = (pos?'+':'') + $f(tu);
    ua.className   = 'pnl-amt ' + (pos ? 'pnl-pos' : 'pnl-neg');
    us.textContent = d.position_count + ' position' + (d.position_count>1?'s':'');
    us.className   = 'pnl-pct ' + (pos ? 'pnl-pos' : 'pnl-neg');
  }

  // Portfolio
  const tb = document.getElementById('tbody');
  if (!d.positions || d.positions.length === 0) {
    tb.innerHTML = '<tr class="empty"><td colspan="12">No open positions</td></tr>';
    return;
  }
  tb.innerHTML = d.positions.map(p => {
    const unr   = p.unrealized ?? 0;
    const unrP  = p.unrealized_pct ?? 0;
    const ucls  = unr > 0 ? 'up' : unr < 0 ? 'un' : 'uz';
    const usign = unr >= 0 ? '+' : '';
    const ar    = p.analyst_rating_score;
    const arCls = ar == null ? 'd' : ar > 0.15 ? 'g' : ar < -0.15 ? 'r' : 'y';
    const arTxt = ar == null
      ? '—'
      : `${ar > 0 ? '+' : ''}${(+ar).toFixed(2)}${p.analyst_rating_total != null ? '/' + p.analyst_rating_total : ''}`;
    const buyVotes = (p.analyst_rating_strong_buy ?? null) == null && (p.analyst_rating_buy ?? null) == null
      ? null
      : (+p.analyst_rating_strong_buy || 0) + (+p.analyst_rating_buy || 0);
    const holdVotes = p.analyst_rating_hold == null ? null : (+p.analyst_rating_hold || 0);
    const sellVotes = (p.analyst_rating_sell ?? null) == null && (p.analyst_rating_strong_sell ?? null) == null
      ? null
      : (+p.analyst_rating_sell || 0) + (+p.analyst_rating_strong_sell || 0);
    const votesTxt = buyVotes == null && holdVotes == null && sellVotes == null
      ? '—'
      : `B:${buyVotes ?? 0} H:${holdVotes ?? 0} S:${sellVotes ?? 0}`;
    const prot = cleanLabel(p.protection_status || 'unknown');
    const protCls = p.protection_status === 'confirmed' ? 'g' : p.protection_status === 'pending' ? 'y' : 'd';
    return `<tr>
      <td>${p.symbol}</td>
      <td>${$f(p.entry_price)}</td>
      <td class="c" style="font-size:11px">${p.unit_price != null ? $f(p.unit_price) : '<span style="color:var(--dim)">pending</span>'}</td>
      <td>${$f(p.current_price)}</td>
      <td>${Math.round(+p.qty)}</td>
      <td>${$f(p.total_amount)}</td>
      <td class="${ucls}">${usign}${$f(unr)}<br><span style="font-size:10px;opacity:.8">${usign}${unrP.toFixed(2)}%</span></td>
      <td class="sl">${$f(p.effective_stop ?? p.stop_loss)}${p.effective_stop > p.stop_loss ? ' ↑' : ''}</td>
      <td class="${arCls}">${arTxt}</td>
      <td class="${arCls}" style="font-size:10px">${votesTxt}</td>
      <td class="${protCls}" style="font-size:10px">${prot}</td>
      <td class="hn">${holdLabel(p.hold_hours)}</td>
    </tr>`;
  }).join('');
}

// ── API auth ─────────────────────────────────────────────────────────────────
const params = new URLSearchParams(window.location.search);
const urlToken = params.get('token') || '';
if (urlToken) localStorage.setItem('velocity_dashboard_token', urlToken);
const storedToken = localStorage.getItem('velocity_dashboard_token') || '';
function api(path) {
  return storedToken ? path + '?token=' + encodeURIComponent(storedToken) : path;
}

// ── Equity chart ─────────────────────────────────────────────────────────────
let eqChart = null;
async function refreshChart() {
  try {
    const r = await fetch(api('/api/equity_history'));
    if (!r.ok) return;
    const hist = await r.json();
    if (!hist || hist.length === 0) return;
    const dates = hist.map(e => new Date(e.ts));
    const etDay = d => d.toLocaleDateString('en-CA', {timeZone:'America/New_York'});
    const intraday = new Set(dates.map(etDay)).size <= 1;
    const windowLabel = document.getElementById('eq-window');
    if (windowLabel) windowLabel.textContent = intraday ? 'INTRADAY TODAY' : '60-DAY ROLLING';
    const labels = hist.map(e => {
      const d = new Date(e.ts);
      return intraday
        ? d.toLocaleTimeString('en-US', {hour:'2-digit', minute:'2-digit', hour12:false, timeZone:'America/New_York'})
        : d.toLocaleDateString('en-US', {month:'short', day:'numeric', timeZone:'America/New_York'});
    });
    const data = hist.map(e => e.eq);
    const baseline = data[0];
    const borderColor = data[data.length-1] >= baseline ? '#00d68f' : '#ff4d6d';
    const gradientColor = data[data.length-1] >= baseline
      ? 'rgba(0,214,143,0.15)' : 'rgba(255,77,109,0.15)';
    const ctx = document.getElementById('eqChart').getContext('2d');
    const grad = ctx.createLinearGradient(0, 0, 0, 180);
    grad.addColorStop(0, gradientColor);
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    if (eqChart) eqChart.destroy();
    eqChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          data,
          borderColor,
          backgroundColor: grad,
          borderWidth: 2,
          pointRadius: hist.length > 30 ? 0 : 3,
          tension: 0.3,
          fill: true,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: {
          callbacks: {
            title: items => {
              const e = hist[items[0].dataIndex];
              return new Date(e.ts).toLocaleString('en-US', {
                month:'short', day:'numeric', hour:'2-digit', minute:'2-digit',
                timeZone:'America/New_York'
              }) + ' ET';
            },
            label: c => ' $' + c.parsed.y.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})
          }
        }},
        scales: {
          x: { ticks: { color:'#4e6070', font:{size:10}, maxTicksLimit:10 }, grid:{ color:'#1c2d45' } },
          y: { ticks: { color:'#4e6070', font:{size:10},
                        callback: v => '$'+v.toLocaleString('en-US',{minimumFractionDigits:0}) },
               grid:{ color:'#1c2d45' } }
        }
      }
    });
  } catch(e) { /* chart fetch failed silently */ }
}
refreshChart();
setInterval(refreshChart, 60000);   // chart refreshes once per minute

// ── Fetch loop ────────────────────────────────────────────────────────────────
async function refresh() {
  flash();
  try {
    const r = await fetch(api('/api/state'));
    if (!r.ok) throw new Error(r.status);
    render(await r.json());
  } catch(e) {
    document.getElementById('gw').innerHTML =
      '<span class="dot dr"></span><span class="r">SERVER OFFLINE</span>';
  }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""

_ACTIVE_PROFILE = get_strategy_profile(STRATEGY_PROFILE)
_ACTIVE_SCORING_MODEL = _ACTIVE_PROFILE.scoring_model.upper()


def _pct_text(value: float, digits: int = 0, *, signed: bool = False) -> str:
    pct_value = float(value) * 100.0
    sign = "+" if signed and pct_value > 0 else ""
    return f"{sign}{pct_value:.{digits}f}%"


_eod_stop_text = (
    "the protective stop is confirmed"
    if EOD_HOLD_REQUIRE_STOP_CONFIRMED
    else "protective stop confirmation is not required"
)
_analyst_exit_rule = (
    f"Enabled: rating score <= {ANALYST_RATING_SELL_THRESHOLD:+.2f} can exit only when price confirms weakness "
    "(at/below entry, below MA20, or EMA/SMA failure)"
    if ANALYST_RATING_EXIT_ENABLED
    else "Disabled: analyst ratings adjust entry ranking only; downgrade exits are not active"
)
_eod_rule = (
    (
        f"Enabled for this profile: same-day at/after "
        f"{EOD_EXIT_TIME[0] % 12 or 12}:{EOD_EXIT_TIME[1]:02d} PM ET: carry only if "
        f"profit >= {EOD_HOLD_MIN_PROFIT_PCT * 100:.0f}%, price is above VWAP or entry, "
        f"close is in the top {(1 - EOD_HOLD_DAY_RANGE_LOCATION_MIN) * 100:.0f}% of the day range, "
        f"relative strength is >= {EOD_HOLD_RELATIVE_STRENGTH_MIN * 100:.0f}%, "
        f"and {_eod_stop_text}; otherwise Market SELL before the close"
    )
    if _ACTIVE_PROFILE.eod_quality_cleanup
    else "Disabled for the swing profile: normal positions are not churned out by same-day EOD quality cleanup"
)
_time_stop_rule = (
    f"After {_ACTIVE_PROFILE.time_stop_bars or SWING_TIME_STOP_BARS} trading bars, "
    "close positions that are not above breakeven"
    if _ACTIVE_PROFILE.time_stop_bars is not None
    else "Disabled for this profile"
)

_HTML = (
    _HTML
    .replace("__PROFILE_LABEL__", _ACTIVE_PROFILE.label)
    .replace("__SCORING_MODEL__", _ACTIVE_SCORING_MODEL)
    .replace("__ACTIVE_MIN_SCORE__", f"{(_ACTIVE_PROFILE.min_score or INDICATOR_SWING_MIN_SCORE):.0f}")
    .replace("__ANALYST_WEIGHT__", f"{ANALYST_RATING_SCORE_WEIGHT:.0f}")
    .replace("__EOD_PROFIT_CLEANUP_RULE__", _eod_rule)
    .replace("__SWING_TIME_STOP_RULE__", _time_stop_rule)
    .replace("__ANALYST_EXIT_RULE__", _analyst_exit_rule)
    .replace("__TRAIL_PCT__", f"{TRAIL_PCT * 100:.4g}")
    .replace("__SWING_RS_63D__", _pct_text(SWING_RS_MIN_63D, signed=True))
    .replace("__SWING_RS_126D__", _pct_text(SWING_RS_MIN_126D, signed=True))
    .replace("__SWING_RET_13W__", _pct_text(SWING_MIN_13W_RETURN))
    .replace("__SWING_RET_26W__", _pct_text(SWING_MIN_26W_RETURN))
    .replace("__SWING_52W__", _pct_text(SWING_MIN_PRICE_VS_52W_HIGH))
    .replace("__SWING_MA20_EXT__", _pct_text(SWING_MAX_MA20_EXTENSION))
    .replace("__SWING_VOL_PACE__", f"{SWING_MIN_VOLUME_PACE:.2f}x")
)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_HTML)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="VelocityEngine Web Dashboard")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port", default=8080, type=int, help="Port (default: 8080)")
    args = p.parse_args()

    print(f"\n  ⚡  VelocityEngine Dashboard")
    if args.host not in ("127.0.0.1", "localhost") and not DASHBOARD_TOKEN:
        raise SystemExit(
            "Refusing external dashboard bind without VELOCITY_DASHBOARD_TOKEN; "
            "this would expose trading state."
        )
    print(f"  Open → http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
