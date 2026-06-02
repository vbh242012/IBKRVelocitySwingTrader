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
import os
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
    VIX_THRESHOLD,
    HOLD_TRADING_BARS,
    PROFIT_MIN_THRESHOLD,
    VELOCITY_EXIT_TIME,
    DASHBOARD_ALLOWED_ORIGINS,
    DASHBOARD_TOKEN,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="VelocityEngine Dashboard", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=DASHBOARD_ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["Authorization"],
)


def _request_authorized(request: Request) -> bool:
    """Optional bearer/query-token auth for non-local dashboard deployments."""
    if not DASHBOARD_TOKEN:
        return True
    auth = request.headers.get("authorization", "")
    if auth == f"Bearer {DASHBOARD_TOKEN}":
        return True
    return request.query_params.get("token") == DASHBOARD_TOKEN


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


def _pnl(equity_now: float) -> dict:
    """Compute daily / weekly / monthly / overall P&L from equity history."""
    history = _read_history()
    tz_ny   = pytz.timezone('US/Eastern')
    now     = datetime.now(tz_ny)

    def _parse_ts(ts: str) -> datetime:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo is not None else tz_ny.localize(dt)

    def _find_base(days_ago: int) -> Optional[float]:
        cutoff = now - timedelta(days=days_ago)
        past   = [e for e in history if _parse_ts(e["ts"]) <= cutoff]
        if past:
            return float(past[-1]["eq"])
        # No snapshot older than the lookback — not enough history yet
        return None

    def _entry(base: Optional[float]) -> dict:
        if base is None or base == 0:
            return {"amount": None, "pct": None}
        amount = round(equity_now - base, 2)
        pct    = round(amount / base * 100, 2)
        return {"amount": amount, "pct": pct}

    # Overall: oldest snapshot in history (first real IBKR reading)
    overall_base = float(history[0]["eq"]) if history else None

    return {
        "daily":   _entry(_find_base(1)),
        "weekly":  _entry(_find_base(7)),
        "monthly": _entry(_find_base(30)),
        "overall": _entry(overall_base),
    }


def _market_open() -> bool:
    tz_ny  = pytz.timezone("US/Eastern")
    now_ny = datetime.now(tz_ny)
    if now_ny.weekday() >= 5:
        return False
    return (9, 30) <= (now_ny.hour, now_ny.minute) <= (16, 0)


# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/api/state")
def get_state():
    state     = _read_json(STATE_FILE)
    dash_data = _read_json(DASHBOARD_FILE)

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

    tz_ny = pytz.timezone('US/Eastern')
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
        entry_ts = d.get("time", now.isoformat())
        try:
            entry_dt = datetime.fromisoformat(entry_ts)
            if entry_dt.tzinfo is None:
                entry_dt = tz_ny.localize(entry_dt)
            hold_h = (now - entry_dt).total_seconds() / 3600
        except ValueError:
            hold_h = 0.0
        unreal     = round((cur - ep) * qty, 2)
        unreal_pct = round((cur - ep) / ep * 100, 2) if ep else 0.0
        total_unrealized += unreal
        positions.append({
            "symbol":          sym,
            "entry_price":     ep,
            "unit_price":      unit_price,
            "current_price":   cur,
            "qty":             qty,
            "total_amount":    round(ep * qty, 2),
            "unrealized":      unreal,
            "unrealized_pct":  unreal_pct,
            "stop_loss":       sl,
            "effective_stop":  effective_sl,
            "volume":          vol,
            "hold_hours":      round(hold_h, 2),
            "entry_time":      entry_ts,
            "score":           d.get("score"),
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
        "hold_trading_bars": HOLD_TRADING_BARS,
        "velocity_exit_time": f"{VELOCITY_EXIT_TIME[0]:02d}:{VELOCITY_EXIT_TIME[1]:02d}",
        "last_scan":         dash_data.get("last_scan"),
        "next_scan":         dash_data.get("next_scan"),
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
.tbl-wrap{overflow-x:auto;border-radius:6px;}
table{width:100%;border-collapse:collapse;font-size:12px;}
thead tr{background:var(--bg4);}
th{
  padding:10px 14px;text-align:right;font-size:9px;
  letter-spacing:2px;color:var(--dim);font-weight:600;
  border-bottom:2px solid var(--border2);white-space:nowrap;
}
th:first-child{text-align:center;}
tbody tr{border-bottom:1px solid var(--border);transition:background .12s;}
tbody tr:hover{background:var(--bg4);}
tbody tr:last-child{border-bottom:none;}
td{padding:11px 14px;text-align:right;white-space:nowrap;}
td:first-child{text-align:center;font-weight:700;color:var(--cyan);font-size:13px;}
.sl{color:var(--red);font-weight:600;}
.tp{color:var(--green);font-weight:600;}
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
  <div class="sub">INTERACTIVE BROKERS &nbsp;·&nbsp; MOMENTUM STRATEGY &nbsp;·&nbsp; REAL-TIME</div>
  <div class="badge">localhost:8080 &nbsp;·&nbsp; auto-refresh 5 s</div>
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
          <th>SCORE</th>
          <th>ENTRY PRICE</th>
          <th>UNIT PRICE</th>
          <th>CURRENT PRICE</th>
          <th>QTY</th>
          <th>TOTAL COST</th>
          <th>UNREALIZED P&amp;L</th>
          <th>STOP (TRAIL)</th>
          <th>VOLUME</th>
          <th>HOLD TIME</th>
        </tr>
      </thead>
      <tbody id="tbody">
        <tr class="empty"><td colspan="11">Waiting for data…</td></tr>
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
  <div class="ptitle entry-title"><span class="icon">✅</span> ENTRY CONDITIONS &nbsp;—&nbsp; ALL MUST BE MET SIMULTANEOUSLY FOR A BUY SIGNAL</div>
  <div class="cond-grid" id="entry-conds"></div>
</div>

<!-- EXIT CONDITIONS -->
<div class="panel">
  <div class="ptitle exit-title"><span class="icon">🚪</span> EXIT CONDITIONS &nbsp;—&nbsp; ANY ONE TRIGGERS POSITION CLOSE</div>
  <div class="cond-grid" id="exit-conds"></div>
</div>

<footer>
  VELOCITY ENGINE &nbsp;·&nbsp; LAST UPDATED: <span id="lu">—</span>
  &nbsp;·&nbsp; <a href="/api/state" target="_blank">API JSON</a>
</footer>

<script>
// ── Entry / Exit conditions ─────────────────────────────────────────────────
const ENTRY_CONDITIONS = [
  ["1",  "ORB Breakout",    "en", "Live price above the 15-min Opening Range High established by 09:45 ET"],
  ["2",  "RSI Acceleration","en", "RSI(14) must be above threshold and rising by the configured minimum delta"],
  ["3",  "Close Location",  "en", "Price must be closing in the upper part of today's range"],
  ["4",  "Universe Filter", "en", "IB scan: broad active corporate stocks | Price > $20 | Vol > 2M | Avg dollar volume ≥ threshold | Spread ≤ 0.5%"],
  ["5",  "SPY Regime",      "en", "SPY uptrend uses normal rules; SPY weakness requires stricter bear-phase rules and smaller risk"],
  ["6",  "Gap / ATR Risk",  "en", "Skip overextended breakouts and names with ATR too large relative to price"],
  ["7",  "VIX Filter",      "en", "Live or delayed VIX regime data required; VIX > 35 or missing VIX data blocks new entries (Risk-Off)"],
  ["8",  "Session Window",  "en", "Entries only 10:00–15:30 ET, Monday–Friday (30-min post-open buffer to avoid ORB noise)"],
  ["9",  "Position Limit",  "en", "Max position capacity compounds with total equity; new entries still require settled cash; max 2 in same sector; correlation ≤ 0.70 with any open position"],
  ["10", "Score Ranking",   "en", "All IBKR scanner results are evaluated; scored: Trend 30pts + Rel.Volume 25pts + Momentum 25pts + Liquidity 20pts = 100"],
  ["11", "Friday Filter",   "en", "Dollar-volume threshold doubled to 2× on Fridays for higher conviction"],
  ["12", "Day Strength",    "en", "Price must be in the upper part of today's range and at least 1.0% above today's open"],
  ["13", "Chandelier Stop", "en", "Chandelier trailing stop = ATR(22) × 2 from peak; bucket = settled cash ÷ open slots, recalculated every 60-sec cycle"],
];
const EXIT_CONDITIONS = [
  ["1", "Chandelier Trail", "ex", "TRAIL SELL at ATR(22)×2.0 from peak price; IB raises stop automatically as price climbs — only stop type used"],
  ["2", "Velocity Exit",    "ex", "__VELOCITY_EXIT_RULE__"],
  ["3", "Hard Stop",        "ex", "7% drawdown from fill price triggers immediate Market SELL regardless of ATR distance"],
  ["4", "Break-Even Floor", "ex", "Once profit exceeds 4%, chandelier stop floored at fill price — eliminates risk of turning a winner into a loser"],
  ["5", "Friday Close",     "ex", "After 3 PM ET on Fridays, positions with < 3% profit are liquidated to avoid weekend gap risk"],
  ["6", "VIX Risk-Off",     "ex", "VIX > 35 blocks new entries; existing positions exit via chandelier stop, velocity exit, or hard stop"],
  ["7", "Daily Loss Halt",  "ex", "3% intraday equity drawdown halts all new entries for the remainder of the trading day"],
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
  document.getElementById('poscount').textContent = `${d.position_count||0} / ${d.max_positions||3}`;

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
    tb.innerHTML = '<tr class="empty"><td colspan="11">No open positions</td></tr>';
    return;
  }
  tb.innerHTML = d.positions.map(p => {
    const warn  = p.hold_hours >= (d.hold_trading_bars ?? 2) * 24 * 0.875;
    const unr   = p.unrealized ?? 0;
    const unrP  = p.unrealized_pct ?? 0;
    const ucls  = unr > 0 ? 'up' : unr < 0 ? 'un' : 'uz';
    const usign = unr >= 0 ? '+' : '';
    const sc    = p.score != null ? p.score.toFixed(1) : '—';
    const scCls = p.score != null ? (p.score >= 70 ? 'g' : p.score >= 45 ? 'y' : 'r') : 'd';
    return `<tr>
      <td>${p.symbol}</td>
      <td class="${scCls}" style="font-weight:700">${sc}</td>
      <td>${$f(p.entry_price)}</td>
      <td class="c" style="font-size:11px">${p.unit_price != null ? $f(p.unit_price) : '<span style="color:var(--dim)">pending</span>'}</td>
      <td>${$f(p.current_price)}</td>
      <td>${(+p.qty).toFixed(4)}</td>
      <td>${$f(p.total_amount)}</td>
      <td class="${ucls}">${usign}${$f(unr)}<br><span style="font-size:10px;opacity:.8">${usign}${unrP.toFixed(2)}%</span></td>
      <td class="sl">${$f(p.effective_stop ?? p.stop_loss)}${p.effective_stop > p.stop_loss ? ' ↑' : ''}</td>
      <td>${vol(p.volume)}</td>
      <td class="${warn?'hw':'hn'}">${(+p.hold_hours).toFixed(1)}h${warn?' ⚠':''}</td>
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

_HTML = _HTML.replace(
    "__VELOCITY_EXIT_RULE__",
    (
        f"After {HOLD_TRADING_BARS} trading day(s), at/after "
        f"{VELOCITY_EXIT_TIME[0] % 12 or 12}:{VELOCITY_EXIT_TIME[1]:02d} PM ET: if profit < "
        f"{PROFIT_MIN_THRESHOLD * 100:.0f}%, force-liquidate via Market SELL; "
        "frees capital for T+1 settlement"
    ),
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
