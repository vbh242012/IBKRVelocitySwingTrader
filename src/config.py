import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR         = os.getenv(
    "VELOCITY_BASE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
)
STATE_FILE       = os.path.join(BASE_DIR, "engine_state.json")
DASHBOARD_FILE   = os.path.join(BASE_DIR, "dashboard_data.json")
EQUITY_HIST_FILE = os.path.join(BASE_DIR, "equity_history.json")
READINESS_FILE   = os.path.join(BASE_DIR, "readiness_snapshot.json")
HALT_FILE        = os.path.join(BASE_DIR, "HALT_TRADING")
FORCE_EXIT_FILE  = os.path.join(BASE_DIR, "FORCE_EXIT_ALL")
INSTANCE_LOCK_FILE = os.path.join(BASE_DIR, "velocity_engine.lock")
LOG_DIR          = os.path.join(BASE_DIR, "logs")
LOG_FILE         = os.path.join(LOG_DIR,  "trading_engine.log")

# ── IB Gateway ───────────────────────────────────────────────────────────────
TRADING_MODE  = os.getenv("VELOCITY_TRADING_MODE", "paper").strip().lower()
IB_HOST        = os.getenv("VELOCITY_IB_HOST", "127.0.0.1")
IB_PORT        = int(os.getenv("VELOCITY_IB_PORT", "4002" if TRADING_MODE == "paper" else "4001"))
IB_CLIENT_ID   = int(os.getenv("VELOCITY_IB_CLIENT_ID", "1"))
MARKET_DATA_TYPE = int(os.getenv("VELOCITY_MARKET_DATA_TYPE", "1"))  # stock entries require real-time market data type 1
VIX_MARKET_DATA_TYPE = int(os.getenv("VELOCITY_VIX_MARKET_DATA_TYPE", "3"))  # 3 = delayed; VIX is a regime filter only
ACCOUNT_CURRENCY = 'USD'
LIVE_TRADING_ACK_PHRASE = "I_UNDERSTAND_LIVE_RISK"
LIVE_TRADING_ACK = os.getenv("VELOCITY_LIVE_TRADING_ACK", "")
LIVE_IB_PORTS = {4001, 7496}
PAPER_IB_PORTS = {4002, 7497}

# Optional external launcher for IB Gateway / IBC.
# This app does not store broker credentials. Put credentials in an external
# IBC config or OS secret mechanism, then point VELOCITY_IB_GATEWAY_START_CMD
# at that launcher command. The command is parsed with shlex, not through a
# shell; wrap complex setup in a script.
IB_GATEWAY_AUTO_START = os.getenv("VELOCITY_IB_GATEWAY_AUTO_START", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
IB_GATEWAY_START_CMD = os.getenv("VELOCITY_IB_GATEWAY_START_CMD", "").strip()
IB_GATEWAY_START_TIMEOUT_SEC = float(os.getenv("VELOCITY_IB_GATEWAY_START_TIMEOUT_SEC", "180"))
IB_GATEWAY_START_POLL_SEC = float(os.getenv("VELOCITY_IB_GATEWAY_START_POLL_SEC", "2"))
IB_GATEWAY_STOP_ON_EXIT = os.getenv("VELOCITY_IB_GATEWAY_STOP_ON_EXIT", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
IB_GATEWAY_LOG_FILE = os.getenv(
    "VELOCITY_IB_GATEWAY_LOG_FILE",
    os.path.join(LOG_DIR, "ib_gateway_launcher.log"),
)

# ── Alerts ───────────────────────────────────────────────────────────────────
# Optional JSON webhook. Payload: {"severity": "...", "message": "..."}.
ALERT_WEBHOOK_URL = os.getenv("VELOCITY_ALERT_WEBHOOK_URL", "")
ALERT_TIMEOUT_SEC = float(os.getenv("VELOCITY_ALERT_TIMEOUT_SEC", "2.0"))

# ── Dashboard security ────────────────────────────────────────────────────────
# Keep empty for localhost-only use.  If binding the dashboard to any shared
# interface, set VELOCITY_DASHBOARD_TOKEN and open /?token=<token>.
DASHBOARD_TOKEN = os.getenv("VELOCITY_DASHBOARD_TOKEN", "")
DASHBOARD_ALLOWED_ORIGINS = [
    "http://127.0.0.1:8080",
    "http://localhost:8080",
]

# ── Capital, position sizing, and broker-derived values ───────────────────────
# Live trading never uses a fixed capital seed: NetLiquidation and SettledCash
# are fetched from IBKR accountSummary() at startup and every cycle.
# Backtests still need explicit assumptions because they run without IBKR.
BACKTEST_INITIAL_CAPITAL      = float(os.getenv("VELOCITY_BACKTEST_INITIAL_CAPITAL", "2000.0"))
BACKTEST_SCAN_COUNT           = int(os.getenv("VELOCITY_BACKTEST_SCAN_COUNT", "0"))  # 0 = all scanner-passed stocks
BACKTEST_MAX_SYMBOLS          = int(os.getenv("VELOCITY_BACKTEST_MAX_SYMBOLS", "0"))  # 0 = full filtered universe
BACKTEST_COMMISSION_PER_ORDER = float(os.getenv("VELOCITY_BACKTEST_COMMISSION_PER_ORDER", "1.00"))

# Maximum concurrent position capacity compounds with total account equity.
# New entries are still additionally constrained by SettledCash so the cash
# account never spends unsettled sale proceeds.  Only a configurable fraction
# of settled cash is deployed into buckets; the rest stays as a broker buffer.
MIN_BUCKET_SIZE      = float(os.getenv("VELOCITY_MIN_BUCKET_SIZE", "500.0"))
MAX_POSITIONS_CAP    = int(os.getenv("VELOCITY_MAX_POSITIONS_CAP", "20"))
SETTLED_CASH_DEPLOYMENT_PCT = float(os.getenv("VELOCITY_SETTLED_CASH_DEPLOYMENT_PCT", "0.95"))

# ── Chandelier Exit trailing stop ─────────────────────────────────────────────
CHANDELIER_PERIOD = 22     # lookback for ATR and highest-high (standard setting)
CHANDELIER_MULT   = 2.0    # ATR multiplier — kept after rule-combo sweep; 1.9 improved DD but gave up total return

# ── Risk rules ────────────────────────────────────────────────────────────────
VIX_THRESHOLD        = 35
HOLD_TRADING_BARS    = 1       # Mon-Fri trading session before velocity exit fires; weekend days are excluded
PROFIT_MIN_THRESHOLD = 0.05    # 5% min gain to avoid velocity exit (cross-validated: 0.05 > 0.03 on both 2023-24 and 2025-26)
GAP_MAX_PCT          = 0.10    # max allowed ORB extension; >10% = chasing, skip entry
MAX_DAILY_LOSS_PCT   = 0.03    # 3% intraday equity drawdown halts new entries for the day
RSI_MIN_DELTA        = 2.0     # minimum RSI point rise; full-universe sweep improved quality vs 1.0
DAY_RANGE_LOCATION_MIN = 0.55  # current price must close in upper 45% of range; improves DD/Sharpe after combo validation
INTRADAY_GAIN_MIN   = 0.010    # current price must be at least +1.0% above today's open; improves signal quality
ATR_PCT_MAX         = 0.07     # ATR_CHAND / price cap; filters excessively noisy names while preserving enough trade count
HARD_STOP_PCT        = 0.07    # 7% drawdown from entry triggers forced market exit regardless of ATR
RISK_PER_TRADE_PCT   = 0.02    # risk 2% of current equity per trade (ATR-based position sizing)
BREAK_EVEN_PCT       = 0.04    # once profit exceeds 4%, floor stop at entry — improves WR +4pp vs 3% threshold
FRIDAY_CLOSE_HOUR    = 15      # ET hour after which Friday positions are evaluated for early close
FRIDAY_MIN_PROFIT_PCT = 0.03   # Friday close: exit if profit < 3% to avoid carrying weekend gap risk
EOD_EXIT_TIME        = (15, 45)  # ET — daily flat: liquidate any position not in profit before close

# ── Bear-phase participation ──────────────────────────────────────────────────
# Broad-market bear tape must not be treated like normal risk.  These settings
# keep the engine active when SPY fails its regime check, but only for exceptional
# relative-strength breakouts and at reduced dollars-at-risk.  RVOL/VCP values
# remain available for diagnostics and ranking, but 8096 does not gate entries
# on them.
BEAR_PHASE_TRADING_ENABLED = os.getenv(
    "VELOCITY_BEAR_PHASE_TRADING_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
BEAR_PHASE_RISK_MULT       = 0.35   # 2% base risk → 0.7% risk in hostile tape
BEAR_PHASE_DOLLAR_VOL_MULT = 1.50   # require deeper liquidity in bear tape
BEAR_RVOL_MIN              = 4.0    # legacy/ranking reference; not an active 8096 entry gate
BEAR_BACKTEST_RVOL_MIN     = 2.0    # legacy daily-volume proxy; not an active 8096 entry gate
BEAR_VCP_RATIO             = 0.80   # legacy/diagnostic reference; not an active 8096 entry gate
BEAR_BREAKOUT_PCT          = 0.02   # legacy optimizer parameter; 10-day-high proximity is not an active entry gate
BEAR_RSI_THRESHOLD         = 65     # stronger momentum floor
BEAR_RSI_MIN_DELTA         = 3.0    # clearer RSI acceleration
BEAR_GAP_MAX_PCT           = 0.04   # less chasing when broad liquidity is poor

# ── Session timing ────────────────────────────────────────────────────────────
ENTRY_START          = (10, 0)   # first valid entry time — 30 min after market open
ENTRY_END            = (15, 30)
STOP_ACTIVATION_TIME = (9, 32)   # protective TRAIL stops activate after the first 2 opening minutes
VOL_MULT_FRIDAY      = 2.0   # Friday liquidity gate: 2× normal dollar-volume threshold
PRE_ENTRY_SYNC_TIME  = (9, 15)  # ET — position re-sync + stop audit before the entry window
POST_OPEN_AUDIT_TIME = (9, 35)  # ET — re-check protective stops shortly after open
PREMARKET_READINESS_TIME = (8, 45)  # ET — account/regime readiness snapshot before open
POST_CLOSE_MAINTENANCE_TIME = (17, 0)  # ET — post-close reconciliation and readiness snapshot
MARKET_CLOSE_TIME    = (16, 0)  # ET — stop software-managed market exits after regular session
ENTRY_PARENT_TIF     = 'DAY'   # Entry BUYs must not survive overnight if cancellation fails
ENTRY_ALL_OR_NONE    = True    # Avoid partial parent fills with full-size attached stop legs

# ── Indicators ────────────────────────────────────────────────────────────────
RSI_PERIOD    = 14
ATR_PERIOD    = 14
MA_FAST       = 50
MA_SLOW       = 200
RSI_THRESHOLD = 55

# ── Historical data requests ──────────────────────────────────────────────────
ORB_LOOKBACK   = '3600 S'   # 1-hour window ending at 9:45 AM → only the 9:30 bar falls inside
ORB_BAR_SIZE   = '15 mins'
DAILY_LOOKBACK = '1 Y'
DAILY_BAR_SIZE = '1 day'

# ── Scanner filters (shared by live engine and backtester) ───────────────────
SCAN_MIN_PRICE      = 20.0
SCAN_MIN_VOLUME     = 2_000_000
SCAN_MIN_MKTCAP     = 2_000_000_000
IB_SCANNER_SCAN_CODE = (
    os.getenv("VELOCITY_IB_SCANNER_SCAN_CODE", "MOST_ACTIVE").strip()
    or "MOST_ACTIVE"
)
IB_SCANNER_SCAN_CODES: list = [
    c.strip()
    for c in os.getenv(
        "VELOCITY_IB_SCANNER_SCAN_CODES",
        "MOST_ACTIVE,TOP_PERC_GAIN,HOT_BY_VOLUME",
    ).split(",")
    if c.strip()
] or ["MOST_ACTIVE"]
IB_SCANNER_LOCATION_CODE = (
    os.getenv("VELOCITY_IB_SCANNER_LOCATION_CODE", "STK.US.MAJOR").strip()
    or "STK.US.MAJOR"
)
IB_SCANNER_ROWS = int(os.getenv("VELOCITY_IB_SCANNER_ROWS", "-1"))  # -1 lets IBKR use its scanner default/maximum
# 20-day average dollar volume proxy for market cap — mirrors the IB $2B
# market-cap gate.  Using a rolling average (not the single day's value)
# prevents micro-cap news/pump spikes from passing the filter.
# $50M/day average corresponds to roughly $1B–$2B+ market cap with normal
# institutional turnover.  All S&P 500 stocks clear this comfortably.
SCAN_MIN_DOLLAR_VOL = 100_000_000

# ── Ticker blocklist ─────────────────────────────────────────────────────────
# Leveraged and inverse ETFs that can appear in broad IB stock scans but are
# structurally incompatible with our trend-following strategy.  Skip them
# before any historical-data fetch to avoid wasting scanner-cycle time.
TICKER_BLOCKLIST: set = {
    # Broad-market ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'VEA', 'VWO',
    # Sector / theme ETFs (SPDR, iShares, Vanguard)
    'XLV', 'XLK', 'XLF', 'XLE', 'XLI', 'XLU', 'XLY', 'XLP', 'XLRE', 'XLB', 'XLC',
    'IHI', 'IBB', 'IYH', 'IYW', 'IYF', 'IYE', 'IYJ', 'IYC', 'IYK', 'IYR',
    'VHT', 'VGT', 'VFH', 'VDE', 'VIS', 'VCR', 'VDC', 'VNQ', 'VAW',
    'GLD', 'SLV', 'USO', 'UNG',                          # commodities
    'TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'AGG', 'BND',   # bonds
    'ARKK', 'ARKG', 'ARKW', 'ARKF', 'ARKQ',              # thematic
    # Inverse / leveraged ETFs (bear)
    'SQQQ', 'SPXS', 'SDOW', 'SRTY', 'TZA', 'FAZ', 'SPXU',
    'SOXS', 'LABD', 'YANG', 'DRV',  'TECS', 'DUST', 'DRIP',
    'UVXY', 'SVXY',
    # Leveraged ETFs (bull) — distorted MA/RSI readings
    'TQQQ', 'UPRO', 'SPXL', 'UDOW', 'URTY', 'SOXL', 'LABU', 'TECL',
}

# ── Screener production rules ─────────────────────────────────────────────────
MIN_CANDLES          = 210     # minimum daily bars (SMA200 needs 200 + slope buffer)
VCP_RATIO            = 1.00    # legacy/diagnostic ATR5 / ATR20 reference; not an active 8096 entry gate
BREAKOUT_PCT         = 0.10    # legacy optimizer parameter; 10-day-high proximity is not an active entry gate
RVOL_MIN             = 2.5     # live scoring/ranking reference; not an active 8096 entry gate
BACKTEST_RVOL_MIN    = 1.1     # legacy daily close RVOL proxy; scanner ranking only in 8096 backtests
SPREAD_MAX_PCT       = 0.005   # maximum bid-ask spread (0.5%)
CORR_MAX             = 0.7     # max daily-return correlation with any current position
MAX_SECTOR_COUNT     = 2       # max simultaneous positions in the same sector
SMA200_SLOPE_LOOKBACK   = 5     # days over which SMA200 slope is measured
ENTRY_REPRICE_MAX_AGE_SEC = 60  # stale scan prices must be refreshed before entry
ENTRY_MAX_PRICE_DRIFT_PCT = 0.02 # max allowed scan-to-order price drift after refresh
ENTRY_LIMIT_ASK_CUSHION_PCT = float(os.getenv("VELOCITY_ENTRY_LIMIT_ASK_CUSHION_PCT", "0.0005"))  # add 5 bps over ask for marketable limit
ENTRY_LIMIT_MIN_TICK = float(os.getenv("VELOCITY_ENTRY_LIMIT_MIN_TICK", "0.01"))  # at least one cent above ask
ENTRY_LIMIT_MAX_OVER_MARKET_PCT = float(os.getenv("VELOCITY_ENTRY_LIMIT_MAX_OVER_MARKET_PCT", "0.002"))  # retain old 0.2% max cap

# ── Loop timing ───────────────────────────────────────────────────────────────
SCAN_INTERVAL          = 60    # seconds between cycles (1 minute)
ERROR_WAIT             = 60
LOG_BACKUP_COUNT       = 30   # keep 30 daily log files
EQUITY_RETRY_INTERVAL  = 5     # seconds between retries when equity fetch fails at startup
