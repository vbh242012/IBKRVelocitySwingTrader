import os


def _parse_hhmm(value: str, default: tuple[int, int]) -> tuple[int, int]:
    try:
        hour_s, minute_s = str(value or "").strip().split(":", 1)
        hour = int(hour_s)
        minute = int(minute_s)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return (hour, minute)
    except (TypeError, ValueError):
        pass
    return default

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR         = os.getenv(
    "VELOCITY_BASE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
)
STATE_FILE       = os.path.join(BASE_DIR, "engine_state.json")
DASHBOARD_FILE   = os.path.join(BASE_DIR, "dashboard_data.json")
EQUITY_HIST_FILE = os.path.join(BASE_DIR, "equity_history.json")
READINESS_FILE   = os.path.join(BASE_DIR, "readiness_snapshot.json")
HEALTH_REPORT_FILE = os.path.join(BASE_DIR, "daily_health_report.json")
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
VIX_CACHE_TTL_SEC = float(os.getenv("VELOCITY_VIX_CACHE_TTL_SEC", "300"))
VIX_FAILURE_COOLDOWN_BASE_SEC = float(os.getenv("VELOCITY_VIX_FAILURE_COOLDOWN_BASE_SEC", "300"))
VIX_FAILURE_COOLDOWN_MAX_SEC = float(os.getenv("VELOCITY_VIX_FAILURE_COOLDOWN_MAX_SEC", "600"))
HISTORICAL_DATA_TIMEOUT_SEC = float(os.getenv("VELOCITY_HISTORICAL_DATA_TIMEOUT_SEC", "25"))
HISTORICAL_DATA_WARMUP_ENABLED = os.getenv(
    "VELOCITY_HISTORICAL_DATA_WARMUP_ENABLED", "1"
).strip().lower() in {"1", "true", "yes", "on"}
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
MAX_DAILY_LOSS_PCT   = 0.03    # 3% intraday equity drawdown halts new entries for the day
ATR_PCT_MAX         = 0.07     # ATR_CHAND / price cap; filters excessively noisy names while preserving enough trade count
HARD_STOP_PCT        = 0.07    # 7% drawdown from entry triggers forced market exit regardless of ATR
RISK_PER_TRADE_PCT   = 0.02    # risk 2% of current equity per trade (ATR-based position sizing)
TIERED_PROFIT_EXIT_ENABLED = os.getenv(
    "VELOCITY_TIERED_PROFIT_EXIT_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
# Each tuple is (R-multiple target, cumulative position fraction that should be sold).
# R is the original per-share risk distance captured from the entry Chandelier stop.
# Example: a 6-share entry trims 1 share at +1R, 1 more at +1.5R, and 2 more at +2R.
TIERED_PROFIT_EXIT_R_LEVELS = (
    (1.0, 0.20),
    (1.5, 0.40),
    (2.0, 0.60),
)
BREAK_EVEN_R_MULT = float(os.getenv(
    "VELOCITY_BREAK_EVEN_R_MULT",
    str(TIERED_PROFIT_EXIT_R_LEVELS[0][0] if TIERED_PROFIT_EXIT_R_LEVELS else 1.0),
))
FRIDAY_CLOSE_HOUR    = 15      # ET hour after which Friday positions are evaluated for early close
FRIDAY_MIN_PROFIT_PCT = 0.03   # Friday close: exit if profit < 3% to avoid carrying weekend gap risk
EOD_EXIT_TIME        = (15, 50)  # ET — same-day EOD quality cleanup before overnight carry
EOD_HOLD_MIN_PROFIT_PCT = float(os.getenv("VELOCITY_EOD_HOLD_MIN_PROFIT_PCT", "0.0"))
EOD_HOLD_DAY_RANGE_LOCATION_MIN = float(os.getenv("VELOCITY_EOD_HOLD_DAY_RANGE_LOCATION_MIN", "0.70"))
EOD_HOLD_RELATIVE_STRENGTH_MIN = float(os.getenv("VELOCITY_EOD_HOLD_RELATIVE_STRENGTH_MIN", "0.0"))
EOD_HOLD_REQUIRE_STOP_CONFIRMED = os.getenv(
    "VELOCITY_EOD_HOLD_REQUIRE_STOP_CONFIRMED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
FRIDAY_ENTRY_CUTOFF_TIME = (12, 0)  # ET — avoid opening new swing positions that will be force-reviewed hours later

# ── Bear-phase participation ──────────────────────────────────────────────────
# Broad-market bear tape must not be treated like normal risk.
BEAR_PHASE_TRADING_ENABLED = os.getenv(
    "VELOCITY_BEAR_PHASE_TRADING_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
BEAR_PHASE_RISK_MULT       = 0.35   # 2% base risk → 0.7% risk in hostile tape
BEAR_PHASE_DOLLAR_VOL_MULT = 1.50   # require deeper liquidity in bear tape

# ── Session timing ────────────────────────────────────────────────────────────
ENTRY_START          = (9, 45)  # first valid new-entry time — wait past the noisy opening rotation
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
PROTECTIVE_STOP_CONFIRM_TIMEOUT_SEC = float(os.getenv("VELOCITY_PROTECTIVE_STOP_CONFIRM_TIMEOUT_SEC", "15"))
PROTECTIVE_STOP_CONFIRM_POLL_SEC = float(os.getenv("VELOCITY_PROTECTIVE_STOP_CONFIRM_POLL_SEC", "2"))

# ── Indicators ────────────────────────────────────────────────────────────────
RSI_PERIOD    = 14
ATR_PERIOD    = 14
MA_FAST       = 50
MA_SLOW       = 200

# ── Historical data requests ──────────────────────────────────────────────────
DAILY_LOOKBACK = '1 Y'
DAILY_BAR_SIZE = '1 day'

# ── Scanner filters (shared by live engine and backtester) ───────────────────
STRATEGY_PROFILE = os.getenv("VELOCITY_STRATEGY_PROFILE", "indicator_swing").strip().lower()
SCAN_MIN_PRICE      = float(os.getenv("VELOCITY_SCAN_MIN_PRICE", "5.0"))
SCAN_MIN_VOLUME     = int(os.getenv("VELOCITY_SCAN_MIN_VOLUME", "2000000"))
SCAN_MIN_MKTCAP     = float(os.getenv("VELOCITY_SCAN_MIN_MKTCAP", "2000000000"))
IB_SCANNER_SCAN_CODES: list = [
    c.strip()
    for c in os.getenv(
        "VELOCITY_IB_SCANNER_SCAN_CODES",
        (
            "MOST_ACTIVE,"
            "MOST_ACTIVE_USD,"
            "MOST_ACTIVE_AVG_USD,"
            "HOT_BY_VOLUME,"
            "TOP_VOLUME_RATE,"
            "HIGH_STVOLUME_5MIN,"
            "HIGH_STVOLUME_10MIN,"
            "TOP_OPEN_PERC_GAIN,"
            "HIGH_OPEN_GAP,"
            "BULLISH_MACD_DIST_VS_LAST,"
            "HIGH_LAST_VS_EMA20,"
            "HIGH_LAST_VS_EMA50"
        ),
    ).split(",")
    if c.strip()
] or ["MOST_ACTIVE"]
IB_SCANNER_LOCATION_CODE = (
    os.getenv("VELOCITY_IB_SCANNER_LOCATION_CODE", "STK.US.MAJOR").strip()
    or "STK.US.MAJOR"
)
IB_SCANNER_ROWS = int(os.getenv("VELOCITY_IB_SCANNER_ROWS", "-1"))  # -1 lets IBKR use its scanner default/maximum
IB_SCANNER_FILTERS_ENABLED = os.getenv(
    "VELOCITY_IB_SCANNER_FILTERS_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}

# Application candidate source.  IBKR's scanner does not expose one clean
# "all US common stocks" response, so the app can combine IBKR scanner hits
# with a rotating full-symbol universe loaded from NASDAQ Trader listings or a
# local file.  Final buy/no-buy decisions still come from the local screener.
APP_SCANNER_SOURCE = os.getenv(
    "VELOCITY_APP_SCANNER_SOURCE", "hybrid"
).strip().lower()  # ibkr | universe | hybrid
APP_SCANNER_BATCH_SIZE = int(os.getenv("VELOCITY_APP_SCANNER_BATCH_SIZE", "25"))
APP_SCANNER_MAX_SYMBOLS = int(os.getenv("VELOCITY_APP_SCANNER_MAX_SYMBOLS", "0"))  # 0 = no cap
APP_SCANNER_UNIVERSE_FILE = os.getenv("VELOCITY_APP_SCANNER_UNIVERSE_FILE", "").strip()
APP_SCANNER_UNIVERSE_CACHE_FILE = os.getenv(
    "VELOCITY_APP_SCANNER_UNIVERSE_CACHE_FILE",
    os.path.join(BASE_DIR, "symbol_universe_cache.json"),
)
APP_SCANNER_UNIVERSE_TTL_SEC = float(os.getenv(
    "VELOCITY_APP_SCANNER_UNIVERSE_TTL_SEC", str(24 * 3600)
))
APP_PREFILTER_ENABLED = os.getenv(
    "VELOCITY_APP_PREFILTER_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
APP_PREFILTER_START_TIME = _parse_hhmm(
    os.getenv("VELOCITY_APP_PREFILTER_START_TIME", "06:30"),
    (6, 30),
)
APP_PREFILTER_CACHE_FILE = os.getenv(
    "VELOCITY_APP_PREFILTER_CACHE_FILE",
    os.path.join(BASE_DIR, "premarket_universe_prefilter.json"),
)
APP_PREFILTER_HISTORY_SLEEP_SEC = float(os.getenv(
    "VELOCITY_APP_PREFILTER_HISTORY_SLEEP_SEC", "0.05"
))
APP_PREFILTER_PROGRESS_EVERY = int(os.getenv(
    "VELOCITY_APP_PREFILTER_PROGRESS_EVERY", "100"
))
APP_PREFILTER_STOP_AT_ENTRY_START = os.getenv(
    "VELOCITY_APP_PREFILTER_STOP_AT_ENTRY_START", "1"
).strip().lower() not in {"0", "false", "no", "off"}

# IBKR scanner-side generic filters.  These are upstream copies of filters the
# live screener still validates locally.  They reduce noisy IBKR candidates, but
# they are not trusted as final risk checks because broker scanner metadata can
# lag or behave differently by venue/session.
IB_SCANNER_CHANGE_OPEN_PCT_ABOVE = float(os.getenv(
    "VELOCITY_IB_SCANNER_CHANGE_OPEN_PCT_ABOVE",
    "0",
))
IB_SCANNER_OPEN_GAP_PCT_BELOW = float(os.getenv(
    "VELOCITY_IB_SCANNER_OPEN_GAP_PCT_BELOW",
    "15",
))
IB_SCANNER_LAST_VS_EMA20_PCT_ABOVE = float(os.getenv(
    "VELOCITY_IB_SCANNER_LAST_VS_EMA20_PCT_ABOVE", "0"
))
IB_SCANNER_LAST_VS_EMA50_PCT_ABOVE = float(os.getenv(
    "VELOCITY_IB_SCANNER_LAST_VS_EMA50_PCT_ABOVE", "0"
))
IB_SCANNER_MACD_HISTOGRAM_ABOVE = float(os.getenv(
    "VELOCITY_IB_SCANNER_MACD_HISTOGRAM_ABOVE", "0"
))
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
SPREAD_MAX_PCT       = 0.005   # maximum bid-ask spread (0.5%)
CORR_MAX             = 0.7     # max daily-return correlation with any current position
MAX_SECTOR_COUNT     = 2       # max simultaneous positions in the same sector
SMA200_SLOPE_LOOKBACK   = 5     # days over which SMA200 slope is measured
ENTRY_REPRICE_MAX_AGE_SEC = 60  # stale scan prices must be refreshed before entry
ENTRY_MAX_PRICE_DRIFT_PCT = 0.02 # max allowed scan-to-order price drift after refresh
ENTRY_LIMIT_ASK_CUSHION_PCT = float(os.getenv("VELOCITY_ENTRY_LIMIT_ASK_CUSHION_PCT", "0.0005"))  # add 5 bps over ask for marketable limit
ENTRY_LIMIT_MIN_TICK = float(os.getenv("VELOCITY_ENTRY_LIMIT_MIN_TICK", "0.01"))  # at least one cent above ask
ENTRY_LIMIT_MAX_OVER_MARKET_PCT = float(os.getenv("VELOCITY_ENTRY_LIMIT_MAX_OVER_MARKET_PCT", "0.002"))  # retain old 0.2% max cap

# ── Indicator swing system ───────────────────────────────────────────────────
# Primary edge: daily/weekly momentum, relative strength, and indicator timing.
SWING_RS_MIN_63D = float(os.getenv("VELOCITY_SWING_RS_MIN_63D", "0.08"))
SWING_RS_MIN_126D = float(os.getenv("VELOCITY_SWING_RS_MIN_126D", "0.10"))
SWING_MIN_13W_RETURN = float(os.getenv("VELOCITY_SWING_MIN_13W_RETURN", "0.12"))
SWING_MIN_26W_RETURN = float(os.getenv("VELOCITY_SWING_MIN_26W_RETURN", "0.18"))
SWING_MIN_PRICE_VS_52W_HIGH = float(os.getenv("VELOCITY_SWING_MIN_PRICE_VS_52W_HIGH", "0.85"))
SWING_MAX_PULLBACK_FROM_HIGH20 = float(os.getenv("VELOCITY_SWING_MAX_PULLBACK_FROM_HIGH20", "0.12"))
SWING_MAX_MA20_EXTENSION = float(os.getenv("VELOCITY_SWING_MAX_MA20_EXTENSION", "0.12"))
SWING_MIN_VOLUME_PACE = float(os.getenv("VELOCITY_SWING_MIN_VOLUME_PACE", "1.20"))
SWING_MIN_SCORE = float(os.getenv("VELOCITY_SWING_MIN_SCORE", "50.0"))
SWING_TIME_STOP_BARS = int(os.getenv("VELOCITY_SWING_TIME_STOP_BARS", "10"))
SWING_TIME_STOP_MIN_PROFIT_PCT = float(os.getenv("VELOCITY_SWING_TIME_STOP_MIN_PROFIT_PCT", "0.0"))

# ── Multi-indicator swing profile ────────────────────────────────────────────
INDICATOR_SWING_MIN_SCORE = float(os.getenv("VELOCITY_INDICATOR_SWING_MIN_SCORE", "50.0"))
INDICATOR_SWING_TIME_STOP_BARS = int(os.getenv("VELOCITY_INDICATOR_SWING_TIME_STOP_BARS", "10"))
INDICATOR_SWING_TIME_STOP_MIN_PROFIT_PCT = float(os.getenv("VELOCITY_INDICATOR_SWING_TIME_STOP_MIN_PROFIT_PCT", "0.0"))
INDICATOR_SWING_MIN_VOLUME_PACE = float(os.getenv("VELOCITY_INDICATOR_SWING_MIN_VOLUME_PACE", "1.2"))
INDICATOR_SWING_RSI_OVERSOLD = float(os.getenv("VELOCITY_INDICATOR_SWING_RSI_OVERSOLD", "35.0"))
INDICATOR_SWING_RSI_OVERBOUGHT = float(os.getenv("VELOCITY_INDICATOR_SWING_RSI_OVERBOUGHT", "70.0"))
INDICATOR_SWING_STOCH_OVERSOLD = float(os.getenv("VELOCITY_INDICATOR_SWING_STOCH_OVERSOLD", "20.0"))
INDICATOR_SWING_STOCH_OVERBOUGHT = float(os.getenv("VELOCITY_INDICATOR_SWING_STOCH_OVERBOUGHT", "80.0"))
INDICATOR_SWING_STRATEGIES = tuple(
    s.strip().lower()
    for s in os.getenv(
        "VELOCITY_INDICATOR_SWING_STRATEGIES",
        "ma_cross",
    ).split(",")
    if s.strip()
)

# ── Analyst rating integration ───────────────────────────────────────────────
# Live ratings use a local CSV first, then Finnhub when a key is configured, and
# finally Yahoo/yfinance as a free fallback. Backtests use only dated local CSV
# snapshots to avoid look-ahead bias.
ANALYST_RATINGS_ENABLED = os.getenv(
    "VELOCITY_ANALYST_RATINGS_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
FINNHUB_API_KEY = os.getenv("VELOCITY_FINNHUB_API_KEY", "").strip()
ANALYST_RATINGS_FREE_SOURCE = os.getenv(
    "VELOCITY_ANALYST_RATINGS_FREE_SOURCE", "yahoo"
).strip().lower()
ANALYST_RATINGS_FILE = os.getenv("VELOCITY_ANALYST_RATINGS_FILE", "").strip()
ANALYST_RATINGS_CACHE_FILE = os.getenv(
    "VELOCITY_ANALYST_RATINGS_CACHE_FILE",
    os.path.join(BASE_DIR, "analyst_ratings_cache.json"),
)
ANALYST_RATINGS_TTL_SEC = float(os.getenv("VELOCITY_ANALYST_RATINGS_TTL_SEC", str(6 * 3600)))
ANALYST_RATING_MIN_ANALYSTS = int(os.getenv("VELOCITY_ANALYST_RATING_MIN_ANALYSTS", "5"))
ANALYST_RATING_SCORE_WEIGHT = float(os.getenv("VELOCITY_ANALYST_RATING_SCORE_WEIGHT", "8.0"))
ANALYST_RATING_SELL_THRESHOLD = float(os.getenv("VELOCITY_ANALYST_RATING_SELL_THRESHOLD", "-0.35"))
ANALYST_RATING_EXIT_ENABLED = os.getenv(
    "VELOCITY_ANALYST_RATING_EXIT_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}

# ── Loop timing ───────────────────────────────────────────────────────────────
SCAN_INTERVAL          = 60    # seconds between cycles (1 minute)
ERROR_WAIT             = 60
LOG_BACKUP_COUNT       = 30   # keep 30 daily log files
EQUITY_RETRY_INTERVAL  = 5     # seconds between retries when equity fetch fails at startup

# ── Reconnect behavior ────────────────────────────────────────────────────────
RECONNECT_INITIAL_WAIT_SEC = float(os.getenv("VELOCITY_RECONNECT_INITIAL_WAIT_SEC", "5"))
RECONNECT_MAX_WAIT_SEC = float(os.getenv("VELOCITY_RECONNECT_MAX_WAIT_SEC", "300"))
# Suppress identical CRITICAL/ERROR alerts within this window (seconds) to
# prevent webhook and log floods during prolonged outages.
ALERT_DEDUP_WINDOW_SEC = float(os.getenv("VELOCITY_ALERT_DEDUP_WINDOW_SEC", "600"))

# ── HMDS warmup retry ─────────────────────────────────────────────────────────
HMDS_WARMUP_MAX_RETRIES = int(os.getenv("VELOCITY_HMDS_WARMUP_MAX_RETRIES", "3"))
HMDS_WARMUP_RETRY_WAIT_SEC = float(os.getenv("VELOCITY_HMDS_WARMUP_RETRY_WAIT_SEC", "60"))

# ── Break-even exit ────────────────────────────────────────────────────────────
# Exit fires when current price falls below entry + (peak_gain × this fraction).
# 0.25 means we keep at least 25% of the peak gain before exiting (vs. 0.0 = exit at entry).
BREAK_EVEN_PEAK_RETAIN_FRACTION = float(os.getenv("VELOCITY_BREAK_EVEN_PEAK_RETAIN_FRACTION", "0.25"))

# ── Stale losing position exit ────────────────────────────────────────────────
STALE_POSITION_MIN_BARS = int(os.getenv("VELOCITY_STALE_POSITION_MIN_BARS", "3"))
STALE_POSITION_MAX_LOSS_PCT = float(os.getenv("VELOCITY_STALE_POSITION_MAX_LOSS_PCT", "-0.02"))
STALE_POSITION_MAX_PEAK_PCT = float(os.getenv("VELOCITY_STALE_POSITION_MAX_PEAK_PCT", "0.01"))

# ── Late entry gate ────────────────────────────────────────────────────────────
LATE_ENTRY_CUTOFF_TIME = _parse_hhmm(
    os.getenv("VELOCITY_LATE_ENTRY_CUTOFF_TIME", "14:30"), (14, 30)
)
LATE_ENTRY_MIN_SCORE = float(os.getenv("VELOCITY_LATE_ENTRY_MIN_SCORE", "92.0"))

# ── Market data blackout detection ────────────────────────────────────────────
# If ≥ RATIO of scanner candidates miss live price for ≥ STREAK consecutive
# cycles, emit one CRITICAL alert and suspend new entries until data restores.
DATA_BLACKOUT_RATIO_THRESHOLD = float(os.getenv("VELOCITY_DATA_BLACKOUT_RATIO_THRESHOLD", "0.70"))
DATA_BLACKOUT_MIN_CANDIDATES = int(os.getenv("VELOCITY_DATA_BLACKOUT_MIN_CANDIDATES", "5"))
DATA_BLACKOUT_STREAK_ALERT = int(os.getenv("VELOCITY_DATA_BLACKOUT_STREAK_ALERT", "2"))
