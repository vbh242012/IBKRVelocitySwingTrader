import json
import logging
import os
import copy
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, time as datetime_time, timedelta
from logging.handlers import TimedRotatingFileHandler
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from ib_async import IB, Index, Stock, MarketOrder, Order, util

from src.config import (
    BASE_DIR, STATE_FILE, DASHBOARD_FILE, EQUITY_HIST_FILE, READINESS_FILE,
    HEALTH_REPORT_FILE, HALT_FILE, FORCE_EXIT_FILE,
    LOG_DIR, LOG_FILE,
    IB_HOST, IB_PORT, IB_CLIENT_ID, MARKET_DATA_TYPE, VIX_MARKET_DATA_TYPE,
    VIX_CACHE_TTL_SEC, VIX_FAILURE_COOLDOWN_BASE_SEC, VIX_FAILURE_COOLDOWN_MAX_SEC,
    HISTORICAL_DATA_TIMEOUT_SEC, HISTORICAL_DATA_WARMUP_ENABLED,
    ACCOUNT_CURRENCY,
    TRADING_MODE, LIVE_TRADING_ACK, LIVE_TRADING_ACK_PHRASE, LIVE_IB_PORTS, PAPER_IB_PORTS,
    ALERT_WEBHOOK_URL, ALERT_TIMEOUT_SEC,
    MAX_POSITIONS_CAP, MIN_BUCKET_SIZE, SETTLED_CASH_DEPLOYMENT_PCT,
    VIX_THRESHOLD, HOLD_TRADING_BARS, PROFIT_MIN_THRESHOLD, VELOCITY_EXIT_TIME, BREAK_EVEN_PCT,
    ENTRY_START, ENTRY_END, STOP_ACTIVATION_TIME, VOL_MULT_FRIDAY, PRE_ENTRY_SYNC_TIME,
    POST_OPEN_AUDIT_TIME, PREMARKET_READINESS_TIME, POST_CLOSE_MAINTENANCE_TIME,
    MARKET_CLOSE_TIME, ENTRY_PARENT_TIF, ENTRY_ALL_OR_NONE,
    RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW, RSI_THRESHOLD,
    ORB_LOOKBACK, ORB_BAR_SIZE, DAILY_LOOKBACK, DAILY_BAR_SIZE,
    SCAN_MIN_DOLLAR_VOL,
    SCAN_INTERVAL, ERROR_WAIT,
    LOG_BACKUP_COUNT,
    EQUITY_RETRY_INTERVAL,
    TICKER_BLOCKLIST,
    GAP_MAX_PCT,
    MAX_DAILY_LOSS_PCT,
    RSI_MIN_DELTA,
    DAY_RANGE_LOCATION_MIN,
    INTRADAY_GAIN_MIN,
    ATR_PCT_MAX,
    HARD_STOP_PCT,
    RISK_PER_TRADE_PCT,
    BEAR_PHASE_TRADING_ENABLED,
    BEAR_PHASE_RISK_MULT,
    BEAR_PHASE_DOLLAR_VOL_MULT,
    BEAR_RVOL_MIN,
    BEAR_VCP_RATIO,
    BEAR_RSI_THRESHOLD,
    BEAR_RSI_MIN_DELTA,
    BEAR_GAP_MAX_PCT,
    FRIDAY_CLOSE_HOUR,
    FRIDAY_MIN_PROFIT_PCT,
    FRIDAY_ENTRY_CUTOFF_TIME,
    EOD_EXIT_TIME,
    MIN_CANDLES, VCP_RATIO, BREAKOUT_PCT, RVOL_MIN, SPREAD_MAX_PCT, SCORING_MODEL,
    SCAN_MIN_PRICE,
    CORR_MAX, MAX_SECTOR_COUNT, SMA200_SLOPE_LOOKBACK,
    ENTRY_REPRICE_MAX_AGE_SEC, ENTRY_MAX_PRICE_DRIFT_PCT,
    ENTRY_LIMIT_ASK_CUSHION_PCT, ENTRY_LIMIT_MIN_TICK, ENTRY_LIMIT_MAX_OVER_MARKET_PCT,
    CHANDELIER_PERIOD, CHANDELIER_MULT,
    PROTECTIVE_STOP_CONFIRM_TIMEOUT_SEC, PROTECTIVE_STOP_CONFIRM_POLL_SEC,
)
from src.ib_gateway import ensure_ib_gateway_ready
from src.indicators import apply_all
from src.scanner import build_momentum_scanners
from src.scoring import score_candidate, volume_pace_from_intraday

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# Single timezone object reused throughout the module.
# The machine runs on IST (UTC+5:30); all times must be anchored to US/Eastern
# so market-hours checks, timestamps, and log lines are unambiguous.
_TZ_NY = pytz.timezone('US/Eastern')
_REJECTED_ORDER_STATUSES = {'Inactive', 'ApiCancelled', 'Cancelled'}


class AccountDataUnavailable(RuntimeError):
    """Raised when IBKR account summary data is not fresh enough for entries."""


def _count_trading_days(entry_dt: datetime, now: datetime) -> int:
    """Count complete Mon-Fri trading sessions elapsed between entry_dt and now."""
    entry_date = entry_dt.date()
    now_date   = now.date()
    count      = 0
    cursor     = entry_date
    while cursor < now_date:
        if cursor.weekday() < 5:
            count += 1
        cursor += timedelta(days=1)
    return count

# ── Logging ───────────────────────────────────────────────────────────────────
class _EasternFormatter(logging.Formatter):
    def formatTime(self, record, _datefmt=None):
        dt = datetime.fromtimestamp(record.created, _TZ_NY)
        return dt.strftime('%Y-%m-%d %H:%M:%S %Z')


class _EasternTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Rotate daily logs at US/Eastern midnight, independent of host timezone."""

    def computeRollover(self, currentTime):
        now_et = datetime.fromtimestamp(currentTime, _TZ_NY)
        next_midnight_naive = datetime.combine(
            now_et.date() + timedelta(days=1),
            datetime_time.min,
        )
        next_midnight_et = _TZ_NY.localize(next_midnight_naive)
        return int(next_midnight_et.timestamp())

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        suffix_date = datetime.fromtimestamp(
            self.rolloverAt - 1,
            _TZ_NY,
        ).strftime(self.suffix)
        dfn = self.rotation_filename(f"{self.baseFilename}.{suffix_date}")
        if os.path.exists(dfn):
            os.remove(dfn)
        self.rotate(self.baseFilename, dfn)

        if self.backupCount > 0:
            for old_file in self.getFilesToDelete():
                os.remove(old_file)

        if not self.delay:
            self.stream = self._open()

        current_time = int(time.time())
        new_rollover_at = self.computeRollover(current_time)
        while new_rollover_at <= current_time:
            new_rollover_at = self.computeRollover(new_rollover_at)
        self.rolloverAt = new_rollover_at


def _log_namer(default_name: str) -> str:
    # TimedRotatingFileHandler default suffix: logs/trading_engine.log.2026-05-12
    # We produce the cleaner form:             logs/trading_engine_2026-05-12.log
    if '.log.' in default_name:
        base, date_suffix = default_name.rsplit('.log.', 1)
        return f"{base}_{date_suffix}.log"
    return default_name


logger = logging.getLogger('VelocityEngine')
logger.setLevel(logging.INFO)
# Guard against duplicate handlers when the module is re-imported (tests, restarts).
if not logger.handlers:
    _handler = _EasternTimedRotatingFileHandler(
        LOG_FILE, when='midnight', backupCount=LOG_BACKUP_COUNT
    )
    _handler.namer = _log_namer
    _handler.setFormatter(_EasternFormatter('%(asctime)s | %(levelname)s | %(message)s'))
    _console = logging.StreamHandler(sys.stdout)
    _console.setFormatter(_EasternFormatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(_handler)
    logger.addHandler(_console)


# ── Engine ────────────────────────────────────────────────────────────────────
class VelocityEngine:
    def __init__(self):
        self.ib           = IB()
        self.state        = self.load_state()

        # Metrics written to dashboard_data.json after every cycle
        self._last_equity:      float            = 0.0
        self._last_settled_cash: float           = 0.0
        self._equity_initialized: bool         = False   # True after first real IBKR fetch
        self._last_vix:         Optional[float] = None
        self._last_vix_ts:      float           = 0.0   # time.time() of last successful VIX fetch
        self._last_vix_source:  Optional[str]   = None
        self._vix_failure_count: int            = 0
        self._next_vix_retry_ts: float          = 0.0
        self._last_vix_failure_ts: float        = 0.0
        self._last_scan_ts:     Optional[str]   = None
        self._next_scan_dt:     Optional[str]   = None   # ISO string for the web UI
        self._historical_data_health: Dict[str, dict] = {}

        # Daily loss circuit breaker
        self._day_start_equity: Optional[float] = None
        self._day_start_date:   Optional[str]   = None

        # Bar cache: keyed by symbol, invalidated daily (date-scoped)
        self._bar_cache: Dict[str, dict] = {}
        # Contract cache: avoids re-qualifying the same symbol every cycle
        self._contract_cache: Dict[str, object] = {}
        # VIX contract cached after first successful qualification
        self._vix_contract = None
        # SPY regime cache: refreshed once per trading day
        self._spy_cache: dict = {}
        # Sector cache: symbol → industry string (stable; never invalidated)
        self._sector_cache: Dict[str, str] = {}
        # Symbols that failed stable same-day scan requirements. This is an IBKR
        # pacing guard, not an alpha rule; dynamic intraday failures are never cached.
        self._daily_scan_skip: Dict[str, str] = {}
        # Last trading date a full protective stop-order audit was run.
        self._last_audit_date: Optional[str] = None
        self._last_audit_at: Optional[datetime] = None
        # Last trading date the post-open protective audit was run.
        self._last_post_open_audit_date: Optional[str] = None
        # Last trading dates for non-trading operational jobs.
        self._last_premarket_readiness_date: Optional[str] = None
        self._last_post_close_maintenance_date: Optional[str] = None
        # Daily EOD flat: track the date so the exit fires once per trading day.
        self._last_eod_exit_date: Optional[str] = None
        # Avoid deleting state on one transient/partial IBKR positions snapshot.
        self._missing_position_counts: Dict[str, int] = {}
        # Daily operational counters written to HEALTH_REPORT_FILE. These are
        # deliberately lightweight so a long paper run can be reviewed from one
        # compact JSON summary instead of only raw logs.
        self._health_date = datetime.now(_TZ_NY).strftime('%Y-%m-%d')
        self._health_metrics = self._new_health_metrics()

        self.connect()

    # ── IB connection ──────────────────────────────────────────────────────────
    def connect(self):
        self._validate_deployment_mode()
        try:
            if not ensure_ib_gateway_ready():
                raise RuntimeError(
                    f"IB Gateway API port {IB_HOST}:{IB_PORT} is unavailable "
                    "and auto-start is disabled."
                )
            self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
            self.ib.reqMarketDataType(MARKET_DATA_TYPE)
            self.ib.errorEvent            += self._on_ib_error
            self.ib.disconnectedEvent     += self._on_ib_disconnect
            self.ib.commissionReportEvent += self._on_commission_report
            logger.info(
                f"ENGINE CONNECTED: IB Gateway Ready "
                f"(mode={TRADING_MODE}, host={IB_HOST}, port={IB_PORT}, clientId={IB_CLIENT_ID})."
            )
            self._write_dashboard_data(connected=True)
            self._warmup_historical_data(reason="connect")
        except Exception as e:
            self._alert("CRITICAL", f"CONNECTION FAILED: Is IB Gateway open? {e}")
            sys.exit()

    def _on_ib_error(self, reqId, errorCode, errorString, contract):
        # Codes that are purely informational and produce no actionable log noise.
        # 162  : scanner subscription ended after reqScannerData — expected behaviour.
        # 202  : order cancellation confirmation (startup orphan cleanup) — expected.
        # 2104 : market data farm connected (info).
        # 2106 : HMDS data farm connected (info).
        # 2107 : HMDS data farm inactive (info).
        # 2108 : market data farm inactive but available on demand (info).
        # 2119 : market data farm connecting (info).
        # 2158 : sec-def data farm connected (info).
        # 10167: delayed data notice — expected when MARKET_DATA_TYPE=3.
        # 135  : "Can't find order" — cascade when child bracket orders are already
        #        cancelled by IB before our explicit cancel loop runs.
        # 10147: "OrderId not found" — same cascade as 135.
        SILENT = {135, 162, 202, 2104, 2106, 2107, 2108, 2119, 2158, 10147, 10167}
        if errorCode in SILENT:
            return
        self._metric_inc('ib_errors')
        self._metric_inc('ib_error_codes', subkey=str(errorCode))
        if errorCode == 10349:
            logger.warning(f"IB preset override (10349) reqId={reqId}: {errorString}")
            return
        logger.warning(f"IB error {errorCode} reqId={reqId}: {errorString}")

    def _on_ib_disconnect(self):
        self._metric_inc('ib_disconnects')
        self._alert("ERROR", "IB DISCONNECTED: connection lost — will attempt reconnect on next cycle.")
        self._write_dashboard_data(connected=False)

    def _on_commission_report(self, trade, fill, report):
        """Async callback: IB commission reports arrive after the fill confirmation.
        Match by the entry order ID stored at fill time and persist to state."""
        try:
            commission = float(report.commission)
        except (TypeError, ValueError):
            return
        if not np.isfinite(commission) or commission <= 0:
            return
        order_id = trade.order.orderId
        for sym, data in self.state.items():
            if data.get('entry_order_id') == order_id:
                data['commission'] = round(commission, 4)
                self.save_state()
                logger.info(
                    f"COMMISSION: {sym} order={order_id} "
                    f"commission=${commission:.4f}"
                )
                break

    def _reconnect(self) -> bool:
        self._validate_deployment_mode()
        logger.info("RECONNECT: attempting to reconnect to IB Gateway...")
        for attempt in range(1, 11):
            try:
                if not ensure_ib_gateway_ready():
                    raise RuntimeError(
                        f"IB Gateway API port {IB_HOST}:{IB_PORT} is unavailable "
                        "and auto-start is disabled."
                    )
                self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
                self.ib.reqMarketDataType(MARKET_DATA_TYPE)
                logger.info(f"RECONNECT: success on attempt {attempt}")
                self._metric_inc('reconnect_successes')
                self._write_dashboard_data(connected=True)
                self._warmup_historical_data(reason="reconnect")
                return True
            except Exception as e:
                logger.warning(f"RECONNECT: attempt {attempt}/10 failed: {e}")
                time.sleep(5)
        self._metric_inc('reconnect_failures')
        self._alert("CRITICAL", "RECONNECT: all 10 attempts failed — skipping trading cycles until restored")
        return False

    def _request_vix_tickers(self):
        """Request VIX with its own market-data type, then restore stock data mode."""
        restore_type = None
        if VIX_MARKET_DATA_TYPE != MARKET_DATA_TYPE:
            restore_type = MARKET_DATA_TYPE
            self.ib.reqMarketDataType(VIX_MARKET_DATA_TYPE)
        try:
            return self.ib.reqTickers(self._vix_contract)
        finally:
            if restore_type is not None:
                self.ib.reqMarketDataType(restore_type)

    def _fetch_vix_price(self) -> Optional[float]:
        """Return the current/delayed VIX value, with a robust historical fallback.

        VIX is a regime filter — precision matters less than availability. Strategy:
        1. If a valid VIX was fetched within the configured TTL, return it from
           cache to avoid hammering IBKR with repeated reqHistoricalData calls.
        2. Try the live/delayed ticker, checking multiple price fields.
        3. On ticker miss, fetch bounded historical data.
        4. If both fresh paths fail, enter a cooldown. A stale VIX reading must
           not authorize new risk.
        """
        now = time.time()
        # Return cached value if still fresh enough.
        if self._last_vix is not None and (now - self._last_vix_ts) < VIX_CACHE_TTL_SEC:
            return self._last_vix

        next_retry_ts = float(getattr(self, '_next_vix_retry_ts', 0.0) or 0.0)
        if now < next_retry_ts:
            wait_s = int(next_retry_ts - now)
            self._metric_inc('vix_retry_suppressed')
            logger.warning(
                f"VIX data unavailable; retry cooldown active for {wait_s}s. "
                "Skipping entries as precaution."
            )
            return None

        try:
            vix_tickers = self._request_vix_tickers()
        except Exception as e:
            logger.warning(f"VIX ticker request failed ({e}); trying historical fallback.")
            self._metric_inc('vix_ticker_failures')
            vix_tickers = []

        if vix_tickers:
            vix_ticker = vix_tickers[0]
            # Check multiple fields: marketPrice() covers last/ask/bid; also try
            # close and prevClose which are populated by delayed subscriptions.
            vix_price = (
                self._coerce_positive_price(vix_ticker.marketPrice())
                or self._coerce_positive_price(getattr(vix_ticker, 'close', None))
                or self._coerce_positive_price(getattr(vix_ticker, 'last', None))
                or self._coerce_positive_price(getattr(vix_ticker, 'prevClose', None))
            )
            if vix_price is not None:
                self._record_vix_success(vix_price, source="ticker")
                return vix_price
            logger.info("VIX ticker returned no usable price; using historical fallback.")
            self._metric_inc('vix_ticker_misses')
        else:
            logger.info("VIX ticker unavailable; using historical fallback.")
            self._metric_inc('vix_ticker_misses')

        bars = self._request_historical_bars(
            self._vix_contract,
            label="VIX",
            duration='5 D',
            bar_size=DAILY_BAR_SIZE,
            what='TRADES',
            use_rth=False,
            timeout=HISTORICAL_DATA_TIMEOUT_SEC,
            metric_prefix='vix',
        )
        if not bars:
            logger.warning("VIX historical fallback returned no bars.")
            self._metric_inc('vix_fallback_failures')
            self._record_vix_failure("historical_no_bars")
            return None
        hist_price = self._coerce_positive_price(getattr(bars[-1], 'close', None))
        if hist_price is None:
            logger.warning("VIX historical fallback returned an invalid close.")
            self._metric_inc('vix_fallback_failures')
            self._record_vix_failure("historical_invalid_close")
            return None
        logger.info(f"VIX fallback: using latest historical close {hist_price:.2f}")
        self._metric_inc('vix_fallback_successes')
        self._record_vix_success(hist_price, source="historical")
        return hist_price

    def _ensure_connected(self) -> bool:
        if not self.ib.isConnected():
            logger.warning("ENGINE: not connected — attempting reconnect before cycle")
            return self._reconnect()
        return True

    def _safe_sleep(self, seconds: float, context: str = "sleep"):
        """Sleep through IB's event loop, but treat socket loss as reconnectable."""
        try:
            self.ib.sleep(seconds)
        except ConnectionError as e:
            self._metric_inc('ib_sleep_disconnects')
            logger.warning(
                f"{context}: IB socket disconnected during sleep ({e}); "
                "next cycle will reconnect."
            )
            self._write_dashboard_data(connected=False)
            time.sleep(min(max(float(seconds), 1.0), ERROR_WAIT))

    def _record_vix_success(self, price: float, source: str):
        self._last_vix = price
        self._last_vix_ts = time.time()
        self._last_vix_source = source
        self._vix_failure_count = 0
        self._next_vix_retry_ts = 0.0
        self._last_vix_failure_ts = 0.0

    def _record_vix_failure(self, reason: str):
        if not hasattr(self, '_vix_failure_count'):
            self._vix_failure_count = 0
        if not hasattr(self, '_next_vix_retry_ts'):
            self._next_vix_retry_ts = 0.0
        self._vix_failure_count = int(getattr(self, '_vix_failure_count', 0) or 0) + 1
        self._last_vix_failure_ts = time.time()
        base = max(float(VIX_FAILURE_COOLDOWN_BASE_SEC), 0.0)
        max_wait = max(float(VIX_FAILURE_COOLDOWN_MAX_SEC), base)
        cooldown = min(base * self._vix_failure_count, max_wait)
        self._next_vix_retry_ts = self._last_vix_failure_ts + cooldown
        next_retry = datetime.fromtimestamp(self._next_vix_retry_ts, _TZ_NY)
        logger.warning(
            f"VIX failure recorded ({reason}); retry in {cooldown:.0f}s "
            f"at {next_retry.strftime('%H:%M:%S %Z')}."
        )

    def _request_historical_bars(
        self,
        contract,
        label: str,
        duration: str,
        bar_size: str,
        what: str,
        use_rth: bool,
        timeout: float,
        metric_prefix: Optional[str] = None,
    ):
        """Bounded historical request with compact health bookkeeping."""
        if not hasattr(self, '_historical_data_health'):
            self._historical_data_health = {}
        started = time.time()
        try:
            bars = self.ib.reqHistoricalData(
                contract, '', duration, bar_size, what, use_rth, timeout=timeout
            )
            latency = time.time() - started
            ok = bool(bars)
            self._historical_data_health[label] = {
                'ok': ok,
                'last_checked_at': datetime.now(_TZ_NY).isoformat(),
                'latency_sec': round(latency, 3),
                'bars': len(bars) if bars else 0,
                'error': None if ok else 'no_bars',
            }
            if metric_prefix and not ok:
                self._metric_inc(f'{metric_prefix}_historical_no_bars')
            return bars
        except Exception as e:
            latency = time.time() - started
            self._historical_data_health[label] = {
                'ok': False,
                'last_checked_at': datetime.now(_TZ_NY).isoformat(),
                'latency_sec': round(latency, 3),
                'bars': 0,
                'error': str(e),
            }
            if metric_prefix:
                self._metric_inc(f'{metric_prefix}_historical_exceptions')
            logger.warning(f"{label} historical request failed after {latency:.1f}s: {e}")
            return []

    def _warmup_historical_data(self, reason: str = "manual") -> bool:
        """Probe SPY then VIX so HMDS wakes before the entry gate depends on it."""
        if not HISTORICAL_DATA_WARMUP_ENABLED:
            return True
        try:
            spy = Stock('SPY', 'SMART', 'USD', primaryExchange='ARCA')
            qualified = self.ib.qualifyContracts(spy)
            if qualified:
                spy = qualified[0]
        except Exception as e:
            logger.warning(f"HMDS WARMUP[{reason}]: SPY qualification failed: {e}")
            self._metric_inc('historical_warmup_failures')
            return False

        spy_bars = self._request_historical_bars(
            spy,
            label="SPY",
            duration='5 D',
            bar_size=DAILY_BAR_SIZE,
            what='TRADES',
            use_rth=True,
            timeout=HISTORICAL_DATA_TIMEOUT_SEC,
            metric_prefix='spy',
        )
        if not spy_bars:
            logger.warning(
                f"HMDS WARMUP[{reason}]: SPY historical failed. "
                "IBKR historical farm is not healthy yet."
            )
            self._metric_inc('historical_warmup_failures')
            return False

        if not self._ensure_vix_contract():
            logger.warning(f"HMDS WARMUP[{reason}]: VIX contract unavailable.")
            self._metric_inc('historical_warmup_failures')
            return False

        vix_bars = self._request_historical_bars(
            self._vix_contract,
            label="VIX",
            duration='5 D',
            bar_size=DAILY_BAR_SIZE,
            what='TRADES',
            use_rth=False,
            timeout=HISTORICAL_DATA_TIMEOUT_SEC,
            metric_prefix='vix',
        )
        if not vix_bars:
            logger.warning(
                f"HMDS WARMUP[{reason}]: SPY historical OK but VIX historical failed. "
                "Treating VIX as unavailable until a later retry succeeds."
            )
            self._record_vix_failure("warmup_vix_failed")
            self._metric_inc('historical_warmup_failures')
            return False

        hist_price = self._coerce_positive_price(getattr(vix_bars[-1], 'close', None))
        if hist_price is not None:
            self._record_vix_success(hist_price, source="historical_warmup")
        vix_text = f"{hist_price:.2f}" if hist_price is not None else "unknown"
        logger.info(
            f"HMDS WARMUP[{reason}]: SPY and VIX historical data OK "
            f"(VIX={vix_text})."
        )
        self._metric_inc('historical_warmup_successes')
        return True

    def _operator_halt_active(self) -> bool:
        """Manual kill switch: HALT_FILE presence blocks new entries only."""
        return os.path.exists(HALT_FILE)

    def _force_exit_active(self) -> bool:
        """Emergency kill switch: FORCE_EXIT_FILE liquidates all broker positions."""
        return os.path.exists(FORCE_EXIT_FILE)

    def _alert(self, severity: str, message: str):
        """Send high-priority operational alerts without adding external deps."""
        severity = severity.upper()
        self._metric_inc('alerts', subkey=severity.lower())
        log_fn = logger.error if severity in {'CRITICAL', 'ERROR'} else logger.warning
        log_fn(f"ALERT[{severity}]: {message}")
        if not ALERT_WEBHOOK_URL:
            return
        payload = json.dumps({
            "severity": severity,
            "message": message,
            "ts": datetime.now(_TZ_NY).isoformat(),
            "mode": TRADING_MODE,
        }).encode("utf-8")
        req = urllib.request.Request(
            ALERT_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=ALERT_TIMEOUT_SEC):
                pass
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.warning(f"ALERT: webhook delivery failed: {e}")

    @staticmethod
    def _new_health_metrics() -> dict:
        return {
            'cycles': 0,
            'alerts': {},
            'ib_errors': 0,
            'ib_error_codes': {},
            'ib_disconnects': 0,
            'ib_sleep_disconnects': 0,
            'reconnect_successes': 0,
            'reconnect_failures': 0,
            'vix_ticker_misses': 0,
            'vix_ticker_failures': 0,
            'vix_fallback_successes': 0,
            'vix_fallback_failures': 0,
            'vix_retry_suppressed': 0,
            'vix_historical_no_bars': 0,
            'vix_historical_exceptions': 0,
            'spy_historical_no_bars': 0,
            'spy_historical_exceptions': 0,
            'historical_warmup_successes': 0,
            'historical_warmup_failures': 0,
            'account_summary_cancelled': 0,
            'account_summary_cancel_failures': 0,
            'scanner_runs': 0,
            'scanner_candidates': 0,
            'scanner_skipped_no_slots': 0,
            'scanner_skipped_friday_cutoff': 0,
            'protective_stop_confirmed': 0,
            'protective_stop_unconfirmed': 0,
            'protective_stop_rebuilt': 0,
            'protective_stop_rejected': 0,
        }

    def _ensure_health_metrics(self, now_ny: Optional[datetime] = None):
        now_ny = now_ny or datetime.now(_TZ_NY)
        today = now_ny.strftime('%Y-%m-%d')
        if not hasattr(self, '_health_metrics') or not isinstance(self._health_metrics, dict):
            self._health_metrics = self._new_health_metrics()
        if not hasattr(self, '_health_date') or not self._health_date:
            self._health_date = today
        if self._health_date != today:
            previous_date = self._health_date
            self._write_health_report(
                reason='date_rollover',
                now_ny=now_ny,
                report_date=previous_date,
                roll=False,
            )
            self._health_date = today
            self._health_metrics = self._new_health_metrics()

    def _metric_inc(self, key: str, amount: int = 1, subkey: Optional[str] = None):
        self._ensure_health_metrics()
        if subkey is None:
            current = self._health_metrics.get(key, 0)
            try:
                self._health_metrics[key] = int(current) + int(amount)
            except (TypeError, ValueError):
                self._health_metrics[key] = int(amount)
            return
        bucket = self._health_metrics.setdefault(key, {})
        bucket[str(subkey)] = int(bucket.get(str(subkey), 0)) + int(amount)

    def _write_health_report(
        self,
        reason: str = 'cycle',
        now_ny: Optional[datetime] = None,
        report_date: Optional[str] = None,
        roll: bool = True,
    ):
        """Persist one compact daily operations snapshot for paper/live review."""
        now_ny = now_ny or datetime.now(_TZ_NY)
        if roll:
            self._ensure_health_metrics(now_ny)
        report_date = report_date or getattr(self, '_health_date', now_ny.strftime('%Y-%m-%d'))
        try:
            connected = bool(self.ib.isConnected())
        except Exception:
            connected = False
        positions = []
        unconfirmed = []
        for sym, data in sorted((getattr(self, 'state', {}) or {}).items()):
            status = data.get('protection_status', 'unknown')
            if status != 'confirmed' and not data.get('pending_exit'):
                unconfirmed.append(sym)
            positions.append({
                'symbol': sym,
                'qty': float(data.get('qty', 0) or 0),
                'entry_price': float(data.get('fill_price', data.get('price', 0)) or 0),
                'current_price': float(data.get('current_price', data.get('price', 0)) or 0),
                'protection_status': status,
                'pending_exit': bool(data.get('pending_exit', False)),
            })
        payload = {
            'date': report_date,
            'generated_at': now_ny.isoformat(),
            'reason': reason,
            'trading_mode': TRADING_MODE,
            'connected': connected,
            'account': {
                'equity': round(float(getattr(self, '_last_equity', 0.0) or 0.0), 2),
                'settled_cash': round(float(getattr(self, '_last_settled_cash', 0.0) or 0.0), 2),
            },
            'regime': {
                'vix': (
                    round(float(self._last_vix), 2)
                    if getattr(self, '_last_vix', None) is not None else None
                ),
                'vix_threshold': VIX_THRESHOLD,
                'vix_source': getattr(self, '_last_vix_source', None),
                'vix_last_success_at': (
                    datetime.fromtimestamp(self._last_vix_ts, _TZ_NY).isoformat()
                    if getattr(self, '_last_vix_ts', 0.0) else None
                ),
                'vix_failure_count': int(getattr(self, '_vix_failure_count', 0) or 0),
                'vix_last_failure_at': (
                    datetime.fromtimestamp(self._last_vix_failure_ts, _TZ_NY).isoformat()
                    if getattr(self, '_last_vix_failure_ts', 0.0) else None
                ),
                'vix_next_retry_at': (
                    datetime.fromtimestamp(self._next_vix_retry_ts, _TZ_NY).isoformat()
                    if getattr(self, '_next_vix_retry_ts', 0.0) else None
                ),
            },
            'market_data': {
                'historical': copy.deepcopy(
                    getattr(self, '_historical_data_health', {}) or {}
                ),
            },
            'scanner': {
                'last_scan': getattr(self, '_last_scan_ts', None),
                'next_scan': getattr(self, '_next_scan_dt', None),
            },
            'risk': {
                'unconfirmed_protection_symbols': unconfirmed,
            },
            'positions': positions,
            'metrics': copy.deepcopy(getattr(self, '_health_metrics', {})),
        }
        tmp = HEALTH_REPORT_FILE + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(payload, f, indent=4, default=str)
            os.replace(tmp, HEALTH_REPORT_FILE)
        except OSError as e:
            logger.warning(f"Could not write health report: {e}")

    def _validate_deployment_mode(self):
        """Fail closed when config points at live trading without explicit acknowledgement."""
        mode = TRADING_MODE.strip().lower()
        if mode not in {"paper", "live"}:
            self._alert("CRITICAL", f"Invalid VELOCITY_TRADING_MODE={TRADING_MODE!r}; expected paper or live.")
            sys.exit()
        if MARKET_DATA_TYPE != 1:
            self._alert(
                "CRITICAL",
                f"MARKET_DATA_TYPE={MARKET_DATA_TYPE}; live entries require real-time market data type 1.",
            )
            sys.exit()
        if mode == "paper" and IB_PORT in LIVE_IB_PORTS:
            self._alert(
                "CRITICAL",
                f"Paper mode refuses to connect to live-looking IB port {IB_PORT}. "
                "Set VELOCITY_TRADING_MODE=live and explicit acknowledgement for live trading.",
            )
            sys.exit()
        if mode == "live" and IB_PORT in PAPER_IB_PORTS:
            self._alert(
                "CRITICAL",
                f"Live mode refuses to connect to paper-looking IB port {IB_PORT}. "
                "Use the live IB Gateway/TWS port before enabling live trading.",
            )
            sys.exit()
        if mode == "live" and LIVE_TRADING_ACK != LIVE_TRADING_ACK_PHRASE:
            self._alert(
                "CRITICAL",
                "Live trading blocked: set VELOCITY_LIVE_TRADING_ACK="
                f"{LIVE_TRADING_ACK_PHRASE} only after paper validation.",
            )
            sys.exit()

    @staticmethod
    def _coerce_positive_price(value) -> Optional[float]:
        """Return a finite positive float price, or None for unusable broker values."""
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return None
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        if np.isfinite(price) and price > 0:
            return price
        return None

    @staticmethod
    def _coerce_order_number(value) -> Optional[float]:
        """Return a usable broker order number, ignoring IB's UNSET_DOUBLE sentinel."""
        number = VelocityEngine._coerce_positive_price(value)
        if number is None:
            return None
        if number >= util.UNSET_DOUBLE * 0.5:
            return None
        return number

    @staticmethod
    def _trail_order_protection(order, pos_data: dict) -> Tuple[bool, str, float, float]:
        """Interpret IB TRAIL SELL fields as (valid, display, stop_dist, stop_loss).

        IB can return a TRAIL order either as a dollar trail in auxPrice or as a
        percent trail with auxPrice left at UNSET_DOUBLE. Treating that sentinel
        as a dollar amount makes the dashboard/audit logic think the stop is
        absurdly wide even when the real trailStopPrice/trailingPercent is valid.
        """
        ref_price = (
            VelocityEngine._coerce_positive_price(pos_data.get('current_price'))
            or VelocityEngine._coerce_positive_price(pos_data.get('peak_price'))
            or VelocityEngine._coerce_positive_price(pos_data.get('price'))
            or VelocityEngine._coerce_positive_price(pos_data.get('entry_price'))
        )
        aux_dist = VelocityEngine._coerce_order_number(getattr(order, 'auxPrice', None))
        trail_stop = VelocityEngine._coerce_order_number(getattr(order, 'trailStopPrice', None))
        trail_pct = VelocityEngine._coerce_order_number(getattr(order, 'trailingPercent', None))

        if aux_dist is not None:
            if ref_price is not None and aux_dist >= ref_price:
                return False, f"dollar trail ${aux_dist:.2f} >= reference price ${ref_price:.2f}", 0.0, 0.0
            stop_loss = trail_stop or (ref_price - aux_dist if ref_price is not None else 0.0)
            return True, f"dist=${aux_dist:.2f}", aux_dist, max(stop_loss, 0.0)

        if trail_pct is not None and trail_stop is not None:
            if trail_pct >= 99.0:
                return False, f"trailing percent {trail_pct:.4g}% is unusable", 0.0, 0.0
            if ref_price is not None:
                if trail_stop >= ref_price:
                    return False, f"trail stop ${trail_stop:.2f} >= reference price ${ref_price:.2f}", 0.0, 0.0
                stop_dist = ref_price - trail_stop
            else:
                stop_dist = trail_stop * (trail_pct / max(100.0 - trail_pct, 1e-9))
            return True, f"stop=${trail_stop:.2f} trail={trail_pct:.4g}%", stop_dist, trail_stop

        return False, "missing dollar trail or percent trail fields", 0.0, 0.0

    @staticmethod
    def _round_up_to_cent(value: float) -> float:
        """Round up to the nearest US equity penny tick."""
        return round(float(np.ceil(float(value) * 100.0 - 1e-9) / 100.0), 2)

    @staticmethod
    def _round_down_to_cent(value: float) -> float:
        """Round down to the nearest US equity penny tick."""
        return round(float(np.floor(float(value) * 100.0 + 1e-9) / 100.0), 2)

    @staticmethod
    def _calc_entry_limit_price(price, bid, ask) -> Optional[float]:
        """Return a marketable, spread-aware BUY limit price, or None.

        MOST_ACTIVE names are usually liquid, so the parent BUY should key off
        the real ask instead of blindly adding a fixed percentage to the last or
        midpoint. The old 0.2% cushion remains as a hard max over the validated
        reference price, while the working limit is ask plus a small tick/cushion.
        """
        ref_price = VelocityEngine._coerce_positive_price(price)
        bid_price = VelocityEngine._coerce_positive_price(bid)
        ask_price = VelocityEngine._coerce_positive_price(ask)
        if ref_price is None or bid_price is None or ask_price is None:
            return None
        if ask_price <= bid_price:
            return None

        mid = (bid_price + ask_price) / 2.0
        if mid <= 0:
            return None
        spread_pct = (ask_price - bid_price) / mid
        if spread_pct > SPREAD_MAX_PCT:
            return None

        max_limit = ref_price * (1.0 + max(0.0, ENTRY_LIMIT_MAX_OVER_MARKET_PCT))
        if ask_price > max_limit:
            return None

        cushion = max(ENTRY_LIMIT_MIN_TICK, ask_price * max(0.0, ENTRY_LIMIT_ASK_CUSHION_PCT))
        raw_limit = min(ask_price + cushion, max_limit)
        limit = round(raw_limit, 2)
        if limit < ask_price:
            limit = VelocityEngine._round_up_to_cent(ask_price)
        if limit > max_limit:
            limit = VelocityEngine._round_down_to_cent(max_limit)
        if limit < ask_price:
            return None
        return limit

    @staticmethod
    def _account_currency_matches(item) -> bool:
        """Ignore non-USD account summary rows in multi-currency IBKR accounts."""
        currency = getattr(item, 'currency', '')
        if not isinstance(currency, str):
            currency = ''
        return currency in ('', 'BASE', ACCOUNT_CURRENCY)

    def _stock_contract(self, sym: str):
        if sym not in self._contract_cache:
            contract = Stock(sym, 'SMART', 'USD')
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                raise RuntimeError(f"IBKR could not qualify contract for {sym}")
            self._contract_cache[sym] = qualified[0]
        return self._contract_cache[sym]

    def _fresh_market_price(self, sym: str) -> Optional[float]:
        """Fetch a fresh market price from IBKR for exit/risk decisions."""
        try:
            contract = self._stock_contract(sym)
            tickers = self.ib.reqTickers(contract)
            if not tickers:
                return None
            ticker = tickers[0]
            for candidate in (
                ticker.marketPrice(),
                getattr(ticker, 'last', None),
            ):
                price = self._coerce_positive_price(candidate)
                if price is not None:
                    return price
            bid = self._coerce_positive_price(getattr(ticker, 'bid', None))
            ask = self._coerce_positive_price(getattr(ticker, 'ask', None))
            if bid is not None and ask is not None and ask >= bid:
                return (bid + ask) / 2.0
            if bid is not None:
                return bid
        except Exception as e:
            logger.warning(f"PRICE {sym}: fresh market price unavailable ({e})")
        return None

    # ── State persistence ──────────────────────────────────────────────────────
    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                if isinstance(state, dict):
                    return state
                raise ValueError(f"state root must be dict, got {type(state).__name__}")
            except (json.JSONDecodeError, OSError) as e:
                backup = f"{STATE_FILE}.corrupt.{datetime.now(_TZ_NY).strftime('%Y%m%d_%H%M%S')}"
                try:
                    shutil.copy2(STATE_FILE, backup)
                    self._alert(
                        "CRITICAL",
                        f"STATE: Could not load state file ({e}). Backed up to {backup}; starting empty.",
                    )
                except OSError as backup_e:
                    self._alert(
                        "CRITICAL",
                        f"STATE: Could not load state file ({e}) and backup failed ({backup_e}); starting empty.",
                    )
            except ValueError as e:
                backup = f"{STATE_FILE}.invalid.{datetime.now(_TZ_NY).strftime('%Y%m%d_%H%M%S')}"
                try:
                    shutil.copy2(STATE_FILE, backup)
                    self._alert("CRITICAL", f"STATE: Invalid state file ({e}). Backed up to {backup}; starting empty.")
                except OSError as backup_e:
                    self._alert("CRITICAL", f"STATE: Invalid state file ({e}) and backup failed ({backup_e}); starting empty.")
        return {}

    def save_state(self):
        tmp = STATE_FILE + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(self.state, f, indent=4)
            os.replace(tmp, STATE_FILE)
        except OSError as e:
            self._alert("CRITICAL", f"STATE: failed to persist engine state: {e}")
            raise

    # ── Dashboard data writer ─────────────────────────────────────────────────
    def _write_dashboard_data(self, connected: bool = True):
        """Write engine metrics to dashboard_data.json for the web dashboard."""
        now = datetime.now(_TZ_NY)
        data = {
            "equity":       self._last_equity,
            "settled_cash": self._last_settled_cash,
            "vix":          self._last_vix,
            "connected":    connected,
            "last_scan":    self._last_scan_ts,
            "next_scan":    self._next_scan_dt,
            "last_updated": now.isoformat(),
        }
        try:
            with open(DASHBOARD_FILE, 'w') as f:
                json.dump(data, f, indent=4)
        except OSError as e:
            logger.warning(f"Could not write dashboard data: {e}")
        self._write_health_report(reason='dashboard_update', now_ny=now)

        # Append equity snapshot to rolling history (kept for 60 days)
        # Skip until the first real IBKR accountSummary() reading arrives.
        if self._last_equity > 0 and self._equity_initialized:
            try:
                cutoff = now.timestamp() - 60 * 86400   # 60 days ago
                try:
                    with open(EQUITY_HIST_FILE, 'r') as f:
                        history = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    history = []
                history = [e for e in history if e.get('ts_epoch', 0) >= cutoff]
                history.append({"ts": now.isoformat(), "ts_epoch": now.timestamp(),
                                 "eq": round(self._last_equity, 2)})
                with open(EQUITY_HIST_FILE, 'w') as f:
                    json.dump(history, f)
            except OSError as e:
                logger.warning(f"Could not write equity history: {e}")

    def _write_readiness_snapshot(self, snapshot: dict):
        """Atomically write non-trading operational readiness data."""
        tmp = READINESS_FILE + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(snapshot, f, indent=4, default=str)
            os.replace(tmp, READINESS_FILE)
        except OSError as e:
            logger.warning(f"Could not write readiness snapshot: {e}")

    # ── Account ────────────────────────────────────────────────────────────────
    def _request_account_summary_snapshot(self):
        """Request account summary as a bounded one-shot and cancel the stream.

        ib_async.accountSummary() is convenient, but if Gateway disconnects while
        a request is open, IBKR can keep counting it against the account-summary
        subscription limit. For real IB instances we use the lower-level request
        API and always send cancelAccountSummary(reqId) in finally.
        """
        if not isinstance(self.ib, IB):
            return self.ib.accountSummary()

        client = getattr(self.ib, 'client', None)
        wrapper = getattr(self.ib, 'wrapper', None)
        if client is None or wrapper is None:
            return self.ib.accountSummary()

        req_id = client.getReqId()
        future = wrapper.startReq(req_id)
        tags = (
            "AccountType,NetLiquidation,TotalCashValue,SettledCash,"
            "AvailableFunds,ExcessLiquidity,BuyingPower,$LEDGER:ALL"
        )
        try:
            try:
                wrapper.acctSummary.clear()
            except Exception:
                pass
            client.reqAccountSummary(req_id, "All", tags)
            self.ib._run(future)
            return list(wrapper.acctSummary.values())
        finally:
            try:
                client.cancelAccountSummary(req_id)
                self._metric_inc('account_summary_cancelled')
            except Exception as e:
                self._metric_inc('account_summary_cancel_failures')
                logger.warning(f"ACCOUNT: cancelAccountSummary({req_id}) failed: {e}")

    def _get_account_values(self) -> Tuple[float, float]:
        """Return (net_liquidation, settled_cash) using fresh IBKR data only.

        For a cash account, AvailableFunds is not a safe substitute for
        SettledCash. If SettledCash is missing or non-positive, new entries get
        zero buying cash and naturally fail closed.
        """
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                summary = self._request_account_summary_snapshot()
                if summary:
                    net_liq = settled = 0.0
                    settled_seen = False
                    for item in summary:
                        if not self._account_currency_matches(item):
                            continue
                        if item.tag == 'NetLiquidation':
                            net_liq = float(item.value)
                        elif item.tag == 'SettledCash':
                            settled_seen = True
                            settled = float(item.value)
                    if net_liq > 0:
                        if not settled_seen:
                            logger.warning(
                                "ACCOUNT: SettledCash tag missing — treating settled cash as $0.00 "
                                "for cash-account safety"
                            )
                        elif settled <= 0:
                            logger.warning(
                                f"ACCOUNT: SettledCash=${settled:.2f} <= 0 — "
                                "blocking new entries until cash settles"
                            )
                        return net_liq, max(settled, 0.0) if settled_seen else 0.0
                logger.warning(
                    f"ACCOUNT: accountSummary attempt {attempt}/{max_attempts} "
                    "returned no usable data"
                )
            except Exception as e:
                logger.warning(
                    f"ACCOUNT: accountSummary attempt {attempt}/{max_attempts} failed: {e}"
                )
            if attempt < max_attempts:
                self.ib.sleep(2)

        raise AccountDataUnavailable(
            f"ACCOUNT: all {max_attempts} accountSummary attempts failed; "
            "fresh settled cash is unavailable."
        )

    # ── Startup gate ──────────────────────────────────────────────────────────
    def _fetch_equity_with_retry(self) -> float:
        """Poll IBKR for NetLiquidation until a positive value is returned. Never gives up."""
        attempt = 0
        while True:
            attempt += 1
            try:
                summary = self._request_account_summary_snapshot()
                if summary:
                    for item in summary:
                        if not self._account_currency_matches(item):
                            continue
                        if item.tag == 'NetLiquidation':
                            val = float(item.value)
                            if val > 0:
                                logger.info(f"INIT: NetLiquidation=${val:.2f} (attempt {attempt})")
                                return val
                logger.warning(
                    f"INIT: Equity attempt {attempt}: NetLiquidation missing or ≤0, "
                    f"retrying in {EQUITY_RETRY_INTERVAL}s..."
                )
            except Exception as e:
                logger.warning(
                    f"INIT: Equity attempt {attempt} exception: {e}, "
                    f"retrying in {EQUITY_RETRY_INTERVAL}s..."
                )
            self.ib.sleep(EQUITY_RETRY_INTERVAL)

    def _log_startup_summary(self, equity: float):
        """Log per-position table and capital totals at startup."""
        if not self.state:
            logger.info("INIT: No open positions. Full capital available.")
            logger.info(
                f"INIT READY | Equity=${equity:.2f} | Cash≈${equity:.2f} | Positions=0/{self._calc_max_positions(equity)}"
            )
            return
        total_cost       = 0.0
        total_unrealized = 0.0
        logger.info("INIT: ── Open Positions ──────────────────────────────────")
        for sym, d in self.state.items():
            ep      = float(d.get('price', 0))
            qty     = float(d.get('qty', 0))
            cur     = float(d.get('current_price', ep))
            sl      = float(d.get('stop_loss', 0))
            unreal  = float(d.get('unrealized_pnl', (cur - ep) * qty))
            unreal_pct = float(d.get('unrealized_pnl_pct', (cur - ep) / ep * 100 if ep else 0))
            cost    = ep * qty
            total_cost       += cost
            total_unrealized += unreal
            logger.info(
                f"INIT:  {sym:6s} | entry=${ep:.2f} cur=${cur:.2f} qty={qty:.4g} "
                f"cost=${cost:.2f} | unreal={unreal:+.2f} ({unreal_pct:+.1f}%) "
                f"| SL=${sl:.2f}"
            )
        cash_approx = equity - total_cost
        logger.info("INIT: ────────────────────────────────────────────────────")
        logger.info(
            f"INIT READY | Equity=${equity:.2f} | Invested≈${total_cost:.2f} "
            f"| Cash≈${cash_approx:.2f} | Unrealized={total_unrealized:+.2f} "
            f"| Positions={len(self.state)}/{self._calc_max_positions(equity)}"
        )

    def _initialize(self):
        """
        Startup safety gate.

        Phase 1 (immediate): fetch equity, cancel orphaned orders, sync
        positions, audit protective stops, refresh prices, and write dashboard
        so the UI is live straight away.

        Phase 2 (timed): sleep until PRE_ENTRY_SYNC_TIME (09:15 ET) so the
        definitive position re-sync and chandelier stop audit happen with the
        morning IBKR position/order snapshot before the 10:00 AM entry window opens.
        If the engine starts after the sync time (intraday restart, evening),
        Phase 2 runs immediately with no sleep.

        The run loop performs a separate post-open audit at POST_OPEN_AUDIT_TIME
        after TRAIL orders are active and opening-auction broker state has settled.
        """
        # ── Phase 1: immediate startup snapshot ──────────────────────────────
        logger.info("INIT: Waiting for account equity from IBKR...")
        equity = self._fetch_equity_with_retry()
        self._last_equity        = equity
        self._last_settled_cash  = 0.0      # exact settled cash is fetched in run_cycle()
        self._equity_initialized = True
        max_pos     = self._calc_max_positions(equity)
        open_slots  = max(0, max_pos - len(self.state))
        bucket_size = 0.0
        logger.info(
            f"INIT: Equity=${equity:.2f} | EntrySlots {open_slots}/{max_pos} | "
            f"Bucket≈${bucket_size:.2f} until SettledCash is fetched in the first cycle"
        )

        # Cancel orphaned pending orders from a previous engine session.
        # "Orphaned" = a symbol whose order is not attached to local state and
        # has no live IBKR position.  We must NOT cancel active-position stop
        # orders — those are audited in Phase 2.
        open_orders = self.ib.reqAllOpenOrders()
        ibkr_symbols = {
            p.contract.symbol for p in self.ib.positions()
            if float(getattr(p, 'position', 0) or 0) > 0
        }
        orphaned = [
            t for t in open_orders
            if str(getattr(t.order, 'action', '')).upper() in {'BUY', 'SELL'}
            and t.contract.symbol not in self.state
            and t.contract.symbol not in ibkr_symbols
        ]
        if orphaned:
            logger.info(
                f"INIT: Cancelling {len(orphaned)} orphaned orders "
                f"({len(open_orders) - len(orphaned)} active-position orders preserved)."
            )
            for trade in orphaned:
                try:
                    self.ib.cancelOrder(trade.order)
                except Exception as e:
                    logger.warning(
                        f"INIT: failed to cancel orphaned "
                        f"{getattr(trade.order, 'action', '?')} order for "
                        f"{getattr(trade.contract, 'symbol', '?')}: {e}"
                    )
            self._safe_sleep(2, context="INIT orphan-cancel settle")

        logger.info("INIT: Phase 1 — immediate position sync and stop-order audit...")
        self._sync_positions_from_ibkr()
        if self.state:
            logger.info("INIT: Auditing stop orders for open positions immediately...")
            self._audit_stop_orders()
            now_ny = datetime.now(_TZ_NY)
            self._last_audit_date = now_ny.strftime('%Y-%m-%d')
            self._last_audit_at = now_ny
            logger.info("INIT: Fetching live prices for open positions...")
            self._update_position_prices()
        self._write_dashboard_data(connected=True)

        # ── Phase 2: pre-entry definitive sync + stop audit ──────────────────
        self._wait_for_pre_entry_sync()

        logger.info("INIT: Phase 2 — pre-entry re-sync and stop-order audit...")
        self._sync_positions_from_ibkr()
        if self.state:
            logger.info("INIT: Auditing stop orders for open positions...")
            self._audit_stop_orders()
            now_ny = datetime.now(_TZ_NY)
            self._last_audit_date = now_ny.strftime('%Y-%m-%d')
            self._last_audit_at = now_ny
            logger.info("INIT: Fetching live prices for open positions...")
            self._update_position_prices()

        self._log_startup_summary(equity)
        self._write_dashboard_data(connected=True)

    def _wait_for_pre_entry_sync(self):
        """
        Sleep until PRE_ENTRY_SYNC_TIME (default 09:15 ET) so the definitive
        position re-sync and stop audit run before the 10:00 AM entry window.

        If the engine starts after the sync time (intraday restart, evening run),
        the method returns immediately without sleeping.
        """
        h, m    = PRE_ENTRY_SYNC_TIME
        now_ny  = datetime.now(_TZ_NY)
        target  = now_ny.replace(hour=h, minute=m, second=0, microsecond=0)

        if now_ny >= target:
            logger.info(
                f"INIT: Already at or past {h:02d}:{m:02d} ET — "
                f"running pre-entry sync immediately."
            )
            return

        wait_s = (target - now_ny).total_seconds()
        logger.info(
            f"INIT: Waiting {wait_s / 60:.1f} min until {h:02d}:{m:02d} ET "
            f"for pre-entry position sync & stop audit "
            f"(entry window opens at {ENTRY_START[0]:02d}:{ENTRY_START[1]:02d} ET)."
        )
        self._safe_sleep(wait_s, context="INIT pre-entry wait")

    def _entry_good_after_time(self) -> str:
        """Return a 10:00 ET activation string only while that time is still future."""
        now_ny = datetime.now(_TZ_NY)
        entry_gate = now_ny.replace(
            hour=ENTRY_START[0],
            minute=ENTRY_START[1],
            second=0,
            microsecond=0,
        )
        if now_ny >= entry_gate:
            return ""
        return (
            f"{now_ny.strftime('%Y%m%d')} "
            f"{ENTRY_START[0]:02d}:{ENTRY_START[1]:02d}:00 US/Eastern"
        )

    def _stop_good_after_time(self) -> str:
        """Return the protective-stop activation time while it is still future."""
        now_ny = datetime.now(_TZ_NY)
        stop_gate = now_ny.replace(
            hour=STOP_ACTIVATION_TIME[0],
            minute=STOP_ACTIVATION_TIME[1],
            second=0,
            microsecond=0,
        )
        if now_ny >= stop_gate:
            return ""
        return (
            f"{now_ny.strftime('%Y%m%d')} "
            f"{STOP_ACTIVATION_TIME[0]:02d}:{STOP_ACTIVATION_TIME[1]:02d}:00 US/Eastern"
        )

    def _make_trail_replacement_order(self, source_order, qty: float, gate_str: str) -> Optional[Order]:
        """Build an equivalent TRAIL SELL order, optionally delayed to the stop gate."""
        replacement = Order()
        replacement.action        = 'SELL'
        replacement.orderType     = 'TRAIL'
        replacement.totalQuantity = qty
        replacement.tif           = 'GTC'
        replacement.goodAfterTime = gate_str
        replacement.transmit      = True

        aux_dist = self._coerce_order_number(getattr(source_order, 'auxPrice', None))
        trail_stop = self._coerce_order_number(getattr(source_order, 'trailStopPrice', None))
        trail_pct = self._coerce_order_number(getattr(source_order, 'trailingPercent', None))
        if aux_dist is not None:
            replacement.auxPrice = aux_dist
            return replacement
        if trail_stop is not None and trail_pct is not None:
            replacement.trailStopPrice = trail_stop
            replacement.trailingPercent = trail_pct
            return replacement
        return None

    def _replace_trail_with_stop_activation_gate(self, trade, sym: str, qty: float, gate_str: str) -> Tuple[object, bool]:
        """Replace an existing TRAIL order so it cannot activate before the stop gate.

        New pre-market/audit TRAIL orders already set goodAfterTime. Existing
        GTC orders recovered from IBKR may not have it. Orders obtained through
        reqAllOpenOrders() are not always modifiable in place by a fresh API
        session, so use cancel-and-replace before the regular session opens.
        """
        order = trade.order
        current_gat = str(getattr(order, 'goodAfterTime', '') or '').strip()
        if not gate_str and not current_gat:
            return trade, True
        if current_gat == gate_str:
            return trade, True

        try:
            order_client_id = int(getattr(order, 'clientId', IB_CLIENT_ID) or IB_CLIENT_ID)
        except (TypeError, ValueError):
            order_client_id = IB_CLIENT_ID
        if order_client_id != IB_CLIENT_ID:
            self._alert(
                "CRITICAL",
                f"AUDIT: {sym} — existing TRAIL order belongs to IB clientId={order_client_id}, "
                f"but this engine is clientId={IB_CLIENT_ID}; configure the engine with the "
                "same clientId before it can modify/cancel that protective order.",
            )
            return trade, False

        replacement = self._make_trail_replacement_order(order, qty, gate_str)
        if replacement is None:
            self._alert(
                "CRITICAL",
                f"AUDIT: {sym} — existing TRAIL stop lacks a usable trail amount; "
                "cannot rebuild it with the stop activation gate.",
            )
            return trade, False

        logger.warning(
            f"AUDIT: {sym} — existing TRAIL SELL has stale/missing stop activation gate; "
            f"cancelling and replacing with goodAfterTime={gate_str or '<active now>'}"
        )
        try:
            self.ib.cancelOrder(order)
            self.ib.sleep(2)
        except Exception as e:
            self._alert(
                "CRITICAL",
                f"AUDIT: {sym} — failed to cancel pre-gate TRAIL stop before replacement: {e}. "
                "Existing broker order may activate before the configured stop gate.",
            )
            return trade, False

        try:
            replacement_trade = self.ib.placeOrder(trade.contract, replacement)
            self.ib.sleep(2)
            status = getattr(replacement_trade.orderStatus, 'status', '')
            if status in _REJECTED_ORDER_STATUSES:
                self._alert(
                    "CRITICAL",
                    f"AUDIT: {sym} — replacement TRAIL stop with stop gate rejected "
                    f"(status={status}); attempting to restore original protection.",
                )
                restored = self._restore_trail_without_entry_gate(trade, sym, qty)
                return restored, False
            logger.info(
                f"AUDIT: {sym} — TRAIL SELL replaced with stop activation gate "
                f"(new_id={replacement_trade.order.orderId})"
            )
            return replacement_trade, True
        except Exception as e:
            self._alert(
                "CRITICAL",
                f"AUDIT: {sym} — failed to place replacement TRAIL stop with stop gate: {e}; "
                "attempting to restore original protection.",
            )
            restored = self._restore_trail_without_entry_gate(trade, sym, qty)
            return restored, False

    def _restore_trail_without_entry_gate(self, trade, sym: str, qty: float):
        """Best-effort restoration if a gated replacement fails after cancellation."""
        restore = self._make_trail_replacement_order(trade.order, qty, '')
        if restore is None:
            self._alert("CRITICAL", f"AUDIT: {sym} — cannot restore original TRAIL stop; position may be unprotected.")
            return trade
        try:
            restored_trade = self.ib.placeOrder(trade.contract, restore)
            self.ib.sleep(2)
            status = getattr(restored_trade.orderStatus, 'status', '')
            if status in _REJECTED_ORDER_STATUSES:
                self._alert(
                    "CRITICAL",
                    f"AUDIT: {sym} — original TRAIL stop restore rejected (status={status}); "
                    "position may be unprotected.",
                )
                return trade
            logger.warning(
                f"AUDIT: {sym} — restored original TRAIL protection without stop activation gate "
                f"(id={restored_trade.order.orderId})"
            )
            return restored_trade
        except Exception as e:
            self._alert("CRITICAL", f"AUDIT: {sym} — original TRAIL stop restore failed: {e}; position may be unprotected.")
            return trade

    def _preflight_order(
        self,
        contract,
        order,
        sym: str,
        *,
        allow_protective_sell_fail_open: bool = False,
    ) -> bool:
        """
        Call IBKR's whatIf API to verify an order won't be rejected before placing it.

        The live order is not mutated.  IBKR requires what-if orders to be
        transmitted, so bracket parents that are deliberately created with
        transmit=False are preflighted through a copied order with transmit=True.
        The canonical rejection signal is OrderState.warningText being non-empty
        — IBKR sets this for every detectable problem: insufficient buying
        power, invalid parameters, unsupported order type, market-hours
        violations, etc.

        Returns True  → order appears acceptable; caller may call placeOrder().
        Returns False → IBKR flagged a problem; caller should skip this order.

        On whatIf exceptions, fail closed by default. The only exception is an
        already-open position stop audit where failing to submit a protective
        SELL can leave inventory completely unprotected.
        """
        try:
            preflight_order = copy.copy(order)
            preflight_order.whatIf = True
            preflight_order.transmit = True
            state = self.ib.whatIfOrder(contract, preflight_order)
            if isinstance(state, list):
                state = state[0] if state else None
            if state is None:
                raise ValueError("whatIfOrder returned empty result")
            warning = (state.warningText or '').strip()
            if warning:
                logger.warning(
                    f"PREFLIGHT {sym} [{order.action} {order.orderType}]: {warning}"
                )
                return False
            return True
        except Exception as e:
            logger.warning(
                f"PREFLIGHT {sym}: whatIf check failed ({e})"
            )
            return (
                allow_protective_sell_fail_open
                and getattr(order, 'action', '') == 'SELL'
                and sym in self.state
            )

    def _mark_position_protection(
        self,
        sym: str,
        status: str,
        reason: str = '',
        order_id: Optional[int] = None,
    ):
        if sym not in self.state:
            return
        self.state[sym]['protection_status'] = status
        self.state[sym]['protection_checked_at'] = datetime.now(_TZ_NY).isoformat()
        if reason:
            self.state[sym]['protection_reason'] = reason
        if status == 'confirmed':
            self.state[sym]['protected_at'] = datetime.now(_TZ_NY).isoformat()
            self.state[sym].pop('protection_reason', None)
            self._metric_inc('protective_stop_confirmed')
        elif status == 'unconfirmed':
            self._metric_inc('protective_stop_unconfirmed')
        try:
            clean_order_id = int(order_id) if order_id is not None else None
        except (TypeError, ValueError):
            clean_order_id = None
        if clean_order_id is not None and clean_order_id > 0:
            self.state[sym]['stop_order_id'] = clean_order_id
        self.save_state()

    def _trade_is_matching_trail_sell(
        self,
        trade,
        sym: str,
        qty: float,
        *,
        expected_order=None,
    ) -> bool:
        """Return True when a trade/order can protect the full long quantity."""
        order = expected_order if expected_order is not None else getattr(trade, 'order', None)
        if order is None:
            return False

        status = getattr(getattr(trade, 'orderStatus', None), 'status', '')
        if isinstance(status, str) and status in _REJECTED_ORDER_STATUSES:
            return False

        action = str(getattr(order, 'action', '') or '').upper()
        order_type = str(getattr(order, 'orderType', '') or '').upper()
        if action != 'SELL' or order_type != 'TRAIL':
            return False

        if expected_order is None:
            contract = getattr(trade, 'contract', None)
            trade_sym = str(getattr(contract, 'symbol', '') or '').upper()
            if trade_sym and trade_sym != sym.upper():
                return False

        try:
            order_qty = float(getattr(order, 'totalQuantity', 0) or 0)
        except (TypeError, ValueError):
            return False
        try:
            expected_qty = float(qty)
        except (TypeError, ValueError):
            return False
        if abs(order_qty - expected_qty) > 1e-6:
            return False

        pos_data = self.state.get(sym, {'qty': qty})
        trail_ok, _, _, _ = self._trail_order_protection(order, pos_data)
        return bool(trail_ok)

    def _confirm_protective_stop(
        self,
        sym: str,
        qty: float,
        *,
        known_trade=None,
        expected_order=None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Wait briefly until a valid TRAIL SELL is visible for a filled BUY.

        A cash-account BUY is sent first, then the protective SELL is submitted
        after the fill. This confirmation gate prevents the engine from adding
        more exposure when the first position may still be unprotected.
        """
        timeout = (
            PROTECTIVE_STOP_CONFIRM_TIMEOUT_SEC
            if timeout is None else max(float(timeout), 0.0)
        )
        poll_s = max(float(PROTECTIVE_STOP_CONFIRM_POLL_SEC), 0.1)

        if known_trade is not None and self._trade_is_matching_trail_sell(
            known_trade,
            sym,
            qty,
            expected_order=expected_order,
        ):
            return True

        deadline = time.monotonic() + timeout
        while True:
            try:
                for trade in self._open_trades_for_audit():
                    if self._trade_is_matching_trail_sell(trade, sym, qty):
                        return True
            except Exception as e:
                logger.warning(f"STOP CONFIRM {sym}: open-order check failed: {e}")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self.ib.sleep(min(poll_s, remaining))

    def _audit_stop_orders(self):
        """
        For every open position ensure exactly one chandelier TRAIL SELL order exists.

        Steps per symbol:
        1. Find all open SELL orders for the symbol.
        2. Cancel any that are NOT order type TRAIL (stale LMT take-profits, STP, etc.).
        3. If no TRAIL SELL remains after cancellations, fetch ATR(22) from daily bars
           and place a new chandelier TRAIL SELL (GTC, transmit=True).

        The entry trading window (10:00–15:30 ET) applies only to new BUY entries.
        Stop orders for existing positions are placed immediately regardless of time,
        but when the audit runs before the configured stop gate their activation is
        delayed with goodAfterTime. After placeOrder() we wait 2 s and verify the
        order status so any unexpected rejection is caught and logged; state is only
        updated once IB confirms the order is live (PreSubmitted / Submitted).
        """
        if not self.state:
            return

        # Log whether we are placing in or outside regular market hours so the
        # operator can correlate any IB rejection messages in the log.
        now_ny    = datetime.now(_TZ_NY)
        in_market = (
            now_ny.weekday() < 5
            and (9, 30) <= (now_ny.hour, now_ny.minute) <= (16, 0)
        )
        if not in_market:
            logger.info(
                "AUDIT: market is currently closed — TRAIL stops will be submitted "
                "as GTC orders and delayed until the configured stop gate when applicable."
            )

        open_trades = self._open_trades_for_audit()

        sell_by_sym: Dict[str, list] = {}
        for t in open_trades:
            if t.order.action == 'SELL':
                sell_by_sym.setdefault(t.contract.symbol, []).append(t)

        for sym, pos_data in list(self.state.items()):
            if getattr(self, '_missing_position_counts', {}).get(sym, 0) > 0:
                logger.warning(
                    f"AUDIT: {sym} skipped — IBKR position was missing in the latest "
                    "snapshot; avoiding orphan protective order until confirmed."
                )
                continue

            qty = float(pos_data.get('qty', 0))
            if qty <= 0:
                logger.warning(f"AUDIT: {sym} — qty={qty}, cannot place stop")
                continue

            sell_orders  = sell_by_sym.get(sym, [])
            raw_trail_orders = [
                t for t in sell_orders
                if t.order.orderType == 'TRAIL'
                and getattr(t.orderStatus, 'status', '') not in _REJECTED_ORDER_STATUSES
            ]
            non_trail    = [t for t in sell_orders if t.order.orderType != 'TRAIL']
            trail_orders = []
            trail_meta = {}
            mismatched_trail = []
            for t in raw_trail_orders:
                try:
                    order_qty = float(getattr(t.order, 'totalQuantity', 0) or 0)
                except (TypeError, ValueError):
                    order_qty = 0.0
                trail_ok, trail_desc, trail_dist, trail_stop = self._trail_order_protection(t.order, pos_data)
                if abs(order_qty - qty) <= 1e-6 and trail_ok:
                    aux_dist = self._coerce_order_number(getattr(t.order, 'auxPrice', None))
                    trail_pct = self._coerce_order_number(getattr(t.order, 'trailingPercent', None))
                    trail_meta[id(t)] = {
                        'desc': trail_desc,
                        'stop_dist': trail_dist,
                        'stop_loss': trail_stop,
                        'trail_pct': trail_pct if aux_dist is None else None,
                    }
                    trail_orders.append(t)
                else:
                    trail_meta[id(t)] = {'desc': trail_desc}
                    mismatched_trail.append(t)

            for t in non_trail:
                logger.info(
                    f"AUDIT: {sym} — cancelling non-TRAIL {t.order.orderType} SELL "
                    f"(id={t.order.orderId})"
                )
                try:
                    self.ib.cancelOrder(t.order)
                except Exception as e:
                    logger.warning(f"AUDIT: {sym} — cancel failed: {e}")
            for t in mismatched_trail:
                logger.warning(
                    f"AUDIT: {sym} — TRAIL SELL invalid "
                    f"(order_qty={getattr(t.order, 'totalQuantity', None)} "
                    f"state_qty={qty:g}, {trail_meta.get(id(t), {}).get('desc', 'unknown reason')}); "
                    f"cancelling and rebuilding"
                )
                try:
                    self.ib.cancelOrder(t.order)
                except Exception as e:
                    logger.warning(f"AUDIT: {sym} — mismatched TRAIL cancel failed: {e}")
            if non_trail or mismatched_trail:
                self.ib.sleep(1)

            if len(trail_orders) > 1:
                for t in trail_orders[1:]:
                    logger.warning(
                        f"AUDIT: {sym} — duplicate TRAIL SELL found "
                        f"(id={t.order.orderId}); cancelling extra protective order"
                    )
                    try:
                        self.ib.cancelOrder(t.order)
                    except Exception as e:
                        logger.warning(f"AUDIT: {sym} — duplicate cancel failed: {e}")

            if trail_orders:
                primary_trail = trail_orders[0]
                meta = trail_meta.get(id(primary_trail), {})
                primary_trail, gate_ok = self._replace_trail_with_stop_activation_gate(
                    primary_trail,
                    sym,
                    qty,
                    self._stop_good_after_time(),
                )
                stop_dist = float(meta.get('stop_dist') or 0.0)
                stop_loss = float(meta.get('stop_loss') or 0.0)
                if stop_dist > 0:
                    self.state[sym]['stop_dist'] = round(stop_dist, 2)
                if stop_loss > 0:
                    self.state[sym]['stop_loss'] = round(stop_loss, 2)
                    self.state[sym]['effective_stop'] = round(stop_loss, 2)
                trail_pct = meta.get('trail_pct')
                if trail_pct:
                    self.state[sym]['stop_mode'] = 'percent'
                    self.state[sym]['trailing_percent'] = round(float(trail_pct), 4)
                else:
                    self.state[sym]['stop_mode'] = 'dollar'
                    self.state[sym].pop('trailing_percent', None)
                self._mark_position_protection(
                    sym,
                    'confirmed',
                    order_id=getattr(primary_trail.order, 'orderId', None),
                )
                logger.info(
                    f"AUDIT: {sym} — TRAIL SELL confirmed "
                    f"(id={primary_trail.order.orderId} "
                    f"{meta.get('desc', 'protection fields unavailable')} "
                    f"entry_gate={'ok' if gate_ok else 'failed'})"
                )
                continue

            logger.info(f"AUDIT: {sym} — no TRAIL SELL found; placing chandelier stop...")
            try:
                contract = self._stock_contract(sym)

                bars = self.ib.reqHistoricalData(
                    contract, '', DAILY_LOOKBACK, DAILY_BAR_SIZE, 'TRADES', True
                )
                if not isinstance(bars, list) or len(bars) < CHANDELIER_PERIOD:
                    self._mark_position_protection(
                        sym,
                        'unconfirmed',
                        'insufficient_history_for_audit_stop',
                    )
                    self._alert(
                        "CRITICAL",
                        f"AUDIT: {sym} — insufficient history "
                        f"({len(bars) if isinstance(bars, list) else 0} bars), "
                        f"cannot place stop; position is unprotected."
                    )
                    continue

                df = util.df(bars)
                df = apply_all(df)
                atr_chandelier = float(df['ATR_CHAND'].iloc[-1])
                if np.isnan(atr_chandelier) or atr_chandelier <= 0:
                    self._mark_position_protection(
                        sym,
                        'unconfirmed',
                        'invalid_atr_for_audit_stop',
                    )
                    self._alert(
                        "CRITICAL",
                        f"AUDIT: {sym} — ATR_CHAND invalid ({atr_chandelier}), "
                        f"cannot place stop; position is unprotected."
                    )
                    continue

                chandelier_dist = round(atr_chandelier * CHANDELIER_MULT, 2)

                stop_order               = Order()
                stop_order.action        = 'SELL'
                stop_order.orderType     = 'TRAIL'
                stop_order.totalQuantity = qty
                stop_order.auxPrice      = chandelier_dist
                stop_order.tif           = 'GTC'
                stop_order.goodAfterTime = self._stop_good_after_time()
                # outsideRth = False (default): stop only triggers during regular
                # trading hours, preventing premature exits on thin pre/post-market moves.
                stop_order.transmit      = True

                if not self._preflight_order(
                    contract,
                    stop_order,
                    sym,
                    allow_protective_sell_fail_open=True,
                ):
                    self._mark_position_protection(
                        sym,
                        'unconfirmed',
                        'audit_stop_preflight_rejected',
                    )
                    self._alert(
                        "CRITICAL",
                        f"AUDIT: {sym} — TRAIL stop pre-flight rejected by IB; "
                        f"position is unprotected — will retry on next cycle."
                    )
                    continue

                stop_trade = self.ib.placeOrder(contract, stop_order)
                # Give IB 2 s to acknowledge and assign a terminal/live status.
                self.ib.sleep(2)

                status = stop_trade.orderStatus.status
                if status in _REJECTED_ORDER_STATUSES:
                    self._mark_position_protection(
                        sym,
                        'unconfirmed',
                        f'audit_stop_rejected_{status}',
                        order_id=getattr(stop_trade.order, 'orderId', None),
                    )
                    self._metric_inc('protective_stop_rejected')
                    self._alert(
                        "CRITICAL",
                        f"AUDIT: {sym} — TRAIL stop rejected by IB "
                        f"(status={status} id={stop_trade.order.orderId}); "
                        f"position is unprotected — will retry on next cycle."
                    )
                    continue  # do not update state; leave for next audit cycle

                self.state[sym]['stop_dist'] = chandelier_dist
                self.state[sym]['stop_loss'] = round(
                    float(pos_data.get('price', 0)) - chandelier_dist, 2
                )
                self._mark_position_protection(
                    sym,
                    'confirmed',
                    order_id=getattr(stop_trade.order, 'orderId', None),
                )
                logger.info(
                    f"AUDIT: {sym} — TRAIL SELL live "
                    f"(qty={qty:.4f} dist=${chandelier_dist:.2f} "
                    f"status={status} id={stop_trade.order.orderId})"
                )
            except Exception as e:
                self._mark_position_protection(
                    sym,
                    'unconfirmed',
                    f'audit_stop_exception_{type(e).__name__}',
                )
                self._alert("CRITICAL", f"AUDIT: {sym} — failed to place TRAIL stop: {e}")

    def _open_trades_for_audit(self) -> list:
        """Return all visible open trades before deciding protection is missing.

        ib.openTrades() can be empty for still-live GTC orders until the API
        client has requested the broker's all-open-orders feed. A stop audit
        must never conclude "no stop exists" from a partial local cache.
        """
        trades = []
        seen = set()
        try:
            all_open = self.ib.reqAllOpenOrders()
            if isinstance(all_open, list):
                trades.extend(all_open)
            self.ib.sleep(1)
        except Exception as e:
            logger.warning(f"AUDIT: reqAllOpenOrders failed; falling back to openTrades cache ({e})")

        try:
            cached = self.ib.openTrades()
            if isinstance(cached, list):
                trades.extend(cached)
        except Exception as e:
            logger.warning(f"AUDIT: openTrades cache unavailable ({e})")

        unique = []
        for trade in trades:
            order = getattr(trade, 'order', None)
            contract = getattr(trade, 'contract', None)
            key = (
                getattr(order, 'permId', None),
                getattr(order, 'orderId', None),
                getattr(contract, 'conId', None),
                getattr(contract, 'symbol', None),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(trade)
        return unique

    def _position_needs_stop_audit(self) -> bool:
        """True when local state suggests a position may lack protection."""
        for data in self.state.values():
            if data.get('pending_exit'):
                continue
            if data.get('protection_status') == 'unconfirmed':
                return True
            try:
                qty = float(data.get('qty', 0) or 0)
                stop_dist = float(data.get('stop_dist', 0) or 0)
                stop_loss = float(data.get('stop_loss', 0) or 0)
            except (TypeError, ValueError):
                return True
            if qty > 0 and (stop_dist <= 0 or stop_loss <= 0):
                return True
        return False

    def _maybe_audit_stop_orders(self):
        """Run stop-order audit once per trading day, and immediately for unprotected state."""
        if not self.state:
            return

        now_ny = datetime.now(_TZ_NY)
        today = now_ny.strftime('%Y-%m-%d')
        needs_audit = self._position_needs_stop_audit()
        if getattr(self, '_last_audit_date', None) == today and not needs_audit:
            return

        reason = "unprotected position state" if needs_audit else "daily safety check"
        logger.info(f"AUDIT: running protective stop audit ({reason})")
        try:
            self._audit_stop_orders()
            self._last_audit_date = today
            self._last_audit_at = now_ny
        except Exception as e:
            self._alert(
                "CRITICAL",
                f"AUDIT: protective stop audit failed unexpectedly; "
                f"positions may be unprotected ({e})",
            )

    def _maybe_post_open_stop_audit(self):
        """
        Run one extra position sync and stop audit after the market opens.

        The 09:15 ET startup audit catches stale state early, but TRAIL orders
        and broker order status can change after the 09:30 opening auction. This
        audit is intentionally separate from the daily audit so it cannot be
        skipped just because the early audit already ran.
        """
        today_dt = datetime.now(_TZ_NY)
        if today_dt.weekday() >= 5:
            return

        today = today_dt.strftime('%Y-%m-%d')
        if getattr(self, '_last_post_open_audit_date', None) == today:
            return

        h, m = POST_OPEN_AUDIT_TIME
        audit_time = today_dt.replace(hour=h, minute=m, second=0, microsecond=0)
        if today_dt < audit_time:
            return

        last_audit_at = getattr(self, '_last_audit_at', None)
        if last_audit_at is not None and last_audit_at >= audit_time:
            self._last_post_open_audit_date = today
            logger.info(
                "AUDIT: post-open checkpoint already covered by a protective "
                f"stop audit at {last_audit_at.strftime('%H:%M:%S %Z')}"
            )
            return

        logger.info(
            f"AUDIT: running post-open position sync and stop audit "
            f"({h:02d}:{m:02d} ET checkpoint)"
        )
        try:
            self._sync_positions_from_ibkr()
            if self.state:
                self._audit_stop_orders()
                self._update_position_prices()
            else:
                logger.info("AUDIT: post-open checkpoint found no open positions")
            self._last_post_open_audit_date = today
            self._last_audit_date = today
            self._last_audit_at = today_dt
            self._write_dashboard_data(connected=True)
        except Exception as e:
            self._alert(
                "CRITICAL",
                f"AUDIT: post-open protective stop audit failed unexpectedly; "
                f"positions may be unprotected ({e})",
            )

    def _regular_management_active(self, now_ny: Optional[datetime] = None) -> bool:
        """True only during the regular-session window where software exits may run."""
        now_ny = now_ny or datetime.now(_TZ_NY)
        if now_ny.weekday() >= 5:
            return False
        hhmm = (now_ny.hour, now_ny.minute)
        return STOP_ACTIVATION_TIME <= hhmm < MARKET_CLOSE_TIME

    def _ensure_vix_contract(self) -> bool:
        """Qualify and cache the VIX contract; fail closed for entry logic."""
        if self._vix_contract is not None:
            return True
        try:
            vix_contracts = self.ib.qualifyContracts(Index('VIX', 'CBOE'))
            if not vix_contracts:
                logger.warning("VIX contract unavailable.")
                return False
            self._vix_contract = vix_contracts[0]
            return True
        except Exception as e:
            logger.warning(f"VIX contract qualification failed: {e}")
            return False

    def _build_readiness_snapshot(self, checkpoint: str) -> dict:
        """
        Collect non-trading data that helps the next trading session.

        This deliberately avoids scanner/entry calls. It may reconcile account
        values, VIX/SPY regime, open orders, and current local/broker position
        state, but it must not create new BUY orders.
        """
        now_ny = datetime.now(_TZ_NY)
        account_error = None
        try:
            equity, settled = self._get_account_values()
            self._last_equity = equity
            self._last_settled_cash = settled
            self._equity_initialized = True
        except AccountDataUnavailable as e:
            account_error = str(e)
            equity = self._last_equity
            settled = self._last_settled_cash

        allocation = self._calc_entry_allocation(equity, settled, len(self.state)) if equity > 0 else {
            'max_pos': 0,
            'capacity_slots': 0,
            'cash_slots': 0,
            'entry_slots': 0,
            'deployable_cash': 0.0,
            'bucket_size': 0.0,
        }

        vix_price = None
        if self._ensure_vix_contract():
            vix_price = self._fetch_vix_price()
        self._last_vix = vix_price

        try:
            spy_trend = self._fetch_spy_trend()
        except Exception as e:
            logger.warning(f"READINESS: SPY regime snapshot failed: {e}")
            spy_trend = False

        open_order_count = 0
        open_sell_count = 0
        try:
            open_orders = self.ib.reqAllOpenOrders()
            open_order_count = len(open_orders or [])
            open_sell_count = sum(
                1 for trade in (open_orders or [])
                if str(getattr(trade.order, 'action', '')).upper() == 'SELL'
            )
        except Exception as e:
            logger.warning(f"READINESS: open-order snapshot failed: {e}")

        positions = []
        for sym, data in sorted(self.state.items()):
            positions.append({
                'symbol': sym,
                'qty': float(data.get('qty', 0) or 0),
                'entry_price': float(data.get('fill_price', data.get('price', 0)) or 0),
                'current_price': float(data.get('current_price', data.get('price', 0)) or 0),
                'stop_loss': float(data.get('stop_loss', 0) or 0),
                'effective_stop': float(data.get('effective_stop', data.get('stop_loss', 0)) or 0),
                'pending_exit': bool(data.get('pending_exit', False)),
            })

        snapshot = {
            'checkpoint': checkpoint,
            'timestamp': now_ny.isoformat(),
            'trading_mode': TRADING_MODE,
            'active_management_window': self._regular_management_active(now_ny),
            'account': {
                'equity': round(float(equity or 0), 2),
                'settled_cash': round(float(settled or 0), 2),
                'deployable_cash': round(float(allocation['deployable_cash']), 2),
                'bucket_size': round(float(allocation['bucket_size']), 2),
                'max_positions': int(allocation['max_pos']),
                'entry_slots': int(allocation['entry_slots']),
                'error': account_error,
            },
            'regime': {
                'vix': round(float(vix_price), 2) if vix_price is not None else None,
                'vix_threshold': VIX_THRESHOLD,
                'spy_trend': bool(spy_trend),
            },
            'orders': {
                'open_order_count': open_order_count,
                'open_sell_order_count': open_sell_count,
            },
            'positions': positions,
        }
        self._write_readiness_snapshot(snapshot)
        return snapshot

    def _run_operational_maintenance(self, checkpoint: str):
        """Run a non-entry broker reconciliation/readiness checkpoint."""
        logger.info(f"OFFHOURS: running {checkpoint} maintenance checkpoint")
        self._sync_positions_from_ibkr()
        if self.state:
            self._audit_stop_orders()
            now_ny = datetime.now(_TZ_NY)
            self._last_audit_date = now_ny.strftime('%Y-%m-%d')
            self._last_audit_at = now_ny
            self._update_position_prices()

        snapshot = self._build_readiness_snapshot(checkpoint)
        logger.info(
            f"OFFHOURS: {checkpoint} snapshot | "
            f"equity=${snapshot['account']['equity']:.2f} "
            f"settled=${snapshot['account']['settled_cash']:.2f} "
            f"positions={len(snapshot['positions'])} "
            f"orders={snapshot['orders']['open_order_count']}"
        )
        self._write_dashboard_data(connected=True)

    def _maybe_run_off_hours_jobs(self) -> bool:
        """Run scheduled non-trading jobs once per trading date."""
        now_ny = datetime.now(_TZ_NY)
        if now_ny.weekday() >= 5:
            return False

        today = now_ny.strftime('%Y-%m-%d')
        hhmm = (now_ny.hour, now_ny.minute)
        ran = False

        if (PREMARKET_READINESS_TIME <= hhmm < ENTRY_START
                and getattr(self, '_last_premarket_readiness_date', None) != today):
            try:
                self._run_operational_maintenance('premarket_readiness')
                self._last_premarket_readiness_date = today
                ran = True
            except Exception as e:
                self._alert(
                    "ERROR",
                    f"OFFHOURS: premarket readiness checkpoint failed: {e}",
                )

        if (hhmm >= POST_CLOSE_MAINTENANCE_TIME
                and getattr(self, '_last_post_close_maintenance_date', None) != today):
            try:
                self._run_operational_maintenance('post_close_reconciliation')
                self._last_post_close_maintenance_date = today
                ran = True
            except Exception as e:
                self._alert(
                    "ERROR",
                    f"OFFHOURS: post-close maintenance checkpoint failed: {e}",
                )

        return ran

    # ── Regime / sector / correlation helpers ──────────────────────────────────
    def _fetch_spy_trend(self) -> bool:
        """True when SPY price > SMA50 > SMA200 and SMA200 is rising."""
        tz_ny  = pytz.timezone('US/Eastern')
        today  = datetime.now(tz_ny).strftime('%Y-%m-%d')
        if self._spy_cache.get('date') == today:
            return self._spy_cache['trend']
        try:
            if 'SPY' not in self._contract_cache:
                self._contract_cache['SPY'] = self._stock_contract('SPY')
            bars = self.ib.reqHistoricalData(
                self._contract_cache['SPY'], '', DAILY_LOOKBACK, DAILY_BAR_SIZE, 'TRADES', True
            )
            if not isinstance(bars, list) or len(bars) < MA_SLOW + SMA200_SLOPE_LOOKBACK:
                logger.warning("SPY regime check has insufficient history — blocking new entries")
                self._spy_cache = {'date': today, 'trend': False}
                return False
            df = util.df(bars)
            df['MA50']  = df['close'].rolling(50).mean()
            df['MA200'] = df['close'].rolling(200).mean()
            df['SMA200_SLOPE'] = df['MA200'] - df['MA200'].shift(SMA200_SLOPE_LOOKBACK)
            last  = df.iloc[-1]
            trend = bool(
                last['close'] > last['MA50']
                and last['MA50'] > last['MA200']
                and last['SMA200_SLOPE'] > 0
            )
            self._spy_cache = {'date': today, 'trend': trend}
            return trend
        except Exception as e:
            logger.warning(f"SPY regime check failed: {e} — blocking new entries")
            return False

    def _get_sector(self, symbol: str, contract) -> str:
        """Return IB industry string for symbol.  Cached; returns 'Unknown' on error."""
        if symbol in self._sector_cache:
            return self._sector_cache[symbol]
        try:
            details = self.ib.reqContractDetails(contract)
            sector  = str(details[0].industry) if isinstance(details, list) and details else 'Unknown'
        except Exception:
            sector = 'Unknown'
        self._sector_cache[symbol] = sector
        return sector

    @staticmethod
    def _daily_returns(df: pd.DataFrame) -> pd.Series:
        """Return a date-indexed pct_change series (last 60 rows) for correlation.

        ib_async util.df() gives an integer-indexed DataFrame with a 'date'
        column.  Two DataFrames from separate reqHistoricalData calls have
        independent integer indices, so pd.concat(join='inner') on the default
        index produces zero overlap.  Setting the date column as the index
        lets the inner-join align by calendar date instead.
        """
        s = df.copy()
        if 'date' in s.columns:
            s = s.set_index('date')
        # Normalise to plain date so tz-aware and tz-naive indices align.
        if hasattr(s.index, 'normalize'):
            try:
                s.index = s.index.normalize().date
            except Exception:
                pass
        return s['close'].pct_change().dropna().tail(60)

    def _compute_book_correlation(self, sym: str, df_daily: pd.DataFrame) -> float:
        """Max absolute Pearson correlation of candidate vs current book.

        Correlation is a portfolio risk gate. If the engine cannot evaluate a
        current holding, fail closed by returning 1.0 so the entry is skipped.
        """
        if not self.state:
            return 0.0
        cand_ret = self._daily_returns(df_daily)
        max_corr  = 0.0
        for book_sym in self.state:
            if book_sym == sym:
                continue
            try:
                cached = self._bar_cache.get(book_sym)
                if cached and 'bars_daily' in cached:
                    book_df = util.df(cached['bars_daily'])
                else:
                    bars = self.ib.reqHistoricalData(
                        self._stock_contract(book_sym), '', '90 D', DAILY_BAR_SIZE, 'TRADES', True
                    )
                    if not isinstance(bars, list) or not bars:
                        logger.warning(
                            f"CORRELATION {sym}: no history for held {book_sym}; failing closed."
                        )
                        return 1.0
                    book_df = util.df(bars)
                    # Cache the freshly fetched bars so subsequent candidates in this
                    # scan cycle don't re-fetch and get a mismatched integer index.
                    self._bar_cache.setdefault(book_sym, {})['bars_daily'] = bars
                book_ret = self._daily_returns(book_df)
                aligned  = pd.concat([cand_ret, book_ret], axis=1, join='inner').dropna()
                if len(aligned) < 20:
                    logger.warning(
                        f"CORRELATION {sym}: insufficient overlap with held {book_sym}; failing closed."
                    )
                    return 1.0
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                if not np.isnan(corr):
                    max_corr = max(max_corr, abs(float(corr)))
            except Exception as e:
                logger.warning(
                    f"CORRELATION {sym}: failed against held {book_sym} ({e}); failing closed."
                )
                return 1.0
        return max_corr

    def _calc_max_positions(self, equity: float) -> int:
        """
        Compute maximum simultaneous position capacity from total account equity.

        Formula: floor(NetLiquidation / MIN_BUCKET_SIZE), capped at
        MAX_POSITIONS_CAP. Entry placement is still separately constrained by
        SettledCash, so a cash account cannot spend unsettled sale proceeds.
        """
        try:
            equity = float(equity)
        except (TypeError, ValueError):
            return 0
        if not np.isfinite(equity) or equity < MIN_BUCKET_SIZE:
            return 0
        return min(int(equity / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)

    @staticmethod
    def _calc_cash_entry_slots(settled: float) -> int:
        """Return how many new-entry buckets settled cash can fund right now."""
        deployable = VelocityEngine._deployable_settled_cash(settled)
        if deployable < MIN_BUCKET_SIZE:
            return 0
        return int(deployable / MIN_BUCKET_SIZE)

    @staticmethod
    def _deployable_settled_cash(settled: float) -> float:
        """Return the settled-cash amount allowed for new entry buckets."""
        try:
            settled = float(settled)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(settled) or settled <= 0:
            return 0.0
        pct = min(max(float(SETTLED_CASH_DEPLOYMENT_PCT), 0.0), 1.0)
        return settled * pct

    def _calc_entry_allocation(self, equity: float, settled: float, open_count: int) -> Dict[str, float]:
        """Dynamically derive entry slots and bucket size from live account values."""
        max_pos = self._calc_max_positions(equity)
        capacity_slots = max(0, max_pos - max(0, int(open_count or 0)))
        deployable_cash = self._deployable_settled_cash(settled)
        cash_slots = (
            int(deployable_cash / MIN_BUCKET_SIZE)
            if deployable_cash >= MIN_BUCKET_SIZE else 0
        )
        entry_slots = min(capacity_slots, cash_slots)
        bucket_size = deployable_cash / entry_slots if entry_slots > 0 else 0.0
        return {
            'max_pos': max_pos,
            'capacity_slots': capacity_slots,
            'cash_slots': cash_slots,
            'entry_slots': entry_slots,
            'deployable_cash': deployable_cash,
            'bucket_size': bucket_size,
        }

    # ── Scanner ────────────────────────────────────────────────────────────────
    def get_institutional_scan(self):
        """Dynamic discovery of institutional momentum from every unique IBKR scanner hit."""
        seen: set = set()
        symbols: list = []
        for sub in build_momentum_scanners():
            try:
                scan = self.ib.reqScannerData(sub)
                for item in scan:
                    sym = item.contractDetails.contract.symbol
                    if sym not in seen:
                        seen.add(sym)
                        symbols.append(sym)
            except Exception as e:
                logger.warning(
                    f"SCAN: IB scanner ({sub.scanCode}) failed ({e}); skipping"
                )
        if not symbols:
            logger.warning("SCAN: all scanner queries failed or returned no candidates")
        return symbols

    # ── Technical context ──────────────────────────────────────────────────────
    def _remember_daily_scan_skip(self, symbol: str, reason: str):
        """Cache only stable same-day scan failures to reduce IBKR pacing load."""
        if not hasattr(self, '_daily_scan_skip') or self._daily_scan_skip is None:
            self._daily_scan_skip = {}
        self._daily_scan_skip[symbol] = reason

    def get_technical_context(self, symbol):
        # Contract cache — avoids re-qualifying the same symbol every 60-second cycle
        if symbol not in self._contract_cache:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            self._contract_cache[symbol] = contract
        contract = self._contract_cache[symbol]

        # Bar cache — daily bars are valid for one trading day; re-fetch on date change.
        # ORB bars are also anchored to today's 9:45 AM, so same daily scope applies.
        tz_ny      = pytz.timezone('US/Eastern')
        now_ny     = datetime.now(tz_ny)
        today_str  = now_ny.strftime('%Y-%m-%d')

        cached = self._bar_cache.get(symbol)
        if cached and cached.get('date') == today_str:
            bars_orb   = cached['bars_orb']
            bars_daily = cached['bars_daily']
        else:
            # 1. Opening Range High — the 9:30–9:45 AM bar of TODAY.
            #
            #    Using an empty endDateTime with '1 D' duration is WRONG: at 10 AM
            #    it starts at yesterday 10 AM, so bars[0] is a yesterday bar.
            #    Fix: anchor endDateTime to today's 9:45 AM ET so the 1-hour lookback
            #    window (8:45–9:45 AM) contains exactly one RTH bar — the 9:30 bar.
            #    bars[-1] is then reliably today's ORB bar regardless of current time.
            end_of_orb = now_ny.replace(hour=9, minute=45, second=0, microsecond=0)
            bars_orb   = self.ib.reqHistoricalData(
                contract, end_of_orb, ORB_LOOKBACK, ORB_BAR_SIZE, 'TRADES', True
            )

            # 2. Daily Context (Trends, ATR, RSI) — prior completed daily bars
            bars_daily = self.ib.reqHistoricalData(
                contract, '', DAILY_LOOKBACK, DAILY_BAR_SIZE, 'TRADES', True
            )

            if bars_orb and bars_daily:
                self._bar_cache[symbol] = {
                    'date':       today_str,
                    'bars_orb':   bars_orb,
                    'bars_daily': bars_daily,
                }

        if not bars_orb:
            return None
        orb_bar = bars_orb[-1]         # last bar = the 9:30 AM bar that closed at 9:45
        orb_high = orb_bar.high
        day_open = float(getattr(orb_bar, 'open', np.nan))
        if not orb_high or np.isnan(float(orb_high)) or orb_high <= 0:
            logger.warning(f"SCAN {symbol}: ORB high invalid ({orb_high}), skipping")
            return None
        if np.isnan(day_open) or day_open <= 0:
            logger.warning(f"SCAN {symbol}: day open invalid ({day_open}), skipping")
            return None

        if not bars_daily:
            return None
        df = util.df(bars_daily)
        if len(df) < MIN_CANDLES:
            logger.warning(
                f"SCAN {symbol}: only {len(df)} daily bars — need {MIN_CANDLES}, skipping"
            )
            self._remember_daily_scan_skip(symbol, f"insufficient daily history (<{MIN_CANDLES})")
            return None
        df = apply_all(df, RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW, SMA200_SLOPE_LOOKBACK, CHANDELIER_PERIOD)
        if np.isnan(float(df['MA200'].iloc[-1])):
            logger.warning(f"SCAN {symbol}: MA200 is NaN (data gaps in history), skipping")
            self._remember_daily_scan_skip(symbol, "invalid daily MA200")
            return None

        # 20-day average dollar volume — used to block low-liquidity pumps
        avg_20d_vol    = float(df['volume'].tail(20).mean())
        dollar_vol_20d = float((df['close'] * df['volume']).tail(20).mean())

        # 3. Live price + bid-ask spread + intraday relative volume
        tickers = self.ib.reqTickers(contract)
        if not tickers:
            logger.warning(f"SCAN {symbol}: live ticker unavailable, skipping")
            return None
        ticker = tickers[0]
        live_price = (
            self._coerce_positive_price(ticker.marketPrice())
            or self._coerce_positive_price(getattr(ticker, 'last', None))
            or self._coerce_positive_price(getattr(ticker, 'close', None))
        )
        if live_price is None:
            logger.warning(f"SCAN {symbol}: live price unavailable, skipping")
            return None

        bid = float(ticker.bid) if not pd.isna(ticker.bid) else 0.0
        ask = float(ticker.ask) if not pd.isna(ticker.ask) else 0.0
        if bid > 0 and ask > bid:
            spread_pct = (ask - bid) / ((bid + ask) / 2)
        else:
            spread_pct = float('inf')   # unavailable → fail-closed

        intraday_vol = float(ticker.volume) if not pd.isna(ticker.volume) else 0.0
        raw_rvol = intraday_vol / avg_20d_vol if avg_20d_vol > 0 else 0.0
        volume_pace = volume_pace_from_intraday(intraday_vol, avg_20d_vol, now_ny)
        intraday_gain = (live_price - day_open) / day_open
        try:
            day_high = float(getattr(ticker, 'high', np.nan))
        except (TypeError, ValueError):
            day_high = np.nan
        try:
            day_low = float(getattr(ticker, 'low', np.nan))
        except (TypeError, ValueError):
            day_low = np.nan
        if not pd.isna(day_high) and not pd.isna(day_low) and day_high > 0 and day_low > 0:
            day_high = max(day_high, live_price)
            day_low  = min(day_low, live_price)
            day_range_location = (
                (live_price - day_low) / (day_high - day_low)
                if day_high > day_low else None
            )
        else:
            day_range_location = None

        return {
            'orb_high':         orb_high,
            'day_open':         day_open,
            'ma50':             float(df['MA50'].iloc[-1]),
            'ma200':            float(df['MA200'].iloc[-1]),
            'atr':              float(df['ATR'].iloc[-1]),
            'atr5':             float(df['ATR5'].iloc[-1]),
            'atr20':            float(df['ATR20'].iloc[-1]),
            'atr_chandelier':   float(df['ATR_CHAND'].iloc[-1]),
            'sma200_slope':     float(df['SMA200_SLOPE'].iloc[-1]),
            'high10':           float(df['HIGH10'].iloc[-1]),
            'rsi':              float(df['RSI'].iloc[-1]),
            'rsi_prev':         float(df['RSI'].iloc[-2]),
            'close':            float(df['close'].iloc[-1]),
            'live_price':       live_price,
            'bid':              bid,
            'ask':              ask,
            'spread_pct':       spread_pct,
            # Live ranking uses time-normalized volume pace.  Raw intraday
            # RVOL stays available for diagnostics, but early-session ranking
            # should not punish a stock just because only part of the day has
            # elapsed.
            'rvol':             volume_pace,
            'rvol_raw':         raw_rvol,
            'volume_pace':      volume_pace,
            'intraday_volume':  intraday_vol,
            'avg_20d_volume':   avg_20d_vol,
            'day_range_location': day_range_location,
            'intraday_gain':    intraday_gain,
            'volume':           int(df['volume'].iloc[-1]),
            'dollar_vol_20d':   dollar_vol_20d,
            'price_fetched_at': datetime.now(tz_ny),
            'contract':         contract,
            'df_daily':         df,
        }

    # ── Exit management ────────────────────────────────────────────────────────
    def manage_position_exits(self):
        """Manage live software exits for existing positions.

        Broker-side trailing stops remain the primary protection. Software exits
        require a fresh broker price so stale dashboard/cache values cannot
        liquidate a valid swing position.
        """
        now_et          = datetime.now(_TZ_NY)
        is_friday_close = (now_et.weekday() == 4 and now_et.hour >= FRIDAY_CLOSE_HOUR)
        today_str       = now_et.strftime('%Y-%m-%d')
        hhmm            = (now_et.hour, now_et.minute)
        is_eod_window   = (hhmm >= EOD_EXIT_TIME and now_et.weekday() < 5)
        is_velocity_exit_window = (hhmm >= VELOCITY_EXIT_TIME and now_et.weekday() < 5)
        eod_exit_due    = (
            is_eod_window
            and getattr(self, '_last_eod_exit_date', None) != today_str
        )
        changed          = False
        eod_exit_checked = False

        for sym in list(self.state.keys()):
            data        = self.state[sym]
            if data.get('pending') or data.get('pending_exit'):
                continue
            entry_price = float(data.get('price', 0))
            if entry_price <= 0:
                logger.warning(f"EXIT: {sym} has invalid entry price, skipping.")
                continue

            cached_cur = self._coerce_positive_price(data.get('current_price', 0))
            fresh_cur  = self._fresh_market_price(sym)
            if fresh_cur is None:
                logger.warning(
                    f"EXIT: {sym} fresh price unavailable; skipping software exit checks "
                    f"(cached current_price={cached_cur if cached_cur else 'unavailable'})."
                )
                continue

            cur = fresh_cur
            self.state[sym]['current_price'] = round(cur, 2)
            self.state[sym]['price_checked_at'] = now_et.isoformat()
            changed = True

            raw_time = data.get('time', '')
            trading_days_held = None
            if raw_time:
                try:
                    entry_dt = datetime.fromisoformat(raw_time)
                    if entry_dt.tzinfo is None:
                        entry_dt = _TZ_NY.localize(entry_dt)
                    trading_days_held = _count_trading_days(entry_dt, now_et)
                except (ValueError, TypeError):
                    logger.warning(
                        f"EXIT: {sym} — malformed entry timestamp {raw_time!r}; "
                        "minimum-hold exits are disabled for this cycle"
                    )
            else:
                logger.warning(
                    f"EXIT: {sym} — missing entry timestamp; minimum-hold exits "
                    "are disabled for this cycle"
                )

            # ── 1. Intraday hard stop — requires a fresh broker price
            drawdown = (cur - entry_price) / entry_price
            if drawdown <= -HARD_STOP_PCT:
                logger.warning(
                    f"HARD STOP: {sym} down {drawdown*100:.1f}% from entry "
                    f"(${cur:.2f} vs entry ${entry_price:.2f}). Forcing exit."
                )
                self.liquidate(sym)
                continue

            peak_price = max(float(data.get('peak_price', entry_price) or entry_price), cur)
            if peak_price != float(data.get('peak_price', entry_price) or entry_price):
                self.state[sym]['peak_price'] = round(peak_price, 2)
                changed = True
            if peak_price >= entry_price * (1 + BREAK_EVEN_PCT) and cur <= entry_price:
                logger.warning(
                    f"BREAK-EVEN EXIT: {sym} gave back a prior "
                    f"{BREAK_EVEN_PCT*100:.0f}%+ profit "
                    f"(peak=${peak_price:.2f}, current=${cur:.2f}, entry=${entry_price:.2f})."
                )
                self.liquidate(sym)
                continue

            # ── 2. Friday afternoon close — explicit weekend-risk policy
            if is_friday_close:
                friday_profit = (cur - entry_price) / entry_price
                if friday_profit < FRIDAY_MIN_PROFIT_PCT:
                    logger.warning(
                        f"FRIDAY CLOSE: {sym} profit={friday_profit*100:.1f}% < "
                        f"{FRIDAY_MIN_PROFIT_PCT*100:.0f}% — closing to avoid weekend risk."
                    )
                    self.liquidate(sym)
                    continue

            # ── 3. EOD flat — never rejects a same-day swing entry.
            #
            # Fires once per trading day after EOD_EXIT_TIME (default 15:45 ET),
            # but only after HOLD_TRADING_BARS has elapsed. This keeps the
            # strategy aligned with its swing mandate while retaining an
            # end-of-day stale-capital cleanup for older positions.
            if eod_exit_due:
                eod_exit_checked = True
                if trading_days_held is None:
                    logger.warning(
                        f"EOD FLAT: {sym} skipped — entry timestamp is unavailable "
                        "or malformed; cannot prove minimum hold window."
                    )
                    continue
                if trading_days_held < HOLD_TRADING_BARS:
                    logger.info(
                        f"EOD FLAT: {sym} skipped — held {trading_days_held} "
                        f"trading sessions (< {HOLD_TRADING_BARS}); swing hold window active."
                    )
                    continue
                eod_profit = (cur - entry_price) / entry_price
                if eod_profit <= 0:
                    logger.warning(
                        f"EOD FLAT: {sym} not in profit at end of day "
                        f"(profit={eod_profit*100:.2f}%, cur=${cur:.2f}, entry=${entry_price:.2f}) "
                        f"— closing position."
                    )
                    self.liquidate(sym)
                    continue

            # ── 4. Velocity exit — counts Mon-Fri trading sessions, not weekend hours
            #
            # Stagnant-trade recycling is intentionally an end-of-day decision.
            # A valid swing position should not be liquidated just because it has
            # not reached the profit threshold during the morning or midday.
            if trading_days_held is not None and trading_days_held >= HOLD_TRADING_BARS:
                if not is_velocity_exit_window:
                    logger.debug(
                        f"VELOCITY EXIT: {sym} skipped until "
                        f"{VELOCITY_EXIT_TIME[0]:02d}:{VELOCITY_EXIT_TIME[1]:02d} ET "
                        f"(held={trading_days_held} trading sessions)."
                    )
                    continue
                profit = (cur - entry_price) / entry_price
                if profit < PROFIT_MIN_THRESHOLD:
                    logger.info(f"VELOCITY EXIT: {sym} stagnant. Freeing capital for T+1.")
                    self.liquidate(sym)

        # Mark EOD exit as done only after at least one live-price evaluation.
        if eod_exit_due and eod_exit_checked:
            self._last_eod_exit_date = today_str

        if changed:
            self.save_state()

    def check_velocity_exits(self):
        """Backward-compatible wrapper for older callers."""
        return self.manage_position_exits()

    def _active_open_trades_for_symbol(self, symbol: str) -> list:
        """Return non-terminal open trades for one symbol."""
        try:
            open_trades = self.ib.openTrades()
        except Exception as e:
            logger.warning(f"LIQUIDATE {symbol}: could not query open trades: {e}")
            return []
        active = []
        for trade in open_trades:
            trade_symbol = getattr(getattr(trade, 'contract', None), 'symbol', None)
            if trade_symbol != symbol:
                continue
            status = str(getattr(getattr(trade, 'orderStatus', None), 'status', '') or '')
            if status in _REJECTED_ORDER_STATUSES or status == 'Filled':
                continue
            active.append(trade)
        return active

    def _active_sell_trades_for_symbol(self, symbol: str) -> list:
        """Return active SELL orders that can block a cash-account market exit."""
        return [
            trade for trade in self._active_open_trades_for_symbol(symbol)
            if str(getattr(getattr(trade, 'order', None), 'action', '')).upper() == 'SELL'
        ]

    def _cancel_open_orders_before_market_exit(self, symbol: str) -> bool:
        """Cancel open symbol orders, then wait until active SELLs are gone.

        IBKR cash accounts reject a market SELL if a full-quantity protective
        SELL is already live, because both orders together could oversell the
        position.  The unavoidable live-trading sequence is therefore:
        cancel protection, wait for cancellation, submit the market exit, and
        rebuild protection immediately if the exit is not accepted.
        """
        open_trades = self._active_open_trades_for_symbol(symbol)
        if not open_trades:
            return True

        logger.warning(
            f"LIQUIDATE {symbol}: cancelling {len(open_trades)} existing open order(s) "
            "before cash-account market exit"
        )
        for trade in open_trades:
            try:
                self.ib.cancelOrder(trade.order)
            except Exception as e:
                logger.warning(f"LIQUIDATE {symbol}: cancel failed before market exit: {e}")

        for _ in range(20):
            self.ib.sleep(0.25)
            if not self._active_sell_trades_for_symbol(symbol):
                return True

        self._alert(
            "ERROR",
            f"LIQUIDATE {symbol}: active SELL orders remained after cancellation wait; "
            "market exit deferred to avoid IBKR cash-account oversell rejection."
        )
        return False

    def liquidate(self, symbol):

        found_position = False
        for p in self.ib.positions():
            if p.contract.symbol == symbol:
                qty = float(p.position)
                if qty <= 0:
                    continue
                found_position = True
                if not self._cancel_open_orders_before_market_exit(symbol):
                    if symbol in self.state:
                        self.state[symbol].pop('pending_exit', None)
                        self.save_state()
                    continue

                if symbol in self.state:
                    self.state[symbol]['pending_exit'] = True
                    self.save_state()
                sell_order = MarketOrder('SELL', qty)
                sell_order.tif = 'DAY'
                sell_order.goodAfterTime = ''
                sell_contract = copy.copy(p.contract)
                sell_contract.exchange = 'SMART'
                try:
                    trade = self.ib.placeOrder(sell_contract, sell_order)
                except Exception as e:
                    if symbol in self.state:
                        self.state[symbol].pop('pending_exit', None)
                        self.save_state()
                    self._alert(
                        "CRITICAL",
                        f"LIQUIDATE {symbol}: market SELL placement failed; "
                        f"state retained for retry ({e})"
                    )
                    self._audit_stop_orders()
                    continue
                try:
                    for _ in self.ib.loopUntil(trade.isDone, timeout=30):
                        pass
                except Exception as e:
                    logger.warning(f"LIQUIDATE {symbol}: status wait failed: {e}")

                status = str(getattr(trade.orderStatus, 'status', '') or '')
                try:
                    filled_qty = float(getattr(trade.orderStatus, 'filled', 0) or 0)
                except (TypeError, ValueError):
                    filled_qty = 0.0

                if status in _REJECTED_ORDER_STATUSES:
                    if symbol in self.state:
                        self.state[symbol].pop('pending_exit', None)
                        self.save_state()
                    self._alert(
                        "CRITICAL",
                        f"LIQUIDATE {symbol}: market SELL rejected "
                        f"(status={status}); state retained for retry"
                    )
                    self._audit_stop_orders()
                    continue

                if status == 'Filled' or filled_qty >= qty:
                    if symbol in self.state:
                        self.state[symbol]['pending_exit'] = True
                    self.save_state()
                    logger.info(
                        f"LIQUIDATE {symbol}: market SELL filled "
                        f"(qty={filled_qty:g}, status={status}); state pending until IBKR sync confirms flat"
                    )
                else:
                    self._alert(
                        "ERROR",
                        f"LIQUIDATE {symbol}: market SELL submitted but not confirmed "
                        f"filled (status={status}, filled={filled_qty:g}); state retained"
                    )

        if not found_position and symbol in self.state:
            logger.info(
                f"LIQUIDATE {symbol}: no IBKR position found; "
                "cancelling orphaned exits and removing stale state"
            )
            self._cancel_orphaned_exit_orders(symbol)
            del self.state[symbol]
            self.save_state()

    def _cancel_orphaned_exit_orders(self, symbol: str) -> int:
        """Cancel leftover SELL orders only after IBKR confirms no position exists."""
        cancelled = 0
        try:
            open_trades = self.ib.openTrades()
        except Exception as e:
            logger.warning(f"SYNC: {symbol} — could not query open trades for orphan cleanup: {e}")
            return 0

        for trade in open_trades:
            trade_symbol = getattr(getattr(trade, 'contract', None), 'symbol', None)
            action = str(getattr(getattr(trade, 'order', None), 'action', '')).upper()
            if trade_symbol != symbol or action != 'SELL':
                continue
            try:
                self.ib.cancelOrder(trade.order)
                cancelled += 1
            except Exception as e:
                logger.warning(f"SYNC: {symbol} — orphan SELL cancel failed: {e}")

        if cancelled:
            logger.info(f"SYNC: {symbol} — cancelled {cancelled} orphaned SELL exit orders")
            self.ib.sleep(1)
        return cancelled

    def _score_candidate(
        self,
        ctx: dict,
        *,
        volume_floor: float = RVOL_MIN,
        gap_max_pct: float = GAP_MAX_PCT,
    ) -> float:
        """Rank a passing candidate with the shared live/backtest scorer."""
        return score_candidate(
            ctx,
            model=SCORING_MODEL,
            volume_floor=volume_floor,
            spread_max_pct=SPREAD_MAX_PCT,
            atr_pct_max=ATR_PCT_MAX,
            gap_max_pct=gap_max_pct,
        )

    def _entry_price_is_still_valid(
        self,
        sym: str,
        ctx: dict,
        price: float,
        breakout_pct: float = BREAKOUT_PCT,
        gap_max_pct: float = GAP_MAX_PCT,
    ) -> bool:
        """Re-check fast-moving entry gates immediately before sending an order."""
        if np.isnan(price) or price <= 0:
            logger.warning(f"SKIP {sym}: refreshed entry price invalid ({price})")
            return False

        if price < SCAN_MIN_PRICE:
            logger.warning(
                f"SKIP {sym}: refreshed entry price ${price:.2f} below "
                f"minimum ${SCAN_MIN_PRICE:.2f}"
            )
            return False

        orb_h = float(ctx.get('orb_high', 0.0))
        if orb_h <= 0 or price <= orb_h:
            logger.warning(
                f"SKIP {sym}: refreshed price ${price:.2f} no longer clears ORB ${orb_h:.2f}"
            )
            return False

        day_open = float(ctx.get('day_open', price) or price)
        if day_open > orb_h * (1 + gap_max_pct):
            logger.warning(
                f"SKIP {sym}: day open ${day_open:.2f} is beyond "
                f"{gap_max_pct*100:.0f}% opening-gap cap from ORB ${orb_h:.2f}"
            )
            return False

        return True

    def _sync_positions_from_ibkr(self):
        """
        Reconcile self.state against actual IBKR positions every cycle.
        - Symbols in IBKR but missing from state → added (e.g. filled while engine restarted)
        - Symbols in state but NOT in IBKR     → removed (stop/target/manual close hit)
        This makes state always match reality, regardless of order fill timing.
        """
        ibkr_pos = {p.contract.symbol: p for p in self.ib.positions()}
        missing_counts = getattr(self, '_missing_position_counts', {})
        changed  = False

        # Add missing IBKR positions and update changed quantities. IBKR is the
        # source of truth for quantity after partial fills, manual adjustments,
        # splits, or broker-side corrections.
        for sym, pos in ibkr_pos.items():
            ibkr_qty = float(pos.position)
            if ibkr_qty <= 0:
                continue

            qty = round(ibkr_qty, 4)
            avg_cost = float(pos.avgCost) if pos.avgCost else 0.0
            if avg_cost <= 0:
                logger.warning(
                    f"SYNC: {sym} — avgCost={avg_cost} from IBKR; "
                    f"velocity-exit profit check will be skipped until price is corrected."
                )

            if sym not in self.state:
                self.state[sym] = {
                    'price':           round(avg_cost, 2),
                    'fill_price':      round(avg_cost, 2),
                    'broker_avg_cost': round(avg_cost, 4),
                    'time':            datetime.now(_TZ_NY).isoformat(),
                    'qty':             qty,
                    'stop_loss':       0.0,
                    'peak_price':      round(avg_cost, 2),
                    'volume':          0,
                    'score':           None,
                }
                logger.info(f"SYNC: Added {sym} from IBKR (qty={pos.position} avg=${avg_cost:.2f})")
                changed = True
            else:
                missing_counts.pop(sym, None)
                if self.state[sym].pop('pending_exit', None):
                    logger.warning(
                        f"SYNC: {sym} still present at IBKR after pending exit; "
                        "clearing pending_exit and continuing risk management."
                    )
                    changed = True
                state_qty = float(self.state[sym].get('qty', 0))
                if abs(state_qty - qty) > 1e-6:
                    logger.info(
                        f"SYNC: Updated {sym} qty from state={state_qty:g} "
                        f"to IBKR={qty:g}"
                    )
                    self.state[sym]['qty'] = qty
                    changed = True

                if avg_cost > 0:
                    if self.state[sym].get('broker_avg_cost') != round(avg_cost, 4):
                        self.state[sym]['broker_avg_cost'] = round(avg_cost, 4)
                        changed = True
                    if float(self.state[sym].get('price', 0)) <= 0:
                        self.state[sym]['price'] = round(avg_cost, 2)
                        changed = True
                    if float(self.state[sym].get('fill_price', 0)) <= 0:
                        self.state[sym]['fill_price'] = round(avg_cost, 2)
                        changed = True
                    if float(self.state[sym].get('peak_price', 0)) <= 0:
                        self.state[sym]['peak_price'] = round(avg_cost, 2)
                        changed = True

        # Remove state entries whose IBKR position is gone or zero
        for sym in list(self.state.keys()):
            ibkr_qty = float(ibkr_pos[sym].position) if sym in ibkr_pos else 0.0
            if ibkr_qty <= 0:
                miss_count = missing_counts.get(sym, 0) + 1
                missing_counts[sym] = miss_count
                if miss_count < 2 and not self._force_exit_active():
                    logger.warning(
                        f"SYNC: {sym} missing from IBKR positions snapshot once; "
                        "deferring state removal until a second confirming snapshot."
                    )
                    continue
                logger.info(f"SYNC: Removed {sym} from state — no IBKR position found")
                self._cancel_orphaned_exit_orders(sym)
                del self.state[sym]
                missing_counts.pop(sym, None)
                changed = True

        self._missing_position_counts = missing_counts
        if changed:
            self.save_state()

    def _update_position_prices(self):
        """Fetch live price for every open position and persist to state.
        Also backfills volume when it is 0 (e.g. positions synced from IBKR
        that never went through the normal entry path).
        """
        if not self.state:
            return
        changed = False
        for sym in list(self.state.keys()):
            if getattr(self, '_missing_position_counts', {}).get(sym, 0) > 0:
                logger.info(
                    f"PRICE: {sym} skipped — IBKR position was missing in the latest "
                    "snapshot; waiting for confirmation before refreshing market data."
                )
                continue
            try:
                contract = self._stock_contract(sym)
                price = self._fresh_market_price(sym)
                if price is not None:
                    cur = price
                    self.state[sym]['current_price'] = round(cur, 2)
                    self.state[sym]['price_checked_at'] = datetime.now(_TZ_NY).isoformat()
                    ep  = float(self.state[sym].get('price', 0))
                    qty = float(self.state[sym].get('qty', 0))
                    if ep > 0 and qty > 0:
                        self.state[sym]['unrealized_pnl']     = round((cur - ep) * qty, 2)
                        self.state[sym]['unrealized_pnl_pct'] = round((cur - ep) / ep * 100, 2)
                    # Track trailing stop high-watermark so dashboard shows live stop level
                    sd = float(self.state[sym].get('stop_dist', 0))
                    if sd > 0:
                        peak = max(float(self.state[sym].get('peak_price', cur)), cur)
                        self.state[sym]['peak_price']     = round(peak, 2)
                        initial_sl = float(self.state[sym].get('stop_loss', 0))
                        if str(self.state[sym].get('stop_mode', '')).lower() == 'percent':
                            # IBKR owns the moving stop for percent TRAIL orders.
                            # Do not invent a fixed-dollar trail from a snapshot;
                            # that can overstate protection on the dashboard.
                            if initial_sl > 0:
                                self.state[sym]['effective_stop'] = round(initial_sl, 2)
                        else:
                            trail_floor = peak - sd
                            if ep > 0 and peak >= ep * (1 + BREAK_EVEN_PCT):
                                trail_floor = max(trail_floor, ep)
                            self.state[sym]['effective_stop'] = round(max(initial_sl, trail_floor), 2)
                    changed = True

                # Backfill volume once if it is missing/zero — only fetches
                # historical bars on the first cycle where volume is absent.
                # Uses a short 5-day window; no need for a full 1-year pull.
                if self.state[sym].get('volume', 0) == 0:
                    bars = self.ib.reqHistoricalData(
                        contract, '', '5 D', DAILY_BAR_SIZE, 'TRADES', True
                    )
                    if bars:
                        df_vol = util.df(bars)
                        vol    = int(df_vol['volume'].iloc[-1])
                        if vol > 0:
                            self.state[sym]['volume'] = vol
                            changed = True
            except Exception as e:
                logger.warning(f"Could not refresh price for {sym}: {e}")
        if changed:
            self.save_state()

    # ── Main cycle ─────────────────────────────────────────────────────────────
    def run_cycle(self):
        self._metric_inc('cycles')
        if not hasattr(self, '_daily_scan_skip') or self._daily_scan_skip is None:
            self._daily_scan_skip = {}

        # 0. Ensure connection is live before doing anything
        if not self._ensure_connected():
            logger.error("ENGINE: reconnect failed — skipping cycle for safety")
            self._write_dashboard_data(connected=False)
            return

        # 1. Sync state with actual IBKR positions (source of truth)
        self._sync_positions_from_ibkr()

        if self._force_exit_active():
            self._alert(
                "CRITICAL",
                f"FORCE EXIT: {FORCE_EXIT_FILE} exists — liquidating all tracked positions."
            )
            for sym in list(self.state.keys()):
                self.liquidate(sym)
            self._update_position_prices()
            self._write_dashboard_data(connected=True)
            return

        off_hours_ran = self._maybe_run_off_hours_jobs()

        # Stop protection is independent of new-entry logic. Run this before
        # account/VIX gates so existing positions remain protected even when
        # fresh cash data or regime data is temporarily unavailable.  Scheduled
        # off-hours jobs also audit stops and set _last_audit_date, so this call
        # naturally skips duplicate same-day audits.
        self._maybe_audit_stop_orders()
        self._maybe_post_open_stop_audit()

        now_ny = datetime.now(_TZ_NY)
        if not self._regular_management_active(now_ny):
            # During closed/premarket hours the engine may reconcile, audit
            # protective stops, and write readiness data, but it must not run
            # software exits, scanners, or new-entry logic.
            if not off_hours_ran:
                try:
                    equity, settled = self._get_account_values()
                    self._last_equity = equity
                    self._last_settled_cash = settled
                    self._equity_initialized = True
                except AccountDataUnavailable as e:
                    self._alert(
                        "ERROR",
                        f"{e} Off-hours dashboard account refresh skipped."
                    )
                if self.state:
                    self._update_position_prices()
                self._write_dashboard_data(connected=True)
            return

        # 2. Account values — single API call for both equity and settled cash
        try:
            equity, settled = self._get_account_values()
        except AccountDataUnavailable as e:
            self._alert(
                "ERROR",
                f"{e} Managing existing positions only; no scanner or new entries."
            )
            self.manage_position_exits()
            self._update_position_prices()
            self._write_dashboard_data(connected=True)
            return
        allocation = self._calc_entry_allocation(equity, settled, len(self.state))
        max_pos = int(allocation['max_pos'])
        capacity_slots = int(allocation['capacity_slots'])
        cash_slots = int(allocation['cash_slots'])
        open_slots = int(allocation['entry_slots'])
        deployable_cash = allocation['deployable_cash']
        bucket_size = allocation['bucket_size']
        self._last_equity = equity
        self._last_settled_cash = settled
        self._equity_initialized = True
        bucket_text = f"${bucket_size:.2f}" if open_slots > 0 else "N/A"
        logger.info(
            f"HEARTBEAT: Equity ${equity:.2f} | Settled ${settled:.2f} | "
            f"Deployable ${deployable_cash:.2f} ({SETTLED_CASH_DEPLOYMENT_PCT:.0%}) | "
            f"EntrySlots {open_slots}/{max_pos} | CashSlots {cash_slots} | Bucket {bucket_text} | "
            f"Positions: {list(self.state.keys()) or 'none'}"
        )

        # Daily loss circuit breaker — reset at start of each new trading day
        today_str = datetime.now(_TZ_NY).strftime('%Y-%m-%d')
        if self._day_start_date != today_str:
            self._day_start_date   = today_str
            self._day_start_equity = equity
            self._daily_scan_skip.clear()
            self._bar_cache.clear()
        elif (self._day_start_equity is not None
              and equity < self._day_start_equity * (1 - MAX_DAILY_LOSS_PCT)):
            logger.warning(
                f"CIRCUIT BREAKER: daily loss limit hit "
                f"(equity ${equity:.2f} vs open ${self._day_start_equity:.2f} "
                f"= {(1 - equity/self._day_start_equity)*100:.1f}% loss). "
                f"No new entries for the rest of today."
            )
            self.manage_position_exits()
            self._update_position_prices()
            return

        # 3. Manage Existing.  Entry-only gates such as VIX, SPY regime, scanner,
        # and slot checks must not delay software exits for positions already held.
        self.manage_position_exits()

        if self._operator_halt_active():
            logger.warning(
                f"OPERATOR HALT: {HALT_FILE} exists — managing existing positions only."
            )
            self._update_position_prices()
            return

        if MARKET_DATA_TYPE != 1:
            logger.warning(
                f"DATA SAFETY: MARKET_DATA_TYPE={MARKET_DATA_TYPE}; "
                "new entries require real-time data (1). Managing existing positions only."
            )
            self._update_position_prices()
            return

        # 5. Entry Window (10:00 AM – 3:30 PM EST)
        tz_ny  = pytz.timezone('US/Eastern')
        now_ny = datetime.now(tz_ny)

        if now_ny.weekday() < 5 and ENTRY_START <= (now_ny.hour, now_ny.minute) <= ENTRY_END:
            # On Fridays raise the liquidity bar to 2× to avoid holding over weekends.
            is_friday = (now_ny.weekday() == 4)
            if is_friday and (now_ny.hour, now_ny.minute) >= FRIDAY_ENTRY_CUTOFF_TIME:
                self._metric_inc('scanner_skipped_friday_cutoff')
                logger.warning(
                    f"FRIDAY ENTRY CUTOFF: no new entries after "
                    f"{FRIDAY_ENTRY_CUTOFF_TIME[0]:02d}:{FRIDAY_ENTRY_CUTOFF_TIME[1]:02d} ET. "
                    "Managing existing positions only."
                )
                self._update_position_prices()
                return
            dol_vol_threshold = SCAN_MIN_DOLLAR_VOL * (VOL_MULT_FRIDAY if is_friday else 1.0)

            allocation = self._calc_entry_allocation(equity, settled, len(self.state))
            max_pos        = int(allocation['max_pos'])
            capacity_slots = int(allocation['capacity_slots'])
            cash_slots     = int(allocation['cash_slots'])
            slots          = int(allocation['entry_slots'])
            bucket_size    = allocation['bucket_size']
            if slots <= 0:
                self._metric_inc('scanner_skipped_no_slots')
                logger.info(
                    f"SCAN: no entry slots available "
                    f"(capacity={capacity_slots}, settled_cash_slots={cash_slots}, max={max_pos})"
                )
            else:
                # Entry-only VIX gate. It intentionally runs after position
                # management, manual halts, data-mode checks, Friday cutoff, and
                # slot/cash checks so IBKR VIX/HMDS calls are not wasted when no
                # new order can be placed anyway.
                if not self._ensure_vix_contract():
                    logger.warning("VIX contract unavailable. Skipping entries as precaution.")
                    self._update_position_prices()
                    return
                vix_price = self._fetch_vix_price()
                if vix_price is None:
                    logger.warning("VIX price unavailable. Skipping entries as precaution.")
                    self._last_vix = None
                    self._update_position_prices()
                    return
                self._last_vix = vix_price
                if vix_price > VIX_THRESHOLD:
                    logger.warning(f"VIX HIGH ({vix_price:.2f}). Risk Off — no new entries.")
                    self._update_position_prices()
                    return

                # Check SPY regime once per cycle — same answer for all candidates
                spy_trend = self._fetch_spy_trend()
                bear_phase = not spy_trend
                if bear_phase and not BEAR_PHASE_TRADING_ENABLED:
                    logger.warning("REGIME: SPY trend weak/falling — no new entries this cycle")
                    self._update_position_prices()
                    return

                if bear_phase:
                    regime_label = "BEAR"
                    rvol_min = BEAR_RVOL_MIN
                    vcp_ratio = BEAR_VCP_RATIO
                    rsi_threshold = BEAR_RSI_THRESHOLD
                    rsi_min_delta = BEAR_RSI_MIN_DELTA
                    gap_max_pct = BEAR_GAP_MAX_PCT
                    dol_vol_threshold *= BEAR_PHASE_DOLLAR_VOL_MULT
                    risk_per_trade_pct = RISK_PER_TRADE_PCT * BEAR_PHASE_RISK_MULT
                    logger.warning(
                        "REGIME: SPY trend weak/falling — bear-phase participation enabled; "
                        f"using stricter 8096 rules (gap≤{gap_max_pct*100:.0f}%, "
                        f"RSI>{rsi_threshold}, RSIΔ≥{rsi_min_delta}, "
                        f"risk={risk_per_trade_pct*100:.1f}%)."
                    )
                else:
                    regime_label = "BULL"
                    rvol_min = RVOL_MIN
                    vcp_ratio = VCP_RATIO
                    rsi_threshold = RSI_THRESHOLD
                    rsi_min_delta = RSI_MIN_DELTA
                    gap_max_pct = GAP_MAX_PCT
                    risk_per_trade_pct = RISK_PER_TRADE_PCT

                watchlist = self.get_institutional_scan()
                self._metric_inc('scanner_runs')
                self._metric_inc('scanner_candidates', len(watchlist))
                logger.info(f"SCAN: {len(watchlist)} candidates → {watchlist}"
                            + (f" [FRIDAY: dolVol threshold ${dol_vol_threshold/1e6:.0f}M]" if is_friday else ""))

                # Pre-compute sector counts for current book (for clustering check)
                book_sectors: Dict[str, int] = {}
                for book_sym in self.state:
                    if book_sym in self._contract_cache:
                        s = self._get_sector(book_sym, self._contract_cache[book_sym])
                        book_sectors[s] = book_sectors.get(s, 0) + 1

                # ── Phase 1: evaluate ALL candidates, collect those passing ──
                signals = []
                for sym in watchlist:
                    if sym in self.state:
                        logger.info(f"SCAN {sym}: SKIP — already in portfolio")
                        continue

                    if sym in TICKER_BLOCKLIST:
                        logger.debug(f"SCAN {sym}: SKIP — blocklisted (leveraged/inverse ETF)")
                        continue

                    if sym in self._daily_scan_skip:
                        logger.debug(
                            f"SCAN {sym}: SKIP — day-filtered "
                            f"({self._daily_scan_skip[sym]})"
                        )
                        continue

                    ctx = self.get_technical_context(sym)
                    if not ctx:
                        logger.warning(f"SCAN {sym}: SKIP — no technical data")
                        continue

                    price        = ctx['live_price']
                    day_open     = float(ctx.get('day_open') or price)
                    orb_h        = ctx['orb_high']
                    ma50         = ctx['ma50']
                    ma200        = ctx['ma200']
                    rsi          = ctx['rsi']
                    rsi_p        = ctx['rsi_prev']
                    atr          = ctx['atr']
                    atr_chand    = ctx.get('atr_chandelier', atr)
                    atr5         = ctx.get('atr5', atr)
                    atr20        = ctx.get('atr20', atr)
                    sma200_slope = ctx.get('sma200_slope', 0.0)
                    rvol         = ctx.get('rvol', 0.0)
                    day_loc      = ctx.get('day_range_location')
                    intraday_gain = ctx.get('intraday_gain')
                    spread_pct   = ctx.get('spread_pct', 0.0)
                    dol_vol_20d  = ctx['dollar_vol_20d']
                    atr_ratio    = (atr5 / atr20) if atr20 > 0 else float('nan')

                    # ── Production entry filter ───────────────────────────────
                    c_trend    = price > ma50 and ma50 > ma200
                    c_slope    = sma200_slope > 0
                    c_vcp      = atr20 > 0 and (atr5 / atr20) < vcp_ratio
                    c_rvol     = rvol >= rvol_min
                    c_spread   = spread_pct <= SPREAD_MAX_PCT
                    c_dol_vol  = dol_vol_20d >= dol_vol_threshold
                    c_orb      = price > orb_h
                    c_gap      = day_open <= orb_h * (1 + gap_max_pct)
                    c_rsi_rise  = rsi > rsi_p
                    c_rsi_delta = (rsi - rsi_p) >= rsi_min_delta
                    c_rsi_lvl   = rsi > rsi_threshold
                    c_day_loc   = (
                        day_loc is not None
                        and not pd.isna(day_loc)
                        and day_loc >= DAY_RANGE_LOCATION_MIN
                    )
                    c_open_gain = (
                        intraday_gain is not None
                        and not pd.isna(intraday_gain)
                        and intraday_gain >= INTRADAY_GAIN_MIN
                    )
                    c_atr_pct = (
                        price > 0
                        and atr_chand > 0
                        and (atr_chand / price) <= ATR_PCT_MAX
                    )

                    scan_detail = (
                        f"SCAN {sym} [{regime_label}]: price=${price:.2f} ORB=${orb_h:.2f} "
                        f"MA50=${ma50:.2f} MA200=${ma200:.2f} SMA200slope={sma200_slope:+.3f} "
                        f"ATR5/ATR20={atr_ratio:.2f}(vcp<{vcp_ratio}) "
                        f"VolPace={rvol:.1f}(≥{rvol_min}) rawRVOL={ctx.get('rvol_raw', rvol):.1f} "
                        f"spread={spread_pct*100:.2f}%(≤{SPREAD_MAX_PCT*100:.1f}%) "
                        f"Open=${day_open:.2f} OpenGap={(day_open/orb_h - 1) if orb_h > 0 else float('nan'):+.2%}(≤{gap_max_pct:.0%}) "
                        f"DayLoc={day_loc if day_loc is not None else float('nan'):.2f}(≥{DAY_RANGE_LOCATION_MIN:.2f}) "
                        f"OpenGain={intraday_gain if intraday_gain is not None else float('nan'):+.2%}(≥{INTRADAY_GAIN_MIN:.1%}) "
                        f"RSI={rsi:.1f}(Δ{rsi-rsi_p:+.1f}) "
                        f"ATR=${atr:.2f} ATR%={atr_chand/price if price > 0 else float('nan'):.2%}(≤{ATR_PCT_MAX:.0%}) "
                        f"DolVol20d=${dol_vol_20d/1e6:.0f}M(thr=${dol_vol_threshold/1e6:.0f}M) | "
                        f"InfoTrend={c_trend} InfoSlope={c_slope} InfoVCP={c_vcp} "
                        f"InfoVolPace={c_rvol} Spread={c_spread} "
                        f"DayLoc={c_day_loc} OpenGain={c_open_gain} ATRPct={c_atr_pct} "
                        f"ORB={c_orb} Gap={c_gap} InfoRSI↑={c_rsi_rise} "
                        f"RSIΔ≥{rsi_min_delta}={c_rsi_delta} RSI>{rsi_threshold}={c_rsi_lvl} "
                        f"DolVol={c_dol_vol}"
                    )

                    if not (c_spread and c_dol_vol
                            and c_orb and c_gap
                            and c_rsi_delta and c_rsi_lvl
                            and c_day_loc and c_open_gain and c_atr_pct):
                        failed = [n for n, v in [
                            (f'Spread≤{SPREAD_MAX_PCT*100:.1f}%', c_spread),
                            (f'DolVol≥{dol_vol_threshold/1e6:.0f}M', c_dol_vol),
                            (f'DayLoc≥{DAY_RANGE_LOCATION_MIN:.2f}', c_day_loc),
                            (f'OpenGain≥{INTRADAY_GAIN_MIN:.1%}', c_open_gain),
                            (f'ATR%≤{ATR_PCT_MAX:.0%}', c_atr_pct),
                            ('price>ORB', c_orb),
                            (f'open gap≤{gap_max_pct*100:.0f}%', c_gap),
                            (f'RSIΔ≥{rsi_min_delta}', c_rsi_delta),
                            (f'RSI>{rsi_threshold}', c_rsi_lvl),
                        ] if not v]
                        logger.debug(f"{scan_detail}")
                        logger.debug(f"SCAN {sym}: NO SIGNAL — failed: {failed}")
                        if not c_dol_vol:
                            self._remember_daily_scan_skip(
                                sym,
                                f"DolVol20d ${dol_vol_20d/1e6:.0f}M "
                                f"< threshold ${dol_vol_threshold/1e6:.0f}M"
                            )
                        continue

                    # ── Correlation filter (expensive — only for passing candidates) ──
                    df_daily = ctx.get('df_daily')
                    if df_daily is not None and self.state:
                        max_corr = self._compute_book_correlation(sym, df_daily)
                        if max_corr > CORR_MAX:
                            logger.debug(
                                f"SCAN {sym}: SKIP — correlation {max_corr:.2f} > {CORR_MAX} with book"
                            )
                            continue

                    # ── Sector clustering filter ──────────────────────────────
                    sector = self._get_sector(sym, ctx['contract'])
                    if book_sectors.get(sector, 0) >= MAX_SECTOR_COUNT:
                        logger.debug(
                            f"SCAN {sym}: SKIP — sector '{sector}' already has "
                            f"{book_sectors[sector]}/{MAX_SECTOR_COUNT} positions"
                        )
                        continue

                    score = self._score_candidate(
                        ctx,
                        volume_floor=rvol_min,
                        gap_max_pct=gap_max_pct,
                    )
                    signals.append((score, sym, ctx))
                    logger.info(scan_detail)
                    logger.info(
                        f"SIGNAL {sym} [{regime_label}]: score={score:.1f}/100 | "
                        f"VolPace={rvol:.1f}x rawRVOL={ctx.get('rvol_raw', rvol):.1f}x "
                        f"trend_sep={(ma50-ma200)/ma200*100:.1f}% "
                        f"RSI_delta={rsi-rsi_p:.1f} spread={spread_pct*100:.2f}% "
                        f"DolVol20d=${dol_vol_20d/1e6:.0f}M"
                    )

                # ── Phase 2: rank by score; enter in descending order, falling
                #    through to the next eligible stock when one is skipped ──────
                if not signals:
                    logger.info("SCAN: No signals this cycle")
                else:
                    signals.sort(key=lambda x: x[0], reverse=True)
                    ranked = [(s, sym) for s, sym, _ in signals]
                    logger.info(f"RANKED: {ranked} — attempting up to {min(slots, len(signals))}")

                    placed = 0
                    for score, sym, ctx in signals:
                        if placed >= slots:
                            break
                        if placed > 0:
                            self.manage_position_exits()

                        atr   = ctx['atr']

                        # Re-fetch price if the scan price is stale -- the scan
                        # phase can take several minutes across a broad candidate list.
                        quote_bid = ctx.get('bid')
                        quote_ask = ctx.get('ask')
                        fetched_at = ctx.get('price_fetched_at', datetime.now(_TZ_NY))
                        age_s = (datetime.now(_TZ_NY) - fetched_at).total_seconds()
                        if age_s > ENTRY_REPRICE_MAX_AGE_SEC:
                            try:
                                t2         = self.ib.reqTickers(ctx['contract'])[0]
                                new_price  = t2.marketPrice()
                                if pd.isna(new_price):
                                    new_price = t2.last
                                if pd.isna(new_price):
                                    new_price = t2.close
                                new_price = self._coerce_positive_price(new_price)
                                new_bid = self._coerce_positive_price(getattr(t2, 'bid', None))
                                new_ask = self._coerce_positive_price(getattr(t2, 'ask', None))
                                if new_price is not None and new_bid is not None and new_ask is not None:
                                    drift = (
                                        abs(float(new_price) - float(ctx['live_price']))
                                        / float(ctx['live_price'])
                                    )
                                    if drift > ENTRY_MAX_PRICE_DRIFT_PCT:
                                        logger.warning(
                                            f"SKIP {sym}: scan-to-order price drift "
                                            f"{drift*100:.2f}% exceeds "
                                            f"{ENTRY_MAX_PRICE_DRIFT_PCT*100:.1f}% cap"
                                        )
                                        continue
                                    price = new_price
                                    quote_bid = new_bid
                                    quote_ask = new_ask
                                    logger.debug(
                                        f"REPRICE {sym}: ${ctx['live_price']:.2f} -> ${price:.2f} "
                                        f"bid=${quote_bid:.2f} ask=${quote_ask:.2f}"
                                    )
                                else:
                                    logger.warning(
                                        f"SKIP {sym}: stale scan price or bid/ask unavailable for reprice"
                                    )
                                    continue
                            except Exception:
                                logger.warning(
                                    f"SKIP {sym}: stale scan price and live reprice failed"
                                )
                                continue
                        else:
                            price = ctx['live_price']

                        if not self._entry_price_is_still_valid(
                            sym, ctx, float(price), gap_max_pct=gap_max_pct
                        ):
                            continue

                        # Spread-aware marketable limit: price from validated ask,
                        # capped by a small max-over-market guard. This is only
                        # allowed with real-time market data; delayed data is blocked
                        # before the entry window.
                        limit_price = self._calc_entry_limit_price(price, quote_bid, quote_ask)
                        if limit_price is None:
                            logger.warning(
                                f"SKIP {sym}: unusable bid/ask for entry limit "
                                f"(price={price}, bid={quote_bid}, ask={quote_ask})"
                            )
                            continue

                        if np.isnan(atr) or atr <= 0:
                            logger.warning(f"SKIP {sym}: ATR invalid ({atr:.4f}), skipping")
                            continue

                        atr_chandelier  = ctx.get('atr_chandelier', atr)
                        chandelier_dist = round(atr_chandelier * CHANDELIER_MULT, 2)
                        hard_stop_dist  = round(price * HARD_STOP_PCT, 2)
                        risk_stop_dist  = min(chandelier_dist, hard_stop_dist)
                        if np.isnan(risk_stop_dist) or risk_stop_dist <= 0:
                            logger.warning(
                                f"SKIP {sym}: invalid risk stop distance "
                                f"(chandelier=${chandelier_dist:.2f}, hard=${hard_stop_dist:.2f})"
                            )
                            continue

                        # Whole shares only. Size by the stricter of capital bucket
                        # and true dollars-at-risk to keep live sizing aligned with
                        # the backtest risk model. Settled cash must cover the whole
                        # intended order; otherwise skip rather than send dust trades.
                        bucket_qty = int(bucket_size / limit_price)
                        risk_qty   = int((equity * risk_per_trade_pct) / risk_stop_dist)
                        qty        = min(bucket_qty, risk_qty)
                        if qty < 1:
                            logger.warning(
                                f"SKIP {sym}: no whole-share size after risk/cash caps "
                                f"(bucket_qty={bucket_qty}, risk_qty={risk_qty}, "
                                f"risk_dist=${risk_stop_dist:.2f})"
                            )
                            continue

                        order_cost = round(qty * limit_price, 2)
                        if settled < order_cost:
                            logger.warning(
                                f"SKIP {sym}: insufficient settled cash for intended size "
                                f"(need ${order_cost:.2f}, have ${settled:.2f}, qty={qty})"
                            )
                            continue

                        # goodAfterTime must not be set to a time already in
                        # the past, or IBKR can reject the order as invalid.
                        gat_str = self._entry_good_after_time()

                        buy_order               = Order()
                        buy_order.action        = 'BUY'
                        buy_order.orderType     = 'LMT'
                        buy_order.totalQuantity = qty
                        buy_order.lmtPrice      = limit_price
                        buy_order.tif           = ENTRY_PARENT_TIF
                        buy_order.allOrNone     = ENTRY_ALL_OR_NONE
                        buy_order.goodAfterTime = gat_str
                        buy_order.transmit      = True   # standalone BUY, stop attached after fill

                        if not self._preflight_order(ctx['contract'], buy_order, sym):
                            logger.warning(
                                f"SKIP {sym}: BUY LMT pre-flight rejected by IB — skipping"
                            )
                            continue

                        parent_trade = self.ib.placeOrder(ctx['contract'], buy_order)

                        # Wait until IB confirms a terminal state (Filled / Cancelled /
                        # ApiCancelled / Inactive), up to 30 s.  loopUntil() returns
                        # immediately once isDone() is True — much faster than a fixed
                        # sleep for quick limit fills; also avoids a stale-status read.
                        for _ in self.ib.loopUntil(parent_trade.isDone, timeout=30):
                            pass
                        status = parent_trade.orderStatus.status
                        filled = parent_trade.orderStatus.filled
                        logger.info(f"ORDER STATUS: {sym} → {status} (filled={filled})")

                        try:
                            filled_qty = float(filled or 0)
                        except (TypeError, ValueError):
                            filled_qty = float(qty) if status == 'Filled' else 0.0

                        if 0 < filled_qty < qty:
                            self._alert(
                                "ERROR",
                                f"PARTIAL FILL: {sym} filled={filled_qty:g}/{qty:g} "
                                f"status={status}. Rebuilding protection from IBKR state."
                            )

                        # Only an actual fill creates a position. Submitted or
                        # PreSubmitted is still just an order, not inventory.
                        if status != 'Filled' or filled_qty <= 0:
                            logger.warning(
                                f"ORDER NOT FILLED: {sym} status={status} filled={filled_qty} "
                                f"qty={qty} limit=${limit_price:.2f}. Cancelling BUY "
                                f"and trying next candidate."
                            )
                            try:
                                self.ib.cancelOrder(parent_trade.order)
                            except Exception:
                                pass
                            self.ib.sleep(1)
                            # If IB reports a late partial fill during cancellation,
                            # reconcile it immediately and ensure a protective stop.
                            self._sync_positions_from_ibkr()
                            if sym in self.state:
                                self._audit_stop_orders()
                            continue

                        fill_price = (
                            self._coerce_positive_price(parent_trade.orderStatus.avgFillPrice)
                            or limit_price
                        )
                        self.state[sym] = {
                            'fill_price':     fill_price,
                            'price':          fill_price,
                            'entry_order_id': parent_trade.order.orderId,
                            'time':           datetime.now(_TZ_NY).isoformat(),
                            'qty':            filled_qty,
                            'stop_loss':      round(fill_price - chandelier_dist, 2),
                            'stop_dist':      chandelier_dist,
                            'peak_price':     fill_price,
                            'volume':         ctx.get('volume', 0),
                            'score':          score,
                            'regime':         regime_label.lower(),
                            'protection_status': 'pending',
                            'protection_reason': 'awaiting_trail_stop_confirmation',
                        }
                        # Commission report sometimes lands synchronously with the fill.
                        # Capture it now if available; _on_commission_report handles it
                        # if it arrives later during an ib.sleep() or subsequent cycle.
                        if parent_trade.fills:
                            cr = parent_trade.fills[0].commissionReport
                            if cr and not np.isnan(cr.commission) and cr.commission > 0:
                                self.state[sym]['commission'] = round(float(cr.commission), 4)
                        self.save_state()

                        # Place the protective TRAIL stop as a standalone order AFTER
                        # the position is confirmed in IBKR's books. Attaching it as a
                        # bracket child (parentId) is rejected in cash accounts with
                        # "Short stock positions can only be held in a margin account"
                        # because the broker evaluates the child SELL before the parent
                        # BUY settles as a long position.
                        stop_order               = Order()
                        stop_order.action        = 'SELL'
                        stop_order.orderType     = 'TRAIL'
                        stop_order.totalQuantity = filled_qty
                        stop_order.auxPrice      = chandelier_dist
                        stop_order.tif           = 'GTC'
                        stop_order.goodAfterTime = self._stop_good_after_time()
                        stop_order.transmit      = True

                        stop_placed = False
                        stop_trade = None
                        audit_ran = False
                        if self._preflight_order(
                            ctx['contract'], stop_order, sym,
                            allow_protective_sell_fail_open=True,
                        ):
                            stop_trade = self.ib.placeOrder(ctx['contract'], stop_order)
                            self.ib.sleep(2)   # let IB propagate acceptance/rejection
                            stop_status = getattr(stop_trade.orderStatus, 'status', '')
                            if stop_status in _REJECTED_ORDER_STATUSES:
                                self._metric_inc('protective_stop_rejected')
                                logger.error(
                                    f"STOP REJECTED: {sym} TRAIL status={stop_status}. "
                                    "Running immediate stop audit."
                                )
                                self._audit_stop_orders()
                                audit_ran = True
                            else:
                                stop_placed = True
                        else:
                            logger.warning(
                                f"STOP PREFLIGHT FAILED: {sym} — running audit to place protection."
                            )
                            self._audit_stop_orders()
                            audit_ran = True

                        if stop_placed and abs(float(filled_qty) - float(qty)) > 1e-6:
                            logger.warning(
                                f"PARTIAL FILL PROTECTION: {sym} filled={filled_qty:g}/{qty:g}; "
                                "cancelling stop and rebuilding for actual filled quantity."
                            )
                            try:
                                self.ib.cancelOrder(stop_order)
                            except Exception as e:
                                logger.warning(f"PARTIAL FILL {sym}: stop cancel failed: {e}")
                            self.ib.sleep(1)
                            self._audit_stop_orders()
                            audit_ran = True

                        protection_confirmed = False
                        if stop_placed and not audit_ran:
                            protection_confirmed = self._confirm_protective_stop(
                                sym,
                                filled_qty,
                                known_trade=stop_trade,
                                expected_order=stop_order,
                            )
                        if not protection_confirmed:
                            if not audit_ran:
                                logger.warning(
                                    f"STOP CONFIRM: {sym} protective TRAIL not visible yet; "
                                    "running immediate stop audit."
                                )
                                self._audit_stop_orders()
                                audit_ran = True
                            protection_confirmed = self._confirm_protective_stop(
                                sym,
                                filled_qty,
                                timeout=PROTECTIVE_STOP_CONFIRM_TIMEOUT_SEC,
                            )

                        if protection_confirmed:
                            self._mark_position_protection(
                                sym,
                                'confirmed',
                                order_id=getattr(getattr(stop_trade, 'order', None), 'orderId', None)
                                if stop_trade is not None else None,
                            )
                        else:
                            self._mark_position_protection(
                                sym,
                                'unconfirmed',
                                'protective_stop_confirmation_timeout',
                            )
                            self._alert(
                                "CRITICAL",
                                f"STOP UNCONFIRMED: {sym} BUY filled qty={filled_qty:g}, "
                                "but no valid TRAIL SELL was confirmed after audit. "
                                "No further entries will be attempted this cycle.",
                            )

                        actual_order_cost = round(float(filled_qty) * float(fill_price), 2)
                        settled -= actual_order_cost
                        placed  += 1
                        # Recalculate bucket for any further entries in this cycle.
                        # The first order may have used less than one full bucket
                        # (ATR-capped), so the remaining cash could support a larger
                        # bucket per remaining slot.
                        allocation = self._calc_entry_allocation(equity, settled, len(self.state))
                        max_pos        = int(allocation['max_pos'])
                        capacity_slots = int(allocation['capacity_slots'])
                        cash_slots     = int(allocation['cash_slots'])
                        open_slots     = int(allocation['entry_slots'])
                        bucket_size    = allocation['bucket_size']
                        logger.info(
                            f"ORDER CONFIRMED: {sym} [{regime_label}] | Score={score:.1f} Qty={filled_qty:g} "
                            f"ScanPrice=${price:.2f} Limit=${limit_price:.2f} "
                            f"FillPrice=${fill_price:.2f} "
                            f"Commission={'$'+str(self.state[sym]['commission']) if 'commission' in self.state[sym] else 'pending'} "
                            f"ChandelierStop=${round(fill_price-chandelier_dist,2):.2f} "
                            f"(dist=${chandelier_dist:.2f}, risk_dist=${risk_stop_dist:.2f}, "
                            f"risk={risk_per_trade_pct*100:.1f}%) | "
                            f"Protection={'confirmed' if protection_confirmed else 'UNCONFIRMED'} | "
                            f"Settled remaining=${settled:.2f}"
                        )
                        if not protection_confirmed:
                            break

        # Refresh live prices for dashboard unrealized P&L
        self._update_position_prices()

    # ── Headless event loop ───────────────────────────────────────────────────
    def run(self):
        logger.info("=" * 40)
        logger.info("ENGINE DEPLOYED")
        logger.info("=" * 40)
        self._initialize()
        logger.info("ENGINE READY: Starting main loop...")
        while True:
            try:
                self.run_cycle()

                now_ny             = datetime.now(_TZ_NY)
                self._last_scan_ts = now_ny.strftime("%H:%M:%S %Z")
                self._next_scan_dt = (
                    now_ny + timedelta(seconds=SCAN_INTERVAL)
                ).isoformat()
                self._write_dashboard_data(connected=True)

                self._safe_sleep(SCAN_INTERVAL, context="main loop")

            except Exception as e:
                logger.exception("RUNTIME ERROR")
                self._alert("ERROR", f"RUNTIME ERROR: {e}")
                self._next_scan_dt = (
                    datetime.now(_TZ_NY) + timedelta(seconds=ERROR_WAIT)
                ).isoformat()
                self._write_dashboard_data(connected=self.ib.isConnected())
                self._safe_sleep(ERROR_WAIT, context="error backoff")
