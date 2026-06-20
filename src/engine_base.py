import copy
import json
import logging
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, time as datetime_time, timedelta
from logging.handlers import TimedRotatingFileHandler
from typing import Dict, Optional, Tuple

import numpy as np
import pytz
from ib_async import IB, Index, Stock, util

from src.config import (
    BASE_DIR, STATE_FILE, DASHBOARD_FILE, EQUITY_HIST_FILE, READINESS_FILE,
    HEALTH_REPORT_FILE, HALT_FILE, FORCE_EXIT_FILE,
    LOG_DIR, LOG_FILE,
    IB_HOST, IB_PORT, IB_CLIENT_ID, MARKET_DATA_TYPE,
    VIX_THRESHOLD,
    TRADING_MODE, LIVE_TRADING_ACK, LIVE_TRADING_ACK_PHRASE, LIVE_IB_PORTS, PAPER_IB_PORTS,
    ALERT_WEBHOOK_URL, ALERT_TIMEOUT_SEC,
    IB_ERROR_DEDUP_WINDOW_SEC,
    RECONNECT_INITIAL_WAIT_SEC, RECONNECT_MAX_WAIT_SEC, ALERT_DEDUP_WINDOW_SEC,
    CONNECT_MAX_ATTEMPTS,
    ACCOUNT_CURRENCY,
    LOG_BACKUP_COUNT,
    ERROR_WAIT,
    APP_SCANNER_BATCH_SIZE, APP_SCANNER_SOURCE,
    STRATEGY_PROFILE,
)
from src.analyst_ratings import AnalystRatingProvider
from src.ib_gateway import ensure_ib_gateway_ready
from src.strategy_profiles import get_strategy_profile

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Single timezone object reused throughout the module.
# The machine runs on IST (UTC+5:30); all times must be anchored to US/Eastern
# so market-hours checks, timestamps, and log lines are unambiguous.
_TZ_NY = pytz.timezone('US/Eastern')
_REJECTED_ORDER_STATUSES = {'Inactive', 'ApiCancelled', 'Cancelled'}


class AccountDataUnavailable(RuntimeError):
    """Raised when IBKR account summary data is not fresh enough for entries."""


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


class VelocityEngineBase:
    def __init__(self):
        self.ib           = IB()
        self.state        = self.load_state()
        self._strategy_profile = get_strategy_profile(STRATEGY_PROFILE)
        self._analyst_provider = AnalystRatingProvider(allow_remote=True)

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
        self._scanner_universe_symbols: list[str] = []
        self._scanner_universe_offset: int = 0
        self._scanner_universe_date: Optional[str] = None
        self._prefilter_date: Optional[str] = None
        self._prefilter_status: str = "not_started"
        self._prefilter_candidates: list[str] = []
        self._prefilter_stats: dict = {}
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
        # Last trading date the pre-entry confirmation sync/audit was run.
        self._last_pre_entry_sync_date: Optional[str] = None
        # Last trading date the post-open protective audit was run.
        self._last_post_open_audit_date: Optional[str] = None
        # Last trading dates for non-trading operational jobs.
        self._last_premarket_prefilter_date: Optional[str] = None
        self._last_premarket_readiness_date: Optional[str] = None
        self._last_post_close_maintenance_date: Optional[str] = None
        # Daily EOD quality cleanup: track the date so the exit fires once per trading day.
        self._last_eod_exit_date: Optional[str] = None
        # Avoid deleting state on one transient/partial IBKR positions snapshot.
        self._missing_position_counts: Dict[str, int] = {}
        # Daily operational counters written to HEALTH_REPORT_FILE. These are
        # deliberately lightweight so a long paper run can be reviewed from one
        # compact JSON summary instead of only raw logs.
        self._health_date = datetime.now(_TZ_NY).strftime('%Y-%m-%d')
        self._health_metrics = self._new_health_metrics()

        # Fix 9: IB error deduplication — {errorCode: (last_logged_ts, suppressed_count)}
        self._ib_error_dedup: Dict[int, tuple] = {}
        # Fix 6: Alert deduplication — {message_key: last_sent_ts}
        self._alert_dedup_cache: Dict[str, float] = {}
        # Market data blackout tracking (scanner-level)
        self._data_blackout_streak: int = 0
        self._data_blackout_alerted: bool = False
        # Per-symbol consecutive miss counts for position-level price blackout
        self._exit_price_miss_counts: Dict[str, int] = {}
        # Shared log-once-per-day cache — {"YYYY-MM-DD:key": date_str}
        self._log_once_cache: Dict[str, str] = {}
        # Per-symbol daily indicator row cache — {sym: (monotonic_ts, row_or_None)}
        self._indicator_row_cache: Dict[str, tuple] = {}

        self.connect()

    # ── IB connection ──────────────────────────────────────────────────────────
    def connect(self):
        """Connect to IB Gateway with bounded in-process retry and backoff.

        Replaces the old single-attempt pattern that called sys.exit() on any
        failure — a transient Gateway/IBC startup race caused 319 full-process
        deaths in two days. Now retries CONNECT_MAX_ATTEMPTS times with the
        same exponential backoff used by _reconnect(), only exiting after the
        hard maximum is exhausted.
        """
        self._validate_deployment_mode()
        for attempt in range(1, CONNECT_MAX_ATTEMPTS + 1):
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
                    f"(mode={TRADING_MODE}, host={IB_HOST}, port={IB_PORT}, "
                    f"clientId={IB_CLIENT_ID}"
                    + (f", attempt={attempt}" if attempt > 1 else "")
                    + ")."
                )
                self._write_dashboard_data(connected=True)
                self._warmup_historical_data(reason="connect")
                return  # success
            except Exception as e:
                self._alert("CRITICAL", f"CONNECTION FAILED: Is IB Gateway open? {e}")
                if attempt >= CONNECT_MAX_ATTEMPTS:
                    logger.error(
                        f"CONNECT: all {CONNECT_MAX_ATTEMPTS} attempts failed — exiting"
                    )
                    sys.exit()
                wait_s = min(
                    RECONNECT_INITIAL_WAIT_SEC * (2 ** (attempt - 1)),
                    RECONNECT_MAX_WAIT_SEC,
                )
                logger.warning(
                    f"CONNECT: attempt {attempt}/{CONNECT_MAX_ATTEMPTS} failed ({e}); "
                    f"retrying in {wait_s:.0f}s"
                )
                if self.ib.isConnected():
                    try:
                        self.ib.disconnect()
                    except Exception:
                        pass
                time.sleep(wait_s)

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
        if errorCode == 165 and "items retrieved" in str(errorString).lower():
            return
        self._metric_inc('ib_errors')
        self._metric_inc('ib_error_codes', subkey=str(errorCode))

        # Fix 9: Deduplicate repeated error codes within IB_ERROR_DEDUP_WINDOW_SEC.
        # Reconnect storms can produce hundreds of identical errors per minute;
        # we log the first occurrence, suppress duplicates, then log a summary
        # when the window expires.
        now_ts = time.time()
        last_ts, suppressed = self._ib_error_dedup.get(errorCode, (0.0, 0))
        if now_ts - last_ts < IB_ERROR_DEDUP_WINDOW_SEC:
            self._ib_error_dedup[errorCode] = (last_ts, suppressed + 1)
            return
        if suppressed > 0:
            logger.warning(
                f"IB error {errorCode}: suppressed {suppressed} duplicate(s) "
                f"in the last {IB_ERROR_DEDUP_WINDOW_SEC:.0f}s; resuming normal logging"
            )
        self._ib_error_dedup[errorCode] = (now_ts, 0)

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
                # Fix 6: Exponential backoff — doubles each attempt, capped at max.
                wait_s = min(RECONNECT_INITIAL_WAIT_SEC * (2 ** (attempt - 1)), RECONNECT_MAX_WAIT_SEC)
                logger.warning(
                    f"RECONNECT: attempt {attempt}/10 failed: {e} "
                    f"(retrying in {wait_s:.0f}s)"
                )
                time.sleep(wait_s)
        self._metric_inc('reconnect_failures')
        self._alert("CRITICAL", "RECONNECT: all 10 attempts failed — skipping trading cycles until restored")
        return False

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

    def _log_once_per_day(self, key: str, level: str, msg: str) -> None:
        """Emit msg at the given log level at most once per calendar day per key."""
        today = datetime.now(_TZ_NY).strftime('%Y-%m-%d')
        cache_key = f"{today}:{key}"
        if cache_key in self._log_once_cache:
            return
        self._log_once_cache[cache_key] = today
        getattr(logger, level.lower(), logger.info)(msg)

    def _alert(self, severity: str, message: str):
        """Send high-priority operational alerts without adding external deps."""
        severity = severity.upper()
        self._metric_inc('alerts', subkey=severity.lower())

        # Fix 6: Rate-limit CRITICAL/ERROR alerts so prolonged outages (e.g. a
        # Saturday gateway crash loop) don't flood logs and webhooks.  Only the
        # first occurrence within the deduplication window is logged and sent;
        # the next occurrence after the window expires resets the clock.
        if severity in {'CRITICAL', 'ERROR'} and ALERT_DEDUP_WINDOW_SEC > 0:
            alert_key = f"{severity}:{message[:100]}"
            now_ts = time.time()
            if now_ts - self._alert_dedup_cache.get(alert_key, 0) < ALERT_DEDUP_WINDOW_SEC:
                self._metric_inc('alerts', subkey='deduplicated')
                return
            self._alert_dedup_cache[alert_key] = now_ts

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

    def _operator_halt_active(self) -> bool:
        """Manual kill switch: HALT_FILE presence blocks new entries only."""
        return os.path.exists(HALT_FILE)

    def _force_exit_active(self) -> bool:
        """Emergency kill switch: FORCE_EXIT_FILE liquidates all broker positions."""
        return os.path.exists(FORCE_EXIT_FILE)

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
            'prefilter_runs': 0,
            'prefilter_processed': 0,
            'prefilter_candidates': 0,
            'prefilter_rejected': 0,
            'prefilter_failures': 0,
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
                'source': self._normalise_app_scanner_source(APP_SCANNER_SOURCE),
                'universe_size': len(getattr(self, '_scanner_universe_symbols', []) or []),
                'universe_offset': int(getattr(self, '_scanner_universe_offset', 0) or 0),
                'universe_batch_size': int(APP_SCANNER_BATCH_SIZE or 0),
                'prefilter': {
                    'date': getattr(self, '_prefilter_date', None),
                    'status': getattr(self, '_prefilter_status', 'not_started'),
                    'candidates': len(getattr(self, '_prefilter_candidates', []) or []),
                    'stats': copy.deepcopy(getattr(self, '_prefilter_stats', {}) or {}),
                },
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
        number = VelocityEngineBase._coerce_positive_price(value)
        if number is None:
            return None
        if number >= util.UNSET_DOUBLE * 0.5:
            return None
        return number

    @staticmethod
    def _round_up_to_cent(value: float) -> float:
        """Round up to the nearest US equity penny tick."""
        return round(float(np.ceil(float(value) * 100.0 - 1e-9) / 100.0), 2)

    @staticmethod
    def _round_down_to_cent(value: float) -> float:
        """Round down to the nearest US equity penny tick."""
        return round(float(np.floor(float(value) * 100.0 + 1e-9) / 100.0), 2)

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
            "scanner_source": self._normalise_app_scanner_source(APP_SCANNER_SOURCE),
            "scanner_universe_size": len(getattr(self, '_scanner_universe_symbols', []) or []),
            "scanner_universe_offset": int(getattr(self, '_scanner_universe_offset', 0) or 0),
            "scanner_universe_batch_size": int(APP_SCANNER_BATCH_SIZE or 0),
            "scanner_prefilter_date": getattr(self, '_prefilter_date', None),
            "scanner_prefilter_status": getattr(self, '_prefilter_status', 'not_started'),
            "scanner_prefilter_candidates": len(getattr(self, '_prefilter_candidates', []) or []),
            "scanner_prefilter_stats": copy.deepcopy(getattr(self, '_prefilter_stats', {}) or {}),
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
