"""
engine.py — thin orchestrator

VelocityEngine is composed from six cohesive mixin classes:
  EntriesMixin    – account values, position sync, sizing, scoring, lifecycle
  ScannerMixin    – universe, prefilter, technical context
  OrdersMixin     – trail stops, stop audit, liquidation
  ExitsMixin      – all software exit conditions
  MarketDataMixin – VIX, SPY trend, historical bars, correlations
  VelocityEngineBase – __init__, IB connection, alerting, health, state I/O

Only run_cycle() and run() live here.  Everything else is inherited.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from ib_async import Order

from src.config import (
    FORCE_EXIT_FILE, HALT_FILE,
    MARKET_DATA_TYPE,
    VIX_THRESHOLD,
    SETTLED_CASH_DEPLOYMENT_PCT,
    ENTRY_START, ENTRY_END,
    FRIDAY_ENTRY_CUTOFF_TIME,
    MAX_DAILY_LOSS_PCT,
    TRAIL_PCT, HARD_STOP_PCT,
    RISK_PER_TRADE_PCT, BEAR_PHASE_RISK_MULT, BEAR_PHASE_DOLLAR_VOL_MULT,
    BEAR_PHASE_TRADING_ENABLED,
    ENTRY_PARENT_TIF, ENTRY_ALL_OR_NONE,
    ENTRY_REPRICE_MAX_AGE_SEC, ENTRY_MAX_PRICE_DRIFT_PCT,
    PROTECTIVE_STOP_CONFIRM_TIMEOUT_SEC,
    VOL_MULT_FRIDAY,
    CORR_MAX, MAX_SECTOR_COUNT,
    LATE_ENTRY_CUTOFF_TIME, LATE_ENTRY_MIN_SCORE,
    DATA_BLACKOUT_RATIO_THRESHOLD, DATA_BLACKOUT_MIN_CANDIDATES, DATA_BLACKOUT_STREAK_ALERT,
    STRATEGY_PROFILE,
    SCAN_INTERVAL, ERROR_WAIT,
    DATA_BLACKOUT_RATIO_THRESHOLD, DATA_BLACKOUT_MIN_CANDIDATES, DATA_BLACKOUT_STREAK_ALERT,
)
from src.strategy_profiles import (
    evaluate_entry_rules,
    get_strategy_profile,
    indicator_sleeve_label,
    select_entry_strategy,
)
import time  # re-exported: tests patch time.time via src.engine.time

from ib_async import IB, util  # re-exported for test modules that patch via src.engine
from src.config import (  # re-exported: conftest + test modules patch these via src.engine
    IB_CLIENT_ID, STATE_FILE,
    DASHBOARD_FILE, EQUITY_HIST_FILE, READINESS_FILE, HEALTH_REPORT_FILE,
    HALT_FILE, FORCE_EXIT_FILE, LOG_DIR, LOG_FILE,
    PROTECTIVE_STOP_CONFIRM_TIMEOUT_SEC, PROTECTIVE_STOP_CONFIRM_POLL_SEC,
    TRADING_MODE, LIVE_TRADING_ACK, LIVE_TRADING_ACK_PHRASE, IB_PORT,
    APP_SCANNER_SOURCE,
)

from src.engine_base import (
    VelocityEngineBase,
    AccountDataUnavailable,
    logger,
    _TZ_NY,
    _REJECTED_ORDER_STATUSES,
    _EasternTimedRotatingFileHandler,
    _log_namer,
    ensure_ib_gateway_ready,
)
from src.engine_market import MarketDataMixin, HMDS_WARMUP_MAX_RETRIES
from src.engine_exits import ExitsMixin
from src.engine_orders import OrdersMixin
from src.engine_scanner import ScannerMixin
from src.engine_entries import EntriesMixin

# Re-export for backward compatibility — test modules import from src.engine
__all__ = [
    'VelocityEngine',
    'AccountDataUnavailable',
    'logger',
    '_TZ_NY',
    '_REJECTED_ORDER_STATUSES',
    'IB', 'util', 'IB_CLIENT_ID', 'STATE_FILE',
]


class VelocityEngine(
    EntriesMixin,
    ScannerMixin,
    OrdersMixin,
    ExitsMixin,
    MarketDataMixin,
    VelocityEngineBase,
):
    """
    Swing-trading engine.

    Inherits from all mixin classes.  Only the main event-loop methods
    (run_cycle and run) live here; all logic is in the mixins.
    """

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
                self.liquidate(sym, reason='force_exit_all')
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
        self._maybe_pre_entry_sync_audit()
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
            # Drop only stale-dated bar caches.  This rollover executes on the
            # first regular-session cycle (~09:32 ET), which is AFTER the 06:30
            # premarket prefilter has populated _bar_cache with clean completed
            # daily bars stamped with today's date.  A blanket clear() here
            # wiped those bars and forced mid-session re-fetches whose last row
            # was today's partial daily bar — corrupting volume/prev-high/RSI
            # for every signal for the rest of the day.
            self._bar_cache = {
                sym: entry for sym, entry in self._bar_cache.items()
                if isinstance(entry, dict) and entry.get('date') == today_str
            }
            getattr(self, '_exit_price_miss_counts', {}).clear()
            getattr(self, '_correlation_book_failures', set()).clear()
            self._prefilter_date = None
            self._prefilter_status = "not_started"
            self._prefilter_candidates = []
            self._prefilter_stats = {}
            # Fix 2: After resetting prefilter state (e.g. engine restart
            # mid-day), immediately try to restore today's candidates from the
            # on-disk cache so we don't re-run a 2.5-hour sieve unnecessarily.
            _pf_profile = getattr(self, "_strategy_profile", get_strategy_profile(STRATEGY_PROFILE))
            _cached = self._read_prefilter_cache(today_str, _pf_profile.name)
            if _cached and self._prefilter_status in {"complete", "partial"}:
                logger.info(
                    f"DAY RESET: restored {len(self._prefilter_candidates)} prefilter "
                    f"candidates from today's {self._prefilter_status} cache "
                    f"(skipping re-run)"
                )
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

        # 5. Entry Window
        tz_ny  = _TZ_NY
        now_ny = datetime.now(tz_ny)
        profile = getattr(self, "_strategy_profile", None)
        if profile is None:
            profile = get_strategy_profile(STRATEGY_PROFILE)
            self._strategy_profile = profile

        if now_ny.weekday() < 5 and ENTRY_START <= (now_ny.hour, now_ny.minute) <= ENTRY_END:
            # On Fridays raise the liquidity bar to 2× to avoid holding over weekends.
            is_friday = (now_ny.weekday() == 4)
            if is_friday and (now_ny.hour, now_ny.minute) >= FRIDAY_ENTRY_CUTOFF_TIME:
                self._metric_inc('scanner_skipped_friday_cutoff')
                # Fix 8: Log the cutoff warning exactly once per trading day so
                # it doesn't flood the log at every 60-second cycle.
                _cutoff_today = now_ny.strftime('%Y-%m-%d')
                if getattr(self, '_friday_cutoff_logged_date', None) != _cutoff_today:
                    self._friday_cutoff_logged_date = _cutoff_today
                    logger.warning(
                        f"FRIDAY ENTRY CUTOFF: no new entries after "
                        f"{FRIDAY_ENTRY_CUTOFF_TIME[0]:02d}:{FRIDAY_ENTRY_CUTOFF_TIME[1]:02d} ET. "
                        "Managing existing positions only."
                    )
                self._update_position_prices()
                return
            dol_vol_threshold = profile.min_dollar_vol * (VOL_MULT_FRIDAY if is_friday else 1.0)

            allocation = self._calc_entry_allocation(equity, settled, len(self.state))
            max_pos        = int(allocation['max_pos'])
            capacity_slots = int(allocation['capacity_slots'])
            cash_slots     = int(allocation['cash_slots'])
            slots          = int(allocation['entry_slots'])
            bucket_size    = allocation['bucket_size']
            if slots <= 0:
                self._metric_inc('scanner_skipped_no_slots')
                if capacity_slots > 0 and cash_slots <= 0:
                    # Account has open position capacity but no deployable settled cash.
                    # This is the "fully deployed / T+1 pending" state — log once per day
                    # rather than every 60-second cycle.
                    self._log_once_per_day(
                        'fully_deployed_t1',
                        'info',
                        f"SCAN: account fully deployed — awaiting T+1 settlement "
                        f"(equity=${equity:.2f}, settled=${settled:.2f}, "
                        f"positions={len(self.state)}/{max_pos}). "
                        "No new entries until settled cash replenishes.",
                    )
                else:
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
                if bear_phase and (
                    not BEAR_PHASE_TRADING_ENABLED
                    or not profile.allow_bear_phase_entries
                ):
                    logger.warning(
                        f"REGIME: SPY trend weak/falling — no new {profile.name} entries this cycle"
                    )
                    self._update_position_prices()
                    return

                if bear_phase:
                    regime_label = "BEAR"
                    dol_vol_threshold *= BEAR_PHASE_DOLLAR_VOL_MULT
                    risk_per_trade_pct = RISK_PER_TRADE_PCT * BEAR_PHASE_RISK_MULT
                    logger.warning(
                        "REGIME: SPY trend weak/falling — bear-phase participation enabled; "
                        f"profile={profile.name}; using stricter risk/liquidity "
                        f"(risk={risk_per_trade_pct*100:.1f}%)."
                    )
                else:
                    regime_label = "BULL"
                    risk_per_trade_pct = RISK_PER_TRADE_PCT

                watchlist = self.get_institutional_scan()
                self._metric_inc('scanner_runs')
                self._metric_inc('scanner_candidates', len(watchlist))
                logger.info(
                    f"SCAN: profile={profile.name} ({profile.label}) "
                    f"{len(watchlist)} candidates → {watchlist}"
                    + (f" [FRIDAY: dolVol threshold ${dol_vol_threshold/1e6:.0f}M]" if is_friday else "")
                )

                # Pre-compute sector counts for current book (for clustering check)
                book_sectors: dict = {}
                for book_sym in self.state:
                    if book_sym in self._contract_cache:
                        s = self._get_sector(book_sym, self._contract_cache[book_sym])
                        book_sectors[s] = book_sectors.get(s, 0) + 1

                # ── Phase 1: evaluate ALL candidates, collect those passing ──
                signals = []
                reject_counts = {
                    'already_held': 0,
                    'blocklisted': 0,
                    'cached_day_filter': 0,
                    'no_technical_data': 0,
                    'entry_filter': 0,
                    'correlation': 0,
                    'sector_limit': 0,
                }
                entry_filter_reasons: dict = {}
                for sym in watchlist:
                    if sym in self.state:
                        reject_counts['already_held'] += 1
                        logger.info(f"SCAN {sym}: SKIP — already in portfolio")
                        continue

                    from src.config import TICKER_BLOCKLIST
                    if sym in TICKER_BLOCKLIST:
                        reject_counts['blocklisted'] += 1
                        logger.debug(f"SCAN {sym}: SKIP — blocklisted (leveraged/inverse ETF)")
                        continue

                    if sym in self._daily_scan_skip:
                        reject_counts['cached_day_filter'] += 1
                        logger.debug(
                            f"SCAN {sym}: SKIP — day-filtered "
                            f"({self._daily_scan_skip[sym]})"
                        )
                        continue

                    ctx = self.get_technical_context(sym)
                    if not ctx:
                        reject_counts['no_technical_data'] += 1
                        logger.warning(f"SCAN {sym}: SKIP — no technical data")
                        continue

                    price         = ctx['live_price']
                    day_open      = float(ctx.get('day_open') or price)
                    ma50          = ctx['ma50']
                    ma200         = ctx['ma200']
                    rsi           = ctx['rsi']
                    rsi_p         = ctx['rsi_prev']
                    atr           = ctx['atr']
                    atr_chand     = ctx.get('atr_chandelier', atr)
                    atr5          = ctx.get('atr5', atr)
                    atr20         = ctx.get('atr20', atr)
                    sma200_slope  = ctx.get('sma200_slope', 0.0)
                    rvol          = ctx.get('rvol', 0.0)
                    day_loc       = ctx.get('day_range_location')
                    intraday_gain = ctx.get('intraday_gain')
                    spread_pct    = ctx.get('spread_pct', 0.0)
                    dol_vol_20d   = ctx['dollar_vol_20d']
                    atr_ratio     = (atr5 / atr20) if atr20 > 0 else float('nan')
                    atr_pct       = atr_chand / price if price > 0 else float('nan')
                    rs_63d        = ctx.get('relative_strength_63d', float('nan'))
                    ret_13w       = ctx.get('return_13w', float('nan'))
                    px_52w        = ctx.get('price_vs_52w_high', float('nan'))
                    analyst_score = ctx.get('analyst_rating_score', 0.0)
                    analyst_total = ctx.get('analyst_rating_total', 0)

                    entry_overrides = {
                        "min_dollar_vol": dol_vol_threshold,
                    }
                    evaluation = evaluate_entry_rules(ctx, profile, overrides=entry_overrides)

                    scan_detail = (
                        f"SCAN {sym} [{regime_label}/{profile.name}]: price=${price:.2f} "
                        f"PrevHigh=${ctx.get('prev_daily_high', ctx.get('prev_high', float('nan'))):.2f} "
                        f"MA20=${ctx.get('ma20', float('nan')):.2f} MA50=${ma50:.2f} MA200=${ma200:.2f} "
                        f"SMA200slope={sma200_slope:+.3f} ATR5/ATR20={atr_ratio:.2f} "
                        f"VolPace={rvol:.1f} rawRVOL={ctx.get('rvol_raw', rvol):.1f} "
                        f"spread={spread_pct*100:.2f}% "
                        f"Open=${day_open:.2f} DayLoc={day_loc if day_loc is not None else float('nan'):.2f} "
                        f"OpenGain={intraday_gain if intraday_gain is not None else float('nan'):+.2%} "
                        f"RSI={rsi:.1f}(Δ{rsi-rsi_p:+.1f}) "
                        f"MACDh={ctx.get('macd_hist', float('nan')):+.3f} "
                        f"MACDhΔ={ctx.get('macd_hist_delta', float('nan')):+.3f} "
                        f"DistHigh20={ctx.get('dist_high20', float('nan')):+.2%} "
                        f"RS63={rs_63d:+.2%} Ret13w={ret_13w:+.2%} "
                        f"Px52w={px_52w:.2f} WeeklyUp={bool(ctx.get('weekly_uptrend'))} "
                        f"Analyst={float(analyst_score):+.2f}/{int(analyst_total or 0)} "
                        f"ATR=${atr:.2f} ATR%={atr_pct:.2%} "
                        f"DolVol20d=${dol_vol_20d/1e6:.0f}M(thr=${dol_vol_threshold/1e6:.0f}M)"
                    )

                    if not evaluation.passed:
                        reject_counts['entry_filter'] += 1
                        failed = list(evaluation.failed)
                        for _gate in failed:
                            entry_filter_reasons[_gate] = entry_filter_reasons.get(_gate, 0) + 1
                        logger.debug(f"{scan_detail}")
                        logger.debug(f"SCAN {sym}: NO SIGNAL — failed: {failed}")
                        if any(name.startswith("dollar_vol>=") for name in failed):
                            self._remember_daily_scan_skip(
                                sym,
                                f"DolVol20d ${dol_vol_20d/1e6:.0f}M "
                                f"< threshold ${dol_vol_threshold/1e6:.0f}M"
                            )
                        continue

                    entry_strategy = select_entry_strategy(ctx, profile) or profile.name
                    ctx['entry_strategy'] = entry_strategy
                    ctx['entry_strategy_label'] = indicator_sleeve_label(entry_strategy)

                    # ── Correlation filter (expensive — only for passing candidates) ──
                    df_daily = ctx.get('df_daily')
                    if df_daily is not None and self.state:
                        max_corr = self._compute_book_correlation(sym, df_daily)
                        if max_corr > CORR_MAX:
                            reject_counts['correlation'] += 1
                            logger.debug(
                                f"SCAN {sym}: SKIP — correlation {max_corr:.2f} > {CORR_MAX} with book"
                            )
                            continue

                    # ── Sector clustering filter ──────────────────────────────
                    sector = self._get_sector(sym, ctx['contract'])
                    if book_sectors.get(sector, 0) >= MAX_SECTOR_COUNT:
                        reject_counts['sector_limit'] += 1
                        logger.debug(
                            f"SCAN {sym}: SKIP — sector '{sector}' already has "
                            f"{book_sectors[sector]}/{MAX_SECTOR_COUNT} positions"
                        )
                        continue

                    score = self._score_candidate(ctx)
                    if profile.min_score is not None and score < float(profile.min_score):
                        reject_counts['entry_filter'] += 1
                        logger.debug(
                            f"SCAN {sym}: SKIP — score {score:.1f} < "
                            f"{float(profile.min_score):.1f}"
                        )
                        continue

                    # Fix 5: Late-session quality gate — after LATE_ENTRY_CUTOFF_TIME
                    # (default 14:30 ET) the remaining session is too short to
                    # develop a full swing move.  Only allow high-conviction setups
                    # to avoid holding mediocre positions overnight.
                    if (now_ny.hour, now_ny.minute) >= LATE_ENTRY_CUTOFF_TIME:
                        if score < LATE_ENTRY_MIN_SCORE:
                            reject_counts['entry_filter'] += 1
                            logger.debug(
                                f"SCAN {sym}: SKIP — late entry after "
                                f"{LATE_ENTRY_CUTOFF_TIME[0]:02d}:{LATE_ENTRY_CUTOFF_TIME[1]:02d} ET; "
                                f"score {score:.1f} < {LATE_ENTRY_MIN_SCORE:.0f} required"
                            )
                            continue

                    signals.append((score, sym, ctx))
                    logger.info(scan_detail)
                    logger.info(
                        f"SIGNAL {sym} [{regime_label}]: score={score:.1f}/100 | "
                        f"profile={profile.name} strategy={ctx.get('entry_strategy_label')} "
                        f"VolPace={rvol:.1f}x rawRVOL={ctx.get('rvol_raw', rvol):.1f}x "
                        f"trend_sep={(ma50-ma200)/ma200*100:.1f}% "
                        f"RSI_delta={rsi-rsi_p:.1f} RS63={rs_63d:+.2%} "
                        f"Analyst={float(analyst_score):+.2f}/{int(analyst_total or 0)} "
                        f"spread={spread_pct*100:.2f}% "
                        f"DolVol20d=${dol_vol_20d/1e6:.0f}M"
                    )

                # ── Phase 2: rank eligible signals by score; enter in descending
                #    order, falling through to the next candidate when a higher
                #    ranked symbol cannot be ordered because of sizing/cash/broker
                #    checks.
                filtered = sum(reject_counts.values())
                reject_summary = ", ".join(
                    f"{name}={count}"
                    for name, count in reject_counts.items()
                    if count
                ) or "none"
                gate_summary = ", ".join(
                    f"{name}={count}"
                    for name, count in sorted(
                        entry_filter_reasons.items(), key=lambda kv: kv[1], reverse=True
                    )[:8]
                )
                logger.info(
                    f"SCAN SUMMARY: scanner_hits={len(watchlist)} "
                    f"eligible_signals={len(signals)} "
                    f"filtered={filtered} "
                    f"rejects[{reject_summary}]"
                    + (f" entry_filter_gates[{gate_summary}]" if gate_summary else "")
                )

                # Fix 1: Market data blackout detection.
                # If ≥ DATA_BLACKOUT_RATIO_THRESHOLD of a meaningful watchlist
                # returned no live price for DATA_BLACKOUT_STREAK_ALERT
                # consecutive cycles, the IBKR real-time data farm is likely
                # down.  Emit one CRITICAL alert (deduped by _alert) and skip
                # new entries until data recovers.
                _blackout_ratio = (
                    reject_counts['no_technical_data'] / len(watchlist)
                    if len(watchlist) >= DATA_BLACKOUT_MIN_CANDIDATES else 0.0
                )
                if _blackout_ratio >= DATA_BLACKOUT_RATIO_THRESHOLD:
                    self._data_blackout_streak += 1
                    if (self._data_blackout_streak >= DATA_BLACKOUT_STREAK_ALERT
                            and not self._data_blackout_alerted):
                        self._alert(
                            "CRITICAL",
                            f"MARKET DATA BLACKOUT: {reject_counts['no_technical_data']}/"
                            f"{len(watchlist)} scanner candidates missing live price "
                            f"for {self._data_blackout_streak} consecutive cycles. "
                            "New entries suspended until data restores."
                        )
                        self._data_blackout_alerted = True
                else:
                    if self._data_blackout_alerted:
                        logger.info(
                            f"MARKET DATA: live prices restored after "
                            f"{self._data_blackout_streak} cycle blackout; "
                            "entries re-enabled."
                        )
                    self._data_blackout_streak = 0
                    self._data_blackout_alerted = False

                if self._data_blackout_streak >= DATA_BLACKOUT_STREAK_ALERT:
                    logger.warning(
                        f"MARKET DATA BLACKOUT: skipping new entries this cycle "
                        f"(streak={self._data_blackout_streak})."
                    )
                elif not signals:
                    logger.info("SCAN: No signals this cycle")
                else:
                    signals.sort(key=lambda x: x[0], reverse=True)
                    ranked = [(s, sym) for s, sym, _ in signals]
                    logger.info(
                        f"RANKED SIGNALS DESC: {ranked} — attempting up to "
                        f"{min(slots, len(signals))}"
                    )

                    placed = 0
                    for rank_idx, (score, sym, ctx) in enumerate(signals, start=1):
                        if placed >= slots:
                            break
                        logger.info(
                            f"ORDER ATTEMPT: rank={rank_idx}/{len(signals)} "
                            f"symbol={sym} score={score:.1f} "
                            f"filled_slots={placed}/{slots}"
                        )
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

                        if not self._entry_price_is_still_valid(sym, ctx, float(price), profile=profile):
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

                        trail_dist     = round(limit_price * TRAIL_PCT, 2)
                        hard_stop_dist = round(price * HARD_STOP_PCT, 2)
                        risk_stop_dist = trail_dist
                        if np.isnan(risk_stop_dist) or risk_stop_dist <= 0:
                            logger.warning(
                                f"SKIP {sym}: invalid risk stop distance "
                                f"(trail_dist=${trail_dist:.2f}, "
                                f"software_hard=${hard_stop_dist:.2f})"
                            )
                            continue

                        # Whole shares only. Risk-size from the broker-protected
                        # Chandelier distance; hard/break-even exits are software
                        # overlays and must not inflate live size.
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
                            'entry_qty':      filled_qty,
                            'entry_risk_per_share': trail_dist,
                            'initial_stop_loss': round(fill_price - trail_dist, 4),
                            'stop_loss':      round(fill_price - trail_dist, 2),
                            'stop_dist':      trail_dist,
                            'stop_mode':      'percent',
                            'trailing_percent': round(TRAIL_PCT * 100, 4),
                            'peak_price':     fill_price,
                            'volume':         ctx.get('volume', 0),
                            'score':          score,
                            'regime':         regime_label.lower(),
                            'strategy_profile': profile.name,
                            'entry_strategy': ctx.get('entry_strategy', profile.name),
                            'entry_strategy_label': ctx.get('entry_strategy_label', profile.label),
                            'relative_strength_63d': ctx.get('relative_strength_63d'),
                            'relative_strength_126d': ctx.get('relative_strength_126d'),
                            'return_13w':     ctx.get('return_13w'),
                            'return_26w':     ctx.get('return_26w'),
                            'weekly_uptrend': bool(ctx.get('weekly_uptrend')),
                            'price_vs_52w_high': ctx.get('price_vs_52w_high'),
                            'analyst_rating_score': ctx.get('analyst_rating_score'),
                            'analyst_rating_total': ctx.get('analyst_rating_total'),
                            'analyst_rating_source': ctx.get('analyst_rating_source'),
                            'analyst_rating_period': ctx.get('analyst_rating_period'),
                            'day_open':       ctx.get('day_open'),
                            'day_high':       ctx.get('day_high'),
                            'day_low':        ctx.get('day_low'),
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
                        self._ledger_call('open_trade', sym, {
                            **self.state[sym],
                            'spread_pct': ctx.get('spread_pct'),
                            'volume_pace': ctx.get('volume_pace'),
                            'atr_pct': ctx.get('atr_pct'),
                        })

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
                        stop_order.trailingPercent = round(TRAIL_PCT * 100, 2)
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
                            f"TrailStop=${round(fill_price * (1 - TRAIL_PCT), 2):.2f} "
                            f"(trail_pct={TRAIL_PCT*100:.1f}%, software_hard=${hard_stop_dist:.2f}, "
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
