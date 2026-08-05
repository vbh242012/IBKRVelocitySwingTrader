from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.config import (
    ANALYST_RATING_EXIT_ENABLED,
    ANALYST_RATING_SELL_THRESHOLD,
    BOLLINGER_STANDALONE_HARD_STOP_PCT,
    DAILY_BAR_SIZE,
    EOD_EXIT_TIME,
    EOD_HOLD_MIN_PROFIT_PCT,
    EOD_HOLD_DAY_RANGE_LOCATION_MIN,
    EOD_HOLD_RELATIVE_STRENGTH_MIN,
    EOD_HOLD_REQUIRE_STOP_CONFIRMED,
    HARD_STOP_PCT,
    HARD_STOP_STALE_BUFFER_PCT,
    MOMENTUM_HOLD_CHECK_INTERVAL_DAYS,
    MOMENTUM_HOLD_ENABLED,
    MOMENTUM_HOLD_LOOKBACK_DAYS,
    MOMENTUM_HOLD_MIN_MOVE_PCT,
    PRICE_STALE_MAX_AGE_SEC,
    POSITION_PRICE_BLACKOUT_STREAK_ALERT,
    FRIDAY_CLOSE_HOUR,
    FRIDAY_MIN_PROFIT_PCT,
    STALE_POSITION_MIN_BARS, STALE_POSITION_MAX_LOSS_PCT, STALE_POSITION_MAX_PEAK_PCT,
    STRATEGY_PROFILE,
)
from src.strategy_profiles import get_strategy_profile
from src.bollinger_standalone import (
    ENTRY_STRATEGY_NAME as BOLLINGER_STANDALONE_STRATEGY,
    bollinger_standalone_midline_reclaim,
    bollinger_standalone_time_stop_due,
)


_US_HOLIDAY_CACHE: Dict[int, set] = {}


def _us_market_holidays(year: int) -> set:
    """US federal holidays for `year`, cached per year.

    Not a perfect NYSE calendar -- NYSE additionally closes for Good Friday
    (not a federal holiday) and stays open on Columbus Day/Veterans Day
    (federal holidays) -- but it correctly captures the holidays that matter
    most for time-based exit counting (New Year's, MLK, Presidents Day,
    Memorial Day, Juneteenth, July 4th, Labor Day, Thanksgiving, Christmas)
    without adding a new dependency (pandas is already required).
    """
    if year not in _US_HOLIDAY_CACHE:
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=f"{year}-01-01", end=f"{year}-12-31")
        _US_HOLIDAY_CACHE[year] = {d.date() for d in holidays}
    return _US_HOLIDAY_CACHE[year]


def _count_trading_days(entry_dt: datetime, now: datetime) -> int:
    """Count complete Mon-Fri, non-holiday trading sessions elapsed between entry_dt and now."""
    entry_date = entry_dt.date()
    now_date   = now.date()
    count      = 0
    cursor     = entry_date
    while cursor < now_date:
        if cursor.weekday() < 5 and cursor not in _us_market_holidays(cursor.year):
            count += 1
        cursor += timedelta(days=1)
    return count


class ExitsMixin:

    def _eod_quality_hold_passes(
        self,
        sym: str,
        data: dict,
        snapshot: Dict[str, Optional[float]],
        entry_price: float,
        today_str: str,
    ) -> Tuple[bool, str]:
        """Quality gate for carrying a position overnight after EOD cleanup time."""
        cur = snapshot.get('price')
        if cur is None or entry_price <= 0:
            return False, "fresh price unavailable"

        profit = (cur - entry_price) / entry_price
        if profit < EOD_HOLD_MIN_PROFIT_PCT:
            return False, (
                f"profit {profit*100:.2f}% < "
                f"{EOD_HOLD_MIN_PROFIT_PCT*100:.2f}%"
            )

        vwap = snapshot.get('vwap')
        above_vwap_or_entry = (
            (vwap is not None and cur >= vwap)
            or cur > entry_price
        )
        if not above_vwap_or_entry:
            ref = f"VWAP ${vwap:.2f}" if vwap is not None else f"entry ${entry_price:.2f}"
            return False, f"price ${cur:.2f} below {ref}"

        day_high = snapshot.get('high') or self._coerce_positive_price(data.get('day_high'))
        day_low = snapshot.get('low') or self._coerce_positive_price(data.get('day_low'))
        day_loc = None
        if day_high is not None and day_low is not None and day_high > day_low:
            day_loc = (cur - day_low) / (day_high - day_low)
        if day_loc is None or day_loc < EOD_HOLD_DAY_RANGE_LOCATION_MIN:
            shown = "unavailable" if day_loc is None else f"{day_loc:.2f}"
            return False, (
                f"day-range location {shown} < "
                f"{EOD_HOLD_DAY_RANGE_LOCATION_MIN:.2f}"
            )

        stock_open = (
            snapshot.get('open')
            or self._coerce_positive_price(data.get('day_open'))
            or self._coerce_positive_price(data.get('entry_day_open'))
        )
        if stock_open is None or stock_open <= 0:
            return False, "stock intraday open unavailable"
        stock_ret = (cur - stock_open) / stock_open
        spy_ret = self._fresh_spy_intraday_return(today_str)
        if spy_ret is None:
            return False, "SPY intraday return unavailable"
        rel_strength = stock_ret - spy_ret
        if rel_strength < EOD_HOLD_RELATIVE_STRENGTH_MIN:
            return False, (
                f"relative strength {rel_strength*100:.2f}% < "
                f"{EOD_HOLD_RELATIVE_STRENGTH_MIN*100:.2f}%"
            )

        if EOD_HOLD_REQUIRE_STOP_CONFIRMED:
            protection_status = str(data.get('protection_status', '') or '').lower()
            if protection_status != 'confirmed':
                return False, f"protective stop not confirmed ({protection_status or 'missing'})"

        if ANALYST_RATING_EXIT_ENABLED:
            rating_ctx = self._analyst_context(sym)
            self._apply_analyst_context(data, rating_ctx)
            rating_score = data.get('analyst_rating_score')
            try:
                rating_score = float(rating_score)
            except (TypeError, ValueError):
                rating_score = None
            if rating_score is not None and np.isfinite(rating_score):
                if rating_score <= ANALYST_RATING_SELL_THRESHOLD:
                    return False, (
                        f"analyst rating score {rating_score:+.2f} <= "
                        f"{ANALYST_RATING_SELL_THRESHOLD:+.2f}"
                    )

        return True, (
            f"profit={profit*100:.2f}% dayLoc={day_loc:.2f} "
            f"RS={rel_strength*100:.2f}% stop={data.get('protection_status', 'unknown')}"
        )

    def _analyst_context(self, symbol: str) -> dict:
        provider = getattr(self, '_analyst_provider', None)
        if provider is None:
            from src.analyst_ratings import AnalystRatingProvider
            self._analyst_provider = AnalystRatingProvider(allow_remote=True)
            provider = self._analyst_provider
        return provider.get(symbol).as_context()

    @staticmethod
    def _apply_analyst_context(data: dict, ctx: dict) -> None:
        if not isinstance(data, dict) or not isinstance(ctx, dict):
            return
        for key in (
            'analyst_rating_score',
            'analyst_rating_raw_score',
            'analyst_rating_total',
            'analyst_rating_strong_buy',
            'analyst_rating_buy',
            'analyst_rating_hold',
            'analyst_rating_sell',
            'analyst_rating_strong_sell',
            'analyst_rating_source',
            'analyst_rating_period',
        ):
            if key in ctx:
                data[key] = ctx.get(key)
        try:
            score = float(data.get('analyst_rating_score'))
        except (TypeError, ValueError):
            return
        if np.isfinite(score):
            data['analyst_rating_score'] = round(score, 4)

    def _analyst_exit_price_confirms(
        self,
        sym: str,
        current_price: float,
        entry_price: float,
        snapshot: Optional[Dict[str, Optional[float]]] = None,
    ) -> tuple[bool, str]:
        if entry_price > 0 and current_price <= entry_price:
            return True, f"current ${current_price:.2f} <= entry ${entry_price:.2f}"

        prev_close = self._coerce_positive_price((snapshot or {}).get('prev_close'))
        if prev_close is not None and entry_price > 0 and prev_close <= entry_price:
            return True, f"previous close ${prev_close:.2f} <= entry ${entry_price:.2f}"

        row = self._daily_indicator_exit_row(sym)
        if row is None:
            return False, "daily confirmation unavailable"

        close = self._coerce_positive_price(row.get('close'))
        ma20 = self._coerce_positive_price(row.get('MA20'))
        if close is not None and entry_price > 0 and close <= entry_price:
            return True, f"daily close ${close:.2f} <= entry ${entry_price:.2f}"
        if close is not None and ma20 is not None and close < ma20:
            return True, f"daily close ${close:.2f} < MA20 ${ma20:.2f}"
        if bool(row.get('MA_BEAR_CROSS', False)):
            return True, "EMA20 crossed below SMA50"

        close_text = f"${close:.2f}" if close is not None else "unavailable"
        ma20_text = f"${ma20:.2f}" if ma20 is not None else "unavailable"
        return False, f"daily close {close_text}, MA20 {ma20_text}"

    def _analyst_exit_required(
        self,
        sym: str,
        data: dict,
        current_price: float,
        entry_price: float,
        snapshot: Optional[Dict[str, Optional[float]]] = None,
    ) -> tuple[bool, str]:
        """Return True when bearish consensus is confirmed by weak price action."""
        if not ANALYST_RATING_EXIT_ENABLED:
            return False, "disabled"

        rating_ctx = self._analyst_context(sym)
        self._apply_analyst_context(data, rating_ctx)
        rating_score = data.get('analyst_rating_score')
        try:
            rating_score = float(rating_score)
        except (TypeError, ValueError):
            return False, "unavailable"
        if not np.isfinite(rating_score):
            return False, "unavailable"

        if rating_score <= ANALYST_RATING_SELL_THRESHOLD:
            confirmed, confirm_reason = self._analyst_exit_price_confirms(
                sym,
                current_price=current_price,
                entry_price=entry_price,
                snapshot=snapshot,
            )
            if confirmed:
                return True, (
                    f"analyst rating score {rating_score:+.2f} <= "
                    f"{ANALYST_RATING_SELL_THRESHOLD:+.2f}; {confirm_reason}"
                )
            return False, (
                f"analyst rating score {rating_score:+.2f} <= "
                f"{ANALYST_RATING_SELL_THRESHOLD:+.2f}, but price not weak enough "
                f"({confirm_reason})"
            )
        return False, f"analyst rating score {rating_score:+.2f}"

    def _indicator_strategy_exit_required(
        self, sym: str, data: dict, trading_bars_held: int = 0,
    ) -> tuple[bool, str]:
        strategy = str(data.get('entry_strategy') or '').strip().lower()
        if strategy not in {
            'ma_cross', 'bollinger_reversion', 'psar_flip', BOLLINGER_STANDALONE_STRATEGY,
        }:
            return False, "not an indicator-swing position"

        last = self._daily_indicator_exit_row(sym)
        if last is None:
            return False, "indicator check unavailable"

        if strategy == "ma_cross" and bool(last.get('MA_BEAR_CROSS', False)):
            return True, "EMA20 crossed below SMA50"
        if strategy == "bollinger_reversion" and bool(last.get('BB_ABOVE_UPPER_2', False)):
            return True, "two closes above upper Bollinger Band"
        if strategy == "psar_flip" and bool(last.get('PSAR_BEAR_3', False)):
            return True, "three PSAR dots above price"
        if strategy == BOLLINGER_STANDALONE_STRATEGY:
            if bollinger_standalone_midline_reclaim(last):
                return True, "closed back above Bollinger midline (reclaim take-profit)"
            if bollinger_standalone_time_stop_due(trading_bars_held):
                return True, f"held {trading_bars_held} trading days without a midline reclaim"
        return False, "strategy exit not triggered"

    def _momentum_hold_passes(self, sym: str) -> Tuple[Optional[bool], str]:
        """Require at least MOMENTUM_HOLD_MIN_MOVE_PCT close-to-close appreciation
        over the trailing MOMENTUM_HOLD_LOOKBACK_DAYS completed sessions.

        Returns (None, reason) when daily history is unavailable so the caller
        can fail open (retry next cycle) rather than liquidating on a data
        outage. Uses completed daily closes rather than the live intraday
        price so a momentum-continuation judgement isn't made on an
        in-progress bar.
        """
        from src.engine_base import completed_daily_bars, logger, _TZ_NY
        try:
            contract = self._stock_contract(sym)
            bars = self.ib.reqHistoricalData(
                contract, '', '10 D', DAILY_BAR_SIZE, 'TRADES', True
            )
            today_str = datetime.now(_TZ_NY).strftime('%Y-%m-%d')
            bars = completed_daily_bars(bars, today_str)
        except Exception as exc:
            logger.warning(f"MOMENTUM HOLD: {sym} daily history fetch failed: {exc}")
            return None, "daily history unavailable"

        if not isinstance(bars, list) or len(bars) <= MOMENTUM_HOLD_LOOKBACK_DAYS:
            return None, "insufficient daily history"

        window = bars[-(MOMENTUM_HOLD_LOOKBACK_DAYS + 1):]
        closes = [self._coerce_positive_price(getattr(b, 'close', None)) for b in window]
        if any(c is None for c in closes):
            return None, "incomplete daily closes"

        ref_close, latest_close = closes[0], closes[-1]
        move = (latest_close - ref_close) / ref_close
        passed = move >= MOMENTUM_HOLD_MIN_MOVE_PCT
        return passed, (
            f"{move*100:+.2f}% over trailing {MOMENTUM_HOLD_LOOKBACK_DAYS} sessions "
            f"(${ref_close:.2f} -> ${latest_close:.2f})"
        )

    def manage_position_exits(self):
        """Manage live software exits for existing positions.

        Broker-side trailing stops remain the primary protection. Software exits
        require a fresh broker price so stale dashboard/cache values cannot
        liquidate a valid swing position.
        """
        from src.engine_base import logger, _TZ_NY
        now_et        = datetime.now(_TZ_NY)
        profile       = getattr(self, "_strategy_profile", get_strategy_profile(STRATEGY_PROFILE))
        today_str     = now_et.strftime('%Y-%m-%d')
        hhmm          = (now_et.hour, now_et.minute)
        is_eod_window = (hhmm >= EOD_EXIT_TIME and now_et.weekday() < 5)
        is_friday_close = (
            now_et.weekday() == 4
            and now_et.hour >= FRIDAY_CLOSE_HOUR
            and profile.friday_close_enabled
        )
        eod_exit_due  = (
            is_eod_window
            and getattr(self, '_last_eod_exit_date', None) != today_str
            and profile.eod_quality_cleanup
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
            snapshot   = self._fresh_market_snapshot(sym)
            fresh_cur  = snapshot.get('price') if snapshot else None
            if fresh_cur is None:
                miss_counts = getattr(self, '_exit_price_miss_counts', {})
                miss_count = miss_counts.get(sym, 0) + 1
                miss_counts[sym] = miss_count
                self._exit_price_miss_counts = miss_counts

                if miss_count == POSITION_PRICE_BLACKOUT_STREAK_ALERT:
                    cached_desc = f"${cached_cur:.2f}" if cached_cur else "unavailable"
                    self._alert(
                        "CRITICAL",
                        f"EXIT BLACKOUT: {sym} has had no fresh price for "
                        f"{miss_count} consecutive management cycles "
                        f"(cached={cached_desc}). "
                        "Software exits disabled; broker trailing stop is the only protection. "
                        "Check IBKR real-time data subscription for this symbol.",
                    )

                # Degraded-mode hard stop: use the cached price if it is recent
                # enough (within PRICE_STALE_MAX_AGE_SEC). Apply an extra buffer
                # (HARD_STOP_STALE_BUFFER_PCT) to account for adverse moves
                # since the last successful quote. All other software exits
                # require a fresh price and are skipped.
                if cached_cur is not None:
                    price_checked_at = data.get('price_checked_at')
                    cache_age_sec = float('inf')
                    if price_checked_at:
                        try:
                            checked_dt = datetime.fromisoformat(price_checked_at)
                            if checked_dt.tzinfo is None:
                                checked_dt = _TZ_NY.localize(checked_dt)
                            cache_age_sec = (now_et - checked_dt.astimezone(_TZ_NY)).total_seconds()
                        except (TypeError, ValueError):
                            pass
                    if cache_age_sec <= PRICE_STALE_MAX_AGE_SEC:
                        stale_threshold = HARD_STOP_PCT + HARD_STOP_STALE_BUFFER_PCT
                        drawdown = (cached_cur - entry_price) / entry_price
                        if drawdown <= -stale_threshold:
                            logger.warning(
                                f"HARD STOP (stale price): {sym} cached ${cached_cur:.2f} "
                                f"is {drawdown*100:.1f}% from entry "
                                f"(threshold -{stale_threshold*100:.0f}% with stale buffer, "
                                f"cache age {cache_age_sec:.0f}s). Forcing exit."
                            )
                            self.liquidate(sym, reason='hard_stop_stale_price')
                            continue

                cached_desc = f"${cached_cur:.2f}" if cached_cur else "unavailable"
                logger.warning(
                    f"EXIT: {sym} fresh price unavailable; skipping software exit checks "
                    f"(cached={cached_desc}, miss_streak={miss_count})."
                )
                continue

            # Fresh price obtained — reset miss counter
            miss_counts = getattr(self, '_exit_price_miss_counts', {})
            miss_counts.pop(sym, None)
            self._exit_price_miss_counts = miss_counts

            cur = fresh_cur
            self.state[sym]['current_price'] = round(cur, 2)
            self.state[sym]['price_checked_at'] = now_et.isoformat()
            # Cycle-sampled MFE/MAE for the trade ledger.
            self._ledger_call('update_price', sym, cur)
            for state_key, snapshot_key in (
                ('day_open', 'open'),
                ('day_high', 'high'),
                ('day_low', 'low'),
                ('prev_close', 'prev_close'),
                ('vwap', 'vwap'),
            ):
                value = snapshot.get(snapshot_key) if snapshot else None
                if value is not None:
                    self.state[sym][state_key] = round(float(value), 4)
            changed = True

            # ── 1. Intraday hard stop — requires a fresh broker price
            drawdown = (cur - entry_price) / entry_price
            entry_strategy_hint = str(data.get('entry_strategy') or '').strip().lower()
            if (
                entry_strategy_hint == BOLLINGER_STANDALONE_STRATEGY
                and drawdown <= -BOLLINGER_STANDALONE_HARD_STOP_PCT
            ):
                logger.warning(
                    f"BOLLINGER HARD STOP: {sym} down {drawdown*100:.1f}% from entry "
                    f"(${cur:.2f} vs entry ${entry_price:.2f}), tighter "
                    f"{BOLLINGER_STANDALONE_HARD_STOP_PCT*100:.0f}% threshold. Forcing exit."
                )
                self.liquidate(sym, reason='bollinger_hard_stop')
                continue
            if drawdown <= -HARD_STOP_PCT:
                logger.warning(
                    f"HARD STOP: {sym} down {drawdown*100:.1f}% from entry "
                    f"(${cur:.2f} vs entry ${entry_price:.2f}). Forcing exit."
                )
                self.liquidate(sym, reason='hard_stop')
                continue

            peak_price = max(float(data.get('peak_price', entry_price) or entry_price), cur)
            if peak_price != float(data.get('peak_price', entry_price) or entry_price):
                self.state[sym]['peak_price'] = round(peak_price, 2)
                changed = True

            analyst_exit, analyst_reason = self._analyst_exit_required(
                sym,
                data,
                current_price=cur,
                entry_price=entry_price,
                snapshot=snapshot,
            )
            if analyst_exit:
                logger.warning(
                    f"ANALYST DOWNGRADE EXIT: {sym} {analyst_reason}. Closing."
                )
                self.liquidate(sym, reason='analyst_downgrade')
                continue

            entry_time_raw = data.get('time')
            trading_bars_held = 0
            try:
                entry_dt = datetime.fromisoformat(entry_time_raw) if entry_time_raw else now_et
                if entry_dt.tzinfo is None:
                    entry_dt = _TZ_NY.localize(entry_dt)
                else:
                    entry_dt = entry_dt.astimezone(_TZ_NY)
                trading_bars_held = _count_trading_days(entry_dt, now_et)
            except (TypeError, ValueError):
                trading_bars_held = 0

            strategy_exit, strategy_reason = self._indicator_strategy_exit_required(
                sym, data, trading_bars_held,
            )
            if strategy_exit:
                logger.warning(
                    f"STRATEGY EXIT: {sym} [{data.get('entry_strategy', 'unknown')}] "
                    f"{strategy_reason}. Closing."
                )
                self.liquidate(sym, reason='strategy_exit')
                continue

            if profile.time_stop_bars is not None and trading_bars_held >= int(profile.time_stop_bars):
                min_profit = float(profile.time_stop_min_profit or 0.0)
                profit = (cur - entry_price) / entry_price
                if profit <= min_profit:
                    logger.warning(
                        f"SWING TIME STOP: {sym} held {trading_bars_held} trading bars "
                        f"with profit={profit*100:.1f}% <= {min_profit*100:.1f}%. Closing."
                    )
                    self.liquidate(sym, reason='time_stop')
                    continue

            # ── 2. Friday afternoon close — explicit weekend-risk policy
            if is_friday_close:
                friday_profit = (cur - entry_price) / entry_price
                if friday_profit < FRIDAY_MIN_PROFIT_PCT:
                    logger.warning(
                        f"FRIDAY CLOSE: {sym} profit={friday_profit*100:.1f}% < "
                        f"{FRIDAY_MIN_PROFIT_PCT*100:.0f}% — closing to avoid weekend risk."
                    )
                    self.liquidate(sym, reason='friday_close')
                    continue

            # ── 3. EOD quality cleanup — same-day capital recycling.
            #
            # Fires once per trading day after EOD_EXIT_TIME (default 15:50 ET).
            # A position is carried only if it is at least breakeven, closing
            # strong, outperforming SPY intraday, and protected by a confirmed
            # broker stop. This intentionally applies to same-day entries too.
            if eod_exit_due:
                eod_exit_checked = True
                hold_ok, hold_reason = self._eod_quality_hold_passes(
                    sym,
                    data,
                    snapshot,
                    entry_price,
                    today_str,
                )
                if not hold_ok:
                    logger.warning(
                        f"EOD QUALITY CLEANUP: {sym} failed hold quality "
                        f"({hold_reason}) near the close "
                        f"(cur=${cur:.2f}, entry=${entry_price:.2f}) "
                        f"— closing position."
                    )
                    self.liquidate(sym, reason='eod_quality_cleanup')
                    continue
                logger.info(f"EOD QUALITY HOLD: {sym} held overnight ({hold_reason}).")

            # ── 4. Stale losing position exit ───────────────────────────────
            # Fix 4: A position held for ≥ STALE_POSITION_MIN_BARS trading
            # sessions that has never exceeded +STALE_POSITION_MAX_PEAK_PCT
            # and is currently losing more than STALE_POSITION_MAX_LOSS_PCT is
            # dead weight.  Close at EOD to recycle capital.  This fires as a
            # standalone check so it catches positions regardless of whether the
            # EOD quality cleanup already ran today.  Positions liquidated above
            # (hard stop, quality hold, etc.) have already `continue`d, so they
            # won't reach this point.
            if (
                is_eod_window
                and trading_bars_held >= STALE_POSITION_MIN_BARS
                and (cur - entry_price) / entry_price < STALE_POSITION_MAX_LOSS_PCT
                and float(data.get('peak_price') or entry_price) / entry_price - 1
                    < STALE_POSITION_MAX_PEAK_PCT
            ):
                profit_pct = (cur - entry_price) / entry_price * 100
                peak_pct = (float(data.get('peak_price') or entry_price) / entry_price - 1) * 100
                logger.warning(
                    f"STALE LOSING EXIT: {sym} held {trading_bars_held} bars, "
                    f"profit={profit_pct:.1f}% (< {STALE_POSITION_MAX_LOSS_PCT*100:.0f}%), "
                    f"peak={peak_pct:.1f}% (never reached "
                    f"+{STALE_POSITION_MAX_PEAK_PCT*100:.0f}%). "
                    "Closing stale loser to recycle capital."
                )
                self.liquidate(sym, reason='stale_losing_exit')
                continue

            # ── 5. Periodic momentum-stall exit ─────────────────────────────
            # Every MOMENTUM_HOLD_CHECK_INTERVAL_DAYS, require at least
            # MOMENTUM_HOLD_MIN_MOVE_PCT close-to-close appreciation over the
            # trailing MOMENTUM_HOLD_LOOKBACK_DAYS completed sessions. This is
            # a momentum-continuation check, independent of profit/loss versus
            # entry, and can close a position that is still up overall if it
            # has stopped making fresh progress.
            if MOMENTUM_HOLD_ENABLED and trading_bars_held >= MOMENTUM_HOLD_LOOKBACK_DAYS:
                last_check_str = data.get('momentum_check_date')
                check_due = last_check_str is None
                if not check_due:
                    try:
                        last_check_date = datetime.fromisoformat(last_check_str).date()
                        check_due = (
                            (now_et.date() - last_check_date).days
                            >= MOMENTUM_HOLD_CHECK_INTERVAL_DAYS
                        )
                    except (TypeError, ValueError):
                        check_due = True
                if check_due:
                    hold_ok, momentum_reason = self._momentum_hold_passes(sym)
                    if hold_ok is not None:
                        self.state[sym]['momentum_check_date'] = today_str
                        changed = True
                        if not hold_ok:
                            logger.warning(
                                f"MOMENTUM STALL EXIT: {sym} {momentum_reason} "
                                f"(< required +{MOMENTUM_HOLD_MIN_MOVE_PCT*100:.1f}%). "
                                "Closing to recycle capital."
                            )
                            self.liquidate(sym, reason='momentum_stall')
                            continue
                        logger.info(f"MOMENTUM HOLD: {sym} {momentum_reason}.")

        # Mark EOD exit as done only after at least one live-price evaluation.
        if eod_exit_due and eod_exit_checked:
            self._last_eod_exit_date = today_str

        if changed:
            self.save_state()
