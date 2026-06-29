from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from ib_async import IB, util

from src.config import (
    ACCOUNT_CURRENCY,
    MAX_POSITIONS_CAP, MIN_BUCKET_SIZE, MIN_BUCKET_FLOOR, SETTLED_CASH_DEPLOYMENT_PCT,
    EQUITY_RETRY_INTERVAL,
    HARD_STOP_PCT,
    RISK_PER_TRADE_PCT,
    ATR_PCT_MAX,
    SPREAD_MAX_PCT,
    STRATEGY_PROFILE,
    ENTRY_START, ENTRY_END, STOP_ACTIVATION_TIME,
    MARKET_CLOSE_TIME, PREMARKET_READINESS_TIME, POST_CLOSE_MAINTENANCE_TIME,
    APP_PREFILTER_ENABLED, APP_PREFILTER_START_TIME,
    DAILY_LOOKBACK, DAILY_BAR_SIZE,
    VIX_THRESHOLD,
    TRADING_MODE,
    ENTRY_REPRICE_MAX_AGE_SEC, ENTRY_MAX_PRICE_DRIFT_PCT,
    CORR_MAX, MAX_SECTOR_COUNT,
    LATE_ENTRY_CUTOFF_TIME, LATE_ENTRY_MIN_SCORE,
    DATA_BLACKOUT_RATIO_THRESHOLD, DATA_BLACKOUT_MIN_CANDIDATES, DATA_BLACKOUT_STREAK_ALERT,
    BEAR_PHASE_TRADING_ENABLED, BEAR_PHASE_RISK_MULT, BEAR_PHASE_DOLLAR_VOL_MULT,
    FRIDAY_ENTRY_CUTOFF_TIME,
    EOD_EXIT_TIME,
    TRAIL_PCT,
)
from src.scoring import score_candidate
from src.strategy_profiles import (
    evaluate_entry_rules,
    get_strategy_profile,
    select_entry_strategy,
    indicator_sleeve_label,
)


class EntriesMixin:

    def _request_account_summary_snapshot(self):
        """Request account summary as a bounded one-shot and cancel the stream.

        ib_async.accountSummary() is convenient, but if Gateway disconnects while
        a request is open, IBKR can keep counting it against the account-summary
        subscription limit. For real IB instances we use the lower-level request
        API and always send cancelAccountSummary(reqId) in finally.
        """
        from src.engine_base import logger
        from src.config import IB_CLIENT_ID
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
        from src.engine_base import logger, AccountDataUnavailable
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

    def _fetch_equity_with_retry(self) -> float:
        """Poll IBKR for NetLiquidation until a positive value is returned. Never gives up."""
        from src.engine_base import logger
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
        from src.engine_base import logger
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

        Fetch equity, cancel orphaned orders, sync positions, audit protective
        stops, refresh prices, and write dashboard so the UI is live straight
        away.

        The run loop performs the pre-entry audit at PRE_ENTRY_SYNC_TIME and a
        separate post-open audit at POST_OPEN_AUDIT_TIME.  Startup must never
        sleep until 09:15 ET, because the full-universe prefilter is scheduled
        earlier and needs the main loop alive.
        """
        from src.engine_base import logger, _TZ_NY
        from src.config import IB_CLIENT_ID, HARD_STOP_PCT
        # ── Immediate startup snapshot ───────────────────────────────────────
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
        # orders — those are audited immediately and by scheduled checkpoints.
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

        logger.info("INIT: Immediate position sync and stop-order audit...")
        self._sync_positions_from_ibkr()
        if self.state:
            logger.info("INIT: Auditing stop orders for open positions immediately...")
            self._audit_stop_orders()
            now_ny = datetime.now(_TZ_NY)
            self._last_audit_date = now_ny.strftime('%Y-%m-%d')
            self._last_audit_at = now_ny
            logger.info("INIT: Fetching live prices for open positions...")
            self._update_position_prices()

        self._log_startup_summary(equity)
        self._write_dashboard_data(connected=True)

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
        if not np.isfinite(equity) or equity < MIN_BUCKET_FLOOR:
            return 0
        return min(int(equity / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)

    @staticmethod
    def _calc_cash_entry_slots(settled: float) -> int:
        """Return how many new-entry buckets settled cash can fund right now.

        If deployable cash is below MIN_BUCKET_SIZE but above MIN_BUCKET_FLOOR,
        one slot is granted using the full deployable amount.  This prevents a
        small account from getting permanently frozen when settled cash is only
        slightly below the normal bucket minimum — a common outcome on T+1
        accounts that are nearly fully deployed.
        """
        deployable = EntriesMixin._deployable_settled_cash(settled)
        if deployable < MIN_BUCKET_FLOOR:
            return 0
        if deployable < MIN_BUCKET_SIZE:
            return 1
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
        cash_slots = EntriesMixin._calc_cash_entry_slots(settled)
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

    def _score_candidate(
        self,
        ctx: dict,
    ) -> float:
        """Rank a passing candidate with the shared live/backtest scorer."""
        profile = getattr(self, "_strategy_profile", get_strategy_profile(STRATEGY_PROFILE))
        return score_candidate(
            ctx,
            model=profile.scoring_model,
            volume_floor=profile.min_volume_pace or 1.0,
            spread_max_pct=SPREAD_MAX_PCT,
            atr_pct_max=profile.max_atr_pct or ATR_PCT_MAX,
        )

    def _entry_price_is_still_valid(
        self,
        sym: str,
        ctx: dict,
        price: float,
        profile=None,
    ) -> bool:
        """Re-check fast-moving entry gates immediately before sending an order."""
        from src.engine_base import logger
        profile = profile or getattr(self, "_strategy_profile", get_strategy_profile(STRATEGY_PROFILE))
        if np.isnan(price) or price <= 0:
            logger.warning(f"SKIP {sym}: refreshed entry price invalid ({price})")
            return False

        if price < profile.min_price:
            logger.warning(
                f"SKIP {sym}: refreshed entry price ${price:.2f} below "
                f"{profile.name} minimum ${profile.min_price:.2f}"
            )
            return False

        refreshed_ctx = dict(ctx)
        refreshed_ctx['live_price'] = price
        refreshed_ctx['close'] = price
        evaluation = evaluate_entry_rules(
            refreshed_ctx,
            profile,
        )
        if not evaluation.passed:
            logger.warning(
                f"SKIP {sym}: refreshed price no longer passes swing setup "
                f"({list(evaluation.failed)})"
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
        from src.engine_base import logger, _TZ_NY
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
            fallback_risk = round(avg_cost * HARD_STOP_PCT, 4) if avg_cost > 0 else 0.0
            if avg_cost <= 0:
                logger.warning(
                    f"SYNC: {sym} — avgCost={avg_cost} from IBKR; "
                    f"EOD quality cleanup will be skipped until price is corrected."
                )

            if sym not in self.state:
                self.state[sym] = {
                    'price':           round(avg_cost, 2),
                    'fill_price':      round(avg_cost, 2),
                    'broker_avg_cost': round(avg_cost, 4),
                    'time':            datetime.now(_TZ_NY).isoformat(),
                    'qty':             qty,
                    'entry_qty':       qty,
                    'entry_risk_per_share': fallback_risk,
                    'initial_stop_loss': round(avg_cost - fallback_risk, 4) if fallback_risk > 0 else 0.0,
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
                if float(self.state[sym].get('entry_qty', 0) or 0) <= 0:
                    self.state[sym]['entry_qty'] = max(qty, state_qty)
                    changed = True
                if float(self.state[sym].get('entry_risk_per_share', 0) or 0) <= 0:
                    fallback_risk = (
                        float(self.state[sym].get('stop_dist', 0) or 0)
                        or (avg_cost * HARD_STOP_PCT if avg_cost > 0 else 0.0)
                    )
                    if fallback_risk > 0:
                        self.state[sym]['entry_risk_per_share'] = round(float(fallback_risk), 4)
                        changed = True
                if float(self.state[sym].get('initial_stop_loss', 0) or 0) <= 0:
                    entry_px = float(self.state[sym].get('fill_price', self.state[sym].get('price', avg_cost)) or 0)
                    entry_risk = float(self.state[sym].get('entry_risk_per_share', 0) or 0)
                    if entry_px > 0 and entry_risk > 0:
                        self.state[sym]['initial_stop_loss'] = round(entry_px - entry_risk, 4)
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
        from src.engine_base import logger, _TZ_NY
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
                    if sd > 0 or str(self.state[sym].get('stop_mode', '')).lower() == 'percent':
                        peak = max(float(self.state[sym].get('peak_price', cur)), cur)
                        self.state[sym]['peak_price'] = round(peak, 2)
                        ibkr_stop = float(self.state[sym].get('stop_loss', 0))
                        trail_pct_state = self.state[sym].get('trailing_percent')
                        if trail_pct_state is not None:
                            # Percent trail: estimate IBKR's live trailStopPrice as
                            # max(last IBKR-confirmed stop_loss, cur × (1 - pct/100)).
                            # Do NOT use peak_price here — the percent trail order may
                            # have been placed after the historical peak (e.g. after a
                            # dollar→percent conversion), so peak × (1-pct) overstates
                            # the actual broker-side stop.
                            cur_estimate = round(cur * (1 - float(trail_pct_state) / 100), 2)
                            trail_floor = max(ibkr_stop, cur_estimate)
                        elif sd > 0:
                            trail_floor = max(ibkr_stop, peak - sd)
                        else:
                            trail_floor = 0
                        if trail_floor > 0:
                            self.state[sym]['effective_stop'] = round(trail_floor, 2)
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

    def _regular_management_active(self, now_ny=None) -> bool:
        """True only during the regular-session window where software exits may run."""
        from src.engine_base import _TZ_NY
        now_ny = now_ny or datetime.now(_TZ_NY)
        if now_ny.weekday() >= 5:
            return False
        hhmm = (now_ny.hour, now_ny.minute)
        return STOP_ACTIVATION_TIME <= hhmm < MARKET_CLOSE_TIME

    def _build_readiness_snapshot(self, checkpoint: str) -> dict:
        """
        Collect non-trading data that helps the next trading session.

        This deliberately avoids scanner/entry calls. It may reconcile account
        values, VIX/SPY regime, open orders, and current local/broker position
        state, but it must not create new BUY orders.
        """
        from src.engine_base import logger, _TZ_NY, AccountDataUnavailable
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
        from src.engine_base import logger
        logger.info(f"OFFHOURS: running {checkpoint} maintenance checkpoint")
        self._sync_positions_from_ibkr()
        if self.state:
            self._audit_stop_orders()
            from src.engine_base import _TZ_NY
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
        from src.engine_base import logger, _TZ_NY
        now_ny = datetime.now(_TZ_NY)
        if now_ny.weekday() >= 5:
            return False

        today = now_ny.strftime('%Y-%m-%d')
        hhmm = (now_ny.hour, now_ny.minute)
        ran = False

        if (APP_PREFILTER_ENABLED
                and APP_PREFILTER_START_TIME <= hhmm < ENTRY_START
                and getattr(self, '_last_premarket_prefilter_date', None) != today):
            try:
                self._run_premarket_universe_prefilter()
                self._last_premarket_prefilter_date = today
                ran = True
            except Exception as e:
                self._metric_inc('prefilter_failures')
                self._alert(
                    "ERROR",
                    f"OFFHOURS: premarket universe prefilter failed: {e}",
                )

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
