import copy
import math
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
from ib_async import MarketOrder, Order, util

from src.config import (
    IB_CLIENT_ID,
    DAILY_LOOKBACK, DAILY_BAR_SIZE,
    ENTRY_START,
    STOP_ACTIVATION_TIME,
    CHANDELIER_PERIOD,
    TRAIL_PCT,
    PROTECTIVE_STOP_CONFIRM_TIMEOUT_SEC, PROTECTIVE_STOP_CONFIRM_POLL_SEC,
    SPREAD_MAX_PCT,
    ENTRY_LIMIT_ASK_CUSHION_PCT, ENTRY_LIMIT_MIN_TICK, ENTRY_LIMIT_MAX_OVER_MARKET_PCT,
    HARD_STOP_PCT,
    STRATEGY_PROFILE,
)
from src.indicators import apply_all
from src.strategy_profiles import get_strategy_profile
from src.engine_base import VelocityEngineBase


class OrdersMixin:

    @staticmethod
    def _trail_order_protection(order, pos_data: dict) -> Tuple[bool, str, float, float]:
        """Interpret IB TRAIL SELL fields as (valid, display, stop_dist, stop_loss).

        IB can return a TRAIL order either as a dollar trail in auxPrice or as a
        percent trail with auxPrice left at UNSET_DOUBLE. Treating that sentinel
        as a dollar amount makes the dashboard/audit logic think the stop is
        absurdly wide even when the real trailStopPrice/trailingPercent is valid.
        """
        ref_price = (
            VelocityEngineBase._coerce_positive_price(pos_data.get('current_price'))
            or VelocityEngineBase._coerce_positive_price(pos_data.get('peak_price'))
            or VelocityEngineBase._coerce_positive_price(pos_data.get('price'))
            or VelocityEngineBase._coerce_positive_price(pos_data.get('entry_price'))
        )
        aux_dist = VelocityEngineBase._coerce_order_number(getattr(order, 'auxPrice', None))
        trail_stop = VelocityEngineBase._coerce_order_number(getattr(order, 'trailStopPrice', None))
        trail_pct = VelocityEngineBase._coerce_order_number(getattr(order, 'trailingPercent', None))

        if aux_dist is not None:
            if ref_price is not None and aux_dist >= ref_price:
                return False, f"dollar trail ${aux_dist:.2f} >= reference price ${ref_price:.2f}", 0.0, 0.0
            stop_loss = trail_stop or (ref_price - aux_dist if ref_price is not None else 0.0)
            return True, f"dist=${aux_dist:.2f}", aux_dist, max(stop_loss, 0.0)

        if trail_pct is not None:
            if trail_pct >= 99.0:
                return False, f"trailing percent {trail_pct:.4g}% is unusable", 0.0, 0.0
            if trail_stop is not None:
                # A finite trailStopPrice with a sane trailingPercent is a valid
                # protective order. When the stop sits at/above the current
                # reference price it is an in-the-money trailed stop that price has
                # pulled back through — a breached stop that should TRIGGER, not a
                # malformed order. It must never be cancelled/rebuilt here: doing so
                # ratchets a locked-in stop back down to the fallen price and gives
                # up the protected gain (this reset turned the live AMD trade on
                # 2026-07-01 from a locked winner into a loss). Derive the distance
                # from the trail percent when the ref-price gap is non-positive.
                if ref_price is not None and ref_price > trail_stop:
                    stop_dist = ref_price - trail_stop
                else:
                    stop_dist = trail_stop * (trail_pct / max(100.0 - trail_pct, 1e-9))
            elif ref_price is not None:
                # trailStopPrice not yet populated by IBKR (order just submitted); estimate from ref_price
                stop_dist = ref_price * trail_pct / 100
                trail_stop = round(ref_price - stop_dist, 2)
            else:
                return False, "percent trail: no trail stop price or reference price available", 0.0, 0.0
            return True, f"stop=${trail_stop:.2f} trail={trail_pct:.4g}%", stop_dist, max(trail_stop, 0.0)

        return False, "missing dollar trail or percent trail fields", 0.0, 0.0

    def _entry_good_after_time(self) -> str:
        """Return an entry-window activation string only while that time is still future."""
        from src.engine_base import _TZ_NY
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
        from src.engine_base import _TZ_NY
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

    @staticmethod
    def _calc_entry_limit_price(price, bid, ask) -> Optional[float]:
        """Return a marketable, spread-aware BUY limit price, or None.

        MOST_ACTIVE names are usually liquid, so the parent BUY should key off
        the real ask instead of blindly adding a fixed percentage to the last or
        midpoint. The old 0.2% cushion remains as a hard max over the validated
        reference price, while the working limit is ask plus a small tick/cushion.
        """
        ref_price = VelocityEngineBase._coerce_positive_price(price)
        bid_price = VelocityEngineBase._coerce_positive_price(bid)
        ask_price = VelocityEngineBase._coerce_positive_price(ask)
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
            limit = VelocityEngineBase._round_up_to_cent(ask_price)
        if limit > max_limit:
            limit = VelocityEngineBase._round_down_to_cent(max_limit)
        if limit < ask_price:
            return None
        return limit

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
        from src.engine_base import logger, _REJECTED_ORDER_STATUSES
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
        from src.engine_base import logger, _REJECTED_ORDER_STATUSES
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
        from src.engine_base import logger
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
        from src.engine_base import _TZ_NY
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
        from src.engine_base import _REJECTED_ORDER_STATUSES
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
        import time
        from src.engine_base import logger
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
        For every open position ensure exactly one percent TRAIL SELL order exists.

        Steps per symbol:
        1. Find all open SELL orders for the symbol.
        2. Cancel any that are NOT order type TRAIL (stale LMT take-profits, STP, etc.).
        3. If no TRAIL SELL remains after cancellations, place a new percent TRAIL SELL
           (GTC, transmit=True) using TRAIL_PCT from config and entry price from state.

        The configured entry trading window applies only to new BUY entries.
        Stop orders for existing positions are placed immediately regardless of time,
        but when the audit runs before the configured stop gate their activation is
        delayed with goodAfterTime. After placeOrder() we wait 2 s and verify the
        order status so any unexpected rejection is caught and logged; state is only
        updated once IB confirms the order is live (PreSubmitted / Submitted).
        """
        from src.engine_base import logger, _TZ_NY, _REJECTED_ORDER_STATUSES
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

            # Convert any dollar TRAIL to percent trail; if cancelled successfully,
            # the "no trail found" block below places a fresh percent TRAIL at TRAIL_PCT.
            # Respect clientId: only attempt cancellation of orders this engine owns.
            if trail_orders and trail_meta.get(id(trail_orders[0]), {}).get('trail_pct') is None:
                primary = trail_orders[0]
                try:
                    order_client_id = int(
                        getattr(primary.order, 'clientId', IB_CLIENT_ID) or IB_CLIENT_ID
                    )
                except (TypeError, ValueError):
                    order_client_id = IB_CLIENT_ID
                if order_client_id != IB_CLIENT_ID:
                    logger.warning(
                        f"AUDIT: {sym} — dollar TRAIL (id={primary.order.orderId}) "
                        f"belongs to clientId={order_client_id}, not this engine "
                        f"(clientId={IB_CLIENT_ID}); skipping percent-trail conversion"
                    )
                else:
                    logger.warning(
                        f"AUDIT: {sym} — converting dollar TRAIL "
                        f"(id={primary.order.orderId}) to {TRAIL_PCT:.0%} percent trail"
                    )
                    try:
                        self.ib.cancelOrder(primary.order)
                        self.ib.sleep(2)
                        trail_orders = []
                    except Exception as e:
                        self._alert(
                            "CRITICAL",
                            f"AUDIT: {sym} — failed to cancel dollar TRAIL for "
                            f"percent conversion ({e}); retaining dollar trail as protection"
                        )

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

            logger.info(f"AUDIT: {sym} — no TRAIL SELL found; placing percent trail stop...")
            try:
                contract = self._stock_contract(sym)

                entry_px = float(pos_data.get('fill_price') or pos_data.get('price') or 0)
                if entry_px <= 0:
                    self._mark_position_protection(
                        sym,
                        'unconfirmed',
                        'missing_entry_price_for_audit_stop',
                    )
                    self._alert(
                        "CRITICAL",
                        f"AUDIT: {sym} — entry price unavailable in state, "
                        f"cannot place stop; position is unprotected."
                    )
                    continue

                trail_pct_val = float(
                    pos_data.get('trailing_percent') or (TRAIL_PCT * 100)
                )
                trail_dist = round(entry_px * trail_pct_val / 100, 2)

                stop_order               = Order()
                stop_order.action        = 'SELL'
                stop_order.orderType     = 'TRAIL'
                stop_order.totalQuantity = qty
                stop_order.trailingPercent = round(trail_pct_val, 2)
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

                self.state[sym]['stop_dist']        = trail_dist
                self.state[sym]['stop_mode']        = 'percent'
                self.state[sym]['trailing_percent'] = round(trail_pct_val, 4)
                self.state[sym]['stop_loss'] = round(
                    entry_px * (1 - trail_pct_val / 100), 2
                )
                self._mark_position_protection(
                    sym,
                    'confirmed',
                    order_id=getattr(stop_trade.order, 'orderId', None),
                )
                logger.info(
                    f"AUDIT: {sym} — TRAIL SELL live "
                    f"(qty={qty:.4f} trail={trail_pct_val:.2g}% dist=${trail_dist:.2f} "
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
        from src.engine_base import logger
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
        from src.engine_base import logger, _TZ_NY
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

    def _maybe_pre_entry_sync_audit(self):
        """
        Run one confirmation position sync and stop audit before new entries.

        This used to be a blocking startup sleep until 09:15 ET.  Keeping it as
        a scheduled checkpoint lets earlier premarket jobs, especially the
        full-universe prefilter, run on time while preserving the same stop
        safety check before the entry window.
        """
        from src.engine_base import logger, _TZ_NY
        from src.config import PRE_ENTRY_SYNC_TIME
        today_dt = datetime.now(_TZ_NY)
        if today_dt.weekday() >= 5:
            return

        today = today_dt.strftime('%Y-%m-%d')
        if getattr(self, '_last_pre_entry_sync_date', None) == today:
            return

        h, m = PRE_ENTRY_SYNC_TIME
        sync_time = today_dt.replace(hour=h, minute=m, second=0, microsecond=0)
        if today_dt < sync_time:
            return

        last_audit_at = getattr(self, '_last_audit_at', None)
        if last_audit_at is not None and last_audit_at >= sync_time:
            self._last_pre_entry_sync_date = today
            logger.info(
                "AUDIT: pre-entry checkpoint already covered by a protective "
                f"stop audit at {last_audit_at.strftime('%H:%M:%S %Z')}"
            )
            return

        logger.info(
            f"AUDIT: running pre-entry position sync and stop audit "
            f"({h:02d}:{m:02d} ET checkpoint)"
        )
        try:
            self._sync_positions_from_ibkr()
            if self.state:
                self._audit_stop_orders()
                self._update_position_prices()
            else:
                logger.info("AUDIT: pre-entry checkpoint found no open positions")
            now_ny = datetime.now(_TZ_NY)
            self._last_pre_entry_sync_date = today
            self._last_audit_date = today
            self._last_audit_at = now_ny
            self._write_dashboard_data(connected=True)
        except Exception as e:
            self._alert(
                "CRITICAL",
                f"AUDIT: pre-entry protective stop audit failed unexpectedly; "
                f"positions may be unprotected ({e})",
            )

    def _maybe_post_open_stop_audit(self):
        """
        Run one extra position sync and stop audit after the market opens.

        The 09:15 ET pre-entry audit catches stale state early, but TRAIL
        orders and broker order status can change after the 09:30 opening
        auction. This audit is intentionally separate from the daily audit so
        it cannot be skipped just because the early audit already ran.
        """
        from src.engine_base import logger, _TZ_NY
        from src.config import POST_OPEN_AUDIT_TIME
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

    def _active_open_trades_for_symbol(self, symbol: str) -> list:
        """Return non-terminal open trades for one symbol."""
        from src.engine_base import logger, _REJECTED_ORDER_STATUSES
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
        from src.engine_base import logger
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
        from src.engine_base import logger, _TZ_NY, _REJECTED_ORDER_STATUSES

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
        from src.engine_base import logger
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

