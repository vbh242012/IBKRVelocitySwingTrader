"""
Comprehensive validation of three critical VelocityEngine subsystems:

  1. Chandelier trailing stop order construction (standalone BUY + post-fill TRAIL)
     - chandelier_dist = ATR_CHAND × CHANDELIER_MULT (2.0)
     - goodAfterTime is omitted after 10:00 ET so IBKR cannot reject a past activation time
     - BUY order: transmit=True; TRAIL stop: standalone GTC transmit=True after fill
     - state.stop_loss  = fill - chandelier_dist
     - No take-profit order or state key

  2. Screener (IB ScannerSubscription parameters)
     - scanCode from config, instrument='STK', location='STK.US.MAJOR'
     - abovePrice=20.0, aboveVolume=2_000_000 (int), marketCapAbove=2000
     - symbol extraction from contractDetails.contract.symbol
     - all unique scanner symbols are returned

  3. Scoring and shortlisting
     - Trend (30pts) · RVOL (25pts) · Momentum (25pts) · Liquidity (20pts)
     - Maximum achievable score = 100
     - Candidates ranked by score; top slots filled first
     - Slot cap respected; already-held symbols skipped before scoring
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
import pytz

from src.scoring import score_candidate, volume_pace_from_intraday


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_fill(commission=1.0):
    """Build a minimal Fill mock carrying a CommissionReport."""
    fill = MagicMock()
    fill.commissionReport.commission = commission
    return fill


def _mock_price_ticker(price: float, bid=None, ask=None):
    """Build a ticker mock with deterministic quote fields."""
    ticker = MagicMock()
    ticker.marketPrice.return_value = price
    ticker.last = price
    ticker.close = price
    ticker.bid = price * 0.999 if bid is None else bid
    ticker.ask = price * 1.001 if ask is None else ask
    return ticker


def _mock_ib():
    """Minimal IB mock suitable for entry-cycle tests."""
    ib = MagicMock()

    # Equity + settled cash (both tags in one list so both helpers work)
    nl = MagicMock(); nl.tag = 'NetLiquidation'; nl.value = '1400.0'
    sc = MagicMock(); sc.tag = 'SettledCash';    sc.value = '5000.0'
    ib.accountSummary.return_value = [nl, sc]

    # Safe VIX
    vix_ticker = MagicMock()
    vix_ticker.marketPrice.return_value = 20.0
    ib.reqTickers.return_value = [vix_ticker]

    # VIX contract qualification
    ib.qualifyContracts.return_value = [MagicMock()]

    # No open IBKR positions — so _sync_positions_from_ibkr leaves state alone
    ib.positions.return_value = []

    # loopUntil must return an iterable — engine: 'for _ in ib.loopUntil(...): pass'
    ib.loopUntil.return_value = [True]

    # reqAllOpenOrders used at startup for orphan cleanup
    ib.reqAllOpenOrders.return_value = []

    # whatIfOrder pre-flight: empty warningText = IB accepts the order
    ib.whatIfOrder.return_value.warningText = ''

    # Default market order response for exit tests. Entry tests override
    # placeOrder with explicit BUY/TRAIL trade mocks.
    sell_trade = MagicMock()
    sell_trade.orderStatus.status = 'Filled'
    sell_trade.orderStatus.filled = 999.0
    ib.placeOrder.return_value = sell_trade

    return ib


def _make_engine(ib_mock):
    """Build a VelocityEngine instance without __init__ (no IB connection)."""
    from src.engine import VelocityEngine
    engine = VelocityEngine.__new__(VelocityEngine)
    engine.ib                  = ib_mock
    engine.state               = {}
    engine._last_equity        = 0.0
    engine._last_settled_cash  = 0.0
    engine._equity_initialized = False
    engine._last_vix           = None
    engine._last_vix_ts        = 0.0
    engine._last_scan_ts       = None
    engine._next_scan_dt       = None
    engine._day_start_equity   = None
    engine._day_start_date     = None
    engine._bar_cache          = {}
    engine._contract_cache     = {}
    engine._vix_contract       = None
    engine._spy_cache          = {'date': '2024-06-05', 'trend': True}
    engine._sector_cache       = {}
    engine._daily_scan_skip    = {}
    engine._last_audit_date    = None
    engine._last_audit_at      = None
    engine._last_post_open_audit_date = None
    engine._last_premarket_readiness_date = None
    engine._last_post_close_maintenance_date = None
    engine._last_eod_exit_date = None
    engine._missing_position_counts = {}
    return engine


def _ctx(price=100.0, orb=95.0, ma50=105.0, ma200=90.0,
         rsi=62.0, rsi_prev=57.0, atr=3.0,
         dollar_vol=500_000_000,
         rvol=3.5, spread_pct=0.002,
         atr5=2.0, atr20=2.5,
         sma200_slope=0.1,
         atr_chandelier=None,
         day_range_location=0.75,
         intraday_gain=0.01,
         day_open=None,
         bid=None, ask=None):
    """Build a get_technical_context()-style dict with all production-rule fields."""
    high10 = round(price * 1.005, 4)   # retained for dashboard/context compatibility
    bid = round(price * (1 - spread_pct / 2), 4) if bid is None else bid
    ask = round(price * (1 + spread_pct / 2), 4) if ask is None else ask
    if day_open is None:
        day_open = price / (1 + intraday_gain) if intraday_gain > -0.99 else price
    return {
        'orb_high':       orb,
        'day_open':       day_open,
        'ma50':           ma50,
        'ma200':          ma200,
        'rsi':            rsi,
        'rsi_prev':       rsi_prev,
        'atr':            atr,
        'atr5':           atr5,
        'atr20':          atr20,
        'atr_chandelier': atr_chandelier if atr_chandelier is not None else atr,
        'sma200_slope':   sma200_slope,
        'high10':         high10,
        'rvol':           rvol,
        'day_range_location': day_range_location,
        'intraday_gain':  intraday_gain,
        'spread_pct':     spread_pct,
        'bid':            bid,
        'ask':            ask,
        'close':          price - 0.5,
        'live_price':     price,
        'volume':         5_000_000,
        'dollar_vol_20d': dollar_vol,
        'contract':       MagicMock(),
    }


def _run_entry_cycle(ib, engine, ctx, sym='TSLA'):
    """
    Run one full run_cycle() with a single passing signal.
    Returns (buy_trade, stop_trade) — the two order mocks from placeOrder calls.
    """
    tz_ny    = pytz.timezone('US/Eastern')
    fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))   # Wed 10:30 — inside window

    fill_price = ctx['live_price']
    from src.config import CHANDELIER_MULT, HARD_STOP_PCT, RISK_PER_TRADE_PCT

    summary = {item.tag: float(item.value) for item in ib.accountSummary.return_value}
    equity = summary.get('NetLiquidation', 1400.0)
    settled = summary.get('SettledCash', 5000.0)
    limit_price = engine._calc_entry_limit_price(ctx['live_price'], ctx['bid'], ctx['ask'])
    allocation = engine._calc_entry_allocation(equity, settled, len(engine.state))
    risk_dist = min(
        round(ctx.get('atr_chandelier', ctx['atr']) * CHANDELIER_MULT, 2),
        round(ctx['live_price'] * HARD_STOP_PCT, 2),
    )
    expected_qty = 0
    if limit_price and risk_dist > 0:
        expected_qty = min(
            int(allocation['bucket_size'] / limit_price),
            int((equity * RISK_PER_TRADE_PCT) / risk_dist),
        )

    buy_trade  = MagicMock()
    buy_trade.order.orderId             = 1          # must be JSON-serializable
    buy_trade.orderStatus.status        = 'Filled'
    buy_trade.orderStatus.filled        = float(expected_qty)
    buy_trade.orderStatus.avgFillPrice  = fill_price
    buy_trade.fills = [_mock_fill(1.0)]              # commission from IB report

    stop_trade = MagicMock()
    stop_trade.order.orderId = 2

    ib.placeOrder.side_effect = [buy_trade, stop_trade]

    with patch.object(engine, 'get_institutional_scan', return_value=[sym]), \
         patch.object(engine, 'get_technical_context', return_value=ctx), \
         patch.object(engine, '_update_position_prices'), \
         patch('src.engine.datetime') as mock_dt:
        mock_dt.now.return_value  = fake_now
        mock_dt.fromisoformat     = datetime.fromisoformat
        engine.run_cycle()

    return buy_trade, stop_trade


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAILING STOP / TAKE-PROFIT ORDER CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

class TestBracketOrderMath:
    """
    Verify chandelier stop distance, GTC 2-order structure, and state persistence.

    Config values used (from src/config.py):
        CHANDELIER_MULT = 2.0   →  chandelier_dist = ATR_CHAND × 2.0

    With atr_chandelier = 3.00:
        chandelier_dist = 6.00
        stop_loss (state) = fill - 6.00 = 94.00
        No take-profit order or state key.
    """

    ATR_CHAND   = 3.00
    ENTRY       = 100.00
    LIMIT       = 100.15                         # ask 100.10 + 5 bps cushion, capped by 0.2%
    CHAND_DIST  = round(3.00 * 2.0, 2)           # 6.00
    INIT_STOP   = round(100.00 - 3.00 * 2.0, 2) # 94.00

    def _setup(self):
        ib      = _mock_ib()
        engine  = _make_engine(ib)
        context = _ctx(price=self.ENTRY, atr=self.ATR_CHAND,
                       atr_chandelier=self.ATR_CHAND,
                       orb=self.ENTRY - 5,
                       ma50=self.ENTRY - 3,
                       ma200=self.ENTRY - 15,
                       rsi=62.0, rsi_prev=57.0)
        return ib, engine, context

    # ── chandelier_dist computation ───────────────────────────────────────────

    def test_chandelier_dist_equals_atr_chandelier_times_mult(self):
        from src.config import CHANDELIER_MULT
        assert CHANDELIER_MULT == 2.0, "CHANDELIER_MULT must be 2.0 (gap-aware forward-validated)"
        chandelier_dist = round(self.ATR_CHAND * CHANDELIER_MULT, 2)
        assert chandelier_dist == self.CHAND_DIST

    def test_chandelier_dist_is_rounded_to_2_decimals(self):
        from src.config import CHANDELIER_MULT
        # Verify rounding behaviour — result should use current multiplier (2.0)
        assert round(3.123 * CHANDELIER_MULT, 2) == round(3.123 * 2.0, 2)

    # ── Order count and structure ─────────────────────────────────────────────

    def test_two_orders_placed_not_three(self):
        """New structure: BUY LMT + TRAIL stop — no take-profit order."""
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx)
        assert ib.placeOrder.call_count == 2, "Exactly 2 placeOrder calls: BUY + TRAIL stop"
        assert engine.state['TSLA']['protection_status'] == 'confirmed'

    def test_buy_order_type_is_lmt(self):
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx)
        buy_order = ib.placeOrder.call_args_list[0][0][1]
        assert buy_order.orderType == 'LMT'

    def test_buy_order_action_is_buy(self):
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx)
        buy_order = ib.placeOrder.call_args_list[0][0][1]
        assert buy_order.action == 'BUY'

    def test_spy_bear_regime_can_enter_with_stricter_bear_rules(self):
        ib, engine, ctx = self._setup()
        engine._spy_cache = {'date': '2024-06-05', 'trend': False}
        ctx.update({
            'orb_high': 98.0,
            'rvol': 5.0,
            'rsi': 72.0,
            'rsi_prev': 68.0,
            'atr5': 1.8,
            'atr20': 2.5,
        })

        _run_entry_cycle(ib, engine, ctx)

        assert ib.placeOrder.call_count == 2
        assert engine.state['TSLA']['regime'] == 'bear'

    def test_spy_bear_regime_does_not_gate_on_rvol_under_8096(self):
        ib, engine, ctx = self._setup()
        engine._spy_cache = {'date': '2024-06-05', 'trend': False}
        ctx.update({
            'orb_high': 98.0,
            'rvol': 2.7,  # below old BEAR_RVOL_MIN, but RVOL is ranking-only in 8096
            'rsi': 72.0,
            'rsi_prev': 68.0,
        })

        _run_entry_cycle(ib, engine, ctx)

        assert ib.placeOrder.call_count == 2
        assert engine.state['TSLA']['regime'] == 'bear'

    def test_buy_order_tif_is_day(self):
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx)
        buy_order = ib.placeOrder.call_args_list[0][0][1]
        assert buy_order.tif == 'DAY'

    def test_buy_order_is_all_or_none(self):
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx)
        buy_order = ib.placeOrder.call_args_list[0][0][1]
        assert buy_order.allOrNone is True

    def test_buy_order_good_after_time_omitted_after_10am_et(self):
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx)
        buy_order = ib.placeOrder.call_args_list[0][0][1]
        assert buy_order.goodAfterTime == ''

    def test_buy_order_transmit_is_true(self):
        # BUY is now transmitted immediately (standalone); TRAIL stop is placed
        # after the fill is confirmed to avoid cash-account "short sell" rejections.
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx)
        buy_order = ib.placeOrder.call_args_list[0][0][1]
        assert buy_order.transmit == True

    def test_stop_order_type_is_trail(self):
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx)
        stop_order = ib.placeOrder.call_args_list[1][0][1]
        assert stop_order.orderType == 'TRAIL'

    def test_stop_order_aux_price_equals_chandelier_dist(self):
        """auxPrice (trail amount in $) = ATR_CHAND × CHANDELIER_MULT."""
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx)
        stop_order = ib.placeOrder.call_args_list[1][0][1]
        assert stop_order.auxPrice == pytest.approx(self.CHAND_DIST, abs=0.01)

    def test_stop_order_tif_is_gtc(self):
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx)
        stop_order = ib.placeOrder.call_args_list[1][0][1]
        assert stop_order.tif == 'GTC'

    def test_stop_order_good_after_time_omitted_after_10am_et(self):
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx)
        stop_order = ib.placeOrder.call_args_list[1][0][1]
        assert stop_order.goodAfterTime == ''

    def test_stop_order_transmit_is_true(self):
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx)
        stop_order = ib.placeOrder.call_args_list[1][0][1]
        assert stop_order.transmit == True

    def test_buy_order_qty_uses_dynamic_cash_bucket_and_risk_cap(self):
        """qty is whole shares sized by settled-cash bucket and ATR risk cap."""
        from src.config import (
            CHANDELIER_MULT,
            HARD_STOP_PCT,
            MAX_POSITIONS_CAP,
            MIN_BUCKET_SIZE,
            RISK_PER_TRADE_PCT,
            SETTLED_CASH_DEPLOYMENT_PCT,
        )
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx)
        buy_order = ib.placeOrder.call_args_list[0][0][1]
        equity = 1400.0
        settled = 5000.0
        max_pos = min(int(equity / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)
        bucket_size = (settled * SETTLED_CASH_DEPLOYMENT_PCT) / max_pos
        chandelier_dist = ctx['atr_chandelier'] * CHANDELIER_MULT
        hard_stop_dist = self.ENTRY * HARD_STOP_PCT
        risk_stop_dist = min(chandelier_dist, hard_stop_dist)
        bucket_qty = int(bucket_size / self.LIMIT)
        risk_qty = int((equity * RISK_PER_TRADE_PCT) / risk_stop_dist)
        expected_qty = min(bucket_qty, risk_qty)
        assert buy_order.totalQuantity == pytest.approx(expected_qty, abs=0.0001)
        assert buy_order.lmtPrice == pytest.approx(self.LIMIT, abs=0.01)

    def test_high_priced_stock_skipped_when_qty_would_be_zero(self):
        """price above bucket_size → int qty = 0 → skip; no placeOrder calls."""
        ib, engine, ctx = self._setup()
        ctx['live_price']     = 10000.0
        ctx['bid']            = 9990.0
        ctx['ask']            = 10010.0
        ctx['spread_pct']     = (ctx['ask'] - ctx['bid']) / ((ctx['ask'] + ctx['bid']) / 2)
        ctx['day_open']       = ctx['orb_high']
        ctx['atr']            = 10.0
        ctx['atr_chandelier'] = 10.0
        _run_entry_cycle(ib, engine, ctx)
        assert ib.placeOrder.call_count == 0, "Stock above bucket price must be skipped (int qty = 0)"
        assert 'TSLA' not in engine.state

    # ── State persistence ────────────────────────────────────────────────────

    def test_state_stop_loss_equals_fill_minus_chandelier_dist(self):
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx, sym='TSLA')
        assert 'TSLA' in engine.state
        sl = engine.state['TSLA']['stop_loss']
        assert sl == pytest.approx(self.ENTRY - self.CHAND_DIST, abs=0.01)

    def test_state_has_no_take_profit(self):
        """New structure has no take-profit — state must not contain that key."""
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx, sym='TSLA')
        assert 'take_profit' not in engine.state.get('TSLA', {})

    def test_state_records_fill_price_commission_and_order_id(self):
        """state stores raw fill_price, actual commission from IB report, and entry_order_id."""
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx, sym='TSLA')
        s = engine.state['TSLA']
        assert s['price']      == pytest.approx(self.ENTRY, abs=0.01)
        assert s['fill_price'] == pytest.approx(self.ENTRY, abs=0.01)
        assert s['commission'] == pytest.approx(1.0, abs=0.001)
        assert 'entry_order_id' in s

    # ── Unconfirmed order — state must NOT be written ────────────────────────

    def test_no_state_written_when_order_unconfirmed(self):
        """If IBKR returns status other than Filled/Submitted, state stays empty."""
        ib, engine, ctx = self._setup()

        mock_trade = MagicMock()
        mock_trade.orderStatus.status       = 'Cancelled'
        mock_trade.orderStatus.avgFillPrice = 0.0
        mock_trade.fills = []
        ib.placeOrder.return_value = mock_trade

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['TSLA']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert 'TSLA' not in engine.state, "Unconfirmed order must not create state entry"

    def test_fill_price_zero_falls_back_to_limit_price(self):
        """avgFillPrice=0 → state records raw limit_price (scan×1.002) as both price and fill_price."""
        ib, engine, ctx = self._setup()

        buy_trade = MagicMock()
        buy_trade.order.orderId             = 2
        buy_trade.orderStatus.status        = 'Filled'
        buy_trade.orderStatus.avgFillPrice  = 0     # IB sometimes returns 0
        buy_trade.fills = [_mock_fill(1.0)]

        stop_trade = MagicMock()
        ib.placeOrder.side_effect = [buy_trade, stop_trade]

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['TSLA']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert 'TSLA' in engine.state
        assert engine.state['TSLA']['price']      == pytest.approx(self.LIMIT, abs=0.01)
        assert engine.state['TSLA']['fill_price'] == pytest.approx(self.LIMIT, abs=0.01)

    def test_unconfirmed_protective_stop_halts_additional_entries_this_cycle(self):
        """After a filled BUY, no second entry is allowed until protection is confirmed."""
        ib, engine, ctx = self._setup()

        buy_trade = MagicMock()
        buy_trade.order.orderId = 11
        buy_trade.orderStatus.status = 'Filled'
        buy_trade.orderStatus.filled = 4.0
        buy_trade.orderStatus.avgFillPrice = self.ENTRY
        buy_trade.fills = [_mock_fill(1.0)]

        stop_trade = MagicMock()
        stop_trade.orderStatus.status = 'Submitted'
        ib.placeOrder.side_effect = [buy_trade, stop_trade]

        tz_ny = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['TSLA', 'NVDA']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_confirm_protective_stop', return_value=False) as mock_confirm, \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch.object(engine, '_alert') as mock_alert, \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine.run_cycle()

        assert ib.placeOrder.call_count == 2
        assert mock_confirm.call_count == 2
        mock_audit.assert_called_once()
        assert mock_alert.call_args.args[0] == "CRITICAL"
        assert "STOP UNCONFIRMED" in mock_alert.call_args.args[1]
        assert engine.state['TSLA']['protection_status'] == 'unconfirmed'
        assert 'NVDA' not in engine.state

    def test_partial_fill_cancels_oversized_child_stop_and_audits(self):
        """A partial fill must not leave a full-quantity child TRAIL sell live."""
        ib, engine, ctx = self._setup()

        buy_trade = MagicMock()
        buy_trade.order.orderId             = 7
        buy_trade.orderStatus.status        = 'Filled'
        buy_trade.orderStatus.filled        = 1.0
        buy_trade.orderStatus.avgFillPrice  = self.ENTRY
        buy_trade.fills = [_mock_fill(1.0)]

        stop_trade = MagicMock()
        stop_trade.orderStatus.status = 'Submitted'

        ib.placeOrder.side_effect = [buy_trade, stop_trade]

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['TSLA']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        stop_order = ib.placeOrder.call_args_list[1].args[1]
        ib.cancelOrder.assert_any_call(stop_order)
        mock_audit.assert_called_once()
        assert engine.state['TSLA']['qty'] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCREENER — IB ScannerSubscription parameters
# ─────────────────────────────────────────────────────────────────────────────

class TestEntryLimitPricing:
    def test_entry_limit_price_is_ask_based_with_market_cap(self):
        from src.engine import VelocityEngine
        from src.config import ENTRY_LIMIT_ASK_CUSHION_PCT, ENTRY_LIMIT_MIN_TICK

        price = 100.00
        bid = 99.90
        ask = 100.10
        expected = round(ask + max(ENTRY_LIMIT_MIN_TICK, ask * ENTRY_LIMIT_ASK_CUSHION_PCT), 2)

        limit = VelocityEngine._calc_entry_limit_price(price, bid, ask)

        assert limit == pytest.approx(expected)
        assert limit < round(price * 1.002, 2)

    def test_entry_limit_price_rejects_ask_above_max_over_market_cap(self):
        from src.engine import VelocityEngine

        assert VelocityEngine._calc_entry_limit_price(100.00, 100.10, 100.25) is None

    def test_entry_limit_price_rejects_wide_spread(self):
        from src.engine import VelocityEngine

        assert VelocityEngine._calc_entry_limit_price(100.00, 99.00, 100.00) is None


class TestScannerSubscription:
    """
    build_momentum_scanner() must produce a ScannerSubscription whose fields
    exactly match the strategy spec.  All assertions are on the returned object,
    no IB connection needed.
    """

    def setup_method(self):
        from src.scanner import build_momentum_scanner
        self.sub = build_momentum_scanner()

    # ── Subscription fields ───────────────────────────────────────────────────

    def test_scan_code_matches_config(self):
        from src.config import IB_SCANNER_SCAN_CODE
        assert self.sub.scanCode == IB_SCANNER_SCAN_CODE
        assert self.sub.scanCode == 'MOST_ACTIVE'

    def test_instrument_is_stk(self):
        assert self.sub.instrument == 'STK'

    def test_location_is_us_major(self):
        from src.config import IB_SCANNER_LOCATION_CODE
        assert self.sub.locationCode == IB_SCANNER_LOCATION_CODE
        assert self.sub.locationCode == 'STK.US.MAJOR'

    def test_number_of_rows_uses_configured_scanner_limit(self):
        from src.config import IB_SCANNER_ROWS
        assert self.sub.numberOfRows == IB_SCANNER_ROWS

    def test_min_price_matches_config(self):
        from src.config import SCAN_MIN_PRICE
        assert self.sub.abovePrice == SCAN_MIN_PRICE
        assert self.sub.abovePrice == 20.0

    def test_min_volume_matches_config(self):
        from src.config import SCAN_MIN_VOLUME
        assert self.sub.aboveVolume == SCAN_MIN_VOLUME
        assert self.sub.aboveVolume == 2_000_000

    def test_min_volume_is_integer(self):
        """IB rejects float for aboveVolume; must be int."""
        assert isinstance(self.sub.aboveVolume, int)

    def test_market_cap_converted_to_millions(self):
        """
        SCAN_MIN_MKTCAP = 2_000_000_000 (2 billion).
        IB's marketCapAbove field is in millions → must be 2000.
        """
        from src.config import SCAN_MIN_MKTCAP
        assert self.sub.marketCapAbove == SCAN_MIN_MKTCAP / 1_000_000
        assert self.sub.marketCapAbove == 2000

    def test_stock_type_filter_is_corp(self):
        """stockTypeFilter='CORP' excludes ETFs at scanner level."""
        assert self.sub.stockTypeFilter == 'CORP'


class TestGetInstitutionalScan:
    """
    get_institutional_scan() must extract symbols from IB scan results
    via the correct attribute path and return every unique symbol.
    """

    def _make_scan_item(self, symbol):
        item = MagicMock()
        item.contractDetails.contract.symbol = symbol
        return item

    def test_returns_list_of_symbols(self):
        ib     = _mock_ib()
        engine = _make_engine(ib)
        items  = [self._make_scan_item('AAPL'), self._make_scan_item('TSLA')]
        ib.reqScannerData.return_value = items

        result = engine.get_institutional_scan()

        assert result == ['AAPL', 'TSLA']

    def test_symbols_extracted_from_contract_details_contract_symbol(self):
        """Ensures the correct nested attribute path is used."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        item   = self._make_scan_item('NVDA')
        ib.reqScannerData.return_value = [item]

        result = engine.get_institutional_scan()

        assert result == ['NVDA']

    def test_returns_all_unique_scanner_symbols_without_fixed_cap(self):
        ib     = _mock_ib()
        engine = _make_engine(ib)
        items  = [self._make_scan_item(f'SYM{i}') for i in range(55)]
        ib.reqScannerData.return_value = items

        result = engine.get_institutional_scan()

        assert len(result) == 55
        assert result[0] == 'SYM0'
        assert result[-1] == 'SYM54'

    def test_duplicate_scanner_symbols_are_deduped_preserving_order(self):
        ib     = _mock_ib()
        engine = _make_engine(ib)
        items  = [
            self._make_scan_item('AAPL'),
            self._make_scan_item('TSLA'),
            self._make_scan_item('AAPL'),
            self._make_scan_item('NVDA'),
        ]
        ib.reqScannerData.return_value = items

        result = engine.get_institutional_scan()

        assert result == ['AAPL', 'TSLA', 'NVDA']

    def test_empty_scan_returns_empty_list(self):
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ib.reqScannerData.return_value = []

        result = engine.get_institutional_scan()

        assert result == []

    def test_scanner_subscription_passed_to_ib(self):
        """reqScannerData must be called once per configured scan code."""
        from src.config import IB_SCANNER_SCAN_CODES
        from ib_async import ScannerSubscription
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ib.reqScannerData.return_value = []

        engine.get_institutional_scan()

        assert ib.reqScannerData.call_count == len(IB_SCANNER_SCAN_CODES)
        for call in ib.reqScannerData.call_args_list:
            assert isinstance(call.args[0], ScannerSubscription)
        used_codes = {call.args[0].scanCode for call in ib.reqScannerData.call_args_list}
        assert used_codes == set(IB_SCANNER_SCAN_CODES)

    def test_symbols_from_multiple_scanners_are_deduped(self):
        """Symbols appearing in more than one scanner are only returned once."""
        from src.config import IB_SCANNER_SCAN_CODES
        n  = len(IB_SCANNER_SCAN_CODES)
        ib = _mock_ib()
        engine = _make_engine(ib)
        # Each scanner returns AAPL (shared) plus one unique symbol
        ib.reqScannerData.side_effect = [
            [self._make_scan_item('AAPL'), self._make_scan_item(f'SYM{i}')]
            for i in range(n)
        ]

        result = engine.get_institutional_scan()

        assert result.count('AAPL') == 1
        assert len(result) == n + 1  # AAPL + n unique SYMs

    def test_failed_scanner_is_skipped_others_still_run(self):
        """One failing scanner should not stop the remaining scanners."""
        from src.config import IB_SCANNER_SCAN_CODES
        n  = len(IB_SCANNER_SCAN_CODES)
        if n < 2:
            return  # test only meaningful with multiple scan codes
        ib = _mock_ib()
        engine = _make_engine(ib)
        side_effects = [Exception("pacing")] + [
            [self._make_scan_item('NVDA')] for _ in range(n - 1)
        ]
        ib.reqScannerData.side_effect = side_effects

        result = engine.get_institutional_scan()

        assert 'NVDA' in result


class TestCashBucketBuffer:
    def test_deployable_settled_cash_keeps_configured_buffer(self):
        from src.config import SETTLED_CASH_DEPLOYMENT_PCT

        engine = _make_engine(_mock_ib())

        assert SETTLED_CASH_DEPLOYMENT_PCT == pytest.approx(0.95)
        assert engine._deployable_settled_cash(1000.0) == pytest.approx(950.0)

    def test_cash_entry_slots_use_deployable_settled_cash(self):
        engine = _make_engine(_mock_ib())

        assert engine._calc_cash_entry_slots(500.0) == 0
        assert engine._calc_cash_entry_slots(530.0) == 1

    def test_entry_allocation_dynamically_recomputes_slots_and_bucket(self):
        engine = _make_engine(_mock_ib())

        first = engine._calc_entry_allocation(equity=1400.0, settled=5000.0, open_count=0)
        assert first['max_pos'] == 2
        assert first['entry_slots'] == 2
        assert first['bucket_size'] == pytest.approx(2375.0)

        after_one_fill = engine._calc_entry_allocation(equity=1400.0, settled=4500.0, open_count=1)
        assert after_one_fill['entry_slots'] == 1
        assert after_one_fill['bucket_size'] == pytest.approx(4275.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SCORING SYSTEM — _score_candidate()
#    Four components summing to 100:
#      Trend (30 pts) · RVOL (25 pts) · Momentum (25 pts) · Liquidity (20 pts)
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringTrendStrength:
    """
    Trend strength component (0-30 pts):
      sep   = (MA50 - MA200) / MA200 × 100
      trend = max(0, min(sep × 5, 30))

      sep ≥ 6% → 30 pts (saturated)
      sep = 5% → 25 pts
      sep = 3% → 15 pts
      sep = 1% → 5 pts
      Bearish (MA50 < MA200) → 0 pts (floored)
    """

    def _trend_score(self, ma50, ma200):
        """
        Isolate trend component.
        Zero-out other components:
          rvol=RVOL_MIN → rvol_score=0; rsi delta=0 + rsi≤70 → momentum=10;
          spread=0 → liquidity=20  →  subtract 0+10+20=30 from total.
        """
        ctx = _ctx(price=100.5, orb=100.0,
                   rsi=65.0, rsi_prev=65.0,
                   ma50=ma50, ma200=ma200,
                   rvol=2.5, spread_pct=0.0)
        return score_candidate(ctx, model="legacy") - 30

    def test_6pct_separation_gives_30_pts(self):
        assert self._trend_score(106.0, 100.0) == pytest.approx(30.0, abs=0.1)

    def test_above_6pct_capped_at_30(self):
        assert self._trend_score(120.0, 100.0) == pytest.approx(30.0, abs=0.1)

    def test_5pct_separation_gives_25_pts(self):
        assert self._trend_score(105.0, 100.0) == pytest.approx(25.0, abs=0.1)

    def test_3pct_separation_gives_15_pts(self):
        assert self._trend_score(103.0, 100.0) == pytest.approx(15.0, abs=0.1)

    def test_1pct_separation_gives_5_pts(self):
        assert self._trend_score(101.0, 100.0) == pytest.approx(5.0, abs=0.1)

    def test_bearish_trend_floored_at_zero(self):
        assert self._trend_score(95.0, 100.0) == pytest.approx(0.0, abs=0.1)

    def test_equal_mas_gives_zero_trend(self):
        assert self._trend_score(100.0, 100.0) == pytest.approx(0.0, abs=0.1)


class TestScoringRVOL:
    """
    RVOL component (0-25 pts):
      rvol_score = min(max(rvol - RVOL_MIN, 0) / RVOL_MIN × 25, 25)

      rvol = RVOL_MIN (2.5×) → 0 pts (at floor)
      rvol = 3.75×           → 12.5 pts
      rvol = 5.0× (2×floor)  → 25 pts (saturated)
      rvol > 5.0×            → capped at 25
      rvol < floor           → 0 pts
    """

    def _rvol_score(self, rvol):
        """
        Isolate RVOL component.
        Zero-out others: ma50=ma200 → trend=0; rsi delta=0 → accel=0;
        rsi≤70 → level=10; spread=0 → liquidity=20  →  subtract 0+10+20=30.
        """
        ctx = _ctx(price=100.5, orb=100.0,
                   rsi=65.0, rsi_prev=65.0,
                   ma50=100.0, ma200=100.0,
                   rvol=rvol, spread_pct=0.0)
        return score_candidate(ctx, model="legacy") - 30

    def test_rvol_at_floor_gives_zero(self):
        from src.config import RVOL_MIN
        assert self._rvol_score(RVOL_MIN) == pytest.approx(0.0, abs=0.1)

    def test_rvol_at_midpoint_gives_12_5(self):
        # rvol=3.75 = 2.5 + 1.25 (half of 2×floor - floor) → 12.5 pts
        assert self._rvol_score(3.75) == pytest.approx(12.5, abs=0.1)

    def test_rvol_at_2x_floor_gives_25(self):
        assert self._rvol_score(5.0) == pytest.approx(25.0, abs=0.1)

    def test_rvol_above_2x_floor_capped_at_25(self):
        assert self._rvol_score(7.0) == pytest.approx(25.0, abs=0.1)

    def test_rvol_below_floor_gives_zero(self):
        assert self._rvol_score(1.0) == pytest.approx(0.0, abs=0.1)


class TestScoringMomentum:
    """
    Momentum component (0-25 pts):
      accel = min(max(delta × 1.5, 0), 15)
      level: RSI ≤ 70 → 10; RSI ≤ 75 → 5; RSI > 75 → max(0, 10-(RSI-75)×2)
      momentum = accel + level
    """

    def _momentum_score(self, rsi, rsi_prev):
        """
        Isolate momentum component.
        Zero-out others: ma50=ma200 → trend=0; rvol=floor → rvol_score=0;
        spread=0 → liquidity=20  →  subtract 0+0+20=20.
        """
        ctx = _ctx(price=100.5, orb=100.0,
                   rsi=rsi, rsi_prev=rsi_prev,
                   ma50=100.0, ma200=100.0,
                   rvol=2.5, spread_pct=0.0)
        return score_candidate(ctx, model="legacy") - 20

    # ── RSI level tiers ───────────────────────────────────────────────────────

    def test_rsi_65_delta_0_gives_10_pts(self):
        # accel=0, level=10 → 10
        assert self._momentum_score(65.0, 65.0) == pytest.approx(10.0, abs=0.1)

    def test_rsi_72_delta_0_gives_5_pts(self):
        # 70 < 72 ≤ 75 → level=5, accel=0 → 5
        assert self._momentum_score(72.0, 72.0) == pytest.approx(5.0, abs=0.1)

    def test_rsi_78_delta_0_gives_4_pts(self):
        # rsi=78 > 75 → level = max(0, 10-(78-75)×2) = max(0, 4) = 4
        assert self._momentum_score(78.0, 78.0) == pytest.approx(4.0, abs=0.1)

    def test_rsi_80_delta_0_gives_0_pts(self):
        # rsi=80 → level = max(0, 10-5×2) = 0
        assert self._momentum_score(80.0, 80.0) == pytest.approx(0.0, abs=0.1)

    def test_rsi_82_delta_0_gives_0_pts(self):
        # rsi=82 → level = max(0, 10-7×2) = max(0, -4) = 0
        assert self._momentum_score(82.0, 82.0) == pytest.approx(0.0, abs=0.1)

    # ── RSI acceleration ──────────────────────────────────────────────────────

    def test_delta_5_gives_7_5_acceleration(self):
        # delta=5: accel=min(7.5,15)=7.5; level=10; total=17.5
        assert self._momentum_score(65.0, 60.0) == pytest.approx(17.5, abs=0.1)

    def test_delta_10_gives_full_15_acceleration(self):
        # delta=10: accel=min(15,15)=15; level=10; total=25
        assert self._momentum_score(65.0, 55.0) == pytest.approx(25.0, abs=0.1)

    def test_delta_above_10_capped(self):
        # delta=20: accel=min(30,15)=15; level=10; total=25 (same as delta=10)
        assert self._momentum_score(65.0, 45.0) == pytest.approx(25.0, abs=0.1)

    def test_negative_delta_gives_zero_acceleration(self):
        # delta=-5: accel=max(-7.5,0)=0; level=10; total=10
        assert self._momentum_score(60.0, 65.0) == pytest.approx(10.0, abs=0.1)

    def test_max_momentum_score_is_25(self):
        assert self._momentum_score(65.0, 55.0) == pytest.approx(25.0, abs=0.1)


class TestScoringLiquidity:
    """
    Liquidity component (0-20 pts):
      liquidity = max(0, (SPREAD_MAX_PCT - spread_pct) / SPREAD_MAX_PCT × 20)

      spread = 0%    → 20 pts
      spread = 0.25% → 10 pts
      spread = 0.5%  → 0 pts
      spread > 0.5%  → clamped to 0
    """

    def _liquidity_score(self, spread_pct):
        """
        Isolate liquidity component.
        Zero-out others: ma50=ma200 → trend=0; rvol=floor → rvol_score=0;
        rsi delta=0 + rsi≤70 → momentum=10  →  subtract 0+0+10=10.
        """
        ctx = _ctx(price=100.5, orb=100.0,
                   rsi=65.0, rsi_prev=65.0,
                   ma50=100.0, ma200=100.0,
                   rvol=2.5, spread_pct=spread_pct)
        return score_candidate(ctx, model="legacy") - 10

    def test_zero_spread_gives_20_pts(self):
        assert self._liquidity_score(0.0) == pytest.approx(20.0, abs=0.1)

    def test_half_spread_gives_10_pts(self):
        # spread = SPREAD_MAX_PCT / 2 = 0.0025 → 10 pts
        assert self._liquidity_score(0.0025) == pytest.approx(10.0, abs=0.1)

    def test_max_spread_gives_zero(self):
        from src.config import SPREAD_MAX_PCT
        assert self._liquidity_score(SPREAD_MAX_PCT) == pytest.approx(0.0, abs=0.1)

    def test_above_max_spread_clamped_at_zero(self):
        assert self._liquidity_score(0.01) == pytest.approx(0.0, abs=0.1)


class TestScoringMaxAndTotal:
    """Integration: verify total score = trend + rvol_score + momentum + liquidity."""

    def test_maximum_achievable_score_is_100(self):
        """
        Perfect conditions:
          ma50=106, ma200=100 → sep=6% → trend=30
          rvol=5.0            → rvol_score=25
          rsi=65, rsi_prev=55 → delta=10, accel=15, level=10 → momentum=25
          spread=0            → liquidity=20
          Total = 30 + 25 + 25 + 20 = 100
        """
        ctx = _ctx(price=101.0, orb=100.0,
                   rsi=65.0, rsi_prev=55.0,
                   ma50=106.0, ma200=100.0,
                   rvol=5.0, spread_pct=0.0)
        assert score_candidate(ctx, model="legacy") == pytest.approx(100.0, abs=0.1)

    def test_score_never_negative(self):
        """Even with all-bad inputs, score must be ≥ 0."""
        ctx = _ctx(price=90.0, orb=100.0,
                   rsi=82.0, rsi_prev=85.0,
                   ma50=90.0, ma200=100.0,
                   rvol=1.0, spread_pct=0.01)
        assert score_candidate(ctx, model="legacy") >= 0.0

    def test_score_never_exceeds_100(self):
        """No combination of inputs should exceed 100."""
        ctx = _ctx(price=101.0, orb=100.0,
                   rsi=60.0, rsi_prev=20.0,
                   ma50=200.0, ma200=100.0,
                   rvol=10.0, spread_pct=0.0)
        assert score_candidate(ctx, model="legacy") <= 100.0

    def test_score_is_rounded_to_2_decimals(self):
        ctx = _ctx(price=101.0, orb=100.0,
                   rsi=65.0, rsi_prev=55.0,
                   ma50=106.0, ma200=100.0,
                   rvol=5.0, spread_pct=0.0)
        score = score_candidate(ctx, model="legacy")
        assert score == round(score, 2)


class TestSharedEnhancedScoring:
    """Shared scorer checks for the optional enhanced ranking model."""

    def test_live_volume_pace_normalizes_early_session_volume(self):
        tz_ny = pytz.timezone("US/Eastern")
        now = tz_ny.localize(datetime(2026, 6, 1, 10, 0))

        # 100k shares in the first 30 of 390 regular-session minutes is
        # running near a 1.3x full-day pace against a 1M-share average day.
        assert volume_pace_from_intraday(100_000, 1_000_000, now) == pytest.approx(1.3)

    def test_legacy_scorer_uses_volume_pace_when_available(self):
        ctx = _ctx(rvol=1.0)
        ctx["volume_pace"] = 5.0

        with_pace = score_candidate(ctx, model="legacy")
        raw_only = score_candidate({**ctx, "volume_pace": 1.0}, model="legacy")

        assert with_pace > raw_only

    def test_enhanced_score_softens_high_rsi_penalty(self):
        base = _ctx(price=101.5, orb=100.0, rsi=80.0, rsi_prev=80.0,
                    ma50=106.0, ma200=100.0, rvol=5.0, spread_pct=0.0)

        assert score_candidate(base, model="enhanced") > score_candidate(base, model="legacy")

    def test_enhanced_score_rewards_clean_breakout_not_stretched_chase(self):
        clean = _ctx(price=101.5, orb=100.0, rsi=68.0, rsi_prev=62.0,
                     ma50=106.0, ma200=100.0, rvol=5.0, spread_pct=0.0,
                     atr_chandelier=2.0)
        stretched = _ctx(price=109.0, orb=100.0, rsi=68.0, rsi_prev=62.0,
                         ma50=106.0, ma200=100.0, rvol=5.0, spread_pct=0.0,
                         atr_chandelier=2.0)

        assert score_candidate(clean, model="enhanced") > score_candidate(stretched, model="enhanced")

    def test_enhanced_score_prefers_cleaner_atr_risk(self):
        clean = _ctx(price=101.5, orb=100.0, rsi=68.0, rsi_prev=62.0,
                     ma50=106.0, ma200=100.0, rvol=5.0, spread_pct=0.0,
                     atr_chandelier=2.0)
        wild = _ctx(price=101.5, orb=100.0, rsi=68.0, rsi_prev=62.0,
                    ma50=106.0, ma200=100.0, rvol=5.0, spread_pct=0.0,
                    atr_chandelier=12.0)

        assert score_candidate(clean, model="enhanced") > score_candidate(wild, model="enhanced")


class TestLegacyV2Scoring:
    """Legacy v2 keeps the legacy core and adds bounded quality tie-breakers."""

    def test_legacy_v2_rewards_liquid_clean_breakout(self):
        clean = _ctx(price=101.5, orb=100.0, rsi=68.0, rsi_prev=62.0,
                     ma50=103.0, ma200=100.0, rvol=3.0, spread_pct=0.001,
                     atr_chandelier=2.0, dollar_vol=900_000_000)
        weak = _ctx(price=101.5, orb=100.0, rsi=68.0, rsi_prev=62.0,
                    ma50=103.0, ma200=100.0, rvol=3.0, spread_pct=0.001,
                    atr_chandelier=2.0, dollar_vol=110_000_000)

        assert score_candidate(clean, model="legacy_v2") > score_candidate(weak, model="legacy_v2")

    def test_legacy_v2_penalizes_stretched_or_wild_candidates(self):
        clean = _ctx(price=102.0, orb=100.0, rsi=68.0, rsi_prev=62.0,
                     ma50=103.0, ma200=100.0, rvol=3.0, spread_pct=0.001,
                     atr_chandelier=2.0, dollar_vol=500_000_000)
        stretched_wild = _ctx(price=111.0, orb=100.0, rsi=68.0, rsi_prev=62.0,
                              ma50=103.0, ma200=100.0, rvol=3.0, spread_pct=0.001,
                              atr_chandelier=12.0, dollar_vol=500_000_000)

        assert score_candidate(clean, model="legacy_v2") > score_candidate(stretched_wild, model="legacy_v2")

    def test_legacy_v2_softens_high_rsi_only_when_rising(self):
        rising = _ctx(price=101.5, orb=100.0, rsi=80.0, rsi_prev=75.0,
                      ma50=103.0, ma200=100.0, rvol=3.0, spread_pct=0.001,
                      atr_chandelier=2.0, dollar_vol=500_000_000)
        flat = _ctx(price=101.5, orb=100.0, rsi=80.0, rsi_prev=80.0,
                    ma50=103.0, ma200=100.0, rvol=3.0, spread_pct=0.001,
                    atr_chandelier=2.0, dollar_vol=500_000_000)

        assert score_candidate(rising, model="legacy_v2") > score_candidate(flat, model="legacy_v2")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHORTLISTING — ranking, slot limits, portfolio exclusion
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateRanking:
    """
    Signals are sorted descending by score; top N (available slots) are entered.
    """

    def _make_ib_with_positions(self, syms, equity=1400.0, settled=5000.0):
        """Return a mock IB that reports these symbols as open positions."""
        ib = _mock_ib()
        nl = MagicMock(); nl.tag = 'NetLiquidation'; nl.value = str(equity)
        sc = MagicMock(); sc.tag = 'SettledCash';    sc.value = str(settled)
        ib.accountSummary.return_value = [nl, sc]
        mocked = []
        for sym in syms:
            p = MagicMock()
            p.contract.symbol = sym
            p.position        = 10.0
            p.avgCost         = 100.0
            mocked.append(p)
        ib.positions.return_value = mocked
        return ib

    def _run_multi_signal_cycle(self, engine, ib, ctx_map):
        """
        Run a cycle where get_institutional_scan returns ctx_map.keys()
        and get_technical_context dispatches to ctx_map by symbol.
        Returns the set of symbols for which placeOrder was called.
        """
        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        symbols = list(ctx_map.keys())

        mock_trade = MagicMock()
        mock_trade.order.orderId             = 1      # must be JSON-serializable
        mock_trade.orderStatus.status        = 'Filled'
        mock_trade.orderStatus.avgFillPrice  = 100.0
        mock_trade.fills = [_mock_fill(1.0)]
        ib.placeOrder.return_value = mock_trade   # same mock for both BUY and TRAIL calls

        def _ctx_for(sym):
            return ctx_map.get(sym)

        with patch.object(engine, 'get_institutional_scan', return_value=symbols), \
             patch.object(engine, 'get_technical_context', side_effect=_ctx_for), \
             patch.object(engine, '_confirm_protective_stop', return_value=True), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, 'manage_position_exits'), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        # Return the state keys that were added (= symbols entered)
        return set(engine.state.keys())

    # ── Ordering ─────────────────────────────────────────────────────────────

    def test_highest_score_candidate_entered_when_one_slot(self):
        """
        Two signals, 2 existing positions → 1 slot.
        Only the higher-score candidate may be entered.
        HIGH: rvol=5.0 → rvol_score=25 (higher score)
        LOW:  rvol=2.5 → rvol_score=0  (lower score, same other params)
        """
        # ma50=95, ma200=85 ensures price > ma50 > ma200 ✓; orb=100 ensures price > orb ✓
        ctx_high = _ctx(price=101.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                        ma50=95.0, ma200=85.0, rvol=5.0)
        ctx_low  = _ctx(price=102.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                        ma50=95.0, ma200=85.0, rvol=2.5)

        ib = self._make_ib_with_positions(['SYM0', 'SYM1'], equity=1500.0)
        engine = _make_engine(ib)
        engine.state = {
            'SYM0': {'price': 100, 'time': datetime.now().isoformat()},
            'SYM1': {'price': 100, 'time': datetime.now().isoformat()},
        }

        entered = self._run_multi_signal_cycle(
            engine, ib, {'HIGH': ctx_high, 'LOW': ctx_low}
        )

        assert 'HIGH' in entered
        assert 'LOW' not in entered

    def test_lower_score_candidate_skipped_when_slot_filled(self):
        """Converse: the lower-score stock is NOT entered when only 1 slot exists."""
        ctx_high = _ctx(price=101.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                        ma50=95.0, ma200=85.0, rvol=5.0)
        ctx_low  = _ctx(price=102.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                        ma50=95.0, ma200=85.0, rvol=2.5)

        ib = self._make_ib_with_positions(['SYM0', 'SYM1'], equity=1500.0)
        engine = _make_engine(ib)
        engine.state = {
            'SYM0': {'price': 100, 'time': datetime.now().isoformat()},
            'SYM1': {'price': 100, 'time': datetime.now().isoformat()},
        }

        entered = self._run_multi_signal_cycle(
            engine, ib, {'HIGH': ctx_high, 'LOW': ctx_low}
        )
        assert 'LOW' not in entered

    def test_all_candidates_entered_when_enough_slots(self):
        """Two dynamic slots free, two signals → both entered."""
        ctx_a = _ctx(price=101.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                     ma50=95.0, ma200=85.0)
        ctx_b = _ctx(price=104.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                     ma50=95.0, ma200=85.0)

        ib     = _mock_ib()
        engine = _make_engine(ib)
        # state empty and $1,400 equity → two dynamic slots.

        entered = self._run_multi_signal_cycle(
            engine, ib, {'ALPHA': ctx_a, 'BETA': ctx_b}
        )

        assert 'ALPHA' in entered
        assert 'BETA' in entered

    def test_entry_order_is_score_descending(self):
        """
        With 3 candidates and 2 slots, the top 2 by score must be entered
        regardless of the order the scanner returns them.
        Score differentiated by RVOL:
          HIGH:  rvol=5.0  → rvol_score=25 (highest)
          MED:   rvol=3.75 → rvol_score=12.5
          LOW:   rvol=2.5  → rvol_score=0  (lowest)
        """
        ctx_hi  = _ctx(price=101.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                       ma50=95.0, ma200=85.0, rvol=5.0)
        ctx_med = _ctx(price=104.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                       ma50=95.0, ma200=85.0, rvol=3.75)
        ctx_lo  = _ctx(price=108.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                       ma50=95.0, ma200=85.0, rvol=2.5)

        ib = self._make_ib_with_positions(['HELD'], equity=1500.0)
        engine = _make_engine(ib)
        engine.state = {'HELD': {'price': 100, 'time': datetime.now().isoformat()}}

        entered = self._run_multi_signal_cycle(
            engine, ib,
            {'MED': ctx_med, 'HIGH': ctx_hi, 'LOW': ctx_lo}
        )

        assert 'HIGH' in entered
        assert 'MED' in entered
        assert 'LOW' not in entered    # 3rd slot consumed by HELD

    # ── Already-held symbol excluded before scoring ───────────────────────────

    def test_already_held_symbol_not_re_entered(self):
        """
        If 'AAPL' is already in state, the scanner returning 'AAPL' must not
        cause get_technical_context to be called for it (skip happens before scoring).
        """
        ib = self._make_ib_with_positions(['AAPL'])
        engine = _make_engine(ib)
        engine.state = {'AAPL': {'price': 100, 'time': datetime.now().isoformat()}}

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['AAPL']), \
             patch.object(engine, 'get_technical_context') as mock_ctx, \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, 'manage_position_exits'), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        mock_ctx.assert_not_called()

    def test_no_entry_when_get_technical_context_returns_none(self):
        """If data is unavailable (None), the symbol must be silently skipped."""
        ib     = _mock_ib()
        engine = _make_engine(ib)

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['GHOST']), \
             patch.object(engine, 'get_technical_context', return_value=None), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert 'GHOST' not in engine.state

    def test_no_entry_outside_session_window(self):
        """Outside 09:45–15:30 ET no entry must occur even if signal passes."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ctx    = _ctx(price=101.0, orb=100.0)

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 16, 0))  # 16:00 — after close

        with patch.object(engine, 'get_institutional_scan', return_value=['SYM']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert 'SYM' not in engine.state

    def test_insufficient_settled_cash_blocks_entry(self):
        """If settled cash < order cost, the entry must be skipped."""
        ib = _mock_ib()
        # Override settled cash to near-zero
        nl = MagicMock(); nl.tag = 'NetLiquidation'; nl.value = '1400.0'
        af = MagicMock(); af.tag = 'AvailableFunds'; af.value = '1.0'   # only $1
        ib.accountSummary.return_value = [nl, af]

        engine = _make_engine(ib)
        ctx    = _ctx(price=100.0, atr=3.0)

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['CASH']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert ib.placeOrder.call_count == 0, "No order must be placed with insufficient cash"
        assert 'CASH' not in engine.state

    def test_cash_check_uses_total_order_cost_not_single_share_price(self):
        """
        With dynamic cash buckets, settled cash below the minimum bucket blocks
        entries before sizing so the cash account cannot create dust trades.
        """
        ib = _mock_ib()
        nl = MagicMock(); nl.tag = 'NetLiquidation'; nl.value = '1400.0'
        sc = MagicMock(); sc.tag = 'SettledCash';    sc.value = '100.0'
        ib.accountSummary.return_value = [nl, sc]

        engine = _make_engine(ib)
        ctx = _ctx(price=50.0, atr=2.0, orb=46.0, ma50=48.0, ma200=40.0,
                   rsi=62.0, rsi_prev=57.0)

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['CHEAP']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert ib.placeOrder.call_count == 0, \
            "Order must be blocked: settled covers 1 share but not the full order"
        assert 'CHEAP' not in engine.state

    def test_cash_check_passes_when_settled_covers_full_order(self):
        """settled >= qty*price → order proceeds."""
        ib = _mock_ib()
        # With a 95% deployable-cash buffer, settled=$530 creates one cash-qualified slot.
        nl = MagicMock(); nl.tag = 'NetLiquidation'; nl.value = '1400.0'
        sc = MagicMock(); sc.tag = 'SettledCash';    sc.value = '530.0'
        ib.accountSummary.return_value = [nl, sc]

        engine = _make_engine(ib)
        # orb=46: price/orb=1.087 < 1.10 gap limit ✓ (isolates cash-check logic)
        ctx = _ctx(price=50.0, atr=2.0, orb=46.0, ma50=48.0, ma200=40.0,
                   rsi=62.0, rsi_prev=57.0)

        mock_trade = MagicMock()
        mock_trade.order.orderId             = 1
        mock_trade.orderStatus.status        = 'Filled'
        mock_trade.orderStatus.avgFillPrice  = 50.0
        mock_trade.fills = [_mock_fill(1.0)]
        ib.placeOrder.return_value = mock_trade

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['CHEAP']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert 'CHEAP' in engine.state, "Order must proceed when settled covers full order cost"

    def test_stale_scan_price_without_valid_reprice_blocks_entry(self):
        """A stale scan price must not be used when the live reprice is unavailable."""
        from src.config import ENTRY_REPRICE_MAX_AGE_SEC

        ib     = _mock_ib()
        engine = _make_engine(ib)

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))
        ctx = _ctx(price=101.0, orb=100.0, ma50=95.0, ma200=85.0)
        ctx['price_fetched_at'] = fake_now - timedelta(seconds=ENTRY_REPRICE_MAX_AGE_SEC + 1)

        vix_ticker = MagicMock()
        vix_ticker.marketPrice.return_value = 20.0

        bad_ticker = MagicMock()
        bad_ticker.marketPrice.return_value = float('nan')
        bad_ticker.last = float('nan')
        bad_ticker.close = float('nan')

        ib.reqTickers.side_effect = [[vix_ticker], [bad_ticker]]

        with patch.object(engine, 'get_institutional_scan', return_value=['STALE']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert ib.placeOrder.call_count == 0
        assert 'STALE' not in engine.state

    def test_child_stop_preflight_failure_cancels_untransmitted_parent(self):
        """A BUY parent must not go live unless the protective TRAIL child passes preflight."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ctx    = _ctx(price=101.0, orb=100.0, ma50=95.0, ma200=85.0)

        buy_ok = MagicMock()
        buy_ok.warningText = ''
        stop_reject = MagicMock()
        stop_reject.warningText = 'child stop rejected'
        ib.whatIfOrder.side_effect = [buy_ok, stop_reject]

        parent_trade = MagicMock()
        parent_trade.order.orderId = 1
        ib.placeOrder.return_value = parent_trade

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['NOSTOP']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert ib.placeOrder.call_count == 1
        ib.cancelOrder.assert_called_once_with(parent_trade.order)
        assert 'NOSTOP' not in engine.state

    def test_fallthrough_to_next_candidate_when_order_cancelled(self):
        """
        When rank-1's bracket is cancelled by IB, the engine must fall
        through and attempt rank-2 rather than giving up on the slot.
        HIGH has higher rvol so it is ranked first; LOW has lower rvol and is ranked second.
        """
        ctx_hi = _ctx(price=101.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                      ma50=95.0, ma200=85.0, rvol=5.0)
        ctx_lo = _ctx(price=104.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                      ma50=95.0, ma200=85.0, rvol=2.5)

        # $1,500 equity gives three dynamic slots; two held positions leaves one.
        ib = self._make_ib_with_positions(['HELD0', 'HELD1'], equity=1500.0)
        engine = _make_engine(ib)
        engine.state = {
            'HELD0': {'price': 100, 'time': datetime.now().isoformat(),
                      'qty': 1, 'stop_loss': 90, 'volume': 0, 'score': 50},
            'HELD1': {'price': 100, 'time': datetime.now().isoformat(),
                      'qty': 1, 'stop_loss': 90, 'volume': 0, 'score': 50},
        }

        cancelled = MagicMock()
        cancelled.orderStatus.status       = 'Cancelled'
        cancelled.orderStatus.filled       = 0
        cancelled.orderStatus.avgFillPrice = 0.0
        cancelled.fills = []

        filled = MagicMock()
        filled.order.orderId             = 1
        filled.orderStatus.status        = 'Filled'
        filled.orderStatus.filled        = 2.0
        filled.orderStatus.avgFillPrice  = 104.0
        filled.fills = [_mock_fill(1.0)]

        stop_mock = MagicMock()
        stop_mock.orderStatus.status = 'Submitted'

        # Sequential pattern: cancelled BUY → 1 call (no TRAIL placed on miss);
        # filled BUY → 2nd call, then standalone TRAIL → 3rd call.
        ib.placeOrder.side_effect = [
            cancelled,   # HIGH — BUY cancelled; no TRAIL submitted
            filled,      # LOW  — BUY filled
            stop_mock,   # LOW  — standalone TRAIL placed after fill
        ]

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))
        ctx_map  = {'HIGH': ctx_hi, 'LOW': ctx_lo}

        with patch.object(engine, 'get_institutional_scan', return_value=['HIGH', 'LOW']), \
             patch.object(engine, 'get_technical_context', side_effect=lambda s: ctx_map.get(s)), \
             patch.object(engine, '_confirm_protective_stop', return_value=True), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, 'manage_position_exits'), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert 'HIGH' not in engine.state, "Cancelled rank-1 must NOT be written to state"
        assert 'LOW'  in  engine.state,    "Rank-2 must be entered after rank-1 is cancelled"

    def test_exit_checks_run_between_multiple_entries(self):
        """
        A broad scan can place more than one entry in the same cycle. After the
        first fill, existing-position exit checks must run again before the next
        order so software-managed exits are not starved by entry processing.
        """
        ctx_a = _ctx(price=101.0, orb=100.0, rsi=70.0, rsi_prev=62.0,
                     ma50=95.0, ma200=85.0, rvol=5.0)
        ctx_b = _ctx(price=104.0, orb=103.0, rsi=68.0, rsi_prev=61.0,
                     ma50=96.0, ma200=84.0, rvol=4.0)

        ib = self._make_ib_with_positions([], equity=1400.0, settled=5000.0)
        engine = _make_engine(ib)
        tz_ny = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        buy_a = MagicMock()
        buy_a.order.orderId = 11
        buy_a.orderStatus.status = 'Filled'
        buy_a.orderStatus.filled = 2.0
        buy_a.orderStatus.avgFillPrice = 101.0
        buy_a.fills = [_mock_fill(1.0)]

        stop_a = MagicMock()
        stop_a.order.orderId = 12
        stop_a.orderStatus.status = 'Submitted'

        buy_b = MagicMock()
        buy_b.order.orderId = 13
        buy_b.orderStatus.status = 'Filled'
        buy_b.orderStatus.filled = 2.0
        buy_b.orderStatus.avgFillPrice = 104.0
        buy_b.fills = [_mock_fill(1.0)]

        stop_b = MagicMock()
        stop_b.order.orderId = 14
        stop_b.orderStatus.status = 'Submitted'

        ib.placeOrder.side_effect = [buy_a, stop_a, buy_b, stop_b]

        with patch.object(engine, 'get_institutional_scan', return_value=['ALPHA', 'BETA']), \
             patch.object(engine, 'get_technical_context', side_effect=lambda sym: {'ALPHA': ctx_a, 'BETA': ctx_b}[sym]), \
             patch.object(engine, '_confirm_protective_stop', return_value=True), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, 'manage_position_exits') as mock_exits, \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine.run_cycle()

        assert {'ALPHA', 'BETA'} <= set(engine.state)
        assert mock_exits.call_count == 2


class TestDailyScanSkip:
    """
    Same-day scan skip is an IBKR pacing guard for stable rejections only.

    It must cache failures that cannot reasonably improve intraday, such as
    20-day dollar volume below the active threshold or unusable daily history.
    It must not cache live/intraday failures such as ORB, spread, gap, day
    location, open gain, or ATR percentage.
    """

    _TZ_NY = pytz.timezone('US/Eastern')

    def _run_cycle_at(self, engine, fake_now):
        with patch.object(engine, 'manage_position_exits'), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine.run_cycle()

    def test_low_dollar_volume_is_cached_for_rest_of_day(self):
        from src.config import SCAN_MIN_DOLLAR_VOL

        ib = _mock_ib()
        engine = _make_engine(ib)
        ctx = _ctx(
            price=101.0,
            orb=100.0,
            dollar_vol=SCAN_MIN_DOLLAR_VOL - 1,
        )
        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['LOWVOL']), \
             patch.object(engine, 'get_technical_context', return_value=ctx):
            self._run_cycle_at(engine, fake_now)

        assert 'LOWVOL' in engine._daily_scan_skip
        assert 'DolVol20d' in engine._daily_scan_skip['LOWVOL']
        assert 'LOWVOL' not in engine.state

    def test_cached_symbol_skips_technical_context_fetch(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._day_start_date = '2024-06-05'
        engine._day_start_equity = 1400.0
        engine._daily_scan_skip = {'LOWVOL': 'DolVol20d $99M < threshold $100M'}

        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['LOWVOL']), \
             patch.object(engine, 'get_technical_context') as mock_ctx:
            self._run_cycle_at(engine, fake_now)

        mock_ctx.assert_not_called()

    def test_daily_scan_skip_clears_on_new_trading_day(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._day_start_date = '2024-06-04'
        engine._day_start_equity = 1400.0
        engine._daily_scan_skip = {'LOWVOL': 'DolVol20d $99M < threshold $100M'}

        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=[]):
            self._run_cycle_at(engine, fake_now)

        assert engine._daily_scan_skip == {}

    def test_daily_bar_cache_clears_on_new_trading_day(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._day_start_date = '2024-06-04'
        engine._day_start_equity = 1400.0
        engine._daily_scan_skip = {'LOWVOL': 'DolVol20d $99M < threshold $100M'}
        engine._bar_cache = {'OLD': {'bars_daily': ['cached'], 'orb_high': 101.0}}

        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=[]):
            self._run_cycle_at(engine, fake_now)

        assert engine._daily_scan_skip == {}
        assert engine._bar_cache == {}

    def test_daily_bar_cache_is_kept_on_same_trading_day(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._day_start_date = '2024-06-05'
        engine._day_start_equity = 1400.0
        cached = {'OLD': {'bars_daily': ['cached'], 'orb_high': 101.0}}
        engine._bar_cache = dict(cached)

        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=[]):
            self._run_cycle_at(engine, fake_now)

        assert engine._bar_cache == cached

    def test_dynamic_orb_failure_is_not_cached(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        ctx = _ctx(price=99.0, orb=100.0)
        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['WAITORB']), \
             patch.object(engine, 'get_technical_context', return_value=ctx):
            self._run_cycle_at(engine, fake_now)

        assert 'WAITORB' not in engine._daily_scan_skip
        assert 'WAITORB' not in engine.state

    def test_dynamic_spread_failure_is_not_cached(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        ctx = _ctx(price=101.0, orb=100.0, spread_pct=0.01)
        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['WIDESPREAD']), \
             patch.object(engine, 'get_technical_context', return_value=ctx):
            self._run_cycle_at(engine, fake_now)

        assert 'WIDESPREAD' not in engine._daily_scan_skip
        assert 'WIDESPREAD' not in engine.state

    def test_insufficient_daily_history_is_cached(self):
        from src.config import MIN_CANDLES

        ib = _mock_ib()
        engine = _make_engine(ib)

        orb_bar = MagicMock()
        orb_bar.high = 100.0
        orb_bar.open = 99.0
        daily_bars = [MagicMock()] * (MIN_CANDLES - 1)
        ib.reqHistoricalData.side_effect = [[orb_bar], daily_bars]
        too_short = pd.DataFrame({'close': np.ones(MIN_CANDLES - 1)})

        with patch('src.engine.util.df', return_value=too_short):
            assert engine.get_technical_context('NEWIPO') is None

        assert 'NEWIPO' in engine._daily_scan_skip
        assert 'insufficient daily history' in engine._daily_scan_skip['NEWIPO']


class TestRuntimeProtectiveStopAudit:
    """Runtime cycles must not rely only on startup to protect open positions."""

    _TZ_NY = pytz.timezone('US/Eastern')

    def test_post_open_audit_waits_until_checkpoint_time(self):
        from src.config import POST_OPEN_AUDIT_TIME

        engine = _make_engine(_mock_ib())
        engine._last_audit_date = '2024-06-05'
        engine.state = {
            'AAPL': {
                'price': 100.0,
                'qty': 1.0,
                'stop_loss': 94.0,
                'stop_dist': 6.0,
                'time': '2024-06-05T10:00:00-04:00',
            }
        }

        h, m = POST_OPEN_AUDIT_TIME
        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, h, m - 1))

        with patch.object(engine, '_sync_positions_from_ibkr') as mock_sync, \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch.object(engine, '_write_dashboard_data'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine._maybe_post_open_stop_audit()

        mock_sync.assert_not_called()
        mock_audit.assert_not_called()

    def test_post_open_audit_runs_once_after_checkpoint_even_if_daily_audit_ran(self):
        from src.config import POST_OPEN_AUDIT_TIME

        engine = _make_engine(_mock_ib())
        engine._last_audit_date = '2024-06-05'
        engine.state = {
            'AAPL': {
                'price': 100.0,
                'qty': 1.0,
                'stop_loss': 94.0,
                'stop_dist': 6.0,
                'time': '2024-06-05T10:00:00-04:00',
            }
        }

        h, m = POST_OPEN_AUDIT_TIME
        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, h, m))

        with patch.object(engine, '_sync_positions_from_ibkr') as mock_sync, \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch.object(engine, '_update_position_prices') as mock_prices, \
             patch.object(engine, '_write_dashboard_data') as mock_dashboard, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine._maybe_post_open_stop_audit()

        mock_sync.assert_called_once()
        mock_audit.assert_called_once()
        mock_prices.assert_called_once()
        mock_dashboard.assert_called_once_with(connected=True)
        assert engine._last_post_open_audit_date == '2024-06-05'

    def test_post_open_audit_skips_duplicate_same_day_checkpoint(self):
        from src.config import POST_OPEN_AUDIT_TIME

        engine = _make_engine(_mock_ib())
        engine._last_post_open_audit_date = '2024-06-05'
        engine.state = {
            'AAPL': {
                'price': 100.0,
                'qty': 1.0,
                'stop_loss': 94.0,
                'stop_dist': 6.0,
                'time': '2024-06-05T10:00:00-04:00',
            }
        }

        h, m = POST_OPEN_AUDIT_TIME
        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, h, m + 5))

        with patch.object(engine, '_sync_positions_from_ibkr') as mock_sync, \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch.object(engine, '_write_dashboard_data'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine._maybe_post_open_stop_audit()

        mock_sync.assert_not_called()
        mock_audit.assert_not_called()

    def test_unprotected_state_forces_audit_even_if_daily_audit_already_ran(self):
        engine = _make_engine(_mock_ib())
        engine._last_audit_date = '2024-06-05'
        engine.state = {
            'AAPL': {
                'price': 100.0,
                'qty': 1.0,
                'stop_loss': 0.0,
                'stop_dist': 0.0,
                'time': '2024-06-05T10:00:00-04:00',
            }
        }

        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine._maybe_audit_stop_orders()

        mock_audit.assert_called_once()

    def test_protected_state_skips_duplicate_same_day_audit(self):
        engine = _make_engine(_mock_ib())
        engine._last_audit_date = '2024-06-05'
        engine.state = {
            'AAPL': {
                'price': 100.0,
                'qty': 1.0,
                'stop_loss': 94.0,
                'stop_dist': 6.0,
                'time': '2024-06-05T10:00:00-04:00',
            }
        }

        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine._maybe_audit_stop_orders()

        mock_audit.assert_not_called()

    def test_run_cycle_audits_stops_before_account_data_gate(self):
        from src.engine import AccountDataUnavailable

        ib = _mock_ib()
        engine = _make_engine(ib)
        engine.state = {
            'AAPL': {
                'price': 100.0,
                'qty': 1.0,
                'stop_loss': 0.0,
                'stop_dist': 0.0,
                'time': '2024-06-05T10:00:00-04:00',
            }
        }

        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch.object(engine, '_get_account_values',
                          side_effect=AccountDataUnavailable('no account data')), \
             patch.object(engine, 'manage_position_exits'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine.run_cycle()

        mock_audit.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# 8. PORTFOLIO RISK GATES — correlation fail-closed behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioRiskGates:
    def test_correlation_gate_fails_closed_when_book_history_missing(self):
        ib = _mock_ib()
        ib.reqHistoricalData.return_value = []
        engine = _make_engine(ib)
        engine.state = {'HELD': {'price': 100.0, 'qty': 1.0}}

        idx = pd.date_range("2024-01-01", periods=90, freq='B')
        df_daily = pd.DataFrame({'close': np.linspace(100.0, 130.0, len(idx))}, index=idx)

        assert engine._compute_book_correlation('NEW', df_daily) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. EXIT ORDERS — velocity exits, liquidation
# ─────────────────────────────────────────────────────────────────────────────

class TestExitOrders:
    """
    Verify velocity exit (manage_position_exits → liquidate) behaviour:
    - MarketOrder('SELL', position) placed with exact qty reported by IBKR
    - Open symbol orders are cancelled before the market sell
    - Cash-account exits cancel protective SELLs first to avoid oversell rejection
    - MarketOrder TIF is explicit DAY so IBKR presets cannot override it to GTC
    - Exit fires only when stagnant; profitable positions are kept
    """

    def _make_position(self, symbol, qty):
        pos = MagicMock()
        pos.contract.symbol = symbol
        pos.position        = qty
        return pos

    def _make_state_entry(self, price=100.0, qty=1.0, hours_ago=0, entry_time=None):
        tz_ny = pytz.timezone('US/Eastern')
        if entry_time is None:
            entry_time = (datetime.now(tz_ny) - timedelta(hours=hours_ago)).isoformat()
        return {'price': price, 'time': entry_time, 'qty': qty,
                'stop_loss': price * 0.94,
                'volume': 0, 'score': 50}

    def _et(self, year, month, day, hour=10, minute=30):
        return pytz.timezone('US/Eastern').localize(
            datetime(year, month, day, hour, minute)
        )

    def _run_velocity_check(self, engine, now=None):
        now = now or self._et(2024, 6, 5, 15, 50)
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine.manage_position_exits()

    # ── liquidate() ──────────────────────────────────────────────────────────

    def test_liquidate_places_market_sell_with_position_qty(self):
        """Market sell must use exactly the qty reported by IBKR (p.position)."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        qty    = 2.5
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('SYM', qty)]
        engine.state = {'SYM': self._make_state_entry(qty=qty)}

        engine.liquidate('SYM')

        assert ib.placeOrder.call_count == 1
        order = ib.placeOrder.call_args[0][1]
        assert order.orderType        == 'MKT'
        assert order.action           == 'SELL'
        assert order.totalQuantity    == pytest.approx(qty, abs=0.0001)

    def test_liquidate_uses_exact_ibkr_position_qty_for_sell(self):
        """
        liquidate() reads qty from p.position (IBKR source of truth), not from
        self.state.  This guards against discrepancies between state and IBKR
        (e.g. partial fills, stock splits, or manually adjusted positions).
        Sells must always mirror exactly what IBKR reports.
        """
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ibkr_qty = 5        # IBKR reports 5 shares
        state_qty = 4       # state has different value — sell must use IBKR's
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('PRICEY', ibkr_qty)]
        engine.state = {'PRICEY': self._make_state_entry(price=100.0, qty=state_qty)}

        engine.liquidate('PRICEY')

        order = ib.placeOrder.call_args[0][1]
        assert order.totalQuantity == pytest.approx(ibkr_qty, abs=0.0001), \
            "Market sell must use IBKR position qty, not state qty"

    def test_liquidate_cancels_non_trail_orders_before_sell(self):
        """Non-TRAIL open orders are cancelled before the market sell is placed."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        trade  = MagicMock()
        trade.contract.symbol = 'SYM'
        trade.order.orderType = 'LMT'
        ib.openTrades.return_value = [trade]
        ib.positions.return_value  = [self._make_position('SYM', 1.0)]
        engine.state = {'SYM': self._make_state_entry()}

        cancel_count_at_place = []
        def capture(*args):
            cancel_count_at_place.append(ib.cancelOrder.call_count)
            return MagicMock()
        ib.placeOrder.side_effect = capture

        engine.liquidate('SYM')

        assert ib.cancelOrder.called, "cancelOrder must be called for open bracket order"
        assert cancel_count_at_place[0] >= 1, "cancelOrder must be called BEFORE placeOrder"

    def test_liquidate_cancels_trail_stop_before_cash_account_market_sell(self):
        """Cash accounts require protective SELL cancellation before market exit."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        trail  = MagicMock()
        trail.contract.symbol = 'SYM'
        trail.order.action = 'SELL'
        trail.order.orderType = 'TRAIL'
        ib.openTrades.side_effect = [[trail], []]
        ib.positions.return_value  = [self._make_position('SYM', 1.0)]
        engine.state = {'SYM': self._make_state_entry()}

        engine.liquidate('SYM')

        ib.cancelOrder.assert_called_once_with(trail.order)
        assert engine.state['SYM']['pending_exit'] is True

    def test_liquidate_marks_symbol_pending_until_sync_confirms_flat(self):
        """State is retained as pending_exit after a filled sell; sync later removes it."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('SYM', 1.0)]
        engine.state = {'SYM': self._make_state_entry()}

        engine.liquidate('SYM')

        assert engine.state['SYM']['pending_exit'] is True

        ib.positions.return_value = []
        engine._sync_positions_from_ibkr()
        engine._sync_positions_from_ibkr()

        assert 'SYM' not in engine.state

    def test_sync_cancels_leftover_trail_only_after_flat_confirmed(self):
        """After IBKR confirms flat twice, leftover SELL stops must be cancelled."""
        ib     = _mock_ib()
        engine = _make_engine(ib)

        trail = MagicMock()
        trail.contract.symbol = 'SYM'
        trail.order.action = 'SELL'
        trail.order.orderType = 'TRAIL'

        ib.positions.return_value = []
        ib.openTrades.return_value = [trail]
        engine.state = {'SYM': self._make_state_entry()}

        engine._sync_positions_from_ibkr()
        ib.cancelOrder.assert_not_called()
        assert 'SYM' in engine.state

        engine._sync_positions_from_ibkr()

        ib.cancelOrder.assert_called_once_with(trail.order)
        assert 'SYM' not in engine.state

    def test_liquidate_retains_state_when_market_sell_rejected(self):
        """Rejected exit orders must not erase local state; retry on next cycle."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('SYM', 1.0)]

        rejected = MagicMock()
        rejected.orderStatus.status = 'Inactive'
        rejected.orderStatus.filled = 0
        ib.placeOrder.return_value = rejected

        engine.state = {'SYM': self._make_state_entry()}
        with patch.object(engine, '_audit_stop_orders') as mock_audit:
            engine.liquidate('SYM')

        assert 'SYM' in engine.state
        assert 'pending_exit' not in engine.state['SYM']
        mock_audit.assert_called_once()

    def test_liquidate_retains_state_when_market_sell_placement_raises(self):
        """IB API placement exceptions must clear pending_exit so exits can retry."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('SYM', 1.0)]
        ib.placeOrder.side_effect  = RuntimeError("connection dropped")

        engine.state = {'SYM': self._make_state_entry()}

        with patch.object(engine, '_alert') as mock_alert, \
             patch.object(engine, '_audit_stop_orders') as mock_audit:
            engine.liquidate('SYM')

        assert 'SYM' in engine.state
        assert 'pending_exit' not in engine.state['SYM']
        mock_alert.assert_called_once()
        mock_audit.assert_called_once()

    def test_liquidate_uses_smart_routed_sell_contract(self):
        """Direct-routed native contracts can be blocked by IBKR; exits must force SMART."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        pos = self._make_position('SYM', 1.0)
        pos.contract.exchange = 'NASDAQ'
        ib.openTrades.return_value = []
        ib.positions.return_value  = [pos]
        engine.state = {'SYM': self._make_state_entry()}

        engine.liquidate('SYM')

        sell_contract = ib.placeOrder.call_args[0][0]
        assert sell_contract.exchange == 'SMART'

    def test_liquidate_market_sell_is_immediate(self):
        """Liquidation sell must not be delayed by entry-window goodAfterTime."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('SYM', 1.0)]
        engine.state = {'SYM': self._make_state_entry()}

        engine.liquidate('SYM')

        order = ib.placeOrder.call_args[0][1]
        assert order.orderType == 'MKT'
        assert order.tif == 'DAY'
        assert order.goodAfterTime == ''

    # ── manage_position_exits() ──────────────────────────────────────────────

    def test_velocity_exit_triggers_when_stagnant_after_hold_window(self):
        """Position held for configured trading sessions with weak profit → liquidated."""
        from src.config import PROFIT_MIN_THRESHOLD
        ib     = _mock_ib()
        engine = _make_engine(ib)

        entry_price   = 100.0
        stagnant_price = entry_price * (1 + PROFIT_MIN_THRESHOLD - 0.005)
        ib.reqTickers.return_value = [_mock_price_ticker(stagnant_price)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('SLOW', 5.0)]
        engine.state = {'SLOW': self._make_state_entry(price=entry_price,
                                                        entry_time=self._et(2024, 6, 3).isoformat())}

        self._run_velocity_check(engine)

        assert engine.state['SLOW']['pending_exit'] is True, "Stagnant position must be marked pending exit"
        assert ib.placeOrder.called, "Market sell must be issued"

    def test_velocity_exit_waits_until_configured_eod_time(self):
        """Held stagnant positions must not be velocity-sold before 15:50 ET."""
        from src.config import PROFIT_MIN_THRESHOLD
        ib     = _mock_ib()
        engine = _make_engine(ib)

        entry_price = 100.0
        stagnant_price = entry_price * (1 + PROFIT_MIN_THRESHOLD - 0.005)
        ib.reqTickers.return_value = [_mock_price_ticker(stagnant_price)]
        engine.state = {'SLOW': self._make_state_entry(
            price=entry_price,
            entry_time=self._et(2024, 6, 3).isoformat(),
        )}

        self._run_velocity_check(engine, now=self._et(2024, 6, 5, 15, 49))

        assert 'pending_exit' not in engine.state['SLOW']
        assert not ib.placeOrder.called

    def test_pending_exit_blocks_duplicate_sell_on_next_cycle(self):
        """A position with an in-flight sell must not submit another market sell."""
        from src.config import PROFIT_MIN_THRESHOLD
        ib     = _mock_ib()
        engine = _make_engine(ib)

        entry_price = 100.0
        stagnant_price = entry_price * (1 + PROFIT_MIN_THRESHOLD - 0.005)
        ib.reqTickers.return_value = [_mock_price_ticker(stagnant_price)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('SLOW', 5.0)]
        entry = self._make_state_entry(
            price=entry_price,
            entry_time=self._et(2024, 6, 3).isoformat(),
        )
        entry['pending_exit'] = True
        engine.state = {'SLOW': entry}

        self._run_velocity_check(engine)

        assert not ib.placeOrder.called
        assert engine.state['SLOW']['pending_exit'] is True

    def test_velocity_exit_does_not_trigger_when_profitable(self):
        """Position held through the window but profit ≥ threshold → kept."""
        from src.config import PROFIT_MIN_THRESHOLD
        ib     = _mock_ib()
        engine = _make_engine(ib)

        entry_price    = 100.0
        profit_price   = entry_price * (1 + PROFIT_MIN_THRESHOLD + 0.01)
        ib.reqTickers.return_value = [_mock_price_ticker(profit_price)]
        engine.state = {'WINNER': self._make_state_entry(price=entry_price,
                                                          entry_time=self._et(2024, 6, 3).isoformat())}

        self._run_velocity_check(engine)

        assert 'WINNER' in engine.state, "Profitable position must NOT be liquidated"
        assert not ib.placeOrder.called

    def test_velocity_exit_does_not_trigger_before_hold_window(self):
        """Position still within hold window must never be touched."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        engine.state = {'NEW': self._make_state_entry(
            entry_time=self._et(2024, 6, 5, 10, 0).isoformat()
        )}
        ib.reqTickers.return_value = [_mock_price_ticker(100.0)]

        self._run_velocity_check(engine)

        assert 'NEW' in engine.state
        assert not ib.placeOrder.called


# ─────────────────────────────────────────────────────────────────────────────
# 9. EDGE CASES — guards, NaN/zero inputs, boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """
    Covers every critical boundary and defensive guard identified in the audit:
    - Corrupt state file on load → empty state, no crash
    - ATR = 0 or NaN → skip entry, no malformed bracket
    - MA200 = 0 → no division by zero in scoring
    - ORB high = 0 → get_technical_context returns None
    - Entry price = 0 in velocity exit → skipped, no division by zero
    - VIX at threshold (=35) → entries allowed; above (>35) → blocked
    - Strict comparisons: price > ORB, RSI strictly rising
    - Friday dollar-volume threshold doubled
    - IBKR sync adds missing position and warns when avgCost missing
    - Indicator edge: RSI with all-gain period, flat bars
    """

    # ── State persistence ────────────────────────────────────────────────────

    def test_load_state_returns_empty_on_corrupt_json(self, tmp_path):
        """Corrupt STATE_FILE must not crash the engine — returns empty dict."""
        import src.engine as eng_mod
        state_path = tmp_path / "engine_state.json"
        state_path.write_text("{not valid json!!!")  # corrupted

        original = eng_mod.STATE_FILE
        eng_mod.STATE_FILE = str(state_path)
        try:
            engine = eng_mod.VelocityEngine.__new__(eng_mod.VelocityEngine)
            result = engine.load_state()
        finally:
            eng_mod.STATE_FILE = original

        assert result == {}, "Corrupt JSON must yield empty state, not crash"

    def test_load_state_returns_empty_when_file_missing(self, tmp_path):
        """Missing state file → empty state (fresh start)."""
        import src.engine as eng_mod
        original = eng_mod.STATE_FILE
        eng_mod.STATE_FILE = str(tmp_path / "nonexistent.json")
        try:
            engine = eng_mod.VelocityEngine.__new__(eng_mod.VelocityEngine)
            result = engine.load_state()
        finally:
            eng_mod.STATE_FILE = original
        assert result == {}

    # ── ATR guard in entry loop ───────────────────────────────────────────────

    def test_entry_skipped_when_atr_is_zero(self):
        """ATR=0 → stop_dist=0 → invalid bracket; engine must skip before placing."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ctx    = _ctx(price=100.0, atr=0.0)

        _run_entry_cycle(ib, engine, ctx)

        assert ib.placeOrder.call_count == 0, "No order must be placed when ATR=0"

    def test_entry_skipped_when_atr_is_nan(self):
        """ATR=NaN → skip before bracket; must not write NaN into state."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ctx    = _ctx(price=100.0, atr=float('nan'))

        _run_entry_cycle(ib, engine, ctx)

        assert ib.placeOrder.call_count == 0, "No order must be placed when ATR=NaN"
        assert 'TSLA' not in engine.state

    # ── Scoring: MA200 = 0 guard ─────────────────────────────────────────────

    def test_score_candidate_ma200_zero_does_not_raise(self):
        """ma200=0 must not raise ZeroDivisionError; trend component floors to 0."""
        from src.engine import VelocityEngine
        engine = VelocityEngine.__new__(VelocityEngine)
        ctx    = _ctx(price=110.0, orb=100.0, rsi=65.0, rsi_prev=60.0,
                      ma50=105.0, ma200=0.0)
        score = engine._score_candidate(ctx)
        assert isinstance(score, float), "Score must be a float even with ma200=0"
        assert 0.0 <= score <= 100.0

    def test_score_candidate_trend_is_zero_when_ma200_is_zero(self):
        """Trend component must return 0 (not NaN or inf) when MA200=0."""
        from src.engine import VelocityEngine
        engine = VelocityEngine.__new__(VelocityEngine)
        ctx_zero_ma = _ctx(price=110.0, orb=100.0, rsi=65.0, rsi_prev=60.0,
                           ma50=105.0, ma200=0.0)
        ctx_normal  = _ctx(price=110.0, orb=100.0, rsi=65.0, rsi_prev=60.0,
                           ma50=105.0, ma200=105.0)   # equal MAs → trend_sep=0 → trend=0
        score_zero  = engine._score_candidate(ctx_zero_ma)
        score_flat  = engine._score_candidate(ctx_normal)
        assert score_zero == score_flat, "MA200=0 must give same trend=0 as equal MAs"

    # ── VIX threshold boundary ────────────────────────────────────────────────

    def _run_cycle_with_vix(self, vix_val):
        """Helper: run a full cycle with given VIX and one passing signal."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ctx    = _ctx(price=101.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                      ma50=95.0, ma200=85.0)

        vix_ticker = MagicMock()
        vix_ticker.marketPrice.return_value = vix_val
        vix_ticker.close = vix_val
        ib.reqTickers.return_value = [vix_ticker]

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        mock_trade = MagicMock()
        mock_trade.order.orderId             = 1
        mock_trade.orderStatus.status        = 'Filled'
        mock_trade.orderStatus.avgFillPrice  = 101.0
        mock_trade.fills = [_mock_fill(1.0)]
        ib.placeOrder.return_value = mock_trade

        with patch.object(engine, 'get_institutional_scan', return_value=['SYM']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        return engine

    def test_vix_at_threshold_allows_entries(self):
        """VIX=35 (== threshold) must NOT block entries (rule: VIX > 35 blocks)."""
        from src.config import VIX_THRESHOLD
        engine = self._run_cycle_with_vix(float(VIX_THRESHOLD))
        assert 'SYM' in engine.state, f"VIX={VIX_THRESHOLD} must not block entries"

    def test_vix_above_threshold_blocks_entries(self):
        """VIX=35.01 (> threshold) must block all new entries."""
        from src.config import VIX_THRESHOLD
        engine = self._run_cycle_with_vix(VIX_THRESHOLD + 0.01)
        assert 'SYM' not in engine.state, "VIX above threshold must block entries"

    def test_vix_request_uses_delayed_type_then_restores_realtime(self):
        """VIX may use delayed data, but stock scanning must be restored to real-time mode."""
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._vix_contract = MagicMock()
        ib.reqTickers.return_value = [_mock_price_ticker(20.0)]

        engine._request_vix_tickers()

        from src.config import MARKET_DATA_TYPE, VIX_MARKET_DATA_TYPE
        if VIX_MARKET_DATA_TYPE != MARKET_DATA_TYPE:
            ib.reqMarketDataType.assert_has_calls([
                call(VIX_MARKET_DATA_TYPE),
                call(MARKET_DATA_TYPE),
            ])

    def test_vix_price_uses_historical_fallback_with_extended_non_rth_window(self):
        """If ticker data is empty/invalid, VIX fallback must use 5 D and useRTH=False."""
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._vix_contract = MagicMock()

        bad_ticker = MagicMock()
        bad_ticker.marketPrice.return_value = float('nan')
        bad_ticker.close = float('nan')
        hist_bar = MagicMock()
        hist_bar.close = 22.5
        ib.reqTickers.return_value = [bad_ticker]
        ib.reqHistoricalData.return_value = [hist_bar]

        assert engine._fetch_vix_price() == pytest.approx(22.5)
        args = ib.reqHistoricalData.call_args[0]
        assert args[2] == '5 D'
        assert args[5] is False

    def test_vix_ticker_exception_uses_historical_fallback(self):
        """A transient VIX ticker exception must not bypass the fallback path."""
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._vix_contract = MagicMock()

        hist_bar = MagicMock()
        hist_bar.close = 21.75
        ib.reqTickers.side_effect = RuntimeError("ticker farm unavailable")
        ib.reqHistoricalData.return_value = [hist_bar]

        assert engine._fetch_vix_price() == pytest.approx(21.75)

    def test_expired_stale_vix_cache_does_not_authorize_entries(self):
        """If fresh ticker and fallback both fail after TTL, VIX must fail closed."""
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._vix_contract = MagicMock()
        engine._last_vix = 20.0
        engine._last_vix_ts = 1.0

        bad_ticker = MagicMock()
        bad_ticker.marketPrice.return_value = float('nan')
        bad_ticker.close = float('nan')
        bad_ticker.last = float('nan')
        bad_ticker.prevClose = float('nan')
        ib.reqTickers.return_value = [bad_ticker]
        ib.reqHistoricalData.return_value = []

        assert engine._fetch_vix_price() is None

    def test_vix_failure_starts_retry_cooldown(self):
        """Repeated HMDS failures must back off instead of hammering IBKR every minute."""
        from src.config import VIX_FAILURE_COOLDOWN_BASE_SEC

        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._vix_contract = MagicMock()

        bad_ticker = MagicMock()
        bad_ticker.marketPrice.return_value = float('nan')
        bad_ticker.close = float('nan')
        bad_ticker.last = float('nan')
        bad_ticker.prevClose = float('nan')
        ib.reqTickers.return_value = [bad_ticker]
        ib.reqHistoricalData.return_value = []

        with patch('src.engine.time.time', return_value=1_000.0):
            assert engine._fetch_vix_price() is None

        assert engine._vix_failure_count == 1
        assert engine._next_vix_retry_ts == pytest.approx(
            1_000.0 + VIX_FAILURE_COOLDOWN_BASE_SEC
        )

    def test_vix_retry_cooldown_suppresses_ib_requests(self):
        """While VIX is cooling down, no ticker or historical request should be sent."""
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._vix_contract = MagicMock()
        engine._next_vix_retry_ts = 2_000.0

        with patch('src.engine.time.time', return_value=1_500.0):
            assert engine._fetch_vix_price() is None

        ib.reqTickers.assert_not_called()
        ib.reqHistoricalData.assert_not_called()

    def test_vix_success_resets_retry_cooldown(self):
        """A good fresh VIX read clears prior cooldown state."""
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._vix_contract = MagicMock()
        engine._vix_failure_count = 3
        engine._next_vix_retry_ts = 0.0
        engine._last_vix_failure_ts = 900.0

        ticker = _mock_price_ticker(19.5)
        ib.reqTickers.return_value = [ticker]

        with patch('src.engine.time.time', return_value=1_000.0):
            assert engine._fetch_vix_price() == pytest.approx(19.5)

        assert engine._vix_failure_count == 0
        assert engine._next_vix_retry_ts == 0.0
        assert engine._last_vix_source == "ticker"

    # ── Strict comparison boundaries ─────────────────────────────────────────

    def test_price_exactly_at_orb_high_does_not_enter(self):
        """price == orb_h fails c_orb (requires strict >). No entry."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ctx    = _ctx(price=100.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                      ma50=95.0, ma200=85.0)

        _run_entry_cycle(ib, engine, ctx)

        assert ib.placeOrder.call_count == 0, "price==orb_h must not trigger entry"

    def test_rsi_equal_to_prev_does_not_enter(self):
        """rsi == rsi_prev fails the 8096 minimum RSI-delta gate. No entry."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ctx    = _ctx(price=101.0, orb=100.0, rsi=65.0, rsi_prev=65.0,
                      ma50=95.0, ma200=85.0)

        _run_entry_cycle(ib, engine, ctx)

        assert ib.placeOrder.call_count == 0, "flat RSI (equal to prev) must not trigger entry"

    def test_low_day_range_location_does_not_enter(self):
        """Current price in lower half of today's range fails the day-location rule."""
        from src.config import DAY_RANGE_LOCATION_MIN
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ctx    = _ctx(price=101.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                      ma50=95.0, ma200=85.0,
                      day_range_location=DAY_RANGE_LOCATION_MIN - 0.01)

        _run_entry_cycle(ib, engine, ctx)

        assert ib.placeOrder.call_count == 0, "low day-range location must not trigger entry"

    def test_low_intraday_gain_does_not_enter(self):
        """Price not sufficiently above today's open fails the intraday-gain rule."""
        from src.config import INTRADAY_GAIN_MIN
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ctx    = _ctx(price=101.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                      ma50=95.0, ma200=85.0,
                      intraday_gain=INTRADAY_GAIN_MIN - 0.001)

        _run_entry_cycle(ib, engine, ctx)

        assert ib.placeOrder.call_count == 0, "weak intraday gain must not trigger entry"

    def test_reprice_check_blocks_below_minimum_price(self):
        """Live reprice must still enforce the production minimum price."""
        from src.config import SCAN_MIN_PRICE

        engine = _make_engine(_mock_ib())
        ctx = _ctx(price=SCAN_MIN_PRICE - 0.50, orb=SCAN_MIN_PRICE - 1.00,
                   ma50=95.0, ma200=85.0)

        assert not engine._entry_price_is_still_valid(
            "LOWP", ctx, SCAN_MIN_PRICE - 0.50,
        )

    def test_reprice_check_uses_regime_specific_gap_cap(self):
        """Bear-mode reprice validation must keep the stricter opening-gap cap."""
        from src.config import BEAR_GAP_MAX_PCT

        engine = _make_engine(_mock_ib())
        ctx = _ctx(
            price=101.0,
            orb=100.0,
            ma50=95.0,
            ma200=85.0,
            day_open=100.0 * (1 + BEAR_GAP_MAX_PCT + 0.005),
        )
        ctx['high10'] = 101.0

        assert not engine._entry_price_is_still_valid(
            'TSLA', ctx, 104.5,
            gap_max_pct=BEAR_GAP_MAX_PCT,
        )

    def test_reprice_check_allows_current_extension_when_open_gap_passed(self):
        """Backtest-compatible gap check uses day open, not refreshed extension."""
        from src.config import BEAR_GAP_MAX_PCT

        engine = _make_engine(_mock_ib())
        ctx = _ctx(
            price=101.0,
            orb=100.0,
            ma50=95.0,
            ma200=85.0,
            day_open=100.0 * (1 + BEAR_GAP_MAX_PCT),
        )

        assert engine._entry_price_is_still_valid(
            'TSLA', ctx, 106.0,
            gap_max_pct=BEAR_GAP_MAX_PCT,
        )

    # ── Friday dollar-volume multiplier ───────────────────────────────────────

    def test_friday_dollar_volume_threshold_is_doubled(self):
        """On Fridays the dollar-volume gate must be 2× the normal threshold."""
        from src.config import SCAN_MIN_DOLLAR_VOL, VOL_MULT_FRIDAY

        ib     = _mock_ib()
        engine = _make_engine(ib)

        # Stock with dollar_vol exactly at 1× threshold — passes Mon–Thu, fails Fri
        marginal_vol = SCAN_MIN_DOLLAR_VOL   # exactly 1× → fails 2× gate
        ctx = _ctx(price=101.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                   ma50=95.0, ma200=85.0, dollar_vol=marginal_vol)

        tz_ny      = pytz.timezone('US/Eastern')
        friday_now = tz_ny.localize(datetime(2024, 6, 7, 10, 30))  # Friday

        mock_trade = MagicMock()
        mock_trade.orderStatus.status       = 'Filled'
        mock_trade.orderStatus.avgFillPrice = 101.0
        mock_trade.fills = [_mock_fill(1.0)]
        ib.placeOrder.return_value = mock_trade

        with patch.object(engine, 'get_institutional_scan', return_value=['SYM']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = friday_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert 'SYM' not in engine.state, \
            "Stock at 1× dollar-vol must be blocked on Friday (requires 2×)"

    def test_normal_day_dollar_volume_at_1x_passes(self):
        """On a non-Friday, 1× dollar-vol threshold is sufficient for entry."""
        from src.config import SCAN_MIN_DOLLAR_VOL

        ib     = _mock_ib()
        engine = _make_engine(ib)
        ctx    = _ctx(price=101.0, orb=100.0, rsi=65.0, rsi_prev=55.0,
                      ma50=95.0, ma200=85.0, dollar_vol=SCAN_MIN_DOLLAR_VOL)

        _run_entry_cycle(ib, engine, ctx)

        assert 'TSLA' in engine.state, "1× dollar-vol must pass on non-Friday"

    # ── IBKR sync: add missing position ──────────────────────────────────────

    def test_sync_adds_ibkr_position_not_in_state(self):
        """Position filled while engine was down must be added to state on next sync."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        engine.state = {}

        pos = MagicMock()
        pos.contract.symbol = 'AAPL'
        pos.position        = 5.0
        pos.avgCost         = 175.0
        ib.positions.return_value  = [pos]
        ib.openTrades.return_value = []

        engine._sync_positions_from_ibkr()

        assert 'AAPL' in engine.state
        assert engine.state['AAPL']['price'] == pytest.approx(175.0)
        assert engine.state['AAPL']['fill_price'] == pytest.approx(175.0)
        assert engine.state['AAPL']['peak_price'] == pytest.approx(175.0)
        assert engine.state['AAPL']['qty']   == pytest.approx(5.0)

    def test_sync_updates_existing_position_qty_from_ibkr(self):
        """Partial fills/manual adjustments must update state quantity."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        engine.state = {
            'AAPL': {
                'price': 175.0, 'qty': 5.0, 'stop_loss': 160.0,
                'volume': 0, 'score': 50,
                'time': datetime.now(pytz.timezone('US/Eastern')).isoformat(),
            }
        }

        pos = MagicMock()
        pos.contract.symbol = 'AAPL'
        pos.position        = 3.0
        pos.avgCost         = 175.0
        ib.positions.return_value = [pos]

        engine._sync_positions_from_ibkr()

        assert engine.state['AAPL']['qty'] == pytest.approx(3.0)
        assert engine.state['AAPL']['broker_avg_cost'] == pytest.approx(175.0)

    def test_sync_does_not_remove_state_on_first_missing_snapshot(self):
        """One empty/partial IBKR positions snapshot must not delete risk state."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        engine.state = {
            'AAPL': {
                'price': 175.0, 'qty': 5.0, 'stop_loss': 160.0,
                'volume': 0, 'score': 50,
                'time': datetime.now(pytz.timezone('US/Eastern')).isoformat(),
            }
        }
        ib.positions.return_value = []

        engine._sync_positions_from_ibkr()

        assert 'AAPL' in engine.state
        assert engine._missing_position_counts['AAPL'] == 1

    def test_sync_removes_state_after_second_missing_snapshot(self):
        """A second confirming missing snapshot removes stale state."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        engine.state = {
            'AAPL': {
                'price': 175.0, 'qty': 5.0, 'stop_loss': 160.0,
                'volume': 0, 'score': 50,
                'time': datetime.now(pytz.timezone('US/Eastern')).isoformat(),
            }
        }
        ib.positions.return_value = []

        engine._sync_positions_from_ibkr()
        engine._sync_positions_from_ibkr()

        assert 'AAPL' not in engine.state

    def test_sync_avgcost_zero_position_skips_velocity_exit_profit_check(self):
        """Position synced with avgCost=0 must not crash velocity exit (division by zero)."""
        ib     = _mock_ib()
        engine = _make_engine(ib)

        tz_ny = pytz.timezone('US/Eastern')
        entry_time = tz_ny.localize(datetime(2024, 6, 3, 10, 30)).isoformat()
        engine.state = {'GHOST': {'price': 0.0, 'time': entry_time,
                                   'qty': 3.0, 'stop_loss': 0.0,
                                   'volume': 0, 'score': None}}

        ticker = MagicMock()
        ticker.marketPrice.return_value = 100.0
        ticker.close = 100.0
        ib.reqTickers.return_value = [ticker]

        engine.manage_position_exits()   # must not raise

        assert 'GHOST' in engine.state, "Zero-price position must be left alone (not liquidated)"

    # ── Indicator edge cases ─────────────────────────────────────────────────

    def test_rsi_flat_price_does_not_crash(self):
        """RSI with all-zero deltas (gain=0, loss=0) must return NaN, not crash."""
        import pandas as pd
        from src.indicators import compute_rsi
        flat = pd.Series([100.0] * 20)   # perfectly flat — no gain or loss
        result = compute_rsi(flat, period=14)
        # NaN is acceptable; 0/0 = NaN in floating point, not a crash
        assert isinstance(result, pd.Series)
        assert len(result) == len(flat)

    def test_rsi_all_gains_returns_100(self):
        """RSI with only up days (loss=0) must return 100 for those bars."""
        import pandas as pd
        from src.indicators import compute_rsi
        rising = pd.Series([float(i) for i in range(1, 31)])  # 1,2,3,...,30
        result = compute_rsi(rising, period=14)
        # All gains → loss=0 → gain/0=inf → 100/(1+inf)=0 → 100-0=100
        assert result.dropna().iloc[-1] == pytest.approx(100.0, abs=0.01)

    def test_atr_with_identical_bars_returns_zero(self):
        """ATR of a stock with identical high/low/close (halted) must be 0, not crash."""
        import pandas as pd
        from src.indicators import compute_atr
        df = pd.DataFrame({
            'high':  [100.0] * 20,
            'low':   [100.0] * 20,
            'close': [100.0] * 20,
        })
        result = compute_atr(df, period=14)
        assert isinstance(result, pd.Series)
        assert result.dropna().iloc[-1] == pytest.approx(0.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# New feature tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGapFilter:
    """
    Gap filter: day_open > orb_high * (1 + GAP_MAX_PCT) must block the signal.
    This matches the daily-bar backtester, where the gap cap is checked against
    the signal day's open rather than the completed close.
    """

    def test_gap_filter_blocks_excessive_opening_gap(self):
        """
        Day open that is >10% above ORB high must not generate a signal,
        even if all other conditions pass.
        """
        from src.config import GAP_MAX_PCT
        ib     = _mock_ib()
        engine = _make_engine(ib)

        orb_h = 100.0
        price = orb_h * (1 + GAP_MAX_PCT + 0.02)
        day_open = orb_h * (1 + GAP_MAX_PCT + 0.01)   # 11% opening gap — fails gap filter
        ctx = _ctx(
            price=price, orb=orb_h, day_open=day_open,
            ma50=price - 5, ma200=price - 20,       # price > MA50 > MA200 ✓
            rsi=62.0, rsi_prev=57.0,
            dollar_vol=500_000_000,
        )

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['GAPPER']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert ib.placeOrder.call_count == 0, "Excessive opening gap must block order placement"
        assert 'GAPPER' not in engine.state

    def test_gap_filter_passes_when_open_at_boundary(self):
        """Day open exactly at ORB * (1 + GAP_MAX_PCT) must pass the gap filter."""
        from src.config import GAP_MAX_PCT
        ib     = _mock_ib()
        engine = _make_engine(ib)

        orb_h = 100.0
        day_open = orb_h * (1 + GAP_MAX_PCT)        # exactly 10% — passes
        price = day_open * 1.01                     # current extension is scorer quality, not hard gap gate
        ctx = _ctx(
            price=price, orb=orb_h, day_open=day_open,
            ma50=price - 5, ma200=price - 20,
            rsi=62.0, rsi_prev=57.0,
            dollar_vol=500_000_000,
        )

        mock_trade = MagicMock()
        mock_trade.order.orderId             = 1
        mock_trade.orderStatus.status        = 'Filled'
        mock_trade.orderStatus.avgFillPrice  = price
        mock_trade.fills = [_mock_fill(1.0)]
        ib.placeOrder.return_value = mock_trade

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['ATEDGE']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert ib.placeOrder.call_count == 2, "Opening gap at boundary must generate 2 placeOrder calls (BUY + TRAIL)"


class TestDailyLossCircuitBreaker:
    """
    Circuit breaker: when equity drops > MAX_DAILY_LOSS_PCT from the day's
    opening equity, run_cycle() must skip new entries.
    """

    def test_circuit_breaker_halts_entries_on_daily_loss(self):
        """Equity dropping 3%+ from day open must prevent bracketOrder calls."""
        from src.config import MAX_DAILY_LOSS_PCT
        ib     = _mock_ib()
        engine = _make_engine(ib)

        # Simulate: day started at 1400, now equity is 3.1% lower
        engine._day_start_date   = '2024-06-05'
        engine._day_start_equity = 1400.0
        loss_equity = round(1400.0 * (1 - MAX_DAILY_LOSS_PCT - 0.001), 2)

        # Override accountSummary to return the loss equity
        nl = MagicMock(); nl.tag = 'NetLiquidation'; nl.value = str(loss_equity)
        sc = MagicMock(); sc.tag = 'SettledCash';    sc.value = '5000.0'
        ib.accountSummary.return_value = [nl, sc]

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['TSLA']), \
             patch.object(engine, 'get_technical_context', return_value=_ctx()), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, 'manage_position_exits'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert ib.placeOrder.call_count == 0, "Circuit breaker must block all new entries"

    def test_circuit_breaker_resets_on_new_day(self):
        """Day-start equity resets when the date changes — new day, clean slate."""
        ib     = _mock_ib()
        engine = _make_engine(ib)

        # Previous day had a big loss
        engine._day_start_date   = '2024-06-04'
        engine._day_start_equity = 1400.0

        # Today (June 5) equity is fine
        nl = MagicMock(); nl.tag = 'NetLiquidation'; nl.value = '1380.0'
        sc = MagicMock(); sc.tag = 'SettledCash';    sc.value = '5000.0'
        ib.accountSummary.return_value = [nl, sc]

        # Passing ctx: price(101) > orb(100) > ma200(85); price > ma50(95) > ma200
        passing_ctx = _ctx(price=101.0, orb=100.0, ma50=95.0, ma200=85.0,
                           rsi=65.0, rsi_prev=55.0, dollar_vol=500_000_000)

        mock_trade = MagicMock()
        mock_trade.order.orderId             = 1
        mock_trade.orderStatus.status        = 'Filled'
        mock_trade.orderStatus.avgFillPrice  = 101.0
        mock_trade.fills = [_mock_fill(1.0)]
        ib.placeOrder.return_value = mock_trade

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['TSLA']), \
             patch.object(engine, 'get_technical_context', return_value=passing_ctx), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        # Day start should now be June 5 with fresh equity
        assert engine._day_start_date == '2024-06-05'
        assert engine._day_start_equity == pytest.approx(1380.0, abs=0.01)
        # And since it's a fresh day, 2 placeOrder calls (BUY + TRAIL stop) must have been made
        assert ib.placeOrder.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# 10. STARTUP SAFETY — _initialize() orphan-order cancellation
# ─────────────────────────────────────────────────────────────────────────────

class TestInitializeSafeStartup:
    """
    _initialize() must only cancel orders whose symbol is NOT in self.state.
    Cancelling bracket stop/TP orders protecting an active position would
    leave it unprotected — a safety-critical bug.
    """

    def test_orphan_cancel_skips_active_position_orders(self):
        """Bracket orders for a symbol in state must NOT be cancelled at startup."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')
        engine.state = {
            'HELD': {'price': 100.0, 'qty': 5.0, 'stop_loss': 90.0,
                     'volume': 0, 'score': 60,
                     'time': datetime.now(tz_ny).isoformat()},
        }

        held_trade   = MagicMock(); held_trade.contract.symbol   = 'HELD';   held_trade.order.action = 'SELL'
        orphan_trade = MagicMock(); orphan_trade.contract.symbol = 'ORPHAN'; orphan_trade.order.action = 'BUY'
        ib.reqAllOpenOrders.return_value = [held_trade, orphan_trade]
        ib.openTrades.return_value       = []

        with patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_update_position_prices'):
            engine._initialize()

        cancelled = [c[0][0] for c in ib.cancelOrder.call_args_list]
        assert orphan_trade.order in cancelled, "Orphan order must be cancelled"
        assert held_trade.order not in cancelled, \
            "Bracket order protecting active position must not be cancelled"

    def test_no_cancel_when_all_orders_belong_to_active_positions(self):
        """When every open order maps to a state symbol, nothing is cancelled."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')
        engine.state = {
            'SYM': {'price': 50.0, 'qty': 2.0, 'stop_loss': 45.0,
                    'volume': 0, 'score': 70,
                    'time': datetime.now(tz_ny).isoformat()},
        }

        active_trade = MagicMock(); active_trade.contract.symbol = 'SYM'; active_trade.order.action = 'SELL'
        ib.reqAllOpenOrders.return_value = [active_trade]
        ib.openTrades.return_value       = []

        with patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_update_position_prices'):
            engine._initialize()

        assert ib.cancelOrder.call_count == 0, \
            "No orders must be cancelled when all belong to active positions"

    def test_startup_preserves_protective_sell_when_state_missing_but_ibkr_has_position(self):
        """
        If the local state file is missing/corrupt, startup must not cancel a
        SELL stop protecting a real IBKR position. IBKR positions are the source
        of truth during recovery.
        """
        ib     = _mock_ib()
        engine = _make_engine(ib)
        engine.state = {}

        pos = MagicMock()
        pos.contract.symbol = 'HELD'
        pos.position = 5.0
        pos.avgCost = 100.0
        ib.positions.return_value = [pos]

        protective = MagicMock()
        protective.contract.symbol = 'HELD'
        protective.order.action = 'SELL'

        orphan_buy = MagicMock()
        orphan_buy.contract.symbol = 'ORPHAN'
        orphan_buy.order.action = 'BUY'

        ib.reqAllOpenOrders.return_value = [protective, orphan_buy]
        ib.openTrades.return_value       = []

        with patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_audit_stop_orders'):
            engine._initialize()

        cancelled = [c[0][0] for c in ib.cancelOrder.call_args_list]
        assert orphan_buy.order in cancelled
        assert protective.order not in cancelled

    def test_startup_cancels_orphaned_sell_when_no_state_or_position_exists(self):
        """A leftover SELL with no state and no IBKR position is dangerous and must go."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        engine.state = {}
        ib.positions.return_value = []

        orphan_sell = MagicMock()
        orphan_sell.contract.symbol = 'FLAT'
        orphan_sell.order.action = 'SELL'

        ib.reqAllOpenOrders.return_value = [orphan_sell]
        ib.openTrades.return_value       = []

        with patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_audit_stop_orders'):
            engine._initialize()

        ib.cancelOrder.assert_called_once_with(orphan_sell.order)


# ─────────────────────────────────────────────────────────────────────────────
# 11. RSI DELTA GATE — minimum acceleration required
# ─────────────────────────────────────────────────────────────────────────────

class TestRsiDeltaGate:
    """
    c_rsi_delta = (rsi - rsi_prev) >= RSI_MIN_DELTA must block entries where
    RSI is technically rising but by a trivially small amount (noise).
    """

    def test_tiny_rsi_rise_blocks_entry(self):
        """RSI delta below RSI_MIN_DELTA must not generate a signal."""
        from src.config import RSI_MIN_DELTA
        ib     = _mock_ib()
        engine = _make_engine(ib)
        # delta = 0.5 < RSI_MIN_DELTA → c_rsi_delta fails
        ctx = _ctx(price=101.0, orb=100.0, rsi=55.5, rsi_prev=55.0,
                   ma50=95.0, ma200=85.0)
        _run_entry_cycle(ib, engine, ctx)
        assert ib.placeOrder.call_count == 0, \
            f"RSI delta {55.5-55.0:.1f} < {RSI_MIN_DELTA} must block entry"
        assert 'TSLA' not in engine.state

    def test_rsi_delta_at_minimum_allows_entry(self):
        """RSI delta exactly at RSI_MIN_DELTA must pass the gate."""
        from src.config import RSI_MIN_DELTA
        ib     = _mock_ib()
        engine = _make_engine(ib)
        rsi_prev = 60.0
        rsi      = rsi_prev + RSI_MIN_DELTA   # delta == RSI_MIN_DELTA exactly
        ctx = _ctx(price=101.0, orb=100.0, rsi=rsi, rsi_prev=rsi_prev,
                   ma50=95.0, ma200=85.0)

        _run_entry_cycle(ib, engine, ctx)
        assert ib.placeOrder.call_count == 2, \
            f"RSI delta exactly at {RSI_MIN_DELTA} must allow entry (2 calls: BUY + TRAIL)"


# ─────────────────────────────────────────────────────────────────────────────
# 12. SCANNER OPTIMIZATION — skip scan when all slots are full
# ─────────────────────────────────────────────────────────────────────────────

class TestScannerSkipWhenFull:
    """
    get_institutional_scan() must not be called when dynamic equity capacity is
    already full — it wastes an API call and adds latency to every cycle.
    """

    def test_scanner_not_called_when_all_slots_filled(self):
        from src.config import MAX_POSITIONS_CAP, MIN_BUCKET_SIZE
        ib    = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        equity = 1400.0
        max_positions = min(int(equity / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)

        # Populate IBKR and state with all dynamic position slots.
        ibkr_positions = []
        for i in range(max_positions):
            sym = f'SYM{i}'
            pos = MagicMock()
            pos.contract.symbol = sym
            pos.position        = 5.0
            pos.avgCost         = 100.0
            ibkr_positions.append(pos)
            engine.state[sym] = {
                'price': 100.0, 'qty': 5.0, 'current_price': 100.0,
                'stop_loss': 90.0,
                'volume': 0, 'score': 60,
                'time': datetime.now(tz_ny).isoformat(),
            }
        ib.positions.return_value  = ibkr_positions
        ib.openTrades.return_value = []

        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan') as mock_scan, \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, 'manage_position_exits'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.run_cycle()

        mock_scan.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 13. HARD STOP — forced exit when drawdown exceeds threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestHardStop:
    def _make_position(self, symbol, qty):
        pos = MagicMock()
        pos.contract.symbol = symbol
        pos.position        = qty
        return pos

    def _state_entry(self, price, cur, tz_ny, entry_time=None):
        entry_time = entry_time or datetime.now(tz_ny).isoformat()
        return {'price': price, 'qty': 5.0, 'current_price': cur,
                'stop_loss': price * 0.85,
                'volume': 0, 'score': 60, 'time': entry_time}

    def _run_exit_check(self, engine):
        fake_now = pytz.timezone('US/Eastern').localize(datetime(2024, 6, 5, 10, 30))
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine.manage_position_exits()

    def test_hard_stop_triggers_when_down_beyond_threshold(self):
        """Drawdown > HARD_STOP_PCT from entry must force-liquidate immediately."""
        from src.config import HARD_STOP_PCT
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = round(entry * (1 - HARD_STOP_PCT - 0.01), 2)   # e.g. 92.9 (down 7.1%)
        engine.state = {'POS': self._state_entry(entry, cur, tz_ny)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('POS', 5.0)]

        self._run_exit_check(engine)

        assert engine.state['POS']['pending_exit'] is True, "Hard stop must mark position pending exit"
        assert ib.placeOrder.called

    def test_hard_stop_uses_fresh_price_over_stale_cached_price(self):
        """Fresh broker price must override stale state.current_price for hard stops."""
        from src.config import HARD_STOP_PCT
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        stale_cur = 100.0
        fresh_cur = round(entry * (1 - HARD_STOP_PCT - 0.01), 2)
        engine.state = {'POS': self._state_entry(entry, stale_cur, tz_ny)}
        ib.reqTickers.return_value = [_mock_price_ticker(fresh_cur)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('POS', 5.0)]

        self._run_exit_check(engine)

        assert engine.state['POS']['pending_exit'] is True
        assert ib.placeOrder.called

    def test_hard_stop_does_not_trigger_within_threshold(self):
        """Drawdown exactly below HARD_STOP_PCT must leave position open."""
        from src.config import HARD_STOP_PCT
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = round(entry * (1 - HARD_STOP_PCT + 0.01), 2)   # e.g. 94.0 (down 6%)
        engine.state = {'POS': self._state_entry(entry, cur, tz_ny)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]

        self._run_exit_check(engine)

        assert 'POS' in engine.state, "Position within loss threshold must not be force-closed"
        assert not ib.placeOrder.called

    def test_break_even_exit_triggers_after_prior_profit_retraces_to_entry(self):
        """Once peak exceeds break-even threshold, a retrace to entry must force exit."""
        from src.config import BREAK_EVEN_PCT
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 99.95
        state = self._state_entry(entry, cur, tz_ny)
        state['peak_price'] = entry * (1 + BREAK_EVEN_PCT + 0.01)
        engine.state = {'POS': state}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('POS', 5.0)]

        self._run_exit_check(engine)

        assert engine.state['POS']['pending_exit'] is True
        assert ib.placeOrder.called

    def test_break_even_exit_does_not_trigger_before_profit_threshold(self):
        """A normal small loser must not be break-even-exited before profit threshold was reached."""
        from src.config import BREAK_EVEN_PCT
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 99.95
        state = self._state_entry(entry, cur, tz_ny)
        state['peak_price'] = entry * (1 + BREAK_EVEN_PCT - 0.01)
        engine.state = {'POS': state}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]

        self._run_exit_check(engine)

        assert 'pending_exit' not in engine.state['POS']
        assert not ib.placeOrder.called


# ─────────────────────────────────────────────────────────────────────────────
# 14. FRIDAY CLOSE — close under-performing positions before weekend
# ─────────────────────────────────────────────────────────────────────────────

class TestFridayClose:
    def _make_position(self, symbol, qty):
        pos = MagicMock()
        pos.contract.symbol = symbol
        pos.position        = qty
        return pos

    def _state_entry(self, price, cur, tz_ny):
        return {'price': price, 'qty': 5.0, 'current_price': cur,
                'stop_loss': price * 0.90,
                'volume': 0, 'score': 60, 'time': datetime.now(tz_ny).isoformat()}

    def test_friday_close_triggers_below_profit_threshold(self):
        """On Friday after close hour, profit < threshold must force-liquidate."""
        from src.config import FRIDAY_CLOSE_HOUR, FRIDAY_MIN_PROFIT_PCT
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = round(entry * (1 + FRIDAY_MIN_PROFIT_PCT - 0.01), 2)   # just below threshold
        engine.state = {'FRI': self._state_entry(entry, cur, tz_ny)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('FRI', 5.0)]

        friday_after = tz_ny.localize(datetime(2024, 6, 7, FRIDAY_CLOSE_HOUR + 1, 0))

        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = friday_after
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.manage_position_exits()

        assert engine.state['FRI']['pending_exit'] is True, "Friday close must mark below-threshold position pending exit"

    def test_friday_close_does_not_trigger_above_threshold(self):
        """Profit above FRIDAY_MIN_PROFIT_PCT on Friday must keep position open."""
        from src.config import FRIDAY_CLOSE_HOUR, FRIDAY_MIN_PROFIT_PCT
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = round(entry * (1 + FRIDAY_MIN_PROFIT_PCT + 0.01), 2)   # above threshold
        engine.state = {'FRI': self._state_entry(entry, cur, tz_ny)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]

        friday_after = tz_ny.localize(datetime(2024, 6, 7, FRIDAY_CLOSE_HOUR + 1, 0))

        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = friday_after
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.manage_position_exits()

        assert 'FRI' in engine.state, "Profitable position must not be closed on Friday"
        assert not ib.placeOrder.called

    def test_friday_close_does_not_trigger_before_close_hour(self):
        """Before FRIDAY_CLOSE_HOUR, Friday close rule must be inactive."""
        from src.config import FRIDAY_CLOSE_HOUR, FRIDAY_MIN_PROFIT_PCT
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = round(entry * (1 + FRIDAY_MIN_PROFIT_PCT - 0.01), 2)   # below threshold
        engine.state = {'FRI': self._state_entry(entry, cur, tz_ny)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]

        friday_morning = tz_ny.localize(datetime(2024, 6, 7, FRIDAY_CLOSE_HOUR - 2, 0))

        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = friday_morning
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.manage_position_exits()

        assert 'FRI' in engine.state, "Friday close must not trigger before FRIDAY_CLOSE_HOUR"
        assert not ib.placeOrder.called


# ─────────────────────────────────────────────────────────────────────────────
# 15. EOD FLAT — liquidate unprofitable positions before end of trading day
# ─────────────────────────────────────────────────────────────────────────────

class TestEodFlat:
    """
    After EOD_EXIT_TIME (default 15:45 ET) on any trading day, positions that
    are not in profit may be liquidated only after the minimum swing hold window
    has elapsed. Same-day entries are not rejected just because they are flat or
    down near the close. The rule fires at most once per calendar trading day.
    """

    def _state_entry(self, entry, cur, tz_ny, entry_time=None):
        return {
            'price': entry, 'qty': 5.0, 'current_price': cur,
            'stop_loss': entry * 0.93,
            'volume': 0, 'score': 60,
            'time': (entry_time or datetime.now(tz_ny)).isoformat(),
        }

    def _make_position(self, symbol, qty):
        pos = MagicMock()
        pos.contract.symbol = symbol
        pos.position        = qty
        return pos

    def test_eod_flat_triggers_when_older_position_at_loss(self):
        """After EOD_EXIT_TIME an older position with profit < 0 must be liquidated."""
        from src.config import EOD_EXIT_TIME
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 98.0   # -2%, not in profit
        old_entry = tz_ny.localize(datetime(2024, 6, 4, 10, 30))
        engine.state = {'LOSS': self._state_entry(entry, cur, tz_ny, old_entry)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('LOSS', 5.0)]

        # Wednesday at EOD_EXIT_TIME + 5 min
        eod_time = tz_ny.localize(
            datetime(2024, 6, 5, EOD_EXIT_TIME[0], EOD_EXIT_TIME[1] + 5)
        )
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = eod_time
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()

        assert engine.state['LOSS']['pending_exit'] is True, \
            "EOD flat must liquidate position not in profit"

    def test_eod_flat_triggers_when_older_position_exactly_at_entry(self):
        """After EOD_EXIT_TIME an older zero-profit position must also be liquidated."""
        from src.config import EOD_EXIT_TIME
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 100.0  # exactly at entry, profit = 0
        old_entry = tz_ny.localize(datetime(2024, 6, 4, 10, 30))
        engine.state = {'FLAT': self._state_entry(entry, cur, tz_ny, old_entry)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('FLAT', 5.0)]

        eod_time = tz_ny.localize(
            datetime(2024, 6, 5, EOD_EXIT_TIME[0], EOD_EXIT_TIME[1] + 5)
        )
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = eod_time
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()

        assert engine.state['FLAT']['pending_exit'] is True, \
            "EOD flat must liquidate zero-profit position"

    def test_eod_flat_does_not_trigger_when_profit_clears_velocity_threshold(self):
        """A strong older winner after EOD_EXIT_TIME must not be closed."""
        from src.config import EOD_EXIT_TIME, PROFIT_MIN_THRESHOLD
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = entry * (1 + PROFIT_MIN_THRESHOLD + 0.01)
        old_entry = tz_ny.localize(datetime(2024, 6, 4, 10, 30))
        engine.state = {'GAIN': self._state_entry(entry, cur, tz_ny, old_entry)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]

        eod_time = tz_ny.localize(
            datetime(2024, 6, 5, EOD_EXIT_TIME[0], EOD_EXIT_TIME[1] + 5)
        )
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = eod_time
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()

        assert 'GAIN' in engine.state, "Strong profitable position must not be closed at EOD"
        assert not ib.placeOrder.called

    def test_eod_flat_does_not_close_same_day_loss(self):
        """A same-day swing entry must not be closed by EOD flat."""
        from src.config import EOD_EXIT_TIME
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 98.0
        same_day_entry = tz_ny.localize(datetime(2024, 6, 5, 11, 20))
        engine.state = {'NEW': self._state_entry(entry, cur, tz_ny, same_day_entry)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]

        eod_time = tz_ny.localize(
            datetime(2024, 6, 5, EOD_EXIT_TIME[0], EOD_EXIT_TIME[1] + 5)
        )
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = eod_time
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()

        assert 'NEW' in engine.state, "Same-day swing entry must not be closed at EOD"
        assert not ib.placeOrder.called

    def test_eod_flat_does_not_trigger_before_eod_time(self):
        """Before EOD_EXIT_TIME the EOD flat rule must be inactive."""
        from src.config import EOD_EXIT_TIME
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 95.0   # -5%, would trigger if time were right
        same_day_entry = tz_ny.localize(datetime(2024, 6, 5, 10, 0))
        engine.state = {'EARLY': self._state_entry(entry, cur, tz_ny, same_day_entry)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]

        # One minute before EOD_EXIT_TIME
        before_eod = tz_ny.localize(
            datetime(2024, 6, 5, EOD_EXIT_TIME[0], EOD_EXIT_TIME[1] - 1)
        )
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = before_eod
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()

        assert 'EARLY' in engine.state, "EOD flat must not trigger before EOD_EXIT_TIME"
        assert not ib.placeOrder.called

    def test_eod_flat_fires_only_once_per_day(self):
        """EOD flat must not re-liquidate on the second cycle of the same day."""
        from src.config import EOD_EXIT_TIME
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 98.0
        old_entry = tz_ny.localize(datetime(2024, 6, 4, 10, 30))
        engine.state = {'ONCE': self._state_entry(entry, cur, tz_ny, old_entry)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('ONCE', 5.0)]

        eod_time = tz_ny.localize(
            datetime(2024, 6, 5, EOD_EXIT_TIME[0], EOD_EXIT_TIME[1] + 5)
        )
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = eod_time
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()   # first call — fires

        # Simulate position partially cleared, then second call same day
        same_day_entry = tz_ny.localize(datetime(2024, 6, 5, 11, 0))
        engine.state['ONCE'] = self._state_entry(entry, cur, tz_ny, same_day_entry)
        place_count_after_first = ib.placeOrder.call_count

        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = eod_time
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()   # second call same day — must not re-fire

        assert ib.placeOrder.call_count == place_count_after_first, \
            "EOD flat must not re-fire on the second cycle of the same trading day"


# ─────────────────────────────────────────────────────────────────────────────
# 16. COMMISSION REPORT — async IB callback updates state
# ─────────────────────────────────────────────────────────────────────────────

class TestCommissionReport:
    """
    _on_commission_report(trade, fill, report) is the async IB callback.
    It must update state['commission'] only when:
      - report.commission is a valid positive number (not NaN, not 0)
      - trade.order.orderId matches an entry_order_id stored in state
    """

    def _make_state_with_order(self, order_id=42):
        tz_ny = pytz.timezone('US/Eastern')
        return {
            'TSLA': {
                'fill_price': 100.0, 'price': 100.0,
                'entry_order_id': order_id,
                'time': datetime.now(tz_ny).isoformat(),
                'qty': 5.0, 'stop_loss': 94.0,
                'volume': 0, 'score': 60,
            }
        }

    def _fire(self, engine, order_id, commission):
        trade  = MagicMock(); trade.order.orderId = order_id
        fill   = MagicMock()
        report = MagicMock(); report.commission = commission
        with patch.object(engine, 'save_state') as mock_save:
            engine._on_commission_report(trade, fill, report)
        return mock_save

    def test_matching_order_id_stores_commission(self):
        """Commission for the correct entry order ID must be saved to state."""
        engine = _make_engine(_mock_ib())
        engine.state = self._make_state_with_order(order_id=42)

        mock_save = self._fire(engine, order_id=42, commission=1.25)

        assert engine.state['TSLA']['commission'] == pytest.approx(1.25, abs=0.001)
        mock_save.assert_called_once()

    def test_non_matching_order_id_does_not_update_state(self):
        """Commission for a different order ID must be silently ignored."""
        engine = _make_engine(_mock_ib())
        engine.state = self._make_state_with_order(order_id=42)

        mock_save = self._fire(engine, order_id=99, commission=1.25)

        assert 'commission' not in engine.state['TSLA']
        mock_save.assert_not_called()

    def test_nan_commission_is_ignored(self):
        """NaN commission (IB placeholder) must not update state — real value arrives later."""
        engine = _make_engine(_mock_ib())
        engine.state = self._make_state_with_order(order_id=42)

        mock_save = self._fire(engine, order_id=42, commission=float('nan'))

        assert 'commission' not in engine.state['TSLA']
        mock_save.assert_not_called()

    def test_zero_commission_is_ignored(self):
        """Zero commission (not a real fill) must not update state."""
        engine = _make_engine(_mock_ib())
        engine.state = self._make_state_with_order(order_id=42)

        mock_save = self._fire(engine, order_id=42, commission=0.0)

        assert 'commission' not in engine.state['TSLA']
        mock_save.assert_not_called()

    def test_commission_rounded_to_4_decimal_places(self):
        """Commission is stored rounded to 4 dp regardless of IB's raw precision."""
        engine = _make_engine(_mock_ib())
        engine.state = self._make_state_with_order(order_id=42)

        self._fire(engine, order_id=42, commission=1.123456789)

        assert engine.state['TSLA']['commission'] == round(1.123456789, 4)
