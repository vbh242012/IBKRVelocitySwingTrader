"""
Comprehensive validation of three critical VelocityEngine subsystems:

  1. Percent trailing stop order construction (standalone BUY + post-fill TRAIL)
     - trail_dist = limit_price × TRAIL_PCT (0.04 = 4%)
     - trailingPercent on TRAIL order = TRAIL_PCT × 100
     - goodAfterTime is omitted after the configured entry start so IBKR cannot reject a past activation time
     - BUY order: transmit=True; TRAIL stop: standalone GTC transmit=True after fill
     - state.stop_loss  = fill - trail_dist
     - No take-profit order or state key

  2. Screener (IB ScannerSubscription parameters)
     - scanCode from active profile/config, instrument='STK', location='STK.US.MAJOR'
     - price, volume, and market-cap floors mirror the active strategy profile
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

from contextlib import contextmanager
from src.scoring import score_candidate, volume_pace_from_intraday


@contextmanager
def _freeze_all_datetimes(fake_now):
    """Freeze datetime.now() in all engine mixin modules to fake_now."""
    with patch('src.engine.datetime') as m0, \
         patch('src.engine_entries.datetime') as m1, \
         patch('src.engine_orders.datetime') as m2, \
         patch('src.engine_market.datetime') as m3:
        for m in (m0, m1, m2, m3):
            m.now.return_value = fake_now
            m.fromisoformat = datetime.fromisoformat
        yield m0


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_fill(commission=1.0):
    """Build a minimal Fill mock carrying a CommissionReport."""
    fill = MagicMock()
    fill.commissionReport.commission = commission
    return fill


def _mock_price_ticker(price: float, bid=None, ask=None, open=None, high=None, low=None, vwap=None, close=None):
    """Build a ticker mock with deterministic quote fields."""
    ticker = MagicMock()
    ticker.marketPrice.return_value = price
    ticker.last = price
    ticker.close = price if close is None else close
    ticker.bid = price * 0.999 if bid is None else bid
    ticker.ask = price * 1.001 if ask is None else ask
    ticker.open = price if open is None else open
    ticker.high = price if high is None else high
    ticker.low = price if low is None else low
    ticker.vwap = price if vwap is None else vwap
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
    from src.strategy_profiles import get_strategy_profile
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
    engine._prefilter_date     = None
    engine._prefilter_status   = "not_started"
    engine._prefilter_candidates = []
    engine._prefilter_stats    = {}
    engine._last_premarket_prefilter_date = None
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
    engine._strategy_profile = get_strategy_profile("indicator_swing")
    # New instance vars added by fixes
    engine._ib_error_dedup      = {}
    engine._alert_dedup_cache   = {}
    engine._data_blackout_streak = 0
    engine._data_blackout_alerted = False
    engine._log_once_cache      = {}
    engine._indicator_row_cache = {}
    engine._last_pre_entry_sync_date = None
    engine._historical_data_health = {}
    engine._vix_failure_count   = 0
    engine._next_vix_retry_ts   = 0.0
    engine._last_vix_failure_ts = 0.0
    engine._last_vix_source     = None
    engine._health_date = datetime.now().strftime('%Y-%m-%d')
    engine._health_metrics = {}
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
         bid=None, ask=None,
         **updates):
    """Build a get_technical_context()-style dict with all production-rule fields."""
    high10 = round(price * 1.005, 4)   # retained for dashboard/context compatibility
    bid = round(price * (1 - spread_pct / 2), 4) if bid is None else bid
    ask = round(price * (1 + spread_pct / 2), 4) if ask is None else ask
    if day_open is None:
        day_open = price / (1 + intraday_gain) if intraday_gain > -0.99 else price
    ctx = {
        'orb_high':       orb,
        'day_open':       day_open,
        'prev_high':      price - 1.0,
        'prev_daily_high': price - 1.0,
        'ma20':           price * 0.96,
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
        'high20':         price * 1.02,
        'dist_high20':    price / (price * 1.02) - 1.0,
        'rvol':           rvol,
        'volume_pace':    rvol,
        'day_range_location': day_range_location,
        'intraday_gain':  intraday_gain,
        'spread_pct':     spread_pct,
        'bid':            bid,
        'ask':            ask,
        'close':          price - 0.5,
        'live_price':     price,
        'ema20_gt_sma50': True,
        'ma_bull_cross':  False,
        'reclaim_ma20':   False,
        'reclaim_ma50':   False,
        'break_prev_high': True,
        'weekly_uptrend': True,
        'return_13w':     0.25,
        'return_26w':     0.35,
        'relative_strength_63d': 0.15,
        'relative_strength_126d': 0.18,
        'price_vs_52w_high': 0.90,
        'stoch_bull_exit_oversold': True,
        'macd_hist':      0.20,
        'macd_hist_delta': 0.05,
        'macd_bull_divergence': False,
        'obv_slope_5':    100_000.0,
        'obv_uptrend':    True,
        'psar_bull_3':    False,
        'volume':         5_000_000,
        'dollar_vol_20d': dollar_vol,
        'contract':       MagicMock(),
    }
    ctx.update(updates)
    return ctx


def _run_entry_cycle(ib, engine, ctx, sym='TSLA'):
    """
    Run one full run_cycle() with a single passing signal.
    Returns (buy_trade, stop_trade) — the two order mocks from placeOrder calls.
    """
    tz_ny    = pytz.timezone('US/Eastern')
    fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))   # Wed 10:30 — inside window

    fill_price = ctx['live_price']
    from src.config import TRAIL_PCT, RISK_PER_TRADE_PCT

    summary = {item.tag: float(item.value) for item in ib.accountSummary.return_value}
    equity = summary.get('NetLiquidation', 1400.0)
    settled = summary.get('SettledCash', 5000.0)
    limit_price = engine._calc_entry_limit_price(ctx['live_price'], ctx['bid'], ctx['ask'])
    allocation = engine._calc_entry_allocation(equity, settled, len(engine.state))
    risk_dist = round(limit_price * TRAIL_PCT, 2) if limit_price else 0
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
         _freeze_all_datetimes(fake_now):
        engine.run_cycle()

    return buy_trade, stop_trade


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAILING STOP / TAKE-PROFIT ORDER CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

class TestBracketOrderMath:
    """
    Verify percent trail stop distance, GTC 2-order structure, and state persistence.

    Config values used (from src/config.py):
        TRAIL_PCT = 0.04  →  trail_dist = limit_price × 0.04

    With entry=100.00, limit=100.15, TRAIL_PCT=0.04:
        trail_dist  = round(100.15 × 0.04, 2) = 4.01
        stop_loss (state) = fill - trail_dist = 100.00 - 4.01 = 95.99
        trailingPercent on TRAIL order = 4.0
        No take-profit order or state key.
    """

    ENTRY       = 100.00
    LIMIT       = 100.15                         # ask 100.10 + 5 bps cushion, capped by 0.2%
    TRAIL_DIST  = round(100.15 * 0.04, 2)        # 4.01 (4% of limit price)
    INIT_STOP   = round(100.00 - round(100.15 * 0.04, 2), 2)  # 95.99

    def _setup(self):
        ib      = _mock_ib()
        engine  = _make_engine(ib)
        context = _ctx(price=self.ENTRY,
                       orb=self.ENTRY - 5,
                       ma50=self.ENTRY - 3,
                       ma200=self.ENTRY - 15,
                       rsi=62.0, rsi_prev=57.0)
        return ib, engine, context

    # ── trail_dist computation ───────────────────────────────────────────────

    def test_trail_pct_is_four_percent(self):
        from src.config import TRAIL_PCT
        assert TRAIL_PCT == pytest.approx(0.04, abs=1e-6), "TRAIL_PCT must be 0.04 (4%)"

    def test_trail_dist_equals_limit_price_times_trail_pct(self):
        from src.config import TRAIL_PCT
        trail_dist = round(self.LIMIT * TRAIL_PCT, 2)
        assert trail_dist == self.TRAIL_DIST

    def test_trail_dist_is_rounded_to_2_decimals(self):
        from src.config import TRAIL_PCT
        assert round(100.333 * TRAIL_PCT, 2) == round(100.333 * 0.04, 2)

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

    def test_spy_bear_regime_blocks_entries_by_default_profile(self):
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

        assert ib.placeOrder.call_count == 0
        assert 'TSLA' not in engine.state

    def test_spy_bear_regime_blocks_entries_regardless_of_volume_pace_by_default(self):
        ib, engine, ctx = self._setup()
        engine._spy_cache = {'date': '2024-06-05', 'trend': False}
        ctx.update({
            'orb_high': 98.0,
            'rvol': 2.7,
            'rsi': 72.0,
            'rsi_prev': 68.0,
        })

        _run_entry_cycle(ib, engine, ctx)

        assert ib.placeOrder.call_count == 0
        assert 'TSLA' not in engine.state

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

    def test_stop_order_trailing_percent_equals_trail_pct(self):
        """trailingPercent on TRAIL order = TRAIL_PCT × 100 (broker-managed percent trail)."""
        from src.config import TRAIL_PCT
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx)
        stop_order = ib.placeOrder.call_args_list[1][0][1]
        assert stop_order.trailingPercent == pytest.approx(TRAIL_PCT * 100, abs=0.01)

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
        """qty is whole shares sized by settled-cash bucket and percent trail risk."""
        from src.config import (
            TRAIL_PCT,
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
        risk_stop_dist = round(self.LIMIT * TRAIL_PCT, 2)
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
        ctx['atr'] = 10.0
        _run_entry_cycle(ib, engine, ctx)
        assert ib.placeOrder.call_count == 0, "Stock above bucket price must be skipped (int qty = 0)"
        assert 'TSLA' not in engine.state

    # ── State persistence ────────────────────────────────────────────────────

    def test_state_stop_loss_equals_fill_minus_trail_dist(self):
        ib, engine, ctx = self._setup()
        _run_entry_cycle(ib, engine, ctx, sym='TSLA')
        assert 'TSLA' in engine.state
        sl = engine.state['TSLA']['stop_loss']
        assert sl == pytest.approx(self.ENTRY - self.TRAIL_DIST, abs=0.01)

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
             _freeze_all_datetimes(fake_now):
            engine.run_cycle()

        assert 'TSLA' in engine.state
        assert engine.state['TSLA']['price']      == pytest.approx(self.LIMIT, abs=0.01)
        assert engine.state['TSLA']['fill_price'] == pytest.approx(self.LIMIT, abs=0.01)

    def test_unconfirmed_protective_stop_halts_additional_entries_this_cycle(self):
        """After a filled BUY, no second entry is allowed until protection is confirmed."""
        from src.config import (
            TRAIL_PCT, MAX_POSITIONS_CAP, MIN_BUCKET_SIZE,
            RISK_PER_TRADE_PCT, SETTLED_CASH_DEPLOYMENT_PCT,
        )
        ib, engine, ctx = self._setup()

        equity, settled = 1400.0, 5000.0
        max_pos = min(int(equity / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)
        bucket_size = (settled * SETTLED_CASH_DEPLOYMENT_PCT) / max_pos
        risk_dist = round(self.LIMIT * TRAIL_PCT, 2)
        expected_qty = min(int(bucket_size / self.LIMIT), int((equity * RISK_PER_TRADE_PCT) / risk_dist))

        buy_trade = MagicMock()
        buy_trade.order.orderId = 11
        buy_trade.orderStatus.status = 'Filled'
        buy_trade.orderStatus.filled = float(expected_qty)
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
             _freeze_all_datetimes(fake_now):
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
             _freeze_all_datetimes(fake_now):
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
    build_momentum_scanners() must produce ScannerSubscription objects whose
    fields exactly match the strategy spec.  All assertions are on the objects,
    no IB connection needed.
    """

    def setup_method(self):
        from src.config import STRATEGY_PROFILE
        from src.scanner import (
            build_momentum_scanner_filter_options,
            build_momentum_scanners,
        )
        from src.strategy_profiles import get_strategy_profile
        self.profile = get_strategy_profile(STRATEGY_PROFILE)
        self.filter_options = build_momentum_scanner_filter_options()
        self.subscriptions = build_momentum_scanners()
        self.sub = self.subscriptions[0]

    # ── Subscription fields ───────────────────────────────────────────────────

    def test_scan_code_matches_config(self):
        assert [sub.scanCode for sub in self.subscriptions] == list(self.profile.scan_codes)
        assert self.sub.scanCode == self.profile.scan_codes[0]

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
        assert self.sub.abovePrice == self.profile.min_price

    def test_min_volume_matches_config(self):
        assert self.sub.aboveVolume == self.profile.min_volume

    def test_min_volume_is_integer(self):
        """IB rejects float for aboveVolume; must be int."""
        assert isinstance(self.sub.aboveVolume, int)

    def test_market_cap_converted_to_millions(self):
        """
        IB's marketCapAbove field is in millions.
        """
        assert self.sub.marketCapAbove == self.profile.min_market_cap / 1_000_000

    def test_stock_type_filter_is_corp(self):
        """stockTypeFilter='CORP' excludes ETFs at scanner level."""
        assert self.sub.stockTypeFilter == 'CORP'

    def test_scanner_side_filter_options_copy_direct_screener_filters(self):
        from src.config import IB_SCANNER_FILTERS_ENABLED
        if not IB_SCANNER_FILTERS_ENABLED:
            assert self.filter_options == []
            return

        filters = {tv.tag: tv.value for tv in self.filter_options}
        expected = {
            "changeOpenPercAbove": self.profile.scanner_change_open_pct_above,
            "openGapPercBelow": self.profile.scanner_open_gap_pct_below,
            "lastVsEMAChangeRatio20Above": self.profile.scanner_last_vs_ema20_pct_above,
            "lastVsEMAChangeRatio50Above": self.profile.scanner_last_vs_ema50_pct_above,
            "curMACDDistAbove": self.profile.scanner_macd_histogram_above,
        }
        for tag, value in expected.items():
            if value is None:
                assert tag not in filters
            else:
                assert filters[tag] == f"{float(value):g}"


class TestApplicationSymbolUniverse:
    def test_loads_symbols_from_configured_file(self, tmp_path, monkeypatch):
        import src.scanner as scanner

        universe_file = tmp_path / "symbols.csv"
        universe_file.write_text("symbol,name\nAAPL,Apple\nmsft,Microsoft\nAAPL,Duplicate\n")
        monkeypatch.setattr(scanner, "APP_SCANNER_UNIVERSE_FILE", str(universe_file))

        assert scanner.load_application_symbol_universe() == ["AAPL", "MSFT"]

    def test_uses_stale_cache_when_listing_fetch_fails(self, tmp_path, monkeypatch):
        import json
        import src.scanner as scanner

        cache_file = tmp_path / "universe_cache.json"
        cache_file.write_text(json.dumps({"fetched_at": 1.0, "symbols": ["AAPL", "NVDA"]}))
        monkeypatch.setattr(scanner, "APP_SCANNER_UNIVERSE_FILE", "")
        monkeypatch.setattr(scanner, "APP_SCANNER_UNIVERSE_CACHE_FILE", str(cache_file))
        monkeypatch.setattr(scanner, "APP_SCANNER_UNIVERSE_TTL_SEC", 0.0)
        monkeypatch.setattr(
            scanner,
            "fetch_common_stock_universe",
            lambda: (_ for _ in ()).throw(RuntimeError("network unavailable")),
        )

        assert scanner.load_application_symbol_universe(force_refresh=True) == ["AAPL", "NVDA"]


class TestGetInstitutionalScan:
    """
    get_institutional_scan() must extract symbols from IB scan results
    via the correct attribute path and return every unique symbol.
    """

    def setup_method(self):
        self._source_patch = patch("src.engine_scanner.APP_SCANNER_SOURCE", "ibkr")
        self._source_patch.start()

    def teardown_method(self):
        self._source_patch.stop()

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
        from src.config import STRATEGY_PROFILE
        from src.strategy_profiles import get_strategy_profile
        from ib_async import ScannerSubscription
        profile = get_strategy_profile(STRATEGY_PROFILE)
        ib     = _mock_ib()
        engine = _make_engine(ib)
        ib.reqScannerData.return_value = []

        engine.get_institutional_scan()

        assert ib.reqScannerData.call_count == len(profile.scan_codes)
        for call in ib.reqScannerData.call_args_list:
            assert isinstance(call.args[0], ScannerSubscription)
            assert call.args[1] == []
            filter_options = call.args[2]
            filter_tags = {tv.tag for tv in filter_options}
            expected_filters = {
                "changeOpenPercAbove": profile.scanner_change_open_pct_above,
                "openGapPercBelow": profile.scanner_open_gap_pct_below,
                "lastVsEMAChangeRatio20Above": profile.scanner_last_vs_ema20_pct_above,
                "lastVsEMAChangeRatio50Above": profile.scanner_last_vs_ema50_pct_above,
                "curMACDDistAbove": profile.scanner_macd_histogram_above,
            }
            for tag, value in expected_filters.items():
                if value is None:
                    assert tag not in filter_tags
                else:
                    assert tag in filter_tags
        used_codes = {call.args[0].scanCode for call in ib.reqScannerData.call_args_list}
        assert used_codes == set(profile.scan_codes)

    def test_symbols_from_multiple_scanners_are_deduped(self):
        """Symbols appearing in more than one scanner are only returned once."""
        from src.config import STRATEGY_PROFILE
        from src.strategy_profiles import get_strategy_profile
        n  = len(get_strategy_profile(STRATEGY_PROFILE).scan_codes)
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
        from src.config import STRATEGY_PROFILE
        from src.strategy_profiles import get_strategy_profile
        n  = len(get_strategy_profile(STRATEGY_PROFILE).scan_codes)
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

    def test_universe_source_returns_rotating_application_batch(self, monkeypatch):
        """Application scanner can source candidates from the full-symbol universe."""
        import src.engine_scanner as engine_module

        ib = _mock_ib()
        engine = _make_engine(ib)
        monkeypatch.setattr(engine_module, "APP_SCANNER_SOURCE", "universe")
        monkeypatch.setattr(engine_module, "APP_SCANNER_BATCH_SIZE", 2)
        monkeypatch.setattr(engine_module, "APP_SCANNER_MAX_SYMBOLS", 0)
        monkeypatch.setattr(
            engine_module,
            "load_application_symbol_universe",
            lambda: ["AAPL", "MSFT", "NVDA"],
        )

        assert engine.get_institutional_scan() == ["AAPL", "MSFT"]
        assert engine.get_institutional_scan() == ["NVDA", "AAPL"]
        ib.reqScannerData.assert_not_called()

    def test_hybrid_source_dedupes_ibkr_and_universe_candidates(self, monkeypatch):
        """Hybrid source keeps IBKR scanner hits and walks the broader universe."""
        import src.engine_scanner as engine_module

        ib = _mock_ib()
        engine = _make_engine(ib)
        ib.reqScannerData.return_value = [
            self._make_scan_item("AAPL"),
            self._make_scan_item("MSFT"),
        ]
        monkeypatch.setattr(engine_module, "APP_SCANNER_SOURCE", "hybrid")
        monkeypatch.setattr(engine_module, "APP_SCANNER_BATCH_SIZE", 3)
        monkeypatch.setattr(engine_module, "APP_SCANNER_MAX_SYMBOLS", 0)
        monkeypatch.setattr(
            engine_module,
            "load_application_symbol_universe",
            lambda: ["MSFT", "NVDA", "ASML"],
        )

        assert engine.get_institutional_scan() == ["AAPL", "MSFT", "NVDA", "ASML"]

    def test_prefiltered_cache_becomes_the_day_watchlist(self, monkeypatch):
        """Once the premarket sieve exists, do not re-add raw IBKR/universe symbols."""
        import src.engine_scanner as engine_module

        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._prefilter_date = "2024-06-05"
        engine._prefilter_status = "complete"
        engine._prefilter_candidates = ["AAPL", "MSFT"]
        monkeypatch.setattr(engine_module, "APP_SCANNER_SOURCE", "hybrid")
        fake_now = pytz.timezone('US/Eastern').localize(datetime(2024, 6, 5, 10, 30))

        with patch("src.engine_scanner.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            result = engine.get_institutional_scan()

        assert result == ["AAPL", "MSFT"]
        ib.reqScannerData.assert_not_called()

    def test_indicator_swing_static_prefilter_rejects_impossible_ma_sleeve(self):
        from src.engine import VelocityEngine
        from src.strategy_profiles import get_strategy_profile

        profile = get_strategy_profile("indicator_swing")
        ctx = {
            "volume": 2_000_000,
            "dollar_vol_20d": 150_000_000,
            "ma50": 120.0,
            "ma200": 100.0,
            "sma200_slope": 1.0,
            "weekly_ma10_gt_ma30": True,
            "ema20_gt_sma50": False,
            "ma_bull_cross": False,
            "rsi": 58.0,
            "rsi_prev": 55.0,
            "macd_hist_delta": 0.1,
            "obv_uptrend": True,
            "stoch_bull_exit_oversold": False,
            "psar_bull_3": False,
        }

        failures = VelocityEngine._prefilter_static_failures(ctx, profile)

        assert "no_possible_indicator_sleeve" in failures

    def test_premarket_prefilter_writes_candidate_cache(self, tmp_path, monkeypatch):
        import json
        import src.engine_scanner as engine_module

        ib = _mock_ib()
        engine = _make_engine(ib)
        cache_file = tmp_path / "prefilter.json"
        monkeypatch.setattr(engine_module, "APP_PREFILTER_CACHE_FILE", str(cache_file))
        monkeypatch.setattr(engine_module, "APP_SCANNER_MAX_SYMBOLS", 0)
        monkeypatch.setattr(engine_module, "APP_PREFILTER_HISTORY_SLEEP_SEC", 0.0)
        monkeypatch.setattr(engine_module, "APP_PREFILTER_PROGRESS_EVERY", 100)
        monkeypatch.setattr(engine_module, "APP_PREFILTER_STOP_AT_ENTRY_START", False)
        monkeypatch.setattr(
            engine_module,
            "load_application_symbol_universe",
            lambda: ["PASS", "FAIL", "PASS"],
        )

        def fake_prefilter(sym, _profile, _today):
            if sym == "PASS":
                return True, (), ("volume_pace>=1.2x",)
            return False, ("MA50<=MA200",), ()

        with patch.object(engine, "_prefilter_symbol", side_effect=fake_prefilter), \
             patch.object(engine, "_write_dashboard_data"):
            payload = engine._run_premarket_universe_prefilter()

        assert payload["status"] == "complete"
        assert payload["candidates"] == ["PASS"]
        assert payload["stats"]["processed"] == 2
        assert payload["stats"]["candidates"] == 1
        saved = json.loads(cache_file.read_text())
        assert saved["candidates"] == ["PASS"]
        assert saved["deferred_rules"]["PASS"] == ["volume_pace>=1.2x"]
        assert saved["rejections_by_symbol"]["FAIL"] == ["MA50<=MA200"]

    def test_premarket_prefilter_stops_with_partial_cache_at_entry_window(self, tmp_path, monkeypatch):
        import json
        import src.engine_scanner as engine_module

        ib = _mock_ib()
        engine = _make_engine(ib)
        cache_file = tmp_path / "prefilter.json"
        monkeypatch.setattr(engine_module, "APP_PREFILTER_CACHE_FILE", str(cache_file))
        monkeypatch.setattr(engine_module, "APP_SCANNER_MAX_SYMBOLS", 0)
        monkeypatch.setattr(engine_module, "APP_PREFILTER_HISTORY_SLEEP_SEC", 0.0)
        monkeypatch.setattr(engine_module, "APP_PREFILTER_PROGRESS_EVERY", 100)
        monkeypatch.setattr(engine_module, "APP_PREFILTER_STOP_AT_ENTRY_START", True)
        monkeypatch.setattr(
            engine_module,
            "load_application_symbol_universe",
            lambda: ["PASS", "LATE"],
        )

        entry_h, entry_m = engine_module.ENTRY_START
        tz_ny = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, entry_h, entry_m, 0))

        def fake_prefilter(sym, _profile, _today):
            return True, (), ("volume_pace>=1.2x",)

        with patch.object(engine, "_prefilter_symbol", side_effect=fake_prefilter), \
             patch.object(engine, "_write_dashboard_data"), \
             patch.object(engine_module, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            payload = engine._run_premarket_universe_prefilter()

        assert payload["status"] == "partial"
        assert payload["stopped_reason"] == "entry_window_open"
        assert payload["candidates"] == ["PASS"]
        assert payload["stats"]["processed"] == 1
        assert engine._last_premarket_prefilter_date == "2024-06-05"
        saved = json.loads(cache_file.read_text())
        assert saved["status"] == "partial"
        assert saved["processed_symbols"] == ["PASS"]


class TestCashBucketBuffer:
    def test_deployable_settled_cash_keeps_configured_buffer(self):
        from src.config import SETTLED_CASH_DEPLOYMENT_PCT

        engine = _make_engine(_mock_ib())

        assert SETTLED_CASH_DEPLOYMENT_PCT == pytest.approx(0.95)
        assert engine._deployable_settled_cash(1000.0) == pytest.approx(950.0)

    def test_cash_entry_slots_use_deployable_settled_cash(self):
        engine = _make_engine(_mock_ib())

        # Below MIN_BUCKET_FLOOR (150): settled=$100 → deployable=95 < 150 → 0 slots
        assert engine._calc_cash_entry_slots(100.0) == 0
        # Between floor and MIN_BUCKET_SIZE: settled=$500 → deployable=$475, which
        # is below MIN_BUCKET_SIZE (500) but above MIN_BUCKET_FLOOR (150) → 1 slot
        # (prevents small accounts from being permanently frozen on T+1 settlement).
        assert engine._calc_cash_entry_slots(500.0) == 1
        # Above MIN_BUCKET_SIZE: settled=$530 → deployable≈$503 ≥ $500 → 1 full slot
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
# 3. SCORING SYSTEM — indicator_swing only
# ─────────────────────────────────────────────────────────────────────────────

class TestIndicatorSwingScoring:
    """Shared scorer checks for the maintained ranking model."""

    def test_live_volume_pace_normalizes_early_session_volume(self):
        tz_ny = pytz.timezone("US/Eastern")
        now = tz_ny.localize(datetime(2026, 6, 1, 10, 0))

        # 100k shares in the first 30 of 390 regular-session minutes is
        # running near a 1.3x full-day pace against a 1M-share average day.
        assert volume_pace_from_intraday(100_000, 1_000_000, now) == pytest.approx(1.3)

    def test_indicator_scorer_uses_volume_pace_when_available(self):
        ctx = _ctx(rvol=1.0, volume_pace=1.0)
        ctx["volume_pace"] = 5.0

        with_pace = score_candidate(ctx, model="indicator_swing", volume_floor=1.0)
        raw_only = score_candidate({**ctx, "volume_pace": 1.0}, model="indicator_swing", volume_floor=1.0)

        assert with_pace > raw_only

    def test_indicator_score_rewards_relative_strength_leader(self):
        leader = _ctx(relative_strength_63d=0.22, relative_strength_126d=0.30)
        laggard = _ctx(relative_strength_63d=0.02, relative_strength_126d=0.03)

        assert score_candidate(leader, model="indicator_swing") > score_candidate(laggard, model="indicator_swing")

    def test_indicator_score_prefers_ma_cross_over_psar_when_other_inputs_match(self):
        ma_cross = _ctx(entry_strategy="ma_cross")
        psar = _ctx(entry_strategy="psar_flip")

        assert score_candidate(ma_cross, model="indicator_swing") > score_candidate(psar, model="indicator_swing")

    def test_analyst_rating_adjusts_score_within_bounds(self):
        bullish = score_candidate(_ctx(analyst_rating_raw_score=1.0, analyst_rating_total=10))
        bearish = score_candidate(_ctx(analyst_rating_raw_score=-1.0, analyst_rating_total=10))

        assert bullish > bearish
        assert 0.0 <= bearish <= bullish <= 100.0

    def test_unknown_scoring_model_is_rejected(self):
        with pytest.raises(ValueError, match="Valid model: indicator_swing"):
            score_candidate(_ctx(), model="legacy")


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
             _freeze_all_datetimes(fake_now):
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
             _freeze_all_datetimes(fake_now):
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
             _freeze_all_datetimes(fake_now):
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
             _freeze_all_datetimes(fake_now):
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
             _freeze_all_datetimes(fake_now):
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
             _freeze_all_datetimes(fake_now):
            engine.run_cycle()

    def test_low_dollar_volume_is_cached_for_rest_of_day(self):
        from src.strategy_profiles import get_strategy_profile

        ib = _mock_ib()
        engine = _make_engine(ib)
        dollar_floor = get_strategy_profile("indicator_swing").min_dollar_vol
        ctx = _ctx(
            price=101.0,
            orb=100.0,
            dollar_vol=dollar_floor - 1,
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

    def test_scan_summary_reports_filter_reasons_not_non_orderable(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine.state = {'HELD': {'price': 100, 'qty': 1}}
        low_rsi_ctx = _ctx(
            price=101.0,
            orb=100.0,
            rsi=50.0,
            rsi_prev=49.0,
        )
        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        def _ctx_for(sym):
            return {
                'NODATA': None,
                'LOWRSI': low_rsi_ctx,
            }.get(sym)

        with patch.object(engine, 'get_institutional_scan', return_value=['HELD', 'NODATA', 'LOWRSI']), \
             patch.object(engine, 'get_technical_context', side_effect=_ctx_for), \
             patch.object(engine, 'manage_position_exits'), \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.logger') as mock_logger, \
             _freeze_all_datetimes(fake_now):
            engine.run_cycle()

        summary_lines = [
            call_args.args[0]
            for call_args in mock_logger.info.call_args_list
            if call_args.args and str(call_args.args[0]).startswith('SCAN SUMMARY:')
        ]
        assert summary_lines
        summary = summary_lines[-1]
        assert 'eligible_signals=0' in summary
        assert 'filtered=3' in summary
        assert 'already_held=1' in summary
        assert 'no_technical_data=1' in summary
        assert 'entry_filter=1' in summary
        assert 'non_orderable' not in summary

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

    def test_indicator_swing_context_skips_orb_historical_request(self):
        from src.config import DAILY_LOOKBACK, DAILY_BAR_SIZE
        from src.strategy_profiles import get_strategy_profile

        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._strategy_profile = get_strategy_profile("indicator_swing")
        daily_bars = [MagicMock()] * 260
        ib.reqHistoricalData.return_value = daily_bars

        idx = pd.date_range("2025-01-01", periods=260, freq="B")
        close = np.linspace(100.0, 160.0, len(idx))
        daily_df = pd.DataFrame({
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(idx), 3_000_000),
        }, index=idx)
        spy_df = daily_df.copy()
        ticker = _mock_price_ticker(161.0, open=158.0, high=162.0, low=157.0)
        ticker.volume = 2_000_000
        ib.reqTickers.return_value = [ticker]

        with patch('src.engine.util.df', return_value=daily_df), \
             patch.object(engine, '_fetch_spy_daily_frame', return_value=spy_df), \
             patch.object(engine, '_analyst_context', return_value={}):
            ctx = engine.get_technical_context('AAPL')

        assert ctx is not None
        assert ctx['day_open'] == pytest.approx(158.0)
        assert ib.reqHistoricalData.call_count == 1
        args = ib.reqHistoricalData.call_args[0]
        assert args[1] == ''
        assert args[2] == DAILY_LOOKBACK
        assert args[3] == DAILY_BAR_SIZE


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
             patch('src.engine_orders.datetime') as mock_dt:
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
             patch('src.engine_orders.datetime') as mock_dt:
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
             patch('src.engine_orders.datetime') as mock_dt:
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
             patch('src.engine_orders.datetime') as mock_dt:
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
             patch('src.engine_orders.datetime') as mock_dt:
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
             _freeze_all_datetimes(fake_now):
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
# 9. EXIT ORDERS — EOD quality cleanup, liquidation
# ─────────────────────────────────────────────────────────────────────────────

class TestExitOrders:
    """
    Verify EOD quality cleanup (manage_position_exits → liquidate) behaviour:
    - MarketOrder('SELL', position) placed with exact qty reported by IBKR
    - Open symbol orders are cancelled before the market sell
    - Cash-account exits cancel protective SELLs first to avoid oversell rejection
    - MarketOrder TIF is explicit DAY so IBKR presets cannot override it to GTC
    - Exit fires near the close when positions fail the EOD hold-quality gate
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
                'volume': 0, 'score': 50,
                'protection_status': 'confirmed'}

    def _et(self, year, month, day, hour=10, minute=30):
        return pytz.timezone('US/Eastern').localize(
            datetime(year, month, day, hour, minute)
        )

    def _run_exit_check(self, engine, now=None):
        now = now or self._et(2024, 6, 5, 15, 50)
        with patch('src.engine_exits.datetime') as mock_dt:
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

    def test_eod_quality_cleanup_is_disabled_for_indicator_swing_profile(self):
        """Default indicator_swing positions are not churned by EOD quality cleanup."""
        ib     = _mock_ib()
        engine = _make_engine(ib)

        entry_price   = 100.0
        stagnant_price = 102.0
        ib.reqTickers.return_value = [
            _mock_price_ticker(stagnant_price, open=100.0, high=110.0, low=100.0, vwap=101.0)
        ]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('SLOW', 5.0)]
        engine.state = {'SLOW': self._make_state_entry(price=entry_price,
                                                        entry_time=self._et(2024, 6, 3).isoformat())}

        self._run_exit_check(engine)

        assert 'pending_exit' not in engine.state['SLOW']
        assert not ib.placeOrder.called

    def test_eod_quality_cleanup_waits_until_configured_eod_time(self):
        """Held weak positions must not be sold before 15:50 ET."""
        ib     = _mock_ib()
        engine = _make_engine(ib)

        entry_price = 100.0
        stagnant_price = 102.0
        ib.reqTickers.return_value = [
            _mock_price_ticker(stagnant_price, open=100.0, high=110.0, low=100.0, vwap=101.0)
        ]
        engine.state = {'SLOW': self._make_state_entry(
            price=entry_price,
            entry_time=self._et(2024, 6, 3).isoformat(),
        )}

        self._run_exit_check(engine, now=self._et(2024, 6, 5, 15, 49))

        assert 'pending_exit' not in engine.state['SLOW']
        assert not ib.placeOrder.called

    def test_pending_exit_blocks_duplicate_sell_on_next_cycle(self):
        """A position with an in-flight sell must not submit another market sell."""
        ib     = _mock_ib()
        engine = _make_engine(ib)

        entry_price = 100.0
        stagnant_price = 102.0
        ib.reqTickers.return_value = [
            _mock_price_ticker(stagnant_price, open=100.0, high=110.0, low=100.0, vwap=101.0)
        ]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('SLOW', 5.0)]
        entry = self._make_state_entry(
            price=entry_price,
            entry_time=self._et(2024, 6, 3).isoformat(),
        )
        entry['pending_exit'] = True
        engine.state = {'SLOW': entry}

        self._run_exit_check(engine)

        assert not ib.placeOrder.called
        assert engine.state['SLOW']['pending_exit'] is True

    def test_eod_quality_cleanup_does_not_trigger_when_hold_quality_passes(self):
        """Position held through the window with strong EOD quality → kept."""
        ib     = _mock_ib()
        engine = _make_engine(ib)

        entry_price    = 100.0
        profit_price   = 106.0
        ib.reqTickers.return_value = [
            _mock_price_ticker(profit_price, open=100.0, high=107.0, low=100.0, vwap=104.0)
        ]
        engine.state = {'WINNER': self._make_state_entry(price=entry_price,
                                                          entry_time=self._et(2024, 6, 3).isoformat())}

        self._run_exit_check(engine)

        assert 'WINNER' in engine.state, "Profitable position must NOT be liquidated"
        assert not ib.placeOrder.called

    def test_eod_quality_cleanup_does_not_churn_same_day_weak_position(self):
        """Same-day weak positions are left to swing exits and broker stops."""
        ib     = _mock_ib()
        engine = _make_engine(ib)
        engine.state = {'NEW': self._make_state_entry(
            entry_time=self._et(2024, 6, 5, 10, 0).isoformat()
        )}
        ib.reqTickers.return_value = [_mock_price_ticker(99.0, open=100.0, high=101.0, low=98.0)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('NEW', 1.0)]

        self._run_exit_check(engine, now=self._et(2024, 6, 5, 15, 50))

        assert 'pending_exit' not in engine.state['NEW']
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
    - Entry price = 0 in EOD exit management → skipped, no division by zero
    - VIX at threshold (=35) → entries allowed; above (>35) → blocked
    - Strict comparisons: price > ORB, RSI strictly rising
    - Friday dollar-volume threshold doubled
    - IBKR sync adds missing position and warns when avgCost missing
    - Indicator edge: RSI with all-gain period, flat bars
    """

    # ── State persistence ────────────────────────────────────────────────────

    def _bare_engine(self):
        """Minimal VelocityEngine via __new__ with only the attrs needed for load_state."""
        import src.engine as eng_mod
        engine = eng_mod.VelocityEngine.__new__(eng_mod.VelocityEngine)
        engine._alert_dedup_cache = {}
        engine._ib_error_dedup = {}
        engine._log_once_cache = {}
        engine._indicator_row_cache = {}
        engine._health_metrics = {}
        return engine

    def test_load_state_returns_empty_on_corrupt_json(self, tmp_path):
        """Corrupt STATE_FILE must not crash the engine — returns empty dict."""
        import src.engine as eng_mod
        state_path = tmp_path / "engine_state.json"
        state_path.write_text("{not valid json!!!")  # corrupted

        original = eng_mod.STATE_FILE
        eng_mod.STATE_FILE = str(state_path)
        try:
            engine = self._bare_engine()
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
            engine = self._bare_engine()
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

    def test_score_candidate_with_ma200_zero_stays_bounded(self):
        """The indicator scorer must stay finite and bounded when MA200=0."""
        from src.engine import VelocityEngine
        engine = VelocityEngine.__new__(VelocityEngine)
        ctx_zero_ma = _ctx(price=110.0, orb=100.0, rsi=65.0, rsi_prev=60.0,
                           ma50=105.0, ma200=0.0)
        ctx_normal  = _ctx(price=110.0, orb=100.0, rsi=65.0, rsi_prev=60.0,
                           ma50=105.0, ma200=105.0)   # equal MAs → trend_sep=0 → trend=0
        score_zero  = engine._score_candidate(ctx_zero_ma)
        score_flat  = engine._score_candidate(ctx_normal)
        assert 0.0 <= score_zero <= 100.0
        assert 0.0 <= score_flat <= 100.0

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
             _freeze_all_datetimes(fake_now):
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

    def test_reprice_check_blocks_below_minimum_price(self):
        """Live reprice must still enforce the production minimum price."""
        from src.config import SCAN_MIN_PRICE

        engine = _make_engine(_mock_ib())
        ctx = _ctx(price=SCAN_MIN_PRICE - 0.50, orb=SCAN_MIN_PRICE - 1.00,
                   ma50=95.0, ma200=85.0)

        assert not engine._entry_price_is_still_valid(
            "LOWP", ctx, SCAN_MIN_PRICE - 0.50,
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

    def test_sync_avgcost_zero_position_skips_eod_profit_cleanup_check(self):
        """Position synced with avgCost=0 must not crash EOD exit management."""
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

        stop_trade = MagicMock()
        stop_trade.order.orderId = 2

        # side_effect lets the first call (BUY) return a proper filled mock, and the
        # second call (TRAIL stop) return a separate mock — preventing the partial-fill
        # path from firing because a MagicMock's __float__ returns 1.0 ≠ qty.
        def _place_order_side_effect(contract, order):
            if order.action == 'BUY':
                t = MagicMock()
                t.order.orderId = 1
                t.orderStatus.status = 'Filled'
                t.orderStatus.filled = float(order.totalQuantity)
                t.orderStatus.avgFillPrice = 101.0
                t.fills = [_mock_fill(1.0)]
                return t
            return stop_trade
        ib.placeOrder.side_effect = _place_order_side_effect

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'get_institutional_scan', return_value=['TSLA']), \
             patch.object(engine, 'get_technical_context', return_value=passing_ctx), \
             patch.object(engine, '_update_position_prices'), \
             _freeze_all_datetimes(fake_now):
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

    def test_analyst_downgrade_exit_requires_price_confirmation(self):
        """Bearish analyst score alone is not enough when price action is still healthy."""
        from src.strategy_profiles import get_strategy_profile
        ib     = _mock_ib()
        engine = _make_engine(ib)
        engine._strategy_profile = get_strategy_profile("indicator_swing")
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 101.0
        engine.state = {'DOWN': self._state_entry(entry, cur, tz_ny)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('DOWN', 5.0)]

        with patch.object(engine, '_analyst_context', return_value={
            'analyst_rating_score': -0.50,
            'analyst_rating_total': 12,
        }):
            self._run_exit_check(engine)

        assert 'pending_exit' not in engine.state['DOWN']
        assert not ib.placeOrder.called

    def test_analyst_downgrade_exit_triggers_when_price_is_weak(self):
        """Bearish analyst score can exit only after price confirms weakness."""
        from src.strategy_profiles import get_strategy_profile
        ib     = _mock_ib()
        engine = _make_engine(ib)
        engine._strategy_profile = get_strategy_profile("indicator_swing")
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 99.0
        engine.state = {'DOWN': self._state_entry(entry, cur, tz_ny)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('DOWN', 5.0)]

        with patch.object(engine, '_analyst_context', return_value={
            'analyst_rating_score': -0.50,
            'analyst_rating_total': 12,
        }):
            self._run_exit_check(engine)

        assert engine.state['DOWN']['pending_exit'] is True
        assert ib.placeOrder.called


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
                'volume': 0, 'score': 60,
                'protection_status': 'confirmed',
                'time': datetime.now(tz_ny).isoformat()}

    def test_friday_close_disabled_for_indicator_swing_profile(self):
        """The maintained profile does not force Friday liquidation by default."""
        from src.config import FRIDAY_CLOSE_HOUR, FRIDAY_MIN_PROFIT_PCT
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = round(entry * (1 + min(FRIDAY_MIN_PROFIT_PCT, 0.02) - 0.005), 2)
        engine.state = {'FRI': self._state_entry(entry, cur, tz_ny)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('FRI', 5.0)]

        friday_after = tz_ny.localize(datetime(2024, 6, 7, FRIDAY_CLOSE_HOUR + 1, 0))

        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = friday_after
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.manage_position_exits()

        assert 'pending_exit' not in engine.state['FRI']
        assert not ib.placeOrder.called

    def test_friday_close_does_not_trigger_above_threshold(self):
        """Profit above Friday threshold plus EOD quality must keep position open."""
        from src.config import FRIDAY_CLOSE_HOUR
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 106.0
        engine.state = {'FRI': self._state_entry(entry, cur, tz_ny)}
        ib.reqTickers.return_value = [
            _mock_price_ticker(cur, open=100.0, high=107.0, low=100.0, vwap=104.0)
        ]

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
# 15. EOD QUALITY CLEANUP — liquidate weak positions near the close
# ─────────────────────────────────────────────────────────────────────────────

class TestEodFlat:
    """
    After EOD_EXIT_TIME (default 15:50 ET) on any trading day, positions that
    fail the hold-quality gate may be liquidated, including same-day entries.
    The rule fires at most once per calendar trading day.
    """

    def _state_entry(self, entry, cur, tz_ny, entry_time=None):
        return {
            'price': entry, 'qty': 5.0, 'current_price': cur,
            'stop_loss': entry * 0.93,
            'volume': 0, 'score': 60,
            'protection_status': 'confirmed',
            'time': (entry_time or datetime.now(tz_ny)).isoformat(),
        }

    def _make_position(self, symbol, qty):
        pos = MagicMock()
        pos.contract.symbol = symbol
        pos.position        = qty
        return pos

    def test_eod_flat_does_not_churn_older_position_at_loss(self):
        """Default profile leaves older losing positions to swing exits/stops."""
        from src.config import EOD_EXIT_TIME
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 98.0   # -2%, not in profit
        old_entry = tz_ny.localize(datetime(2024, 6, 4, 10, 30))
        engine.state = {'LOSS': self._state_entry(entry, cur, tz_ny, old_entry)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur, open=100.0, high=101.0, low=98.0)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('LOSS', 5.0)]

        # Wednesday at EOD_EXIT_TIME + 5 min
        eod_time = tz_ny.localize(
            datetime(2024, 6, 5, EOD_EXIT_TIME[0], EOD_EXIT_TIME[1] + 5)
        )
        with patch('src.engine_exits.datetime') as mock_dt:
            mock_dt.now.return_value = eod_time
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()

        assert 'pending_exit' not in engine.state['LOSS']
        assert not ib.placeOrder.called

    def test_eod_flat_does_not_churn_older_profitable_position_closing_weak(self):
        """Default profile does not use EOD quality cleanup."""
        from src.config import EOD_EXIT_TIME
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 101.0
        old_entry = tz_ny.localize(datetime(2024, 6, 4, 10, 30))
        engine.state = {'WEAK': self._state_entry(entry, cur, tz_ny, old_entry)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur, open=100.0, high=110.0, low=100.0)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('WEAK', 5.0)]

        eod_time = tz_ny.localize(
            datetime(2024, 6, 5, EOD_EXIT_TIME[0], EOD_EXIT_TIME[1] + 5)
        )
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = eod_time
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()

        assert 'pending_exit' not in engine.state['WEAK']
        assert not ib.placeOrder.called

    def test_eod_flat_does_not_churn_older_position_exactly_at_entry(self):
        """Default profile does not close zero-profit positions solely for EOD cleanup."""
        from src.config import EOD_EXIT_TIME
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 100.0  # exactly at entry, profit = 0
        old_entry = tz_ny.localize(datetime(2024, 6, 4, 10, 30))
        engine.state = {'FLAT': self._state_entry(entry, cur, tz_ny, old_entry)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur, open=100.0, high=105.0, low=99.0)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('FLAT', 5.0)]

        eod_time = tz_ny.localize(
            datetime(2024, 6, 5, EOD_EXIT_TIME[0], EOD_EXIT_TIME[1] + 5)
        )
        with patch('src.engine_exits.datetime') as mock_dt:
            mock_dt.now.return_value = eod_time
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()

        assert 'pending_exit' not in engine.state['FLAT']
        assert not ib.placeOrder.called

    def test_eod_flat_does_not_trigger_when_hold_quality_passes(self):
        """A strong older winner after EOD_EXIT_TIME must not be closed."""
        from src.config import EOD_EXIT_TIME
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 101.0
        old_entry = tz_ny.localize(datetime(2024, 6, 4, 10, 30))
        engine.state = {'GAIN': self._state_entry(entry, cur, tz_ny, old_entry)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur, open=100.0, high=107.0, low=100.0, vwap=104.0)]

        eod_time = tz_ny.localize(
            datetime(2024, 6, 5, EOD_EXIT_TIME[0], EOD_EXIT_TIME[1] + 5)
        )
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = eod_time
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()

        assert 'GAIN' in engine.state, "Strong profitable position must not be closed at EOD"
        assert not ib.placeOrder.called

    def test_eod_flat_does_not_liquidate_when_stop_is_not_confirmed_by_default(self):
        """Stop confirmation is not an EOD cleanup gate when cleanup is disabled."""
        from src.config import EOD_EXIT_TIME
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 101.0
        old_entry = tz_ny.localize(datetime(2024, 6, 4, 10, 30))
        state = self._state_entry(entry, cur, tz_ny, old_entry)
        state['protection_status'] = 'unconfirmed'
        engine.state = {'NO_STOP': state}
        ib.reqTickers.return_value = [
            _mock_price_ticker(cur, open=100.0, high=107.0, low=100.0, vwap=104.0)
        ]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('NO_STOP', 5.0)]

        eod_time = tz_ny.localize(
            datetime(2024, 6, 5, EOD_EXIT_TIME[0], EOD_EXIT_TIME[1] + 5)
        )
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = eod_time
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()

        assert 'pending_exit' not in engine.state['NO_STOP']
        assert not ib.placeOrder.called

    def test_eod_flat_does_not_close_same_day_weak_position(self):
        """Same-day weak position is not closed by disabled EOD cleanup."""
        from src.config import EOD_EXIT_TIME
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 98.0
        same_day_entry = tz_ny.localize(datetime(2024, 6, 5, 11, 20))
        engine.state = {'NEW': self._state_entry(entry, cur, tz_ny, same_day_entry)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur, open=100.0, high=101.0, low=98.0)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('NEW', 5.0)]

        eod_time = tz_ny.localize(
            datetime(2024, 6, 5, EOD_EXIT_TIME[0], EOD_EXIT_TIME[1] + 5)
        )
        with patch('src.engine_exits.datetime') as mock_dt:
            mock_dt.now.return_value = eod_time
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()

        assert 'pending_exit' not in engine.state['NEW']
        assert not ib.placeOrder.called

    def test_eod_flat_does_not_trigger_before_eod_time(self):
        """Before EOD_EXIT_TIME the EOD cleanup rule must be inactive."""
        from src.config import EOD_EXIT_TIME
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 104.0   # +4%, would trigger if time were right
        old_entry = tz_ny.localize(datetime(2024, 6, 4, 10, 0))
        engine.state = {'EARLY': self._state_entry(entry, cur, tz_ny, old_entry)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur, open=100.0, high=110.0, low=100.0)]

        # One minute before EOD_EXIT_TIME
        before_eod = tz_ny.localize(
            datetime(2024, 6, 5, EOD_EXIT_TIME[0], EOD_EXIT_TIME[1] - 1)
        )
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = before_eod
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()

        assert 'EARLY' in engine.state, "EOD cleanup must not trigger before EOD_EXIT_TIME"
        assert not ib.placeOrder.called

    def test_eod_flat_fires_only_once_per_day(self):
        """EOD cleanup must not re-liquidate on the second cycle of the same day."""
        from src.config import EOD_EXIT_TIME
        ib     = _mock_ib()
        engine = _make_engine(ib)
        tz_ny  = pytz.timezone('US/Eastern')

        entry = 100.0
        cur   = 98.0
        old_entry = tz_ny.localize(datetime(2024, 6, 4, 10, 30))
        engine.state = {'ONCE': self._state_entry(entry, cur, tz_ny, old_entry)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur, open=100.0, high=101.0, low=98.0)]
        ib.openTrades.return_value = []
        ib.positions.return_value  = [self._make_position('ONCE', 5.0)]

        eod_time = tz_ny.localize(
            datetime(2024, 6, 5, EOD_EXIT_TIME[0], EOD_EXIT_TIME[1] + 5)
        )
        with patch('src.engine_exits.datetime') as mock_dt:
            mock_dt.now.return_value = eod_time
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()   # first call — fires

        # Simulate position partially cleared, then second call same day
        same_day_entry = tz_ny.localize(datetime(2024, 6, 5, 11, 0))
        engine.state['ONCE'] = self._state_entry(entry, cur, tz_ny, same_day_entry)
        place_count_after_first = ib.placeOrder.call_count

        with patch('src.engine_exits.datetime') as mock_dt:
            mock_dt.now.return_value = eod_time
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.manage_position_exits()   # second call same day — must not re-fire

        assert ib.placeOrder.call_count == place_count_after_first, \
            "EOD cleanup must not re-fire on the second cycle of the same trading day"


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
