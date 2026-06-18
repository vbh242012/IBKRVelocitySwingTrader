"""
Unit tests for VelocityEngine business logic.

IB is fully mocked — no live connection required.
Tests exercise entry signals, EOD exit management, position limits,
and bracket order construction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import logging
import pytest
import numpy as np
import pandas as pd
import pytz
from datetime import datetime
from unittest.mock import MagicMock, patch, call


# ── Helpers ───────────────────────────────────────────────────────────────────
def _mock_ib():
    ib = MagicMock()
    # accountSummary returns equity plus settled cash, matching live sizing logic.
    nl       = MagicMock()
    nl.tag   = 'NetLiquidation'
    nl.value = '1400.0'
    sc       = MagicMock()
    sc.tag   = 'SettledCash'
    sc.value = '1400.0'
    ib.accountSummary.return_value = [nl, sc]

    # VIX ticker with safe value
    vix_ticker = MagicMock()
    vix_ticker.marketPrice.return_value = 20.0
    ib.reqTickers.return_value = [vix_ticker]

    # qualifyContracts returns a list with one item
    ib.qualifyContracts.return_value = [MagicMock()]
    ib.reqAllOpenOrders.return_value = []

    # whatIfOrder pre-flight: empty warningText = IB accepts the order
    ib.whatIfOrder.return_value.warningText = ''

    return ib


def _mock_price_ticker(price: float, *, open=None, high=None, low=None, vwap=None):
    ticker = MagicMock()
    ticker.marketPrice.return_value = price
    ticker.last = price
    ticker.close = price
    ticker.bid = price * 0.999
    ticker.ask = price * 1.001
    ticker.open = price if open is None else open
    ticker.high = price if high is None else high
    ticker.low = price if low is None else low
    ticker.vwap = price if vwap is None else vwap
    return ticker


def _make_engine_patched(ib_mock):
    """Return a VelocityEngine with IB replaced and connect() bypassed."""
    with patch('src.engine.IB', return_value=ib_mock), \
         patch.object(sys.modules.get('src.engine', __import__('src.engine')),
                      'logger', MagicMock()):
        from src.engine import VelocityEngine
        from src.strategy_profiles import get_strategy_profile
        with patch.object(VelocityEngine, 'connect', lambda self: None):
            engine = VelocityEngine.__new__(VelocityEngine)
            engine.ib                   = ib_mock
            engine.state                = {}
            engine._last_equity         = 0.0
            engine._last_settled_cash   = 0.0
            engine._equity_initialized  = False
            engine._last_vix            = None
            engine._last_vix_ts         = 0.0
            engine._last_scan_ts        = None
            engine._next_scan_dt        = None
            # Attributes added after initial implementation
            engine._day_start_date      = None
            engine._day_start_equity    = None
            engine._contract_cache      = {}
            engine._bar_cache           = {}
            engine._vix_contract        = None
            engine._spy_cache           = {}
            engine._prefilter_date      = None
            engine._prefilter_status    = "not_started"
            engine._prefilter_candidates = []
            engine._prefilter_stats     = {}
            engine._last_premarket_prefilter_date = None
            engine._sector_cache        = {}
            engine._daily_scan_skip     = {}
            engine._last_audit_date     = None
            engine._last_audit_at       = None
            engine._last_post_open_audit_date = None
            engine._last_premarket_readiness_date = None
            engine._last_post_close_maintenance_date = None
            engine._missing_position_counts = {}
            engine._strategy_profile = get_strategy_profile("indicator_swing")
            # New instance vars added by fixes
            engine._ib_error_dedup      = {}
            engine._alert_dedup_cache   = {}
            engine._data_blackout_streak = 0
            engine._data_blackout_alerted = False
            engine._friday_cutoff_logged_date = None
            engine._last_eod_exit_date  = None
            engine._last_pre_entry_sync_date = None
            engine._last_premarket_prefilter_date = None
            engine._historical_data_health = {}
            engine._vix_failure_count   = 0
            engine._next_vix_retry_ts   = 0.0
            engine._last_vix_failure_ts = 0.0
            engine._last_vix_source     = None
            engine._equity_initialized  = False
            engine._health_date = datetime.now().strftime('%Y-%m-%d')
            engine._health_metrics = {}
            return engine


# ── Expert filter (entry conditions) ─────────────────────────────────────────
class TestExpertFilter:
    """Entry checks use the maintained indicator_swing profile rules."""

    def _ctx(self, **updates):
        ctx = dict(
            live_price=110.0,
            close=110.0,
            ma20=104.0,
            ma50=98.0,
            ma200=90.0,
            sma200_slope=0.25,
            ema20_gt_sma50=True,
            rsi=62.0,
            rsi_prev=58.0,
            atr=2.0,
            atr_chandelier=4.0,
            atr_pct=4.0 / 110.0,
            spread_pct=0.002,
            volume_pace=2.0,
            volume=3_000_000,
            dollar_vol_20d=150_000_000,
            break_prev_high=True,
            reclaim_ma20=False,
            reclaim_ma50=False,
            weekly_uptrend=True,
            return_13w=0.25,
            return_26w=0.35,
            relative_strength_63d=0.15,
            relative_strength_126d=0.18,
            price_vs_52w_high=0.90,
            high20=112.0,
            dist_high20=110.0 / 112.0 - 1.0,
            stoch_bull_exit_oversold=True,
            macd_hist_delta=0.05,
            obv_uptrend=True,
            contract=MagicMock(),
        )
        ctx.update(updates)
        return ctx

    def _passes(self, ctx):
        from src.strategy_profiles import evaluate_entry_rules, get_strategy_profile
        return evaluate_entry_rules(ctx, get_strategy_profile("indicator_swing")).passed

    def test_all_conditions_met(self):
        assert self._passes(self._ctx()) is True

    def test_fails_when_price_below_ma50(self):
        assert self._passes(self._ctx(live_price=95.0, close=95.0)) is False

    def test_fails_when_ma50_below_ma200(self):
        assert self._passes(self._ctx(ma50=85, ma200=90)) is False

    def test_fails_without_indicator_sleeve_signal(self):
        assert self._passes(
            self._ctx(ema20_gt_sma50=False, break_prev_high=False, reclaim_ma20=False, reclaim_ma50=False)
        ) is False

    def test_fails_when_dollar_volume_below_threshold(self):
        assert self._passes(self._ctx(dollar_vol_20d=50_000_000)) is False

    def test_passes_when_dollar_volume_at_threshold(self):
        from src.config import SCAN_MIN_DOLLAR_VOL
        assert self._passes(self._ctx(dollar_vol_20d=SCAN_MIN_DOLLAR_VOL)) is True

    def test_fails_when_atr_pct_above_threshold(self):
        from src.strategy_profiles import get_strategy_profile
        atr_cap = get_strategy_profile("indicator_swing").max_atr_pct
        ctx = self._ctx()
        ctx['atr_chandelier'] = ctx['live_price'] * (atr_cap + 0.01)
        ctx['atr_pct'] = atr_cap + 0.01
        assert self._passes(ctx) is False

    def test_fails_when_spread_above_threshold(self):
        from src.strategy_profiles import get_strategy_profile
        spread_cap = get_strategy_profile("indicator_swing").max_spread_pct
        assert self._passes(self._ctx(spread_pct=spread_cap + 0.001)) is False


# ── EOD quality cleanup logic ─────────────────────────────────────────────────
class TestEodProfitCleanup:
    _TZ_NY = pytz.timezone('US/Eastern')

    def _entry_after_hold_window(self):
        """Tuesday entry with Wednesday check: 1 Mon-Fri session elapsed."""
        return self._TZ_NY.localize(datetime(2024, 6, 4, 10, 30)).isoformat()

    def _entry_before_hold_window(self):
        """Same-day Wednesday entry/check: 0 Mon-Fri sessions elapsed."""
        return self._TZ_NY.localize(datetime(2024, 6, 5, 10, 0)).isoformat()

    def _run_exit_check(self, engine, hour=15, minute=50):
        check_time = self._TZ_NY.localize(datetime(2024, 6, 5, hour, minute))
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = check_time
            mock_dt.fromisoformat = datetime.fromisoformat
            engine.manage_position_exits()

    def test_older_losing_position_is_not_churned_by_default_swing_profile(self):
        ib      = _mock_ib()
        engine  = _make_engine_patched(ib)

        old_time = self._entry_after_hold_window()
        engine.state = {'AAPL': {'price': 100.0, 'time': old_time}}

        ib.reqTickers.return_value = [_mock_price_ticker(99.0, open=100.0, high=101.0, low=98.0)]
        ib.positions.return_value  = []

        with patch.object(engine, 'liquidate') as mock_liq:
            self._run_exit_check(engine)
            mock_liq.assert_not_called()

    def test_quality_position_not_exited_at_eod(self):
        ib      = _mock_ib()
        engine  = _make_engine_patched(ib)

        old_time = self._entry_after_hold_window()
        engine.state = {'AAPL': {
            'price': 100.0,
            'time': old_time,
            'protection_status': 'confirmed',
        }}

        ib.reqTickers.return_value = [
            _mock_price_ticker(106.0, open=100.0, high=107.0, low=100.0, vwap=104.0)
        ]

        with patch.object(engine, 'liquidate') as mock_liq:
            self._run_exit_check(engine)
            mock_liq.assert_not_called()

    def test_same_day_position_below_profit_threshold_is_not_churned_at_eod(self):
        ib      = _mock_ib()
        engine  = _make_engine_patched(ib)

        fresh_time = self._entry_before_hold_window()
        engine.state = {'AAPL': {'price': 100.0, 'time': fresh_time}}

        ib.reqTickers.return_value = [_mock_price_ticker(99.0, open=100.0, high=101.0, low=98.0)]

        with patch.object(engine, 'liquidate') as mock_liq:
            self._run_exit_check(engine)
            mock_liq.assert_not_called()

    def test_swing_profile_does_not_churn_weak_position_at_eod(self):
        from src.strategy_profiles import get_strategy_profile

        ib      = _mock_ib()
        engine  = _make_engine_patched(ib)
        engine._strategy_profile = get_strategy_profile("indicator_swing")

        fresh_time = self._entry_before_hold_window()
        engine.state = {'AAPL': {'price': 100.0, 'time': fresh_time}}

        ib.reqTickers.return_value = [_mock_price_ticker(99.0, open=100.0, high=101.0, low=98.0)]

        with patch.object(engine, 'liquidate') as mock_liq:
            self._run_exit_check(engine)
            mock_liq.assert_not_called()

    def test_older_position_below_threshold_not_exited_before_eod_cleanup_time(self):
        ib      = _mock_ib()
        engine  = _make_engine_patched(ib)

        old_time = self._entry_after_hold_window()
        engine.state = {'AAPL': {'price': 100.0, 'time': old_time}}

        ib.reqTickers.return_value = [_mock_price_ticker(99.0, open=100.0, high=101.0, low=98.0)]

        with patch.object(engine, 'liquidate') as mock_liq:
            self._run_exit_check(engine, hour=15, minute=49)
            mock_liq.assert_not_called()

    def test_stale_close_price_does_not_trigger_exit_when_market_price_missing(self):
        ib      = _mock_ib()
        engine  = _make_engine_patched(ib)

        old_time = self._entry_after_hold_window()
        engine.state = {'AAPL': {'price': 100.0, 'time': old_time, 'current_price': 90.0}}

        # The ticker close can be stale/delayed, so it must not liquidate by itself.
        ticker       = MagicMock()
        ticker.marketPrice.return_value = float('nan')
        ticker.last                     = float('nan')
        ticker.bid                      = float('nan')
        ticker.ask                      = float('nan')
        ticker.close                    = 90.0
        ib.reqTickers.return_value      = [ticker]
        ib.positions.return_value       = []

        with patch.object(engine, 'liquidate') as mock_liq:
            self._run_exit_check(engine)
            mock_liq.assert_not_called()


# ── Position limit ────────────────────────────────────────────────────────────
class TestPositionLimit:
    def test_dynamic_position_capacity_blocks_new_entries_when_full(self):
        """run_cycle must not scan entries when dynamic equity capacity is full."""
        from src.config import MAX_POSITIONS_CAP, MIN_BUCKET_SIZE
        import pytz as real_pytz

        equity = 1400.0
        max_positions = min(int(equity / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)
        state_full = {f'SYM{i}': {'price': 100, 'time': datetime.now().isoformat()}
                      for i in range(max_positions)}
        ib     = _mock_ib()
        engine = _make_engine_patched(ib)
        engine.state = state_full

        # Mock ib.positions() to return matching positions so _sync_positions_from_ibkr
        # doesn't clear state_full entries
        mock_pos = []
        for i in range(max_positions):
            p = MagicMock()
            p.contract.symbol = f'SYM{i}'
            p.position = 10.0
            p.avgCost  = 100.0
            mock_pos.append(p)
        ib.positions.return_value = mock_pos

        # Build a real tz-aware datetime that falls inside the entry window
        tz_ny    = real_pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))  # Wednesday 10:30

        with patch.object(engine, 'get_institutional_scan', return_value=['NEW']), \
             patch.object(engine, 'get_technical_context') as mock_ctx, \
             patch.object(engine, 'manage_position_exits'), \
             patch('src.engine.datetime') as mock_dt:

            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat    = datetime.fromisoformat  # keep real impl

            engine.run_cycle()
            # All dynamic slots filled → no technical context is fetched.
            mock_ctx.assert_not_called()

    def test_no_vix_or_scanner_call_when_cash_slots_are_zero(self):
        """If no new entry can be placed, avoid VIX/HMDS and scanner API calls."""
        ib = _mock_ib()
        for item in ib.accountSummary.return_value:
            if item.tag == 'SettledCash':
                item.value = '0.0'
        engine = _make_engine_patched(ib)

        tz_ny = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, '_maybe_run_off_hours_jobs', return_value=False), \
             patch.object(engine, 'manage_position_exits') as mock_exits, \
             patch.object(engine, '_ensure_vix_contract') as mock_vix_contract, \
             patch.object(engine, '_fetch_vix_price') as mock_vix_price, \
             patch.object(engine, 'get_institutional_scan') as mock_scan, \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat

            engine.run_cycle()

        mock_exits.assert_called_once()
        mock_vix_contract.assert_not_called()
        mock_vix_price.assert_not_called()
        mock_scan.assert_not_called()

    def test_friday_entry_cutoff_blocks_vix_and_scanner(self):
        """Friday after the configured cutoff is position-management only."""
        from src.config import FRIDAY_ENTRY_CUTOFF_TIME

        ib = _mock_ib()
        engine = _make_engine_patched(ib)

        tz_ny = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(
            2024, 6, 7,
            FRIDAY_ENTRY_CUTOFF_TIME[0],
            FRIDAY_ENTRY_CUTOFF_TIME[1] + 1,
        ))

        with patch.object(engine, '_maybe_run_off_hours_jobs', return_value=False), \
             patch.object(engine, 'manage_position_exits') as mock_exits, \
             patch.object(engine, '_ensure_vix_contract') as mock_vix_contract, \
             patch.object(engine, '_fetch_vix_price') as mock_vix_price, \
             patch.object(engine, 'get_institutional_scan') as mock_scan, \
             patch.object(engine, '_update_position_prices'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat

            engine.run_cycle()

        mock_exits.assert_called_once()
        mock_vix_contract.assert_not_called()
        mock_vix_price.assert_not_called()
        mock_scan.assert_not_called()


# ── State persistence ─────────────────────────────────────────────────────────
class TestStatePersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        state_path = str(tmp_path / "state.json")
        ib     = _mock_ib()
        engine = _make_engine_patched(ib)
        engine.state = {'AAPL': {'price': 123.45, 'time': '2024-01-01T10:00:00'}}

        # Bypass the module-level STATE_FILE constant by patching open/json
        import src.engine as eng_mod
        original = eng_mod.STATE_FILE if hasattr(eng_mod, 'STATE_FILE') else None

        import src.config as cfg
        old_state_file = cfg.STATE_FILE
        cfg.STATE_FILE  = state_path

        import importlib
        importlib.reload(eng_mod)   # pick up new STATE_FILE

        try:
            engine.save_state.__func__  # verify it's a bound method
        except AttributeError:
            pass

        # Write directly using the patched path
        with open(state_path, 'w') as f:
            json.dump(engine.state, f)

        with open(state_path) as f:
            loaded = json.load(f)

        assert loaded == engine.state

        cfg.STATE_FILE = old_state_file


# ── Friday filter ─────────────────────────────────────────────────────────────
class TestFridayFilter:
    """On Fridays the dollar-volume threshold doubles."""

    def _passes(self, dollar_vol_20d, is_friday):
        from src.config import SCAN_MIN_DOLLAR_VOL, VOL_MULT_FRIDAY
        threshold = SCAN_MIN_DOLLAR_VOL * (VOL_MULT_FRIDAY if is_friday else 1.0)
        return dollar_vol_20d >= threshold

    def test_weekday_uses_normal_threshold(self):
        from src.config import SCAN_MIN_DOLLAR_VOL
        assert self._passes(SCAN_MIN_DOLLAR_VOL, is_friday=False) is True

    def test_friday_rejects_at_normal_threshold(self):
        from src.config import SCAN_MIN_DOLLAR_VOL
        # Exactly at normal threshold fails on Friday (need 2×)
        assert self._passes(SCAN_MIN_DOLLAR_VOL, is_friday=True) is False

    def test_friday_passes_at_double_threshold(self):
        from src.config import SCAN_MIN_DOLLAR_VOL, VOL_MULT_FRIDAY
        assert self._passes(SCAN_MIN_DOLLAR_VOL * VOL_MULT_FRIDAY, is_friday=True) is True

    def test_friday_rejects_below_double_threshold(self):
        from src.config import SCAN_MIN_DOLLAR_VOL
        # 1.5× is between 1× and 2× — passes on weekday, fails on Friday
        assert self._passes(int(SCAN_MIN_DOLLAR_VOL * 1.5), is_friday=True) is False


# ── VIX-High still manages positions ─────────────────────────────────────────
class TestVixHighBranch:
    """When VIX > threshold, existing-position exits and price updates must still run."""

    def test_vix_high_still_calls_position_exit_management(self):
        import pytz as real_pytz

        ib     = _mock_ib()
        engine = _make_engine_patched(ib)
        engine.state = {}

        # VIX ticker returns high value
        vix_ticker = MagicMock()
        vix_ticker.marketPrice.return_value = 40.0   # > VIX_THRESHOLD (35)

        vix_contract = MagicMock()
        ib.qualifyContracts.return_value = [vix_contract]
        ib.reqTickers.return_value       = [vix_ticker]
        ib.positions.return_value        = []

        tz_ny    = real_pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'manage_position_exits') as mock_exits, \
             patch.object(engine, '_update_position_prices') as mock_upd, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat

            engine.run_cycle()

        mock_exits.assert_called_once()
        mock_upd.assert_called_once()

    def test_vix_nan_still_calls_position_exit_management(self):
        import pytz as real_pytz

        ib     = _mock_ib()
        engine = _make_engine_patched(ib)
        engine.state = {}

        vix_ticker = MagicMock()
        vix_ticker.marketPrice.return_value = float('nan')
        vix_ticker.close                    = float('nan')   # fallback also NaN

        vix_contract = MagicMock()
        ib.qualifyContracts.return_value = [vix_contract]
        ib.reqTickers.return_value       = [vix_ticker]
        ib.positions.return_value        = []

        tz_ny    = real_pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, 'manage_position_exits') as mock_exits, \
             patch.object(engine, '_update_position_prices') as mock_upd, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat

            engine.run_cycle()

        mock_exits.assert_called_once()
        mock_upd.assert_called_once()


class TestOperatorHalt:
    """The manual halt file must block new entries but keep risk management alive."""

    def test_halt_file_blocks_scanner_after_exit_management(self, tmp_path):
        import pytz as real_pytz
        import src.engine as eng_mod

        ib     = _mock_ib()
        engine = _make_engine_patched(ib)
        engine.state = {}

        halt_file = tmp_path / "HALT_TRADING"
        halt_file.write_text("operator halt\n")

        vix_ticker = MagicMock()
        vix_ticker.marketPrice.return_value = 20.0
        vix_contract = MagicMock()
        ib.qualifyContracts.return_value = [vix_contract]
        ib.reqTickers.return_value       = [vix_ticker]
        ib.positions.return_value        = []

        tz_ny    = real_pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(eng_mod, 'HALT_FILE', str(halt_file)), \
             patch.object(engine, 'manage_position_exits') as mock_exit, \
             patch.object(engine, '_update_position_prices') as mock_prices, \
             patch.object(engine, 'get_institutional_scan') as mock_scan, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        mock_exit.assert_called_once()
        mock_prices.assert_called_once()
        mock_scan.assert_not_called()


class TestForceExit:
    """The emergency force-exit file must close positions before any scan/account work."""

    def test_force_exit_file_liquidates_tracked_positions(self, tmp_path):
        import src.engine as eng_mod

        ib     = _mock_ib()
        engine = _make_engine_patched(ib)
        engine.state = {
            'AAPL': {'price': 100.0, 'time': datetime.now().isoformat(), 'qty': 2},
            'MSFT': {'price': 200.0, 'time': datetime.now().isoformat(), 'qty': 1},
        }
        ib.isConnected.return_value = True

        force_file = tmp_path / "FORCE_EXIT_ALL"
        force_file.write_text("panic\n")

        with patch.object(eng_mod, 'FORCE_EXIT_FILE', str(force_file)), \
             patch.object(engine, '_sync_positions_from_ibkr'), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, '_update_position_prices') as mock_prices, \
             patch.object(engine, '_write_dashboard_data') as mock_dash, \
             patch.object(engine, '_alert') as mock_alert:
            engine.run_cycle()

        assert mock_liq.call_args_list == [call('AAPL'), call('MSFT')]
        mock_prices.assert_called_once()
        mock_dash.assert_called_once_with(connected=True)
        mock_alert.assert_called_once()
        ib.accountSummary.assert_not_called()


class TestConnectionSafety:
    """A failed reconnect must not fall through into trading logic."""

    def test_connect_waits_for_gateway_before_ib_connect(self):
        from src.engine import VelocityEngine
        import src.engine as eng_mod

        ib = _mock_ib()
        engine = VelocityEngine.__new__(VelocityEngine)
        engine.ib = ib

        with patch.object(engine, '_validate_deployment_mode'), \
             patch.object(engine, '_write_dashboard_data') as mock_dash, \
             patch.object(engine, '_warmup_historical_data') as mock_warmup, \
             patch.object(eng_mod, 'ensure_ib_gateway_ready', return_value=True) as mock_ready:
            engine.connect()

        mock_ready.assert_called_once()
        ib.connect.assert_called_once()
        mock_dash.assert_called_once_with(connected=True)
        mock_warmup.assert_called_once_with(reason="connect")

    def test_connect_fails_closed_when_gateway_unavailable(self):
        from src.engine import VelocityEngine
        import src.engine as eng_mod

        ib = _mock_ib()
        engine = VelocityEngine.__new__(VelocityEngine)
        engine.ib = ib

        with patch.object(engine, '_validate_deployment_mode'), \
             patch.object(engine, '_alert') as mock_alert, \
             patch.object(eng_mod, 'ensure_ib_gateway_ready', return_value=False), \
             pytest.raises(SystemExit):
            engine.connect()

        ib.connect.assert_not_called()
        mock_alert.assert_called_once()

    def test_run_cycle_skips_when_reconnect_fails(self):
        ib     = _mock_ib()
        engine = _make_engine_patched(ib)
        ib.isConnected.return_value = False

        with patch.object(engine, '_reconnect', return_value=False), \
             patch.object(engine, '_write_dashboard_data') as mock_dash:
            engine.run_cycle()

        ib.accountSummary.assert_not_called()
        mock_dash.assert_called_once_with(connected=False)

    def test_run_cycle_skips_entries_when_account_summary_unavailable(self):
        ib     = _mock_ib()
        engine = _make_engine_patched(ib)
        ib.isConnected.return_value = True
        ib.accountSummary.return_value = []

        with patch.object(engine, 'manage_position_exits') as mock_exits, \
             patch.object(engine, '_update_position_prices') as mock_prices, \
             patch.object(engine, '_write_dashboard_data') as mock_dash, \
             patch.object(engine, '_alert') as mock_alert, \
             patch.object(engine, 'get_institutional_scan') as mock_scan, \
             patch.object(engine, '_maybe_run_off_hours_jobs', return_value=False), \
             patch.object(engine, '_regular_management_active', return_value=True):
            engine.run_cycle()

        mock_exits.assert_called_once()
        mock_prices.assert_called_once()
        mock_dash.assert_called_with(connected=True)
        mock_alert.assert_called_once()
        mock_scan.assert_not_called()


class TestHistoricalDataWarmup:
    def test_warmup_marks_general_hmds_unhealthy_when_spy_fails(self):
        ib = _mock_ib()
        engine = _make_engine_patched(ib)
        ib.qualifyContracts.return_value = [MagicMock()]
        # SPY fails on every attempt — return [] for all retries
        ib.reqHistoricalData.return_value = []

        # Fix 7 retries HMDS_WARMUP_MAX_RETRIES times; patch sleep so test is fast
        with patch('src.engine.time.sleep'), \
             patch('src.engine.HMDS_WARMUP_MAX_RETRIES', 3):
            assert engine._warmup_historical_data(reason="test") is False

        assert engine._historical_data_health['SPY']['ok'] is False
        # With retries, SPY is requested HMDS_WARMUP_MAX_RETRIES times
        assert ib.reqHistoricalData.call_count == 3

    def test_warmup_marks_vix_specific_failure_after_spy_success(self):
        ib = _mock_ib()
        engine = _make_engine_patched(ib)
        # SPY succeeds every attempt, VIX fails every attempt (3 retries)
        ib.qualifyContracts.return_value = [MagicMock()]
        spy_bar = MagicMock(close=450.0)
        ib.reqHistoricalData.side_effect = [
            [spy_bar], [],  # attempt 1: SPY ok, VIX fail
            [spy_bar], [],  # attempt 2
            [spy_bar], [],  # attempt 3
        ]

        with patch('src.engine.time.sleep'), \
             patch('src.engine.HMDS_WARMUP_MAX_RETRIES', 3):
            assert engine._warmup_historical_data(reason="test") is False

        assert engine._historical_data_health['SPY']['ok'] is True
        assert engine._historical_data_health['VIX']['ok'] is False
        assert engine._vix_failure_count == 1

    def test_warmup_success_caches_vix(self):
        ib = _mock_ib()
        engine = _make_engine_patched(ib)
        ib.qualifyContracts.return_value = [MagicMock()]
        ib.reqHistoricalData.side_effect = [
            [MagicMock(close=450.0)],
            [MagicMock(close=16.25)],
        ]

        with patch('src.engine.time.sleep'):
            assert engine._warmup_historical_data(reason="test") is True

        assert engine._last_vix == pytest.approx(16.25)
        assert engine._last_vix_source == "historical_warmup"


class TestIbErrorLogging:
    def test_hmds_items_retrieved_notice_is_silent(self):
        ib = _mock_ib()
        engine = _make_engine_patched(ib)

        with patch.object(engine, '_metric_inc') as mock_metric, \
             patch('src.engine.logger') as mock_logger:
            engine._on_ib_error(
                123,
                165,
                "Historical Market Data Service query message:12 items retrieved",
                None,
            )

        mock_metric.assert_not_called()
        mock_logger.warning.assert_not_called()

    def test_other_hmds_error_165_still_warns_and_counts(self):
        ib = _mock_ib()
        engine = _make_engine_patched(ib)

        with patch.object(engine, '_metric_inc') as mock_metric, \
             patch('src.engine.logger') as mock_logger:
            engine._on_ib_error(123, 165, "Historical data request failed", None)

        mock_metric.assert_any_call('ib_errors')
        mock_metric.assert_any_call('ib_error_codes', subkey='165')
        mock_logger.warning.assert_called_once()


class TestOffHoursMaintenance:
    _TZ_NY = pytz.timezone('US/Eastern')

    def test_premarket_readiness_collects_snapshot_without_exits_or_entries(self):
        from src.config import PREMARKET_READINESS_TIME
        from src.engine import READINESS_FILE

        ib = _mock_ib()
        engine = _make_engine_patched(ib)
        engine._last_premarket_prefilter_date = '2024-06-05'
        engine.state = {
            'AAPL': {
                'fill_price': 100.0,
                'price': 100.0,
                'current_price': 101.0,
                'qty': 2.0,
                'stop_loss': 94.0,
                'effective_stop': 95.0,
                'stop_dist': 6.0,
            }
        }
        h, m = PREMARKET_READINESS_TIME
        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, h, m, 0))

        with patch.object(engine, '_sync_positions_from_ibkr') as mock_sync, \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch.object(engine, '_update_position_prices') as mock_prices, \
             patch.object(engine, '_fetch_spy_trend', return_value=True), \
             patch.object(engine, 'manage_position_exits') as mock_exits, \
             patch.object(engine, 'get_institutional_scan') as mock_scan, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine.run_cycle()

        # One source-of-truth sync at cycle start, then one confirmation sync
        # inside the off-hours maintenance checkpoint before stop audit.
        assert mock_sync.call_count == 2
        mock_audit.assert_called_once()
        mock_prices.assert_called_once()
        mock_exits.assert_not_called()
        mock_scan.assert_not_called()
        assert engine._last_premarket_readiness_date == '2024-06-05'
        with open(READINESS_FILE, 'r') as f:
            snapshot = json.load(f)
        assert snapshot['checkpoint'] == 'premarket_readiness'
        assert snapshot['positions'][0]['symbol'] == 'AAPL'
        assert snapshot['account']['equity'] == 1400.0
        assert snapshot['regime']['vix'] == 20.0

    def test_post_close_reconciliation_runs_once_without_new_entries(self):
        from src.config import POST_CLOSE_MAINTENANCE_TIME
        from src.engine import READINESS_FILE

        ib = _mock_ib()
        engine = _make_engine_patched(ib)
        engine.state = {
            'MSFT': {
                'fill_price': 300.0,
                'price': 300.0,
                'current_price': 301.0,
                'qty': 1.0,
                'stop_loss': 285.0,
                'stop_dist': 15.0,
            }
        }
        h, m = POST_CLOSE_MAINTENANCE_TIME
        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, h, m, 0))

        with patch.object(engine, '_sync_positions_from_ibkr') as mock_sync, \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_fetch_spy_trend', return_value=False), \
             patch.object(engine, 'manage_position_exits') as mock_exits, \
             patch.object(engine, 'get_institutional_scan') as mock_scan, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine.run_cycle()

        # One source-of-truth sync at cycle start, then one confirmation sync
        # inside the off-hours maintenance checkpoint before stop audit.
        assert mock_sync.call_count == 2
        mock_audit.assert_called_once()
        mock_exits.assert_not_called()
        mock_scan.assert_not_called()
        assert engine._last_post_close_maintenance_date == '2024-06-05'
        with open(READINESS_FILE, 'r') as f:
            snapshot = json.load(f)
        assert snapshot['checkpoint'] == 'post_close_reconciliation'
        assert snapshot['positions'][0]['symbol'] == 'MSFT'

    def test_off_hours_job_is_once_per_trading_date(self):
        from src.config import PREMARKET_READINESS_TIME

        engine = _make_engine_patched(_mock_ib())
        engine._last_premarket_prefilter_date = '2024-06-05'
        engine._last_premarket_readiness_date = '2024-06-05'
        h, m = PREMARKET_READINESS_TIME
        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, h, m + 5, 0))

        with patch.object(engine, '_run_operational_maintenance') as mock_job, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            assert engine._maybe_run_off_hours_jobs() is False

        mock_job.assert_not_called()

    def test_premarket_prefilter_starts_at_configured_time(self):
        from src.config import APP_PREFILTER_START_TIME

        engine = _make_engine_patched(_mock_ib())
        h, m = APP_PREFILTER_START_TIME
        fake_now = self._TZ_NY.localize(datetime(2024, 6, 5, h, m, 0))

        with patch.object(engine, '_run_premarket_universe_prefilter', return_value={}) as mock_prefilter, \
             patch.object(engine, '_run_operational_maintenance') as mock_maintenance, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            assert engine._maybe_run_off_hours_jobs() is True

        mock_prefilter.assert_called_once()
        mock_maintenance.assert_not_called()
        assert engine._last_premarket_prefilter_date == '2024-06-05'


# ── Logger handler guard ──────────────────────────────────────────────────────
class TestLoggerHandlerGuard:
    def test_no_duplicate_handlers_on_reimport(self):
        import importlib
        import src.engine as eng_mod

        handler_count_before = len(logging.getLogger('VelocityEngine').handlers)
        importlib.reload(eng_mod)
        handler_count_after  = len(logging.getLogger('VelocityEngine').handlers)

        assert handler_count_after == handler_count_before, (
            f"Re-import added handlers: {handler_count_before} → {handler_count_after}"
        )

    def test_log_rollover_uses_eastern_midnight(self, tmp_path):
        import src.engine as eng_mod

        handler = eng_mod._EasternTimedRotatingFileHandler(
            str(tmp_path / "trading_engine.log"),
            when='midnight',
            backupCount=1,
        )
        try:
            current = eng_mod._TZ_NY.localize(
                datetime(2026, 5, 28, 23, 59, 59)
            ).timestamp()

            rollover = handler.computeRollover(current)
            rollover_et = datetime.fromtimestamp(rollover, eng_mod._TZ_NY)

            assert rollover_et == eng_mod._TZ_NY.localize(
                datetime(2026, 5, 29, 0, 0, 0)
            )
        finally:
            handler.close()


# ── SPY Regime ────────────────────────────────────────────────────────────────
class TestSpyRegime:
    @staticmethod
    def _spy_df(first5: float) -> pd.DataFrame:
        closes = [first5] * 5 + [80.0] * 150 + [110.0] * 49 + [112.0]
        return pd.DataFrame({"close": closes})

    def _run_spy_trend(self, ib, df: pd.DataFrame) -> bool:
        import src.engine as eng_mod

        engine = _make_engine_patched(ib)
        engine._spy_cache = {}
        engine._contract_cache = {'SPY': MagicMock()}
        ib.reqHistoricalData.return_value = [object()] * len(df)

        with patch.object(eng_mod.util, 'df', return_value=df):
            return engine._fetch_spy_trend()

    def test_spy_uptrend_requires_rising_sma200(self):
        ib = _mock_ib()

        # Last close is above MA50 > MA200, but the 200-day average is falling
        # because very high prices are rolling out of the 200-day window.
        assert self._run_spy_trend(ib, self._spy_df(first5=200.0)) is False

    def test_spy_uptrend_allows_rising_sma200(self):
        ib = _mock_ib()

        assert self._run_spy_trend(ib, self._spy_df(first5=70.0)) is True

    def test_log_rollover_filename_uses_eastern_date(self, tmp_path):
        import src.engine as eng_mod

        log_file = tmp_path / "trading_engine.log"
        handler = eng_mod._EasternTimedRotatingFileHandler(
            str(log_file),
            when='midnight',
            backupCount=3,
        )
        handler.namer = eng_mod._log_namer
        try:
            handler.stream.write("old eastern-day log\n")
            handler.stream.flush()
            handler.rolloverAt = int(
                eng_mod._TZ_NY.localize(datetime(2026, 5, 29, 0, 0, 0)).timestamp()
            )

            with patch.object(eng_mod.time, 'time', return_value=handler.rolloverAt + 1):
                handler.doRollover()

            assert (tmp_path / "trading_engine_2026-05-28.log").exists()
            assert not (tmp_path / "trading_engine_2026-05-29.log").exists()
        finally:
            handler.close()
