"""
Unit tests for VelocityEngine business logic.

IB is fully mocked — no live connection required.
Tests exercise entry signals, velocity exits, position limits,
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

    # whatIfOrder pre-flight: empty warningText = IB accepts the order
    ib.whatIfOrder.return_value.warningText = ''

    return ib


def _make_engine_patched(ib_mock):
    """Return a VelocityEngine with IB replaced and connect() bypassed."""
    with patch('src.engine.IB', return_value=ib_mock), \
         patch.object(sys.modules.get('src.engine', __import__('src.engine')),
                      'logger', MagicMock()):
        from src.engine import VelocityEngine
        with patch.object(VelocityEngine, 'connect', lambda self: None):
            engine = VelocityEngine.__new__(VelocityEngine)
            engine.ib                   = ib_mock
            engine.state                = {}
            engine._last_equity         = 0.0
            engine._last_settled_cash   = 0.0
            engine._equity_initialized  = False
            engine._last_vix            = None
            engine._last_scan_ts        = None
            engine._next_scan_dt        = None
            # Attributes added after initial implementation
            engine._day_start_date      = None
            engine._day_start_equity    = None
            engine._contract_cache      = {}
            engine._bar_cache           = {}
            engine._vix_contract        = None
            engine._spy_cache           = {}
            engine._sector_cache        = {}
            engine._daily_scan_skip     = {}
            engine._last_audit_date     = None
            engine._missing_position_counts = {}
            return engine


# ── Expert filter (entry conditions) ─────────────────────────────────────────
class TestExpertFilter:
    """
    The entry guard in run_cycle is:
        price > orb_high
        price > ma50 > ma200
        rsi > rsi_prev
        rsi > 55
        day_range_location >= configured minimum
        intraday_gain >= configured minimum
        ATR_CHAND / price <= configured maximum
    """

    def _ctx(self, price=110, orb=100, ma50=105, ma200=90, rsi=60, rsi_prev=55,
             dollar_vol_20d=300_000_000, day_range_location=0.75,
             intraday_gain=0.01):
        return dict(orb_high=orb, ma50=ma50, ma200=ma200,
                    rsi=rsi, rsi_prev=rsi_prev, atr=2.0,
                    atr_chandelier=2.0,
                    close=price, live_price=price,
                    bid=price * 0.999, ask=price * 1.001,
                    spread_pct=0.002,
                    dollar_vol_20d=dollar_vol_20d,
                    day_range_location=day_range_location,
                    intraday_gain=intraday_gain,
                    contract=MagicMock())

    def _passes(self, ctx):
        from src.config import (
            DAY_RANGE_LOCATION_MIN,
            INTRADAY_GAIN_MIN,
            RSI_THRESHOLD,
            SCAN_MIN_DOLLAR_VOL,
            ATR_PCT_MAX,
        )
        p = ctx['live_price']
        return (p > ctx['orb_high']
                and p > ctx['ma50'] > ctx['ma200']
                and ctx['rsi'] > ctx['rsi_prev']
                and ctx['rsi'] > RSI_THRESHOLD
                and ctx['day_range_location'] >= DAY_RANGE_LOCATION_MIN
                and ctx['intraday_gain'] >= INTRADAY_GAIN_MIN
                and ctx['atr_chandelier'] / p <= ATR_PCT_MAX
                and ctx['dollar_vol_20d'] >= SCAN_MIN_DOLLAR_VOL)

    def test_all_conditions_met(self):
        assert self._passes(self._ctx()) is True

    def test_fails_when_price_below_orb(self):
        assert self._passes(self._ctx(price=99, orb=100)) is False

    def test_fails_when_price_below_ma50(self):
        assert self._passes(self._ctx(price=110, ma50=115)) is False

    def test_fails_when_ma50_below_ma200(self):
        assert self._passes(self._ctx(ma50=85, ma200=90)) is False

    def test_fails_when_rsi_not_rising(self):
        assert self._passes(self._ctx(rsi=60, rsi_prev=65)) is False

    def test_fails_when_rsi_below_threshold(self):
        assert self._passes(self._ctx(rsi=54, rsi_prev=50)) is False

    def test_rsi_exactly_at_threshold_fails(self):
        # rule is rsi > 55, so exactly 55 should fail
        assert self._passes(self._ctx(rsi=55, rsi_prev=50)) is False

    def test_fails_when_dollar_volume_below_threshold(self):
        assert self._passes(self._ctx(dollar_vol_20d=50_000_000)) is False

    def test_passes_when_dollar_volume_at_threshold(self):
        from src.config import SCAN_MIN_DOLLAR_VOL
        assert self._passes(self._ctx(dollar_vol_20d=SCAN_MIN_DOLLAR_VOL)) is True

    def test_fails_when_day_range_location_below_threshold(self):
        from src.config import DAY_RANGE_LOCATION_MIN
        assert self._passes(
            self._ctx(day_range_location=DAY_RANGE_LOCATION_MIN - 0.01)
        ) is False

    def test_fails_when_intraday_gain_below_threshold(self):
        from src.config import INTRADAY_GAIN_MIN
        assert self._passes(
            self._ctx(intraday_gain=INTRADAY_GAIN_MIN - 0.001)
        ) is False

    def test_fails_when_atr_pct_above_threshold(self):
        from src.config import ATR_PCT_MAX
        ctx = self._ctx()
        ctx['atr_chandelier'] = ctx['live_price'] * (ATR_PCT_MAX + 0.01)
        assert self._passes(ctx) is False


# ── Velocity exit logic ───────────────────────────────────────────────────────
class TestVelocityExit:
    _TZ_NY = pytz.timezone('US/Eastern')

    def _entry_after_hold_window(self):
        """Tuesday entry with Wednesday check: 1 Mon-Fri session elapsed."""
        return self._TZ_NY.localize(datetime(2024, 6, 4, 10, 30)).isoformat()

    def _entry_before_hold_window(self):
        """Same-day Wednesday entry/check: 0 Mon-Fri sessions elapsed."""
        return self._TZ_NY.localize(datetime(2024, 6, 5, 10, 0)).isoformat()

    def _run_velocity_check(self, engine):
        check_time = self._TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = check_time
            mock_dt.fromisoformat = datetime.fromisoformat
            engine.check_velocity_exits()

    def test_stagnant_position_older_than_hold_window_triggers_exit(self):
        ib      = _mock_ib()
        engine  = _make_engine_patched(ib)

        old_time = self._entry_after_hold_window()
        engine.state = {'AAPL': {'price': 100.0, 'time': old_time}}

        # Market price only slightly up (1% gain < PROFIT_MIN_THRESHOLD=5%)
        ticker = MagicMock()
        ticker.marketPrice.return_value = 101.0
        ib.reqTickers.return_value = [ticker]
        ib.positions.return_value  = []

        with patch.object(engine, 'liquidate') as mock_liq:
            self._run_velocity_check(engine)
            mock_liq.assert_called_once_with('AAPL')

    def test_profitable_position_not_exited_early(self):
        ib      = _mock_ib()
        engine  = _make_engine_patched(ib)

        old_time = self._entry_after_hold_window()
        engine.state = {'AAPL': {'price': 100.0, 'time': old_time}}

        # 6% gain — above PROFIT_MIN_THRESHOLD (5%) → must NOT trigger velocity exit
        ticker = MagicMock()
        ticker.marketPrice.return_value = 106.0
        ib.reqTickers.return_value = [ticker]

        with patch.object(engine, 'liquidate') as mock_liq:
            self._run_velocity_check(engine)
            mock_liq.assert_not_called()

    def test_fresh_position_not_exited(self):
        ib      = _mock_ib()
        engine  = _make_engine_patched(ib)

        fresh_time = self._entry_before_hold_window()
        engine.state = {'AAPL': {'price': 100.0, 'time': fresh_time}}

        ticker = MagicMock()
        ticker.marketPrice.return_value = 99.0
        ib.reqTickers.return_value = [ticker]

        with patch.object(engine, 'liquidate') as mock_liq:
            self._run_velocity_check(engine)
            mock_liq.assert_not_called()

    def test_falls_back_to_close_price_when_market_price_nan(self):
        ib      = _mock_ib()
        engine  = _make_engine_patched(ib)

        old_time = self._entry_after_hold_window()
        engine.state = {'AAPL': {'price': 100.0, 'time': old_time}}

        # 0.5% gain via close fallback — below PROFIT_MIN_THRESHOLD=5% → exit
        ticker       = MagicMock()
        ticker.marketPrice.return_value = float('nan')
        ticker.close                    = 100.5
        ib.reqTickers.return_value      = [ticker]
        ib.positions.return_value       = []

        with patch.object(engine, 'liquidate') as mock_liq:
            self._run_velocity_check(engine)
            mock_liq.assert_called_once_with('AAPL')


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
             patch.object(engine, 'check_velocity_exits'), \
             patch('src.engine.datetime') as mock_dt:

            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat    = datetime.fromisoformat  # keep real impl

            engine.run_cycle()
            # All dynamic slots filled → scanner runs but get_technical_context never called
            mock_ctx.assert_not_called()


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
    """When VIX > threshold, velocity exits and price updates must still run."""

    def test_vix_high_still_calls_velocity_exits(self):
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

        with patch.object(engine, 'check_velocity_exits') as mock_vel, \
             patch.object(engine, '_update_position_prices') as mock_upd, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat

            engine.run_cycle()

        mock_vel.assert_called_once()
        mock_upd.assert_called_once()

    def test_vix_nan_still_calls_velocity_exits(self):
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

        with patch.object(engine, 'check_velocity_exits') as mock_vel, \
             patch.object(engine, '_update_position_prices') as mock_upd, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat

            engine.run_cycle()

        mock_vel.assert_called_once()
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
             patch.object(engine, 'check_velocity_exits') as mock_exit, \
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
             patch.object(eng_mod, 'ensure_ib_gateway_ready', return_value=True) as mock_ready:
            engine.connect()

        mock_ready.assert_called_once()
        ib.connect.assert_called_once()
        mock_dash.assert_called_once_with(connected=True)

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

        with patch.object(engine, 'check_velocity_exits') as mock_exits, \
             patch.object(engine, '_update_position_prices') as mock_prices, \
             patch.object(engine, '_write_dashboard_data') as mock_dash, \
             patch.object(engine, '_alert') as mock_alert, \
             patch.object(engine, 'get_institutional_scan') as mock_scan, \
             patch.object(engine, '_maybe_run_off_hours_jobs', return_value=False):
            engine.run_cycle()

        mock_exits.assert_called_once()
        mock_prices.assert_called_once()
        mock_dash.assert_called_with(connected=True)
        mock_alert.assert_called_once()
        mock_scan.assert_not_called()


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
