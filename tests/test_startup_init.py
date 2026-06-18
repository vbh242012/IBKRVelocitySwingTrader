"""
Unit tests for VelocityEngine startup initialisation gate.

Covers:
  - _fetch_equity_with_retry: retries on empty summary, missing NL tag, zero/negative
    equity, exceptions, and succeeds after N attempts
  - _initialize: sets real equity, marks _equity_initialized, syncs positions,
    calls _update_position_prices only when positions exist, writes dashboard
  - _update_position_prices: stores unrealized_pnl and unrealized_pnl_pct
  - run(): calls _initialize before the first run_cycle

IB is fully mocked — no live connection required.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytz
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, call
import src.engine as eng_mod


# ── shared helpers ─────────────────────────────────────────────────────────────

def _nl_item(value: str = '5000.0'):
    item       = MagicMock()
    item.tag   = 'NetLiquidation'
    item.value = value
    return item


def _acct_item(tag: str, value: str, currency: str = 'USD'):
    item          = MagicMock()
    item.tag      = tag
    item.value    = value
    item.currency = currency
    return item


def _other_item(tag: str = 'TotalCashValue', value: str = '100.0'):
    item       = MagicMock()
    item.tag   = tag
    item.value = value
    return item


def _make_engine(ib_mock=None, state=None):
    """Return a VelocityEngine with IB replaced and connect() bypassed."""
    if ib_mock is None:
        ib_mock = MagicMock()
        ib_mock.accountSummary.return_value = [_nl_item('5000.0')]
        ib_mock.positions.return_value = []

    from src.engine import VelocityEngine
    engine = VelocityEngine.__new__(VelocityEngine)
    engine.ib                  = ib_mock
    engine.state               = state if state is not None else {}
    engine._last_equity        = 0.0
    engine._last_settled_cash  = 0.0
    engine._equity_initialized = False
    engine._last_vix           = None
    engine._last_scan_ts       = None
    engine._next_scan_dt       = None
    engine._day_start_date     = None
    engine._day_start_equity   = None
    engine._contract_cache     = {}
    engine._bar_cache          = {}
    engine._vix_contract       = None
    engine._spy_cache          = {}
    engine._sector_cache       = {}
    engine._daily_scan_skip    = {}
    engine._last_audit_date    = None
    engine._last_audit_at      = None
    engine._last_pre_entry_sync_date = None
    engine._last_post_open_audit_date = None
    engine._last_premarket_readiness_date = None
    engine._last_post_close_maintenance_date = None
    engine._missing_position_counts = {}
    return engine


# ── _fetch_equity_with_retry ───────────────────────────────────────────────────

class TestFetchEquityWithRetry:
    def test_returns_immediately_on_first_success(self):
        ib = MagicMock()
        ib.accountSummary.return_value = [_nl_item('5000.0')]
        engine = _make_engine(ib)
        result = engine._fetch_equity_with_retry()
        assert result == 5000.0
        assert ib.accountSummary.call_count == 1
        assert ib.sleep.call_count == 0

    def test_retries_when_summary_is_empty(self):
        ib = MagicMock()
        # first call returns empty, second returns valid
        ib.accountSummary.side_effect = [[], [_nl_item('3000.0')]]
        engine = _make_engine(ib)
        result = engine._fetch_equity_with_retry()
        assert result == 3000.0
        assert ib.accountSummary.call_count == 2
        assert ib.sleep.call_count == 1

    def test_retries_when_nl_tag_is_missing(self):
        ib = MagicMock()
        ib.accountSummary.side_effect = [
            [_other_item()],            # no NL tag
            [_nl_item('2500.0')],
        ]
        engine = _make_engine(ib)
        result = engine._fetch_equity_with_retry()
        assert result == 2500.0
        assert ib.sleep.call_count == 1

    def test_retries_when_equity_is_zero(self):
        ib = MagicMock()
        ib.accountSummary.side_effect = [
            [_nl_item('0.0')],
            [_nl_item('1800.0')],
        ]
        engine = _make_engine(ib)
        result = engine._fetch_equity_with_retry()
        assert result == 1800.0

    def test_retries_when_equity_is_negative(self):
        ib = MagicMock()
        ib.accountSummary.side_effect = [
            [_nl_item('-500.0')],
            [_nl_item('4200.0')],
        ]
        engine = _make_engine(ib)
        result = engine._fetch_equity_with_retry()
        assert result == 4200.0

    def test_retries_on_exception_then_succeeds(self):
        ib = MagicMock()
        ib.accountSummary.side_effect = [
            RuntimeError("connection lost"),
            [_nl_item('6000.0')],
        ]
        engine = _make_engine(ib)
        result = engine._fetch_equity_with_retry()
        assert result == 6000.0
        assert ib.sleep.call_count == 1

    def test_succeeds_after_three_failures(self):
        ib = MagicMock()
        ib.accountSummary.side_effect = [
            [],
            [_nl_item('0.0')],
            [_other_item()],
            [_nl_item('7500.0')],
        ]
        engine = _make_engine(ib)
        result = engine._fetch_equity_with_retry()
        assert result == 7500.0
        assert ib.accountSummary.call_count == 4
        assert ib.sleep.call_count == 3

    def test_sleep_duration_matches_config(self):
        from src.config import EQUITY_RETRY_INTERVAL
        ib = MagicMock()
        ib.accountSummary.side_effect = [[], [_nl_item('1000.0')]]
        engine = _make_engine(ib)
        engine._fetch_equity_with_retry()
        ib.sleep.assert_called_once_with(EQUITY_RETRY_INTERVAL)


# ── _get_account_values ───────────────────────────────────────────────────────

class TestAccountValues:
    def test_account_summary_snapshot_cancels_live_request(self):
        """Live IB account-summary requests must be cancelled to avoid leaks."""
        class FakeIB(eng_mod.IB):
            pass

        ib = FakeIB.__new__(FakeIB)
        ib.disconnect = MagicMock()
        ib.client = MagicMock()
        ib.client.getReqId.return_value = 77
        ib.wrapper = MagicMock()
        ib.wrapper.startReq.return_value = object()
        snapshot = {
            ('U123', 'NetLiquidation', 'USD'): _acct_item('NetLiquidation', '5000.0', 'USD'),
            ('U123', 'SettledCash', 'USD'): _acct_item('SettledCash', '1200.0', 'USD'),
        }
        ib.wrapper.acctSummary = {}
        ib._run = MagicMock(side_effect=lambda _future: ib.wrapper.acctSummary.update(snapshot))

        engine = _make_engine(ib)
        summary = engine._request_account_summary_snapshot()

        assert {item.tag: float(item.value) for item in summary} == {
            'NetLiquidation': 5000.0,
            'SettledCash': 1200.0,
        }
        ib.client.reqAccountSummary.assert_called_once()
        ib.client.cancelAccountSummary.assert_called_once_with(77)

    def test_get_account_values_ignores_non_usd_rows(self):
        ib = MagicMock()
        ib.accountSummary.return_value = [
            _acct_item('NetLiquidation', '999999.0', 'EUR'),
            _acct_item('SettledCash', '999999.0', 'EUR'),
            _acct_item('AvailableFunds', '999999.0', 'EUR'),
            _acct_item('NetLiquidation', '5000.0', 'USD'),
            _acct_item('SettledCash', '1200.0', 'USD'),
            _acct_item('AvailableFunds', '1400.0', 'USD'),
        ]
        engine = _make_engine(ib)

        equity, cash = engine._get_account_values()

        assert equity == 5000.0
        assert cash == 1200.0

    def test_get_account_values_never_uses_available_funds_for_cash_sizing(self):
        ib = MagicMock()
        ib.accountSummary.return_value = [
            _acct_item('NetLiquidation', '5000.0', 'USD'),
            _acct_item('AvailableFunds', '5000.0', 'USD'),
        ]
        engine = _make_engine(ib)

        equity, cash = engine._get_account_values()

        assert equity == 5000.0
        assert cash == 0.0

    def test_get_account_values_raises_when_summary_unusable(self):
        ib = MagicMock()
        ib.accountSummary.return_value = []
        engine = _make_engine(ib)

        with pytest.raises(eng_mod.AccountDataUnavailable):
            engine._get_account_values()

    def test_price_coercion_rejects_unconfigured_mocks(self):
        engine = _make_engine(MagicMock())

        assert engine._coerce_positive_price(MagicMock()) is None
        assert engine._coerce_positive_price(float('nan')) is None
        assert engine._coerce_positive_price(0.0) is None
        assert engine._coerce_positive_price(101.25) == 101.25


# ── Deployment safety ─────────────────────────────────────────────────────────

class TestDeploymentSafety:
    def test_paper_mode_refuses_live_ib_port(self):
        engine = _make_engine(MagicMock())

        with patch.object(eng_mod, 'TRADING_MODE', 'paper'), \
             patch.object(eng_mod, 'IB_PORT', 4001), \
             patch.object(engine, '_alert') as mock_alert:
            with pytest.raises(SystemExit):
                engine._validate_deployment_mode()

        mock_alert.assert_called_once()

    def test_live_mode_requires_explicit_acknowledgement(self):
        engine = _make_engine(MagicMock())

        with patch.object(eng_mod, 'TRADING_MODE', 'live'), \
             patch.object(eng_mod, 'IB_PORT', 4001), \
             patch.object(eng_mod, 'LIVE_TRADING_ACK', ''), \
             patch.object(engine, '_alert') as mock_alert:
            with pytest.raises(SystemExit):
                engine._validate_deployment_mode()

        mock_alert.assert_called_once()

    def test_live_mode_allows_explicit_acknowledgement(self):
        engine = _make_engine(MagicMock())

        with patch.object(eng_mod, 'TRADING_MODE', 'live'), \
             patch.object(eng_mod, 'IB_PORT', 4001), \
             patch.object(eng_mod, 'LIVE_TRADING_ACK', eng_mod.LIVE_TRADING_ACK_PHRASE), \
             patch.object(engine, '_alert') as mock_alert:
            engine._validate_deployment_mode()

        mock_alert.assert_not_called()

    def test_live_mode_refuses_paper_ib_port_even_with_acknowledgement(self):
        engine = _make_engine(MagicMock())

        with patch.object(eng_mod, 'TRADING_MODE', 'live'), \
             patch.object(eng_mod, 'IB_PORT', 4002), \
             patch.object(eng_mod, 'LIVE_TRADING_ACK', eng_mod.LIVE_TRADING_ACK_PHRASE), \
             patch.object(engine, '_alert') as mock_alert:
            with pytest.raises(SystemExit):
                engine._validate_deployment_mode()

        mock_alert.assert_called_once()


# ── State persistence safety ──────────────────────────────────────────────────

class TestStatePersistenceSafety:
    def test_corrupt_state_file_is_backed_up_before_empty_state(self):
        with open(eng_mod.STATE_FILE, 'w') as f:
            f.write('{not valid json')

        engine = _make_engine(MagicMock())
        with patch.object(engine, '_alert') as mock_alert:
            state = engine.load_state()

        assert state == {}
        assert mock_alert.called
        backups = [
            name for name in os.listdir(os.path.dirname(eng_mod.STATE_FILE))
            if name.startswith(os.path.basename(eng_mod.STATE_FILE) + '.corrupt.')
        ]
        assert backups

    def test_non_dict_state_file_is_backed_up_before_empty_state(self):
        with open(eng_mod.STATE_FILE, 'w') as f:
            json.dump([], f)

        engine = _make_engine(MagicMock())
        with patch.object(engine, '_alert') as mock_alert:
            state = engine.load_state()

        assert state == {}
        assert mock_alert.called
        backups = [
            name for name in os.listdir(os.path.dirname(eng_mod.STATE_FILE))
            if name.startswith(os.path.basename(eng_mod.STATE_FILE) + '.invalid.')
        ]
        assert backups


# ── _initialize ────────────────────────────────────────────────────────────────

class TestInitialize:
    def test_sets_last_equity_from_api(self):
        ib = MagicMock()
        ib.accountSummary.return_value = [_nl_item('8000.0')]
        ib.positions.return_value = []
        engine = _make_engine(ib)
        with patch.object(engine, '_write_dashboard_data'):
            engine._initialize()
        assert engine._last_equity == 8000.0
        assert engine._last_settled_cash == 0.0

    def test_marks_equity_initialized(self):
        ib = MagicMock()
        ib.accountSummary.return_value = [_nl_item('5000.0')]
        ib.positions.return_value = []
        engine = _make_engine(ib)
        assert engine._equity_initialized is False
        with patch.object(engine, '_write_dashboard_data'):
            engine._initialize()
        assert engine._equity_initialized is True

    def test_does_not_use_hardcoded_capital_seed(self):
        ib = MagicMock()
        ib.accountSummary.return_value = [_nl_item('9999.0')]
        ib.positions.return_value = []
        engine = _make_engine(ib)
        with patch.object(engine, '_write_dashboard_data'):
            engine._initialize()
        # real equity from IBKR, not a local capital_seed fallback
        assert engine._last_equity == 9999.0
        assert not hasattr(engine, 'capital_seed')

    def test_syncs_positions_from_ibkr(self):
        """Startup performs the immediate source-of-truth position sync once."""
        ib = MagicMock()
        ib.accountSummary.return_value = [_nl_item('5000.0')]
        ib.positions.return_value = []
        engine = _make_engine(ib)
        with patch.object(engine, '_sync_positions_from_ibkr') as mock_sync, \
             patch.object(engine, '_update_position_prices'),                \
             patch.object(engine, '_write_dashboard_data'):
            engine._initialize()
        assert mock_sync.call_count == 1

    def test_updates_prices_when_positions_exist(self):
        """Prices are updated during the immediate startup audit when positions exist."""
        ib = MagicMock()
        ib.accountSummary.return_value = [_nl_item('5000.0')]
        ib.positions.return_value = []
        engine = _make_engine(ib, state={'AAPL': {
            'price': 150.0, 'qty': 10, 'stop_loss': 140.0,
            'stop_dist': 10.0,
            'volume': 5000000, 'score': 70.0, 'time': '2026-01-01T10:00:00',
        }})
        with patch.object(engine, '_sync_positions_from_ibkr'),    \
             patch.object(engine, '_audit_stop_orders'),              \
             patch.object(engine, '_update_position_prices') as mock_up, \
             patch.object(engine, '_write_dashboard_data'):
            engine._initialize()
        assert mock_up.call_count == 1

    def test_audits_stops_when_positions_exist(self):
        ib = MagicMock()
        ib.accountSummary.return_value = [_nl_item('5000.0')]
        ib.positions.return_value = []
        engine = _make_engine(ib, state={'AAPL': {
            'price': 150.0, 'qty': 10, 'stop_loss': 140.0,
            'stop_dist': 10.0,
            'volume': 5000000, 'score': 70.0, 'time': '2026-01-01T10:00:00',
        }})
        with patch.object(engine, '_sync_positions_from_ibkr'),       \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch.object(engine, '_update_position_prices'),          \
             patch.object(engine, '_write_dashboard_data'):
            engine._initialize()
        assert mock_audit.call_count == 1

    def test_immediate_audit_when_unprotected_positions_exist(self):
        """Unprotected positions receive the immediate startup audit without waiting."""
        ib = MagicMock()
        ib.accountSummary.return_value = [_nl_item('5000.0')]
        ib.positions.return_value = []
        engine = _make_engine(ib, state={'AAPL': {
            'price': 150.0, 'qty': 10, 'stop_loss': 0.0,
            'volume': 5000000, 'score': 70.0, 'time': '2026-01-01T10:00:00',
        }})
        with patch.object(engine, '_sync_positions_from_ibkr') as mock_sync, \
             patch.object(engine, '_audit_stop_orders') as mock_audit,       \
             patch.object(engine, '_update_position_prices'),                \
             patch.object(engine, '_write_dashboard_data'):
            engine._initialize()

        assert mock_sync.call_count == 1
        assert mock_audit.call_count == 1

    def test_skips_audit_when_no_positions(self):
        ib = MagicMock()
        ib.accountSummary.return_value = [_nl_item('5000.0')]
        ib.positions.return_value = []
        engine = _make_engine(ib, state={})
        with patch.object(engine, '_sync_positions_from_ibkr'),       \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch.object(engine, '_update_position_prices'),          \
             patch.object(engine, '_write_dashboard_data'):
            engine._initialize()
        mock_audit.assert_not_called()

    def test_skips_price_update_when_no_positions(self):
        ib = MagicMock()
        ib.accountSummary.return_value = [_nl_item('5000.0')]
        ib.positions.return_value = []
        engine = _make_engine(ib, state={})
        with patch.object(engine, '_sync_positions_from_ibkr'),    \
             patch.object(engine, '_update_position_prices') as mock_up, \
             patch.object(engine, '_write_dashboard_data'):
            engine._initialize()
        mock_up.assert_not_called()

    def test_writes_dashboard_at_end(self):
        """Dashboard is written after the immediate startup snapshot."""
        ib = MagicMock()
        ib.accountSummary.return_value = [_nl_item('5000.0')]
        ib.positions.return_value = []
        engine = _make_engine(ib)
        with patch.object(engine, '_sync_positions_from_ibkr'), \
             patch.object(engine, '_write_dashboard_data') as mock_wd:
            engine._initialize()
        mock_wd.assert_called_once_with(connected=True)

    def test_initialize_does_not_sleep_until_pre_entry_sync(self):
        """Startup must enter the main loop promptly so the 06:30 prefilter can run."""
        ib = MagicMock()
        ib.accountSummary.return_value = [_nl_item('5000.0')]
        ib.positions.return_value = []
        ib.reqAllOpenOrders.return_value = []
        engine = _make_engine(ib)
        fake_now = pytz.timezone('US/Eastern').localize(datetime(2026, 5, 19, 6, 30, 0))

        with patch.object(engine, '_write_dashboard_data'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine._initialize()

        ib.sleep.assert_not_called()

    def test_initialize_called_before_run_cycle_in_run(self):
        """run() must call _initialize() before the first run_cycle()."""
        ib = MagicMock()
        ib.accountSummary.return_value = [_nl_item('5000.0')]
        ib.positions.return_value = []
        engine = _make_engine(ib)
        call_order = []

        def fake_init():
            call_order.append('init')

        def fake_cycle():
            call_order.append('cycle')
            raise SystemExit(0)   # SystemExit is BaseException, not caught by except Exception

        with patch.object(engine, '_initialize', side_effect=fake_init), \
             patch.object(engine, 'run_cycle', side_effect=fake_cycle),  \
             patch.object(engine, '_write_dashboard_data'):
            with pytest.raises(SystemExit):
                engine.run()

        assert call_order.index('init') < call_order.index('cycle')

    def test_run_logs_runtime_exception_traceback(self):
        """Runtime loop exceptions must use logger.exception so stack traces survive."""
        ib = MagicMock()
        ib.isConnected.return_value = True
        ib.sleep.side_effect = SystemExit(0)
        engine = _make_engine(ib)

        with patch.object(engine, '_initialize'), \
             patch.object(engine, 'run_cycle', side_effect=RuntimeError('boom')), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_alert'), \
             patch.object(eng_mod.logger, 'exception') as mock_exception:
            with pytest.raises(SystemExit):
                engine.run()

        mock_exception.assert_called_once_with("RUNTIME ERROR")


# ── _update_position_prices — unrealized P&L ──────────────────────────────────

class TestUnrealizedPnl:
    def _ticker(self, price):
        t = MagicMock()
        t.marketPrice.return_value = price
        t.last  = price
        t.close = price
        return t

    def _engine_with_pos(self, entry, qty, cur_price):
        ib = MagicMock()
        ib.reqTickers.return_value = [self._ticker(cur_price)]
        state = {
            'TSLA': {
                'price': entry, 'qty': qty,
                'stop_loss': entry - 5,
                'volume': 1000000, 'score': 75.0,
                'time': '2026-01-01T10:00:00',
            }
        }
        engine = _make_engine(ib, state=state)
        return engine

    def test_unrealized_gain_stored(self):
        engine = self._engine_with_pos(entry=100.0, qty=10, cur_price=110.0)
        engine._update_position_prices()
        assert engine.state['TSLA']['unrealized_pnl']     == 100.0  # (110-100)*10
        assert engine.state['TSLA']['unrealized_pnl_pct'] == 10.0   # 10%

    def test_unrealized_loss_stored(self):
        engine = self._engine_with_pos(entry=100.0, qty=5, cur_price=90.0)
        engine._update_position_prices()
        assert engine.state['TSLA']['unrealized_pnl']     == -50.0  # (90-100)*5
        assert engine.state['TSLA']['unrealized_pnl_pct'] == -10.0  # -10%

    def test_breakeven_stored(self):
        engine = self._engine_with_pos(entry=100.0, qty=8, cur_price=100.0)
        engine._update_position_prices()
        assert engine.state['TSLA']['unrealized_pnl']     == 0.0
        assert engine.state['TSLA']['unrealized_pnl_pct'] == 0.0

    def test_current_price_stored(self):
        engine = self._engine_with_pos(entry=100.0, qty=10, cur_price=115.0)
        engine._update_position_prices()
        assert engine.state['TSLA']['current_price'] == 115.0

    def test_break_even_armed_after_r_based_threshold(self):
        engine = self._engine_with_pos(
            entry=100.0,
            qty=10,
            cur_price=120.0,
        )
        engine.state['TSLA']['stop_dist'] = 20.0
        engine._update_position_prices()

        assert engine.state['TSLA']['break_even_armed'] is True
        assert engine.state['TSLA']['break_even_target_price'] == pytest.approx(120.0)
        assert engine.state['TSLA']['effective_stop'] == pytest.approx(100.0)

    def test_percent_trail_keeps_broker_stop_as_effective_stop(self):
        engine = self._engine_with_pos(entry=100.0, qty=10, cur_price=110.0)
        engine.state['TSLA']['stop_mode'] = 'percent'
        engine.state['TSLA']['stop_loss'] = 94.0
        engine.state['TSLA']['stop_dist'] = 5.0

        engine._update_position_prices()

        assert engine.state['TSLA']['peak_price'] == pytest.approx(110.0)
        assert engine.state['TSLA']['effective_stop'] == pytest.approx(94.0)

    def test_pnl_rounded_to_two_decimals(self):
        engine = self._engine_with_pos(entry=33.33, qty=3, cur_price=34.00)
        engine._update_position_prices()
        pnl = engine.state['TSLA']['unrealized_pnl']
        assert pnl == round(pnl, 2)

    def test_pnl_not_computed_when_price_unavailable(self):
        import numpy as np
        ib = MagicMock()
        ticker = MagicMock()
        ticker.marketPrice.return_value = float('nan')
        ticker.last  = float('nan')
        ticker.close = float('nan')
        ib.reqTickers.return_value = [ticker]
        state = {'MSFT': {
            'price': 300.0, 'qty': 5,
            'stop_loss': 280.0,
            'volume': 5000000, 'score': 70.0,
            'time': '2026-01-01T10:00:00',
        }}
        engine = _make_engine(ib, state=state)
        engine._update_position_prices()
        assert 'unrealized_pnl' not in engine.state['MSFT']

    def test_missing_once_position_skips_market_data_refresh(self):
        ib = MagicMock()
        state = {'DELL': {
            'price': 295.29, 'qty': 2,
            'stop_loss': 0.0,
            'volume': 0, 'score': None,
            'time': '2026-05-26T03:51:33-04:00',
        }}
        engine = _make_engine(ib, state=state)
        engine._missing_position_counts = {'DELL': 1}

        engine._update_position_prices()

        ib.reqTickers.assert_not_called()
        ib.reqHistoricalData.assert_not_called()
        assert 'current_price' not in engine.state['DELL']


# ── _log_startup_summary ───────────────────────────────────────────────────────

class TestLogStartupSummary:
    def test_no_positions_logs_ready(self, caplog):
        import logging
        engine = _make_engine(state={})
        with caplog.at_level(logging.INFO, logger='VelocityEngine'):
            engine._log_startup_summary(5000.0)
        combined = '\n'.join(caplog.messages)
        assert 'No open positions' in combined
        assert 'INIT READY' in combined

    def test_with_positions_logs_each_symbol(self, caplog):
        import logging
        state = {
            'AAPL': {
                'price': 150.0, 'qty': 10, 'current_price': 155.0,
                'stop_loss': 140.0,
                'unrealized_pnl': 50.0, 'unrealized_pnl_pct': 3.33,
                'volume': 5000000, 'score': 70.0, 'time': '2026-01-01T10:00:00',
            }
        }
        engine = _make_engine(state=state)
        with caplog.at_level(logging.INFO, logger='VelocityEngine'):
            engine._log_startup_summary(5000.0)
        combined = '\n'.join(caplog.messages)
        assert 'AAPL' in combined
        assert 'INIT READY' in combined
        assert 'Invested' in combined

    def test_ready_line_shows_correct_equity(self, caplog):
        import logging
        engine = _make_engine(state={})
        with caplog.at_level(logging.INFO, logger='VelocityEngine'):
            engine._log_startup_summary(12345.67)
        assert any('12345.67' in m for m in caplog.messages)


# ── _audit_stop_orders ────────────────────────────────────────────────────────

def _make_trade(
    symbol, action, order_type, order_id=1, aux_price=6.0, total_quantity=10,
    trail_stop_price=None, trailing_percent=None, good_after_time='',
    client_id=eng_mod.IB_CLIENT_ID,
):
    """Return a minimal Trade mock with .contract, .order attributes."""
    t               = MagicMock()
    t.contract      = MagicMock()
    t.contract.symbol = symbol
    t.order         = MagicMock()
    t.order.action  = action
    t.order.orderType = order_type
    t.order.orderId = order_id
    t.order.clientId = client_id
    t.order.auxPrice = aux_price
    t.order.totalQuantity = total_quantity
    t.order.trailStopPrice = (
        eng_mod.util.UNSET_DOUBLE if trail_stop_price is None else trail_stop_price
    )
    t.order.trailingPercent = (
        eng_mod.util.UNSET_DOUBLE if trailing_percent is None else trailing_percent
    )
    t.order.goodAfterTime = good_after_time
    return t


class TestAuditStopOrders:
    _POS = {
        'price': 100.0, 'qty': 10, 'stop_loss': 94.0,
        'volume': 1000000, 'score': 75.0, 'time': '2026-01-01T10:00:00',
    }

    def test_no_action_when_trail_exists(self):
        """TRAIL SELL already present — nothing cancelled, nothing placed."""
        ib = MagicMock()
        ib.openTrades.return_value = [_make_trade('AAPL', 'SELL', 'TRAIL')]
        engine = _make_engine(ib, state={'AAPL': dict(self._POS)})
        with patch.object(engine, '_stop_good_after_time', return_value=''):
            engine._audit_stop_orders()
        ib.cancelOrder.assert_not_called()
        ib.placeOrder.assert_not_called()

    def test_existing_trail_before_stop_gate_is_delayed_to_932_et(self):
        """Existing GTC TRAIL orders must not activate before the stop gate."""
        gate = '20260605 09:32:00 US/Eastern'
        ib = MagicMock()
        trail = _make_trade('AAPL', 'SELL', 'TRAIL', good_after_time='')
        modified = MagicMock()
        modified.orderStatus.status = 'Submitted'
        ib.reqAllOpenOrders.return_value = [trail]
        ib.openTrades.return_value = []
        ib.placeOrder.return_value = modified
        engine = _make_engine(ib, state={'AAPL': dict(self._POS)})

        with patch.object(engine, '_stop_good_after_time', return_value=gate):
            engine._audit_stop_orders()

        ib.cancelOrder.assert_called_once_with(trail.order)
        assert ib.placeOrder.call_count == 1
        placed_contract, placed_order = ib.placeOrder.call_args[0]
        assert placed_contract is trail.contract
        assert placed_order is not trail.order
        assert placed_order.orderType == 'TRAIL'
        assert placed_order.goodAfterTime == gate
        assert placed_order.totalQuantity == pytest.approx(10)

    def test_existing_trail_from_different_client_is_not_modified(self):
        """Do not try to control protective orders owned by another IB client id."""
        gate = '20260605 09:32:00 US/Eastern'
        ib = MagicMock()
        trail = _make_trade('AAPL', 'SELL', 'TRAIL', good_after_time='', client_id=99)
        ib.reqAllOpenOrders.return_value = [trail]
        ib.openTrades.return_value = []
        engine = _make_engine(ib, state={'AAPL': dict(self._POS)})

        with patch.object(engine, '_stop_good_after_time', return_value=gate):
            engine._audit_stop_orders()

        ib.cancelOrder.assert_not_called()
        ib.placeOrder.assert_not_called()

    def test_existing_later_gate_is_removed_after_stop_gate_has_passed(self):
        """A stale future GAT must be removed once the configured stop gate has passed."""
        ib = MagicMock()
        trail = _make_trade(
            'AAPL', 'SELL', 'TRAIL',
            good_after_time='20260605 10:00:00 US/Eastern',
        )
        replacement_trade = MagicMock()
        replacement_trade.orderStatus.status = 'Submitted'
        ib.reqAllOpenOrders.return_value = [trail]
        ib.openTrades.return_value = []
        ib.placeOrder.return_value = replacement_trade
        engine = _make_engine(ib, state={'AAPL': dict(self._POS)})

        with patch.object(engine, '_stop_good_after_time', return_value=''):
            engine._audit_stop_orders()

        ib.cancelOrder.assert_called_once_with(trail.order)
        placed_order = ib.placeOrder.call_args[0][1]
        assert placed_order.goodAfterTime == ''

    def test_accepts_ib_percent_trail_when_aux_is_unset(self):
        """IBKR can return percent TRAIL orders with auxPrice left as UNSET_DOUBLE."""
        ib = MagicMock()
        ib.openTrades.return_value = [
            _make_trade(
                'AAPL', 'SELL', 'TRAIL',
                aux_price=eng_mod.util.UNSET_DOUBLE,
                trail_stop_price=95.0,
                trailing_percent=5.0,
            )
        ]
        state = {'AAPL': dict(self._POS)}
        state['AAPL'].pop('stop_dist', None)
        state['AAPL']['stop_loss'] = 0.0
        engine = _make_engine(ib, state=state)

        with patch.object(engine, '_stop_good_after_time', return_value=''):
            engine._audit_stop_orders()

        ib.cancelOrder.assert_not_called()
        ib.placeOrder.assert_not_called()
        assert engine.state['AAPL']['stop_loss'] == pytest.approx(95.0)
        assert engine.state['AAPL']['effective_stop'] == pytest.approx(95.0)
        assert engine.state['AAPL']['stop_dist'] == pytest.approx(5.0)
        assert engine.state['AAPL']['stop_mode'] == 'percent'
        assert engine.state['AAPL']['trailing_percent'] == pytest.approx(5.0)

    def test_audit_uses_all_open_orders_feed_before_rebuilding_stop(self):
        """Existing GTC stops may be visible through reqAllOpenOrders before openTrades."""
        ib = MagicMock()
        ib.reqAllOpenOrders.return_value = [
            _make_trade(
                'AAPL', 'SELL', 'TRAIL',
                aux_price=eng_mod.util.UNSET_DOUBLE,
                trail_stop_price=95.0,
                trailing_percent=5.0,
            )
        ]
        ib.openTrades.return_value = []
        engine = _make_engine(ib, state={'AAPL': dict(self._POS)})

        with patch.object(engine, '_stop_good_after_time', return_value=''):
            engine._audit_stop_orders()

        ib.reqAllOpenOrders.assert_called_once()
        ib.cancelOrder.assert_not_called()
        ib.placeOrder.assert_not_called()

    def test_missing_once_position_is_not_audited(self):
        """Avoid placing orphan protection when IBKR has not confirmed the position."""
        ib, _ = self._make_ib_with_history()
        engine = _make_engine(ib, state={'DELL': dict(self._POS)})
        engine._missing_position_counts = {'DELL': 1}

        engine._audit_stop_orders()

        ib.reqHistoricalData.assert_not_called()
        ib.placeOrder.assert_not_called()

    def test_cancels_non_trail_sell_order(self):
        """Non-TRAIL SELL (e.g. LMT take-profit) must be cancelled."""
        ib, _ = self._make_ib_with_history()
        lmt_trade = _make_trade('AAPL', 'SELL', 'LMT', order_id=99)
        ib.openTrades.return_value = [lmt_trade]

        engine = _make_engine(ib, state={'AAPL': dict(self._POS)})
        engine._audit_stop_orders()

        ib.cancelOrder.assert_called_once_with(lmt_trade.order)

    def _make_ib_with_history(self):
        """Return (ib_mock, df) with enough daily bars for ATR_CHAND computation."""
        import pandas as pd, numpy as np
        ib = MagicMock()
        ib.openTrades.return_value       = []
        ib.qualifyContracts.return_value = [MagicMock()]
        # whatIfOrder pre-flight passes by default (warningText='')
        ib.whatIfOrder.return_value.warningText = ''
        n     = 250
        close = 100 + np.arange(n) * 0.1
        high  = close + 0.2
        low   = close - 0.2
        idx   = pd.date_range('2025-01-01', periods=n, freq='B')
        df    = pd.DataFrame({'open': close, 'high': high, 'low': low,
                              'close': close, 'volume': 1_000_000}, index=idx)
        from src.indicators import apply_all
        df = apply_all(df)
        from ib_async import util as _util
        ib.reqHistoricalData.return_value = [MagicMock() for _ in range(n)]
        _util.df = lambda _bars: df
        return ib, df

    def test_places_trail_when_no_sell_order(self):
        """Position has no SELL orders → TRAIL SELL placed and state updated."""
        ib, _ = self._make_ib_with_history()
        placed_orders = []
        stop_trade = MagicMock()
        stop_trade.orderStatus.status = 'PreSubmitted'   # IB accepted
        ib.placeOrder.side_effect = lambda c, o: (placed_orders.append(o), stop_trade)[1]

        state = {'AAPL': dict(self._POS)}
        engine = _make_engine(ib, state=state)
        engine._audit_stop_orders()

        assert len(placed_orders) == 1
        order = placed_orders[0]
        assert order.action    == 'SELL'
        assert order.orderType == 'TRAIL'
        assert order.tif       == 'GTC'
        # State must be updated with the new stop distance
        assert 'stop_dist' in engine.state['AAPL']
        assert 'stop_loss' in engine.state['AAPL']

    def test_audit_trail_good_after_time_set_only_before_stop_gate(self):
        """Pre-market audit can defer TRAIL activation; after 09:32 it must omit past GAT."""
        tz_ny = pytz.timezone('US/Eastern')

        ib, _ = self._make_ib_with_history()
        placed_orders = []
        stop_trade = MagicMock()
        stop_trade.orderStatus.status = 'PreSubmitted'
        ib.placeOrder.side_effect = lambda c, o: (placed_orders.append(o), stop_trade)[1]
        engine = _make_engine(ib, state={'AAPL': dict(self._POS)})

        before_gate = tz_ny.localize(datetime(2024, 6, 5, 9, 31))
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = before_gate
            mock_dt.fromisoformat = datetime.fromisoformat
            engine._audit_stop_orders()

        assert '09:32:00 US/Eastern' in placed_orders[0].goodAfterTime

        ib, _ = self._make_ib_with_history()
        placed_orders = []
        stop_trade = MagicMock()
        stop_trade.orderStatus.status = 'PreSubmitted'
        ib.placeOrder.side_effect = lambda c, o: (placed_orders.append(o), stop_trade)[1]
        engine = _make_engine(ib, state={'AAPL': dict(self._POS)})

        after_gate = tz_ny.localize(datetime(2024, 6, 5, 9, 33))
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = after_gate
            mock_dt.fromisoformat = datetime.fromisoformat
            engine._audit_stop_orders()

        assert placed_orders[0].goodAfterTime == ''

    def test_rejected_order_does_not_update_state(self):
        """If IB returns Inactive/rejected status, state must NOT be written."""
        ib, _ = self._make_ib_with_history()
        stop_trade = MagicMock()
        stop_trade.orderStatus.status = 'Inactive'   # IB rejected
        ib.placeOrder.return_value = stop_trade

        state = {'AAPL': dict(self._POS)}
        # Remove stop_dist so we can detect if it was (wrongly) written
        state['AAPL'].pop('stop_dist', None)
        engine = _make_engine(ib, state=state)
        engine._audit_stop_orders()

        ib.placeOrder.assert_called_once()
        assert 'stop_dist' not in engine.state['AAPL']

    def test_api_cancelled_order_does_not_update_state(self):
        """ApiCancelled status (IBKR pre-market edge case) must not write state."""
        ib, _ = self._make_ib_with_history()
        stop_trade = MagicMock()
        stop_trade.orderStatus.status = 'ApiCancelled'
        ib.placeOrder.return_value = stop_trade

        state = {'AAPL': dict(self._POS)}
        state['AAPL'].pop('stop_dist', None)
        engine = _make_engine(ib, state=state)
        engine._audit_stop_orders()

        assert 'stop_dist' not in engine.state['AAPL']

    def test_skips_symbol_with_zero_qty(self):
        """Positions with qty=0 must not trigger a placeOrder call."""
        ib = MagicMock()
        ib.openTrades.return_value = []
        pos = dict(self._POS)
        pos['qty'] = 0
        engine = _make_engine(ib, state={'AAPL': pos})
        engine._audit_stop_orders()
        ib.placeOrder.assert_not_called()

    def test_skips_when_state_empty(self):
        """No positions → openTrades not even checked."""
        ib = MagicMock()
        engine = _make_engine(ib, state={})
        engine._audit_stop_orders()
        ib.openTrades.assert_not_called()

    def test_keeps_trail_cancels_lmt_for_same_symbol(self):
        """One TRAIL and one LMT SELL on same symbol — cancel LMT, keep TRAIL."""
        ib = MagicMock()
        trail_trade = _make_trade('TSLA', 'SELL', 'TRAIL', order_id=10)
        lmt_trade   = _make_trade('TSLA', 'SELL', 'LMT',   order_id=11)
        ib.openTrades.return_value = [trail_trade, lmt_trade]
        engine = _make_engine(ib, state={'TSLA': dict(self._POS)})
        with patch.object(engine, '_stop_good_after_time', return_value=''):
            engine._audit_stop_orders()
        # LMT cancelled; TRAIL kept; no new order placed
        ib.cancelOrder.assert_called_once_with(lmt_trade.order)
        ib.placeOrder.assert_not_called()

    def test_cancels_and_rebuilds_trail_when_qty_mismatches_state(self):
        """A protective stop with stale quantity must be replaced, not trusted."""
        ib, _ = self._make_ib_with_history()
        stale_trail = _make_trade(
            'AAPL', 'SELL', 'TRAIL', order_id=7, total_quantity=3
        )
        ib.openTrades.return_value = [stale_trail]

        placed_orders = []
        stop_trade = MagicMock()
        stop_trade.orderStatus.status = 'PreSubmitted'
        ib.placeOrder.side_effect = lambda c, o: (placed_orders.append(o), stop_trade)[1]

        engine = _make_engine(ib, state={'AAPL': dict(self._POS)})
        engine._audit_stop_orders()

        ib.cancelOrder.assert_called_once_with(stale_trail.order)
        assert len(placed_orders) == 1
        assert placed_orders[0].totalQuantity == pytest.approx(self._POS['qty'])

    def test_cancels_and_rebuilds_trail_when_aux_price_invalid(self):
        """A zero-distance trail is not protective and must be rebuilt."""
        ib, _ = self._make_ib_with_history()
        bad_trail = _make_trade(
            'AAPL', 'SELL', 'TRAIL', order_id=8, aux_price=0.0,
            total_quantity=self._POS['qty'],
        )
        ib.openTrades.return_value = [bad_trail]

        placed_orders = []
        stop_trade = MagicMock()
        stop_trade.orderStatus.status = 'PreSubmitted'
        ib.placeOrder.side_effect = lambda c, o: (placed_orders.append(o), stop_trade)[1]

        engine = _make_engine(ib, state={'AAPL': dict(self._POS)})
        engine._audit_stop_orders()

        ib.cancelOrder.assert_called_once_with(bad_trail.order)
        assert len(placed_orders) == 1
        assert placed_orders[0].auxPrice > 0

    def test_cancels_and_rebuilds_trail_when_ib_unset_fields_have_no_stop(self):
        """UNSET_DOUBLE without trailStopPrice/trailingPercent is not protection."""
        ib, _ = self._make_ib_with_history()
        bad_trail = _make_trade(
            'AAPL', 'SELL', 'TRAIL', order_id=9,
            aux_price=eng_mod.util.UNSET_DOUBLE,
            total_quantity=self._POS['qty'],
        )
        ib.openTrades.return_value = [bad_trail]

        placed_orders = []
        stop_trade = MagicMock()
        stop_trade.orderStatus.status = 'PreSubmitted'
        ib.placeOrder.side_effect = lambda c, o: (placed_orders.append(o), stop_trade)[1]

        engine = _make_engine(ib, state={'AAPL': dict(self._POS)})
        engine._audit_stop_orders()

        ib.cancelOrder.assert_called_once_with(bad_trail.order)
        assert len(placed_orders) == 1
        assert placed_orders[0].auxPrice > 0

    def test_ignores_buy_orders_for_same_symbol(self):
        """Pending BUY order for the same symbol must not interfere with audit."""
        ib, _ = self._make_ib_with_history()
        buy_trade = _make_trade('AAPL', 'BUY', 'LMT', order_id=5)
        ib.openTrades.return_value = [buy_trade]

        placed_orders = []
        ib.placeOrder.side_effect = lambda c, o: (placed_orders.append(o), MagicMock())[1]

        engine = _make_engine(ib, state={'AAPL': dict(self._POS)})
        engine._audit_stop_orders()

        # BUY order not cancelled; a TRAIL SELL must be placed (no existing SELL order)
        ib.cancelOrder.assert_not_called()
        assert len(placed_orders) == 1
        assert placed_orders[0].orderType == 'TRAIL'


# ── _preflight_order ──────────────────────────────────────────────────────────

class TestPreflightOrder:
    def _engine(self):
        return _make_engine(MagicMock())

    def test_returns_true_when_no_warning(self):
        """Empty warningText → order is acceptable."""
        engine = self._engine()
        engine.ib.whatIfOrder.return_value.warningText = ''
        contract = MagicMock()
        order    = MagicMock()
        assert engine._preflight_order(contract, order, 'AAPL') is True

    def test_preflight_uses_transmitted_copy_without_mutating_live_order(self):
        """IBKR what-if requires transmit=True, but live bracket parent stays held."""
        engine = self._engine()
        engine.ib.whatIfOrder.return_value.warningText = ''
        contract = MagicMock()
        order = MagicMock()
        order.action = 'BUY'
        order.orderType = 'LMT'
        order.transmit = False
        order.whatIf = False

        assert engine._preflight_order(contract, order, 'AAPL') is True

        sent_order = engine.ib.whatIfOrder.call_args[0][1]
        assert sent_order is not order
        assert sent_order.transmit is True
        assert sent_order.whatIf is True
        assert order.transmit is False
        assert order.whatIf is False

    def test_unwraps_list_return_from_whatif_order(self):
        """Some IBKR API versions return [OrderState] instead of OrderState."""
        engine = self._engine()
        state = MagicMock()
        state.warningText = ''
        engine.ib.whatIfOrder.return_value = [state]
        contract = MagicMock()
        order    = MagicMock()

        assert engine._preflight_order(contract, order, 'AAPL') is True

    def test_returns_false_when_warning_present(self):
        """Non-empty warningText → IB signals rejection."""
        engine = self._engine()
        engine.ib.whatIfOrder.return_value.warningText = 'Order would exceed buying power'
        contract = MagicMock()
        order    = MagicMock()
        assert engine._preflight_order(contract, order, 'AAPL') is False

    def test_returns_true_on_whatif_exception_for_existing_position_protective_sell(self):
        """Stop-audit SELL may fail open only when an existing position needs protection."""
        engine = self._engine()
        engine.state = {'AAPL': {'qty': 1}}
        engine.ib.whatIfOrder.side_effect = RuntimeError('gateway timeout')
        contract = MagicMock()
        order    = MagicMock()
        order.action = 'SELL'
        assert engine._preflight_order(
            contract,
            order,
            'AAPL',
            allow_protective_sell_fail_open=True,
        ) is True

    def test_returns_false_on_whatif_exception_for_new_child_sell(self):
        """Entry child stops must not fail open before a position exists."""
        engine = self._engine()
        engine.ib.whatIfOrder.side_effect = RuntimeError('gateway timeout')
        contract = MagicMock()
        order    = MagicMock()
        order.action = 'SELL'
        assert engine._preflight_order(contract, order, 'AAPL') is False

    def test_returns_false_on_whatif_exception_for_buy(self):
        """If BUY preflight cannot be verified, new exposure must be blocked."""
        engine = self._engine()
        engine.ib.whatIfOrder.side_effect = RuntimeError('gateway timeout')
        contract = MagicMock()
        order    = MagicMock()
        order.action = 'BUY'
        assert engine._preflight_order(contract, order, 'AAPL') is False

    def test_audit_skips_placeorder_when_preflight_fails(self):
        """When preflight returns False, _audit_stop_orders must not call placeOrder."""
        ib, _ = TestAuditStopOrders()._make_ib_with_history()
        ib.whatIfOrder.return_value.warningText = 'Order rejected: market closed'

        engine = _make_engine(ib, state={'AAPL': {
            'price': 100.0, 'qty': 10, 'stop_loss': 94.0,
            'volume': 1000000, 'score': 75.0, 'time': '2026-01-01T10:00:00',
        }})
        engine._audit_stop_orders()

        ib.placeOrder.assert_not_called()

    def test_whitespace_only_warning_is_ignored(self):
        """warningText containing only whitespace must be treated as empty."""
        engine = self._engine()
        engine.ib.whatIfOrder.return_value.warningText = '   '
        contract = MagicMock()
        order    = MagicMock()
        assert engine._preflight_order(contract, order, 'AAPL') is True


# ── _maybe_pre_entry_sync_audit ───────────────────────────────────────────────

class TestPreEntrySyncAudit:
    _TZ_NY = pytz.timezone('US/Eastern')

    def test_waits_until_checkpoint_time(self):
        from src.config import PRE_ENTRY_SYNC_TIME
        ib = MagicMock()
        engine = _make_engine(ib, state={'AAPL': {
            'price': 100.0,
            'qty': 1.0,
            'stop_loss': 94.0,
            'stop_dist': 6.0,
            'time': '2026-05-19T10:00:00-04:00',
        }})

        h, m = PRE_ENTRY_SYNC_TIME
        fake_now = self._TZ_NY.localize(datetime(2026, 5, 19, h, m - 1, 0))
        with patch.object(engine, '_sync_positions_from_ibkr') as mock_sync, \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine._maybe_pre_entry_sync_audit()

        mock_sync.assert_not_called()
        mock_audit.assert_not_called()
        ib.sleep.assert_not_called()

    def test_runs_once_after_checkpoint(self):
        from src.config import PRE_ENTRY_SYNC_TIME
        ib = MagicMock()
        engine = _make_engine(ib, state={'AAPL': {
            'price': 100.0,
            'qty': 1.0,
            'stop_loss': 94.0,
            'stop_dist': 6.0,
            'time': '2026-05-19T10:00:00-04:00',
        }})

        h, m = PRE_ENTRY_SYNC_TIME
        fake_now = self._TZ_NY.localize(datetime(2026, 5, 19, h, m, 0))
        with patch.object(engine, '_sync_positions_from_ibkr') as mock_sync, \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch.object(engine, '_update_position_prices') as mock_prices, \
             patch.object(engine, '_write_dashboard_data') as mock_dashboard, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine._maybe_pre_entry_sync_audit()

        mock_sync.assert_called_once()
        mock_audit.assert_called_once()
        mock_prices.assert_called_once()
        mock_dashboard.assert_called_once_with(connected=True)
        assert engine._last_pre_entry_sync_date == '2026-05-19'
        assert engine._last_audit_date == '2026-05-19'
        ib.sleep.assert_not_called()

    def test_skips_duplicate_same_day_checkpoint(self):
        from src.config import PRE_ENTRY_SYNC_TIME
        ib = MagicMock()
        engine = _make_engine(ib, state={'AAPL': {
            'price': 100.0,
            'qty': 1.0,
            'stop_loss': 94.0,
            'stop_dist': 6.0,
            'time': '2026-05-19T10:00:00-04:00',
        }})
        engine._last_pre_entry_sync_date = '2026-05-19'

        h, m = PRE_ENTRY_SYNC_TIME
        fake_now = self._TZ_NY.localize(datetime(2026, 5, 19, h, m + 5, 0))
        with patch.object(engine, '_sync_positions_from_ibkr') as mock_sync, \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine._maybe_pre_entry_sync_audit()

        mock_sync.assert_not_called()
        mock_audit.assert_not_called()
        ib.sleep.assert_not_called()

    def test_startup_audit_after_checkpoint_marks_checkpoint_covered(self):
        from src.config import PRE_ENTRY_SYNC_TIME
        ib = MagicMock()
        engine = _make_engine(ib, state={'AAPL': {
            'price': 100.0,
            'qty': 1.0,
            'stop_loss': 94.0,
            'stop_dist': 6.0,
            'time': '2026-05-19T10:00:00-04:00',
        }})

        h, m = PRE_ENTRY_SYNC_TIME
        audit_time = self._TZ_NY.localize(datetime(2026, 5, 19, h, m + 10, 0))
        engine._last_audit_at = audit_time
        fake_now = self._TZ_NY.localize(datetime(2026, 5, 19, h, m + 11, 0))
        with patch.object(engine, '_sync_positions_from_ibkr') as mock_sync, \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine._maybe_pre_entry_sync_audit()

        mock_sync.assert_not_called()
        mock_audit.assert_not_called()
        assert engine._last_pre_entry_sync_date == '2026-05-19'
        ib.sleep.assert_not_called()
