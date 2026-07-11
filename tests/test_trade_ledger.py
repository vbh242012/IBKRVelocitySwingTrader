"""Tests for the per-trade measurement ledger and its engine hooks.

The ledger is the "fix measurement" deliverable: every live round trip must
produce a queryable record (entry, MFE/MAE, exit price, exit reason, P&L)
without ever being able to interrupt the trading path.
"""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import pytz

from src.trade_ledger import TradeLedger

_TZ_NY = pytz.timezone('US/Eastern')


def _ledger(tmp_path) -> TradeLedger:
    return TradeLedger(path=str(tmp_path / "trade_ledger.json"))


def _entry_record(**updates):
    rec = {
        'fill_price': 100.0,
        'price': 100.0,
        'time': '2026-07-06T10:00:00-04:00',
        'qty': 5.0,
        'entry_order_id': 42,
        'commission': 1.0,
        'score': 77.5,
        'entry_strategy': 'ma_cross',
        'strategy_profile': 'indicator_swing',
        'trailing_percent': 2.0,
        'spread_pct': 0.001,
        'volume_pace': 1.8,
    }
    rec.update(updates)
    return rec


# ── Ledger unit behavior ──────────────────────────────────────────────────────
class TestTradeLedger:
    def test_lifecycle_open_update_close(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.open_trade('AAPL', _entry_record())

        ledger.update_price('AAPL', 104.0)   # MFE sample
        ledger.update_price('AAPL', 97.0)    # MAE sample
        ledger.update_price('AAPL', 101.0)   # inside the range — no change

        ledger.close_trade(
            'AAPL',
            exit_price=102.0,
            exit_reason='trail_stop',
            exit_commission=1.0,
            exit_time='2026-07-10T11:00:00-04:00',
        )

        assert not ledger.has_open('AAPL')
        closed = ledger.recent_closed(5)
        assert len(closed) == 1
        rec = closed[0]
        assert rec['status'] == 'closed'
        assert rec['exit_reason'] == 'trail_stop'
        assert rec['exit_price'] == pytest.approx(102.0)
        assert rec['mfe_price'] == pytest.approx(104.0)
        assert rec['mae_price'] == pytest.approx(97.0)
        assert rec['mfe_pct'] == pytest.approx(4.0)
        assert rec['mae_pct'] == pytest.approx(-3.0)
        assert rec['gross_pnl'] == pytest.approx((102.0 - 100.0) * 5.0)
        assert rec['net_pnl'] == pytest.approx(10.0 - 1.0 - 1.0)
        assert rec['pnl_pct'] == pytest.approx(2.0)
        assert rec['commissions_complete'] is True
        assert rec['trading_days_held'] == 4  # Mon Jul 6 → Fri Jul 10 2026

    def test_persists_and_reloads(self, tmp_path):
        path = tmp_path / "trade_ledger.json"
        first = TradeLedger(path=str(path))
        first.open_trade('MSFT', _entry_record())
        first.close_trade('MSFT', exit_price=101.0, exit_reason='time_stop')

        second = TradeLedger(path=str(path))
        assert len(second.recent_closed()) == 1
        assert second.recent_closed()[0]['symbol'] == 'MSFT'

        # Open records survive a restart too.
        second.open_trade('NVDA', _entry_record())
        third = TradeLedger(path=str(path))
        assert third.has_open('NVDA')
        assert len(third.recent_closed()) == 1

    def test_corrupt_file_is_sidelined_not_fatal(self, tmp_path):
        path = tmp_path / "trade_ledger.json"
        path.write_text("{not valid json")

        ledger = TradeLedger(path=str(path))
        ledger.open_trade('AMD', _entry_record())

        assert ledger.has_open('AMD')
        sidecars = list(tmp_path.glob("trade_ledger.json.corrupt-*"))
        assert len(sidecars) == 1
        assert sidecars[0].read_text() == "{not valid json"

    def test_close_without_open_record_is_never_lost(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.close_trade('GHOST', exit_price=50.0, exit_reason='broker_exit')

        rec = ledger.recent_closed()[0]
        assert rec['note'] == 'closed_without_open_record'
        assert rec['exit_price'] == pytest.approx(50.0)
        assert rec['net_pnl'] is None  # no entry data → no fabricated P&L

    def test_recovery_open_does_not_replace_existing_record(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.open_trade('TSLA', _entry_record(score=88.0))
        ledger.open_trade(
            'TSLA', _entry_record(score=None, source='recovered_from_broker'),
            replace=False,
        )

        assert ledger._open['TSLA']['score'] == pytest.approx(88.0)
        assert ledger.recent_closed() == []

    def test_fresh_entry_supersedes_stale_open_record(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.open_trade('INTC', _entry_record())
        ledger.open_trade('INTC', _entry_record(fill_price=110.0))

        assert ledger._open['INTC']['entry_price'] == pytest.approx(110.0)
        closed = ledger.recent_closed()
        assert len(closed) == 1
        assert closed[0]['exit_reason'] == 'superseded_by_new_entry'

    def test_summary_counts_wins_and_losses(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.open_trade('A', _entry_record())
        ledger.close_trade('A', exit_price=105.0, exit_reason='trail_stop',
                           exit_commission=1.0)
        ledger.open_trade('B', _entry_record())
        ledger.close_trade('B', exit_price=98.0, exit_reason='hard_stop',
                           exit_commission=1.0)

        summary = ledger.summary()
        assert summary['closed_trades'] == 2
        assert summary['wins'] == 1
        assert summary['losses'] == 1
        assert summary['win_rate_pct'] == pytest.approx(50.0)


# ── Engine hooks ──────────────────────────────────────────────────────────────
def _bare_engine():
    """Engine via __new__ with only the attributes the tested paths touch."""
    from src.engine import VelocityEngine
    engine = VelocityEngine.__new__(VelocityEngine)
    engine.ib = MagicMock()
    engine.state = {}
    return engine


class TestEngineLedgerHooks:
    def test_ledger_call_is_noop_without_ledger(self):
        engine = _bare_engine()
        assert engine._ledger_call('open_trade', 'X', {}) is None

    def test_ledger_call_is_failsoft(self, tmp_path):
        engine = _bare_engine()
        engine._trade_ledger = _ledger(tmp_path)
        # Nonexistent method must warn, not raise.
        assert engine._ledger_call('no_such_method') is None

    def test_liquidate_records_reason_and_exit_fill(self, tmp_path):
        engine = _bare_engine()
        engine._trade_ledger = _ledger(tmp_path)
        engine.state = {'XOM': {'fill_price': 100.0, 'price': 100.0, 'qty': 4.0}}

        position = MagicMock()
        position.contract.symbol = 'XOM'
        position.position = 4.0
        engine.ib.positions.return_value = [position]
        engine.ib.loopUntil.return_value = [True]

        sell_trade = MagicMock()
        sell_trade.orderStatus.status = 'Filled'
        sell_trade.orderStatus.filled = 4.0
        sell_trade.orderStatus.avgFillPrice = 101.25
        sell_trade.order.orderId = 77
        engine.ib.placeOrder.return_value = sell_trade

        with patch.object(engine, '_cancel_open_orders_before_market_exit',
                          return_value=True), \
             patch.object(engine, 'save_state'):
            engine.liquidate('XOM', reason='hard_stop')

        assert engine.state['XOM']['exit_reason'] == 'hard_stop'
        assert engine.state['XOM']['exit_fill_price'] == pytest.approx(101.25)
        assert engine.state['XOM']['exit_order_id'] == 77
        assert engine.state['XOM']['pending_exit'] is True

    def test_sync_removal_closes_ledger_with_software_fill(self, tmp_path):
        engine = _bare_engine()
        engine._trade_ledger = _ledger(tmp_path)
        engine._trade_ledger.open_trade('XOM', _entry_record())
        engine.state = {'XOM': {
            'fill_price': 100.0, 'price': 100.0, 'qty': 5.0,
            'commission': 1.0,
            'exit_reason': 'hard_stop',
            'exit_fill_price': 101.25,
            'exit_fill_time': '2026-07-10T14:00:00-04:00',
            'exit_commission': 1.0,
        }}
        engine._missing_position_counts = {'XOM': 1}  # second miss confirms flat
        engine.ib.positions.return_value = []

        with patch.object(engine, '_force_exit_active', return_value=False), \
             patch.object(engine, '_cancel_orphaned_exit_orders'), \
             patch.object(engine, 'save_state'):
            engine._sync_positions_from_ibkr()

        assert 'XOM' not in engine.state
        rec = engine._trade_ledger.recent_closed()[0]
        assert rec['exit_reason'] == 'hard_stop'
        assert rec['exit_price'] == pytest.approx(101.25)
        assert rec['exit_price_source'] == 'software_fill'
        assert rec['net_pnl'] == pytest.approx((101.25 - 100.0) * 5.0 - 2.0)

    def test_broker_trail_fill_reconstructed_from_executions(self, tmp_path):
        engine = _bare_engine()
        engine._trade_ledger = _ledger(tmp_path)
        engine._trade_ledger.open_trade('XOM', _entry_record())
        engine.state = {'XOM': {
            'fill_price': 100.0, 'price': 100.0, 'qty': 5.0,
            'commission': 1.0,
            'stop_order_id': 55,
        }}
        engine._missing_position_counts = {'XOM': 1}
        engine.ib.positions.return_value = []

        fill = SimpleNamespace(
            contract=SimpleNamespace(symbol='XOM'),
            execution=SimpleNamespace(
                side='SLD', shares=5.0, price=99.0, orderId=55,
                time=_TZ_NY.localize(datetime(2026, 7, 10, 10, 15)),
            ),
            commissionReport=SimpleNamespace(commission=1.02),
            time=None,
        )
        engine.ib.fills.return_value = [fill]

        with patch.object(engine, '_force_exit_active', return_value=False), \
             patch.object(engine, '_cancel_orphaned_exit_orders'), \
             patch.object(engine, 'save_state'):
            engine._sync_positions_from_ibkr()

        rec = engine._trade_ledger.recent_closed()[0]
        assert rec['exit_reason'] == 'trail_stop'  # matched stop_order_id
        assert rec['exit_price'] == pytest.approx(99.0)
        assert rec['exit_price_source'] == 'broker_fills'
        assert rec['exit_commission'] == pytest.approx(1.02)

    def test_unmatched_broker_fill_uses_fallback_reason(self, tmp_path):
        engine = _bare_engine()
        engine._trade_ledger = _ledger(tmp_path)
        engine._trade_ledger.open_trade('XOM', _entry_record())
        engine.state = {'XOM': {
            'fill_price': 100.0, 'qty': 5.0, 'stop_order_id': 55,
        }}
        engine._missing_position_counts = {'XOM': 1}
        engine.ib.positions.return_value = []

        fill = SimpleNamespace(
            contract=SimpleNamespace(symbol='XOM'),
            execution=SimpleNamespace(side='SLD', shares=5.0, price=99.5,
                                      orderId=901, time=None),
            commissionReport=None,
            time=None,
        )
        engine.ib.fills.return_value = [fill]

        with patch.object(engine, '_force_exit_active', return_value=False), \
             patch.object(engine, '_cancel_orphaned_exit_orders'), \
             patch.object(engine, 'save_state'):
            engine._sync_positions_from_ibkr()

        rec = engine._trade_ledger.recent_closed()[0]
        assert rec['exit_reason'] == 'broker_exit'
        assert rec['exit_price'] == pytest.approx(99.5)

    def test_estimate_used_when_no_fills_available(self, tmp_path):
        engine = _bare_engine()
        engine._trade_ledger = _ledger(tmp_path)
        engine._trade_ledger.open_trade('XOM', _entry_record())
        engine.state = {'XOM': {
            'fill_price': 100.0, 'qty': 5.0, 'current_price': 98.4,
        }}
        engine._missing_position_counts = {'XOM': 1}
        engine.ib.positions.return_value = []
        engine.ib.fills.return_value = []
        engine.ib.reqExecutions.return_value = []

        with patch.object(engine, '_force_exit_active', return_value=False), \
             patch.object(engine, '_cancel_orphaned_exit_orders'), \
             patch.object(engine, 'save_state'):
            engine._sync_positions_from_ibkr()

        rec = engine._trade_ledger.recent_closed()[0]
        assert rec['exit_price_source'] == 'estimate'
        assert rec['exit_price'] == pytest.approx(98.4)
