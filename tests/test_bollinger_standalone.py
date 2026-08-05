"""
Tests for the standalone Bollinger mean-reversion strategy (2026-07-30
session): src/bollinger_standalone.py (entry evaluator, rank, exit
helpers), the new bollinger_reversion_standalone branch in
ExitsMixin._indicator_strategy_exit_required / manage_position_exits, the
new EntriesMixin._scan_and_enter_bollinger_standalone /
_place_bollinger_standalone_order entry path, and the prefilter's
independent bollinger_candidates tagging.

Reuses the existing _mock_ib()/_make_engine()/_freeze_all_datetimes()
helpers from tests/test_trailing_stop_scoring_screener.py rather than
duplicating the engine-construction boilerplate.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest
import pytz
from datetime import datetime
from unittest.mock import MagicMock, patch

from tests.test_trailing_stop_scoring_screener import (
    _mock_ib,
    _make_engine,
    _mock_price_ticker,
    _mock_fill,
    _freeze_all_datetimes,
)

from src.bollinger_standalone import (
    ENTRY_STRATEGY_NAME,
    evaluate_bollinger_standalone_entry,
    bollinger_standalone_rank,
    bollinger_standalone_midline_reclaim,
    bollinger_standalone_time_stop_due,
)
from src.config import (
    BOLLINGER_STANDALONE_HARD_STOP_PCT,
    BOLLINGER_STANDALONE_MIN_DOLLAR_VOL,
    BOLLINGER_STANDALONE_TIME_STOP_DAYS,
)

TZ_NY = pytz.timezone('US/Eastern')


def _base_ctx(**updates):
    ctx = {
        "live_price": 10.0,
        "close": 10.0,
        "bb_mid": 11.0,
        "spread_pct": 0.002,
        "dollar_vol_20d": BOLLINGER_STANDALONE_MIN_DOLLAR_VOL * 2,
        "bb_reclaim_lower": True,
        # get_technical_context() always includes a contract in production
        # (engine_scanner.py); the book-concentration checks in
        # _scan_and_enter_bollinger_standalone() rely on it being present.
        "contract": MagicMock(),
    }
    ctx.update(updates)
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Entry evaluator (pure function)
# ─────────────────────────────────────────────────────────────────────────────

class TestBollingerStandaloneEntryEvaluator:
    def test_passes_with_reclaim_and_liquidity(self):
        evaluation = evaluate_bollinger_standalone_entry(_base_ctx())
        assert evaluation.passed
        assert evaluation.failed == ()

    def test_rejects_without_bb_reclaim(self):
        evaluation = evaluate_bollinger_standalone_entry(_base_ctx(bb_reclaim_lower=False))
        assert not evaluation.passed
        assert "bb_reclaim_lower" in evaluation.failed

    def test_rejects_below_dollar_volume_floor(self):
        thin_ctx = _base_ctx(dollar_vol_20d=BOLLINGER_STANDALONE_MIN_DOLLAR_VOL / 2)
        evaluation = evaluate_bollinger_standalone_entry(thin_ctx)
        assert not evaluation.passed
        assert any(f.startswith("dollar_vol>=") for f in evaluation.failed)

    def test_rejects_wide_spread(self):
        wide_ctx = _base_ctx(spread_pct=0.05)
        evaluation = evaluate_bollinger_standalone_entry(wide_ctx)
        assert not evaluation.passed
        assert any(f.startswith("spread<=") for f in evaluation.failed)

    def test_rejects_missing_price(self):
        bad_ctx = _base_ctx(live_price=float('nan'), close=float('nan'))
        evaluation = evaluate_bollinger_standalone_entry(bad_ctx)
        assert not evaluation.passed
        assert "price>0" in evaluation.failed

    def test_no_trend_or_relative_strength_gate(self):
        """A beaten-down candidate (below MA50/MA200, negative RS) must still
        pass — this is the whole point of the standalone strategy."""
        beaten_down_ctx = _base_ctx(ma50=50.0, ma200=60.0, relative_strength_63d=-0.30)
        assert evaluate_bollinger_standalone_entry(beaten_down_ctx).passed


class TestBollingerStandaloneRank:
    def test_deeper_reclaim_ranks_higher(self):
        shallow = bollinger_standalone_rank(_base_ctx(live_price=10.5, bb_mid=11.0))
        deep = bollinger_standalone_rank(_base_ctx(live_price=9.0, bb_mid=11.0))
        assert deep > shallow

    def test_missing_inputs_rank_last(self):
        assert bollinger_standalone_rank(_base_ctx(bb_mid=float('nan'))) == float('-inf')


# ─────────────────────────────────────────────────────────────────────────────
# Exit helpers (pure functions)
# ─────────────────────────────────────────────────────────────────────────────

class TestBollingerStandaloneExitHelpers:
    def test_midline_reclaim_true_when_close_above_mid(self):
        assert bollinger_standalone_midline_reclaim({"close": 11.0, "BB_MID": 10.0})

    def test_midline_reclaim_false_when_close_below_mid(self):
        assert not bollinger_standalone_midline_reclaim({"close": 9.0, "BB_MID": 10.0})

    def test_time_stop_due_at_threshold(self):
        assert bollinger_standalone_time_stop_due(BOLLINGER_STANDALONE_TIME_STOP_DAYS)
        assert not bollinger_standalone_time_stop_due(BOLLINGER_STANDALONE_TIME_STOP_DAYS - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Exit dispatch, driven through manage_position_exits() like the rest of the suite
# ─────────────────────────────────────────────────────────────────────────────

class TestBollingerStandaloneExitDispatch:
    def _state_entry(self, price, cur, entry_time):
        return {
            'price': price, 'qty': 5.0, 'current_price': cur,
            'stop_loss': price * 0.90,
            'volume': 0, 'score': None,
            'entry_strategy': ENTRY_STRATEGY_NAME,
            'protection_status': 'confirmed',
            'time': entry_time.isoformat(),
        }

    def _run_exit_check(self, engine, now):
        with patch('src.engine_exits.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine.manage_position_exits()

    def _make_position(self, symbol, qty):
        pos = MagicMock()
        pos.contract.symbol = symbol
        pos.position = qty
        return pos

    def test_midline_reclaim_triggers_exit(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        entry_time = TZ_NY.localize(datetime(2026, 7, 20, 10, 0))
        now = TZ_NY.localize(datetime(2026, 7, 22, 10, 30))
        engine.state = {'MRVN': self._state_entry(10.0, 10.2, entry_time)}
        ib.reqTickers.return_value = [_mock_price_ticker(10.2)]
        ib.openTrades.return_value = []
        ib.positions.return_value = [self._make_position('MRVN', 5.0)]

        with patch.object(
            engine, '_daily_indicator_exit_row',
            return_value=pd.Series({'close': 10.5, 'BB_MID': 10.0}),
        ):
            self._run_exit_check(engine, now)

        assert engine.state['MRVN']['pending_exit'] is True

    def test_no_reclaim_within_time_stop_window_is_held(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        entry_time = TZ_NY.localize(datetime(2026, 7, 20, 10, 0))
        now = TZ_NY.localize(datetime(2026, 7, 21, 10, 30))  # 1 trading day held
        engine.state = {'MRVN': self._state_entry(10.0, 9.9, entry_time)}
        ib.reqTickers.return_value = [_mock_price_ticker(9.9)]
        ib.openTrades.return_value = []
        ib.positions.return_value = [self._make_position('MRVN', 5.0)]

        with patch.object(
            engine, '_daily_indicator_exit_row',
            return_value=pd.Series({'close': 9.9, 'BB_MID': 10.5}),
        ):
            self._run_exit_check(engine, now)

        assert 'pending_exit' not in engine.state['MRVN']

    def test_time_stop_fires_without_reclaim(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        entry_time = TZ_NY.localize(datetime(2026, 7, 1, 10, 0))
        now = TZ_NY.localize(datetime(2026, 7, 13, 10, 30))  # >= 7 trading days later
        engine.state = {'MRVN': self._state_entry(10.0, 9.9, entry_time)}
        ib.reqTickers.return_value = [_mock_price_ticker(9.9)]
        ib.openTrades.return_value = []
        ib.positions.return_value = [self._make_position('MRVN', 5.0)]

        with patch.object(
            engine, '_daily_indicator_exit_row',
            return_value=pd.Series({'close': 9.9, 'BB_MID': 10.5}),
        ):
            self._run_exit_check(engine, now)

        assert engine.state['MRVN']['pending_exit'] is True

    def test_tighter_bollinger_hard_stop_fires_before_generic_hard_stop(self):
        """A drawdown between the tighter 5% Bollinger stop and the generic
        7% HARD_STOP_PCT must exit via the Bollinger-specific reason, proving
        the tighter check runs first."""
        from src.config import HARD_STOP_PCT
        assert BOLLINGER_STANDALONE_HARD_STOP_PCT < HARD_STOP_PCT
        ib = _mock_ib()
        engine = _make_engine(ib)
        entry_time = TZ_NY.localize(datetime(2026, 7, 20, 10, 0))
        now = TZ_NY.localize(datetime(2026, 7, 21, 10, 30))
        entry = 10.0
        cur = round(entry * (1 - BOLLINGER_STANDALONE_HARD_STOP_PCT - 0.005), 2)
        engine.state = {'MRVN': self._state_entry(entry, cur, entry_time)}
        ib.reqTickers.return_value = [_mock_price_ticker(cur)]
        ib.openTrades.return_value = []
        ib.positions.return_value = [self._make_position('MRVN', 5.0)]

        self._run_exit_check(engine, now)

        assert engine.state['MRVN']['pending_exit'] is True

    def test_ma_cross_strategy_unaffected_by_new_branch(self):
        """Regression guard: a plain ma_cross position must not be touched
        by the new dispatch branch or the tighter Bollinger hard stop."""
        ib = _mock_ib()
        engine = _make_engine(ib)
        entry_time = TZ_NY.localize(datetime(2026, 7, 20, 10, 0))
        now = TZ_NY.localize(datetime(2026, 7, 21, 10, 30))
        state = self._state_entry(10.0, 9.7, entry_time)  # -3%, below neither stop
        state['entry_strategy'] = 'ma_cross'
        engine.state = {'AAPL': state}
        ib.reqTickers.return_value = [_mock_price_ticker(9.7)]
        ib.openTrades.return_value = []
        ib.positions.return_value = [self._make_position('AAPL', 5.0)]

        with patch.object(
            engine, '_daily_indicator_exit_row',
            return_value=pd.Series({'MA_BEAR_CROSS': False, 'close': 9.7, 'BB_MID': 10.0}),
        ):
            self._run_exit_check(engine, now)

        assert 'pending_exit' not in engine.state['AAPL']


# ─────────────────────────────────────────────────────────────────────────────
# Entry scan gating (1-slot cap, kill switch, candidate selection)
# ─────────────────────────────────────────────────────────────────────────────

class TestBollingerStandaloneEntryScan:
    def _fake_now(self):
        # Matches _make_engine()'s default _spy_cache date (2024-06-05) so
        # _fetch_spy_trend() uses the pre-seeded cached trend instead of
        # attempting a real (unmocked) historical fetch for "today".
        return TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

    def test_kill_switch_off_skips_entirely(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._prefilter_bollinger_candidates = ['XYZ']

        with patch('src.config.BOLLINGER_STANDALONE_ENABLED', False), \
             patch('src.engine_entries.BOLLINGER_STANDALONE_ENABLED', False), \
             patch.object(engine, '_place_bollinger_standalone_order') as mock_place, \
             _freeze_all_datetimes(self._fake_now()):
            engine._scan_and_enter_bollinger_standalone()

        mock_place.assert_not_called()

    def test_one_slot_cap_blocks_second_position(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine.state = {
            'ALREADY': {'entry_strategy': ENTRY_STRATEGY_NAME},
        }
        engine._prefilter_bollinger_candidates = ['XYZ']

        with patch('src.engine_entries.BOLLINGER_STANDALONE_ENABLED', True), \
             patch.object(engine, '_place_bollinger_standalone_order') as mock_place, \
             _freeze_all_datetimes(self._fake_now()):
            engine._scan_and_enter_bollinger_standalone()

        mock_place.assert_not_called()

    def test_no_candidates_skips(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._prefilter_bollinger_candidates = []

        with patch('src.engine_entries.BOLLINGER_STANDALONE_ENABLED', True), \
             patch.object(engine, '_place_bollinger_standalone_order') as mock_place, \
             _freeze_all_datetimes(self._fake_now()):
            engine._scan_and_enter_bollinger_standalone()

        mock_place.assert_not_called()

    def test_best_ranked_passing_candidate_is_placed(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._prefilter_bollinger_candidates = ['WEAK', 'STRONG']

        weak_ctx = _base_ctx(live_price=10.8, bb_mid=11.0)     # shallow reclaim
        strong_ctx = _base_ctx(live_price=9.0, bb_mid=11.0)    # deep reclaim

        def fake_ctx(sym):
            return {'WEAK': weak_ctx, 'STRONG': strong_ctx}[sym]

        with patch('src.engine_entries.BOLLINGER_STANDALONE_ENABLED', True), \
             patch.object(engine, 'get_technical_context', side_effect=fake_ctx), \
             patch.object(engine, '_place_bollinger_standalone_order') as mock_place, \
             _freeze_all_datetimes(self._fake_now()):
            engine._scan_and_enter_bollinger_standalone()

        mock_place.assert_called_once()
        called_sym = mock_place.call_args[0][0]
        assert called_sym == 'STRONG'

    def test_candidate_already_held_is_skipped(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        engine.state = {'HELD': {'entry_strategy': 'ma_cross'}}
        engine._prefilter_bollinger_candidates = ['HELD']

        with patch('src.engine_entries.BOLLINGER_STANDALONE_ENABLED', True), \
             patch.object(engine, '_place_bollinger_standalone_order') as mock_place, \
             _freeze_all_datetimes(self._fake_now()):
            engine._scan_and_enter_bollinger_standalone()

        mock_place.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Order placement mechanics
# ─────────────────────────────────────────────────────────────────────────────

class TestBollingerStandaloneOrderPlacement:
    def test_buy_then_trail_placed_with_correct_tagging(self):
        from src.config import TRAIL_PCT

        ib = _mock_ib()
        engine = _make_engine(ib)
        ctx = _base_ctx(live_price=10.0, bid=9.98, ask=10.02, volume=1_000_000)
        ctx['contract'] = MagicMock()
        ctx['day_open'] = 10.1
        ctx['day_high'] = 10.3
        ctx['day_low'] = 9.9

        buy_trade = MagicMock()
        buy_trade.order.orderId = 101
        buy_trade.orderStatus.status = 'Filled'
        buy_trade.orderStatus.filled = 10.0
        buy_trade.orderStatus.avgFillPrice = 10.0
        buy_trade.fills = [_mock_fill(1.0)]

        stop_trade = MagicMock()
        stop_trade.order.orderId = 102
        stop_trade.orderStatus.status = 'Submitted'

        ib.placeOrder.side_effect = [buy_trade, stop_trade]

        fake_now = TZ_NY.localize(datetime(2026, 7, 20, 10, 30))
        with _freeze_all_datetimes(fake_now), \
             patch.object(engine, '_confirm_protective_stop', return_value=True):
            engine._place_bollinger_standalone_order('QRVO', ctx, bucket_size=100.0, settled=1000.0)

        assert ib.placeOrder.call_count == 2
        buy_order = ib.placeOrder.call_args_list[0][0][1]
        stop_order = ib.placeOrder.call_args_list[1][0][1]

        assert buy_order.action == 'BUY'
        assert buy_order.orderType == 'LMT'
        assert stop_order.action == 'SELL'
        assert stop_order.orderType == 'TRAIL'
        assert stop_order.trailingPercent == pytest.approx(round(TRAIL_PCT * 100, 2))

        assert 'QRVO' in engine.state
        assert engine.state['QRVO']['entry_strategy'] == ENTRY_STRATEGY_NAME
        assert engine.state['QRVO']['strategy_profile'] == ENTRY_STRATEGY_NAME
        assert engine.state['QRVO']['fill_price'] == pytest.approx(10.0)

    def test_insufficient_settled_cash_skips_order(self):
        ib = _mock_ib()
        engine = _make_engine(ib)
        ctx = _base_ctx(live_price=10.0, bid=9.98, ask=10.02)
        ctx['contract'] = MagicMock()

        engine._place_bollinger_standalone_order('QRVO', ctx, bucket_size=100.0, settled=1.0)

        assert not ib.placeOrder.called
        assert 'QRVO' not in engine.state


# ─────────────────────────────────────────────────────────────────────────────
# Prefilter independent tagging
# ─────────────────────────────────────────────────────────────────────────────

class TestBollingerPrefilterTagging:
    def _daily_df(self, n=260, start_price=100.0, end_price=90.0):
        idx = pd.date_range("2025-01-01", periods=n, freq="B")
        close = np.linspace(start_price, end_price, n)  # declining -> fails indicator_swing trend gates
        return pd.DataFrame({
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 3_000_000),
            "MA200": close,  # non-NaN placeholder; real indicator computation is bypassed in this test
        }, index=idx)

    def test_symbol_failing_indicator_swing_but_passing_standalone_check(self):
        from src.strategy_profiles import get_strategy_profile

        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._strategy_profile = get_strategy_profile("indicator_swing")
        engine._prefilter_bollinger_candidates = []
        daily_df = self._daily_df()
        ib.reqHistoricalData.return_value = [MagicMock()] * 260

        forced_ctx = {
            'live_price': 88.0, 'close': 88.0, 'price': 88.0,
            'bb_mid': 92.0, 'bb_reclaim_lower': True,
            'spread_pct': 0.001,
            'dollar_vol_20d': BOLLINGER_STANDALONE_MIN_DOLLAR_VOL * 3,
            'avg_dollar_vol_20': BOLLINGER_STANDALONE_MIN_DOLLAR_VOL * 3,
            'volume': 3_000_000, 'ma50': 95.0, 'ma200': 100.0,
            'sma200_slope': -0.5,
            'df_daily': daily_df,
        }

        with patch('ib_async.util.df', return_value=daily_df), \
             patch('src.engine_base.completed_daily_bars', side_effect=lambda bars, today: bars), \
             patch.object(engine, '_enrich_prefilter_daily_frame', return_value=daily_df), \
             patch.object(engine, '_build_prefilter_context', return_value=forced_ctx):
            passed, failures, _ = engine._prefilter_symbol('WEAK', engine._strategy_profile, '2026-07-20')

        assert passed is False
        assert 'MA50<=MA200' in failures
        assert 'WEAK' in engine._prefilter_bollinger_candidates

    def test_symbol_failing_standalone_liquidity_is_not_tagged(self):
        from src.strategy_profiles import get_strategy_profile

        ib = _mock_ib()
        engine = _make_engine(ib)
        engine._strategy_profile = get_strategy_profile("indicator_swing")
        engine._prefilter_bollinger_candidates = []
        daily_df = self._daily_df()
        ib.reqHistoricalData.return_value = [MagicMock()] * 260

        forced_ctx = {
            'live_price': 88.0, 'close': 88.0, 'price': 88.0,
            'bb_mid': 92.0, 'bb_reclaim_lower': True,
            'spread_pct': 0.001,
            'dollar_vol_20d': BOLLINGER_STANDALONE_MIN_DOLLAR_VOL / 10,  # too thin
            'avg_dollar_vol_20': BOLLINGER_STANDALONE_MIN_DOLLAR_VOL / 10,
            'volume': 3_000_000, 'ma50': 95.0, 'ma200': 100.0,
            'sma200_slope': -0.5,
            'df_daily': daily_df,
        }

        with patch('ib_async.util.df', return_value=daily_df), \
             patch('src.engine_base.completed_daily_bars', side_effect=lambda bars, today: bars), \
             patch.object(engine, '_enrich_prefilter_daily_frame', return_value=daily_df), \
             patch.object(engine, '_build_prefilter_context', return_value=forced_ctx):
            engine._prefilter_symbol('THIN', engine._strategy_profile, '2026-07-20')

        assert 'THIN' not in engine._prefilter_bollinger_candidates
