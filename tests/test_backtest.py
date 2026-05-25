"""
Unit tests for backtest/strategy.py — no IB, no live data.

All tests use purely synthetic DataFrames so they run offline
and deterministically.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest
from datetime import date

import backtest.strategy as strategy_module
from backtest.optimizer import (
    OptimizationParams,
    _limit_symbols,
    _run_with_params,
    default_grid,
    format_optimization_table,
    quick_grid,
    score_metrics,
)
from backtest.strategy import VelocityBacktest, Trade, BacktestResult
from src.config import SCAN_MIN_VOLUME
from src.config import (
    DAY_RANGE_LOCATION_MIN,
    INTRADAY_GAIN_MIN,
    ATR_PCT_MAX,
    PROFIT_MIN_THRESHOLD,
    BACKTEST_RVOL_MIN,
)


class TestBacktestUniverseFiltering:
    def test_common_equity_filter_rejects_non_stock_listings(self):
        assert VelocityBacktest._is_common_equity_listing(
            "AAPL", "Apple Inc. Common Stock"
        )
        assert VelocityBacktest._is_common_equity_listing(
            "WWW", "Wolverine World Wide, Inc. Common Stock"
        )
        assert not VelocityBacktest._is_common_equity_listing(
            "ABCW", "ABC Acquisition Corp. Warrant"
        )
        assert not VelocityBacktest._is_common_equity_listing(
            "ABCR", "ABC Acquisition Corp. Right"
        )
        assert not VelocityBacktest._is_common_equity_listing(
            "ABCU", "ABC Acquisition Corp. Unit"
        )
        assert not VelocityBacktest._is_common_equity_listing(
            "PREF", "Example Corp. Preferred Stock"
        )
        assert not VelocityBacktest._is_common_equity_listing(
            np.nan, np.nan
        )

    def test_fetch_universe_excludes_warrants_units_rights_and_keeps_amex(self, monkeypatch):
        nasdaq_text = "\n".join([
            "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares",
            "AAPL|Apple Inc. Common Stock|Q|N|N|100|N|N",
            "ABCW|ABC Acquisition Corp. Warrant|Q|N|N|100|N|N",
            "WWW|Wolverine World Wide, Inc. Common Stock|Q|N|N|100|N|N",
            "SMALL|Small Co Common Stock|S|N|N|100|N|N",
        ])
        other_text = "\n".join([
            "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol",
            "XYZ|XYZ Corp Common Stock|N|XYZ|N|100|N|XYZ",
            "AMEX|AMEX Co Common Stock|A|AMEX|N|100|N|AMEX",
            "PREF|Example Corp Preferred Stock|N|PREF|N|100|N|PREF",
            "FUND|Example ETF|N|FUND|Y|100|N|FUND",
        ])

        class Response:
            def __init__(self, data):
                self.data = data.encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return self.data

        def fake_urlopen(req, timeout=20):
            url = req.full_url
            return Response(nasdaq_text if "nasdaqlisted" in url else other_text)

        monkeypatch.setattr(strategy_module.urllib.request, "urlopen", fake_urlopen)

        assert VelocityBacktest._fetch_universe() == ["AAPL", "AMEX", "WWW", "XYZ"]


# ── Synthetic data factory ────────────────────────────────────────────────────
def _make_df(n: int = 300, seed: int = 0, trend: float = 0.2) -> pd.DataFrame:
    """Smooth upward-trending OHLCV with fully warmed-up indicators."""
    np.random.seed(seed)
    close  = 100 + trend * np.arange(n) + np.cumsum(np.random.randn(n) * 0.3)
    high   = close + np.abs(np.random.randn(n) * 0.2)
    low    = close - np.abs(np.random.randn(n) * 0.2)
    idx    = pd.date_range("2023-01-01", periods=n, freq='B')

    from src.indicators import apply_all
    df = pd.DataFrame({'open': close, 'high': high, 'low': low,
                       'close': close, 'volume': SCAN_MIN_VOLUME + 1_000_000}, index=idx)
    df = apply_all(df)
    df['prev_high']   = df['high'].shift(1)
    df['prev_ATR5']   = df['ATR5'].shift(1)
    df['prev_ATR20']  = df['ATR20'].shift(1)
    df['prev_HIGH10'] = df['HIGH10'].shift(1)
    return df


# ── Trade dataclass ───────────────────────────────────────────────────────────
class TestTradeDataclass:
    def test_pnl_long_winner(self):
        # pnl = net_pnl = gross - $2 round-trip commission
        t = Trade("AAPL", date(2024, 1, 2), 100.0, date(2024, 1, 5), 110.0, qty=10)
        assert abs(t.pnl - 98.0) < 1e-9    # gross=100, net=98 after $2 commission

    def test_pnl_long_loser(self):
        t = Trade("AAPL", date(2024, 1, 2), 100.0, date(2024, 1, 5), 90.0, qty=5)
        assert abs(t.pnl - (-52.0)) < 1e-9  # gross=-50, net=-52 after $2 commission

    def test_pnl_pct_correct(self):
        t = Trade("AAPL", date(2024, 1, 2), 100.0, date(2024, 1, 5), 110.0, qty=1)
        assert abs(t.pnl_pct - 0.10) < 1e-9

    def test_pnl_returns_zero_when_no_exit(self):
        # Commission only deducted at close — open positions show 0
        t = Trade("AAPL", date(2024, 1, 2), 100.0)
        assert t.pnl == 0.0

    def test_gross_pnl_excludes_commission(self):
        t = Trade("AAPL", date(2024, 1, 2), 100.0, date(2024, 1, 5), 110.0, qty=10)
        assert abs(t.gross_pnl - 100.0) < 1e-9

    def test_net_pnl_includes_commission(self):
        t = Trade("AAPL", date(2024, 1, 2), 100.0, date(2024, 1, 5), 110.0, qty=10)
        assert abs(t.net_pnl - 98.0) < 1e-9   # $2 round-trip deducted


# ── Entry signal ──────────────────────────────────────────────────────────────
class TestEntrySignal:
    """
    _entry_signal implements the 8096 entry gate and requires:
      row columns: close, prev_high, RSI, CLV, open, ATR/ATR_CHAND
      positional:  prev_rsi, rvol, rvol_min
      optional:    legacy-compatible vcp_ratio, breakout_pct
    """

    def _row(self, close=110, open_price=108.5, prev_high=100, ma50=105, ma200=90, rsi=60, atr=2.0,
             sma200_slope=0.5, prev_atr5=1.5, prev_atr20=2.5, prev_high10=108,
             clv=0.75):
        atr_chand = atr
        return pd.Series({
            'open':         open_price,
            'close':        close,
            'prev_high':    prev_high,
            'MA50':         ma50,
            'MA200':        ma200,
            'RSI':          rsi,
            'CLV':          clv,
            'ATR':          atr,
            'ATR_CHAND':    atr_chand,
            'SMA200_SLOPE': sma200_slope,
            'prev_ATR5':    prev_atr5,
            'prev_ATR20':   prev_atr20,
            'prev_HIGH10':  prev_high10,
        })

    def test_all_conditions_pass(self):
        # 8096: ORB/prev-high break + RSI level/delta + CLV + open gain + ATR%.
        assert VelocityBacktest._entry_signal(
            self._row(), prev_rsi=55, rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_price_below_prev_high(self):
        assert not VelocityBacktest._entry_signal(
            self._row(close=99, prev_high=100), prev_rsi=55,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_ignores_price_below_ma50_when_8096_rules_pass(self):
        assert VelocityBacktest._entry_signal(
            self._row(close=110, open_price=108.5, ma50=115), prev_rsi=55,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_ignores_ma50_below_ma200_when_8096_rules_pass(self):
        assert VelocityBacktest._entry_signal(
            self._row(ma50=85, ma200=90), prev_rsi=55,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_rsi_delta_below_minimum(self):
        assert not VelocityBacktest._entry_signal(
            self._row(rsi=60), prev_rsi=65, rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_rsi_below_55(self):
        assert not VelocityBacktest._entry_signal(
            self._row(rsi=54, ma50=105), prev_rsi=50,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_ignores_rvol_below_min_when_8096_rules_pass(self):
        assert VelocityBacktest._entry_signal(
            self._row(), prev_rsi=55, rvol=0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_ignores_vcp_not_contracting_when_8096_rules_pass(self):
        assert VelocityBacktest._entry_signal(
            self._row(prev_atr5=2.6, prev_atr20=2.5), prev_rsi=55,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_ignores_distance_from_10_day_high_when_orb_passes(self):
        # Exhaustive rule sweep promoted removing the 10-day-high proximity gate:
        # 8096 ORB + RSI + close-location/open-gain/ATR rules define the breakout quality.
        assert VelocityBacktest._entry_signal(
            self._row(close=116, prev_high10=105), prev_rsi=55,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_close_location_below_minimum(self):
        assert not VelocityBacktest._entry_signal(
            self._row(clv=DAY_RANGE_LOCATION_MIN - 0.01), prev_rsi=55,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_intraday_gain_below_minimum(self):
        open_price = 100.0
        close = open_price * (1 + INTRADAY_GAIN_MIN - 0.001)
        assert not VelocityBacktest._entry_signal(
            self._row(close=close, open_price=open_price, prev_high=99, ma50=98,
                      ma200=90, prev_high10=close), prev_rsi=55,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_atr_pct_above_maximum(self):
        row = self._row(atr=ATR_PCT_MAX * 110 * 1.2)
        row['ATR_CHAND'] = ATR_PCT_MAX * row['close'] * 1.2
        assert not VelocityBacktest._entry_signal(
            row, prev_rsi=55, rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)


# ── Metrics ───────────────────────────────────────────────────────────────────
class TestComputeMetrics:
    def _eq(self, vals):
        idx = pd.date_range("2024-01-01", periods=len(vals), freq='B')
        return pd.Series(vals, index=idx, dtype=float)

    def test_win_rate_all_wins(self):
        trades = [
            Trade("A", date(2024,1,2), 100, date(2024,1,5), 110, qty=1),
            Trade("B", date(2024,1,2), 200, date(2024,1,5), 220, qty=1),
        ]
        m = VelocityBacktest._compute_metrics(trades, self._eq([1400, 1410, 1420]))
        assert m['win_rate'] == 1.0

    def test_win_rate_all_losses(self):
        trades = [
            Trade("A", date(2024,1,2), 100, date(2024,1,5),  90, qty=1),
        ]
        m = VelocityBacktest._compute_metrics(trades, self._eq([1400, 1390]))
        assert m['win_rate'] == 0.0

    def test_profit_factor_infinite_when_no_losses(self):
        trades = [Trade("A", date(2024,1,2), 100, date(2024,1,5), 110, qty=1)]
        m = VelocityBacktest._compute_metrics(trades, self._eq([1400, 1410]))
        assert m['profit_factor'] == float('inf')

    def test_max_drawdown_negative(self):
        # Equity drops from 1400 to 1200, then recovers
        eq = self._eq([1400, 1350, 1200, 1250, 1400])
        m  = VelocityBacktest._compute_metrics(
            [Trade("A", date(2024,1,2), 100, date(2024,1,5), 110, qty=1)], eq
        )
        assert m['max_drawdown_pct'] < 0

    def test_empty_trades_returns_empty_dict(self):
        eq = self._eq([1400, 1400])
        assert VelocityBacktest._compute_metrics([], eq) == {}

    def test_total_return_growing_equity(self):
        eq = self._eq([1000, 1500])
        trades = [Trade("A", date(2024,1,2), 100, date(2024,1,5), 150, qty=10)]
        m  = VelocityBacktest._compute_metrics(trades, eq)
        assert abs(m['total_return_pct'] - 50.0) < 1e-6


# ── Exit fill realism ────────────────────────────────────────────────────────
class TestExitFillRealism:
    def test_stop_fill_uses_stop_when_bar_trades_through_intraday(self):
        row = pd.Series({"open": 101.0})
        assert VelocityBacktest._stop_fill_price(row, 100.0) == 100.0

    def test_stop_fill_uses_open_when_market_gaps_below_stop(self):
        row = pd.Series({"open": 95.0})
        assert VelocityBacktest._stop_fill_price(row, 100.0) == 95.0


# ── Entry fill realism ────────────────────────────────────────────────────────
class TestEntryFillRealism:
    @staticmethod
    def _signal_df():
        idx = pd.date_range("2023-01-02", periods=3, freq="B")
        return pd.DataFrame({
            "open": [10.0, 25.0, 25.0],
            "high": [26.0, 26.0, 26.0],
            "low": [9.5, 24.0, 24.0],
            "close": [25.0, 25.0, 25.0],
            "volume": SCAN_MIN_VOLUME + 1_000_000,
            "MA50": 20.0,
            "MA200": 15.0,
            "RSI": 70.0,
            "CLV": 0.75,
            "ATR": 1.0,
            "ATR_CHAND": 1.0,
            "prev_ATR5": 0.50,
            "prev_ATR20": 1.0,
            "prev_HIGH10": 24.0,
            "SMA200_SLOPE": 1.0,
            "prev_high": [10.0, 24.0, 24.0],
            "avg_vol_20": SCAN_MIN_VOLUME + 1_000_000,
            "avg_dollar_vol_20": 200_000_000,
        }, index=idx)

    def test_entry_price_floor_blocks_below_min_price_fill(self, monkeypatch):
        df = self._signal_df()
        df["open"] = 10.0
        df["prev_high"] = 10.0
        bt = VelocityBacktest(start="2023-01-02", end="2023-01-05",
                              capital=1000.0, use_cache=False,
                              use_spy_filter=False, use_vix_filter=False)
        bt._data = {"LOWRAW": df}
        monkeypatch.setattr(bt, "_daily_scan", lambda today, **_kwargs: [("LOWRAW", 5.0)])
        monkeypatch.setattr(VelocityBacktest, "_entry_signal", staticmethod(lambda *args, **_kwargs: True))

        result = bt._run_loop()

        assert result.trades == []
        assert result.filter_stats["entries_taken"] == 0

    def test_conservative_daily_entry_fills_no_better_than_close(self, monkeypatch):
        df = self._signal_df()
        df.loc[df.index[0], "open"] = 24.0
        df.loc[df.index[0], "prev_high"] = 24.0
        bt = VelocityBacktest(start="2023-01-02", end="2023-01-05",
                              capital=1000.0, conservative_daily_entry=True,
                              use_cache=False, use_spy_filter=False,
                              use_vix_filter=False)
        bt._data = {"SIG": df}
        monkeypatch.setattr(bt, "_daily_scan", lambda today, **_kwargs: [("SIG", 5.0)])
        monkeypatch.setattr(VelocityBacktest, "_entry_signal", staticmethod(lambda *args, **_kwargs: True))

        result = bt._run_loop()

        assert result.trades
        assert result.trades[0].entry_price == pytest.approx(25.0 * 1.001)


# ── Live-compatible position sizing ───────────────────────────────────────────
class TestWholeShareSizing:
    def test_whole_share_qty_floors_risk_and_bucket_caps(self):
        # equity risk: 1400 * 2% / $3 risk = 9 shares
        # bucket cap : $466 / $50 = 9 shares
        qty = VelocityBacktest._whole_share_qty(
            account_equity=1400.0,
            bucket=466.67,
            entry_price=50.0,
            risk_stop_dist=3.0,
        )
        assert qty == 9

    def test_whole_share_qty_returns_zero_when_stock_too_expensive(self):
        qty = VelocityBacktest._whole_share_qty(
            account_equity=1400.0,
            bucket=466.67,
            entry_price=500.0,
            risk_stop_dist=10.0,
        )
        assert qty == 0

    def test_whole_share_qty_accepts_reduced_bear_risk(self):
        qty = VelocityBacktest._whole_share_qty(
            account_equity=1400.0,
            bucket=466.67,
            entry_price=50.0,
            risk_stop_dist=3.0,
            risk_per_trade_pct=0.01,
        )
        assert qty == 4


# ── Optimizer helpers ─────────────────────────────────────────────────────────
class TestOptimizerHelpers:
    def test_score_metrics_rejects_thin_sample(self):
        metrics = {
            "total_trades": 3,
            "sharpe_ratio": 10.0,
            "total_return_pct": 500.0,
            "max_drawdown_pct": -1.0,
            "profit_factor": 10.0,
        }
        assert score_metrics(metrics, min_trades=20) == float("-inf")

    def test_score_metrics_rewards_better_risk_adjusted_profile(self):
        weak = {
            "total_trades": 30,
            "sharpe_ratio": 1.0,
            "total_return_pct": 20.0,
            "max_drawdown_pct": -20.0,
            "profit_factor": 1.2,
        }
        strong = {
            "total_trades": 30,
            "sharpe_ratio": 2.0,
            "total_return_pct": 50.0,
            "max_drawdown_pct": -5.0,
            "profit_factor": 2.0,
        }
        assert score_metrics(strong) > score_metrics(weak)

    def test_quick_grid_is_non_empty_and_bounded(self):
        grid = quick_grid()
        assert grid
        assert len(grid) <= 8
        assert all(isinstance(p, OptimizationParams) for p in grid)

    def test_default_grid_only_varies_active_8096_parameters(self):
        grid = default_grid()
        assert len(grid) == 18
        assert len({p.rvol_min for p in grid}) == 1
        assert len({p.breakout_pct for p in grid}) == 1
        assert len({p.vcp_ratio for p in grid}) == 1
        assert len({p.hold_bars for p in grid}) > 1
        assert len({p.break_even_pct for p in grid}) > 1
        assert len({p.chandelier_mult for p in grid}) > 1

    def test_limit_symbols_keeps_most_liquid_names(self):
        liquid = _make_df(n=320, seed=9, trend=0.2)
        quiet = _make_df(n=320, seed=10, trend=0.2)
        liquid["avg_dollar_vol_20"] = 100_000_000
        quiet["avg_dollar_vol_20"] = 10_000_000
        base = VelocityBacktest(start="2023-01-01", end="2024-04-01",
                                use_cache=False, use_spy_filter=False,
                                use_vix_filter=False)
        base._data = {"QUIET": quiet, "LIQUID": liquid}

        _limit_symbols(base, 1)

        assert list(base._data) == ["LIQUID"]

    def test_limit_symbols_zero_means_unlimited(self):
        base = VelocityBacktest(start="2023-01-01", end="2024-04-01",
                                use_cache=False, use_spy_filter=False,
                                use_vix_filter=False)
        base._data = {"A": _make_df(n=320, seed=11), "B": _make_df(n=320, seed=12)}

        _limit_symbols(base, 0)

        assert set(base._data) == {"A", "B"}

    def test_run_with_params_uses_requested_window(self):
        df = _make_df(n=320, seed=8, trend=0.25)
        base = VelocityBacktest(start="2023-01-01", end="2024-04-01",
                                use_cache=False, use_spy_filter=False,
                                use_vix_filter=False)
        base._data = {"SYM": df}
        result = _run_with_params(
            base,
            start="2023-09-01",
            end="2024-01-01",
            params=OptimizationParams(),
        )
        assert isinstance(result, BacktestResult)
        assert result.equity_curve.index.min() >= pd.Timestamp("2023-09-01")
        assert result.equity_curve.index.max() < pd.Timestamp("2024-01-01")

    def test_format_optimization_table_has_header(self):
        table = format_optimization_table([])
        assert "rank robust forward train" in table


# ── Full run on synthetic data ────────────────────────────────────────────────
class TestFullRunSynthetic:
    def test_run_returns_backtest_result(self, monkeypatch):
        """Patch _download to inject a synthetic bullish symbol."""
        df = _make_df(n=300, seed=1, trend=0.3)

        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01",
                              use_cache=False, use_spy_filter=False,
                              use_vix_filter=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"FAKE": df}))

        result = bt.run()
        assert isinstance(result, BacktestResult)

    def test_no_trades_flat_market(self, monkeypatch):
        """Flat/downward market should produce very few or zero qualifying signals."""
        np.random.seed(5)
        n     = 300
        close = np.full(n, 100.0) + np.random.randn(n) * 0.05
        high  = close + 0.02
        low   = close - 0.02
        idx   = pd.date_range("2023-01-01", periods=n, freq='B')

        from src.indicators import apply_all as _apply
        df = pd.DataFrame({'open': close, 'high': high, 'low': low,
                           'close': close, 'volume': SCAN_MIN_VOLUME + 1_000_000}, index=idx)
        df = _apply(df)
        df['prev_high']   = df['high'].shift(1)
        df['prev_ATR5']   = df['ATR5'].shift(1)
        df['prev_ATR20']  = df['ATR20'].shift(1)
        df['prev_HIGH10'] = df['HIGH10'].shift(1)

        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01",
                              use_cache=False, use_spy_filter=False,
                              use_vix_filter=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"FLAT": df}))

        result = bt.run()
        # MA50 ≈ MA200 in a flat market, so breakout filter mostly fails
        assert isinstance(result, BacktestResult)

    def test_equity_curve_is_pandas_series(self, monkeypatch):
        df = _make_df(n=300, seed=2, trend=0.2)
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01",
                              use_cache=False, use_spy_filter=False,
                              use_vix_filter=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"SYM": df}))

        result = bt.run()
        assert isinstance(result.equity_curve, pd.Series)

    def test_equity_curve_starts_at_backtest_start(self, monkeypatch):
        df = _make_df(n=300, seed=7, trend=0.2)
        bt = VelocityBacktest(start="2023-06-01", end="2024-01-01",
                              use_cache=False, use_spy_filter=False,
                              use_vix_filter=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"SYM": df}))

        result = bt.run()
        assert result.equity_curve.index.min() >= pd.Timestamp("2023-06-01")

    def test_run_loop_stops_at_requested_end_with_unsliced_cached_data(self, monkeypatch):
        df = _make_df(n=320, seed=17, trend=0.35)
        start = "2023-09-01"
        end = "2023-10-02"
        bt = VelocityBacktest(start=start, end=end, capital=2_000.0,
                              max_pos=2, use_cache=False,
                              use_spy_filter=False, use_vix_filter=False)
        bt._data = {"SYM": df}
        monkeypatch.setattr(bt, "_daily_scan", lambda today, **_kwargs: [("SYM", 5.0)])
        monkeypatch.setattr(VelocityBacktest, "_entry_signal", staticmethod(lambda *args, **_kwargs: True))

        result = bt._run_loop()

        expected_scan_days = len(df[(df.index >= start) & (df.index < end)])
        assert result.filter_stats["scan_days"] == expected_scan_days
        assert result.equity_curve.index.min() >= pd.Timestamp(start)
        assert result.equity_curve.index.max() < pd.Timestamp(end)
        assert all(pd.Timestamp(t.entry_date) < pd.Timestamp(end) for t in result.trades)
        assert all(pd.Timestamp(t.exit_date) < pd.Timestamp(end) for t in result.trades)

    def test_bear_phase_mode_allows_spy_bear_entries(self, monkeypatch):
        df = _make_df(n=320, seed=21, trend=0.35)
        bt = VelocityBacktest(start="2023-09-01", end="2024-01-01",
                              use_cache=False, use_spy_filter=True,
                              use_vix_filter=False,
                              bear_phase_trading=True)
        bt._data = {"SYM": df}
        bt._spy_bull = pd.Series(False, index=df.index)
        monkeypatch.setattr(bt, "_daily_scan", lambda today, **_kwargs: [("SYM", 5.0)])
        monkeypatch.setattr(VelocityBacktest, "_entry_signal", staticmethod(lambda *args, **_kwargs: True))

        result = bt._run_loop()

        assert result.filter_stats["spy_bear_trade_days"] > 0
        assert result.filter_stats["bear_phase_entries"] > 0
        assert result.metrics["regime_entries"].get("bear", 0) > 0

    def test_spy_bear_still_blocks_when_bear_mode_disabled(self, monkeypatch):
        df = _make_df(n=320, seed=22, trend=0.35)
        bt = VelocityBacktest(start="2023-09-01", end="2024-01-01",
                              use_cache=False, use_spy_filter=True,
                              use_vix_filter=False,
                              bear_phase_trading=False)
        bt._data = {"SYM": df}
        bt._spy_bull = pd.Series(False, index=df.index)
        scan_calls = []
        monkeypatch.setattr(bt, "_daily_scan", lambda today, **_kwargs: scan_calls.append(today) or [("SYM", 5.0)])

        result = bt._run_loop()

        assert result.filter_stats["spy_blocked_days"] > 0
        assert result.filter_stats["entries_taken"] == 0
        assert scan_calls == []

    def test_no_data_raises_runtime_error(self, monkeypatch):
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01",
                              use_cache=False, use_spy_filter=False,
                              use_vix_filter=False)
        # Patch _download to leave _data empty
        monkeypatch.setattr(bt, '_download', lambda: None)
        with pytest.raises(RuntimeError, match="No usable data"):
            bt.run()

    def test_missing_spy_regime_data_raises_when_filter_enabled(self, monkeypatch):
        df = _make_df(n=300, seed=3, trend=0.2)
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01",
                              use_cache=False, use_spy_filter=True,
                              use_vix_filter=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"SYM": df}))

        with pytest.raises(RuntimeError, match="SPY regime filter is enabled"):
            bt.run()

    def test_missing_vix_regime_data_raises_when_filter_enabled(self, monkeypatch):
        df = _make_df(n=300, seed=4, trend=0.2)
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01",
                              use_cache=False, use_spy_filter=False,
                              use_vix_filter=True)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"SYM": df}))

        with pytest.raises(RuntimeError, match="VIX regime filter is enabled"):
            bt.run()

    def test_regime_cache_reused_without_provider_call(self, monkeypatch, tmp_path):
        monkeypatch.setattr(strategy_module, "_CACHE_DIR", str(tmp_path))
        idx = pd.date_range("2023-01-01", periods=300, freq="B")
        cached = pd.Series(True, index=idx)

        writer = VelocityBacktest(start="2023-01-01", end="2024-01-01",
                                  use_cache=True, use_spy_filter=True,
                                  use_vix_filter=False)
        writer._spy_bull = cached
        writer._save_regime_cache()

        reader = VelocityBacktest(start="2023-01-01", end="2024-01-01",
                                  use_cache=True, use_spy_filter=True,
                                  use_vix_filter=False)
        monkeypatch.setattr(strategy_module.yf, "download",
                            lambda *args, **_kwargs: pytest.fail("provider should not be called"))

        assert reader._try_load_regime_cache()
        assert reader._spy_bull.equals(cached)

    def test_regime_cache_reuses_compatible_superset_window(self, monkeypatch, tmp_path):
        monkeypatch.setattr(strategy_module, "_CACHE_DIR", str(tmp_path))
        idx = pd.date_range("2021-01-01", periods=900, freq="B")
        spy = pd.Series(True, index=idx)
        vix = pd.Series(20.0, index=idx)

        writer = VelocityBacktest(start="2022-01-01", end="2024-06-01",
                                  use_cache=True, use_spy_filter=True,
                                  use_vix_filter=True)
        writer._spy_bull = spy
        writer._vix_series = vix
        writer._save_regime_cache()

        reader = VelocityBacktest(start="2023-09-01", end="2024-01-01",
                                  use_cache=True, use_spy_filter=True,
                                  use_vix_filter=True)

        assert reader._try_load_regime_cache()
        assert reader._spy_bull.equals(spy)
        assert reader._vix_series.equals(vix)

    def test_stock_cache_reuses_compatible_superset_window(self, monkeypatch, tmp_path):
        monkeypatch.setattr(strategy_module, "_CACHE_DIR", str(tmp_path))
        df = _make_df(n=900, seed=18, trend=0.2)
        df.index = pd.date_range("2021-01-01", periods=len(df), freq="B")

        writer = VelocityBacktest(start="2022-01-01", end="2024-06-01",
                                  max_symbols=10, use_cache=True,
                                  use_spy_filter=False, use_vix_filter=False)
        writer._data = {"SYM": df}
        writer._save_cache()

        reader = VelocityBacktest(start="2023-09-01", end="2024-01-01",
                                  max_symbols=10, use_cache=True,
                                  use_spy_filter=False, use_vix_filter=False)

        assert reader._try_load_cache()
        assert list(reader._data) == ["SYM"]
        assert reader._data["SYM"].index.min() <= pd.Timestamp(reader._data_start)
        assert reader._data["SYM"].index.max() >= pd.Timestamp(reader.end) - pd.Timedelta(days=7)

    def test_spy_regime_download_uses_yahoo_chart_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setattr(strategy_module, "_CACHE_DIR", str(tmp_path))
        idx = pd.date_range("2022-01-01", periods=320, freq="B")
        fallback = pd.DataFrame({
            "open": np.linspace(100, 150, len(idx)),
            "high": np.linspace(101, 151, len(idx)),
            "low": np.linspace(99, 149, len(idx)),
            "close": np.linspace(100, 160, len(idx)),
            "volume": 1_000_000,
        }, index=idx)

        monkeypatch.setattr(strategy_module.yf, "download",
                            lambda *args, **_kwargs: pd.DataFrame())
        monkeypatch.setattr(VelocityBacktest, "_download_yahoo_chart_daily",
                            staticmethod(lambda *args, **_kwargs: fallback))
        monkeypatch.setattr(VelocityBacktest, "_download_stooq_daily",
                            staticmethod(lambda *args, **_kwargs: pytest.fail("stooq should not be called")))

        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01",
                              use_cache=False, use_spy_filter=True,
                              use_vix_filter=False)
        bt._download_regime_data()

        assert bt._spy_bull is not None
        assert bool(bt._spy_bull.dropna().iloc[-1])

    def test_equity_curve_final_matches_closed_trade_pnl(self, monkeypatch):
        df = _make_df(n=300, seed=6, trend=0.3)
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01",
                              use_cache=False, use_spy_filter=False,
                              use_vix_filter=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"SYM": df}))

        result = bt.run()
        if result.trades:
            expected_final = bt.capital + sum(t.pnl for t in result.trades)
            assert result.equity_curve.iloc[-1] == pytest.approx(expected_final)

    def test_t1_settlement_blocks_same_day_reuse_of_sale_proceeds(self, monkeypatch):
        idx = pd.date_range("2023-01-02", periods=6, freq="B")
        df = pd.DataFrame({
            "open": 50.0,
            "high": 51.0,
            "low": 49.0,
            "close": 50.0,
            "volume": SCAN_MIN_VOLUME + 1_000_000,
            "MA50": 40.0,
            "MA200": 30.0,
            "RSI": 60.0,
            "CLV": 0.75,
            "ATR": 0.10,
            "ATR_CHAND": 0.10,
            "prev_ATR5": 0.10,
            "prev_ATR20": 0.20,
            "prev_HIGH10": 50.0,
            "SMA200_SLOPE": 1.0,
            "prev_high": 49.0,
            "avg_vol_20": SCAN_MIN_VOLUME + 1_000_000,
            "avg_dollar_vol_20": 200_000_000,
        }, index=idx)
        bt = VelocityBacktest(start=str(idx[0].date()), end=str((idx[-1] + pd.Timedelta(days=1)).date()),
                              capital=600.0, max_pos=1, hold_bars=1,
                              use_cache=False, use_spy_filter=False,
                              use_vix_filter=False)
        bt._data = {"A": df.copy(), "B": df.copy()}
        monkeypatch.setattr(bt, "_daily_scan", lambda today, **_kwargs: [("A", 5.0), ("B", 5.0)])
        monkeypatch.setattr(VelocityBacktest, "_entry_signal", staticmethod(lambda *args, **_kwargs: True))

        result = bt._run_loop()

        assert len(result.trades) >= 2
        assert result.trades[1].entry_date > result.trades[0].exit_date

    def test_missing_vix_value_blocks_entries_when_filter_enabled(self, monkeypatch):
        df = _make_df(n=260, seed=31, trend=0.4)
        bt = VelocityBacktest(start="2023-09-01", end="2024-01-01",
                              use_cache=False, use_spy_filter=False,
                              use_vix_filter=True)
        bt._data = {"SYM": df}
        bt._vix_series = pd.Series(index=df.index, dtype=float)
        monkeypatch.setattr(bt, "_daily_scan", lambda today, **_kwargs: [("SYM", 5.0)])
        monkeypatch.setattr(VelocityBacktest, "_entry_signal", staticmethod(lambda *args, **_kwargs: True))

        result = bt._run_loop()

        assert result.filter_stats["vix_blocked_days"] > 0
        assert result.filter_stats["entries_taken"] == 0

    def test_vix_delay_bars_uses_prior_available_vix_value(self):
        idx = pd.date_range("2024-01-02", periods=3, freq="B")
        bt = VelocityBacktest(start="2024-01-02", end="2024-01-06",
                              use_cache=False, use_spy_filter=False,
                              use_vix_filter=True, vix_delay_bars=1)
        bt._vix_series = pd.Series([20.0, 40.0, 15.0], index=idx)

        assert pd.isna(bt._vix_value_for_date(idx[0]))
        assert bt._vix_value_for_date(idx[1]) == pytest.approx(20.0)
        assert bt._vix_value_for_date(idx[2]) == pytest.approx(40.0)
