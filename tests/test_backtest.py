"""Offline tests for the maintained indicator_swing forward backtester."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import run_backtest
from backtest.optimizer import OptimizationParams, default_grid, quick_grid, score_metrics
from backtest.strategy import BacktestResult, Trade, VelocityBacktest
from src.strategy_profiles import get_strategy_profile


def _signal_row(**updates) -> pd.Series:
    row = {
        "open": 106.0,
        "high": 112.0,
        "low": 104.0,
        "close": 110.0,
        "volume": 3_000_000,
        "MA20": 104.0,
        "MA50": 98.0,
        "MA200": 90.0,
        "SMA200_SLOPE": 0.25,
        "RSI": 62.0,
        "ATR": 3.0,
        "ATR_CHAND": 4.0,
        "atr_pct": 4.0 / 110.0,
        "CLV": 0.75,
        "prev_high": 106.0,
        "high20": 112.0,
        "dist_high20": 110.0 / 112.0 - 1.0,
        "MACD_HIST": 0.20,
        "MACD_HIST_DELTA": 0.05,
        "MACD_BULL_DIVERGENCE": False,
        "OBV_SLOPE_5": 100_000.0,
        "OBV_UPTREND": True,
        "OBV_BULL_DIVERGENCE": False,
        "EMA20_GT_SMA50": True,
        "MA_BULL_CROSS": False,
        "BB_RECLAIM_LOWER": False,
        "PSAR_BULL_3": False,
        "STOCH_K": 35.0,
        "STOCH_D": 30.0,
        "STOCH_BULL_EXIT_OVERSOLD": True,
        "reclaim_ma20": False,
        "reclaim_ma50": False,
        "break_prev_high": True,
        "weekly_uptrend": True,
        "return_13w": 0.25,
        "return_26w": 0.35,
        "relative_strength_63d": 0.15,
        "relative_strength_126d": 0.18,
        "price_vs_52w_high": 0.90,
        "avg_dollar_vol_20": 150_000_000,
    }
    row.update(updates)
    return pd.Series(row)


def test_common_equity_filter_rejects_non_stock_listings():
    assert VelocityBacktest._is_common_equity_listing("AAPL", "Apple Inc. Common Stock")
    assert VelocityBacktest._is_common_equity_listing("WWW", "Wolverine World Wide, Inc. Common Stock")
    assert not VelocityBacktest._is_common_equity_listing("ABCW", "ABC Acquisition Corp. Warrant")
    assert not VelocityBacktest._is_common_equity_listing("ABCR", "ABC Acquisition Corp. Right")
    assert not VelocityBacktest._is_common_equity_listing("ABCU", "ABC Acquisition Corp. Unit")
    assert not VelocityBacktest._is_common_equity_listing("PREF", "Example Corp. Preferred Stock")
    assert not VelocityBacktest._is_common_equity_listing(np.nan, np.nan)


def test_entry_signal_uses_indicator_swing_rules():
    assert VelocityBacktest._entry_signal(
        _signal_row(),
        prev_rsi=58.0,
        rvol=2.0,
        strategy_profile="indicator_swing",
    )

    assert not VelocityBacktest._entry_signal(
        _signal_row(
            EMA20_GT_SMA50=False,
            MA_BULL_CROSS=False,
            reclaim_ma20=False,
            reclaim_ma50=False,
            break_prev_high=False,
        ),
        prev_rsi=58.0,
        rvol=2.0,
        strategy_profile="indicator_swing",
    )


def test_removed_backtest_kwargs_are_not_accepted():
    with pytest.raises(TypeError):
        VelocityBacktest(hold_bars=1)
    with pytest.raises(TypeError):
        VelocityBacktest(breakout_pct=0.02)
    with pytest.raises(TypeError):
        VelocityBacktest(vcp_ratio=0.8)


def test_optimizer_params_only_cover_active_exit_knobs():
    params = OptimizationParams()
    assert set(params.__dataclass_fields__) == {"trail_pct"}
    assert all(set(p.__dataclass_fields__) == {"trail_pct"} for p in quick_grid())
    assert all(set(p.__dataclass_fields__) == {"trail_pct"} for p in default_grid())


def test_run_backtest_scoring_model_is_profile_owned():
    args = SimpleNamespace(strategy_profile="indicator_swing")
    assert run_backtest._effective_scoring_model(args) == "indicator_swing"


def test_unknown_backtest_profile_is_rejected():
    with pytest.raises(ValueError, match="Valid profile: indicator_swing"):
        VelocityBacktest(strategy_profile="relative_strength_swing")


def test_score_metrics_penalizes_thin_samples():
    assert score_metrics({"total_trades": 2}) == float("-inf")
    assert score_metrics(
        {
            "total_trades": 50,
            "sharpe_ratio": 1.2,
            "total_return_pct": 12.0,
            "max_drawdown_pct": -8.0,
            "profit_factor": 2.0,
        },
        min_trades=20,
    ) > 0


def test_print_report_handles_empty_result(capsys):
    result = BacktestResult(trades=[], equity_curve=pd.Series(dtype=float), metrics={}, filter_stats={})
    VelocityBacktest.print_report(result)
    assert "No trades" in capsys.readouterr().out


def test_trade_net_pnl_includes_round_trip_commission():
    trade = Trade(
        symbol="AAPL",
        entry_date=pd.Timestamp("2026-01-02").date(),
        entry_price=100.0,
        exit_date=pd.Timestamp("2026-01-05").date(),
        exit_price=110.0,
        qty=2,
        round_trip_commission=2.0,
    )
    assert trade.gross_pnl == pytest.approx(20.0)
    assert trade.net_pnl == pytest.approx(18.0)


def test_backtest_analyst_exit_requires_price_confirmation():
    bt = VelocityBacktest()
    trade = Trade(
        symbol="AAPL",
        entry_date=pd.Timestamp("2026-01-02").date(),
        entry_price=100.0,
        qty=5,
    )
    trade.__dict__['_analyst_rating_score'] = -0.50
    today = pd.Timestamp("2026-01-05")

    bt._analyst_context = lambda _symbol, _today: {
        "analyst_rating_score": -0.50,
        "analyst_rating_total": 12,
    }

    assert not bt._analyst_exit_required(
        trade,
        today,
        pd.Series({"close": 105.0, "MA20": 102.0, "MA_BEAR_CROSS": False}),
    )
    assert bt._analyst_exit_required(
        trade,
        today,
        pd.Series({"close": 99.0, "MA20": 102.0, "MA_BEAR_CROSS": False}),
    )
    assert bt._analyst_exit_required(
        trade,
        today,
        pd.Series({"close": 101.0, "MA20": 102.0, "MA_BEAR_CROSS": False}),
    )


def test_profile_names_in_backtest_cli_match_single_profile():
    assert run_backtest.profile_names() == ("indicator_swing",)
    assert get_strategy_profile("indicator_swing").scoring_model == "indicator_swing"
