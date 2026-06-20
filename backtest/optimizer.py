from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Sequence

import math
import pandas as pd

from backtest.strategy import BacktestResult, VelocityBacktest
from src.config import (
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_COMMISSION_PER_ORDER,
    CHANDELIER_MULT,
    BACKTEST_SCAN_COUNT,
    BACKTEST_MAX_SYMBOLS,
)


@dataclass(frozen=True)
class OptimizationParams:
    chandelier_mult: float = CHANDELIER_MULT


@dataclass(frozen=True)
class OptimizationRun:
    params: OptimizationParams
    train_score: float
    forward_score: float
    robust_score: float
    train_metrics: dict
    forward_metrics: dict


def default_grid() -> List[OptimizationParams]:
    """Small robustness grid over active indicator_swing exit parameters."""
    return [
        OptimizationParams(chandelier_mult=chandelier_mult)
        for chandelier_mult in [0.8, 1.0, 1.2]
    ]


def quick_grid() -> List[OptimizationParams]:
    """Very small grid for a fast smoke-test optimization pass."""
    return [
        OptimizationParams(chandelier_mult=chandelier_mult)
        for chandelier_mult in [0.8, 1.0]
    ]


def score_metrics(metrics: dict, min_trades: int = 20) -> float:
    """Risk-adjusted objective that penalizes thin samples and drawdown."""
    if not metrics or metrics.get("total_trades", 0) < min_trades:
        return float("-inf")
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    total_return = float(metrics.get("total_return_pct", 0.0)) / 100.0
    drawdown_penalty = abs(float(metrics.get("max_drawdown_pct", 0.0))) / 20.0
    profit_factor = min(float(metrics.get("profit_factor", 0.0)), 5.0) / 5.0
    return round(sharpe + total_return + profit_factor - drawdown_penalty, 6)


def _prepare_base(
    start: str,
    end: str,
    capital: float,
    scan_count: int,
    commission_per_order: float,
    use_spy_filter: bool,
    use_vix_filter: bool,
    use_cache: bool,
    max_symbols: int | None = None,
    scoring_model: str | None = None,
    strategy_profile: str | None = None,
) -> VelocityBacktest:
    base = VelocityBacktest(
        start=start,
        end=end,
        capital=capital,
        scan_count=scan_count,
        commission_per_order=commission_per_order,
        max_symbols=max_symbols or BACKTEST_MAX_SYMBOLS,
        use_spy_filter=use_spy_filter,
        use_vix_filter=use_vix_filter,
        use_cache=use_cache,
        scoring_model=scoring_model,
        strategy_profile=strategy_profile,
    )
    if use_cache and base._try_load_cache():
        base._download_regime_data()
    else:
        base._download()
        if use_cache:
            base._save_cache()
    if not base._data:
        raise RuntimeError("No usable data downloaded. Check tickers / dates.")
    _limit_symbols(base, max_symbols)
    base._validate_regime_data()
    return base


def _symbol_liquidity_score(df: pd.DataFrame, start: str, end: str) -> float:
    window = df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]
    if window.empty:
        return float("-inf")
    dvol = window.get("avg_dollar_vol_20")
    if dvol is None:
        dvol = window["close"] * window["volume"]
    recent = dvol.dropna().tail(60)
    if recent.empty:
        return float("-inf")
    score = float(recent.median())
    return score if math.isfinite(score) else float("-inf")


def _limit_symbols(base: VelocityBacktest, max_symbols: int | None) -> None:
    """Keep the most liquid symbols for fast optimization smoke passes."""
    if max_symbols is None or max_symbols <= 0 or len(base._data) <= max_symbols:
        return

    ranked = sorted(
        (
            (_symbol_liquidity_score(df, base.start, base.end), sym)
            for sym, df in base._data.items()
        ),
        reverse=True,
    )
    keep = {sym for score, sym in ranked[:max_symbols] if math.isfinite(score)}
    base._data = {sym: df for sym, df in base._data.items() if sym in keep}
    if not base._data:
        raise RuntimeError("Symbol limit removed all usable data.")


def _slice_data(base: VelocityBacktest, end: str) -> dict:
    end_ts = pd.Timestamp(end)
    return {
        sym: df[df.index < end_ts].copy()
        for sym, df in base._data.items()
        if not df[df.index < end_ts].empty
    }


def _run_with_params(
    base: VelocityBacktest,
    start: str,
    end: str,
    params: OptimizationParams,
) -> BacktestResult:
    bt = VelocityBacktest(
        start=start,
        end=end,
        capital=base.capital,
        max_pos=base.max_pos,
        scan_count=base._scan_count,
        min_price=base._min_price,
        min_volume=base._min_volume,
        min_dollar_vol=base._min_dollar_vol,
        use_spy_filter=base._use_spy_filter,
        use_vix_filter=base._use_vix_filter,
        chandelier_mult=params.chandelier_mult,
        bear_phase_trading=base._bear_phase_trading,
        commission_per_order=base._round_trip_cost / 2.0,
        use_cache=False,
        scoring_model=base._scoring_model,
        strategy_profile=base._profile.name,
    )
    bt._data = _slice_data(base, end)
    bt._spy_bull = base._spy_bull
    bt._spy_return = base._spy_return
    bt._vix_series = base._vix_series
    bt._validate_regime_data()
    return bt._run_loop()


def run_optimization(
    start: str = "2025-01-01",
    split: str = "2026-01-01",
    end: str = "2026-05-01",
    capital: float = BACKTEST_INITIAL_CAPITAL,
    scan_count: int = BACKTEST_SCAN_COUNT,
    commission_per_order: float = BACKTEST_COMMISSION_PER_ORDER,
    grid: Sequence[OptimizationParams] | None = None,
    top_n: int = 10,
    min_train_trades: int = 40,
    min_forward_trades: int = 15,
    use_spy_filter: bool = True,
    use_vix_filter: bool = True,
    use_cache: bool = True,
    max_symbols: int | None = BACKTEST_MAX_SYMBOLS,
    scoring_model: str | None = None,
    strategy_profile: str | None = None,
    progress: bool = False,
) -> List[OptimizationRun]:
    base = _prepare_base(
        start=start,
        end=end,
        capital=capital,
        scan_count=scan_count,
        commission_per_order=commission_per_order,
        use_spy_filter=use_spy_filter,
        use_vix_filter=use_vix_filter,
        use_cache=use_cache,
        max_symbols=max_symbols,
        scoring_model=scoring_model,
        strategy_profile=strategy_profile,
    )
    candidates = list(grid or default_grid())
    runs: List[OptimizationRun] = []
    if progress:
        symbol_scope = len(base._data)
        print(
            f"  Optimizing {len(candidates)} parameter sets over "
            f"{symbol_scope:,} symbols...",
            flush=True,
        )

    for idx, params in enumerate(candidates, start=1):
        train = _run_with_params(base, start, split, params)
        forward = _run_with_params(base, split, end, params)
        train_score = score_metrics(train.metrics, min_train_trades)
        forward_score = score_metrics(forward.metrics, min_forward_trades)
        robust_score = round(min(train_score, forward_score), 6)
        runs.append(
            OptimizationRun(
                params=params,
                train_score=train_score,
                forward_score=forward_score,
                robust_score=robust_score,
                train_metrics=train.metrics,
                forward_metrics=forward.metrics,
            )
        )
        if progress:
            print(
                f"  [{idx:>2}/{len(candidates)}] robust={robust_score:.2f} "
                f"forward={forward_score:.2f} trades_f={forward.metrics.get('total_trades', 0)} "
                f"chand={params.chandelier_mult}",
                flush=True,
            )
    runs.sort(key=lambda r: (r.robust_score, r.forward_score, r.train_score), reverse=True)
    return runs[:top_n]


def format_optimization_table(runs: Iterable[OptimizationRun]) -> str:
    lines = [
        "rank robust forward train trades_f ret_f sharpe_f dd_f params",
        "-" * 120,
    ]
    for rank, run in enumerate(runs, start=1):
        fm = run.forward_metrics
        tm = run.train_metrics
        p = run.params
        lines.append(
            f"{rank:>4} {run.robust_score:>6.2f} {run.forward_score:>7.2f} "
            f"{run.train_score:>6.2f} {fm.get('total_trades', 0):>8} "
            f"{fm.get('total_return_pct', 0.0):>6.1f}% "
            f"{fm.get('sharpe_ratio', 0.0):>7.2f} "
            f"{fm.get('max_drawdown_pct', 0.0):>6.1f}% "
            f"chand={p.chandelier_mult}, "
            f"train_trades={tm.get('total_trades', 0)}"
        )
    return "\n".join(lines)
