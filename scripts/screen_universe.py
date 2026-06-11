#!/usr/bin/env python
"""Screen the research equity universe with the active strategy profile.

This is an offline/research snapshot. Live/paper candidate sourcing is handled
by the application scanner (`VELOCITY_APP_SCANNER_SOURCE`), which can combine
IBKR scanner hits with a rotating full-symbol universe.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.strategy import VelocityBacktest
from run_backtest import _load_backtest_data
from src.config import (
    ATR_PCT_MAX,
    BACKTEST_MAX_SYMBOLS,
    SPREAD_MAX_PCT,
    STRATEGY_PROFILE,
)
from src.scoring import score_candidate
from src.strategy_profiles import (
    evaluate_entry_rules,
    get_strategy_profile,
    indicator_sleeve_label,
    select_entry_strategy,
)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _latest_available_date(bt: VelocityBacktest, as_of: date) -> pd.Timestamp:
    cutoff = pd.Timestamp(as_of)
    candidates: list[pd.Timestamp] = []
    if getattr(bt, "_spy_close", None) is not None and not bt._spy_close.empty:
        candidates.extend(pd.Timestamp(d) for d in bt._spy_close.index if pd.Timestamp(d) <= cutoff)
    if not candidates:
        for df in bt._data.values():
            if df is None or df.empty:
                continue
            candidates.extend(pd.Timestamp(d) for d in df.index if pd.Timestamp(d) <= cutoff)
    if not candidates:
        raise RuntimeError(f"No market data available on or before {as_of}.")
    return max(candidates)


def _series_return(series: pd.Series | None, today, bars_back: int) -> float:
    return VelocityBacktest._series_return(series, today, bars_back)


def _candidate_context(
    bt: VelocityBacktest,
    symbol: str,
    row: pd.Series,
    prev_row: pd.Series,
    today: pd.Timestamp,
    rvol: float,
) -> dict:
    spy_ret_63d = _series_return(bt._spy_close, today, 63)
    spy_ret_126d = _series_return(bt._spy_close, today, 126)
    return_13w = _finite(row.get("return_13w"))
    return_26w = _finite(row.get("return_26w"))
    close_price = _finite(row.get("close"))
    open_price = _finite(row.get("open"))
    ctx = {
        "symbol": symbol,
        "ma20": row.get("MA20"),
        "ma50": row.get("MA50"),
        "ma200": row.get("MA200"),
        "sma200_slope": row.get("SMA200_SLOPE"),
        "rsi": row.get("RSI"),
        "rsi_prev": prev_row.get("RSI"),
        "rvol": rvol,
        "rvol_raw": rvol,
        "volume_pace": rvol,
        "spread_pct": 0.0,
        "live_price": close_price,
        "close": close_price,
        "day_open": open_price,
        "orb_high": row.get("prev_high"),
        "prev_high": row.get("prev_high"),
        "prev_daily_high": row.get("prev_high"),
        "high20": row.get("high20"),
        "dist_high20": row.get("dist_high20"),
        "day_range_location": row.get("CLV"),
        "intraday_gain": (
            (close_price - open_price) / open_price
            if np.isfinite(close_price) and np.isfinite(open_price) and open_price > 0
            else np.nan
        ),
        "atr": row.get("ATR"),
        "ATR_CHAND": row.get("ATR_CHAND"),
        "atr_chandelier": row.get("ATR_CHAND", row.get("ATR")),
        "atr_pct": row.get("atr_pct"),
        "macd_hist": row.get("MACD_HIST"),
        "macd_hist_delta": row.get("MACD_HIST_DELTA"),
        "macd_bull_divergence": bool(row.get("MACD_BULL_DIVERGENCE", False)),
        "macd_bear_divergence": bool(row.get("MACD_BEAR_DIVERGENCE", False)),
        "obv_slope_5": row.get("OBV_SLOPE_5"),
        "obv_uptrend": bool(row.get("OBV_UPTREND", False)),
        "obv_bull_divergence": bool(row.get("OBV_BULL_DIVERGENCE", False)),
        "obv_bear_divergence": bool(row.get("OBV_BEAR_DIVERGENCE", False)),
        "ema20_gt_sma50": bool(row.get("EMA20_GT_SMA50", False)),
        "ma_bull_cross": bool(row.get("MA_BULL_CROSS", False)),
        "ma_bear_cross": bool(row.get("MA_BEAR_CROSS", False)),
        "bb_below_lower_2": bool(row.get("BB_BELOW_LOWER_2", False)),
        "bb_above_upper_2": bool(row.get("BB_ABOVE_UPPER_2", False)),
        "bb_reclaim_lower": bool(row.get("BB_RECLAIM_LOWER", False)),
        "psar_bull_3": bool(row.get("PSAR_BULL_3", False)),
        "psar_bear_3": bool(row.get("PSAR_BEAR_3", False)),
        "stoch_k": row.get("STOCH_K"),
        "stoch_d": row.get("STOCH_D"),
        "stoch_bull_exit_oversold": bool(row.get("STOCH_BULL_EXIT_OVERSOLD", False)),
        "stoch_bear_exit_overbought": bool(row.get("STOCH_BEAR_EXIT_OVERBOUGHT", False)),
        "reclaim_ma20": bool(row.get("reclaim_ma20", False)),
        "reclaim_ma50": bool(row.get("reclaim_ma50", False)),
        "break_prev_high": bool(row.get("break_prev_high", False)),
        "weekly_uptrend": bool(row.get("weekly_uptrend", False)),
        "return_13w": return_13w,
        "return_26w": return_26w,
        "relative_strength_63d": (
            return_13w - spy_ret_63d
            if np.isfinite(return_13w) and np.isfinite(spy_ret_63d)
            else np.nan
        ),
        "relative_strength_126d": (
            return_26w - spy_ret_126d
            if np.isfinite(return_26w) and np.isfinite(spy_ret_126d)
            else np.nan
        ),
        "price_vs_52w_high": row.get("price_vs_52w_high"),
        "volume": row.get("volume"),
        "dollar_vol_20d": row.get("avg_dollar_vol_20"),
    }
    ctx.update(bt._analyst_context(symbol, today))
    return ctx


def screen(bt: VelocityBacktest, today: pd.Timestamp, top: int = 100) -> pd.DataFrame:
    profile = bt._profile
    output_columns = [
        "symbol", "score", "strategy", "strategy_label", "close", "volume",
        "volume_pace", "dollar_vol_20d", "rsi", "rs_63d", "rs_126d",
        "return_13w", "return_26w", "price_vs_52w_high", "atr_pct",
        "dist_high20", "weekly_uptrend", "ma20", "ma50", "ma200",
        "ema20_gt_sma50", "break_prev_high", "reclaim_ma20", "reclaim_ma50",
        "macd_hist_delta", "obv_slope_5", "analyst_rating_score",
        "analyst_rating_total",
    ]
    rows: list[dict] = []
    coarse = 0
    failed_profile = 0
    failed_score = 0
    dollar_vol_floor = bt._min_dollar_vol

    for symbol, df in bt._data.items():
        if df is None or today not in df.index:
            continue
        idx = df.index.get_loc(today)
        if idx < 1:
            continue
        row = df.loc[today]
        prev_row = df.iloc[idx - 1]

        close_price = _finite(row.get("close"))
        volume = _finite(row.get("volume"), 0.0)
        if not np.isfinite(close_price) or close_price < bt._min_price:
            continue
        if volume < bt._min_volume:
            continue
        avg_dvol = _finite(row.get("avg_dollar_vol_20"), close_price * volume)
        if not np.isfinite(avg_dvol) or avg_dvol < dollar_vol_floor:
            continue
        avg_vol = _finite(row.get("avg_vol_20"), 0.0)
        if avg_vol <= 0:
            continue

        coarse += 1
        rvol = volume / avg_vol
        ctx = _candidate_context(bt, symbol, row, prev_row, today, rvol)
        entry_strategy = select_entry_strategy(ctx, profile)
        if entry_strategy:
            ctx["entry_strategy"] = entry_strategy
            ctx["entry_strategy_label"] = indicator_sleeve_label(entry_strategy)

        evaluation = evaluate_entry_rules(ctx, profile)
        if not evaluation.passed:
            failed_profile += 1
            continue

        score = score_candidate(
            ctx,
            model=bt._scoring_model,
            volume_floor=profile.min_volume_pace or 1.0,
            spread_max_pct=SPREAD_MAX_PCT,
            atr_pct_max=profile.max_atr_pct or ATR_PCT_MAX,
        )
        if profile.min_score is not None and score < float(profile.min_score):
            failed_score += 1
            continue

        rows.append({
            "symbol": symbol,
            "score": score,
            "strategy": entry_strategy or profile.name,
            "strategy_label": indicator_sleeve_label(entry_strategy or profile.name),
            "close": round(close_price, 4),
            "volume": int(volume),
            "volume_pace": round(rvol, 3),
            "dollar_vol_20d": round(avg_dvol, 2),
            "rsi": round(_finite(ctx.get("rsi")), 2),
            "rs_63d": round(_finite(ctx.get("relative_strength_63d")), 4),
            "rs_126d": round(_finite(ctx.get("relative_strength_126d")), 4),
            "return_13w": round(_finite(ctx.get("return_13w")), 4),
            "return_26w": round(_finite(ctx.get("return_26w")), 4),
            "price_vs_52w_high": round(_finite(ctx.get("price_vs_52w_high")), 4),
            "atr_pct": round(_finite(ctx.get("atr_pct")), 4),
            "dist_high20": round(_finite(ctx.get("dist_high20")), 4),
            "weekly_uptrend": bool(ctx.get("weekly_uptrend")),
            "ma20": round(_finite(ctx.get("ma20")), 4),
            "ma50": round(_finite(ctx.get("ma50")), 4),
            "ma200": round(_finite(ctx.get("ma200")), 4),
            "ema20_gt_sma50": bool(ctx.get("ema20_gt_sma50")),
            "break_prev_high": bool(ctx.get("break_prev_high")),
            "reclaim_ma20": bool(ctx.get("reclaim_ma20")),
            "reclaim_ma50": bool(ctx.get("reclaim_ma50")),
            "macd_hist_delta": round(_finite(ctx.get("macd_hist_delta")), 4),
            "obv_slope_5": round(_finite(ctx.get("obv_slope_5")), 2),
            "analyst_rating_score": round(_finite(ctx.get("analyst_rating_score"), 0.0), 4),
            "analyst_rating_total": int(_finite(ctx.get("analyst_rating_total"), 0.0)),
        })

    out = pd.DataFrame(rows, columns=output_columns)
    if not out.empty:
        out = out.sort_values(
            ["score", "volume_pace", "dollar_vol_20d"],
            ascending=[False, False, False],
        )
    if top and top > 0:
        out = out.head(top)
    out.attrs["coarse_candidates"] = coarse
    out.attrs["failed_profile"] = failed_profile
    out.attrs["failed_score"] = failed_score
    return out.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screen the US common-stock universe with Velocity profile rules.")
    p.add_argument("--as-of", default=str(date.today()), help="Screen date YYYY-MM-DD; uses latest available market date <= this date.")
    p.add_argument("--strategy-profile", default=STRATEGY_PROFILE, help=f"Strategy profile (default: {STRATEGY_PROFILE})")
    p.add_argument("--max-symbols", type=int, default=BACKTEST_MAX_SYMBOLS, help="0 means full filtered universe; use 300 for quick cache smoke test.")
    p.add_argument("--top", type=int, default=100, help="Rows to print/save; 0 saves all passing candidates.")
    p.add_argument("--output", default="", help="CSV path. Default: runtime/research/universe_screen_<date>.csv")
    p.add_argument("--no-cache", action="store_true", help="Force fresh universe/data download.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    as_of = _parse_date(args.as_of)
    end = as_of + timedelta(days=1)
    profile = get_strategy_profile(args.strategy_profile)

    bt = VelocityBacktest(
        start=str(as_of),
        end=str(end),
        max_symbols=args.max_symbols,
        scan_count=0,
        strategy_profile=args.strategy_profile,
        use_cache=not args.no_cache,
    )
    _load_backtest_data(bt)
    screen_day = _latest_available_date(bt, as_of)
    results = screen(bt, screen_day, top=args.top)

    output = args.output
    if not output:
        out_dir = os.path.join(ROOT, "runtime", "research")
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, f"universe_screen_{screen_day.date()}.csv")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    results.to_csv(output, index=False)

    print("\nVELOCITY UNIVERSE SCREEN")
    print(f"  Date              : {screen_day.date()}")
    print(f"  Profile           : {profile.name} ({profile.label})")
    print(f"  Scoring model     : {scoring_model}")
    print(f"  Data universe     : {len(bt._data):,} symbols loaded")
    print(f"  Coarse candidates : {results.attrs.get('coarse_candidates', 0):,}")
    print(f"  Profile rejects   : {results.attrs.get('failed_profile', 0):,}")
    print(f"  Score rejects     : {results.attrs.get('failed_score', 0):,}")
    print(f"  Passing candidates: {len(results):,}{' shown/saved' if args.top else ''}")
    print(f"  Output CSV        : {output}")
    if results.empty:
        print("\nNo symbols passed the selected profile rules.")
    else:
        cols = ["symbol", "score", "strategy_label", "close", "volume_pace", "rs_63d", "rs_126d", "rsi"]
        print("\n" + results[cols].head(min(len(results), 20)).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
