#!/usr/bin/env python
"""
CLI entry-point for the Velocity Strategy forward backtester.

Defaults to 2025-01-01 → 2026-05-01.

Key assumptions baked in:
  • The maintained strategy profile is indicator_swing
  • Shared live/backtest entry rules and scorer
  • ATR-based position sizing (2% equity risk per trade)
  • Chandelier trailing stop as the primary exit mechanism
  • T+1 cash settlement: sale proceeds cannot fund same-day replacement buys
  • VIX regime gate enabled by default to match live-entry risk control
  • Gap-aware stop fills and no same-bar stop ratchet look-ahead
  • Close-or-worse daily entry fills, 0.1% entry slippage, and commissions
  • Data caching to backtest/.cache/ (use --no-cache to force re-download)
  • Filter funnel stats printed after each run

Usage:
    .venv/bin/python run_backtest.py
    .venv/bin/python run_backtest.py --start 2025-01-01 --end 2026-05-01
    .venv/bin/python run_backtest.py --capital 2000
    .venv/bin/python run_backtest.py --no-spy-filter
    .venv/bin/python run_backtest.py --chandelier-mult 1.9
    .venv/bin/python run_backtest.py --start 2020-01-01 --end 2026-05-22 --max-symbols 300 --yearly
    .venv/bin/python run_backtest.py --trades
    .venv/bin/python run_backtest.py --no-cache        # force fresh download
    .venv/bin/python run_backtest.py --optimize --opt-grid quick
    .venv/bin/python run_backtest.py --optimize --opt-grid default --opt-symbol-limit 0
"""

import argparse
import sys
import os
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(__file__))

from backtest.optimizer import format_optimization_table, quick_grid, run_optimization
from backtest.strategy import VelocityBacktest
from src.config import (
    BACKTEST_SCAN_COUNT, BACKTEST_INITIAL_CAPITAL, BACKTEST_COMMISSION_PER_ORDER,
    BACKTEST_MAX_SYMBOLS,
    CHANDELIER_PERIOD,
    TRAIL_PCT,
    EOD_HOLD_MIN_PROFIT_PCT, EOD_HOLD_DAY_RANGE_LOCATION_MIN,
    EOD_HOLD_RELATIVE_STRENGTH_MIN,
    MAX_POSITIONS_CAP, MIN_BUCKET_SIZE, SETTLED_CASH_DEPLOYMENT_PCT,
    STRATEGY_PROFILE,
)
from src.strategy_profiles import get_strategy_profile, profile_names


def parse_args():
    p = argparse.ArgumentParser(description="Velocity Strategy Forward Backtester")
    p.add_argument("--start",           default="2025-01-01",        help="Start date YYYY-MM-DD")
    p.add_argument("--end",             default="2026-05-01",         help="End date YYYY-MM-DD")
    p.add_argument("--capital",         default=BACKTEST_INITIAL_CAPITAL, type=float,
                   help=f"Starting capital USD (default: ${BACKTEST_INITIAL_CAPITAL:,.0f})")
    p.add_argument("--scan-count",      default=BACKTEST_SCAN_COUNT,      type=int,
                   help="Top-N daily scanner picks; 0 means all scanner-passed stocks (default: all)")
    p.add_argument("--commission-per-order", default=BACKTEST_COMMISSION_PER_ORDER, type=float,
                   help=f"Backtest commission assumption per order (default: ${BACKTEST_COMMISSION_PER_ORDER:.2f}; live uses IBKR commission reports)")
    p.add_argument("--max-symbols", default=BACKTEST_MAX_SYMBOLS, type=int,
                   help="Cap downloaded symbols for bounded validation; 0 means full filtered universe")
    p.add_argument("--strategy-profile", choices=profile_names(), default=STRATEGY_PROFILE,
                   help=f"Screening/entry profile to test (default: {STRATEGY_PROFILE})")
    p.add_argument("--min-price", default=None, type=float,
                   help="Override the selected profile's scanner minimum price")
    p.add_argument("--min-volume", default=None, type=float,
                   help="Override the selected profile's scanner minimum daily volume")
    p.add_argument("--min-dollar-vol", default=None, type=float,
                   help="Override the selected profile's 20-day average dollar-volume floor")
    p.add_argument("--trail-pct", default=TRAIL_PCT, type=float,
                   help=f"Flat %% trailing stop distance from peak (default: {TRAIL_PCT:.0%})")
    p.add_argument("--no-spy-filter",   action="store_true",
                   help="Disable SPY regime filter (allow entries in bear market)")
    p.set_defaults(vix_filter=True)
    p.add_argument("--vix-filter", dest="vix_filter", action="store_true",
                   help="Enable VIX regime gate (default; kept for compatibility)")
    p.add_argument("--no-vix-filter", dest="vix_filter", action="store_false",
                   help="Disable VIX regime gate for research only")
    p.add_argument("--vix-delay-bars", default=0, type=int,
                   help="Daily-bar delayed VIX proxy. 0=current VIX bar; 1=prior available VIX bar for delayed-VIX research")
    p.add_argument("--trades",          action="store_true",
                   help="Print top-20 trade log after summary")
    p.add_argument("--yearly",          action="store_true",
                   help="Print separate calendar-year forward results using one consistent downloaded universe")
    p.add_argument("--no-cache",        action="store_true",
                   help="Force fresh data download (ignore backtest/.cache/)")
    p.add_argument("--optimize",        action="store_true",
                   help="Run train/forward parameter validation instead of one backtest")
    p.add_argument("--opt-split",       default="2026-01-01",
                   help="Train/forward split date for --optimize")
    p.add_argument("--opt-top",         default=10, type=int,
                   help="Rows to print from optimization ranking")
    p.add_argument("--opt-grid",        choices=["quick", "default"], default="quick",
                   help="Optimization grid size (quick is recommended for iteration)")
    p.add_argument("--opt-symbol-limit", default=None, type=int,
                   help="Limit symbols for optimization smoke runs. Default: 800 for quick, all for default. Use 0 for all.")
    return p.parse_args()


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _year_ranges(start: str, end: str):
    start_d = _parse_date(start)
    end_d = _parse_date(end)
    year = start_d.year
    while year <= end_d.year:
        y_start = max(start_d, date(year, 1, 1))
        y_end = min(end_d, date(year + 1, 1, 1))
        if y_start < y_end:
            yield str(y_start), str(y_end), year
        year += 1


def _effective_scoring_model(args) -> str:
    profile = get_strategy_profile(args.strategy_profile)
    return profile.scoring_model.strip().lower()


def _daily_fill_label() -> str:
    return "Close-or-worse daily swing fill; exits use the selected profile's stop stack"


def _build_backtest(args, *, start: str, end: str, use_cache: bool) -> VelocityBacktest:
    return VelocityBacktest(
        start          = start,
        end            = end,
        capital        = args.capital,
        scan_count     = args.scan_count,
        commission_per_order = args.commission_per_order,
        max_symbols    = args.max_symbols,
        trail_pct      = args.trail_pct,
        use_spy_filter = not args.no_spy_filter,
        use_vix_filter = args.vix_filter,
        vix_delay_bars = args.vix_delay_bars,
        scoring_model  = _effective_scoring_model(args),
        strategy_profile = args.strategy_profile,
        min_price      = args.min_price,
        min_volume     = args.min_volume,
        min_dollar_vol = args.min_dollar_vol,
        use_cache      = use_cache,
    )


def _load_backtest_data(bt: VelocityBacktest) -> None:
    if bt._use_cache and bt._try_load_cache():
        bt._download_regime_data()
    else:
        bt._download()
        if bt._use_cache:
            bt._save_cache()
    if not bt._data:
        raise RuntimeError("No usable data downloaded. Check tickers / dates.")
    bt._validate_regime_data()


def _print_yearly_report(args) -> None:
    base = _build_backtest(args, start=args.start, end=args.end, use_cache=not args.no_cache)
    _load_backtest_data(base)

    rows = []
    for y_start, y_end, year in _year_ranges(args.start, args.end):
        bt = _build_backtest(args, start=y_start, end=y_end, use_cache=False)
        bt._data = base._data
        bt._spy_bull = base._spy_bull
        bt._spy_return = base._spy_return
        bt._spy_close = getattr(base, "_spy_close", None)
        bt._vix_series = base._vix_series
        bt._vix_delay_bars = base._vix_delay_bars
        bt._validate_regime_data()
        result = bt._run_loop()
        metrics = result.metrics
        rows.append((year, metrics, result.filter_stats))

    print("\nYEARLY FORWARD RESULTS")
    print("Year | Trades | Return% | Win% | PF | MaxDD% | Sharpe | Fine | Entries")
    print("-" * 78)
    for year, metrics, stats in rows:
        if not metrics:
            print(f"{year} |      0 |    0.00 |  0.0 | 0.00 |   0.00 |   0.00 | {stats.get('fine_signals', 0):4d} | {stats.get('entries_taken', 0):7d}")
            continue
        print(
            f"{year} | "
            f"{metrics.get('total_trades', 0):6d} | "
            f"{metrics.get('total_return_pct', 0.0):7.2f} | "
            f"{metrics.get('win_rate', 0.0) * 100:4.1f} | "
            f"{metrics.get('profit_factor', 0.0):4.2f} | "
            f"{metrics.get('max_drawdown_pct', 0.0):6.2f} | "
            f"{metrics.get('sharpe_ratio', 0.0):6.2f} | "
            f"{stats.get('fine_signals', 0):4d} | "
            f"{stats.get('entries_taken', 0):7d}"
        )


def main():
    args = parse_args()
    print(f"\nVELOCITY STRATEGY — FORWARD BACKTEST  (v2)")
    print(f"{'─' * 50}")
    print(f"  Period        : {args.start} → {args.end}")
    print(f"  Capital       : ${args.capital:,.2f}")
    profile = get_strategy_profile(args.strategy_profile)
    print(f"  Profile       : {profile.name} ({profile.label})")
    _deployable_capital = args.capital * SETTLED_CASH_DEPLOYMENT_PCT
    _init_slots = (
        min(int(_deployable_capital / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)
        if _deployable_capital >= MIN_BUCKET_SIZE else 0
    )
    _bucket = _deployable_capital / _init_slots if _init_slots > 0 else 0.0
    print(f"  Max pos       : {MAX_POSITIONS_CAP} cap | Dynamic max=floor(equity/${MIN_BUCKET_SIZE:.0f}) | Initial slots={_init_slots}, bucket≈${_bucket:,.2f} using {SETTLED_CASH_DEPLOYMENT_PCT:.0%} deployable cash")
    print(f"  Entry rules   : {profile.description}")
    scoring_model = _effective_scoring_model(args)
    print(f"  Scoring model : {scoring_model}")
    print(f"  Exit          : Percent trail ({args.trail_pct:.0%} from peak) + 7% hard stop + confirmed analyst downgrade")
    if profile.eod_quality_cleanup:
        print(
            "  EOD cleanup   : same-day quality hold "
            f"(profit>={EOD_HOLD_MIN_PROFIT_PCT:.0%}, "
            f"dayLoc>={EOD_HOLD_DAY_RANGE_LOCATION_MIN:.0%}, "
            f"RS>={EOD_HOLD_RELATIVE_STRENGTH_MIN:.0%}, stop confirmed)"
        )
    else:
        time_stop = (
            f"{profile.time_stop_bars} bars with profit<={profile.time_stop_min_profit:.0%}"
            if profile.time_stop_bars is not None else "disabled"
        )
        print(f"  Swing exits   : no EOD churn | time stop={time_stop}")
    print(f"  Position size : Whole shares, broker-Chandelier risk (2% equity) capped by bucket")
    print(f"  Daily fill     : {_daily_fill_label()}")
    print(f"  Slippage      : 0.1% entry  |  Commission: ${args.commission_per_order*2:.2f}/round-trip")
    print(f"  Symbol cap    : {'FULL filtered universe' if args.max_symbols <= 0 else f'{args.max_symbols:,} symbols'}")
    print(f"  SPY filter    : {'OFF' if args.no_spy_filter else 'ON (SPY > SMA50 > SMA200 and SMA200 rising)'}")
    if args.vix_filter:
        vix_label = "ON (missing VIX or VIX > 35 blocks entries)"
        if args.vix_delay_bars > 0:
            vix_label += f"; delayed proxy={args.vix_delay_bars} daily bar(s)"
    else:
        vix_label = "OFF"
    print(f"  VIX filter    : {vix_label}")
    print(f"  Cache         : {'OFF (forced re-download)' if args.no_cache else 'ON (backtest/.cache/)'}")
    print()

    if args.optimize:
        grid = quick_grid() if args.opt_grid == "quick" else None
        opt_symbol_limit = args.opt_symbol_limit
        if opt_symbol_limit is None and args.max_symbols > 0:
            opt_symbol_limit = args.max_symbols
        if opt_symbol_limit is None and args.opt_grid == "quick":
            opt_symbol_limit = 800
        if opt_symbol_limit is not None and opt_symbol_limit <= 0:
            opt_symbol_limit = None
        runs = run_optimization(
            start          = args.start,
            split          = args.opt_split,
            end            = args.end,
            capital        = args.capital,
            scan_count     = args.scan_count,
            max_symbols    = opt_symbol_limit,
            commission_per_order = args.commission_per_order,
            grid           = grid,
            top_n          = args.opt_top,
            use_spy_filter = not args.no_spy_filter,
            use_vix_filter = args.vix_filter,
            use_cache      = not args.no_cache,
            scoring_model  = scoring_model,
            strategy_profile = args.strategy_profile,
            progress       = True,
        )
        print(format_optimization_table(runs))
        return

    if args.yearly:
        _print_yearly_report(args)
        return

    bt = _build_backtest(args, start=args.start, end=args.end, use_cache=not args.no_cache)
    result = bt.run()
    VelocityBacktest.print_report(result, capital=args.capital)

    if args.trades:
        VelocityBacktest.print_trades(result)


if __name__ == "__main__":
    main()
