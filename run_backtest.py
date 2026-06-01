#!/usr/bin/env python
"""
CLI entry-point for the Velocity Strategy forward backtester (v2).

Defaults to 2025-01-01 → 2026-05-01 (out-of-sample relative to the
2023-2024 development period).

Key v2 improvements baked in:
  • RVOL-aware daily scanner ranking; 8096 entries no longer gate on RVOL
  • ATR-based position sizing (2% equity risk per trade)
  • Break-even stop floor at 4% profit
  • Trading-bar hold count (not calendar days)
  • T+1 cash settlement: sale proceeds cannot fund same-day replacement buys
  • VIX regime gate enabled by default to match live-entry risk control
  • Gap-aware stop fills and no same-bar stop ratchet look-ahead
  • 0.1% entry slippage + configurable commission per round trip
  • Data caching to backtest/.cache/ (use --no-cache to force re-download)
  • Filter funnel stats printed after each run

Usage:
    .venv/bin/python run_backtest.py
    .venv/bin/python run_backtest.py --start 2025-01-01 --end 2026-05-01
    .venv/bin/python run_backtest.py --capital 2000
    .venv/bin/python run_backtest.py --no-spy-filter
    .venv/bin/python run_backtest.py --hold-bars 7 --chandelier-mult 1.9
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
    BACKTEST_RVOL_MIN, BREAK_EVEN_PCT,
    CHANDELIER_MULT, CHANDELIER_PERIOD, PROFIT_MIN_THRESHOLD, HOLD_TRADING_BARS,
    MAX_POSITIONS_CAP, MIN_BUCKET_SIZE, SETTLED_CASH_DEPLOYMENT_PCT,
    SCORING_MODEL,
)

DEFAULT_HOLD_BARS = HOLD_TRADING_BARS


def parse_args():
    p = argparse.ArgumentParser(description="Velocity Strategy Forward Backtester v2")
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
    p.add_argument("--hold-bars",       default=DEFAULT_HOLD_BARS,    type=int,
                   help=f"Trading bars before velocity-exit check (default: live HOLD_TRADING_BARS={DEFAULT_HOLD_BARS})")
    p.add_argument("--rvol",            default=BACKTEST_RVOL_MIN,    type=float,
                   help=f"Legacy RVOL/ranking reference; 8096 does not gate entries on RVOL (default: {BACKTEST_RVOL_MIN}×)")
    p.add_argument("--scoring-model", choices=["legacy", "legacy_v2", "enhanced"], default=SCORING_MODEL,
                   help=f"Candidate ranking model used by live/backtest scoring (default: {SCORING_MODEL})")
    p.add_argument("--break-even-pct",  default=BREAK_EVEN_PCT,       type=float,
                   help=f"Break-even stop activation threshold (default: {BREAK_EVEN_PCT * 100:.0f}%%)")
    p.add_argument("--chandelier-mult", default=CHANDELIER_MULT,      type=float,
                   help=f"ATR multiple for Chandelier trailing stop (default: {CHANDELIER_MULT})")
    p.add_argument("--no-spy-filter",   action="store_true",
                   help="Disable SPY regime filter (allow entries in bear market)")
    p.set_defaults(vix_filter=True)
    p.add_argument("--vix-filter", dest="vix_filter", action="store_true",
                   help="Enable VIX regime gate (default; kept for compatibility)")
    p.add_argument("--no-vix-filter", dest="vix_filter", action="store_false",
                   help="Disable VIX regime gate for research only")
    p.add_argument("--vix-delay-bars", default=0, type=int,
                   help="Daily-bar delayed VIX proxy. 0=current VIX bar; 1=prior available VIX bar for delayed-VIX research")
    p.add_argument("--conservative-daily-entry", action="store_true",
                   help="Research mode: when a daily signal uses the completed close/RSI/CLV, fill no better than that close")
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


def _build_backtest(args, *, start: str, end: str, use_cache: bool) -> VelocityBacktest:
    return VelocityBacktest(
        start          = start,
        end            = end,
        capital        = args.capital,
        scan_count     = args.scan_count,
        commission_per_order = args.commission_per_order,
        max_symbols    = args.max_symbols,
        hold_bars      = args.hold_bars,
        rvol_min       = args.rvol,
        break_even_pct = args.break_even_pct,
        chandelier_mult= args.chandelier_mult,
        use_spy_filter = not args.no_spy_filter,
        use_vix_filter = args.vix_filter,
        vix_delay_bars = args.vix_delay_bars,
        scoring_model  = args.scoring_model,
        conservative_daily_entry = args.conservative_daily_entry,
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
    _deployable_capital = args.capital * SETTLED_CASH_DEPLOYMENT_PCT
    _init_slots = (
        min(int(_deployable_capital / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)
        if _deployable_capital >= MIN_BUCKET_SIZE else 0
    )
    _bucket = _deployable_capital / _init_slots if _init_slots > 0 else 0.0
    print(f"  Max pos       : {MAX_POSITIONS_CAP} cap | Dynamic max=floor(equity/${MIN_BUCKET_SIZE:.0f}) | Initial slots={_init_slots}, bucket≈${_bucket:,.2f} using {SETTLED_CASH_DEPLOYMENT_PCT:.0%} deployable cash")
    print(f"  Entry rules   : 8096 momentum/risk screener")
    print(f"  Scoring model : {args.scoring_model}")
    print(f"  RVOL ref      : {args.rvol:.1f}× (scanner ranking only; not an entry gate)")
    print(f"  Exit          : Chandelier (ATR{CHANDELIER_PERIOD}×{args.chandelier_mult}) + 7% hard stop + {args.break_even_pct:.0%} break-even")
    print(f"  Velocity exit : profit_min {PROFIT_MIN_THRESHOLD:.0%} after {args.hold_bars} bars")
    print(f"  Hold bars     : {args.hold_bars} trading days before velocity check")
    print(f"  Position size : Whole shares, ATR-based (2% equity risk) capped by bucket")
    print(f"  Daily fill     : {'Conservative close-or-worse' if args.conservative_daily_entry else 'Legacy open/prev-high proxy'}")
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
            scoring_model  = args.scoring_model,
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
