"""Standalone Bollinger mean-reversion strategy.

Separate from the `indicator_swing` "bollinger_reversion" sleeve in
`src/strategy_profiles.py`, which still requires the full trend/relative-
strength gate stack (price > MA50, MA50 > MA200, near 52-week highs, etc.)
and uses only the standard percent trailing stop as its exit. This module is
an ungated mean-reversion strategy: entry on `BB_RECLAIM_LOWER` alone (two
prior closes below the lower Bollinger band, then a reclaim), no trend/RS
requirement. Exit is midline reclaim (take profit), a tighter hard stop, or
a short time-stop for stragglers.

Kept in its own module rather than added to `strategy_profiles.py` because
that module's `get_strategy_profile()` / `evaluate_entry_rules()` explicitly
declare themselves single-profile and raise `ValueError` for anything else —
this strategy is not a `StrategyProfile` variant, it runs as an independent,
additional entry/exit path alongside `indicator_swing`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple

from src.config import (
    BOLLINGER_STANDALONE_MIN_DOLLAR_VOL,
    BOLLINGER_STANDALONE_TIME_STOP_DAYS,
    SPREAD_MAX_PCT,
)

ENTRY_STRATEGY_NAME = "bollinger_reversion_standalone"


@dataclass(frozen=True)
class BollingerEntryEvaluation:
    passed: bool
    failed: Tuple[str, ...]
    checks: Mapping[str, bool]


def _number(ctx: Mapping, *keys: str, default: float = math.nan) -> float:
    for key in keys:
        if key not in ctx:
            continue
        try:
            value = float(ctx.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return default


def _present(value: float) -> bool:
    return math.isfinite(value)


def _bool(ctx: Mapping, key: str) -> bool:
    value = ctx.get(key)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        return bool(value)
    except Exception:
        return False


def evaluate_bollinger_standalone_entry(ctx: Mapping) -> BollingerEntryEvaluation:
    """Evaluate a candidate for the standalone Bollinger reversion entry.

    Deliberately has no trend/relative-strength gates — a beaten-down stock
    is exactly what BB_RECLAIM_LOWER is meant to catch. Only liquidity,
    spread, and the reclaim signal itself are checked.
    """
    checks: dict[str, bool] = {}

    price = _number(ctx, "live_price", "price", "close")
    # No default=0.0 override: the live scanner sets spread_pct to
    # float('inf') as an explicit "unavailable -> fail closed" sentinel.
    # Overriding to 0.0 silently turned an unknown spread into a perfect 0%
    # spread, defeating the spread gate below. Backtest/prefilter contexts
    # always set an explicit finite spread_pct (0.0), so they are
    # unaffected by the NaN default.
    spread_pct = _number(ctx, "spread_pct")
    dollar_vol = _number(ctx, "dollar_vol_20d", "avg_dollar_vol_20")

    checks["price>0"] = _present(price) and price > 0
    checks[f"spread<={SPREAD_MAX_PCT*100:.2f}%"] = _present(spread_pct) and spread_pct <= SPREAD_MAX_PCT
    checks[f"dollar_vol>={BOLLINGER_STANDALONE_MIN_DOLLAR_VOL/1e6:.0f}M"] = (
        _present(dollar_vol) and dollar_vol >= BOLLINGER_STANDALONE_MIN_DOLLAR_VOL
    )
    checks["bb_reclaim_lower"] = _bool(ctx, "bb_reclaim_lower")

    failed = tuple(label for label, ok in checks.items() if not ok)
    return BollingerEntryEvaluation(passed=not failed, failed=failed, checks=checks)


def bollinger_standalone_rank(ctx: Mapping) -> float:
    """Rank candidates by depth of reclaim opportunity (higher = preferred).

    Mirrors the backtested RANK formula: (BB_MID - close) / BB_MID — a
    reclaim from deeper below the midline ranks higher. Returns -inf when
    the inputs aren't available so an incomplete candidate always sorts last
    rather than raising.
    """
    price = _number(ctx, "live_price", "price", "close")
    bb_mid = _number(ctx, "bb_mid", "BB_MID")
    if not (_present(price) and _present(bb_mid) and bb_mid > 0):
        return float("-inf")
    return (bb_mid - price) / bb_mid


def bollinger_standalone_midline_reclaim(last_row: Mapping) -> bool:
    """True when the last completed daily bar closed back above the BB midline.

    Uses completed daily-bar data (matching the other indicator-strategy exit
    checks in `_indicator_strategy_exit_required`), not the live intraday
    price, so the take-profit decision isn't made on an in-progress bar.
    """
    try:
        close = float(last_row.get("close"))
        bb_mid = float(last_row.get("BB_MID"))
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(close) and math.isfinite(bb_mid)):
        return False
    return close > bb_mid


def bollinger_standalone_time_stop_due(trading_bars_held: int) -> bool:
    return int(trading_bars_held or 0) >= BOLLINGER_STANDALONE_TIME_STOP_DAYS
