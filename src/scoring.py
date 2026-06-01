"""Shared candidate scoring for live trading and forward backtests.

The entry gates decide whether a setup is allowed.  This module only ranks
already-eligible candidates so scarce cash/slots go to the best names first.
"""

from __future__ import annotations

from datetime import datetime, time as dt_time
from typing import Mapping, Optional

import math

from src.config import (
    ATR_PCT_MAX,
    GAP_MAX_PCT,
    RVOL_MIN,
    SCAN_MIN_DOLLAR_VOL,
    SPREAD_MAX_PCT,
)


VALID_SCORING_MODELS = {"legacy", "legacy_v2", "enhanced"}
REGULAR_SESSION_MINUTES = 390.0


def _finite_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def regular_session_elapsed_fraction(now: Optional[datetime]) -> float:
    """Return elapsed US regular-session fraction for live volume pacing.

    Before the open, return a small non-zero fraction so the calculation is
    bounded.  After the close, return 1.0.  The live engine only enters after
    the opening range is available, but this helper stays defensive for tests
    and operational edge cases.
    """
    if now is None:
        return 1.0
    current = now.time()
    open_t = dt_time(9, 30)
    close_t = dt_time(16, 0)
    if current >= close_t:
        return 1.0
    if current <= open_t:
        return 1.0 / REGULAR_SESSION_MINUTES
    minutes = (
        (current.hour - open_t.hour) * 60
        + (current.minute - open_t.minute)
        + current.second / 60.0
    )
    return min(max(minutes / REGULAR_SESSION_MINUTES, 1.0 / REGULAR_SESSION_MINUTES), 1.0)


def volume_pace_from_intraday(
    intraday_volume: float,
    average_daily_volume: float,
    now: Optional[datetime],
) -> float:
    """Normalize live intraday volume to a full-day volume pace.

    Raw intraday volume makes a 10:00 ET stock look artificially quiet because
    only a small part of the session has elapsed.  Volume pace estimates the
    full-day run-rate: current volume divided by expected volume through this
    point in the day.
    """
    intraday_volume = _finite_float(intraday_volume)
    average_daily_volume = _finite_float(average_daily_volume)
    if intraday_volume <= 0 or average_daily_volume <= 0:
        return 0.0
    elapsed = regular_session_elapsed_fraction(now)
    return intraday_volume / (average_daily_volume * elapsed)


def legacy_candidate_score(
    ctx: Mapping,
    *,
    volume_floor: float = RVOL_MIN,
    spread_max_pct: float = SPREAD_MAX_PCT,
) -> float:
    """Original live score: trend, raw RVOL, RSI momentum, spread quality."""
    ma50 = _finite_float(ctx.get("ma50"))
    ma200 = _finite_float(ctx.get("ma200"))
    rsi = _finite_float(ctx.get("rsi"))
    rsi_prev = _finite_float(ctx.get("rsi_prev"))
    volume_pace = _finite_float(
        ctx.get("volume_pace", ctx.get("rvol", ctx.get("rvol_raw", volume_floor)))
    )
    spread_pct = _finite_float(ctx.get("spread_pct"))

    sep = (ma50 - ma200) / ma200 * 100.0 if ma200 else 0.0
    trend = max(0.0, min(sep * 5.0, 30.0))

    floor = max(float(volume_floor), 0.01)
    volume_score = min(max(volume_pace - floor, 0.0) / floor * 25.0, 25.0)

    rsi_delta = rsi - rsi_prev
    accel = min(max(rsi_delta * 1.5, 0.0), 15.0)
    if rsi <= 70:
        level = 10.0
    elif rsi <= 75:
        level = 5.0
    else:
        level = max(0.0, 10.0 - (rsi - 75.0) * 2.0)
    momentum = accel + level

    spread_max = max(float(spread_max_pct), 0.000001)
    liquidity = max(0.0, (spread_max - spread_pct) / spread_max * 20.0)

    return round(max(0.0, min(trend + volume_score + momentum + liquidity, 100.0)), 2)


def legacy_v2_candidate_score(
    ctx: Mapping,
    *,
    volume_floor: float = RVOL_MIN,
    spread_max_pct: float = SPREAD_MAX_PCT,
    atr_pct_max: float = ATR_PCT_MAX,
    gap_max_pct: float = GAP_MAX_PCT,
    dollar_vol_floor: float = SCAN_MIN_DOLLAR_VOL,
) -> float:
    """Legacy score plus small quality tie-breakers.

    The base legacy model remains dominant.  These adjustments are intentionally
    bounded so they can reorder close candidates without turning the scorer into
    a new strategy:
      - volume pace follow-through
      - dollar liquidity depth
      - clean but not stretched ORB extension
      - ATR risk cleanliness
      - mild relief for high RSI when RSI is still rising
    """
    base = legacy_candidate_score(
        ctx,
        volume_floor=volume_floor,
        spread_max_pct=spread_max_pct,
    )
    price = _finite_float(ctx.get("live_price", ctx.get("close")))
    orb_high = _finite_float(ctx.get("orb_high", ctx.get("prev_high")))
    rsi = _finite_float(ctx.get("rsi"))
    rsi_prev = _finite_float(ctx.get("rsi_prev"))
    volume_pace = _finite_float(
        ctx.get("volume_pace", ctx.get("rvol", ctx.get("rvol_raw", volume_floor)))
    )
    atr_chand = _finite_float(ctx.get("atr_chandelier", ctx.get("ATR_CHAND", ctx.get("atr"))))
    dollar_vol = _finite_float(ctx.get("dollar_vol_20d", ctx.get("avg_dollar_vol_20")))

    floor = max(float(volume_floor), 0.01)
    volume_ratio = volume_pace / floor
    volume_bonus = min(max((volume_ratio - 1.0) / 2.0, 0.0), 1.0) * 1.0

    liquidity_bonus = 0.0
    dollar_floor = max(float(dollar_vol_floor), 1.0)
    if dollar_vol > dollar_floor:
        # 10x the minimum dollar-liquidity floor earns the full 1.25 points.
        liquidity_bonus = min(max(math.log10(dollar_vol / dollar_floor), 0.0), 1.0) * 1.25

    gap_cap = max(float(gap_max_pct), 0.005)
    extension_bonus = 0.0
    extension = (price - orb_high) / orb_high if price > 0 and orb_high > 0 else 0.0
    if extension > 0:
        sweet_low = 0.002
        sweet_high = min(0.04, gap_cap * 0.60)
        if extension < sweet_low:
            extension_bonus = 0.50
        elif extension <= sweet_high:
            extension_bonus = 1.50
        elif extension <= gap_cap and gap_cap > sweet_high:
            extension_bonus = 1.50 * (gap_cap - extension) / (gap_cap - sweet_high)
            if extension >= gap_cap * 0.80:
                extension_bonus -= 0.75
        else:
            extension_bonus = -2.00

    atr_bonus = 0.0
    atr_pct = atr_chand / price if price > 0 and atr_chand > 0 else float("inf")
    atr_cap = max(float(atr_pct_max), 0.001)
    clean_zone = min(0.025, atr_cap * 0.40)
    if atr_pct <= 0 or not math.isfinite(atr_pct):
        atr_bonus = 0.0
    elif atr_pct <= clean_zone:
        atr_bonus = 1.25
    elif atr_pct <= atr_cap:
        atr_bonus = 1.25 * (atr_cap - atr_pct) / (atr_cap - clean_zone)
    else:
        atr_bonus = -1.50

    rsi_delta = rsi - rsi_prev
    high_rsi_adjustment = 0.0
    if 75.0 < rsi <= 85.0 and rsi_delta >= 2.0:
        high_rsi_adjustment = min((rsi_delta - 2.0) / 4.0, 1.0) * 1.00
    elif rsi > 90.0:
        high_rsi_adjustment = -1.00

    adjustment = (
        volume_bonus
        + liquidity_bonus
        + extension_bonus
        + atr_bonus
        + high_rsi_adjustment
    )
    return round(max(0.0, min(base + adjustment, 100.0)), 2)


def enhanced_candidate_score(
    ctx: Mapping,
    *,
    volume_floor: float = RVOL_MIN,
    spread_max_pct: float = SPREAD_MAX_PCT,
    atr_pct_max: float = ATR_PCT_MAX,
    gap_max_pct: float = GAP_MAX_PCT,
) -> float:
    """Enhanced ranking score tuned for quality momentum breakouts.

    Components sum to 100:
      trend              22
      volume pace        18
      momentum           20
      liquidity          15
      extension quality  15
      ATR risk quality   10
    """
    ma50 = _finite_float(ctx.get("ma50"))
    ma200 = _finite_float(ctx.get("ma200"))
    price = _finite_float(ctx.get("live_price", ctx.get("close")))
    orb_high = _finite_float(ctx.get("orb_high", ctx.get("prev_high")))
    rsi = _finite_float(ctx.get("rsi"))
    rsi_prev = _finite_float(ctx.get("rsi_prev"))
    volume_pace = _finite_float(
        ctx.get("volume_pace", ctx.get("rvol", ctx.get("rvol_raw", volume_floor)))
    )
    spread_pct = _finite_float(ctx.get("spread_pct"))
    atr_chand = _finite_float(ctx.get("atr_chandelier", ctx.get("ATR_CHAND", ctx.get("atr"))))

    sep = (ma50 - ma200) / ma200 if ma200 else 0.0
    trend = max(0.0, min(sep / 0.08 * 22.0, 22.0))

    floor = max(float(volume_floor), 0.01)
    volume_score = min(max(volume_pace - floor, 0.0) / floor * 18.0, 18.0)

    rsi_delta = rsi - rsi_prev
    accel = min(max(rsi_delta * 1.2, 0.0), 12.0)
    if rsi < 55:
        level = 0.0
    elif rsi <= 72:
        level = 8.0
    elif rsi <= 82:
        level = 7.0
    elif rsi <= 90:
        level = 5.0
    else:
        level = max(2.0, 5.0 - (rsi - 90.0) * 0.5)
    momentum = accel + level

    spread_max = max(float(spread_max_pct), 0.000001)
    liquidity = max(0.0, (spread_max - spread_pct) / spread_max * 15.0)

    extension = (price - orb_high) / orb_high if price > 0 and orb_high > 0 else 0.0
    gap_cap = max(float(gap_max_pct), 0.005)
    sweet_low = 0.002
    sweet_high = min(0.04, gap_cap * 0.60)
    if extension <= 0:
        extension_quality = 0.0
    elif extension < sweet_low:
        extension_quality = 8.0 + (extension / sweet_low) * 4.0
    elif extension <= sweet_high:
        extension_quality = 15.0
    elif extension <= gap_cap and gap_cap > sweet_high:
        extension_quality = 15.0 * (gap_cap - extension) / (gap_cap - sweet_high)
    else:
        extension_quality = 0.0

    atr_pct = atr_chand / price if price > 0 and atr_chand > 0 else float("inf")
    atr_cap = max(float(atr_pct_max), 0.001)
    clean_zone = min(0.03, atr_cap * 0.50)
    if atr_pct <= 0 or not math.isfinite(atr_pct):
        atr_quality = 0.0
    elif atr_pct <= clean_zone:
        atr_quality = 10.0
    elif atr_pct <= atr_cap:
        atr_quality = 10.0 * (atr_cap - atr_pct) / (atr_cap - clean_zone)
    else:
        atr_quality = 0.0

    total = trend + volume_score + momentum + liquidity + extension_quality + atr_quality
    return round(max(0.0, min(total, 100.0)), 2)


def score_candidate(
    ctx: Mapping,
    *,
    model: str = "legacy",
    volume_floor: float = RVOL_MIN,
    spread_max_pct: float = SPREAD_MAX_PCT,
    atr_pct_max: float = ATR_PCT_MAX,
    gap_max_pct: float = GAP_MAX_PCT,
) -> float:
    """Score one candidate using the requested production/research model."""
    model = (model or "legacy").strip().lower()
    if model not in VALID_SCORING_MODELS:
        raise ValueError(f"Unknown scoring model: {model!r}")
    if model == "legacy_v2":
        return legacy_v2_candidate_score(
            ctx,
            volume_floor=volume_floor,
            spread_max_pct=spread_max_pct,
            atr_pct_max=atr_pct_max,
            gap_max_pct=gap_max_pct,
        )
    if model == "enhanced":
        return enhanced_candidate_score(
            ctx,
            volume_floor=volume_floor,
            spread_max_pct=spread_max_pct,
            atr_pct_max=atr_pct_max,
            gap_max_pct=gap_max_pct,
        )
    return legacy_candidate_score(
        ctx,
        volume_floor=volume_floor,
        spread_max_pct=spread_max_pct,
    )
