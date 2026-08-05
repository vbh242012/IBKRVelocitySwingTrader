"""Shared candidate scoring for the maintained indicator swing strategy.

Entry gates decide whether a setup is allowed.  This module only ranks
already-eligible candidates so scarce cash/slots go to the best names first.
"""

from __future__ import annotations

import math
from datetime import datetime, time as dt_time
from typing import Mapping, Optional

from src.config import (
    ANALYST_RATING_MIN_ANALYSTS,
    ANALYST_RATING_SCORE_WEIGHT,
    ATR_PCT_MAX,
    INDICATOR_SWING_STOCH_OVERSOLD,
    RECLAIM_TRIGGER_BONUS,
    SCAN_MIN_DOLLAR_VOL,
    SPREAD_MAX_PCT,
)


VALID_SCORING_MODELS = {"indicator_swing"}
REGULAR_SESSION_MINUTES = 390.0


def _finite_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def regular_session_elapsed_fraction(now: Optional[datetime]) -> float:
    """Return elapsed US regular-session fraction for live volume pacing."""
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
    """Normalize live intraday volume to a full-day volume pace."""
    intraday_volume = _finite_float(intraday_volume)
    average_daily_volume = _finite_float(average_daily_volume)
    if intraday_volume <= 0 or average_daily_volume <= 0:
        return 0.0
    elapsed = regular_session_elapsed_fraction(now)
    return intraday_volume / (average_daily_volume * elapsed)


def analyst_rating_adjustment(ctx: Mapping) -> float:
    """Bounded score boost/penalty from analyst consensus."""
    if "analyst_rating_raw_score" in ctx:
        raw = max(-1.0, min(1.0, _finite_float(ctx.get("analyst_rating_raw_score"), 0.0)))
    else:
        raw = max(-1.0, min(1.0, _finite_float(ctx.get("analyst_rating_score"), 0.0)))
    total = _finite_float(ctx.get("analyst_rating_total"), 0.0)
    min_analysts = max(float(ANALYST_RATING_MIN_ANALYSTS), 1.0)
    # No special-case for total<=0: the formula already yields 0.0 confidence
    # there. The prior `else 1.0` granted FULL confidence when the analyst
    # count was missing/zero -- inverted from analyst_ratings.py's sibling
    # `confidence` property, which correctly returns 0.0 in that case.
    confidence = max(0.0, min(total / min_analysts, 1.0))
    return raw * confidence * float(ANALYST_RATING_SCORE_WEIGHT)


def indicator_swing_score(
    ctx: Mapping,
    *,
    volume_floor: float = 1.0,
    spread_max_pct: float = SPREAD_MAX_PCT,
    atr_pct_max: float = ATR_PCT_MAX,
) -> float:
    """Score RS-first indicator swing candidates after hard profile gates."""
    sleeve = str(ctx.get("entry_strategy") or "")
    trigger_score = {
        "ma_cross": 20.0,
        "bollinger_reversion": 18.0,
        "psar_flip": 12.0,
    }.get(sleeve, 12.0)
    # Pullback-reclaim entries carry materially better per-trade economics than
    # prior-high breakouts (2026-07-11 attribution: PF 1.83 vs 1.25), so a
    # bounded bonus lets reclaims win the slot when both signal the same day.
    if (
        sleeve == "ma_cross"
        and RECLAIM_TRIGGER_BONUS > 0
        and (bool(ctx.get("reclaim_ma20")) or bool(ctx.get("reclaim_ma50")))
    ):
        trigger_score += float(RECLAIM_TRIGGER_BONUS)

    rsi = _finite_float(ctx.get("rsi"), float("nan"))
    rsi_prev = _finite_float(ctx.get("rsi_prev", ctx.get("prev_rsi")), float("nan"))
    stoch_k = _finite_float(ctx.get("stoch_k"), float("nan"))
    stoch_d = _finite_float(ctx.get("stoch_d"), float("nan"))
    macd_hist = _finite_float(ctx.get("macd_hist"), 0.0)
    macd_delta = _finite_float(ctx.get("macd_hist_delta"), 0.0)
    volume_pace = _finite_float(ctx.get("volume_pace", ctx.get("rvol")), 0.0)
    spread_pct = _finite_float(ctx.get("spread_pct"), spread_max_pct)
    dollar_vol = _finite_float(ctx.get("dollar_vol_20d", ctx.get("avg_dollar_vol_20")), 0.0)
    price = _finite_float(ctx.get("live_price", ctx.get("close")), 0.0)
    ma20 = _finite_float(ctx.get("ma20", ctx.get("MA20")), float("nan"))
    ma50 = _finite_float(ctx.get("ma50", ctx.get("MA50")), float("nan"))
    ma200 = _finite_float(ctx.get("ma200", ctx.get("MA200")), float("nan"))
    sma200_slope = _finite_float(ctx.get("sma200_slope", ctx.get("SMA200_SLOPE")), float("nan"))
    rs_63d = _finite_float(ctx.get("relative_strength_63d", ctx.get("rs_63d")), 0.0)
    rs_126d = _finite_float(ctx.get("relative_strength_126d", ctx.get("rs_126d")), 0.0)
    ret_13w = _finite_float(ctx.get("return_13w", ctx.get("ret_13w")), 0.0)
    ret_26w = _finite_float(ctx.get("return_26w", ctx.get("ret_26w")), 0.0)
    price_vs_52w_high = _finite_float(ctx.get("price_vs_52w_high"), 0.0)
    atr_chand = _finite_float(ctx.get("atr_chandelier", ctx.get("ATR_CHAND", ctx.get("atr"))), 0.0)
    atr_pct = _finite_float(ctx.get("atr_pct"), float("nan"))
    if not math.isfinite(atr_pct):
        atr_pct = atr_chand / price if price > 0 and atr_chand > 0 else float("inf")

    leadership = 0.0
    leadership += min(max(rs_63d / 0.20, 0.0), 1.0) * 5.0
    leadership += min(max(rs_126d / 0.25, 0.0), 1.0) * 5.0
    leadership += min(max(ret_13w / 0.30, 0.0), 1.0) * 4.0
    leadership += min(max(ret_26w / 0.40, 0.0), 1.0) * 4.0
    if price_vs_52w_high >= 0.90:
        leadership += 4.0
    elif price_vs_52w_high >= 0.85:
        leadership += 2.0
    if ctx.get("weekly_uptrend"):
        leadership += 3.0
    if price > 0 and math.isfinite(ma50) and price > ma50:
        leadership += 2.0
    if math.isfinite(ma50) and math.isfinite(ma200) and ma50 > ma200:
        leadership += 2.0
    if math.isfinite(sma200_slope) and sma200_slope > 0:
        leadership += 1.0

    momentum = 0.0
    if math.isfinite(rsi):
        if 50.0 <= rsi <= 68.0:
            momentum += 8.0
        elif 40.0 <= rsi < 50.0 and sleeve == "bollinger_reversion":
            momentum += 5.0
        elif rsi > 68.0:
            momentum += 3.0
    if math.isfinite(rsi) and math.isfinite(rsi_prev) and rsi > rsi_prev:
        momentum += 3.0
    if ctx.get("stoch_bull_exit_oversold"):
        momentum += 5.0
    elif (
        math.isfinite(stoch_k)
        and math.isfinite(stoch_d)
        and stoch_k > stoch_d
        and stoch_k <= INDICATOR_SWING_STOCH_OVERSOLD + 15
    ):
        momentum += 3.0
    if ctx.get("macd_bull_divergence"):
        momentum += 6.0
    elif macd_delta > 0:
        momentum += min(macd_delta / max(abs(macd_hist), 0.01), 1.0) * 5.0
    if ctx.get("psar_bull_3"):
        momentum += 3.0
    if price > 0 and math.isfinite(ma20) and ma20 > 0 and price <= ma20 * 1.08:
        momentum += 2.0

    volume = 0.0
    if ctx.get("obv_bull_divergence"):
        volume += 6.0
    elif ctx.get("obv_uptrend") or _finite_float(ctx.get("obv_slope_5"), 0.0) > 0:
        volume += 4.0
    floor = max(float(volume_floor or 1.0), 0.01)
    volume += min(max((volume_pace / floor - 1.0) / 1.5, 0.0), 1.0) * 6.0

    liquidity = (
        min(max(math.log10(max(dollar_vol, 1.0) / max(SCAN_MIN_DOLLAR_VOL, 1.0)), 0.0), 1.0)
        * 4.0
    )
    spread_score = (
        max(0.0, (max(spread_max_pct, 0.000001) - spread_pct) / max(spread_max_pct, 0.000001))
        * 3.0
    )

    atr_cap = max(float(atr_pct_max), 0.001)
    if not math.isfinite(atr_pct) or atr_pct <= 0 or atr_pct > atr_cap:
        risk = 0.0
    elif atr_pct <= 0.06:
        risk = 10.0
    else:
        risk = 10.0 * max((atr_cap - atr_pct) / max(atr_cap - 0.06, 0.001), 0.0)

    total = (
        trigger_score
        + min(leadership, 30.0)
        + min(momentum, 25.0)
        + min(volume, 12.0)
        + liquidity
        + spread_score
        + risk
    )
    return round(max(0.0, min(total, 100.0)), 2)


def score_candidate(
    ctx: Mapping,
    *,
    model: str = "indicator_swing",
    volume_floor: float = 1.0,
    spread_max_pct: float = SPREAD_MAX_PCT,
    atr_pct_max: float = ATR_PCT_MAX,
) -> float:
    """Score one candidate using the maintained production model."""
    model = (model or "indicator_swing").strip().lower()
    if model not in VALID_SCORING_MODELS:
        raise ValueError(f"Unknown scoring model: {model!r}. Valid model: indicator_swing")
    base = indicator_swing_score(
        ctx,
        volume_floor=volume_floor,
        spread_max_pct=spread_max_pct,
        atr_pct_max=atr_pct_max,
    )
    return round(max(0.0, min(base + analyst_rating_adjustment(ctx), 100.0)), 2)
