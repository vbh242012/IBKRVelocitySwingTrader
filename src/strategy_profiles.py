"""Single production screening and entry profile.

The project now supports one maintained strategy: ``indicator_swing``.  Older
ORB, experimental momentum, and standalone research profiles were intentionally
removed to keep the live app and backtester aligned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

from src.config import (
    ATR_PCT_MAX,
    INDICATOR_SWING_MIN_SCORE,
    INDICATOR_SWING_MIN_VOLUME_PACE,
    INDICATOR_SWING_RSI_OVERSOLD,
    INDICATOR_SWING_STRATEGIES,
    INDICATOR_SWING_TIME_STOP_BARS,
    INDICATOR_SWING_TIME_STOP_MIN_PROFIT_PCT,
    SCAN_MIN_DOLLAR_VOL,
    SCAN_MIN_MKTCAP,
    SCAN_MIN_PRICE,
    SCAN_MIN_VOLUME,
    SPREAD_MAX_PCT,
    SWING_MIN_13W_RETURN,
    SWING_MIN_26W_RETURN,
    SWING_MIN_PRICE_VS_52W_HIGH,
    SWING_RS_MIN_63D,
    SWING_RS_MIN_126D,
)


PROFILE_NAME = "indicator_swing"

INDICATOR_SWING_SCAN_CODES = (
    "MOST_ACTIVE_AVG_USD",
    "MOST_ACTIVE_USD",
    "HOT_BY_VOLUME",
    "TOP_VOLUME_RATE",
    "TOP_PERC_GAIN",
    "HIGH_LAST_VS_EMA20",
    "HIGH_LAST_VS_EMA50",
    "BULLISH_MACD_DIST_VS_LAST",
)

VALID_INDICATOR_SLEEVES = ("ma_cross", "bollinger_reversion", "psar_flip")
INDICATOR_SLEEVE_LABELS = {
    "ma_cross": "EMA/SMA Cross",
    "bollinger_reversion": "Bollinger Reclaim",
    "psar_flip": "PSAR Flip",
}


@dataclass(frozen=True)
class StrategyProfile:
    name: str
    label: str
    description: str
    scan_codes: tuple[str, ...]
    min_price: float = SCAN_MIN_PRICE
    min_volume: float = SCAN_MIN_VOLUME
    min_market_cap: float = SCAN_MIN_MKTCAP
    min_dollar_vol: float = SCAN_MIN_DOLLAR_VOL
    max_atr_pct: Optional[float] = ATR_PCT_MAX
    max_spread_pct: float = SPREAD_MAX_PCT
    require_above_ma50: bool = True
    require_ma50_above_ma200: bool = True
    require_sma200_slope_positive: bool = True
    require_weekly_uptrend: bool = True
    min_rs_63d: Optional[float] = None
    min_rs_126d: Optional[float] = None
    min_13w_return: Optional[float] = None
    min_26w_return: Optional[float] = None
    min_price_vs_52w_high: Optional[float] = None
    max_pullback_from_high20: Optional[float] = None
    max_ma20_extension: Optional[float] = None
    min_volume_pace: Optional[float] = None
    indicator_sleeves: tuple[str, ...] = ()
    min_score: Optional[float] = None
    eod_quality_cleanup: bool = False
    friday_close_enabled: bool = False
    allow_bear_phase_entries: bool = False
    time_stop_bars: Optional[int] = None
    time_stop_min_profit: Optional[float] = None
    scanner_filters_enabled: bool = True
    scanner_change_open_pct_above: Optional[float] = None
    scanner_open_gap_pct_below: Optional[float] = 15.0
    scanner_last_vs_ema20_pct_above: Optional[float] = None
    scanner_last_vs_ema50_pct_above: Optional[float] = None
    scanner_macd_histogram_above: Optional[float] = None
    scoring_model: str = PROFILE_NAME


@dataclass(frozen=True)
class EntryEvaluation:
    passed: bool
    failed: tuple[str, ...]
    checks: Mapping[str, bool]


PROFILE = StrategyProfile(
    name=PROFILE_NAME,
    label="Multi-Indicator Swing",
    description=(
        "Relative-strength first swing profile. EMA20>SMA50 trend timing and "
        "MA reclaims/prior-high breaks are the maintained production timing "
        "sleeve; optional Bollinger/PSAR sleeves are disabled unless explicitly "
        "enabled in VELOCITY_INDICATOR_SWING_STRATEGIES."
    ),
    scan_codes=INDICATOR_SWING_SCAN_CODES,
    min_price=10.0,
    min_volume=1_000_000,
    min_market_cap=1_000_000_000,
    min_dollar_vol=75_000_000,
    max_atr_pct=0.12,
    max_spread_pct=0.010,
    require_above_ma50=True,
    require_ma50_above_ma200=True,
    require_sma200_slope_positive=True,
    require_weekly_uptrend=True,
    min_rs_63d=SWING_RS_MIN_63D,
    min_rs_126d=SWING_RS_MIN_126D,
    min_13w_return=SWING_MIN_13W_RETURN,
    min_26w_return=SWING_MIN_26W_RETURN,
    min_price_vs_52w_high=SWING_MIN_PRICE_VS_52W_HIGH,
    max_pullback_from_high20=0.15,
    max_ma20_extension=0.10,
    indicator_sleeves=tuple(
        s for s in INDICATOR_SWING_STRATEGIES if s in VALID_INDICATOR_SLEEVES
    ) or ("ma_cross",),
    min_volume_pace=INDICATOR_SWING_MIN_VOLUME_PACE,
    min_score=INDICATOR_SWING_MIN_SCORE,
    time_stop_bars=INDICATOR_SWING_TIME_STOP_BARS,
    time_stop_min_profit=INDICATOR_SWING_TIME_STOP_MIN_PROFIT_PCT,
    scoring_model=PROFILE_NAME,
)


def profile_names() -> tuple[str, ...]:
    return (PROFILE_NAME,)


def get_strategy_profile(name: str | None = None) -> StrategyProfile:
    profile_name = (name or PROFILE_NAME).strip().lower()
    if profile_name == PROFILE_NAME:
        return PROFILE
    raise ValueError(f"Unknown strategy profile {name!r}. Valid profile: {PROFILE_NAME}")


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


def _check(checks: dict[str, bool], label: str, condition: bool) -> None:
    checks[label] = bool(condition)


def indicator_sleeve_label(sleeve: str | None) -> str:
    return INDICATOR_SLEEVE_LABELS.get(str(sleeve or ""), str(sleeve or "Unknown"))


def indicator_sleeve_signals(ctx: Mapping, profile: StrategyProfile = PROFILE) -> tuple[str, ...]:
    sleeves = profile.indicator_sleeves or ("ma_cross",)
    signals: list[str] = []
    ma_trend_active = _bool(ctx, "ema20_gt_sma50") or _bool(ctx, "ma_bull_cross")
    ma_timing = (
        _bool(ctx, "ma_bull_cross")
        or (
            ma_trend_active
            and (
                _bool(ctx, "break_prev_high")
                or _bool(ctx, "reclaim_ma20")
                or _bool(ctx, "reclaim_ma50")
            )
        )
    )
    if "ma_cross" in sleeves and ma_timing:
        signals.append("ma_cross")
    if "bollinger_reversion" in sleeves and _bool(ctx, "bb_reclaim_lower"):
        signals.append("bollinger_reversion")
    if "psar_flip" in sleeves and _bool(ctx, "psar_bull_3"):
        signals.append("psar_flip")
    return tuple(signals)


def select_entry_strategy(ctx: Mapping, profile: StrategyProfile = PROFILE) -> Optional[str]:
    signals = indicator_sleeve_signals(ctx, profile)
    if not signals:
        return None
    priority = {"ma_cross": 0, "bollinger_reversion": 1, "psar_flip": 2}
    return sorted(signals, key=lambda s: priority.get(s, 99))[0]


def evaluate_entry_rules(
    ctx: Mapping,
    profile: StrategyProfile,
    *,
    overrides: Optional[Mapping[str, float]] = None,
) -> EntryEvaluation:
    """Evaluate one candidate against the maintained swing profile."""
    if profile.name != PROFILE_NAME:
        raise ValueError(f"Unsupported strategy profile: {profile.name!r}")

    overrides = overrides or {}
    checks: dict[str, bool] = {}

    price = _number(ctx, "live_price", "price", "close")
    spread_pct = _number(ctx, "spread_pct", default=0.0)
    volume = _number(ctx, "volume")
    dollar_vol = _number(ctx, "dollar_vol_20d", "avg_dollar_vol_20")
    atr_pct = _number(ctx, "atr_pct")
    if not _present(atr_pct):
        atr_chand = _number(ctx, "atr_chandelier", "ATR_CHAND", "atr", "ATR")
        atr_pct = atr_chand / price if _present(atr_chand) and _present(price) and price > 0 else math.nan

    min_price = float(overrides.get("min_price", profile.min_price))
    min_volume = float(overrides.get("min_volume", profile.min_volume))
    min_dollar_vol = float(overrides.get("min_dollar_vol", profile.min_dollar_vol))
    max_atr_pct = overrides.get("max_atr_pct", profile.max_atr_pct)
    max_spread_pct = float(overrides.get("max_spread_pct", profile.max_spread_pct))

    _check(checks, f"price>={min_price:.2f}", _present(price) and price >= min_price)
    if min_volume > 0:
        _check(checks, f"volume>={min_volume/1e6:.1f}M", _present(volume) and volume >= min_volume)
    _check(
        checks,
        f"dollar_vol>={min_dollar_vol/1e6:.0f}M",
        _present(dollar_vol) and dollar_vol >= min_dollar_vol,
    )
    _check(checks, f"spread<={max_spread_pct*100:.2f}%", _present(spread_pct) and spread_pct <= max_spread_pct)
    if max_atr_pct is not None:
        _check(checks, f"ATR%<={float(max_atr_pct)*100:.0f}%", _present(atr_pct) and atr_pct <= float(max_atr_pct))

    ma50 = _number(ctx, "ma50", "MA50")
    ma200 = _number(ctx, "ma200", "MA200")
    _check(checks, "price>MA50", _present(price) and _present(ma50) and price > ma50)
    _check(checks, "MA50>MA200", _present(ma50) and _present(ma200) and ma50 > ma200)
    sma200_slope = _number(ctx, "sma200_slope", "SMA200_SLOPE")
    _check(checks, "SMA200_slope>0", _present(sma200_slope) and sma200_slope > 0)

    signals = indicator_sleeve_signals(ctx, profile)
    _check(checks, "indicator_sleeve_signal", bool(signals))
    _check(checks, "weekly_uptrend", _bool(ctx, "weekly_uptrend"))

    rs_63d = _number(ctx, "relative_strength_63d", "rs_63d")
    _check(
        checks,
        f"RS_63d>={float(profile.min_rs_63d)*100:.0f}%",
        _present(rs_63d) and rs_63d >= float(profile.min_rs_63d),
    )
    rs_126d = _number(ctx, "relative_strength_126d", "rs_126d")
    _check(
        checks,
        f"RS_126d>={float(profile.min_rs_126d)*100:.0f}%",
        _present(rs_126d) and rs_126d >= float(profile.min_rs_126d),
    )
    ret_13w = _number(ctx, "return_13w", "ret_13w")
    _check(
        checks,
        f"return_13w>={float(profile.min_13w_return)*100:.0f}%",
        _present(ret_13w) and ret_13w >= float(profile.min_13w_return),
    )
    ret_26w = _number(ctx, "return_26w", "ret_26w")
    _check(
        checks,
        f"return_26w>={float(profile.min_26w_return)*100:.0f}%",
        _present(ret_26w) and ret_26w >= float(profile.min_26w_return),
    )
    price_vs_52w_high = _number(ctx, "price_vs_52w_high")
    _check(
        checks,
        f"price_vs_52w_high>={float(profile.min_price_vs_52w_high):.2f}",
        _present(price_vs_52w_high)
        and price_vs_52w_high >= float(profile.min_price_vs_52w_high),
    )
    dist_high20 = _number(ctx, "dist_high20")
    _check(
        checks,
        f"pullback_high20<={float(profile.max_pullback_from_high20)*100:.0f}%",
        _present(dist_high20) and dist_high20 >= -float(profile.max_pullback_from_high20),
    )
    ma20 = _number(ctx, "ma20", "MA20")
    ma20_extension = (
        price / ma20 - 1
        if _present(price) and _present(ma20) and ma20 > 0 else math.nan
    )
    _check(
        checks,
        f"MA20_extension<={float(profile.max_ma20_extension)*100:.0f}%",
        _present(ma20_extension) and ma20_extension <= float(profile.max_ma20_extension),
    )
    pace = _number(ctx, "volume_pace", "rvol")
    _check(
        checks,
        f"volume_pace>={float(profile.min_volume_pace):.1f}x",
        _present(pace) and pace >= float(profile.min_volume_pace),
    )

    rsi = _number(ctx, "rsi", "RSI")
    rsi_prev = _number(ctx, "rsi_prev", "prev_rsi")
    rsi_momentum = _present(rsi) and rsi >= 50.0
    rsi_recovery = (
        "bollinger_reversion" in (profile.indicator_sleeves or ())
        and _present(rsi)
        and _present(rsi_prev)
        and rsi >= INDICATOR_SWING_RSI_OVERSOLD
        and rsi > rsi_prev
    )
    _check(checks, "RSI_momentum_or_recovery", rsi_momentum or rsi_recovery)

    stoch_bull = _bool(ctx, "stoch_bull_exit_oversold")
    macd_delta = _number(ctx, "macd_hist_delta", "MACD_HIST_DELTA")
    macd_bull = _bool(ctx, "macd_bull_divergence") or (_present(macd_delta) and macd_delta > 0)
    obv_bull = _bool(ctx, "obv_uptrend") or _bool(ctx, "obv_bull_divergence")
    psar_confirm = _bool(ctx, "psar_bull_3")
    volume_confirm = _present(pace) and pace >= float(profile.min_volume_pace)
    confirmations = sum(bool(v) for v in (stoch_bull, macd_bull, obv_bull, psar_confirm, volume_confirm))
    _check(checks, "two_momentum_volume_confirmations", confirmations >= 2)

    failed = tuple(label for label, ok in checks.items() if not ok)
    return EntryEvaluation(passed=not failed, failed=failed, checks=checks)
