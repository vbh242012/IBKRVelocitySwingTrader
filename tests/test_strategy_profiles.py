from dataclasses import replace

import pytest

from src.strategy_profiles import (
    evaluate_entry_rules,
    get_strategy_profile,
    profile_names,
    select_entry_strategy,
)


def _base_ctx(**updates):
    ctx = {
        "live_price": 110.0,
        "close": 110.0,
        "day_open": 104.0,
        "prev_high": 106.0,
        "prev_daily_high": 106.0,
        "ma20": 104.0,
        "ma50": 98.0,
        "ma200": 90.0,
        "sma200_slope": 0.25,
        "ema20_gt_sma50": True,
        "ma_bull_cross": False,
        "rsi": 62.0,
        "rsi_prev": 58.0,
        "rvol": 2.0,
        "volume_pace": 2.0,
        "spread_pct": 0.001,
        "atr_chandelier": 4.0,
        "atr_pct": 4.0 / 110.0,
        "high20": 112.0,
        "dist_high20": 110.0 / 112.0 - 1,
        "macd_hist": 0.20,
        "macd_hist_delta": 0.05,
        "macd_bull_divergence": False,
        "obv_slope_5": 100_000.0,
        "obv_uptrend": True,
        "stoch_bull_exit_oversold": True,
        "psar_bull_3": False,
        "reclaim_ma20": False,
        "reclaim_ma50": False,
        "break_prev_high": True,
        "weekly_uptrend": True,
        "return_13w": 0.25,
        "return_26w": 0.35,
        "relative_strength_63d": 0.15,
        "relative_strength_126d": 0.18,
        "price_vs_52w_high": 0.90,
        "volume": 3_000_000,
        "dollar_vol_20d": 150_000_000,
    }
    ctx.update(updates)
    return ctx


def test_profile_names_only_exposes_maintained_profile():
    assert profile_names() == ("indicator_swing",)
    assert get_strategy_profile().name == "indicator_swing"
    assert get_strategy_profile("indicator_swing").name == "indicator_swing"


def test_indicator_swing_min_entry_score_is_50():
    assert get_strategy_profile("indicator_swing").min_score == pytest.approx(50.0)


@pytest.mark.parametrize(
    "old_name",
    [
        "current",
        "relative_strength_swing",
        "indicator_ma_cross",
        "indicator_bollinger",
        "indicator_psar",
        "reversal_reclaim",
        "five_day_momentum",
        "safer_liquid_momentum",
    ],
)
def test_old_profiles_are_removed(old_name):
    with pytest.raises(ValueError, match="Valid profile: indicator_swing"):
        get_strategy_profile(old_name)


def test_indicator_swing_ma_cross_setup_passes():
    profile = get_strategy_profile("indicator_swing")
    ctx = _base_ctx(ma_bull_cross=True)

    result = evaluate_entry_rules(ctx, profile)

    assert result.passed
    assert select_entry_strategy(ctx, profile) == "ma_cross"


def test_indicator_swing_rejects_weak_trend_and_rs():
    profile = get_strategy_profile("indicator_swing")

    weak_weekly = evaluate_entry_rules(_base_ctx(weekly_uptrend=False), profile)
    weak_rs = evaluate_entry_rules(_base_ctx(relative_strength_63d=-0.01), profile)
    weak_ma = evaluate_entry_rules(_base_ctx(ma50=88.0, ma200=90.0), profile)

    assert "weekly_uptrend" in weak_weekly.failed
    assert "RS_63d>=8%" in weak_rs.failed
    assert "MA50>MA200" in weak_ma.failed


def test_indicator_swing_requires_enabled_sleeve_signal():
    profile = get_strategy_profile("indicator_swing")
    result = evaluate_entry_rules(
        _base_ctx(
            ema20_gt_sma50=False,
            ma_bull_cross=False,
            break_prev_high=False,
            reclaim_ma20=False,
            reclaim_ma50=False,
        ),
        profile,
    )

    assert not result.passed
    assert "indicator_sleeve_signal" in result.failed


def test_optional_bollinger_and_psar_sleeves_are_current_profile_configuration():
    profile = get_strategy_profile("indicator_swing")
    bollinger_profile = replace(profile, indicator_sleeves=("bollinger_reversion",))
    psar_profile = replace(profile, indicator_sleeves=("psar_flip",))

    bollinger_ctx = _base_ctx(
        ema20_gt_sma50=False,
        break_prev_high=False,
        bb_reclaim_lower=True,
        rsi=42.0,
        rsi_prev=38.0,
    )
    psar_ctx = _base_ctx(
        ema20_gt_sma50=False,
        break_prev_high=False,
        psar_bull_3=True,
    )

    assert evaluate_entry_rules(bollinger_ctx, bollinger_profile).passed
    assert select_entry_strategy(bollinger_ctx, bollinger_profile) == "bollinger_reversion"
    assert evaluate_entry_rules(psar_ctx, psar_profile).passed
    assert select_entry_strategy(psar_ctx, psar_profile) == "psar_flip"
