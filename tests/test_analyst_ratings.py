import csv

import pytest

from src.analyst_ratings import AnalystRatingProvider, rating_from_counts
from src.scoring import score_candidate


def _score_ctx(**updates):
    ctx = {
        "ma50": 110.0,
        "ma200": 100.0,
        "ma20": 115.0,
        "sma200_slope": 0.2,
        "rsi": 62.0,
        "rsi_prev": 58.0,
        "volume_pace": 3.0,
        "spread_pct": 0.001,
        "live_price": 120.0,
        "atr_chandelier": 4.0,
        "atr_pct": 4.0 / 120.0,
        "dollar_vol_20d": 200_000_000,
        "entry_strategy": "ma_cross",
        "relative_strength_63d": 0.16,
        "relative_strength_126d": 0.18,
        "return_13w": 0.25,
        "return_26w": 0.35,
        "price_vs_52w_high": 0.92,
        "weekly_uptrend": True,
        "stoch_bull_exit_oversold": True,
        "macd_hist_delta": 0.05,
        "obv_uptrend": True,
    }
    ctx.update(updates)
    return ctx


def test_rating_from_counts_is_bounded_and_confidence_adjusted():
    rating = rating_from_counts(
        "AAPL",
        strong_buy=4,
        buy=3,
        hold=2,
        sell=1,
        strong_sell=0,
        source="test",
    )

    assert rating.score == pytest.approx(0.50)
    assert rating.total == 10
    assert rating.adjusted_score == pytest.approx(0.50)


def test_provider_uses_latest_dated_file_row_without_remote_lookahead(tmp_path):
    ratings_file = tmp_path / "ratings.csv"
    with ratings_file.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["symbol", "period", "strongBuy", "buy", "hold", "sell", "strongSell"],
        )
        writer.writeheader()
        writer.writerow({
            "symbol": "MSFT",
            "period": "2025-01-01",
            "strongBuy": 1,
            "buy": 1,
            "hold": 3,
            "sell": 0,
            "strongSell": 0,
        })
        writer.writerow({
            "symbol": "MSFT",
            "period": "2025-06-01",
            "strongBuy": 4,
            "buy": 2,
            "hold": 0,
            "sell": 0,
            "strongSell": 0,
        })

    provider = AnalystRatingProvider(
        ratings_file=str(ratings_file),
        cache_file=str(tmp_path / "cache.json"),
        allow_remote=False,
    )

    early = provider.get("MSFT", as_of="2025-03-01")
    late = provider.get("MSFT", as_of="2025-07-01")
    missing = provider.get("AAPL", as_of="2025-07-01")

    assert early.period == "2025-01-01"
    assert late.period == "2025-06-01"
    assert late.adjusted_score > early.adjusted_score
    assert missing.source == "no_historical_rating"
    assert missing.adjusted_score == 0.0


def test_analyst_rating_adjusts_candidate_score_but_cannot_exceed_bounds():
    bullish = score_candidate(
        _score_ctx(analyst_rating_raw_score=1.0, analyst_rating_total=10),
        model="indicator_swing",
    )
    neutral = score_candidate(_score_ctx(), model="indicator_swing")
    bearish = score_candidate(
        _score_ctx(analyst_rating_raw_score=-1.0, analyst_rating_total=10),
        model="indicator_swing",
    )

    assert bullish > neutral
    assert bearish < neutral
    assert 0.0 <= bearish <= bullish <= 100.0
