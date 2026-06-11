"""Analyst recommendation integration for live scoring and forward tests.

Ratings are confirmation, not a primary alpha signal.  The provider returns a
bounded score in [-1, 1] derived from recommendation counts:

    strongBuy=+2, buy=+1, hold=0, sell=-1, strongSell=-2

Live mode can fetch Finnhub recommendation trends when configured.  Backtests
should use dated local snapshots only, so historical research does not leak
today's analyst consensus into old trades.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Dict, Iterable, Optional

from src.config import (
    ANALYST_RATING_MIN_ANALYSTS,
    ANALYST_RATINGS_CACHE_FILE,
    ANALYST_RATINGS_ENABLED,
    ANALYST_RATINGS_FILE,
    ANALYST_RATINGS_TTL_SEC,
    FINNHUB_API_KEY,
)


@dataclass(frozen=True)
class AnalystRating:
    symbol: str
    score: float = 0.0
    total: int = 0
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    period: Optional[str] = None
    source: str = "neutral"
    fetched_at: float = 0.0

    @property
    def confidence(self) -> float:
        floor = max(int(ANALYST_RATING_MIN_ANALYSTS), 1)
        return max(0.0, min(float(self.total) / floor, 1.0))

    @property
    def adjusted_score(self) -> float:
        return max(-1.0, min(1.0, self.score)) * self.confidence

    def as_context(self) -> dict:
        return {
            "analyst_rating_score": self.adjusted_score,
            "analyst_rating_raw_score": self.score,
            "analyst_rating_total": self.total,
            "analyst_rating_source": self.source,
            "analyst_rating_period": self.period,
        }


def neutral_rating(symbol: str, source: str = "neutral") -> AnalystRating:
    return AnalystRating(symbol=str(symbol or "").upper(), source=source, fetched_at=time.time())


def rating_from_counts(
    symbol: str,
    *,
    strong_buy: int = 0,
    buy: int = 0,
    hold: int = 0,
    sell: int = 0,
    strong_sell: int = 0,
    period: Optional[str] = None,
    source: str = "counts",
    fetched_at: Optional[float] = None,
) -> AnalystRating:
    counts = {
        "strong_buy": max(0, int(strong_buy or 0)),
        "buy": max(0, int(buy or 0)),
        "hold": max(0, int(hold or 0)),
        "sell": max(0, int(sell or 0)),
        "strong_sell": max(0, int(strong_sell or 0)),
    }
    total = sum(counts.values())
    if total <= 0:
        return neutral_rating(symbol, source=source)
    raw = (
        2 * counts["strong_buy"]
        + counts["buy"]
        - counts["sell"]
        - 2 * counts["strong_sell"]
    )
    score = raw / (2.0 * total)
    if not math.isfinite(score):
        score = 0.0
    return AnalystRating(
        symbol=str(symbol or "").upper(),
        score=max(-1.0, min(1.0, score)),
        total=total,
        period=period,
        source=source,
        fetched_at=time.time() if fetched_at is None else float(fetched_at),
        **counts,
    )


def _coerce_int(row: dict, *names: str) -> int:
    for name in names:
        if name in row and row[name] not in (None, ""):
            try:
                return int(float(row[name]))
            except (TypeError, ValueError):
                continue
    return 0


def _coerce_float(row: dict, *names: str) -> Optional[float]:
    for name in names:
        if name in row and row[name] not in (None, ""):
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return None


def _row_period(row: dict) -> Optional[str]:
    for key in ("period", "date", "as_of", "asof"):
        value = row.get(key)
        if value:
            return str(value)[:10]
    return None


def rating_from_row(row: dict, *, source: str) -> AnalystRating:
    symbol = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
    explicit_score = _coerce_float(row, "score", "analyst_score", "rating_score")
    total = _coerce_int(row, "total", "total_analysts", "analyst_count")
    if explicit_score is not None:
        score = max(-1.0, min(1.0, explicit_score))
        return AnalystRating(
            symbol=symbol,
            score=score,
            total=total,
            period=_row_period(row),
            source=source,
            fetched_at=time.time(),
        )
    return rating_from_counts(
        symbol,
        strong_buy=_coerce_int(row, "strongBuy", "strong_buy", "strong_buy_count"),
        buy=_coerce_int(row, "buy", "buy_count"),
        hold=_coerce_int(row, "hold", "hold_count"),
        sell=_coerce_int(row, "sell", "sell_count"),
        strong_sell=_coerce_int(row, "strongSell", "strong_sell", "strong_sell_count"),
        period=_row_period(row),
        source=source,
    )


class AnalystRatingProvider:
    """Rating provider with CSV snapshots, JSON cache, and optional Finnhub live fetch."""

    def __init__(
        self,
        *,
        enabled: bool = ANALYST_RATINGS_ENABLED,
        allow_remote: bool = True,
        api_key: str = FINNHUB_API_KEY,
        ratings_file: str = ANALYST_RATINGS_FILE,
        cache_file: str = ANALYST_RATINGS_CACHE_FILE,
        ttl_sec: float = ANALYST_RATINGS_TTL_SEC,
    ):
        self.enabled = bool(enabled)
        self.allow_remote = bool(allow_remote)
        self.api_key = str(api_key or "").strip()
        self.ratings_file = str(ratings_file or "").strip()
        self.cache_file = str(cache_file or "").strip()
        self.ttl_sec = max(60.0, float(ttl_sec or 0))
        self._file_rows: Dict[str, list[AnalystRating]] = {}
        self._cache: Dict[str, AnalystRating] = {}
        self._load_file_rows()
        self._load_cache()

    def _load_file_rows(self) -> None:
        if not self.ratings_file or not os.path.exists(self.ratings_file):
            return
        try:
            with open(self.ratings_file, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    rating = rating_from_row(row, source="file")
                    if not rating.symbol:
                        continue
                    self._file_rows.setdefault(rating.symbol, []).append(rating)
            for rows in self._file_rows.values():
                rows.sort(key=lambda r: r.period or "")
        except OSError:
            self._file_rows = {}

    def _load_cache(self) -> None:
        if not self.cache_file or not os.path.exists(self.cache_file):
            return
        try:
            with open(self.cache_file) as fh:
                payload = json.load(fh)
            if not isinstance(payload, dict):
                return
            for symbol, row in payload.items():
                if not isinstance(row, dict):
                    continue
                row.setdefault("symbol", symbol)
                rating = AnalystRating(**{
                    "symbol": str(row.get("symbol", symbol)).upper(),
                    "score": float(row.get("score", 0.0)),
                    "total": int(row.get("total", 0)),
                    "strong_buy": int(row.get("strong_buy", 0)),
                    "buy": int(row.get("buy", 0)),
                    "hold": int(row.get("hold", 0)),
                    "sell": int(row.get("sell", 0)),
                    "strong_sell": int(row.get("strong_sell", 0)),
                    "period": row.get("period"),
                    "source": str(row.get("source", "cache")),
                    "fetched_at": float(row.get("fetched_at", 0.0)),
                })
                self._cache[rating.symbol] = rating
        except (OSError, ValueError, TypeError):
            self._cache = {}

    def _save_cache(self) -> None:
        if not self.cache_file:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
            with open(self.cache_file, "w") as fh:
                json.dump(
                    {symbol: asdict(rating) for symbol, rating in self._cache.items()},
                    fh,
                    indent=2,
                    sort_keys=True,
                )
        except OSError:
            return

    @staticmethod
    def _date_key(as_of: Optional[date | datetime | str]) -> Optional[str]:
        if as_of is None:
            return None
        if isinstance(as_of, datetime):
            return as_of.date().isoformat()
        if isinstance(as_of, date):
            return as_of.isoformat()
        return str(as_of)[:10]

    def _from_file(self, symbol: str, as_of: Optional[date | datetime | str]) -> Optional[AnalystRating]:
        rows = self._file_rows.get(symbol)
        if not rows:
            return None
        as_of_key = self._date_key(as_of)
        if as_of_key is None:
            return rows[-1]
        eligible = [
            row for row in rows
            if not row.period or row.period <= as_of_key
        ]
        return eligible[-1] if eligible else None

    def _from_cache(self, symbol: str) -> Optional[AnalystRating]:
        rating = self._cache.get(symbol)
        if rating is None:
            return None
        if time.time() - float(rating.fetched_at or 0.0) > self.ttl_sec:
            return None
        return rating

    def _fetch_finnhub(self, symbol: str) -> Optional[AnalystRating]:
        if not self.allow_remote or not self.api_key:
            return None
        encoded_symbol = urllib.parse.quote(symbol)
        token = urllib.parse.quote(self.api_key)
        url = (
            "https://finnhub.io/api/v1/stock/recommendation"
            f"?symbol={encoded_symbol}&token={token}"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "VelocitySwingTrader/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not isinstance(payload, list) or not payload:
            return None
        latest = payload[0]
        return rating_from_counts(
            symbol,
            strong_buy=_coerce_int(latest, "strongBuy"),
            buy=_coerce_int(latest, "buy"),
            hold=_coerce_int(latest, "hold"),
            sell=_coerce_int(latest, "sell"),
            strong_sell=_coerce_int(latest, "strongSell"),
            period=_row_period(latest),
            source="finnhub",
        )

    def get(self, symbol: str, *, as_of: Optional[date | datetime | str] = None) -> AnalystRating:
        symbol = str(symbol or "").strip().upper()
        if not symbol or not self.enabled:
            return neutral_rating(symbol, source="disabled")

        file_rating = self._from_file(symbol, as_of)
        if file_rating is not None:
            return file_rating

        # Historical backtests must not fall back to current remote ratings.
        if as_of is not None:
            return neutral_rating(symbol, source="no_historical_rating")

        cached = self._from_cache(symbol)
        if cached is not None:
            return cached

        try:
            fetched = self._fetch_finnhub(symbol)
        except Exception:
            fetched = None
        if fetched is None:
            return neutral_rating(symbol, source="unavailable")
        self._cache[symbol] = fetched
        self._save_cache()
        return fetched


def apply_rating_context(ctx: dict, rating: AnalystRating) -> dict:
    ctx.update(rating.as_context())
    return ctx
