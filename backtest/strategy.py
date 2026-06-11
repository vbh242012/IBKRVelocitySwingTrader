"""Forward backtester for the maintained ``indicator_swing`` strategy.

The backtester intentionally mirrors the live swing-entry code path:
candidate rows are enriched with the same indicator fields, screened through
``evaluate_entry_rules``, ranked with the shared scorer, and exited with the
same tiered profit trim, Chandelier, hard-stop, break-even,
analyst-downgrade, strategy-exit, and time-stop stack.

Daily bars cannot know the intraday fill sequence, so entries fill no better
than the completed signal-day close plus the configured slippage. Stop fills
use the open when price gaps through the stop.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import random
import re
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    ANALYST_RATING_EXIT_ENABLED,
    ANALYST_RATING_SELL_THRESHOLD,
    EOD_HOLD_MIN_PROFIT_PCT,
    EOD_HOLD_DAY_RANGE_LOCATION_MIN,
    EOD_HOLD_RELATIVE_STRENGTH_MIN,
    EOD_HOLD_REQUIRE_STOP_CONFIRMED,
    RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW,
    BACKTEST_INITIAL_CAPITAL, MAX_POSITIONS_CAP, MIN_BUCKET_SIZE, BACKTEST_SCAN_COUNT,
    BACKTEST_MAX_SYMBOLS,
    SETTLED_CASH_DEPLOYMENT_PCT,
    SCAN_MIN_PRICE, SCAN_MIN_VOLUME, SCAN_MIN_DOLLAR_VOL,
    STRATEGY_PROFILE,
    VIX_THRESHOLD,
    MIN_CANDLES,
    CHANDELIER_PERIOD, CHANDELIER_MULT,
    ATR_PCT_MAX, HARD_STOP_PCT,
    SMA200_SLOPE_LOOKBACK,
    SPREAD_MAX_PCT,
    RISK_PER_TRADE_PCT, BREAK_EVEN_PCT,
    TIERED_PROFIT_EXIT_ENABLED, TIERED_PROFIT_EXIT_R_LEVELS,
    BACKTEST_COMMISSION_PER_ORDER,
    BEAR_PHASE_TRADING_ENABLED, BEAR_PHASE_RISK_MULT,
    BEAR_PHASE_DOLLAR_VOL_MULT,
)
from src.analyst_ratings import AnalystRatingProvider
from src.indicators import apply_all
from src.scoring import score_candidate
from src.strategy_profiles import (
    evaluate_entry_rules,
    get_strategy_profile,
    indicator_sleeve_label,
    select_entry_strategy,
)

warnings.filterwarnings("ignore", category=FutureWarning)

_NASDAQ_LISTED_URL = 'https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt'
_OTHER_LISTED_URL  = 'https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt'
_CACHE_DIR         = os.path.join(os.path.dirname(__file__), ".cache")
_DEFAULT_ROUND_TRIP_COST = BACKTEST_COMMISSION_PER_ORDER * 2
_CACHE_VERSION = "v7_indicator_swing"
_CACHE_COMPATIBLE_VERSIONS = {
    _CACHE_VERSION,
    # Earlier swing cache versions contain the same enriched OHLCV columns;
    # scanner floors and scoring are applied at run time.
    "v6_strict_swing_momentum",
    "v5_relative_strength_swing",
}
_CACHE_BASE_COLUMNS = {"open", "high", "low", "close", "volume"}
_CACHE_REQUIRED_COLUMNS = _CACHE_BASE_COLUMNS | {
    "MA50", "MA200", "ATR_CHAND", "RSI", "CLV", "SMA200_SLOPE",
    "prev_close", "prev_high", "high20", "high50", "dist_high20", "dist_high50",
    "atr_pct", "MACD_HIST", "MACD_HIST_DELTA", "OBV", "OBV_SLOPE_5",
    "reclaim_ma20", "reclaim_ma50", "break_prev_high", "avg_vol_20",
    "avg_dollar_vol_20", "return_13w", "return_26w", "high_52w",
    "price_vs_52w_high", "weekly_uptrend", "prev_ATR5", "prev_ATR20",
    "prev_HIGH10", "EMA20_GT_SMA50", "MA_BULL_CROSS", "MA_BEAR_CROSS",
    "BB_BELOW_LOWER_2", "BB_ABOVE_UPPER_2", "BB_RECLAIM_LOWER",
    "PSAR_BULL_3", "PSAR_BEAR_3",
    "STOCH_K", "STOCH_D",
    "STOCH_BULL_EXIT_OVERSOLD", "STOCH_BEAR_EXIT_OVERBOUGHT",
    "MACD_BULL_DIVERGENCE", "MACD_BEAR_DIVERGENCE", "OBV_UPTREND",
    "OBV_BULL_DIVERGENCE", "OBV_BEAR_DIVERGENCE",
}
_NON_COMMON_NAME_RE = re.compile(
    r"\b("
    r"warrant|warrants|right|rights|unit|units|preferred|preference|"
    r"note|notes|bond|bonds|debenture|debentures|etf|etn|fund"
    r")\b",
    re.IGNORECASE,
)
_COMMON_EQUITY_NAME_RE = re.compile(
    r"\b(common stock|common shares|ordinary shares|american depositary shares|"
    r"american depository shares|ads|adr)\b",
    re.IGNORECASE,
)


# ── Data types ────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    symbol:      str
    entry_date:  date
    entry_price: float
    exit_date:   Optional[date]  = None
    exit_price:  Optional[float] = None
    exit_reason: str             = ""
    entry_strategy: str          = ""
    qty:         float           = 0.0
    round_trip_commission: float = _DEFAULT_ROUND_TRIP_COST

    @property
    def gross_pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def net_pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return self.gross_pnl - self.round_trip_commission

    # Alias kept for backward compat with print_report / metrics
    @property
    def pnl(self) -> float:
        return self.net_pnl

    @property
    def pnl_pct(self) -> float:
        if self.exit_price is None or self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price


@dataclass
class BacktestResult:
    trades:       List[Trade]
    equity_curve: pd.Series
    metrics:      Dict
    filter_stats: Dict


# ── Core backtester ───────────────────────────────────────────────────────────
class VelocityBacktest:
    """
    Replay the full VelocityEngine production signal logic on historical
    daily OHLCV data downloaded via yfinance.

    Parameters
    ----------
    start           : backtest start date  (YYYY-MM-DD)
    end             : backtest end date    (YYYY-MM-DD)
    capital         : starting capital in USD
    max_pos         : safety cap for dynamic max simultaneous positions
    scan_count      : top-N from daily scanner considered per bar; 0 means all
    max_symbols     : download cap for bounded validation; 0/None means full filtered universe
    min_price       : minimum close price filter
    min_volume      : minimum daily share volume filter
    min_dollar_vol  : minimum 20-day avg dollar volume
    use_spy_filter  : if True, skip entries unless SPY close > SMA50 > SMA200 and SMA200 is rising
    use_vix_filter  : if True, skip new entries when VIX is missing or > VIX_THRESHOLD
    vix_delay_bars  : daily-bar proxy for delayed VIX data; 0=current bar,
                      1=prior available VIX bar (used for 15-minute delayed research)
    break_even_pct       : once profit exceeds this, floor the stop at entry (0.04 optimal)
    chandelier_mult      : ATR multiplier for trailing stop
    use_cache            : load/save downloaded data from backtest/.cache/
    """

    def __init__(
        self,
        start:          str   = "2025-01-01",
        end:            str   = "2026-05-01",
        capital:        float = BACKTEST_INITIAL_CAPITAL,
        max_pos:        int   = MAX_POSITIONS_CAP,
        scan_count:     int   = BACKTEST_SCAN_COUNT,
        min_price:      Optional[float] = None,
        min_volume:     Optional[float] = None,
        min_dollar_vol: Optional[float] = None,
        use_spy_filter: bool  = True,
        use_vix_filter: bool  = True,
        vix_delay_bars: int   = 0,
        break_even_pct:       float = BREAK_EVEN_PCT,
        chandelier_mult:      float = CHANDELIER_MULT,
        bear_phase_trading:   bool  = BEAR_PHASE_TRADING_ENABLED,
        commission_per_order: float = BACKTEST_COMMISSION_PER_ORDER,
        max_symbols:          int   = BACKTEST_MAX_SYMBOLS,
        use_cache:            bool  = True,
        scoring_model:        Optional[str] = None,
        strategy_profile:     str   = STRATEGY_PROFILE,
    ):
        self._profile              = get_strategy_profile(strategy_profile)
        self.start                 = start
        self.end                   = end
        self.capital               = capital
        self.max_pos               = max_pos
        self._scan_count           = scan_count
        self._min_price            = self._profile.min_price if min_price is None else min_price
        self._min_volume           = self._profile.min_volume if min_volume is None else min_volume
        self._min_dollar_vol       = self._profile.min_dollar_vol if min_dollar_vol is None else min_dollar_vol
        self._use_spy_filter       = use_spy_filter
        self._use_vix_filter       = use_vix_filter
        self._vix_delay_bars       = max(0, int(vix_delay_bars or 0))
        self._break_even_pct       = break_even_pct
        self._chandelier_mult      = chandelier_mult
        self._bear_phase_trading   = bear_phase_trading
        self._round_trip_cost      = max(0.0, float(commission_per_order)) * 2.0
        self._max_symbols          = max(0, int(max_symbols or 0))
        self._use_cache            = use_cache
        self._scoring_model        = (scoring_model or self._profile.scoring_model).strip().lower()
        self._analyst_provider     = AnalystRatingProvider(allow_remote=False)

        self._data:        Dict[str, pd.DataFrame] = {}
        self._vix_series:  Optional[pd.Series]     = None
        self._spy_bull:    Optional[pd.Series]      = None
        self._spy_return:  Optional[pd.Series]      = None
        self._spy_close:   Optional[pd.Series]      = None

        # Download starts early enough to warm up MA200 + chandelier ATR
        _trade_start     = date.fromisoformat(start)
        self._data_start = (_trade_start - timedelta(days=400)).isoformat()

        # Filter funnel accumulators (populated during run)
        self._filter_stats: Dict = {
            'scan_days':            0,
            'coarse_candidates':    0,   # pass price/vol/dollar-vol coarse scan
            'fine_signals':         0,   # pass the selected profile's _entry_signal
            'entries_taken':        0,   # actually opened a position
            'entries_skipped_full': 0,   # scanner candidate skipped because no entry slot/cash was available
            'spy_blocked_days':     0,   # trading days blocked by SPY filter
            'spy_bear_trade_days':  0,   # SPY-bear days allowed by bear-phase mode
            'bear_phase_signals':   0,
            'bull_phase_entries':   0,
            'bear_phase_entries':   0,
            'vix_blocked_days':     0,   # trading days blocked by VIX filter
            'total_commissions':    0.0,
        }

    # ── Universe discovery ────────────────────────────────────────────────────
    @staticmethod
    def _is_common_equity_listing(symbol: str, security_name: str) -> bool:
        """Approximate IBKR stockTypeFilter='CORP' for historical backtests.

        NASDAQ symbol files contain warrants, rights, units, preferred shares,
        notes, and funds. The live scanner excludes most of those at source; the
        backtest must do the same or results are polluted and downloads explode.
        """
        if pd.isna(symbol):
            return False
        symbol = str(symbol or "").strip()
        name = "" if pd.isna(security_name) else str(security_name or "").strip()
        if not symbol or len(symbol) > 5:
            return False
        if re.search(r"[\^+$\.]", symbol):
            return False

        if _NON_COMMON_NAME_RE.search(name):
            return False

        # If a symbol ends with common warrant/right/unit suffixes, only keep it
        # when the listing name explicitly says it is common/ordinary equity.
        if symbol[-1:] in {"W", "R", "U"} and not _COMMON_EQUITY_NAME_RE.search(name):
            return False

        return True

    @staticmethod
    def _fetch_universe() -> List[str]:
        """Fetch liquid-research universe aligned with the live corporate-stock scanner."""
        import io
        import urllib.request

        ua = {'User-Agent': 'Mozilla/5.0 (compatible; VelocityBacktest/1.0)'}

        def _get(url: str) -> bytes:
            req = urllib.request.Request(url, headers=ua)
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read()

        tickers: set = set()

        try:
            text  = _get(_NASDAQ_LISTED_URL).decode('utf-8')
            df_nq = pd.read_csv(__import__('io').StringIO(text), sep='|')
            df_nq = df_nq[
                (df_nq['ETF']               == 'N') &
                (df_nq['Test Issue']        == 'N') &
                (df_nq['Market Category'].isin(['Q', 'G']))
            ]
            for _, row in df_nq.iterrows():
                symbol = row.get('Symbol', '')
                name = row.get('Security Name', '')
                if VelocityBacktest._is_common_equity_listing(symbol, name):
                    tickers.add(str(symbol).strip().upper())
        except Exception as e:
            raise RuntimeError(f"Failed to fetch NASDAQ listing: {e}")

        try:
            text   = _get(_OTHER_LISTED_URL).decode('utf-8')
            df_oth = pd.read_csv(__import__('io').StringIO(text), sep='|')
            df_oth = df_oth[
                (df_oth['ETF']             == 'N') &
                (df_oth['Test Issue']      == 'N') &
                (df_oth['Exchange'].isin(['N', 'A']))
            ]
            for _, row in df_oth.iterrows():
                symbol = row.get('ACT Symbol', '')
                name = row.get('Security Name', '')
                if VelocityBacktest._is_common_equity_listing(symbol, name):
                    tickers.add(str(symbol).strip().upper())
        except Exception as e:
            raise RuntimeError(f"Failed to fetch other-exchange listing: {e}")

        return sorted(tickers)

    # ── Cache helpers ─────────────────────────────────────────────────────────
    def _cache_path(self) -> str:
        key = (
            f"{_CACHE_VERSION}_{self._data_start}_{self.end}_"
            f"profile{self._profile.name}_"
            f"p{self._min_price:g}_v{int(self._min_volume)}_"
            f"dv{int(self._min_dollar_vol/1e6)}_max{self._max_symbols}"
        )
        h   = hashlib.md5(key.encode()).hexdigest()[:10]
        os.makedirs(_CACHE_DIR, exist_ok=True)
        return os.path.join(_CACHE_DIR, f"bt_{h}.pkl")

    def _regime_cache_path(self) -> str:
        key = (
            f"regime_{self._data_start}_{self.start}_{self.end}_"
            f"spy{int(self._use_spy_filter)}_vix{int(self._use_vix_filter)}"
        )
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        os.makedirs(_CACHE_DIR, exist_ok=True)
        return os.path.join(_CACHE_DIR, f"regime_{h}.pkl")

    def _cache_metadata(self) -> Dict:
        return {
            'version': _CACHE_VERSION,
            'data_start': self._data_start,
            'end': self.end,
            'strategy_profile': self._profile.name,
            'min_price': self._min_price,
            'min_volume': self._min_volume,
            'min_dollar_vol': self._min_dollar_vol,
            'max_symbols': self._max_symbols,
        }

    def _cache_data_covers_request(self, data: Dict[str, pd.DataFrame]) -> bool:
        if not data:
            return False
        requested_start = pd.Timestamp(self._data_start)
        requested_end = pd.Timestamp(self.end)
        min_seen = None
        max_seen = None
        for df in data.values():
            if df is None or df.empty:
                continue
            idx = df.index
            cur_min = pd.Timestamp(idx.min())
            cur_max = pd.Timestamp(idx.max())
            min_seen = cur_min if min_seen is None else min(min_seen, cur_min)
            max_seen = cur_max if max_seen is None else max(max_seen, cur_max)
        if min_seen is None or max_seen is None:
            return False
        # yfinance end dates are exclusive and weekends/holidays shift the last
        # available bar, so allow a short grace window at the right edge.
        return (
            min_seen <= requested_start + pd.Timedelta(days=5)
            and max_seen >= requested_end - pd.Timedelta(days=7)
        )

    def _repair_cached_data_columns(self) -> Tuple[int, int]:
        repaired = 0
        dropped = 0
        for sym, df in list(self._data.items()):
            if df is None or df.empty or not _CACHE_BASE_COLUMNS.issubset(df.columns):
                self._data.pop(sym, None)
                dropped += 1
                continue
            if _CACHE_REQUIRED_COLUMNS.issubset(df.columns):
                continue

            fixed = df.copy()
            fixed = apply_all(
                fixed, RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW,
                SMA200_SLOPE_LOOKBACK, CHANDELIER_PERIOD
            )

            if "MA10" not in fixed.columns:
                fixed["MA10"] = fixed["close"].rolling(10).mean()
            if "MA20" not in fixed.columns:
                fixed["MA20"] = fixed["close"].rolling(20).mean()
            if "prev_close" not in fixed.columns:
                fixed["prev_close"] = fixed["close"].shift(1)
            if "prev_high" not in fixed.columns:
                fixed["prev_high"] = fixed["high"].shift(1)
            if "high20" not in fixed.columns:
                fixed["high20"] = fixed["high"].rolling(20).max().shift(1)
            if "high50" not in fixed.columns:
                fixed["high50"] = fixed["high"].rolling(50).max().shift(1)
            if "dist_high20" not in fixed.columns:
                fixed["dist_high20"] = fixed["close"] / fixed["high20"] - 1
            if "dist_high50" not in fixed.columns:
                fixed["dist_high50"] = fixed["close"] / fixed["high50"] - 1
            if "atr_pct" not in fixed.columns:
                fixed["atr_pct"] = fixed["ATR_CHAND"] / fixed["close"]
            if "MACD_HIST" not in fixed.columns or "MACD_HIST_DELTA" not in fixed.columns:
                ema12 = fixed["close"].ewm(span=12, adjust=False).mean()
                ema26 = fixed["close"].ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                macd_signal = macd.ewm(span=9, adjust=False).mean()
                fixed["MACD_HIST"] = macd - macd_signal
                fixed["MACD_HIST_DELTA"] = fixed["MACD_HIST"] - fixed["MACD_HIST"].shift(1)
            if "OBV" not in fixed.columns or "OBV_SLOPE_5" not in fixed.columns:
                signed_volume = np.where(fixed["close"].diff() >= 0, fixed["volume"], -fixed["volume"])
                fixed["OBV"] = pd.Series(signed_volume, index=fixed.index).cumsum()
                fixed["OBV_SLOPE_5"] = fixed["OBV"] - fixed["OBV"].shift(5)
            if "reclaim_ma20" not in fixed.columns:
                fixed["reclaim_ma20"] = (fixed["prev_close"] <= fixed["MA20"].shift(1)) & (fixed["close"] > fixed["MA20"])
            if "reclaim_ma50" not in fixed.columns:
                fixed["reclaim_ma50"] = (fixed["prev_close"] <= fixed["MA50"].shift(1)) & (fixed["close"] > fixed["MA50"])
            if "break_prev_high" not in fixed.columns:
                fixed["break_prev_high"] = fixed["close"] > fixed["prev_high"]
            if "avg_vol_20" not in fixed.columns:
                fixed["avg_vol_20"] = fixed["volume"].rolling(20).mean()
            if "avg_dollar_vol_20" not in fixed.columns:
                fixed["avg_dollar_vol_20"] = (fixed["close"] * fixed["volume"]).rolling(20).mean()
            if "return_13w" not in fixed.columns:
                fixed["return_13w"] = fixed["close"] / fixed["close"].shift(63) - 1
            if "return_26w" not in fixed.columns:
                fixed["return_26w"] = fixed["close"] / fixed["close"].shift(126) - 1
            if "high_52w" not in fixed.columns:
                fixed["high_52w"] = fixed["high"].rolling(252).max().shift(1)
            if "price_vs_52w_high" not in fixed.columns:
                fixed["price_vs_52w_high"] = fixed["close"] / fixed["high_52w"]
            if "weekly_uptrend" not in fixed.columns:
                weekly_close = fixed["close"].resample("W-FRI").last()
                weekly_ma10 = weekly_close.rolling(10).mean()
                weekly_ma30 = weekly_close.rolling(30).mean()
                weekly_up = (
                    (weekly_close > weekly_ma10)
                    & (weekly_ma10 > weekly_ma30)
                    & (weekly_close / weekly_close.shift(13) - 1 > 0)
                )
                fixed["weekly_uptrend"] = weekly_up.reindex(fixed.index, method="ffill").fillna(False)
            if "prev_ATR5" not in fixed.columns:
                fixed["prev_ATR5"] = fixed["ATR5"].shift(1)
            if "prev_ATR20" not in fixed.columns:
                fixed["prev_ATR20"] = fixed["ATR20"].shift(1)
            if "prev_HIGH10" not in fixed.columns:
                fixed["prev_HIGH10"] = fixed["HIGH10"].shift(1)

            if not _CACHE_REQUIRED_COLUMNS.issubset(fixed.columns):
                self._data.pop(sym, None)
                dropped += 1
                continue
            self._data[sym] = fixed
            repaired += 1
        return repaired, dropped

    def _cache_meta_matches_request(self, meta: Dict, data: Dict[str, pd.DataFrame]) -> bool:
        if not meta:
            # Legacy cache payloads did not store metadata. Reuse them only for
            # bounded symbol-cap runs where the cache visibly covers the period.
            return self._max_symbols > 0 and len(data) <= self._max_symbols
        try:
            max_symbols = int(meta.get('max_symbols', 0))
            symbol_cap_ok = (
                max_symbols == int(self._max_symbols)
                or (self._max_symbols > 0 and len(data) <= self._max_symbols)
            )
            return (
                str(meta.get('version')) in _CACHE_COMPATIBLE_VERSIONS
                and symbol_cap_ok
                and pd.Timestamp(str(meta.get('data_start'))) <= pd.Timestamp(self._data_start)
                and pd.Timestamp(str(meta.get('end'))) >= pd.Timestamp(self.end)
            )
        except Exception:
            return False

    def _load_cache_payload(self, path: str, *, exact: bool) -> bool:
        try:
            print(f"  Loading cached data from {path} …")
            with open(path, 'rb') as f:
                cached = pickle.load(f)
            data = cached.get('data', {}) if isinstance(cached, dict) else {}
            meta = cached.get('meta', {}) if isinstance(cached, dict) else {}
            if not exact:
                if not self._cache_data_covers_request(data):
                    return False
                if not self._cache_meta_matches_request(meta, data):
                    return False
            self._data = data
            repaired, dropped = self._repair_cached_data_columns()
            if not self._data:
                return False
            print(f"  Loaded {len(self._data):,} symbols from cache.")
            if repaired:
                print(f"  Repaired {repaired:,} cached symbols with current indicator columns.")
            if dropped:
                print(f"  Dropped {dropped:,} incompatible cached symbols.")
            return True
        except Exception as e:
            if exact:
                print(f"  Cache load failed ({e}), re-downloading …")
            return False

    def _try_load_cache(self) -> bool:
        path = self._cache_path()
        if os.path.exists(path) and self._load_cache_payload(path, exact=True):
            return True

        if not os.path.isdir(_CACHE_DIR):
            return False

        for name in sorted(os.listdir(_CACHE_DIR), reverse=True):
            if not name.startswith("bt_") or not name.endswith(".pkl"):
                continue
            candidate = os.path.join(_CACHE_DIR, name)
            if candidate == path:
                continue
            if self._load_cache_payload(candidate, exact=False):
                print("  Reused compatible superset cache for requested date window.")
                return True
        return False

    def _save_cache(self) -> None:
        path = self._cache_path()
        try:
            with open(path, 'wb') as f:
                pickle.dump({'meta': self._cache_metadata(), 'data': self._data}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Data cached → {path}")
        except Exception as e:
            print(f"  Cache save failed: {e}")

    def _regime_requirements_met(self) -> bool:
        return (
            self._spy_return is not None
            and self._spy_close is not None
            and (not self._use_spy_filter or self._spy_bull is not None)
            and (not self._use_vix_filter or self._vix_series is not None)
        )

    def _regime_cache_metadata(self) -> Dict:
        return {
            'data_start': self._data_start,
            'start': self.start,
            'end': self.end,
            'use_spy_filter': self._use_spy_filter,
            'use_vix_filter': self._use_vix_filter,
        }

    @staticmethod
    def _series_covers_window(series: Optional[pd.Series], start: str, end: str) -> bool:
        if series is None or series.empty:
            return False
        min_seen = pd.Timestamp(series.index.min())
        max_seen = pd.Timestamp(series.index.max())
        return (
            min_seen <= pd.Timestamp(start) + pd.Timedelta(days=5)
            and max_seen >= pd.Timestamp(end) - pd.Timedelta(days=7)
        )

    def _regime_payload_covers_request(self, cached: Dict) -> bool:
        if not self._series_covers_window(
            cached.get('spy_return'), self._data_start, self.end
        ):
            return False
        if not self._series_covers_window(
            cached.get('spy_close'), self._data_start, self.end
        ):
            return False
        if self._use_spy_filter and not self._series_covers_window(
            cached.get('spy_bull'), self._data_start, self.end
        ):
            return False
        if self._use_vix_filter:
            if not self._series_covers_window(
                cached.get('vix_series'), self.start, self.end
            ):
                return False
        return True

    def _regime_meta_matches_request(self, meta: Dict) -> bool:
        if not meta:
            return True
        try:
            return (
                bool(meta.get('use_spy_filter')) >= bool(self._use_spy_filter)
                and bool(meta.get('use_vix_filter')) >= bool(self._use_vix_filter)
                and pd.Timestamp(str(meta.get('data_start'))) <= pd.Timestamp(self._data_start)
                and pd.Timestamp(str(meta.get('start'))) <= pd.Timestamp(self.start)
                and pd.Timestamp(str(meta.get('end'))) >= pd.Timestamp(self.end)
            )
        except Exception:
            return False

    def _load_regime_cache_payload(self, path: str, *, exact: bool) -> bool:
        try:
            with open(path, 'rb') as f:
                cached = pickle.load(f)
            if not isinstance(cached, dict):
                return False
            meta = cached.get('meta', {})
            if not exact:
                if not self._regime_meta_matches_request(meta):
                    return False
                if not self._regime_payload_covers_request(cached):
                    return False
            self._spy_return = cached.get('spy_return')
            self._spy_close = cached.get('spy_close')
            if self._use_spy_filter:
                self._spy_bull = cached.get('spy_bull')
            if self._use_vix_filter:
                self._vix_series = cached.get('vix_series')
            if self._regime_requirements_met():
                print(f"  Loaded regime data from {path}.")
                return True
        except Exception as e:
            if exact:
                print(f"  Regime cache load failed ({e}), re-downloading …")
        return False

    def _try_load_regime_cache(self) -> bool:
        path = self._regime_cache_path()
        if os.path.exists(path) and self._load_regime_cache_payload(path, exact=True):
            return True

        if not os.path.isdir(_CACHE_DIR):
            return False

        for name in sorted(os.listdir(_CACHE_DIR), reverse=True):
            if not name.startswith("regime_") or not name.endswith(".pkl"):
                continue
            candidate = os.path.join(_CACHE_DIR, name)
            if candidate == path:
                continue
            if self._load_regime_cache_payload(candidate, exact=False):
                print("  Reused compatible superset regime cache for requested date window.")
                return True
        return False

    def _save_regime_cache(self) -> None:
        path = self._regime_cache_path()
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'meta': self._regime_cache_metadata(),
                    'spy_bull': self._spy_bull,
                    'spy_return': self._spy_return,
                    'spy_close': self._spy_close,
                    'vix_series': self._vix_series,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"  Regime cache save failed: {e}")

    @staticmethod
    def _download_stooq_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
        api_key = os.getenv("STOOQ_APIKEY", "").strip()
        if not api_key:
            raise RuntimeError("STOOQ_APIKEY is not configured.")
        d1 = pd.Timestamp(start).strftime("%Y%m%d")
        d2 = pd.Timestamp(end).strftime("%Y%m%d")
        encoded = urllib.parse.quote(symbol.lower())
        url = (
            f"https://stooq.com/q/d/l/?s={encoded}&d1={d1}&d2={d2}"
            f"&i=d&apikey={urllib.parse.quote(api_key)}"
        )
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; VelocityBacktest/1.0)'},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
        df = pd.read_csv(io.BytesIO(raw))
        if df.empty or "Date" not in df.columns:
            raise RuntimeError(f"Stooq returned no usable rows for {symbol}.")
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()

    @staticmethod
    def _download_yahoo_chart_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
        period1 = int(pd.Timestamp(start).timestamp())
        period2 = int(pd.Timestamp(end).timestamp())
        encoded = urllib.parse.quote(symbol)
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
            f"?period1={period1}&period2={period2}&interval=1d"
            "&events=history&includeAdjustedClose=true"
        )
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; VelocityBacktest/1.0)'},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            payload = json.loads(r.read().decode("utf-8"))

        chart = payload.get("chart", {})
        if chart.get("error"):
            raise RuntimeError(chart["error"])
        results = chart.get("result") or []
        if not results:
            raise RuntimeError(f"Yahoo chart returned no results for {symbol}.")

        result = results[0]
        timestamps = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        if not timestamps or "close" not in quote:
            raise RuntimeError(f"Yahoo chart returned no usable bars for {symbol}.")

        df = pd.DataFrame({
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }, index=pd.to_datetime(timestamps, unit="s").tz_localize("UTC").tz_convert("America/New_York").tz_localize(None).normalize())

        adj = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
        if adj:
            raw_close = pd.Series(quote.get("close"), index=df.index, dtype=float)
            adj_close = pd.Series(adj, index=df.index, dtype=float)
            ratio = adj_close / raw_close.replace(0, np.nan)
            for col in ("open", "high", "low", "close"):
                df[col] = pd.Series(df[col], index=df.index, dtype=float) * ratio
        return df.dropna(subset=["close"]).sort_index()

    # ── Data download ─────────────────────────────────────────────────────────
    def _download(self) -> None:
        """
        Download and indicator-enrich daily OHLCV for all institutional-grade
        US-listed equities.
        """
        tickers = self._fetch_universe()
        stable_seed = int(hashlib.md5(self.start.encode()).hexdigest()[:8], 16)
        rng = random.Random(stable_seed)
        rng.shuffle(tickers)
        if self._max_symbols > 0:
            tickers = tickers[:self._max_symbols]
        print(
            f"  Universe : {len(tickers):,} US equities "
            f"(NASDAQ Global Select/Market + NYSE/NYSE American common equities)\n"
            f"  Profile  : {self._profile.name} ({self._profile.label})\n"
            f"  Filters  : price>${self._min_price:.0f}  |  "
            f"vol>{self._min_volume/1e6:.0f}M shares  |  "
            f"20d avg dollar-vol>${self._min_dollar_vol/1e6:.0f}M  |  "
            f"top-{self._scan_count} by RVOL-weighted momentum"
        )
        print(f"  Downloading {len(tickers):,} tickers (from {self._data_start}) …")

        try:
            raw = yf.download(
                tickers,
                start=self._data_start,
                end=self.end,
                auto_adjust=True,
                progress=True,
                group_by='ticker',
                threads=True,
            )
        except Exception as e:
            raise RuntimeError(f"Data download failed: {e}")

        loaded = 0
        single = len(tickers) == 1
        for sym in tickers:
            try:
                df = raw.copy() if single else raw[sym].copy()
                df.columns = df.columns.str.lower()
                df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
                if len(df) < MIN_CANDLES + 5:
                    continue
                df = apply_all(
                    df, RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW,
                    SMA200_SLOPE_LOOKBACK, CHANDELIER_PERIOD
                )
                df['MA10']              = df['close'].rolling(10).mean()
                df['MA20']              = df['close'].rolling(20).mean()
                df['prev_close']        = df['close'].shift(1)
                df['prev_high']         = df['high'].shift(1)
                df['high20']            = df['high'].rolling(20).max().shift(1)
                df['high50']            = df['high'].rolling(50).max().shift(1)
                df['dist_high20']       = df['close'] / df['high20'] - 1
                df['dist_high50']       = df['close'] / df['high50'] - 1
                df['atr_pct']           = df['ATR_CHAND'] / df['close']
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                macd_signal = macd.ewm(span=9, adjust=False).mean()
                df['MACD_HIST']         = macd - macd_signal
                df['MACD_HIST_DELTA']   = df['MACD_HIST'] - df['MACD_HIST'].shift(1)
                signed_volume = np.where(df['close'].diff() >= 0, df['volume'], -df['volume'])
                df['OBV']               = pd.Series(signed_volume, index=df.index).cumsum()
                df['OBV_SLOPE_5']       = df['OBV'] - df['OBV'].shift(5)
                prev_ma20 = df['MA20'].shift(1)
                prev_ma50 = df['MA50'].shift(1)
                df['reclaim_ma20']      = (df['prev_close'] <= prev_ma20) & (df['close'] > df['MA20'])
                df['reclaim_ma50']      = (df['prev_close'] <= prev_ma50) & (df['close'] > df['MA50'])
                df['break_prev_high']   = df['close'] > df['prev_high']
                df['avg_vol_20']        = df['volume'].rolling(20).mean()
                df['avg_dollar_vol_20'] = (
                    (df['close'] * df['volume']).rolling(20).mean()
                )
                df['return_13w']        = df['close'] / df['close'].shift(63) - 1
                df['return_26w']        = df['close'] / df['close'].shift(126) - 1
                df['high_52w']          = df['high'].rolling(252).max().shift(1)
                df['price_vs_52w_high'] = df['close'] / df['high_52w']
                weekly_close = df['close'].resample('W-FRI').last()
                weekly_ma10 = weekly_close.rolling(10).mean()
                weekly_ma30 = weekly_close.rolling(30).mean()
                weekly_up = (
                    (weekly_close > weekly_ma10)
                    & (weekly_ma10 > weekly_ma30)
                    & (weekly_close / weekly_close.shift(13) - 1 > 0)
                )
                df['weekly_uptrend'] = weekly_up.reindex(df.index, method='ffill').fillna(False)
                # Previous-bar diagnostic values kept for compatibility with
                # older research reports. The default swing profile no longer
                # gates entries on VCP or 10-day-high proximity.
                df['prev_ATR5']   = df['ATR5'].shift(1)
                df['prev_ATR20']  = df['ATR20'].shift(1)
                df['prev_HIGH10'] = df['HIGH10'].shift(1)
                self._data[sym] = df
                loaded += 1
            except Exception:
                continue

        print(f"  Loaded   : {loaded:,} symbols with ≥{MIN_CANDLES + 5} bars.")
        self._download_regime_data()

    def _download_regime_data(self) -> None:
        """
        Download or load SPY and VIX regime data.

        Enabled regime filters are mandatory: if the data provider fails, the
        backtest stops instead of silently producing an unfiltered result. The
        cache exists to keep validation reproducible and avoid provider rate
        limits on repeated local research runs.
        """
        if self._use_cache and self._try_load_regime_cache():
            return

        if self._use_vix_filter:
            try:
                vix_raw = yf.download('^VIX', start=self.start, end=self.end,
                                      auto_adjust=True, progress=False)
                if vix_raw.empty:
                    raise RuntimeError("VIX yfinance download returned no rows.")
            except Exception as e:
                try:
                    vix_raw = self._download_yahoo_chart_daily("^VIX", self.start, self.end)
                    print("  VIX yfinance unavailable; loaded regime data from Yahoo chart.")
                except Exception as yahoo_e:
                    try:
                        vix_raw = self._download_stooq_daily("^vix", self.start, self.end)
                        print("  VIX yfinance unavailable; loaded regime data from Stooq.")
                    except Exception as stooq_e:
                        raise RuntimeError(
                            f"VIX regime data download failed: {e}; "
                            f"Yahoo chart fallback failed: {yahoo_e}; "
                            f"Stooq fallback failed: {stooq_e}"
                        ) from stooq_e
            vix_raw.columns = [
                c[0].lower() if isinstance(c, tuple) else c.lower()
                for c in vix_raw.columns
            ]
            self._vix_series = vix_raw['close']

        try:
            spy_raw = yf.download('SPY', start=self._data_start, end=self.end,
                                  auto_adjust=True, progress=False)
            if spy_raw.empty:
                raise RuntimeError("SPY yfinance download returned no rows.")
        except Exception as e:
            try:
                spy_raw = self._download_yahoo_chart_daily("SPY", self._data_start, self.end)
                print("  SPY yfinance unavailable; loaded regime data from Yahoo chart.")
            except Exception as yahoo_e:
                try:
                    spy_raw = self._download_stooq_daily("spy.us", self._data_start, self.end)
                    print("  SPY yfinance unavailable; loaded regime data from Stooq.")
                except Exception as stooq_e:
                    raise RuntimeError(
                        f"SPY regime data download failed: {e}; "
                        f"Yahoo chart fallback failed: {yahoo_e}; "
                        f"Stooq fallback failed: {stooq_e}"
                    ) from stooq_e
        spy_raw.columns = [
            c[0].lower() if isinstance(c, tuple) else c.lower()
            for c in spy_raw.columns
        ]
        sc    = spy_raw['close']
        so    = spy_raw['open']
        self._spy_close = sc
        self._spy_return = (sc - so) / so
        ma50  = sc.rolling(50).mean()
        ma200 = sc.rolling(200).mean()
        # Require price > MA50 > MA200 and a rising SMA200. This blocks
        # low-quality recovery rallies while the long-term trend is still
        # falling.
        # This blocks entries during corrections AND recovery — the
        # recovery phase from a deep correction produces many false
        # breakouts before the trend is genuinely re-established.
        ma200_slope = ma200 - ma200.shift(SMA200_SLOPE_LOOKBACK)
        self._spy_bull = (sc > ma50) & (ma50 > ma200) & (ma200_slope > 0)

        if self._use_cache and self._regime_requirements_met():
            self._save_regime_cache()

    def _validate_regime_data(self) -> None:
        if self._spy_return is None:
            raise RuntimeError(
                "SPY daily return data is unavailable; EOD relative-strength cleanup cannot be tested."
            )
        if self._use_spy_filter and self._spy_bull is None:
            raise RuntimeError(
                "SPY regime filter is enabled, but SPY regime data is unavailable."
            )
        if self._use_vix_filter and self._vix_series is None:
            raise RuntimeError(
                "VIX regime filter is enabled, but VIX regime data is unavailable."
            )

    def _vix_value_for_date(self, today) -> float:
        """Return the VIX value available to the strategy for a given daily bar.

        Daily data cannot faithfully represent a 15-minute intraday delay.  When
        vix_delay_bars > 0, use the prior available daily VIX bar as a strict
        no-look-ahead approximation for delayed VIX research.
        """
        if self._vix_series is None:
            return np.nan
        if self._vix_delay_bars <= 0:
            return self._vix_series.get(today, np.nan)
        try:
            loc = self._vix_series.index.get_loc(today)
        except KeyError:
            return np.nan
        if isinstance(loc, slice):
            loc = loc.start
        elif isinstance(loc, np.ndarray):
            locs = np.flatnonzero(loc)
            loc = int(locs[0]) if len(locs) else -1
        src_idx = int(loc) - self._vix_delay_bars
        if src_idx < 0:
            return np.nan
        return self._vix_series.iloc[src_idx]

    @staticmethod
    def _series_return(series: Optional[pd.Series], today, bars_back: int) -> float:
        if series is None or today not in series.index:
            return np.nan
        try:
            loc = series.index.get_loc(today)
            if isinstance(loc, slice):
                loc = loc.start
            elif isinstance(loc, np.ndarray):
                locs = np.flatnonzero(loc)
                loc = int(locs[0]) if len(locs) else -1
            src = int(loc) - int(bars_back)
            if src < 0:
                return np.nan
            cur = float(series.iloc[int(loc)])
            ref = float(series.iloc[src])
        except Exception:
            return np.nan
        return cur / ref - 1 if ref > 0 else np.nan

    def _analyst_context(self, symbol: str, today) -> dict:
        return self._analyst_provider.get(symbol, as_of=today).as_context()

    # ── Daily scanner simulation ──────────────────────────────────────────────
    def _daily_scan(
        self,
        today,
        min_dollar_vol: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Simulate the broad IB active-stock scanner with production pre-filters.
        Returns list of (symbol, rvol) tuples, sorted by the same shared
        candidate scorer used by live trading.  scan_count <= 0 means every
        scanner-passed stock is returned.
        Fine signal rules are applied in _entry_signal.
        """
        scored: List[tuple] = []
        dollar_vol_floor = self._min_dollar_vol if min_dollar_vol is None else min_dollar_vol

        for sym, df in self._data.items():
            if today not in df.index:
                continue
            idx = df.index.get_loc(today)
            if idx < 1:
                continue

            row      = df.loc[today]
            prev_row = df.iloc[idx - 1]

            # Price and volume floor (mirrors IB scanner parameters)
            if row['close'] < self._min_price:
                continue
            if row['volume'] < self._min_volume:
                continue

            # Dollar-volume gate (20-day average)
            avg_dvol = row.get('avg_dollar_vol_20', row['close'] * row['volume'])
            if pd.isna(avg_dvol) or avg_dvol < dollar_vol_floor:
                continue

            # RVOL ranking input (not a default swing entry gate)
            avg_vol = row.get('avg_vol_20', 0)
            if pd.isna(avg_vol) or avg_vol <= 0:
                continue
            rvol = row['volume'] / avg_vol

            self._filter_stats['coarse_candidates'] += 1
            spy_ret_63d = self._series_return(self._spy_close, today, 63)
            spy_ret_126d = self._series_return(self._spy_close, today, 126)
            try:
                return_13w = float(row.get('return_13w', np.nan))
            except (TypeError, ValueError):
                return_13w = np.nan
            try:
                return_26w = float(row.get('return_26w', np.nan))
            except (TypeError, ValueError):
                return_26w = np.nan

            # Daily bars are complete sessions, so full-day RVOL is already the
            # daily equivalent of live time-normalized volume pace.
            ctx = {
                'ma50':           row.get('MA50'),
                'ma200':          row.get('MA200'),
                'ma20':           row.get('MA20'),
                'sma200_slope':   row.get('SMA200_SLOPE'),
                'rsi':            row.get('RSI'),
                'rsi_prev':       prev_row.get('RSI'),
                'rvol':           rvol,
                'rvol_raw':       rvol,
                'volume_pace':    rvol,
                'spread_pct':     0.0,
                'live_price':     row.get('close'),
                'close':          row.get('close'),
                'day_open':       row.get('open'),
                'orb_high':       row.get('prev_high'),
                'prev_high':      row.get('prev_high'),
                'prev_daily_high': row.get('prev_high'),
                'high20':         row.get('high20'),
                'dist_high20':    row.get('dist_high20'),
                'day_range_location': row.get('CLV'),
                'intraday_gain':  (
                    (row.get('close') - row.get('open')) / row.get('open')
                    if row.get('open') and row.get('open') > 0 else np.nan
                ),
                'atr':            row.get('ATR'),
                'ATR_CHAND':      row.get('ATR_CHAND'),
                'atr_chandelier': row.get('ATR_CHAND', row.get('ATR')),
                'atr_pct':        row.get('atr_pct'),
                'macd_hist':      row.get('MACD_HIST'),
                'macd_hist_delta': row.get('MACD_HIST_DELTA'),
                'macd_bull_divergence': bool(row.get('MACD_BULL_DIVERGENCE', False)),
                'macd_bear_divergence': bool(row.get('MACD_BEAR_DIVERGENCE', False)),
                'obv_slope_5':    row.get('OBV_SLOPE_5'),
                'obv_uptrend':     bool(row.get('OBV_UPTREND', False)),
                'obv_bull_divergence': bool(row.get('OBV_BULL_DIVERGENCE', False)),
                'obv_bear_divergence': bool(row.get('OBV_BEAR_DIVERGENCE', False)),
                'ema20_gt_sma50':  bool(row.get('EMA20_GT_SMA50', False)),
                'ma_bull_cross':   bool(row.get('MA_BULL_CROSS', False)),
                'ma_bear_cross':   bool(row.get('MA_BEAR_CROSS', False)),
                'bb_below_lower_2': bool(row.get('BB_BELOW_LOWER_2', False)),
                'bb_above_upper_2': bool(row.get('BB_ABOVE_UPPER_2', False)),
                'bb_reclaim_lower': bool(row.get('BB_RECLAIM_LOWER', False)),
                'psar_bull_3':     bool(row.get('PSAR_BULL_3', False)),
                'psar_bear_3':     bool(row.get('PSAR_BEAR_3', False)),
                'stoch_k':         row.get('STOCH_K'),
                'stoch_d':         row.get('STOCH_D'),
                'stoch_bull_exit_oversold': bool(row.get('STOCH_BULL_EXIT_OVERSOLD', False)),
                'stoch_bear_exit_overbought': bool(row.get('STOCH_BEAR_EXIT_OVERBOUGHT', False)),
                'reclaim_ma20':   bool(row.get('reclaim_ma20', False)),
                'reclaim_ma50':   bool(row.get('reclaim_ma50', False)),
                'break_prev_high': bool(row.get('break_prev_high', False)),
                'weekly_uptrend':  bool(row.get('weekly_uptrend', False)),
                'return_13w':      return_13w,
                'return_26w':      return_26w,
                'relative_strength_63d': (
                    float(return_13w) - spy_ret_63d
                    if np.isfinite(return_13w) and np.isfinite(spy_ret_63d) else np.nan
                ),
                'relative_strength_126d': (
                    float(return_26w) - spy_ret_126d
                    if np.isfinite(return_26w) and np.isfinite(spy_ret_126d) else np.nan
                ),
                'price_vs_52w_high': row.get('price_vs_52w_high'),
                'volume':         row.get('volume'),
                'dollar_vol_20d': row.get('avg_dollar_vol_20'),
            }
            ctx.update(self._analyst_context(sym, today))
            entry_strategy = select_entry_strategy(ctx, self._profile)
            if entry_strategy:
                ctx['entry_strategy'] = entry_strategy
                ctx['entry_strategy_label'] = indicator_sleeve_label(entry_strategy)
            score = score_candidate(
                ctx,
                model=self._scoring_model,
                volume_floor=self._profile.min_volume_pace or 1.0,
                spread_max_pct=SPREAD_MAX_PCT,
                atr_pct_max=self._profile.max_atr_pct or ATR_PCT_MAX,
            )
            if self._profile.min_score is not None and score < float(self._profile.min_score):
                continue
            scored.append((sym, score, rvol))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored if self._scan_count <= 0 else scored[:self._scan_count]
        return [(sym, rvol) for sym, _, rvol in selected]

    # ── Signal check ─────────────────────────────────────────────────────────
    @staticmethod
    def _entry_signal(row: pd.Series, prev_rsi: float, rvol: float,
                      strategy_profile: str = STRATEGY_PROFILE) -> bool:
        """Profile-aware fine entry filter (daily-bar approximation)."""
        profile = get_strategy_profile(strategy_profile)
        open_price = float(row.get('open', np.nan))
        close_price = float(row.get('close', np.nan))
        intraday_gain = (
            (close_price - open_price) / open_price
            if np.isfinite(open_price) and open_price > 0 else np.nan
        )
        ctx = {
            'live_price': close_price,
            'close': close_price,
            'day_open': open_price,
            'orb_high': row.get('prev_high'),
            'prev_high': row.get('prev_high'),
            'prev_daily_high': row.get('prev_high'),
            'ma20': row.get('MA20'),
            'ma50': row.get('MA50'),
            'ma200': row.get('MA200'),
            'sma200_slope': row.get('SMA200_SLOPE'),
            'rsi': row.get('RSI'),
            'rsi_prev': prev_rsi,
            'rvol': rvol,
            'volume_pace': rvol,
            'spread_pct': 0.0,
            'day_range_location': row.get('CLV'),
            'intraday_gain': intraday_gain,
            'atr': row.get('ATR'),
            'atr_chandelier': row.get('ATR_CHAND', row.get('ATR', np.nan)),
            'atr_pct': row.get('atr_pct', (
                row.get('ATR_CHAND', row.get('ATR', np.nan)) / close_price
                if np.isfinite(close_price) and close_price > 0 else np.nan
            )),
            'high20': row.get('high20'),
            'dist_high20': row.get('dist_high20'),
            'macd_hist': row.get('MACD_HIST'),
            'macd_hist_delta': row.get('MACD_HIST_DELTA'),
            'macd_bull_divergence': bool(row.get('MACD_BULL_DIVERGENCE', False)),
            'macd_bear_divergence': bool(row.get('MACD_BEAR_DIVERGENCE', False)),
            'obv_slope_5': row.get('OBV_SLOPE_5'),
            'obv_uptrend': bool(row.get('OBV_UPTREND', False)),
            'obv_bull_divergence': bool(row.get('OBV_BULL_DIVERGENCE', False)),
            'obv_bear_divergence': bool(row.get('OBV_BEAR_DIVERGENCE', False)),
            'ema20_gt_sma50': bool(row.get('EMA20_GT_SMA50', False)),
            'ma_bull_cross': bool(row.get('MA_BULL_CROSS', False)),
            'ma_bear_cross': bool(row.get('MA_BEAR_CROSS', False)),
            'bb_below_lower_2': bool(row.get('BB_BELOW_LOWER_2', False)),
            'bb_above_upper_2': bool(row.get('BB_ABOVE_UPPER_2', False)),
            'bb_reclaim_lower': bool(row.get('BB_RECLAIM_LOWER', False)),
            'psar_bull_3': bool(row.get('PSAR_BULL_3', False)),
            'psar_bear_3': bool(row.get('PSAR_BEAR_3', False)),
            'stoch_k': row.get('STOCH_K'),
            'stoch_d': row.get('STOCH_D'),
            'stoch_bull_exit_oversold': bool(row.get('STOCH_BULL_EXIT_OVERSOLD', False)),
            'stoch_bear_exit_overbought': bool(row.get('STOCH_BEAR_EXIT_OVERBOUGHT', False)),
            'reclaim_ma20': bool(row.get('reclaim_ma20', False)),
            'reclaim_ma50': bool(row.get('reclaim_ma50', False)),
            'break_prev_high': bool(row.get('break_prev_high', False)),
            'weekly_uptrend': bool(row.get('weekly_uptrend', False)),
            'return_13w': row.get('return_13w'),
            'return_26w': row.get('return_26w'),
            'relative_strength_63d': row.get('relative_strength_63d'),
            'relative_strength_126d': row.get('relative_strength_126d'),
            'price_vs_52w_high': row.get('price_vs_52w_high'),
            'volume': row.get('volume', 0),
            'dollar_vol_20d': row.get('avg_dollar_vol_20', 0),
        }
        overrides = {
            'min_volume': 0.0,
            'min_dollar_vol': 0.0,
        }
        evaluation = evaluate_entry_rules(ctx, profile, overrides=overrides)
        return evaluation.passed

    @staticmethod
    def _entry_strategy_for_row(row: pd.Series, prev_rsi: float, rvol: float,
                                strategy_profile: str) -> Optional[str]:
        profile = get_strategy_profile(strategy_profile)
        close_price = float(row.get('close', np.nan))
        open_price = float(row.get('open', np.nan))
        ctx = {
            'live_price': close_price,
            'close': close_price,
            'day_open': open_price,
            'rsi': row.get('RSI'),
            'rsi_prev': prev_rsi,
            'rvol': rvol,
            'volume_pace': rvol,
            'ema20_gt_sma50': bool(row.get('EMA20_GT_SMA50', False)),
            'ma_bull_cross': bool(row.get('MA_BULL_CROSS', False)),
            'bb_below_lower_2': bool(row.get('BB_BELOW_LOWER_2', False)),
            'bb_reclaim_lower': bool(row.get('BB_RECLAIM_LOWER', False)),
            'psar_bull_3': bool(row.get('PSAR_BULL_3', False)),
            'reclaim_ma20': bool(row.get('reclaim_ma20', False)),
            'reclaim_ma50': bool(row.get('reclaim_ma50', False)),
            'break_prev_high': bool(row.get('break_prev_high', False)),
            'stoch_bull_exit_oversold': bool(row.get('STOCH_BULL_EXIT_OVERSOLD', False)),
            'macd_bull_divergence': bool(row.get('MACD_BULL_DIVERGENCE', False)),
            'macd_hist': row.get('MACD_HIST'),
            'macd_hist_delta': row.get('MACD_HIST_DELTA'),
            'obv_uptrend': bool(row.get('OBV_UPTREND', False)),
            'obv_bull_divergence': bool(row.get('OBV_BULL_DIVERGENCE', False)),
            'obv_slope_5': row.get('OBV_SLOPE_5'),
        }
        return select_entry_strategy(ctx, profile) or profile.name

    @staticmethod
    def _stop_fill_price(row: pd.Series, effective_stop: float) -> float:
        """Long stop-market fill approximation for daily bars."""
        bar_open = float(row['open'])
        if bar_open <= effective_stop:
            return round(bar_open, 4)
        return round(effective_stop, 4)

    def _spy_return_for_date(self, today) -> float:
        if self._spy_return is None:
            return np.nan
        value = self._spy_return.get(today, np.nan)
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    def _eod_quality_hold_passes(self, row: pd.Series, trade: Trade, today) -> bool:
        """Daily-bar approximation of the live EOD quality hold rule."""
        try:
            close_px = float(row['close'])
            entry_px = float(trade.entry_price)
            open_px = float(row['open'])
            high_px = float(row['high'])
            low_px = float(row['low'])
        except (TypeError, ValueError):
            return False

        if not all(np.isfinite(v) for v in (close_px, entry_px, open_px, high_px, low_px)):
            return False
        if entry_px <= 0 or open_px <= 0 or high_px <= low_px:
            return False

        profit_pct = (close_px - entry_px) / entry_px
        if profit_pct < EOD_HOLD_MIN_PROFIT_PCT:
            return False

        # Daily bars do not contain true 15:50 VWAP.  The closest
        # no-look-ahead proxy for "above VWAP or above entry" is close > entry.
        if close_px <= entry_px:
            return False

        day_loc = (close_px - low_px) / (high_px - low_px)
        if day_loc < EOD_HOLD_DAY_RANGE_LOCATION_MIN:
            return False

        stock_ret = (close_px - open_px) / open_px
        spy_ret = self._spy_return_for_date(today)
        if not np.isfinite(spy_ret):
            return False
        if (stock_ret - spy_ret) < EOD_HOLD_RELATIVE_STRENGTH_MIN:
            return False

        if EOD_HOLD_REQUIRE_STOP_CONFIRMED:
            atr_chand = trade.__dict__.get('_atr_chand', np.nan)
            try:
                atr_chand = float(atr_chand)
            except (TypeError, ValueError):
                return False
            if not np.isfinite(atr_chand) or atr_chand <= 0:
                return False

        if ANALYST_RATING_EXIT_ENABLED:
            rating_score = trade.__dict__.get('_analyst_rating_score', np.nan)
            try:
                rating_score = float(rating_score)
            except (TypeError, ValueError):
                rating_score = np.nan
            if np.isfinite(rating_score) and rating_score <= ANALYST_RATING_SELL_THRESHOLD:
                return False

        return True

    def _analyst_exit_required(self, trade: Trade, today) -> bool:
        if not ANALYST_RATING_EXIT_ENABLED:
            return False
        rating_ctx = self._analyst_context(trade.symbol, today)
        rating_score = rating_ctx.get(
            'analyst_rating_score',
            trade.__dict__.get('_analyst_rating_score', np.nan),
        )
        trade.__dict__['_analyst_rating_score'] = rating_score
        trade.__dict__['_analyst_rating_total'] = rating_ctx.get(
            'analyst_rating_total',
            trade.__dict__.get('_analyst_rating_total', 0),
        )
        try:
            rating_score = float(rating_score)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(rating_score) and rating_score <= ANALYST_RATING_SELL_THRESHOLD)

    @staticmethod
    def _whole_share_qty(account_equity: float, bucket: float,
                         entry_price: float, risk_stop_dist: float,
                         risk_per_trade_pct: float = RISK_PER_TRADE_PCT) -> int:
        """Live-compatible whole-share size capped by risk and cash bucket."""
        if account_equity <= 0 or bucket <= 0 or entry_price <= 0 or risk_stop_dist <= 0:
            return 0
        risk_dollars = account_equity * max(risk_per_trade_pct, 0.0)
        qty_risk     = int(risk_dollars / risk_stop_dist)
        qty_bucket   = int(bucket / entry_price)
        return max(0, min(qty_risk, qty_bucket))

    @staticmethod
    def _nearest_whole_share(value: float) -> int:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0
        if not math.isfinite(value) or value <= 0:
            return 0
        return int(math.floor(value + 0.5))

    @staticmethod
    def _profit_tier_id(target_r: float) -> str:
        return f"{float(target_r):.2f}R"

    def _entry_risk_per_share(self, trade: Trade) -> float:
        for key in ('_entry_risk_per_share', '_risk_per_share', '_stop_dist'):
            try:
                risk = float(trade.__dict__.get(key, 0) or 0)
            except (TypeError, ValueError):
                risk = 0.0
            if np.isfinite(risk) and risk > 0:
                return risk
        try:
            atr_chand = float(trade.__dict__.get('_atr_chand', 0) or 0)
        except (TypeError, ValueError):
            atr_chand = 0.0
        risk = atr_chand * self._chandelier_mult
        if np.isfinite(risk) and risk > 0:
            return risk
        fallback = trade.entry_price * HARD_STOP_PCT
        return fallback if np.isfinite(fallback) and fallback > 0 else 0.0

    def _profit_tier_exit_plan(self, trade: Trade, row: pd.Series) -> Optional[dict]:
        if not TIERED_PROFIT_EXIT_ENABLED or trade.entry_price <= 0 or trade.qty <= 0:
            return None

        row_high = float(row['high'])
        risk_per_share = self._entry_risk_per_share(trade)
        if risk_per_share <= 0:
            return None
        fired = {
            str(tier_id)
            for tier_id in (trade.__dict__.get('_profit_tiers_fired') or [])
        }
        entry_qty = float(trade.__dict__.get('_entry_qty', trade.qty) or trade.qty)
        sold_so_far = float(trade.__dict__.get('_profit_tier_sold_qty', 0.0) or 0.0)
        available_whole = int(math.floor(max(float(trade.qty), 0.0)))
        planned = []
        remaining_available = available_whole

        for target_r, cumulative_fraction in TIERED_PROFIT_EXIT_R_LEVELS:
            target_r = float(target_r)
            tier_id = self._profit_tier_id(target_r)
            if tier_id in fired:
                continue
            target_price = trade.entry_price + risk_per_share * target_r
            if row_high + 1e-9 < target_price:
                break

            cumulative_target = self._nearest_whole_share(
                entry_qty * float(cumulative_fraction)
            )
            tier_qty = max(0, cumulative_target - self._nearest_whole_share(sold_so_far))
            tier_qty = min(tier_qty, remaining_available)
            fill_price = max(target_price, float(row['open']))
            planned.append({
                'tier_id': tier_id,
                'target_r': target_r,
                'target_price': round(float(target_price), 4),
                'target_pct': round((risk_per_share * target_r) / trade.entry_price, 6),
                'cumulative_fraction': float(cumulative_fraction),
                'planned_qty': int(tier_qty),
                'fill_price': round(float(fill_price), 4),
            })
            sold_so_far += tier_qty
            remaining_available -= tier_qty

        if not planned:
            return None
        return {
            'risk_per_share': risk_per_share,
            'sell_qty': int(sum(item['planned_qty'] for item in planned)),
            'tiers': planned,
        }

    def _allocated_commission(self, trade: Trade, qty: float) -> float:
        entry_qty = float(trade.__dict__.get('_entry_qty', trade.qty) or trade.qty)
        if entry_qty <= 0 or qty <= 0:
            return 0.0
        entry_commission = float(trade.__dict__.get('_entry_commission_total', self._round_trip_cost / 2.0))
        sell_commission = float(trade.__dict__.get('_sell_commission_per_exit', self._round_trip_cost / 2.0))
        return sell_commission + entry_commission * (float(qty) / entry_qty)

    def _set_final_exit_commission(self, trade: Trade) -> None:
        trade.round_trip_commission = self._allocated_commission(trade, trade.qty)

    @staticmethod
    def _deployable_settled_cash(settled_cash: float) -> float:
        """Cash allowed for new-entry buckets after keeping the configured buffer."""
        try:
            settled_cash = float(settled_cash)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(settled_cash) or settled_cash <= 0:
            return 0.0
        pct = min(max(float(SETTLED_CASH_DEPLOYMENT_PCT), 0.0), 1.0)
        return settled_cash * pct

    def _calc_dynamic_max_positions(self, account_equity: float) -> int:
        try:
            account_equity = float(account_equity)
        except (TypeError, ValueError):
            return 0
        if not np.isfinite(account_equity) or account_equity < MIN_BUCKET_SIZE:
            return 0
        return min(int(account_equity / MIN_BUCKET_SIZE), self.max_pos)

    def _calc_entry_allocation(self, account_equity: float, settled_cash: float,
                               open_count: int) -> Dict[str, float]:
        """Live-compatible dynamic slots and bucket sizing."""
        max_pos = self._calc_dynamic_max_positions(account_equity)
        capacity_slots = max(0, max_pos - max(0, int(open_count or 0)))
        deployable_cash = self._deployable_settled_cash(settled_cash)
        cash_slots = (
            int(deployable_cash / MIN_BUCKET_SIZE)
            if deployable_cash >= MIN_BUCKET_SIZE else 0
        )
        entry_slots = min(capacity_slots, cash_slots)
        bucket_size = deployable_cash / entry_slots if entry_slots > 0 else 0.0
        return {
            'max_pos': max_pos,
            'capacity_slots': capacity_slots,
            'cash_slots': cash_slots,
            'entry_slots': entry_slots,
            'deployable_cash': deployable_cash,
            'bucket_size': bucket_size,
        }

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self) -> BacktestResult:
        if self._use_cache and self._try_load_cache():
            # Cache only stores stock data — always refresh regime signals live
            self._download_regime_data()
        else:
            self._download()   # _download() calls _download_regime_data() at the end
            if self._use_cache:
                self._save_cache()
        if not self._data:
            raise RuntimeError("No usable data downloaded. Check tickers / dates.")
        self._validate_regime_data()
        return self._run_loop()

    def _run_loop(self) -> BacktestResult:
        """
        Strategy loop:
        - Tiered profit trims + Chandelier trailing stop + hard stop + break-even floor exit
        - ATR-based position sizing with 0.1% entry slippage
        - $2 round-trip commission per trade
        - Trading-bar hold count (not calendar days)
        """
        trades: List[Trade]             = []
        open_positions: Dict            = {}
        settled_cash                    = self.capital   # settled cash available for new buys
        pending_settlements: Dict       = {}             # sale proceeds settling on future sessions
        equity_curve: Dict[date, float] = {}

        all_dates   = sorted(set(
            d for df in self._data.values() for d in df.index
        ))
        trade_start = pd.Timestamp(self.start)
        trade_end = pd.Timestamp(self.end)
        last_trade_day = None

        def next_trading_session(today):
            for candidate in all_dates:
                if candidate > today:
                    return candidate
            return today

        def release_settled_cash(today) -> None:
            nonlocal settled_cash
            due_dates = [d for d in pending_settlements if d <= today]
            for due in sorted(due_dates):
                settled_cash += pending_settlements.pop(due)

        def pending_cash_total() -> float:
            return float(sum(pending_settlements.values()))

        def mark_to_market(today) -> float:
            total = settled_cash + pending_cash_total()
            for sym, trade in open_positions.items():
                df_pos = self._data.get(sym)
                if df_pos is not None and today in df_pos.index:
                    last_px = float(df_pos.loc[today]['close'])
                    trade.__dict__['_last_close'] = last_px
                else:
                    last_px = float(trade.__dict__.get('_last_close', trade.entry_price))
                total += last_px * trade.qty
            return total

        for today in all_dates:
            today_ts = pd.Timestamp(today)
            if today_ts < trade_start:
                continue
            if today_ts >= trade_end:
                break
            last_trade_day = today

            release_settled_cash(today)
            # bucket is computed per-entry from settled_cash (see entry block below)

            # ── Exit checks ───────────────────────────────────────────────
            for sym in list(open_positions.keys()):
                t  = open_positions[sym]
                df = self._data.get(sym)
                if df is None or today not in df.index:
                    continue

                row = df.loc[today]
                t.__dict__['_last_close'] = float(row['close'])

                # Increment trading-bar count (real trading days, not calendar)
                t.__dict__['_bars_held'] = t.__dict__.get('_bars_held', 0) + 1
                bars_held = t.__dict__['_bars_held']

                atr_chand = t.__dict__.get('_atr_chand', float(row['ATR']))

                # Use the stop known at the start of the bar. With daily data we
                # cannot know whether today's high happened before today's low,
                # so ratcheting from same-bar highs would introduce look-ahead.
                peak_high = t.__dict__.get('_peak_high', t.entry_price)

                # Chandelier stop: peak_high - ATR_CHAND × multiplier
                chand_stop = peak_high - atr_chand * self._chandelier_mult

                # Hard stop: flat 7% below entry
                hard_stop = t.entry_price * (1 - HARD_STOP_PCT)

                # Break-even floor: once up BREAK_EVEN_PCT, stop ≥ entry
                if peak_high >= t.entry_price * (1 + self._break_even_pct):
                    be_stop = t.entry_price
                else:
                    be_stop = 0.0   # inactive until profit threshold reached

                stop_candidates = [
                    ("chandelier_stop", chand_stop),
                    ("hard_stop", hard_stop),
                ]
                if be_stop > 0:
                    stop_candidates.append(("break_even_stop", be_stop))
                exit_stop_reason, effective_stop = max(stop_candidates, key=lambda item: item[1])

                exit_reason = None
                exit_price  = float(row['close'])

                if float(row['low']) <= effective_stop:
                    exit_reason = exit_stop_reason
                    exit_price = self._stop_fill_price(row, effective_stop)

                if exit_reason is None:
                    tier_plan = self._profit_tier_exit_plan(t, row)
                    if tier_plan:
                        fired = list(t.__dict__.get('_profit_tiers_fired') or [])
                        sold_qty = 0.0
                        for tier in tier_plan.get('tiers') or []:
                            tier_id = str(tier.get('tier_id'))
                            planned_qty = int(tier.get('planned_qty') or 0)
                            if planned_qty <= 0:
                                if tier_id not in fired:
                                    fired.append(tier_id)
                                continue

                            partial_qty = min(float(planned_qty), float(t.qty))
                            if partial_qty <= 0:
                                break
                            fill_price = float(tier.get('fill_price') or row['close'])
                            partial = Trade(
                                symbol=t.symbol,
                                entry_date=t.entry_date,
                                entry_price=t.entry_price,
                                exit_date=today.date() if hasattr(today, 'date') else today,
                                exit_price=fill_price,
                                exit_reason=(
                                    f"profit_tier_"
                                    f"{str(tier.get('tier_id') or '').replace('.', '_').lower()}"
                                ),
                                entry_strategy=t.entry_strategy,
                                qty=partial_qty,
                                round_trip_commission=self._allocated_commission(t, partial_qty),
                            )
                            partial.__dict__['_bars_held'] = bars_held
                            partial.__dict__['_regime'] = t.__dict__.get('_regime', 'unknown')
                            partial.__dict__['_partial_exit'] = True
                            sale_proceeds = partial.exit_price * partial.qty - partial.round_trip_commission
                            settle_date = next_trading_session(today)
                            pending_settlements[settle_date] = (
                                pending_settlements.get(settle_date, 0.0) + sale_proceeds
                            )
                            self._filter_stats['total_commissions'] += partial.round_trip_commission
                            trades.append(partial)
                            t.qty = max(0.0, float(t.qty) - partial_qty)
                            sold_qty += partial_qty
                            t.__dict__['_profit_tier_sold_qty'] = (
                                float(t.__dict__.get('_profit_tier_sold_qty', 0.0) or 0.0)
                                + partial_qty
                            )
                            if tier_id not in fired:
                                fired.append(tier_id)

                        if fired:
                            t.__dict__['_profit_tiers_fired'] = fired
                        if sold_qty > 0:
                            if t.qty < 1:
                                del open_positions[sym]
                            else:
                                t.__dict__['_peak_high'] = max(peak_high, float(row['high']))
                            continue

                if exit_reason is None and self._analyst_exit_required(t, today):
                    exit_reason = "analyst_downgrade"
                    exit_price = float(row['close'])
                elif exit_reason is None and t.entry_strategy == "ma_cross" and bool(row.get('MA_BEAR_CROSS', False)):
                    exit_reason = "ma_cross_exit"
                    exit_price = float(row['close'])
                elif exit_reason is None and t.entry_strategy == "bollinger_reversion" and bool(row.get('BB_ABOVE_UPPER_2', False)):
                    exit_reason = "bollinger_exit"
                    exit_price = float(row['close'])
                elif exit_reason is None and t.entry_strategy == "psar_flip" and bool(row.get('PSAR_BEAR_3', False)):
                    exit_reason = "psar_exit"
                    exit_price = float(row['close'])
                elif exit_reason is None and (
                    self._profile.eod_quality_cleanup
                    and not self._eod_quality_hold_passes(row, t, today)
                ):
                    exit_reason = "eod_profit_cleanup"
                elif (
                    exit_reason is None
                    and self._profile.time_stop_bars is not None
                    and bars_held >= int(self._profile.time_stop_bars)
                ):
                    profit_pct = (float(row['close']) - t.entry_price) / t.entry_price
                    min_profit = float(self._profile.time_stop_min_profit or 0.0)
                    if profit_pct <= min_profit:
                        exit_reason = "time_stop"
                        exit_price = float(row['close'])

                if exit_reason:
                    t.exit_date   = today.date() if hasattr(today, 'date') else today
                    t.exit_price  = exit_price
                    t.exit_reason = exit_reason
                    self._set_final_exit_commission(t)
                    sale_proceeds = t.exit_price * t.qty - t.round_trip_commission
                    settle_date = next_trading_session(today)
                    pending_settlements[settle_date] = (
                        pending_settlements.get(settle_date, 0.0) + sale_proceeds
                    )
                    self._filter_stats['total_commissions'] += t.round_trip_commission
                    trades.append(t)
                    del open_positions[sym]
                else:
                    # Survivor gets today's high as the trailing reference for
                    # the next bar.
                    t.__dict__['_peak_high'] = max(peak_high, float(row['high']))

            # ── Regime gates ──────────────────────────────────────────────
            skip_entries  = False
            bear_phase    = False
            past_start    = True

            if self._use_vix_filter and self._vix_series is not None and past_start:
                try:
                    vix_val = self._vix_value_for_date(today)
                    if vix_val is None or pd.isna(vix_val) or float(vix_val) > VIX_THRESHOLD:
                        skip_entries = True
                        self._filter_stats['vix_blocked_days'] += 1
                except Exception:
                    skip_entries = True
                    self._filter_stats['vix_blocked_days'] += 1

            if (not skip_entries and self._use_spy_filter
                    and self._spy_bull is not None and past_start):
                try:
                    bull = self._spy_bull.get(today)
                    if bull is None or pd.isna(bull):
                        skip_entries = True
                        self._filter_stats['spy_blocked_days'] += 1
                    elif not bool(bull):
                        if self._bear_phase_trading and self._profile.allow_bear_phase_entries:
                            bear_phase = True
                            self._filter_stats['spy_bear_trade_days'] += 1
                        else:
                            skip_entries = True
                            self._filter_stats['spy_blocked_days'] += 1
                except Exception:
                    pass

            if bear_phase:
                regime_min_dollar_vol = self._min_dollar_vol * BEAR_PHASE_DOLLAR_VOL_MULT
                regime_risk_pct = RISK_PER_TRADE_PCT * BEAR_PHASE_RISK_MULT
            else:
                regime_min_dollar_vol = self._min_dollar_vol
                regime_risk_pct = RISK_PER_TRADE_PCT

            account_equity = mark_to_market(today)
            allocation = self._calc_entry_allocation(
                account_equity, settled_cash, len(open_positions)
            )
            dynamic_max_pos = int(allocation['max_pos'])

            # ── Entry checks — scanner-driven ──────────────────────────────
            if past_start:
                self._filter_stats['scan_days'] += 1

            if (
                not skip_entries
                and len(open_positions) < dynamic_max_pos
                and past_start
            ):
                for sym, rvol in self._daily_scan(
                    today,
                    min_dollar_vol=regime_min_dollar_vol,
                ):
                    allocation = self._calc_entry_allocation(
                        account_equity, settled_cash, len(open_positions)
                    )
                    entry_slots = int(allocation['entry_slots'])
                    if sym in open_positions or entry_slots <= 0:
                        if sym not in open_positions:
                            self._filter_stats['entries_skipped_full'] += 1
                        continue

                    df = self._data.get(sym)
                    if df is None or today not in df.index:
                        continue

                    idx = df.index.get_loc(today)
                    if idx < 1:
                        continue

                    row      = df.loc[today].copy()
                    prev_rsi = float(df.iloc[idx - 1]['RSI'])
                    spy_ret_63d = self._series_return(self._spy_close, today, 63)
                    spy_ret_126d = self._series_return(self._spy_close, today, 126)
                    try:
                        return_13w = float(row.get('return_13w', np.nan))
                    except (TypeError, ValueError):
                        return_13w = np.nan
                    try:
                        return_26w = float(row.get('return_26w', np.nan))
                    except (TypeError, ValueError):
                        return_26w = np.nan
                    if np.isfinite(return_13w) and np.isfinite(spy_ret_63d):
                        row['relative_strength_63d'] = return_13w - spy_ret_63d
                    if np.isfinite(return_26w) and np.isfinite(spy_ret_126d):
                        row['relative_strength_126d'] = return_26w - spy_ret_126d

                    required_cols = ['open', 'RSI', 'CLV', 'ATR', 'ATR_CHAND', 'prev_high']
                    if pd.isna(row[required_cols]).any():
                        continue

                    if self._entry_signal(
                        row,
                        prev_rsi,
                        rvol,
                        strategy_profile=self._profile.name,
                    ):
                        self._filter_stats['fine_signals'] += 1
                        if bear_phase:
                            self._filter_stats['bear_phase_signals'] += 1
                        entry_strategy = self._entry_strategy_for_row(
                            row,
                            prev_rsi,
                            rvol,
                            self._profile.name,
                        )

                        # Daily bars only know the completed day after the fact,
                        # so swing entries fill no better than the signal close.
                        raw_entry = max(float(row['open']), float(row['close']))
                        if raw_entry < self._min_price:
                            continue
                        entry_price = round(raw_entry * 1.001, 4)

                        # ATR-based position sizing: risk 2% of equity per trade
                        atr_chand_val   = float(row['ATR_CHAND'])
                        chand_dist      = round(atr_chand_val * self._chandelier_mult, 2)
                        # Size from the broker-protected Chandelier distance.
                        # Hard/break-even exits are software overlays.
                        risk_stop_dist  = chand_dist
                        risk_stop_dist  = max(risk_stop_dist, 0.01)  # floor at 1¢

                        # Max position capacity compounds with equity; spending
                        # remains limited by settled cash so T+1 proceeds are not reused.
                        bucket = allocation['bucket_size']
                        qty = self._whole_share_qty(
                            account_equity, bucket, entry_price, risk_stop_dist,
                            regime_risk_pct,
                        )

                        if qty < 1:   # no live-tradable whole-share size
                            continue

                        # Deduct actual cost from settled cash immediately
                        settled_cash -= entry_price * qty

                        t = Trade(
                            symbol      = sym,
                            entry_date  = today.date() if hasattr(today, 'date') else today,
                            entry_price = entry_price,
                            entry_strategy = entry_strategy,
                            qty         = qty,
                            round_trip_commission = self._round_trip_cost,
                        )
                        t.__dict__['_atr_chand']  = atr_chand_val
                        t.__dict__['_peak_high']  = entry_price
                        t.__dict__['_last_close'] = entry_price
                        t.__dict__['_bars_held']  = 0
                        t.__dict__['_regime']     = 'bear' if bear_phase else 'bull'
                        t.__dict__['_entry_qty']  = float(qty)
                        t.__dict__['_entry_risk_per_share'] = float(risk_stop_dist)
                        t.__dict__['_initial_stop_loss'] = entry_price - float(risk_stop_dist)
                        t.__dict__['_entry_commission_total'] = self._round_trip_cost / 2.0
                        t.__dict__['_sell_commission_per_exit'] = self._round_trip_cost / 2.0
                        t.__dict__['_profit_tiers_fired'] = []
                        t.__dict__['_profit_tier_sold_qty'] = 0.0
                        # A daily close-fill is modeled after the live EOD audit
                        # point.  Auditing it against the same close would make
                        # carry impossible because entry includes slippage.
                        t.__dict__['_skip_entry_day_eod'] = True
                        rating_ctx = self._analyst_context(sym, today)
                        t.__dict__['_analyst_rating_score'] = rating_ctx.get('analyst_rating_score', 0.0)
                        t.__dict__['_analyst_rating_total'] = rating_ctx.get('analyst_rating_total', 0)
                        open_positions[sym]        = t
                        self._filter_stats['entries_taken'] += 1
                        if bear_phase:
                            self._filter_stats['bear_phase_entries'] += 1
                        else:
                            self._filter_stats['bull_phase_entries'] += 1

            # Daily-bar approximation of the live 15:50 ET EOD quality cleanup.
            # Daily data only has the final close, so apply the same carry
            # quality rule after all same-day entries have been selected. This
            # does not free same-day buying power because sale proceeds settle
            # on the next trading session.
            if self._profile.eod_quality_cleanup:
                today_date = today.date() if hasattr(today, 'date') else today
                for sym in list(open_positions.keys()):
                    t = open_positions[sym]
                    if t.entry_date != today_date or t.__dict__.get('_bars_held', 0) != 0:
                        continue
                    if t.__dict__.get('_skip_entry_day_eod', False):
                        continue
                    df = self._data.get(sym)
                    if df is None or today not in df.index:
                        continue
                    row = df.loc[today]
                    close_px = float(row['close'])
                    t.__dict__['_last_close'] = close_px
                    if self._eod_quality_hold_passes(row, t, today):
                        continue

                    t.exit_date   = today_date
                    t.exit_price  = close_px
                    t.exit_reason = "eod_profit_cleanup"
                    self._set_final_exit_commission(t)
                    sale_proceeds = t.exit_price * t.qty - t.round_trip_commission
                    settle_date = next_trading_session(today)
                    pending_settlements[settle_date] = (
                        pending_settlements.get(settle_date, 0.0) + sale_proceeds
                    )
                    self._filter_stats['total_commissions'] += t.round_trip_commission
                    trades.append(t)
                    del open_positions[sym]

            if past_start:
                equity_curve[today] = mark_to_market(today)

        # Close any positions still open at end of period
        for sym, t in open_positions.items():
            df = self._data.get(sym)
            if df is not None and not df.empty and last_trade_day is not None:
                period_df = df.loc[df.index <= last_trade_day]
                if period_df.empty:
                    continue
                exit_idx = period_df.index[-1]
                t.exit_price  = float(period_df['close'].iloc[-1])
                t.exit_date   = exit_idx.date()
                t.exit_reason = "end_of_period"
                self._set_final_exit_commission(t)
                settled_cash += t.exit_price * t.qty - t.round_trip_commission
                self._filter_stats['total_commissions'] += t.round_trip_commission
                trades.append(t)
                equity_curve[exit_idx] = settled_cash + pending_cash_total()

        eq_series = pd.Series(equity_curve).sort_index()
        metrics   = self._compute_metrics(trades, eq_series)
        return BacktestResult(
            trades=trades,
            equity_curve=eq_series,
            metrics=metrics,
            filter_stats=dict(self._filter_stats),
        )

    # ── Performance metrics ───────────────────────────────────────────────────
    @staticmethod
    def _compute_metrics(trades: List[Trade], equity: pd.Series) -> Dict:
        if not trades:
            return {}

        completed = [t for t in trades if t.exit_price is not None]
        pnls      = [t.pnl     for t in completed]   # net pnl (post-commission)
        pnl_pcts  = [t.pnl_pct for t in completed]
        wins      = [p for p in pnls if p > 0]
        losses    = [p for p in pnls if p < 0]

        eq_vals  = equity.values.astype(float)
        peak     = np.maximum.accumulate(eq_vals)
        drawdown = (eq_vals - peak) / np.where(peak == 0, 1, peak)
        max_dd   = drawdown.min()

        daily_ret = equity.pct_change().dropna()
        sharpe    = (
            (daily_ret.mean() / daily_ret.std() * np.sqrt(252))
            if daily_ret.std() > 0 else 0.0
        )

        total_return = (
            (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
            if len(equity) > 1 else 0.0
        )

        avg_hold_bars = (
            np.mean([t.__dict__.get('_bars_held', 0) for t in completed])
            if completed else 0.0
        )

        return {
            "total_trades":     len(pnls),
            "win_rate":         len(wins) / len(pnls) if pnls else 0.0,
            "total_pnl":        sum(pnls),
            "total_return_pct": total_return * 100,
            "avg_win":          np.mean(wins)    if wins    else 0.0,
            "avg_loss":         np.mean(losses)  if losses  else 0.0,
            "avg_win_pct":      np.mean([p for p in pnl_pcts if p > 0]) * 100 if wins else 0.0,
            "avg_loss_pct":     np.mean([p for p in pnl_pcts if p < 0]) * 100 if losses else 0.0,
            "avg_hold_bars":    avg_hold_bars,
            "profit_factor":    (
                sum(wins) / abs(sum(losses))
                if losses and sum(losses) != 0 else float('inf')
            ),
            "max_drawdown_pct": max_dd * 100,
            "sharpe_ratio":     sharpe,
            "exit_reasons":     pd.Series(
                [t.exit_reason for t in completed]
            ).value_counts().to_dict(),
            "regime_entries":   pd.Series(
                [t.__dict__.get('_regime', 'unknown') for t in completed]
            ).value_counts().to_dict(),
        }

    # ── Console report ────────────────────────────────────────────────────────
    @staticmethod
    def print_report(result: BacktestResult, capital: float = BACKTEST_INITIAL_CAPITAL) -> None:
        m  = result.metrics
        fs = result.filter_stats
        if not m:
            print("No trades executed.")
            VelocityBacktest._print_filter_stats(fs)
            return

        final_equity = capital + m['total_pnl']

        print("\n" + "=" * 65)
        print("  VELOCITY STRATEGY — FORWARD BACKTEST REPORT  (v2)")
        print("  Profile-aware swing entry | T+1 cash settlement | Chandelier stop")
        print("=" * 65)
        print(f"  Starting Capital  : ${capital:,.2f}")
        print(f"  Final Equity      : ${final_equity:,.2f}")
        print(f"  Total Return      : {m['total_return_pct']:.2f}%  (net of commissions)")
        print(f"  Total P&L (net)   : ${m['total_pnl']:,.2f}  "
              f"(comm: ${fs.get('total_commissions', 0):,.2f})")
        print("-" * 65)
        print(f"  Total Trades      : {m['total_trades']}")
        print(f"  Win Rate          : {m['win_rate']:.1%}")
        print(f"  Avg Win (net)     : ${m['avg_win']:,.2f}  ({m['avg_win_pct']:.2f}%)")
        print(f"  Avg Loss (net)    : ${m['avg_loss']:,.2f}  ({m['avg_loss_pct']:.2f}%)")
        print(f"  Profit Factor     : {m['profit_factor']:.2f}")
        print(f"  Avg Hold          : {m['avg_hold_bars']:.1f} trading bars")
        print("-" * 65)
        print(f"  Max Drawdown      : {m['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio      : {m['sharpe_ratio']:.2f}")
        print("  Exit Breakdown    :")
        for reason, count in sorted(m['exit_reasons'].items(), key=lambda x: -x[1]):
            print(f"    {reason:<22}: {count}")
        print("=" * 65)

        VelocityBacktest._print_filter_stats(fs)

    @staticmethod
    def _print_filter_stats(fs: Dict) -> None:
        if not fs:
            return
        print("\n  FILTER FUNNEL")
        print("  " + "-" * 40)
        print(f"  Scan days           : {fs.get('scan_days', 0):,}")
        spy_d = fs.get('spy_blocked_days', 0)
        vix_d = fs.get('vix_blocked_days', 0)
        if spy_d:
            print(f"  SPY-blocked days    : {spy_d:,}")
        bear_days = fs.get('spy_bear_trade_days', 0)
        if bear_days:
            print(f"  Bear trade days     : {bear_days:,}")
        if vix_d:
            print(f"  VIX-blocked days    : {vix_d:,}")
        print(f"  Coarse candidates   : {fs.get('coarse_candidates', 0):,}  "
              f"(price/vol/dollar-vol pass)")
        print(f"  Fine signals        : {fs.get('fine_signals', 0):,}  "
              f"(profile gates pass)")
        print(f"  Entries taken       : {fs.get('entries_taken', 0):,}")
        bear_entries = fs.get('bear_phase_entries', 0)
        if bear_entries:
            print(f"  Bear entries        : {bear_entries:,}  "
                  f"(signals: {fs.get('bear_phase_signals', 0):,})")
        skipped = fs.get('entries_skipped_full', 0)
        if skipped:
            print(f"  Skipped (slot/cash) : {skipped:,}")
        print()

    # ── Per-trade log ─────────────────────────────────────────────────────────
    @staticmethod
    def print_trades(result: BacktestResult, top_n: int = 20) -> None:
        """Print the top_n trades by absolute gross P&L."""
        trades = sorted(
            [t for t in result.trades if t.exit_price is not None],
            key=lambda t: abs(t.gross_pnl), reverse=True
        )[:top_n]
        if not trades:
            print("No completed trades.")
            return
        print(f"\n{'SYM':<6}  {'ENTRY':>10}  {'EXIT':>10}  "
              f"{'ENTRY $':>8}  {'EXIT $':>8}  {'NET PNL':>8}  {'PNL%':>7}  "
              f"{'BARS':>4}  {'STRATEGY':<20}  {'REASON'}")
        print("-" * 115)
        for t in trades:
            bars = t.__dict__.get('_bars_held', '?')
            print(
                f"{t.symbol:<6}  {str(t.entry_date):>10}  {str(t.exit_date):>10}  "
                f"${t.entry_price:>7.2f}  ${t.exit_price:>7.2f}  "
                f"${t.net_pnl:>7.2f}  {t.pnl_pct*100:>6.1f}%  "
                f"{str(bars):>4}  {t.entry_strategy or '-':<20}  {t.exit_reason}"
            )
