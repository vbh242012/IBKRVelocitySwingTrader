"""
Velocity Strategy Backtester — v2 (Production-Grade Edition)
─────────────────────────────────────────────────────────────
Improvements over v1:
  1. RVOL is used for scanner ranking, not as an entry gate in the 8096 rule set.
  2. bars_held counts actual trading bars open (not calendar days).
     Previously, Friday entry → Saturday + Sunday counted as 2 bars,
     firing velocity_exit after only 1 real trading session.
  3. Break-even floor: once profit ≥ BREAK_EVEN_PCT (4%), the effective
     stop cannot fall below entry price — locks in break-even.
  4. ATR-based whole-share position sizing: risk RISK_PER_TRADE_PCT (2%)
     of equity per trade, using the tighter of chandelier or 7% hard-stop
     as the risk distance.  Capped by the per-bucket dollar limit.
  5. 0.1% entry slippage added to entry_price for realism.
  6. Configurable backtest commission deducted per closed trade.
  7. Composite scanner score = pct_daily_gain × RVOL (volume-weighted
     momentum) — ranks high-conviction setups ahead of thin movers.
  8. Data caching: downloaded + indicator-enriched DataFrames are
     pickled to backtest/.cache/ so re-runs skip the market-data download.
  9. Stop fills use the open when price gaps through the stop.
 10. Filter funnel stats printed at end: shows exactly where signals
     are lost across each filter stage.

Universe discovery (mirrors the live IB scanner):
  - Candidate pool  : current NASDAQ Global Select/Market + NYSE/NYSE American
                      common equities, excluding warrants/rights/units/preferreds
  - Daily scan      : each bar, rank by composite score and keep top
                      all scanner-passed stocks unless scan_count is set

Important research caveat:
  - This uses currently listed symbols. It is good for regression and rough
    forward validation, but it is not a survivorship-free institutional
    historical database.

ORB approximation: previous day's high acts as the opening-range breakout
level.  A close above it on the signal day mirrors the live "price > orb_high"
check.

Entry rules (8096 production filter):
  1. Data sufficiency    : ≥ MIN_CANDLES (210) bars of history
  2. ORB                 : close > previous day's high
  3. RSI acceleration    : RSI delta ≥ RSI_MIN_DELTA
  4. RSI level           : RSI > RSI_THRESHOLD
  5. Close location      : close in upper half of day's range
  6. Intraday gain       : close at least INTRADAY_GAIN_MIN above open
  7. ATR% cap            : ATR_CHAND / close ≤ ATR_PCT_MAX
  8. Gap cap             : open ≤ previous high × (1 + GAP_MAX_PCT)
  9. Spread              : not available in daily data — skipped
 10. SPY regime          : SPY close > SMA50 > SMA200 (optional)
 11. Correlation         : not practical in daily batch — skipped
 12. Sector clustering   : not practical in daily batch — skipped

Exit rules (production):
  • Chandelier trailing stop : peak_high - ATR_CHAND × CHANDELIER_MULT
  • Hard stop                : entry × (1 - HARD_STOP_PCT) = 7% from entry
  • Break-even floor         : if profit > BREAK_EVEN_PCT, stop ≥ entry
  • Velocity time exit       : held ≥ hold_bars and profit < 5%
  • (No take-profit bracket — removed from production)
"""

from __future__ import annotations

import hashlib
import io
import json
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
    PROFIT_MIN_THRESHOLD,
    RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW, RSI_THRESHOLD,
    BACKTEST_INITIAL_CAPITAL, MAX_POSITIONS_CAP, MIN_BUCKET_SIZE, BACKTEST_SCAN_COUNT,
    BACKTEST_MAX_SYMBOLS,
    SETTLED_CASH_DEPLOYMENT_PCT,
    SCAN_MIN_PRICE, SCAN_MIN_VOLUME, SCAN_MIN_DOLLAR_VOL,
    VIX_THRESHOLD, HOLD_TRADING_BARS,
    MIN_CANDLES, VCP_RATIO, BREAKOUT_PCT, BACKTEST_RVOL_MIN, GAP_MAX_PCT,
    CHANDELIER_PERIOD, CHANDELIER_MULT,
    RSI_MIN_DELTA, DAY_RANGE_LOCATION_MIN, INTRADAY_GAIN_MIN, ATR_PCT_MAX, HARD_STOP_PCT,
    SMA200_SLOPE_LOOKBACK,
    RISK_PER_TRADE_PCT, BREAK_EVEN_PCT,
    BACKTEST_COMMISSION_PER_ORDER,
    BEAR_PHASE_TRADING_ENABLED, BEAR_PHASE_RISK_MULT,
    BEAR_PHASE_DOLLAR_VOL_MULT, BEAR_BACKTEST_RVOL_MIN,
    BEAR_VCP_RATIO, BEAR_BREAKOUT_PCT, BEAR_RSI_THRESHOLD,
    BEAR_RSI_MIN_DELTA, BEAR_GAP_MAX_PCT,
)
from src.indicators import apply_all

warnings.filterwarnings("ignore", category=FutureWarning)

_NASDAQ_LISTED_URL = 'https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt'
_OTHER_LISTED_URL  = 'https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt'
_CACHE_DIR         = os.path.join(os.path.dirname(__file__), ".cache")
_DEFAULT_ROUND_TRIP_COST = BACKTEST_COMMISSION_PER_ORDER * 2
_CACHE_VERSION = "v3_common_equity_universe"
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
    hold_bars       : trading bars before velocity time-exit check
    scan_count      : top-N from daily scanner considered per bar; 0 means all
    max_symbols     : download cap for bounded validation; 0/None means full filtered universe
    min_price       : minimum close price filter
    min_volume      : minimum daily share volume filter
    min_dollar_vol  : minimum 20-day avg dollar volume
    use_spy_filter  : if True, skip entries when SPY < SMA50 or SMA50 < SMA200
    use_vix_filter  : if True, skip new entries when VIX is missing or > VIX_THRESHOLD
    vix_delay_bars  : daily-bar proxy for delayed VIX data; 0=current bar,
                      1=prior available VIX bar (used for 15-minute delayed research)
    rvol_min             : legacy optimizer parameter; 8096 uses RVOL for ranking, not as an entry gate
    break_even_pct       : once profit exceeds this, floor the stop at entry (0.04 optimal)
    profit_min_threshold : velocity exit fires if profit < this after hold_bars
    chandelier_mult      : ATR multiplier for trailing stop
    breakout_pct         : legacy optimizer parameter; 10-day-high proximity is no longer an entry gate
    vcp_ratio            : legacy optimizer parameter; 8096 does not gate on VCP
    conservative_daily_entry : if True, daily-bar entries fill no better than the signal-day close
    use_cache            : load/save downloaded data from backtest/.cache/
    """

    def __init__(
        self,
        start:          str   = "2025-01-01",
        end:            str   = "2026-05-01",
        capital:        float = BACKTEST_INITIAL_CAPITAL,
        max_pos:        int   = MAX_POSITIONS_CAP,
        hold_bars:      int   = HOLD_TRADING_BARS,
        scan_count:     int   = BACKTEST_SCAN_COUNT,
        min_price:      float = SCAN_MIN_PRICE,
        min_volume:     float = SCAN_MIN_VOLUME,
        min_dollar_vol: float = SCAN_MIN_DOLLAR_VOL,
        use_spy_filter: bool  = True,
        use_vix_filter: bool  = True,
        vix_delay_bars: int   = 0,
        rvol_min:             float = BACKTEST_RVOL_MIN,
        break_even_pct:       float = BREAK_EVEN_PCT,
        profit_min_threshold: float = PROFIT_MIN_THRESHOLD,
        chandelier_mult:      float = CHANDELIER_MULT,
        breakout_pct:         float = BREAKOUT_PCT,
        vcp_ratio:            float = VCP_RATIO,
        bear_phase_trading:   bool  = BEAR_PHASE_TRADING_ENABLED,
        commission_per_order: float = BACKTEST_COMMISSION_PER_ORDER,
        max_symbols:          int   = BACKTEST_MAX_SYMBOLS,
        conservative_daily_entry: bool = False,
        use_cache:            bool  = True,
    ):
        self.start                 = start
        self.end                   = end
        self.capital               = capital
        self.max_pos               = max_pos
        self.hold_bars             = hold_bars
        self._scan_count           = scan_count
        self._min_price            = min_price
        self._min_volume           = min_volume
        self._min_dollar_vol       = min_dollar_vol
        self._use_spy_filter       = use_spy_filter
        self._use_vix_filter       = use_vix_filter
        self._vix_delay_bars       = max(0, int(vix_delay_bars or 0))
        self._rvol_min             = rvol_min
        self._break_even_pct       = break_even_pct
        self._profit_min_threshold = profit_min_threshold
        self._chandelier_mult      = chandelier_mult
        self._breakout_pct         = breakout_pct
        self._vcp_ratio            = vcp_ratio
        self._bear_phase_trading   = bear_phase_trading
        self._round_trip_cost      = max(0.0, float(commission_per_order)) * 2.0
        self._max_symbols          = max(0, int(max_symbols or 0))
        self._conservative_daily_entry = bool(conservative_daily_entry)
        self._use_cache            = use_cache

        self._data:        Dict[str, pd.DataFrame] = {}
        self._vix_series:  Optional[pd.Series]     = None
        self._spy_bull:    Optional[pd.Series]      = None

        # Download starts early enough to warm up MA200 + chandelier ATR
        _trade_start     = date.fromisoformat(start)
        self._data_start = (_trade_start - timedelta(days=400)).isoformat()

        # Filter funnel accumulators (populated during run)
        self._filter_stats: Dict = {
            'scan_days':            0,
            'coarse_candidates':    0,   # pass price/vol/dollar-vol coarse scan
            'fine_signals':         0,   # pass full 8096 _entry_signal
            'entries_taken':        0,   # actually opened a position
            'entries_skipped_full': 0,   # signal fired but max_pos already full
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

    def _cache_meta_matches_request(self, meta: Dict, data: Dict[str, pd.DataFrame]) -> bool:
        if not meta:
            # Legacy cache payloads did not store metadata. Reuse them only for
            # bounded symbol-cap runs where the cache visibly covers the period.
            return self._max_symbols > 0 and len(data) <= self._max_symbols
        try:
            return (
                meta.get('version') == _CACHE_VERSION
                and float(meta.get('min_dollar_vol')) == float(self._min_dollar_vol)
                and int(meta.get('max_symbols')) == int(self._max_symbols)
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
            repaired = 0
            for sym, df in list(self._data.items()):
                if 'CLV' not in df.columns:
                    day_range = df['high'] - df['low']
                    df = df.copy()
                    df['CLV'] = (df['close'] - df['low']) / day_range.where(day_range != 0)
                    self._data[sym] = df
                    repaired += 1
            print(f"  Loaded {len(self._data):,} symbols from cache.")
            if repaired:
                print(f"  Repaired {repaired:,} cached symbols with CLV columns.")
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
            (not self._use_spy_filter or self._spy_bull is not None)
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
        if self._use_spy_filter:
            if not self._series_covers_window(
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
                df['prev_high']         = df['high'].shift(1)
                df['avg_vol_20']        = df['volume'].rolling(20).mean()
                df['avg_dollar_vol_20'] = (
                    (df['close'] * df['volume']).rolling(20).mean()
                )
                # Previous-bar diagnostic values kept for compatibility with
                # older research reports. 8096 no longer gates entries on VCP
                # or 10-day-high proximity.
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

        if self._use_spy_filter:
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
            ma50  = sc.rolling(50).mean()
            ma200 = sc.rolling(200).mean()
            # Require both: price > MA50 AND MA50 > MA200 (golden cross).
            # This blocks entries during corrections AND recovery — the
            # recovery phase from a deep correction produces many false
            # breakouts before the trend is genuinely re-established.
            self._spy_bull = (sc > ma50) & (ma50 > ma200)

        if self._use_cache and self._regime_requirements_met():
            self._save_regime_cache()

    def _validate_regime_data(self) -> None:
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

    # ── Daily scanner simulation ──────────────────────────────────────────────
    def _daily_scan(
        self,
        today,
        rvol_min: Optional[float] = None,
        min_dollar_vol: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Simulate the broad IB active-stock scanner with production pre-filters.
        Returns list of (symbol, rvol) tuples, sorted by composite score
        (% daily gain × RVOL) descending. scan_count <= 0 means every
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

            # RVOL ranking input (not an 8096 entry gate)
            avg_vol = row.get('avg_vol_20', 0)
            if pd.isna(avg_vol) or avg_vol <= 0:
                continue
            rvol = row['volume'] / avg_vol

            self._filter_stats['coarse_candidates'] += 1

            pct   = (row['close'] - prev_row['close']) / prev_row['close']
            score = pct * max(rvol, 1.0)   # composite: momentum × volume
            scored.append((sym, score, rvol))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored if self._scan_count <= 0 else scored[:self._scan_count]
        return [(sym, rvol) for sym, _, rvol in selected]

    # ── Signal check ─────────────────────────────────────────────────────────
    @staticmethod
    def _entry_signal(row: pd.Series, prev_rsi: float, rvol: float,
                      rvol_min: float, vcp_ratio: float = VCP_RATIO,
                      breakout_pct: float = BREAKOUT_PCT,
                      rsi_threshold: float = RSI_THRESHOLD,
                      rsi_min_delta: float = RSI_MIN_DELTA) -> bool:
        """8096 production filter (daily-bar approximation)."""
        # ORB proxy: close above previous day's high
        c_orb = not pd.isna(row['prev_high']) and row['close'] > row['prev_high']

        # RSI level + minimum delta. RSI rising is implied by positive delta.
        # No upper RSI cap: high RSI on a breakout day is a FEATURE, not a bug.
        # Momentum leaders (like OKLO, NVDA breakouts) have RSI 75-85 on signal day.
        rsi_delta   = row['RSI'] - prev_rsi
        c_rsi_delta = rsi_delta >= rsi_min_delta
        c_rsi_lvl   = row['RSI'] > rsi_threshold
        c_clv        = (
            not pd.isna(row['CLV'])
            and row['CLV'] >= DAY_RANGE_LOCATION_MIN
        )
        c_open_gain  = (
            not pd.isna(row['open'])
            and row['open'] > 0
            and (row['close'] - row['open']) / row['open'] >= INTRADAY_GAIN_MIN
        )
        atr_chand = row.get('ATR_CHAND', row.get('ATR', np.nan))
        c_atr_pct = (
            not pd.isna(atr_chand)
            and row['close'] > 0
            and (atr_chand / row['close']) <= ATR_PCT_MAX
        )

        return (
            c_orb
            and c_rsi_delta and c_rsi_lvl
            and c_clv and c_open_gain and c_atr_pct
        )

    @staticmethod
    def _stop_fill_price(row: pd.Series, effective_stop: float) -> float:
        """Long stop-market fill approximation for daily bars."""
        bar_open = float(row['open'])
        if bar_open <= effective_stop:
            return round(bar_open, 4)
        return round(effective_stop, 4)

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
        - Chandelier trailing stop + hard stop + break-even floor exit
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

                effective_stop = max(chand_stop, hard_stop, be_stop)

                profit_pct = (float(row['close']) - t.entry_price) / t.entry_price

                exit_reason = None
                exit_price  = float(row['close'])

                if float(row['low']) <= effective_stop:
                    exit_reason = "chandelier_stop"
                    exit_price = self._stop_fill_price(row, effective_stop)
                elif bars_held >= self.hold_bars and profit_pct < self._profit_min_threshold:
                    exit_reason = "velocity_exit"

                if exit_reason:
                    t.exit_date   = today.date() if hasattr(today, 'date') else today
                    t.exit_price  = exit_price
                    t.exit_reason = exit_reason
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
                        if self._bear_phase_trading:
                            bear_phase = True
                            self._filter_stats['spy_bear_trade_days'] += 1
                        else:
                            skip_entries = True
                            self._filter_stats['spy_blocked_days'] += 1
                except Exception:
                    pass

            if bear_phase:
                regime_rvol_min = BEAR_BACKTEST_RVOL_MIN
                regime_vcp_ratio = BEAR_VCP_RATIO
                regime_breakout_pct = BEAR_BREAKOUT_PCT
                regime_rsi_threshold = BEAR_RSI_THRESHOLD
                regime_rsi_min_delta = BEAR_RSI_MIN_DELTA
                regime_gap_max_pct = BEAR_GAP_MAX_PCT
                regime_min_dollar_vol = self._min_dollar_vol * BEAR_PHASE_DOLLAR_VOL_MULT
                regime_risk_pct = RISK_PER_TRADE_PCT * BEAR_PHASE_RISK_MULT
            else:
                regime_rvol_min = self._rvol_min
                regime_vcp_ratio = self._vcp_ratio
                regime_breakout_pct = self._breakout_pct
                regime_rsi_threshold = RSI_THRESHOLD
                regime_rsi_min_delta = RSI_MIN_DELTA
                regime_gap_max_pct = GAP_MAX_PCT
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
                    rvol_min=regime_rvol_min,
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

                    row      = df.loc[today]
                    prev_rsi = float(df.iloc[idx - 1]['RSI'])

                    required_cols = ['open', 'RSI', 'CLV', 'ATR', 'ATR_CHAND', 'prev_high']
                    if pd.isna(row[required_cols]).any():
                        continue

                    # Gap cap: skip if already gapped beyond GAP_MAX_PCT
                    if float(row['open']) > float(row['prev_high']) * (1 + regime_gap_max_pct):
                        continue

                    if self._entry_signal(
                        row, prev_rsi, rvol, regime_rvol_min,
                        regime_vcp_ratio, regime_breakout_pct,
                        regime_rsi_threshold, regime_rsi_min_delta,
                    ):
                        self._filter_stats['fine_signals'] += 1
                        if bear_phase:
                            self._filter_stats['bear_phase_signals'] += 1

                        # Daily bars only know the completed day after the fact.
                        # The legacy approximation filled at open/previous high even
                        # though the signal used the same day close/RSI/CLV; that can
                        # create impossible penny-to-close fills on intraday blowoffs.
                        raw_entry = max(float(row['open']), float(row['prev_high']))
                        if self._conservative_daily_entry:
                            raw_entry = max(raw_entry, float(row['close']))
                        if raw_entry < self._min_price:
                            continue
                        entry_price = round(raw_entry * 1.001, 4)

                        # ATR-based position sizing: risk 2% of equity per trade
                        atr_chand_val   = float(row['ATR_CHAND'])
                        chand_dist      = atr_chand_val * self._chandelier_mult
                        hard_stop_dist  = entry_price * HARD_STOP_PCT
                        # The tighter stop is the one that fires first → defines risk
                        risk_stop_dist  = min(chand_dist, hard_stop_dist)
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
                            qty         = qty,
                            round_trip_commission = self._round_trip_cost,
                        )
                        t.__dict__['_atr_chand']  = atr_chand_val
                        t.__dict__['_peak_high']  = entry_price
                        t.__dict__['_last_close'] = entry_price
                        t.__dict__['_bars_held']  = 0
                        t.__dict__['_regime']     = 'bear' if bear_phase else 'bull'
                        open_positions[sym]        = t
                        self._filter_stats['entries_taken'] += 1
                        if bear_phase:
                            self._filter_stats['bear_phase_entries'] += 1
                        else:
                            self._filter_stats['bull_phase_entries'] += 1

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
        print("  8096 entry | T+1 cash settlement | Chandelier stop")
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
              f"(8096 pass)")
        print(f"  Entries taken       : {fs.get('entries_taken', 0):,}")
        bear_entries = fs.get('bear_phase_entries', 0)
        if bear_entries:
            print(f"  Bear entries        : {bear_entries:,}  "
                  f"(signals: {fs.get('bear_phase_signals', 0):,})")
        skipped = fs.get('entries_skipped_full', 0)
        if skipped:
            print(f"  Skipped (pos full)  : {skipped:,}")
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
              f"{'BARS':>4}  {'REASON'}")
        print("-" * 90)
        for t in trades:
            bars = t.__dict__.get('_bars_held', '?')
            print(
                f"{t.symbol:<6}  {str(t.entry_date):>10}  {str(t.exit_date):>10}  "
                f"${t.entry_price:>7.2f}  ${t.exit_price:>7.2f}  "
                f"${t.net_pnl:>7.2f}  {t.pnl_pct*100:>6.1f}%  "
                f"{str(bars):>4}  {t.exit_reason}"
            )
