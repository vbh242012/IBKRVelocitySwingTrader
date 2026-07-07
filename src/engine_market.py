import time
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytz
from ib_async import Index, Stock, util

from src.config import (
    DAILY_LOOKBACK, DAILY_BAR_SIZE,
    HISTORICAL_DATA_TIMEOUT_SEC,
    HISTORICAL_DATA_WARMUP_ENABLED,
    HMDS_WARMUP_MAX_RETRIES, HMDS_WARMUP_RETRY_WAIT_SEC,
    MA_SLOW, SMA200_SLOPE_LOOKBACK,
    MIN_CANDLES,
    RSI_PERIOD, ATR_PERIOD, MA_FAST,
    CHANDELIER_PERIOD,
    VIX_MARKET_DATA_TYPE, MARKET_DATA_TYPE,
    VIX_CACHE_TTL_SEC,
    VIX_FAILURE_COOLDOWN_BASE_SEC, VIX_FAILURE_COOLDOWN_MAX_SEC,
)
from src.indicators import apply_all

# Import logger and _TZ_NY from engine_base at runtime to avoid circular imports
# They will be accessible via self (mixin pattern)


class MarketDataMixin:

    def _request_vix_tickers(self):
        """Request VIX with its own market-data type, then restore stock data mode."""
        restore_type = None
        if VIX_MARKET_DATA_TYPE != MARKET_DATA_TYPE:
            restore_type = MARKET_DATA_TYPE
            self.ib.reqMarketDataType(VIX_MARKET_DATA_TYPE)
        try:
            return self.ib.reqTickers(self._vix_contract)
        finally:
            if restore_type is not None:
                self.ib.reqMarketDataType(restore_type)

    def _fetch_vix_price(self) -> Optional[float]:
        """Return the current/delayed VIX value, with a robust historical fallback.

        VIX is a regime filter — precision matters less than availability. Strategy:
        1. If a valid VIX was fetched within the configured TTL, return it from
           cache to avoid hammering IBKR with repeated reqHistoricalData calls.
        2. Try the live/delayed ticker, checking multiple price fields.
        3. On ticker miss, fetch bounded historical data.
        4. If both fresh paths fail, enter a cooldown. A stale VIX reading must
           not authorize new risk.
        """
        now = time.time()
        # Return cached value if still fresh enough.
        if self._last_vix is not None and (now - self._last_vix_ts) < VIX_CACHE_TTL_SEC:
            return self._last_vix

        next_retry_ts = float(getattr(self, '_next_vix_retry_ts', 0.0) or 0.0)
        if now < next_retry_ts:
            wait_s = int(next_retry_ts - now)
            self._metric_inc('vix_retry_suppressed')
            from src.engine_base import logger
            logger.warning(
                f"VIX data unavailable; retry cooldown active for {wait_s}s. "
                "Skipping entries as precaution."
            )
            return None

        try:
            vix_tickers = self._request_vix_tickers()
        except Exception as e:
            from src.engine_base import logger
            logger.warning(f"VIX ticker request failed ({e}); trying historical fallback.")
            self._metric_inc('vix_ticker_failures')
            vix_tickers = []

        if vix_tickers:
            vix_ticker = vix_tickers[0]
            # Check multiple fields: marketPrice() covers last/ask/bid; also try
            # close and prevClose which are populated by delayed subscriptions.
            vix_price = (
                self._coerce_positive_price(vix_ticker.marketPrice())
                or self._coerce_positive_price(getattr(vix_ticker, 'close', None))
                or self._coerce_positive_price(getattr(vix_ticker, 'last', None))
                or self._coerce_positive_price(getattr(vix_ticker, 'prevClose', None))
            )
            if vix_price is not None:
                self._record_vix_success(vix_price, source="ticker")
                return vix_price
            from src.engine_base import logger
            logger.info("VIX ticker returned no usable price; using historical fallback.")
            self._metric_inc('vix_ticker_misses')
        else:
            from src.engine_base import logger
            logger.info("VIX ticker unavailable; using historical fallback.")
            self._metric_inc('vix_ticker_misses')

        bars = self._request_historical_bars(
            self._vix_contract,
            label="VIX",
            duration='5 D',
            bar_size=DAILY_BAR_SIZE,
            what='TRADES',
            use_rth=False,
            timeout=HISTORICAL_DATA_TIMEOUT_SEC,
            metric_prefix='vix',
        )
        if not bars:
            from src.engine_base import logger
            logger.warning("VIX historical fallback returned no bars.")
            self._metric_inc('vix_fallback_failures')
            self._record_vix_failure("historical_no_bars")
            return None
        hist_price = self._coerce_positive_price(getattr(bars[-1], 'close', None))
        if hist_price is None:
            from src.engine_base import logger
            logger.warning("VIX historical fallback returned an invalid close.")
            self._metric_inc('vix_fallback_failures')
            self._record_vix_failure("historical_invalid_close")
            return None
        from src.engine_base import logger
        logger.info(f"VIX fallback: using latest historical close {hist_price:.2f}")
        self._metric_inc('vix_fallback_successes')
        self._record_vix_success(hist_price, source="historical")
        return hist_price

    def _record_vix_success(self, price: float, source: str):
        self._last_vix = price
        self._last_vix_ts = time.time()
        self._last_vix_source = source
        self._vix_failure_count = 0
        self._next_vix_retry_ts = 0.0
        self._last_vix_failure_ts = 0.0

    def _record_vix_failure(self, reason: str):
        from src.engine_base import logger, _TZ_NY
        if not hasattr(self, '_vix_failure_count'):
            self._vix_failure_count = 0
        if not hasattr(self, '_next_vix_retry_ts'):
            self._next_vix_retry_ts = 0.0
        self._vix_failure_count = int(getattr(self, '_vix_failure_count', 0) or 0) + 1
        self._last_vix_failure_ts = time.time()
        base = max(float(VIX_FAILURE_COOLDOWN_BASE_SEC), 0.0)
        max_wait = max(float(VIX_FAILURE_COOLDOWN_MAX_SEC), base)
        cooldown = min(base * self._vix_failure_count, max_wait)
        self._next_vix_retry_ts = self._last_vix_failure_ts + cooldown
        next_retry = datetime.fromtimestamp(self._next_vix_retry_ts, _TZ_NY)
        logger.warning(
            f"VIX failure recorded ({reason}); retry in {cooldown:.0f}s "
            f"at {next_retry.strftime('%H:%M:%S %Z')}."
        )

    def _ensure_vix_contract(self) -> bool:
        """Qualify and cache the VIX contract; fail closed for entry logic."""
        from src.engine_base import logger
        if self._vix_contract is not None:
            return True
        try:
            vix_contracts = self.ib.qualifyContracts(Index('VIX', 'CBOE'))
            if not vix_contracts:
                logger.warning("VIX contract unavailable.")
                return False
            self._vix_contract = vix_contracts[0]
            return True
        except Exception as e:
            logger.warning(f"VIX contract qualification failed: {e}")
            return False

    def _request_historical_bars(
        self,
        contract,
        label: str,
        duration: str,
        bar_size: str,
        what: str,
        use_rth: bool,
        timeout: float,
        metric_prefix: Optional[str] = None,
    ):
        """Bounded historical request with compact health bookkeeping."""
        from src.engine_base import logger, _TZ_NY
        if not hasattr(self, '_historical_data_health'):
            self._historical_data_health = {}
        started = time.time()
        try:
            bars = self.ib.reqHistoricalData(
                contract, '', duration, bar_size, what, use_rth, timeout=timeout
            )
            latency = time.time() - started
            ok = bool(bars)
            self._historical_data_health[label] = {
                'ok': ok,
                'last_checked_at': datetime.now(_TZ_NY).isoformat(),
                'latency_sec': round(latency, 3),
                'bars': len(bars) if bars else 0,
                'error': None if ok else 'no_bars',
            }
            if metric_prefix and not ok:
                self._metric_inc(f'{metric_prefix}_historical_no_bars')
            return bars
        except Exception as e:
            latency = time.time() - started
            self._historical_data_health[label] = {
                'ok': False,
                'last_checked_at': datetime.now(_TZ_NY).isoformat(),
                'latency_sec': round(latency, 3),
                'bars': 0,
                'error': str(e),
            }
            if metric_prefix:
                self._metric_inc(f'{metric_prefix}_historical_exceptions')
            logger.warning(f"{label} historical request failed after {latency:.1f}s: {e}")
            return []

    def _warmup_historical_data(self, reason: str = "manual") -> bool:
        """Probe SPY then VIX so HMDS wakes before the entry gate depends on it.

        Fix 7: Retries up to HMDS_WARMUP_MAX_RETRIES times with a wait between
        attempts so a momentarily degraded HMDS farm doesn't permanently block
        entries on every startup or reconnect.
        """
        from src.engine_base import logger
        if not HISTORICAL_DATA_WARMUP_ENABLED:
            return True

        for attempt in range(1, HMDS_WARMUP_MAX_RETRIES + 1):
            try:
                spy = Stock('SPY', 'SMART', 'USD', primaryExchange='ARCA')
                qualified = self.ib.qualifyContracts(spy)
                if qualified:
                    spy = qualified[0]
            except Exception as e:
                logger.warning(f"HMDS WARMUP[{reason}] attempt {attempt}: SPY qualification failed: {e}")
                if attempt < HMDS_WARMUP_MAX_RETRIES:
                    logger.info(f"HMDS WARMUP[{reason}]: retrying in {HMDS_WARMUP_RETRY_WAIT_SEC:.0f}s...")
                    self._safe_sleep(HMDS_WARMUP_RETRY_WAIT_SEC, context="HMDS warmup retry")
                    continue
                self._metric_inc('historical_warmup_failures')
                return False

            spy_bars = self._request_historical_bars(
                spy,
                label="SPY",
                duration='5 D',
                bar_size=DAILY_BAR_SIZE,
                what='TRADES',
                use_rth=True,
                timeout=HISTORICAL_DATA_TIMEOUT_SEC,
                metric_prefix='spy',
            )
            if not spy_bars:
                logger.warning(
                    f"HMDS WARMUP[{reason}] attempt {attempt}/{HMDS_WARMUP_MAX_RETRIES}: "
                    "SPY historical failed — IBKR historical farm not healthy yet."
                )
                if attempt < HMDS_WARMUP_MAX_RETRIES:
                    logger.info(f"HMDS WARMUP[{reason}]: retrying in {HMDS_WARMUP_RETRY_WAIT_SEC:.0f}s...")
                    self._safe_sleep(HMDS_WARMUP_RETRY_WAIT_SEC, context="HMDS warmup retry")
                    continue
                self._metric_inc('historical_warmup_failures')
                return False

            if not self._ensure_vix_contract():
                logger.warning(f"HMDS WARMUP[{reason}] attempt {attempt}: VIX contract unavailable.")
                if attempt < HMDS_WARMUP_MAX_RETRIES:
                    logger.info(f"HMDS WARMUP[{reason}]: retrying in {HMDS_WARMUP_RETRY_WAIT_SEC:.0f}s...")
                    self._safe_sleep(HMDS_WARMUP_RETRY_WAIT_SEC, context="HMDS warmup retry")
                    continue
                self._metric_inc('historical_warmup_failures')
                return False

            vix_bars = self._request_historical_bars(
                self._vix_contract,
                label="VIX",
                duration='5 D',
                bar_size=DAILY_BAR_SIZE,
                what='TRADES',
                use_rth=False,
                timeout=HISTORICAL_DATA_TIMEOUT_SEC,
                metric_prefix='vix',
            )
            if not vix_bars:
                logger.warning(
                    f"HMDS WARMUP[{reason}] attempt {attempt}/{HMDS_WARMUP_MAX_RETRIES}: "
                    "SPY OK but VIX historical failed."
                )
                if attempt < HMDS_WARMUP_MAX_RETRIES:
                    logger.info(f"HMDS WARMUP[{reason}]: retrying in {HMDS_WARMUP_RETRY_WAIT_SEC:.0f}s...")
                    self._safe_sleep(HMDS_WARMUP_RETRY_WAIT_SEC, context="HMDS warmup retry")
                    continue
                logger.warning(
                    f"HMDS WARMUP[{reason}]: all {HMDS_WARMUP_MAX_RETRIES} attempts exhausted; "
                    "treating VIX as unavailable until a later retry succeeds."
                )
                self._record_vix_failure("warmup_vix_failed")
                self._metric_inc('historical_warmup_failures')
                return False

            hist_price = self._coerce_positive_price(getattr(vix_bars[-1], 'close', None))
            if hist_price is not None:
                self._record_vix_success(hist_price, source="historical_warmup")
            vix_text = f"{hist_price:.2f}" if hist_price is not None else "unknown"
            attempt_text = f" (attempt {attempt})" if attempt > 1 else ""
            logger.info(
                f"HMDS WARMUP[{reason}]{attempt_text}: SPY and VIX historical data OK "
                f"(VIX={vix_text})."
            )
            self._metric_inc('historical_warmup_successes')
            return True

        self._metric_inc('historical_warmup_failures')
        return False

    def _fresh_market_price(self, sym: str) -> Optional[float]:
        """Fetch a fresh market price from IBKR for exit/risk decisions."""
        snapshot = self._fresh_market_snapshot(sym)
        return snapshot.get('price') if snapshot else None

    def _fresh_market_snapshot(self, sym: str) -> Optional[Dict[str, Optional[float]]]:
        """Fetch a fresh IBKR quote snapshot with fields needed by exit rules."""
        from src.engine_base import logger
        try:
            contract = self._stock_contract(sym)
            tickers = self.ib.reqTickers(contract)
            if not tickers:
                return None
            ticker = tickers[0]
            price = None
            for candidate in (
                ticker.marketPrice(),
                getattr(ticker, 'last', None),
            ):
                price = self._coerce_positive_price(candidate)
                if price is not None:
                    break
            bid = self._coerce_positive_price(getattr(ticker, 'bid', None))
            ask = self._coerce_positive_price(getattr(ticker, 'ask', None))
            if price is None and bid is not None and ask is not None and ask >= bid:
                price = (bid + ask) / 2.0
            if price is None and bid is not None:
                price = bid
            if price is None:
                return None

            day_open = self._coerce_positive_price(getattr(ticker, 'open', None))
            day_high = self._coerce_positive_price(getattr(ticker, 'high', None))
            day_low  = self._coerce_positive_price(getattr(ticker, 'low', None))
            prev_close = self._coerce_positive_price(getattr(ticker, 'close', None))
            vwap     = self._coerce_positive_price(getattr(ticker, 'vwap', None))
            if day_high is not None:
                day_high = max(day_high, price)
            if day_low is not None:
                day_low = min(day_low, price)
            return {
                'price': price,
                'bid': bid,
                'ask': ask,
                'open': day_open,
                'high': day_high,
                'low': day_low,
                'prev_close': prev_close,
                'vwap': vwap,
            }
        except Exception as e:
            logger.warning(f"PRICE {sym}: fresh market snapshot unavailable ({e})")
        return None

    def _fresh_spy_intraday_return(self, today_str: str) -> Optional[float]:
        """Return SPY's intraday return from IBKR, cached per EOD cycle."""
        from src.engine_base import logger
        cached = getattr(self, '_eod_spy_return_cache', None)
        if cached and cached.get('date') == today_str:
            return cached.get('return')

        snapshot = self._fresh_market_snapshot('SPY')
        spy_ret = None
        if snapshot:
            price = snapshot.get('price')
            day_open = snapshot.get('open')
            if price is not None and day_open is not None and day_open > 0:
                spy_ret = (price - day_open) / day_open
        self._eod_spy_return_cache = {'date': today_str, 'return': spy_ret}
        if spy_ret is None:
            logger.warning("EOD QUALITY: SPY intraday return unavailable; hold test will fail closed.")
        return spy_ret

    def _fetch_spy_daily_frame(self) -> Optional[pd.DataFrame]:
        """Return cached SPY daily bars with MA context for regime/RS checks."""
        from src.engine_base import logger, _TZ_NY
        tz_ny = _TZ_NY
        today = datetime.now(tz_ny).strftime('%Y-%m-%d')
        cached = self._spy_cache.get('df') if self._spy_cache.get('date') == today else None
        if cached is not None:
            return cached
        try:
            if 'SPY' not in self._contract_cache:
                self._contract_cache['SPY'] = self._stock_contract('SPY')
            bars = self.ib.reqHistoricalData(
                self._contract_cache['SPY'], '', DAILY_LOOKBACK, DAILY_BAR_SIZE, 'TRADES', True
            )
            if not isinstance(bars, list) or len(bars) < MA_SLOW + SMA200_SLOPE_LOOKBACK:
                logger.warning("SPY context has insufficient history — blocking new entries")
                self._spy_cache = {'date': today, 'trend': False, 'df': None}
                return None
            df = util.df(bars)
            df['MA50'] = df['close'].rolling(50).mean()
            df['MA200'] = df['close'].rolling(200).mean()
            df['SMA200_SLOPE'] = df['MA200'] - df['MA200'].shift(SMA200_SLOPE_LOOKBACK)
            last = df.iloc[-1]
            trend = bool(
                last['close'] > last['MA50']
                and last['MA50'] > last['MA200']
                and last['SMA200_SLOPE'] > 0
            )
            self._spy_cache = {'date': today, 'trend': trend, 'df': df}
            return df
        except Exception as e:
            logger.warning(f"SPY context fetch failed: {e} — blocking new entries")
            self._spy_cache = {'date': today, 'trend': False, 'df': None}
            return None

    def _fetch_spy_trend(self) -> bool:
        """True when SPY price > SMA50 > SMA200 and SMA200 is rising."""
        from src.engine_base import _TZ_NY
        tz_ny  = _TZ_NY
        today  = datetime.now(tz_ny).strftime('%Y-%m-%d')
        if self._spy_cache.get('date') == today:
            return self._spy_cache['trend']
        df = self._fetch_spy_daily_frame()
        return bool(self._spy_cache.get('trend')) if df is not None else False

    @staticmethod
    def _return_from_daily(df: pd.DataFrame, latest_price: float, bars_back: int) -> float:
        if df is None or len(df) <= bars_back or latest_price <= 0:
            return float('nan')
        try:
            ref = float(df['close'].iloc[-bars_back])
        except Exception:
            return float('nan')
        return latest_price / ref - 1 if ref > 0 else float('nan')

    @staticmethod
    def _weekly_uptrend_from_daily(df: pd.DataFrame, latest_price: float) -> bool:
        if df is None or df.empty or latest_price <= 0:
            return False
        try:
            d = df.copy()
            if 'date' in d.columns:
                d = d.set_index(pd.to_datetime(d['date']))
            weekly = d['close'].resample('W-FRI').last().dropna()
            if len(weekly) < 30:
                return False
            ma10w = weekly.rolling(10).mean().iloc[-1]
            ma30w = weekly.rolling(30).mean().iloc[-1]
            ret_13w = latest_price / float(weekly.iloc[-13]) - 1 if len(weekly) > 13 else float('nan')
            return bool(
                latest_price > ma10w
                and ma10w > ma30w
                and np.isfinite(ret_13w)
                and ret_13w > 0
            )
        except Exception:
            return False

    def _build_swing_context(self, df_daily: pd.DataFrame, live_price: float) -> dict:
        spy_df = self._fetch_spy_daily_frame()
        ret_63d = self._return_from_daily(df_daily, live_price, 63)
        ret_126d = self._return_from_daily(df_daily, live_price, 126)
        ret_13w = ret_63d
        ret_26w = ret_126d
        spy_ret_63d = self._return_from_daily(spy_df, float(spy_df['close'].iloc[-1]), 63) if spy_df is not None and not spy_df.empty else float('nan')
        spy_ret_126d = self._return_from_daily(spy_df, float(spy_df['close'].iloc[-1]), 126) if spy_df is not None and not spy_df.empty else float('nan')
        high_52w = float(df_daily['high'].tail(252).max()) if df_daily is not None and len(df_daily) else float('nan')
        return {
            'return_13w': ret_13w,
            'return_26w': ret_26w,
            'relative_strength_63d': (
                ret_63d - spy_ret_63d if np.isfinite(ret_63d) and np.isfinite(spy_ret_63d) else float('nan')
            ),
            'relative_strength_126d': (
                ret_126d - spy_ret_126d if np.isfinite(ret_126d) and np.isfinite(spy_ret_126d) else float('nan')
            ),
            'weekly_uptrend': self._weekly_uptrend_from_daily(df_daily, live_price),
            'high_52w': high_52w,
            'price_vs_52w_high': live_price / high_52w if high_52w > 0 else float('nan'),
        }

    @staticmethod
    def _daily_returns(df: pd.DataFrame) -> pd.Series:
        """Return a date-indexed pct_change series (last 60 rows) for correlation.

        ib_async util.df() gives an integer-indexed DataFrame with a 'date'
        column.  Two DataFrames from separate reqHistoricalData calls have
        independent integer indices, so pd.concat(join='inner') on the default
        index produces zero overlap.  Setting the date column as the index
        lets the inner-join align by calendar date instead.
        """
        s = df.copy()
        if 'date' in s.columns:
            s = s.set_index('date')
        # Normalise to plain date so tz-aware and tz-naive indices align.
        if hasattr(s.index, 'normalize'):
            try:
                s.index = s.index.normalize().date
            except Exception:
                pass
        return s['close'].pct_change().dropna().tail(60)

    def _compute_book_correlation(self, sym: str, df_daily: pd.DataFrame) -> float:
        """Max absolute Pearson correlation of candidate vs current book.

        Correlation is a portfolio risk gate. If the engine cannot evaluate a
        current holding, that pair is treated as maximally correlated (1.0) and
        the candidate is blocked. However, we no longer return early — previous
        behaviour returned 1.0 immediately on the first failing book symbol,
        which silently blocked ALL new entries whenever any held symbol had
        unavailable history (e.g. a stale HMDS connection on one name).

        Changed behaviour:
        - Failures for a specific book_sym are cached in _correlation_book_failures
          so subsequent candidates skip the re-fetch without hammering IBKR again.
        - max_corr is set to 1.0 for the failing pair (still fail-closed for
          THIS candidate), but the loop continues to the next book symbol.
        - Each candidate call starts fresh and is independent.
        """
        from src.engine_base import logger
        if not self.state:
            return 0.0
        cand_ret = self._daily_returns(df_daily)
        max_corr  = 0.0
        failures = getattr(self, '_correlation_book_failures', set())
        for book_sym in self.state:
            if book_sym == sym:
                continue
            if book_sym in failures:
                logger.debug(
                    f"CORRELATION {sym}: skipping {book_sym} "
                    "(history unavailable this cycle; cached failure)"
                )
                max_corr = max(max_corr, 1.0)
                continue
            try:
                cached = self._bar_cache.get(book_sym)
                if cached and 'bars_daily' in cached:
                    book_df = util.df(cached['bars_daily'])
                else:
                    bars = self.ib.reqHistoricalData(
                        self._stock_contract(book_sym), '', '90 D', DAILY_BAR_SIZE, 'TRADES', True
                    )
                    if not isinstance(bars, list) or not bars:
                        logger.warning(
                            f"CORRELATION {sym}: no history for held {book_sym}; "
                            "treating as maximally correlated for this candidate."
                        )
                        failures.add(book_sym)
                        self._correlation_book_failures = failures
                        max_corr = max(max_corr, 1.0)
                        continue
                    book_df = util.df(bars)
                    # Cache the freshly fetched bars so subsequent candidates in this
                    # scan cycle don't re-fetch and get a mismatched integer index.
                    self._bar_cache.setdefault(book_sym, {})['bars_daily'] = bars
                book_ret = self._daily_returns(book_df)
                aligned  = pd.concat([cand_ret, book_ret], axis=1, join='inner').dropna()
                if len(aligned) < 20:
                    logger.warning(
                        f"CORRELATION {sym}: insufficient overlap with held {book_sym}; "
                        "treating as maximally correlated for this candidate."
                    )
                    failures.add(book_sym)
                    self._correlation_book_failures = failures
                    max_corr = max(max_corr, 1.0)
                    continue
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                if not np.isnan(corr):
                    max_corr = max(max_corr, abs(float(corr)))
            except Exception as e:
                logger.warning(
                    f"CORRELATION {sym}: failed against held {book_sym} ({e}); "
                    "treating as maximally correlated for this candidate."
                )
                failures.add(book_sym)
                self._correlation_book_failures = failures
                max_corr = max(max_corr, 1.0)
        return max_corr

    def _daily_indicator_exit_row(self, sym: str) -> Optional[pd.Series]:
        from src.engine_base import logger
        now_ts = time.monotonic()
        cached = self._indicator_row_cache.get(sym)
        if cached is not None:
            ts, row = cached
            if now_ts - ts < 30.0:
                return row
        try:
            contract = self._stock_contract(sym)
            bars = self.ib.reqHistoricalData(
                contract, '', DAILY_LOOKBACK, DAILY_BAR_SIZE, 'TRADES', True
            )
            if not isinstance(bars, list) or len(bars) < MIN_CANDLES:
                result = None
            else:
                df = apply_all(
                    util.df(bars),
                    RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW,
                    SMA200_SLOPE_LOOKBACK, CHANDELIER_PERIOD,
                )
                result = df.iloc[-1]
        except Exception as exc:
            logger.warning(f"DAILY EXIT DATA: {sym} indicator check failed: {exc}")
            result = None
        self._indicator_row_cache[sym] = (now_ts, result)
        return result

    def _get_sector(self, symbol: str, contract) -> str:
        """Return IB industry string for symbol.  Cached; returns 'Unknown' on error."""
        if symbol in self._sector_cache:
            return self._sector_cache[symbol]
        try:
            details = self.ib.reqContractDetails(contract)
            sector  = str(details[0].industry) if isinstance(details, list) and details else 'Unknown'
        except Exception:
            sector = 'Unknown'
        self._sector_cache[symbol] = sector
        return sector
