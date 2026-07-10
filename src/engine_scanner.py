import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    APP_SCANNER_SOURCE, APP_SCANNER_BATCH_SIZE, APP_SCANNER_MAX_SYMBOLS,
    APP_PREFILTER_ENABLED, APP_PREFILTER_START_TIME, APP_PREFILTER_CACHE_FILE,
    APP_PREFILTER_HISTORY_SLEEP_SEC, APP_PREFILTER_PROGRESS_EVERY,
    APP_PREFILTER_STOP_AT_ENTRY_START,
    ENTRY_START,
    DAILY_LOOKBACK, DAILY_BAR_SIZE,
    MIN_CANDLES,
    RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW,
    SMA200_SLOPE_LOOKBACK, CHANDELIER_PERIOD,
    TICKER_BLOCKLIST,
    STRATEGY_PROFILE,
)
from src.indicators import apply_all
from src.scanner import (
    build_momentum_scanner_filter_options,
    build_momentum_scanners,
    load_application_symbol_universe,
)
from src.strategy_profiles import evaluate_entry_rules, get_strategy_profile


class ScannerMixin:

    @staticmethod
    def _normalise_app_scanner_source(source: str) -> str:
        value = str(source or "").strip().lower()
        if value in {"ib", "ibkr", "broker"}:
            return "ibkr"
        if value in {"all", "full", "symbols", "symbol_universe", "universe"}:
            return "universe"
        if value in {"hybrid", "mixed", "both"}:
            return "hybrid"
        return "hybrid"

    @staticmethod
    def _dedupe_symbols(symbols) -> list:
        seen: set = set()
        out: list = []
        for raw in symbols or []:
            sym = str(raw or "").strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                out.append(sym)
        return out

    def _ibkr_scanner_symbols(self) -> list:
        """Dynamic discovery of institutional momentum from IBKR scanner hits."""
        from src.engine_base import logger
        seen: set = set()
        symbols: list = []
        filter_options = build_momentum_scanner_filter_options()
        for sub in build_momentum_scanners():
            try:
                scan = self.ib.reqScannerData(sub, [], filter_options)
                for item in scan:
                    sym = item.contractDetails.contract.symbol
                    if sym not in seen:
                        seen.add(sym)
                        symbols.append(sym)
            except Exception as e:
                logger.warning(
                    f"SCAN: IB scanner ({sub.scanCode}) failed ({e}); skipping"
                )
        if not symbols:
            logger.warning("SCAN: all scanner queries failed or returned no candidates")
        return symbols

    def _application_universe_symbols(self) -> list:
        from src.engine_base import logger, _TZ_NY
        import pytz
        tz_ny = _TZ_NY
        today = datetime.now(tz_ny).strftime('%Y-%m-%d')
        cached_day = getattr(self, '_scanner_universe_date', None)
        cached_symbols = getattr(self, '_scanner_universe_symbols', None)
        if cached_day != today or not cached_symbols:
            symbols = load_application_symbol_universe()
            max_symbols = int(APP_SCANNER_MAX_SYMBOLS or 0)
            if max_symbols > 0:
                symbols = symbols[:max_symbols]
            self._scanner_universe_symbols = self._dedupe_symbols(symbols)
            self._scanner_universe_offset = 0
            self._scanner_universe_date = today
            logger.info(
                f"SCAN SOURCE: loaded application universe "
                f"({len(self._scanner_universe_symbols)} symbols)"
            )
        return list(getattr(self, '_scanner_universe_symbols', []) or [])

    def _next_universe_batch(self) -> list:
        from src.engine_base import logger
        symbols = self._application_universe_symbols()
        if not symbols:
            return []
        batch_size = int(APP_SCANNER_BATCH_SIZE or 0)
        if batch_size <= 0 or batch_size >= len(symbols):
            self._scanner_universe_offset = 0
            return symbols

        start = int(getattr(self, '_scanner_universe_offset', 0) or 0) % len(symbols)
        end = start + batch_size
        if end <= len(symbols):
            batch = symbols[start:end]
        else:
            batch = symbols[start:] + symbols[: end - len(symbols)]
        self._scanner_universe_offset = end % len(symbols)
        logger.info(
            f"SCAN SOURCE: universe batch {len(batch)}/{len(symbols)} "
            f"offset={start}->{self._scanner_universe_offset}"
        )
        return batch

    def get_institutional_scan(self):
        """Return app scanner candidates before local profile rules screen them."""
        from src.engine_base import logger
        source = self._normalise_app_scanner_source(APP_SCANNER_SOURCE)
        ibkr_symbols: list = []
        universe_symbols: list = []
        prefiltered = (
            self._prefilter_candidates_for_today()
            if source in {"universe", "hybrid"} else None
        )

        if source in {"ibkr", "hybrid"} and prefiltered is None:
            ibkr_symbols = self._ibkr_scanner_symbols()

        if source in {"universe", "hybrid"}:
            if prefiltered is not None:
                universe_symbols = prefiltered
                logger.info(
                    f"SCAN SOURCE: using premarket prefiltered universe "
                    f"({len(universe_symbols)} candidates)"
                )
            else:
                try:
                    universe_symbols = self._next_universe_batch()
                except Exception as e:
                    logger.warning(f"SCAN SOURCE: application universe unavailable ({e})")
                    if source == "universe":
                        return []

        symbols = self._dedupe_symbols(ibkr_symbols + universe_symbols)
        logger.info(
            f"SCAN SOURCE: mode={source} ibkr={len(ibkr_symbols)} "
            f"universe_batch={len(universe_symbols)} combined={len(symbols)}"
        )
        return symbols

    def _remember_daily_scan_skip(self, symbol: str, reason: str):
        """Cache only stable same-day scan failures to reduce IBKR pacing load."""
        if not hasattr(self, '_daily_scan_skip') or self._daily_scan_skip is None:
            self._daily_scan_skip = {}
        self._daily_scan_skip[symbol] = reason

    def _read_prefilter_cache(self, today: str, profile_name: str) -> Optional[dict]:
        try:
            with open(APP_PREFILTER_CACHE_FILE, "r") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("date") != today or payload.get("profile") != profile_name:
            return None
        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list):
            return None
        self._prefilter_date = today
        self._prefilter_status = str(payload.get("status") or "complete")
        self._prefilter_candidates = self._dedupe_symbols(candidates)
        self._prefilter_stats = dict(payload.get("stats") or {})
        if self._prefilter_status == "complete":
            self._last_premarket_prefilter_date = today
        return payload

    def _write_prefilter_cache(self, payload: dict) -> None:
        os.makedirs(os.path.dirname(APP_PREFILTER_CACHE_FILE) or ".", exist_ok=True)
        tmp = f"{APP_PREFILTER_CACHE_FILE}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, APP_PREFILTER_CACHE_FILE)

    def _prefilter_candidates_for_today(self) -> Optional[list]:
        if not APP_PREFILTER_ENABLED:
            return None
        from src.engine_base import _TZ_NY
        today = datetime.now(_TZ_NY).strftime('%Y-%m-%d')
        profile = getattr(self, "_strategy_profile", get_strategy_profile(STRATEGY_PROFILE))
        if getattr(self, '_prefilter_date', None) == today and getattr(self, '_prefilter_status', None):
            status = getattr(self, '_prefilter_status', 'not_started')
            if status in {"complete", "partial"}:
                return list(getattr(self, '_prefilter_candidates', []) or [])
        payload = self._read_prefilter_cache(today, profile.name)
        if payload and str(payload.get("status")) in {"complete", "partial"}:
            return list(getattr(self, '_prefilter_candidates', []) or [])
        return None

    def _enrich_prefilter_daily_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        fixed = apply_all(
            df, RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW,
            SMA200_SLOPE_LOOKBACK, CHANDELIER_PERIOD
        )
        fixed['MA10'] = fixed['close'].rolling(10).mean()
        fixed['MA20'] = fixed['close'].rolling(20).mean()
        fixed['HIGH20'] = fixed['high'].rolling(20).max()
        fixed['HIGH50'] = fixed['high'].rolling(50).max()
        return fixed

    @staticmethod
    def _weekly_structure_from_daily(df: pd.DataFrame) -> dict:
        try:
            d = df.copy()
            if 'date' in d.columns:
                d = d.set_index(pd.to_datetime(d['date']))
            weekly = d['close'].resample('W-FRI').last().dropna()
            if len(weekly) < 30:
                return {'weekly_ma10_gt_ma30': False}
            ma10w = weekly.rolling(10).mean().iloc[-1]
            ma30w = weekly.rolling(30).mean().iloc[-1]
            return {'weekly_ma10_gt_ma30': bool(ma10w > ma30w)}
        except Exception:
            return {'weekly_ma10_gt_ma30': False}

    def _build_prefilter_context(self, df: pd.DataFrame) -> dict:
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        price = float(last['close'])
        ma20 = float(last['MA20']) if not pd.isna(last['MA20']) else float('nan')
        ma50 = float(last['MA50']) if not pd.isna(last['MA50']) else float('nan')
        ma200 = float(last['MA200']) if not pd.isna(last['MA200']) else float('nan')
        high20 = float(last['HIGH20']) if not pd.isna(last['HIGH20']) else float('nan')
        high50 = float(last['HIGH50']) if not pd.isna(last['HIGH50']) else float('nan')
        atr_chandelier = float(last['ATR_CHAND']) if not pd.isna(last['ATR_CHAND']) else float('nan')
        avg_20d_vol = float(df['volume'].tail(20).mean())
        dollar_vol_20d = float((df['close'] * df['volume']).tail(20).mean())
        ctx = {
            'price': price,
            'close': price,
            'live_price': price,
            'ma10': float(last['MA10']) if not pd.isna(last['MA10']) else float('nan'),
            'ma20': ma20,
            'ma50': ma50,
            'ma200': ma200,
            'atr': float(last['ATR']) if not pd.isna(last['ATR']) else float('nan'),
            'atr5': float(last['ATR5']) if not pd.isna(last['ATR5']) else float('nan'),
            'atr20': float(last['ATR20']) if not pd.isna(last['ATR20']) else float('nan'),
            'atr_chandelier': atr_chandelier,
            'atr_pct': atr_chandelier / price if price > 0 and np.isfinite(atr_chandelier) else float('nan'),
            'sma200_slope': float(last['SMA200_SLOPE']) if not pd.isna(last['SMA200_SLOPE']) else float('nan'),
            'high10': float(last['HIGH10']) if not pd.isna(last['HIGH10']) else float('nan'),
            'high20': high20,
            'high50': high50,
            'dist_high20': price / high20 - 1 if high20 > 0 else float('nan'),
            'dist_high50': price / high50 - 1 if high50 > 0 else float('nan'),
            'rsi': float(last['RSI']) if not pd.isna(last['RSI']) else float('nan'),
            'rsi_prev': float(prev['RSI']) if not pd.isna(prev['RSI']) else float('nan'),
            'prev_close': float(prev['close']) if not pd.isna(prev['close']) else price,
            'prev_daily_high': float(prev['high']) if not pd.isna(prev['high']) else float('nan'),
            'prev_high': float(prev['high']) if not pd.isna(prev['high']) else float('nan'),
            'spread_pct': 0.0,
            'rvol': 0.0,
            'volume_pace': 0.0,
            'volume': int(last['volume']) if not pd.isna(last['volume']) else 0,
            'avg_20d_volume': avg_20d_vol,
            'dollar_vol_20d': dollar_vol_20d,
            'macd_hist': float(last['MACD_HIST']) if not pd.isna(last['MACD_HIST']) else float('nan'),
            'macd_hist_delta': (
                float(last['MACD_HIST'] - prev['MACD_HIST'])
                if not pd.isna(last['MACD_HIST']) and not pd.isna(prev['MACD_HIST']) else float('nan')
            ),
            'obv_slope_5': float(last['OBV_SLOPE_5']) if not pd.isna(last['OBV_SLOPE_5']) else float('nan'),
            'obv_uptrend': bool(last.get('OBV_UPTREND', False)),
            'obv_bull_divergence': bool(last.get('OBV_BULL_DIVERGENCE', False)),
            'obv_bear_divergence': bool(last.get('OBV_BEAR_DIVERGENCE', False)),
            'ema20_gt_sma50': bool(last.get('EMA20_GT_SMA50', False)),
            'ma_bull_cross': bool(last.get('MA_BULL_CROSS', False)),
            'ma_bear_cross': bool(last.get('MA_BEAR_CROSS', False)),
            'bb_below_lower_2': bool(last.get('BB_BELOW_LOWER_2', False)),
            'bb_above_upper_2': bool(last.get('BB_ABOVE_UPPER_2', False)),
            'bb_reclaim_lower': bool(last.get('BB_RECLAIM_LOWER', False)),
            'psar_bull_3': bool(last.get('PSAR_BULL_3', False)),
            'psar_bear_3': bool(last.get('PSAR_BEAR_3', False)),
            'stoch_k': float(last['STOCH_K']) if not pd.isna(last['STOCH_K']) else float('nan'),
            'stoch_d': float(last['STOCH_D']) if not pd.isna(last['STOCH_D']) else float('nan'),
            'stoch_bull_exit_oversold': bool(last.get('STOCH_BULL_EXIT_OVERSOLD', False)),
            'stoch_bear_exit_overbought': bool(last.get('STOCH_BEAR_EXIT_OVERBOUGHT', False)),
            'macd_bull_divergence': bool(last.get('MACD_BULL_DIVERGENCE', False)),
            'macd_bear_divergence': bool(last.get('MACD_BEAR_DIVERGENCE', False)),
            'reclaim_ma20': False,
            'reclaim_ma50': False,
            'break_prev_high': False,
            'df_daily': df,
        }
        ctx.update(self._weekly_structure_from_daily(df))
        ctx.update(self._build_swing_context(df, price))
        return ctx

    @staticmethod
    def _prefilter_static_failures(ctx: dict, profile) -> tuple[str, ...]:
        failures: list[str] = []

        def finite(name: str) -> float:
            try:
                value = float(ctx.get(name))
            except (TypeError, ValueError):
                return float('nan')
            return value if np.isfinite(value) else float('nan')

        volume = finite('volume')
        dollar_vol = finite('dollar_vol_20d')
        rsi = finite('rsi')
        rsi_prev = finite('rsi_prev')
        ma50 = finite('ma50')
        ma200 = finite('ma200')
        sma200_slope = finite('sma200_slope')
        macd_hist = finite('macd_hist')
        macd_delta = finite('macd_hist_delta')
        obv_slope = finite('obv_slope_5')

        if profile.min_volume and (not np.isfinite(volume) or volume < float(profile.min_volume)):
            failures.append(f"historical_volume<{float(profile.min_volume):.0f}")
        if profile.min_dollar_vol and (
            not np.isfinite(dollar_vol) or dollar_vol < float(profile.min_dollar_vol)
        ):
            failures.append(f"historical_dollar_vol<{float(profile.min_dollar_vol):.0f}")
        if profile.require_ma50_above_ma200 and (
            not np.isfinite(ma50) or not np.isfinite(ma200) or ma50 <= ma200
        ):
            failures.append("MA50<=MA200")
        if profile.require_sma200_slope_positive and (
            not np.isfinite(sma200_slope) or sma200_slope <= 0
        ):
            failures.append("SMA200_slope<=0")
        sleeves = profile.indicator_sleeves or ()
        ma_trend_active = bool(ctx.get('ema20_gt_sma50')) or bool(ctx.get('ma_bull_cross'))
        stoch_bull = bool(ctx.get('stoch_bull_exit_oversold'))
        macd_bull = bool(ctx.get('macd_bull_divergence')) or (
            np.isfinite(macd_delta) and macd_delta > 0
        )
        obv_bull = bool(ctx.get('obv_uptrend')) or bool(ctx.get('obv_bull_divergence'))
        psar_confirm = bool(ctx.get('psar_bull_3'))
        rsi_momentum = np.isfinite(rsi) and rsi >= 50.0
        rsi_recovery_possible = (
            "bollinger_reversion" in sleeves
            and np.isfinite(rsi)
            and np.isfinite(rsi_prev)
            and rsi >= 40.0
            and rsi > rsi_prev
        )
        # Volume pace no longer counts as a confirmation (it is a hard gate),
        # so the requirement is two of the four static daily-bar indicators.
        # These are computed from completed bars and cannot change intraday,
        # which makes this check fully decidable premarket.
        static_confirmations = sum(bool(v) for v in (stoch_bull, macd_bull, obv_bull, psar_confirm))

        possible_sleeves: list[str] = []
        if "ma_cross" in sleeves and ma_trend_active and (macd_bull or obv_bull):
            possible_sleeves.append("ma_cross")
        if "bollinger_reversion" in sleeves and bool(ctx.get('bb_reclaim_lower')):
            possible_sleeves.append("bollinger_reversion")
        if "psar_flip" in sleeves and bool(ctx.get('psar_bull_3')):
            possible_sleeves.append("psar_flip")

        if profile.require_weekly_uptrend and not bool(ctx.get('weekly_ma10_gt_ma30')):
            failures.append("weekly_MA10<=MA30")
        if not possible_sleeves:
            failures.append("no_possible_indicator_sleeve")
        if not (rsi_momentum or rsi_recovery_possible):
            failures.append("daily_RSI_cannot_confirm")
        if static_confirmations < 2:
            failures.append("static_indicator_confirmations<2")

        return tuple(failures)

    def _prefilter_symbol(self, symbol: str, profile, today: str) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
        from ib_async import util
        from src.engine_base import completed_daily_bars
        if symbol in TICKER_BLOCKLIST:
            return False, ("blocklisted",), ()
        try:
            contract = self._stock_contract(symbol)
            bars_daily = self.ib.reqHistoricalData(
                contract, '', DAILY_LOOKBACK, DAILY_BAR_SIZE, 'TRADES', True
            )
            # Premarket runs get completed bars anyway; this protects manual
            # intraday prefilter runs from today's partial daily bar.
            bars_daily = completed_daily_bars(bars_daily, today)
            if not isinstance(bars_daily, list) or len(bars_daily) < MIN_CANDLES:
                return False, (f"insufficient_daily_history<{MIN_CANDLES}",), ()
            df = util.df(bars_daily)
            if df is None or len(df) < MIN_CANDLES:
                return False, (f"insufficient_daily_history<{MIN_CANDLES}",), ()
            df = self._enrich_prefilter_daily_frame(df)
            if np.isnan(float(df['MA200'].iloc[-1])):
                return False, ("invalid_MA200",), ()
            ctx = self._build_prefilter_context(df)
            static_failures = self._prefilter_static_failures(ctx, profile)
            evaluation = evaluate_entry_rules(ctx, profile)
            deferred = tuple(
                failure for failure in evaluation.failed
                if failure not in static_failures
            )
            if static_failures:
                return False, static_failures, deferred

            self._bar_cache[symbol] = {
                'date': today,
                'bars_daily': bars_daily,
            }
            return True, (), deferred
        except Exception as e:
            return False, (f"historical_fetch_failed:{type(e).__name__}",), ()

    def _run_premarket_universe_prefilter(self) -> dict:
        """Scan the full application universe once and cache daily-pass candidates."""
        from src.engine_base import logger, _TZ_NY
        now_ny = datetime.now(_TZ_NY)
        today = now_ny.strftime('%Y-%m-%d')
        profile = getattr(self, "_strategy_profile", get_strategy_profile(STRATEGY_PROFILE))

        cached = self._read_prefilter_cache(today, profile.name)
        if cached and cached.get("status") == "complete":
            logger.info(
                f"PREFILTER: using cached {today} universe sieve "
                f"({len(self._prefilter_candidates)} candidates)"
            )
            return cached

        universe = load_application_symbol_universe()
        max_symbols = int(APP_SCANNER_MAX_SYMBOLS or 0)
        if max_symbols > 0:
            universe = universe[:max_symbols]
        universe = self._dedupe_symbols(universe)
        processed: set[str] = set()
        candidates: list[str] = []
        deferred_by_symbol: dict[str, list[str]] = {}
        reject_reasons: dict[str, int] = {}
        rejections_by_symbol: dict[str, list[str]] = {}
        processed_count = 0

        if cached:
            processed = set(str(s).upper() for s in cached.get("processed_symbols", []) or [])
            candidates = self._dedupe_symbols(cached.get("candidates", []) or [])
            deferred_by_symbol = dict(cached.get("deferred_rules", {}) or {})
            reject_reasons = dict(cached.get("reject_reasons", {}) or {})
            rejections_by_symbol = dict(cached.get("rejections_by_symbol", {}) or {})
            processed_count = len(processed)

        self._metric_inc('prefilter_runs')
        self._prefilter_date = today
        self._prefilter_status = "running"
        logger.info(
            f"PREFILTER: starting full-universe historical sieve at {now_ny.strftime('%H:%M:%S %Z')} "
            f"profile={profile.name} universe={len(universe)} already_done={len(processed)}"
        )

        progress_every = max(1, int(APP_PREFILTER_PROGRESS_EVERY or 100))

        def write_checkpoint(status: str, stopped_reason: Optional[str] = None) -> dict:
            self._prefilter_candidates = self._dedupe_symbols(candidates)
            self._prefilter_status = status
            self._prefilter_stats = {
                'processed': processed_count,
                'universe': len(universe),
                'candidates': len(self._prefilter_candidates),
                'rejected': max(0, processed_count - len(self._prefilter_candidates)),
                'reject_reasons': reject_reasons,
            }
            payload = {
                'date': today,
                'generated_at': datetime.now(_TZ_NY).isoformat(),
                'profile': profile.name,
                'status': status,
                'source': 'application_universe_historical_prefilter',
                'universe': len(universe),
                'processed_symbols': sorted(processed),
                'candidates': self._prefilter_candidates,
                'deferred_rules': deferred_by_symbol,
                'reject_reasons': reject_reasons,
                'rejections_by_symbol': rejections_by_symbol,
                'stats': self._prefilter_stats,
            }
            if stopped_reason:
                payload['stopped_reason'] = stopped_reason
                payload['stopped_at'] = payload['generated_at']
                self._prefilter_stats['stopped_reason'] = stopped_reason
            self._write_prefilter_cache(payload)
            self._write_dashboard_data(connected=True)
            return payload

        for sym in universe:
            if sym in processed:
                continue
            passed, failures, deferred = self._prefilter_symbol(sym, profile, today)
            processed.add(sym)
            processed_count += 1
            self._metric_inc('prefilter_processed')
            if passed:
                candidates.append(sym)
                deferred_by_symbol[sym] = list(deferred)
                rejections_by_symbol.pop(sym, None)
            else:
                self._metric_inc('prefilter_rejected')
                reason_key = failures[0] if failures else "unknown"
                reject_reasons[reason_key] = int(reject_reasons.get(reason_key, 0)) + 1
                rejections_by_symbol[sym] = list(failures or ("unknown",))

            if processed_count % progress_every == 0:
                write_checkpoint('partial')
                logger.info(
                    f"PREFILTER: progress {processed_count}/{len(universe)} "
                    f"candidates={len(self._prefilter_candidates)}"
                )

            cutoff_now = datetime.now(_TZ_NY)
            if (APP_PREFILTER_STOP_AT_ENTRY_START
                    and cutoff_now.weekday() < 5
                    and (cutoff_now.hour, cutoff_now.minute) >= ENTRY_START):
                payload = write_checkpoint('partial', stopped_reason='entry_window_open')
                self._last_premarket_prefilter_date = today
                logger.warning(
                    f"PREFILTER: stopped at entry window "
                    f"{ENTRY_START[0]:02d}:{ENTRY_START[1]:02d} ET after "
                    f"{processed_count}/{len(universe)} symbols; using "
                    f"{len(self._prefilter_candidates)} cached candidates today."
                )
                return payload

            sleep_s = max(0.0, float(APP_PREFILTER_HISTORY_SLEEP_SEC or 0.0))
            if sleep_s > 0:
                self.ib.sleep(sleep_s)

        payload = write_checkpoint('complete')
        self._last_premarket_prefilter_date = today
        self._metric_inc('prefilter_candidates', len(self._prefilter_candidates))
        logger.info(
            f"PREFILTER COMPLETE: processed={processed_count}/{len(universe)} "
            f"candidates={len(self._prefilter_candidates)} "
            f"rejected={self._prefilter_stats['rejected']}"
        )
        return payload

    def get_technical_context(self, symbol):
        from src.engine_base import logger, _TZ_NY
        from ib_async import Stock, util
        import pytz
        # Contract cache — avoids re-qualifying the same symbol every 60-second cycle
        if symbol not in self._contract_cache:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            self._contract_cache[symbol] = contract
        contract = self._contract_cache[symbol]

        # Bar cache — daily bars are valid for one trading day; re-fetch on date change.
        from src.engine_base import _TZ_NY
        tz_ny      = _TZ_NY
        now_ny     = datetime.now(tz_ny)
        today_str  = now_ny.strftime('%Y-%m-%d')

        cached = self._bar_cache.get(symbol)
        bars_daily = cached.get('bars_daily') if cached and cached.get('date') == today_str else None

        if not bars_daily:
            # Daily context (trends, ATR, RSI) — prior completed daily bars.
            # During the regular session IBKR appends today's partial bar;
            # strip it so signals always see completed bars only, matching the
            # premarket prefilter and the backtester.
            from src.engine_base import completed_daily_bars
            bars_daily = self.ib.reqHistoricalData(
                contract, '', DAILY_LOOKBACK, DAILY_BAR_SIZE, 'TRADES', True
            )
            bars_daily = completed_daily_bars(bars_daily, today_str)

        if bars_daily:
            self._bar_cache[symbol] = {
                'date':       today_str,
                'bars_daily': bars_daily,
            }

        orb_high = float('nan')

        if not bars_daily:
            return None
        df = util.df(bars_daily)
        if len(df) < MIN_CANDLES:
            logger.warning(
                f"SCAN {symbol}: only {len(df)} daily bars — need {MIN_CANDLES}, skipping"
            )
            self._remember_daily_scan_skip(symbol, f"insufficient daily history (<{MIN_CANDLES})")
            return None
        df = apply_all(df, RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW, SMA200_SLOPE_LOOKBACK, CHANDELIER_PERIOD)
        if np.isnan(float(df['MA200'].iloc[-1])):
            logger.warning(f"SCAN {symbol}: MA200 is NaN (data gaps in history), skipping")
            self._remember_daily_scan_skip(symbol, "invalid daily MA200")
            return None

        # Extra profile inputs used by the plug-and-play entry sleeves.  Daily
        # bars here are prior completed sessions; today's live price is applied
        # below for reclaims and distance-to-level checks.
        df['MA10'] = df['close'].rolling(10).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['HIGH20'] = df['high'].rolling(20).max()
        df['HIGH50'] = df['high'].rolling(50).max()
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df['MACD_HIST'] = macd - macd_signal
        signed_volume = np.where(df['close'].diff() >= 0, df['volume'], -df['volume'])
        df['OBV'] = pd.Series(signed_volume, index=df.index).cumsum()
        df['OBV_SLOPE_5'] = df['OBV'] - df['OBV'].shift(5)

        # 20-day average dollar volume — used to block low-liquidity pumps
        avg_20d_vol    = float(df['volume'].tail(20).mean())
        dollar_vol_20d = float((df['close'] * df['volume']).tail(20).mean())

        # 3. Live price + bid-ask spread + intraday relative volume
        tickers = self.ib.reqTickers(contract)
        if not tickers:
            logger.warning(f"SCAN {symbol}: live ticker unavailable, skipping")
            return None
        ticker = tickers[0]
        live_price = (
            self._coerce_positive_price(ticker.marketPrice())
            or self._coerce_positive_price(getattr(ticker, 'last', None))
            or self._coerce_positive_price(getattr(ticker, 'close', None))
        )
        if live_price is None:
            logger.warning(f"SCAN {symbol}: live price unavailable, skipping")
            return None
        day_open = (
            self._coerce_positive_price(getattr(ticker, 'open', None))
            or live_price
        )

        bid = float(ticker.bid) if not pd.isna(ticker.bid) else 0.0
        ask = float(ticker.ask) if not pd.isna(ticker.ask) else 0.0
        if bid > 0 and ask > bid:
            spread_pct = (ask - bid) / ((bid + ask) / 2)
        else:
            spread_pct = float('inf')   # unavailable → fail-closed

        intraday_vol = float(ticker.volume) if not pd.isna(ticker.volume) else 0.0
        raw_rvol = intraday_vol / avg_20d_vol if avg_20d_vol > 0 else 0.0
        from src.scoring import volume_pace_from_intraday
        volume_pace = volume_pace_from_intraday(intraday_vol, avg_20d_vol, now_ny)
        intraday_gain = (live_price - day_open) / day_open
        try:
            day_high = float(getattr(ticker, 'high', np.nan))
        except (TypeError, ValueError):
            day_high = np.nan
        try:
            day_low = float(getattr(ticker, 'low', np.nan))
        except (TypeError, ValueError):
            day_low = np.nan
        if not pd.isna(day_high) and not pd.isna(day_low) and day_high > 0 and day_low > 0:
            day_high = max(day_high, live_price)
            day_low  = min(day_low, live_price)
            day_range_location = (
                (live_price - day_low) / (day_high - day_low)
                if day_high > day_low else None
            )
        else:
            day_range_location = None

        ma10 = float(df['MA10'].iloc[-1]) if not pd.isna(df['MA10'].iloc[-1]) else float('nan')
        ma20 = float(df['MA20'].iloc[-1]) if not pd.isna(df['MA20'].iloc[-1]) else float('nan')
        ma50 = float(df['MA50'].iloc[-1])
        ma200 = float(df['MA200'].iloc[-1])
        prev_close = float(df['close'].iloc[-1])
        prev_daily_high = float(df['high'].iloc[-1])
        high20 = float(df['HIGH20'].iloc[-1]) if not pd.isna(df['HIGH20'].iloc[-1]) else float('nan')
        high50 = float(df['HIGH50'].iloc[-1]) if not pd.isna(df['HIGH50'].iloc[-1]) else float('nan')
        macd_hist = float(df['MACD_HIST'].iloc[-1]) if not pd.isna(df['MACD_HIST'].iloc[-1]) else float('nan')
        macd_hist_prev = float(df['MACD_HIST'].iloc[-2]) if len(df) > 1 and not pd.isna(df['MACD_HIST'].iloc[-2]) else float('nan')
        obv_slope_5 = float(df['OBV_SLOPE_5'].iloc[-1]) if not pd.isna(df['OBV_SLOPE_5'].iloc[-1]) else float('nan')
        atr_chandelier = float(df['ATR_CHAND'].iloc[-1])
        swing_context = self._build_swing_context(df, live_price)

        ctx = {
            'orb_high':         orb_high,
            'day_open':         day_open,
            'day_high':         day_high if not pd.isna(day_high) else None,
            'day_low':          day_low if not pd.isna(day_low) else None,
            'ma10':             ma10,
            'ma20':             ma20,
            'ma50':             ma50,
            'ma200':            ma200,
            'atr':              float(df['ATR'].iloc[-1]),
            'atr5':             float(df['ATR5'].iloc[-1]),
            'atr20':            float(df['ATR20'].iloc[-1]),
            'atr_chandelier':   atr_chandelier,
            'atr_pct':          atr_chandelier / live_price if live_price > 0 else float('nan'),
            'sma200_slope':     float(df['SMA200_SLOPE'].iloc[-1]),
            'high10':           float(df['HIGH10'].iloc[-1]),
            'high20':           high20,
            'high50':           high50,
            'dist_high20':      live_price / high20 - 1 if high20 > 0 else float('nan'),
            'dist_high50':      live_price / high50 - 1 if high50 > 0 else float('nan'),
            'rsi':              float(df['RSI'].iloc[-1]),
            'rsi_prev':         float(df['RSI'].iloc[-2]),
            'close':            prev_close,
            'prev_close':       prev_close,
            'prev_daily_high':  prev_daily_high,
            'prev_high':        prev_daily_high,
            'live_price':       live_price,
            'bid':              bid,
            'ask':              ask,
            'spread_pct':       spread_pct,
            # Live ranking uses time-normalized volume pace.  Raw intraday
            # RVOL stays available for diagnostics, but early-session ranking
            # should not punish a stock just because only part of the day has
            # elapsed.
            'rvol':             volume_pace,
            'rvol_raw':         raw_rvol,
            'volume_pace':      volume_pace,
            'intraday_volume':  intraday_vol,
            'avg_20d_volume':   avg_20d_vol,
            'day_range_location': day_range_location,
            'intraday_gain':    intraday_gain,
            'macd_hist':        macd_hist,
            'macd_hist_prev':   macd_hist_prev,
            'macd_hist_delta':  macd_hist - macd_hist_prev if not pd.isna(macd_hist) and not pd.isna(macd_hist_prev) else float('nan'),
            'obv_slope_5':      obv_slope_5,
            'obv_uptrend':      bool(df.get('OBV_UPTREND', pd.Series(False, index=df.index)).iloc[-1]),
            'obv_bull_divergence': bool(df.get('OBV_BULL_DIVERGENCE', pd.Series(False, index=df.index)).iloc[-1]),
            'obv_bear_divergence': bool(df.get('OBV_BEAR_DIVERGENCE', pd.Series(False, index=df.index)).iloc[-1]),
            'ema20_gt_sma50':   bool(df.get('EMA20_GT_SMA50', pd.Series(False, index=df.index)).iloc[-1]),
            'ma_bull_cross':    bool(df.get('MA_BULL_CROSS', pd.Series(False, index=df.index)).iloc[-1]),
            'ma_bear_cross':    bool(df.get('MA_BEAR_CROSS', pd.Series(False, index=df.index)).iloc[-1]),
            'bb_below_lower_2': bool(df.get('BB_BELOW_LOWER_2', pd.Series(False, index=df.index)).iloc[-1]),
            'bb_above_upper_2': bool(df.get('BB_ABOVE_UPPER_2', pd.Series(False, index=df.index)).iloc[-1]),
            'bb_reclaim_lower': bool(df.get('BB_RECLAIM_LOWER', pd.Series(False, index=df.index)).iloc[-1]),
            'psar_bull_3':      bool(df.get('PSAR_BULL_3', pd.Series(False, index=df.index)).iloc[-1]),
            'psar_bear_3':      bool(df.get('PSAR_BEAR_3', pd.Series(False, index=df.index)).iloc[-1]),
            'stoch_k':          float(df.get('STOCH_K', pd.Series(float('nan'), index=df.index)).iloc[-1]),
            'stoch_d':          float(df.get('STOCH_D', pd.Series(float('nan'), index=df.index)).iloc[-1]),
            'stoch_bull_exit_oversold': bool(df.get('STOCH_BULL_EXIT_OVERSOLD', pd.Series(False, index=df.index)).iloc[-1]),
            'stoch_bear_exit_overbought': bool(df.get('STOCH_BEAR_EXIT_OVERBOUGHT', pd.Series(False, index=df.index)).iloc[-1]),
            'macd_bull_divergence': bool(df.get('MACD_BULL_DIVERGENCE', pd.Series(False, index=df.index)).iloc[-1]),
            'macd_bear_divergence': bool(df.get('MACD_BEAR_DIVERGENCE', pd.Series(False, index=df.index)).iloc[-1]),
            'reclaim_ma20':     ma20 > 0 and prev_close <= ma20 and live_price > ma20,
            'reclaim_ma50':     ma50 > 0 and prev_close <= ma50 and live_price > ma50,
            'break_prev_high':  prev_daily_high > 0 and live_price > prev_daily_high,
            # Last COMPLETED daily bar volume (yesterday during live sessions)
            # — same measure the premarket prefilter validated, so the
            # volume>=min gate stays consistent all day.  Intraday activity is
            # judged by volume_pace, never by this field.
            'volume':           int(df['volume'].iloc[-1]),
            'dollar_vol_20d':   dollar_vol_20d,
            'price_fetched_at': datetime.now(tz_ny),
            'contract':         contract,
            'df_daily':         df,
        }
        ctx.update(swing_context)
        ctx.update(self._analyst_context(symbol))
        return ctx
