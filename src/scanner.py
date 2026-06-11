import csv
import json
import os
import re
import time
import urllib.request

from ib_async import ScannerSubscription, TagValue

from src.config import (
    APP_SCANNER_UNIVERSE_CACHE_FILE,
    APP_SCANNER_UNIVERSE_FILE,
    APP_SCANNER_UNIVERSE_TTL_SEC,
    IB_SCANNER_CHANGE_OPEN_PCT_ABOVE,
    IB_SCANNER_FILTERS_ENABLED,
    IB_SCANNER_LAST_VS_EMA20_PCT_ABOVE,
    IB_SCANNER_LAST_VS_EMA50_PCT_ABOVE,
    IB_SCANNER_MACD_HISTOGRAM_ABOVE,
    IB_SCANNER_OPEN_GAP_PCT_BELOW,
    IB_SCANNER_SCAN_CODES,
    IB_SCANNER_LOCATION_CODE,
    IB_SCANNER_ROWS,
    SCAN_MIN_PRICE,
    SCAN_MIN_VOLUME,
    SCAN_MIN_MKTCAP,
    STRATEGY_PROFILE,
)
from src.strategy_profiles import get_strategy_profile

_NASDAQ_LISTED_URL = 'https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt'
_OTHER_LISTED_URL = 'https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt'
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


def _scanner_filter_value(value: float) -> str:
    """Format numeric scanner filter values the way IBKR expects TagValue text."""
    return f"{float(value):.6f}".rstrip("0").rstrip(".") or "0"


def _active_profile():
    return get_strategy_profile(STRATEGY_PROFILE)


def _env_set(name: str) -> bool:
    return name in os.environ and os.environ.get(name, "").strip() != ""


def _is_common_equity_listing(symbol: str, security_name: str) -> bool:
    """Approximate IBKR stockTypeFilter='CORP' for a local symbol universe."""
    symbol = str(symbol or "").strip().upper()
    name = str(security_name or "").strip()
    if not symbol or len(symbol) > 5:
        return False
    if re.search(r"[\^+$\.]", symbol):
        return False
    if _NON_COMMON_NAME_RE.search(name):
        return False
    if symbol[-1:] in {"W", "R", "U"} and not _COMMON_EQUITY_NAME_RE.search(name):
        return False
    return True


def _normalise_symbol_list(symbols) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols or []:
        sym = str(raw or "").strip().upper()
        if not sym or sym in seen or not _is_common_equity_listing(sym, "Common Stock"):
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _read_symbol_file(path: str) -> list[str]:
    with open(path, "r", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        if "," in first_line:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                symbol_field = next(
                    (
                        name for name in reader.fieldnames
                        if str(name or "").strip().lower() in {"symbol", "ticker", "act symbol"}
                    ),
                    reader.fieldnames[0],
                )
                return _normalise_symbol_list(row.get(symbol_field) for row in reader)

        symbols = []
        for line in f:
            token = re.split(r"[\s,|]+", line.strip())[0] if line.strip() else ""
            if token and token.lower() not in {"symbol", "ticker"}:
                symbols.append(token)
        return _normalise_symbol_list(symbols)


def _fetch_listing_text(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (compatible; VelocityTrader/1.0)'},
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        return response.read().decode("utf-8")


def fetch_common_stock_universe() -> list[str]:
    """Fetch US common-stock symbols from NASDAQ Trader listing files."""
    symbols: set[str] = set()

    nq_rows = csv.DictReader(_fetch_listing_text(_NASDAQ_LISTED_URL).splitlines(), delimiter="|")
    for row in nq_rows:
        if row.get("ETF") != "N" or row.get("Test Issue") != "N":
            continue
        if row.get("Market Category") not in {"Q", "G"}:
            continue
        symbol = row.get("Symbol", "")
        name = row.get("Security Name", "")
        if _is_common_equity_listing(symbol, name):
            symbols.add(str(symbol).strip().upper())

    other_rows = csv.DictReader(_fetch_listing_text(_OTHER_LISTED_URL).splitlines(), delimiter="|")
    for row in other_rows:
        if row.get("ETF") != "N" or row.get("Test Issue") != "N":
            continue
        if row.get("Exchange") not in {"N", "A"}:
            continue
        symbol = row.get("ACT Symbol", "")
        name = row.get("Security Name", "")
        if _is_common_equity_listing(symbol, name):
            symbols.add(str(symbol).strip().upper())

    return sorted(symbols)


def _read_universe_cache(path: str) -> tuple[list[str], float]:
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return [], 0.0
    if isinstance(payload, list):
        return _normalise_symbol_list(payload), 0.0
    if not isinstance(payload, dict):
        return [], 0.0
    symbols = _normalise_symbol_list(payload.get("symbols", []))
    try:
        fetched_at = float(payload.get("fetched_at", 0.0) or 0.0)
    except (TypeError, ValueError):
        fetched_at = 0.0
    return symbols, fetched_at


def _write_universe_cache(path: str, symbols: list[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    payload = {
        "fetched_at": time.time(),
        "symbols": list(symbols),
    }
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def load_application_symbol_universe(force_refresh: bool = False) -> list[str]:
    """Return the application scanner's full-symbol universe.

    A configured local file wins for deterministic paper/live runs.  Otherwise
    the NASDAQ Trader listing cache is used while fresh; on network failure, a
    stale cache is still safer than returning no symbols.
    """
    if APP_SCANNER_UNIVERSE_FILE:
        return _read_symbol_file(APP_SCANNER_UNIVERSE_FILE)

    cached_symbols, fetched_at = _read_universe_cache(APP_SCANNER_UNIVERSE_CACHE_FILE)
    fresh = (
        cached_symbols
        and fetched_at > 0
        and (time.time() - fetched_at) <= max(0.0, APP_SCANNER_UNIVERSE_TTL_SEC)
    )
    if fresh and not force_refresh:
        return cached_symbols

    try:
        symbols = fetch_common_stock_universe()
        if symbols:
            _write_universe_cache(APP_SCANNER_UNIVERSE_CACHE_FILE, symbols)
            return symbols
    except Exception:
        if cached_symbols:
            return cached_symbols
        raise

    return cached_symbols


def build_momentum_scanner_filter_options() -> list:
    """IBKR generic filters that mirror local screener gates where possible."""
    profile = _active_profile()
    filters_enabled = (
        IB_SCANNER_FILTERS_ENABLED
        if _env_set("VELOCITY_IB_SCANNER_FILTERS_ENABLED")
        else profile.scanner_filters_enabled
    )
    if not filters_enabled:
        return []
    values = [
        (
            "changeOpenPercAbove",
            "VELOCITY_IB_SCANNER_CHANGE_OPEN_PCT_ABOVE",
            IB_SCANNER_CHANGE_OPEN_PCT_ABOVE,
            profile.scanner_change_open_pct_above,
        ),
        (
            "openGapPercBelow",
            "VELOCITY_IB_SCANNER_OPEN_GAP_PCT_BELOW",
            IB_SCANNER_OPEN_GAP_PCT_BELOW,
            profile.scanner_open_gap_pct_below,
        ),
        (
            "lastVsEMAChangeRatio20Above",
            "VELOCITY_IB_SCANNER_LAST_VS_EMA20_PCT_ABOVE",
            IB_SCANNER_LAST_VS_EMA20_PCT_ABOVE,
            profile.scanner_last_vs_ema20_pct_above,
        ),
        (
            "lastVsEMAChangeRatio50Above",
            "VELOCITY_IB_SCANNER_LAST_VS_EMA50_PCT_ABOVE",
            IB_SCANNER_LAST_VS_EMA50_PCT_ABOVE,
            profile.scanner_last_vs_ema50_pct_above,
        ),
        (
            "curMACDDistAbove",
            "VELOCITY_IB_SCANNER_MACD_HISTOGRAM_ABOVE",
            IB_SCANNER_MACD_HISTOGRAM_ABOVE,
            profile.scanner_macd_histogram_above,
        ),
    ]
    filters = []
    for tag, env_name, config_value, profile_value in values:
        value = config_value if _env_set(env_name) else profile_value
        if value is not None:
            filters.append(TagValue(tag, _scanner_filter_value(value)))
    return filters


def build_momentum_scanners() -> list:
    """One ScannerSubscription per configured scan code; caller deduplicates results."""
    profile = _active_profile()
    scan_codes = IB_SCANNER_SCAN_CODES if _env_set("VELOCITY_IB_SCANNER_SCAN_CODES") else profile.scan_codes
    min_price = SCAN_MIN_PRICE if _env_set("VELOCITY_SCAN_MIN_PRICE") else profile.min_price
    min_volume = SCAN_MIN_VOLUME if _env_set("VELOCITY_SCAN_MIN_VOLUME") else profile.min_volume
    min_mktcap = SCAN_MIN_MKTCAP if _env_set("VELOCITY_SCAN_MIN_MKTCAP") else profile.min_market_cap
    return [
        ScannerSubscription(
            numberOfRows=IB_SCANNER_ROWS,
            instrument='STK',
            locationCode=IB_SCANNER_LOCATION_CODE,
            scanCode=code,
            abovePrice=float(min_price),
            aboveVolume=int(min_volume),
            marketCapAbove=float(min_mktcap) / 1_000_000,
            stockTypeFilter='CORP',
        )
        for code in scan_codes
    ]
