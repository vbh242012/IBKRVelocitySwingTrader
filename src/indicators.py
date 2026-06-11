import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low']  - df['close'].shift()),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def compute_ma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def compute_psar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """Compute Parabolic SAR for long/short swing state detection."""
    if df.empty:
        return pd.Series(dtype=float, index=df.index)

    high = df['high'].astype(float)
    low = df['low'].astype(float)
    psar = pd.Series(index=df.index, dtype=float)
    if len(df) < 2:
        psar.iloc[0] = low.iloc[0]
        return psar

    bull = True
    af = step
    ep = high.iloc[0]
    psar.iloc[0] = low.iloc[0]
    psar.iloc[1] = low.iloc[0]

    for i in range(2, len(df)):
        prior_psar = psar.iloc[i - 1]
        if bull:
            sar = prior_psar + af * (ep - prior_psar)
            sar = min(sar, low.iloc[i - 1], low.iloc[i - 2])
            if low.iloc[i] < sar:
                bull = False
                sar = ep
                ep = low.iloc[i]
                af = step
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_step)
        else:
            sar = prior_psar + af * (ep - prior_psar)
            sar = max(sar, high.iloc[i - 1], high.iloc[i - 2])
            if high.iloc[i] > sar:
                bull = True
                sar = ep
                ep = high.iloc[i]
                af = step
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_step)
        psar.iloc[i] = sar
    return psar


def _rolling_all_true(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).sum().eq(window)


def apply_all(df: pd.DataFrame,
              rsi_period:         int = 14,
              atr_period:         int = 14,
              ma_fast:            int = 50,
              ma_slow:            int = 200,
              slope_lookback:     int = 5,
              chandelier_period:  int = 22) -> pd.DataFrame:
    """Attach indicator columns to an OHLCV dataframe."""
    df = df.copy()
    df['MA50']         = compute_ma(df['close'], ma_fast)
    df['MA200']        = compute_ma(df['close'], ma_slow)
    df['ATR']          = compute_atr(df, atr_period)
    df['ATR5']         = compute_atr(df, 5)
    df['ATR20']        = compute_atr(df, 20)
    df['ATR_CHAND']    = compute_atr(df, chandelier_period)
    df['RSI']          = compute_rsi(df['close'], rsi_period)
    day_range          = df['high'] - df['low']
    df['CLV']          = (df['close'] - df['low']) / day_range.where(day_range != 0)
    df['SMA200_SLOPE'] = df['MA200'] - df['MA200'].shift(slope_lookback)
    df['HIGH10']       = df['high'].rolling(10).max()
    df['EMA20']        = df['close'].ewm(span=20, adjust=False).mean()
    df['SMA50']        = df['MA50']
    df['EMA20_GT_SMA50'] = df['EMA20'] > df['SMA50']
    ema20_gt_prev = df['EMA20_GT_SMA50'].shift(1, fill_value=False)
    df['MA_BULL_CROSS'] = df['EMA20_GT_SMA50'] & ~ema20_gt_prev
    df['MA_BEAR_CROSS'] = ~df['EMA20_GT_SMA50'] & ema20_gt_prev

    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['BB_MID']       = bb_mid
    df['BB_UPPER']     = bb_mid + 2.0 * bb_std
    df['BB_LOWER']     = bb_mid - 2.0 * bb_std
    close_below_lower = df['close'] < df['BB_LOWER']
    close_above_upper = df['close'] > df['BB_UPPER']
    below_lower_prev1 = close_below_lower.shift(1, fill_value=False)
    below_lower_prev2 = close_below_lower.shift(2, fill_value=False)
    above_upper_prev1 = close_above_upper.shift(1, fill_value=False)
    df['BB_BELOW_LOWER_2'] = close_below_lower & below_lower_prev1
    df['BB_ABOVE_UPPER_2'] = close_above_upper & above_upper_prev1
    two_prior_below = below_lower_prev1 & below_lower_prev2
    df['BB_RECLAIM_LOWER'] = (df['close'] > df['BB_LOWER']) & two_prior_below

    df['PSAR']         = compute_psar(df)
    df['PSAR_BELOW_PRICE'] = df['PSAR'] < df['close']
    df['PSAR_ABOVE_PRICE'] = df['PSAR'] > df['close']
    df['PSAR_BULL_3'] = _rolling_all_true(df['PSAR_BELOW_PRICE'], 3)
    df['PSAR_BEAR_3'] = _rolling_all_true(df['PSAR_ABOVE_PRICE'], 3)

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD']         = macd
    df['MACD_SIGNAL']  = macd_signal
    df['MACD_HIST']    = macd - macd_signal
    df['MACD_HIST_DELTA'] = df['MACD_HIST'] - df['MACD_HIST'].shift(1)
    price_low5 = df['close'].rolling(5).min()
    price_high5 = df['close'].rolling(5).max()
    macd_low5 = df['MACD_HIST'].rolling(5).min()
    macd_high5 = df['MACD_HIST'].rolling(5).max()
    df['MACD_BULL_DIVERGENCE'] = (price_low5 < price_low5.shift(5)) & (macd_low5 > macd_low5.shift(5))
    df['MACD_BEAR_DIVERGENCE'] = (price_high5 > price_high5.shift(5)) & (macd_high5 < macd_high5.shift(5))

    lowest_low = df['low'].rolling(14).min()
    highest_high = df['high'].rolling(14).max()
    stoch_den = (highest_high - lowest_low).where((highest_high - lowest_low) != 0)
    df['STOCH_K'] = (df['close'] - lowest_low) / stoch_den * 100.0
    df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
    stoch_cross_up = (df['STOCH_K'] > df['STOCH_D']) & (df['STOCH_K'].shift(1) <= df['STOCH_D'].shift(1))
    stoch_cross_down = (df['STOCH_K'] < df['STOCH_D']) & (df['STOCH_K'].shift(1) >= df['STOCH_D'].shift(1))
    df['STOCH_BULL_EXIT_OVERSOLD'] = stoch_cross_up & (df['STOCH_K'] > 20.0) & (df['STOCH_D'] > 20.0)
    df['STOCH_BEAR_EXIT_OVERBOUGHT'] = stoch_cross_down & (df['STOCH_K'] < 80.0) & (df['STOCH_D'] < 80.0)

    signed_volume = (df['volume'].where(df['close'].diff() >= 0, -df['volume'])).fillna(df['volume'])
    df['OBV'] = signed_volume.cumsum()
    df['OBV_SLOPE_5'] = df['OBV'] - df['OBV'].shift(5)
    df['OBV_UPTREND'] = df['OBV_SLOPE_5'] > 0
    price_slope5 = df['close'] - df['close'].shift(5)
    df['OBV_BULL_DIVERGENCE'] = (price_slope5 < 0) & (df['OBV_SLOPE_5'] > 0)
    df['OBV_BEAR_DIVERGENCE'] = (price_slope5 > 0) & (df['OBV_SLOPE_5'] < 0)
    return df
