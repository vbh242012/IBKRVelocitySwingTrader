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
    return df
