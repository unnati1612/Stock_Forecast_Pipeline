"""
features.py
============
Turns raw daily OHLCV rows into model-ready features: lag returns,
rolling averages, and a couple of simple technical indicators (RSI).

Kept in its own file so both train.py (training) and app.py (live
dashboard predictions) can import the exact same feature logic —
using different code for training vs. inference is a classic bug
source (train/serve skew), so don't duplicate this logic elsewhere.
"""

import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index — a common momentum indicator (0-100)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: a DataFrame for ONE ticker, sorted by date ascending, with
           columns ['date', 'open', 'high', 'low', 'close', 'volume'].
    Output: same DataFrame with added feature columns. Rows at the start
            will have NaNs (not enough history for rolling windows yet) —
            drop those before training.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()

    # Daily return
    df["return_1d"] = df["close"].pct_change()

    # Lag features: yesterday's and 2-days-ago return
    df["lag_return_1"] = df["return_1d"].shift(1)
    df["lag_return_2"] = df["return_1d"].shift(2)

    # Rolling averages (5-day and 10-day)
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_ratio"] = df["ma_5"] / df["ma_10"]  # >1 means short-term momentum is up

    # Rolling volatility (std dev of returns)
    df["volatility_5"] = df["return_1d"].rolling(5).std()

    # RSI
    df["rsi_14"] = compute_rsi(df["close"], window=14)

    # Volume relative to its own recent average (spikes can signal moves)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(10).mean()

    # TARGET: did the price go up (1) or down (0) the NEXT day?
    # This must be computed from FUTURE data relative to each row, so it's
    # only valid for historical training data — never available at prediction time.
    df["next_day_return"] = df["close"].pct_change().shift(-1)
    df["target_direction"] = (df["next_day_return"] > 0).astype(int)

    return df


FEATURE_COLUMNS = [
    "lag_return_1", "lag_return_2", "ma_ratio",
    "volatility_5", "rsi_14", "volume_ratio",
]
