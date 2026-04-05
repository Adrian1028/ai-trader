"""
Minimal Technical Agent — Pure rule-based, no LLM.

Indicators:
  1. SMA(50) / SMA(200) crossover  (trend)
  2. RSI(14)                        (momentum / mean-reversion)
  3. MACD(12,26,9)                  (momentum confirmation)

Output: continuous signal in [-1, +1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_signal(prices: pd.DataFrame) -> float:
    """
    Compute technical signal from OHLCV DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Must contain 'Close' column, indexed by date, sorted ascending.
        Needs at least 200 rows for SMA-200.

    Returns
    -------
    float in [-1, +1]  (positive = bullish, negative = bearish)
    """
    close = prices["Close"]

    if len(close) < 200:
        return 0.0  # insufficient data

    # --- 1. SMA crossover signal [-1, +1] ---
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)

    # Normalised distance: (SMA50 - SMA200) / SMA200, clamped
    sma_diff = (sma50.iloc[-1] - sma200.iloc[-1]) / sma200.iloc[-1]
    # Scale so ±5% maps to ±1
    sma_signal = float(np.clip(sma_diff / 0.05, -1, 1))

    # --- 2. RSI signal [-1, +1] ---
    rsi = _rsi(close, 14)
    rsi_val = rsi.iloc[-1]

    if np.isnan(rsi_val):
        rsi_signal = 0.0
    elif rsi_val >= 70:
        # Overbought → bearish; scale 70-100 to 0..-1
        rsi_signal = -((rsi_val - 70) / 30)
    elif rsi_val <= 30:
        # Oversold → bullish; scale 30-0 to 0..+1
        rsi_signal = (30 - rsi_val) / 30
    else:
        # Neutral zone 30-70 → slight directional bias
        rsi_signal = (50 - rsi_val) / 40  # mild contrarian

    rsi_signal = float(np.clip(rsi_signal, -1, 1))

    # --- 3. MACD signal [-1, +1] ---
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)

    macd_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
    # Normalise by price level so signal is scale-invariant
    price_level = close.iloc[-1]
    if price_level > 0:
        macd_norm = macd_diff / price_level
    else:
        macd_norm = 0.0
    # Scale so ±1% of price maps to ±1
    macd_signal = float(np.clip(macd_norm / 0.01, -1, 1))

    # --- Combine: equal weight across three indicators ---
    combined = (sma_signal + rsi_signal + macd_signal) / 3.0
    return float(np.clip(combined, -1, 1))
