"""
Liquidity & Crowding Risk Filter
==================================

Cross-sectional filter using:
  1. Realized Amihud (14d EMA) — price impact per unit volume
  2. Inelasticity (21d rolling) — return volatility / volume volatility

Stocks with Crowding_Risk z-score > threshold are excluded.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_AMIHUD_WINDOW = 14
DEFAULT_INELASTICITY_WINDOW = 21
DEFAULT_ZSCORE_THRESHOLD = 1.5


def realized_amihud(prices: pd.DataFrame, window: int = DEFAULT_AMIHUD_WINDOW) -> pd.Series:
    """
    Realized Amihud illiquidity ratio (EMA smoothed).

    RA = EMA_window( (High - Low) / (Close * Volume) )

    Higher = less liquid = more price impact.
    """
    high = prices["High"]
    low = prices["Low"]
    close = prices["Close"]
    volume = prices["Volume"]

    # Avoid division by zero
    denom = close * volume
    denom = denom.replace(0, np.nan)

    raw_amihud = (high - low) / denom
    # EMA smoothing
    ra = raw_amihud.ewm(span=window, min_periods=window).mean()
    return ra


def inelasticity(prices: pd.DataFrame, window: int = DEFAULT_INELASTICITY_WINDOW) -> pd.Series:
    """
    Price inelasticity measure.

    M = sigma_price / sigma_volume

    Where:
      sigma_price  = rolling std of daily returns
      sigma_volume = rolling std of daily volume

    High M = small volume changes cause big price moves = fragile.
    """
    close = prices["Close"]
    volume = prices["Volume"]

    daily_returns = close.pct_change()
    sigma_p = daily_returns.rolling(window, min_periods=window).std()
    sigma_q = volume.rolling(window, min_periods=window).std()

    # Avoid division by zero
    sigma_q = sigma_q.replace(0, np.nan)

    m = sigma_p / sigma_q
    return m


def compute_crowding_risk(
    all_prices: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    amihud_window: int = DEFAULT_AMIHUD_WINDOW,
    inelasticity_window: int = DEFAULT_INELASTICITY_WINDOW,
) -> dict[str, float]:
    """
    Compute cross-sectional crowding risk z-scores for all stocks on a given date.

    Parameters
    ----------
    all_prices : dict[str, pd.DataFrame]
        {ticker: OHLCV DataFrame} — must contain data up to `date`.
    date : pd.Timestamp
        The date to compute scores for.

    Returns
    -------
    dict[str, float] — {ticker: crowding_risk_zscore}
    """
    ra_values = {}
    m_values = {}

    for ticker, df in all_prices.items():
        hist = df.loc[df.index <= date]
        if len(hist) < max(amihud_window, inelasticity_window) + 5:
            continue

        ra = realized_amihud(hist, amihud_window)
        m = inelasticity(hist, inelasticity_window)

        if not ra.empty and not pd.isna(ra.iloc[-1]):
            ra_values[ticker] = float(ra.iloc[-1])
        if not m.empty and not pd.isna(m.iloc[-1]):
            m_values[ticker] = float(m.iloc[-1])

    # Cross-sectional z-score normalisation
    tickers_with_both = set(ra_values.keys()) & set(m_values.keys())
    if len(tickers_with_both) < 3:
        # Not enough stocks for meaningful z-score
        return {t: 0.0 for t in all_prices}

    ra_arr = np.array([ra_values[t] for t in tickers_with_both])
    m_arr = np.array([m_values[t] for t in tickers_with_both])

    ra_mean, ra_std = ra_arr.mean(), ra_arr.std()
    m_mean, m_std = m_arr.mean(), m_arr.std()

    if ra_std == 0:
        ra_std = 1.0
    if m_std == 0:
        m_std = 1.0

    result = {}
    for ticker in tickers_with_both:
        z_ra = (ra_values[ticker] - ra_mean) / ra_std
        z_m = (m_values[ticker] - m_mean) / m_std
        crowding_risk = 0.5 * z_ra + 0.5 * z_m
        result[ticker] = float(crowding_risk)

    # For tickers without enough data, assign neutral score
    for ticker in all_prices:
        if ticker not in result:
            result[ticker] = 0.0

    return result


def filter_stocks(
    crowding_scores: dict[str, float],
    threshold: float = DEFAULT_ZSCORE_THRESHOLD,
) -> tuple[list[str], list[str]]:
    """
    Filter stocks based on crowding risk score.

    Returns (retained_tickers, excluded_tickers)
    """
    retained = []
    excluded = []

    for ticker, score in crowding_scores.items():
        if score > threshold:
            excluded.append(ticker)
        else:
            retained.append(ticker)

    return retained, excluded


def compute_crowding_risk_series(
    all_prices: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    amihud_window: int = DEFAULT_AMIHUD_WINDOW,
    inelasticity_window: int = DEFAULT_INELASTICITY_WINDOW,
) -> pd.DataFrame:
    """
    Compute crowding risk z-scores for all dates.

    Returns DataFrame with tickers as columns, dates as index.
    """
    records = []
    for i, date in enumerate(dates):
        scores = compute_crowding_risk(all_prices, date, amihud_window, inelasticity_window)
        scores["date"] = date
        records.append(scores)
        if (i + 1) % 100 == 0:
            logger.info("Crowding risk: computed %d/%d days", i + 1, len(dates))

    df = pd.DataFrame(records).set_index("date")
    return df
