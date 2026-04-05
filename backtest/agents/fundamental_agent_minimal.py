"""
Minimal Fundamental Agent — Pure rule-based, no LLM.

Indicators (updated quarterly):
  1. P/E ratio vs sector median
  2. Revenue growth rate (YoY)
  3. Free Cash Flow Yield (FCF / Market Cap)

Output: continuous signal in [-1, +1].

Data source: yfinance (free, no API key needed).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

# Sector median P/E approximations (as of typical values).
# Used when live sector data unavailable.
SECTOR_PE_MEDIANS = {
    "Technology": 28.0,
    "Consumer Cyclical": 22.0,
    "Communication Services": 20.0,
    "Healthcare": 25.0,
    "Financial Services": 14.0,
    "Industrials": 20.0,
    "Consumer Defensive": 22.0,
    "Energy": 12.0,
    "Utilities": 18.0,
    "Real Estate": 35.0,
    "Basic Materials": 15.0,
    "default": 20.0,
}


def _fetch_fundamentals(ticker: str) -> dict:
    """Fetch fundamental data via yfinance. Returns dict with needed fields."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    info = t.info or {}

    # P/E
    pe_ratio = info.get("trailingPE") or info.get("forwardPE")

    # Sector
    sector = info.get("sector", "default")

    # Revenue growth
    revenue_growth = info.get("revenueGrowth")  # YoY as decimal

    # Free cash flow yield = FCF / Market Cap
    fcf = info.get("freeCashflow")
    market_cap = info.get("marketCap")
    if fcf and market_cap and market_cap > 0:
        fcf_yield = fcf / market_cap
    else:
        fcf_yield = None

    return {
        "pe_ratio": pe_ratio,
        "sector": sector,
        "revenue_growth": revenue_growth,
        "fcf_yield": fcf_yield,
    }


# Cache fundamentals for 24 hours (simulate quarterly updates in backtest)
_fundamentals_cache: dict[str, tuple[str, dict]] = {}


def compute_signal(ticker: str, current_date: str | None = None) -> float:
    """
    Compute fundamental signal for a ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL')
    current_date : str, optional
        Current date string for cache key (quarterly refresh in backtest)

    Returns
    -------
    float in [-1, +1]  (positive = undervalued/bullish, negative = overvalued/bearish)
    """
    # Quarter-based cache key
    cache_key = f"{ticker}_{(current_date or 'live')[:7]}"

    if cache_key in _fundamentals_cache:
        data = _fundamentals_cache[cache_key]
    else:
        try:
            data = _fetch_fundamentals(ticker)
            _fundamentals_cache[cache_key] = data
        except Exception as e:
            logger.warning("Failed to fetch fundamentals for %s: %s", ticker, e)
            return 0.0

    signals = []

    # --- 1. P/E vs sector median [-1, +1] ---
    pe = data.get("pe_ratio")
    sector = data.get("sector", "default")
    sector_median = SECTOR_PE_MEDIANS.get(sector, SECTOR_PE_MEDIANS["default"])

    if pe is not None and pe > 0:
        # Negative = overvalued (PE higher than median), Positive = undervalued
        pe_deviation = (sector_median - pe) / sector_median
        # Scale: ±50% deviation from median maps to ±1
        pe_signal = float(np.clip(pe_deviation / 0.50, -1, 1))
        signals.append(pe_signal)

    # --- 2. Revenue growth [-1, +1] ---
    rev_growth = data.get("revenue_growth")
    if rev_growth is not None:
        # 0% growth = neutral, +30% = very bullish, -30% = very bearish
        rev_signal = float(np.clip(rev_growth / 0.30, -1, 1))
        signals.append(rev_signal)

    # --- 3. Free Cash Flow Yield [-1, +1] ---
    fcf_yield = data.get("fcf_yield")
    if fcf_yield is not None:
        # FCF yield of 5% is good (+1 at 8%), negative FCF is bearish
        # Centre at 3%, scale so 8% → +1, -2% → -1
        fcf_signal = float(np.clip((fcf_yield - 0.03) / 0.05, -1, 1))
        signals.append(fcf_signal)

    if not signals:
        return 0.0

    return float(np.clip(np.mean(signals), -1, 1))


def clear_cache():
    """Clear the fundamentals cache (useful between backtest runs)."""
    _fundamentals_cache.clear()
