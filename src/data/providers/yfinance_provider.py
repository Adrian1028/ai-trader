"""
YFinance Data Provider — Universal Market Data
===============================================
Free, unlimited data provider using Yahoo Finance (yfinance library).
Supports all global markets: US, UK (LSE), EU, HK, JP, etc.

Ticker formats:
  - US stocks:  AAPL, MSFT
  - UK stocks:  BARC.L, HSBA.L, BP.L
  - HK stocks:  0005.HK, 0700.HK

Usage:
  - Primary data source for NON-US stocks (UK, EU, HK)
  - Fallback data source for US stocks when Polygon quota exhausted
  - No API key required, no rate limits

Provides:
  - Daily OHLCV bars (historical)
  - Company fundamentals (P/E, ROE, market cap, etc.)
  - Basic quote data (current price, volume)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from .base_provider import BaseDataProvider, OHLCVBar

logger = logging.getLogger(__name__)


class YFinanceProvider(BaseDataProvider):
    """
    Yahoo Finance data provider via yfinance library.
    No API key required — completely free and unlimited.
    """

    def __init__(self, cache_ttl: float = 300.0) -> None:
        # No API key needed; pass empty string
        super().__init__("yfinance", "", "https://finance.yahoo.com", cache_ttl_seconds=cache_ttl)
        self._yf = None  # lazy import

    def _get_yf(self) -> Any:
        """Lazy import yfinance to avoid import errors if not installed."""
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
        return self._yf

    async def health_check(self) -> bool:
        """Check if yfinance is importable and can fetch data."""
        try:
            yf = self._get_yf()
            ticker = yf.Ticker("AAPL")
            info = ticker.fast_info
            return info is not None
        except Exception as exc:
            logger.warning("[yfinance] Health check failed: %s", exc)
            return False

    # -- OHLCV data ------------------------------------------------

    async def daily_bars(
        self,
        ticker: str,
        days: int = 365,
    ) -> list[OHLCVBar]:
        """
        Fetch daily OHLCV bars for any global ticker.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker (e.g., 'AAPL', 'BARC.L', '0005.HK')
        days : int
            Number of days of history to fetch.

        Returns
        -------
        list[OHLCVBar] : Sorted oldest-first.
        """
        cache_key = f"daily_bars:{ticker}:{days}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        try:
            yf = self._get_yf()
            self._request_count += 1

            t = yf.Ticker(ticker)
            period = "2y" if days > 365 else "1y" if days > 180 else "6mo"
            df = t.history(period=period, auto_adjust=True)

            if df is None or df.empty:
                logger.warning("[yfinance] No data for %s", ticker)
                return []

            bars: list[OHLCVBar] = []
            for idx, row in df.iterrows():
                bars.append(OHLCVBar(
                    timestamp=idx.strftime("%Y-%m-%d"),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                ))

            # Trim to requested days
            if len(bars) > days:
                bars = bars[-days:]

            self._set_cached(cache_key, bars)
            logger.debug("[yfinance] %s: fetched %d daily bars", ticker, len(bars))
            return bars

        except Exception as exc:
            self._error_count += 1
            logger.warning("[yfinance] Failed to fetch daily bars for %s: %s", ticker, exc)
            return []

    # -- Fundamentals ----------------------------------------------

    async def company_overview(self, ticker: str) -> dict[str, Any]:
        """
        Fetch company fundamentals (P/E, ROE, market cap, etc.).
        Returns a dict compatible with AlphaVantage's overview format.
        """
        cache_key = f"overview:{ticker}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        try:
            yf = self._get_yf()
            self._request_count += 1

            t = yf.Ticker(ticker)
            info = t.info or {}

            # Map to AlphaVantage-compatible format
            overview = {
                "Symbol": ticker,
                "Name": info.get("longName", info.get("shortName", ticker)),
                "Exchange": info.get("exchange", ""),
                "Currency": info.get("currency", ""),
                "MarketCapitalization": str(info.get("marketCap", "")),
                "PERatio": str(info.get("trailingPE", "")),
                "ForwardPE": str(info.get("forwardPE", "")),
                "PriceToBookRatio": str(info.get("priceToBook", "")),
                "ReturnOnEquityTTM": str(info.get("returnOnEquity", "")),
                "ProfitMargin": str(info.get("profitMargins", "")),
                "Beta": str(info.get("beta", "")),
                "DividendYield": str(info.get("dividendYield", "")),
                "52WeekHigh": str(info.get("fiftyTwoWeekHigh", "")),
                "52WeekLow": str(info.get("fiftyTwoWeekLow", "")),
                "50DayMovingAverage": str(info.get("fiftyDayAverage", "")),
                "200DayMovingAverage": str(info.get("twoHundredDayAverage", "")),
                "QuarterlyRevenueGrowthYOY": str(info.get("revenueGrowth", "")),
                "QuarterlyEarningsGrowthYOY": str(info.get("earningsGrowth", "")),
            }

            self._set_cached(cache_key, overview)
            return overview

        except Exception as exc:
            self._error_count += 1
            logger.warning("[yfinance] Failed to fetch overview for %s: %s", ticker, exc)
            return {}

    # -- Quote data ------------------------------------------------

    async def get_quote(self, ticker: str) -> dict[str, Any]:
        """Get current price and basic quote data."""
        cache_key = f"quote:{ticker}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        try:
            yf = self._get_yf()
            self._request_count += 1

            t = yf.Ticker(ticker)
            info = t.fast_info

            quote = {
                "symbol": ticker,
                "price": float(info.get("lastPrice", 0) or info.get("previousClose", 0)),
                "previous_close": float(info.get("previousClose", 0)),
                "market_cap": float(info.get("marketCap", 0)),
                "currency": str(info.get("currency", "USD")),
            }

            self._set_cached(cache_key, quote)
            return quote

        except Exception as exc:
            self._error_count += 1
            logger.warning("[yfinance] Failed to get quote for %s: %s", ticker, exc)
            return {}
