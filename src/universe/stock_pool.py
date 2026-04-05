"""
Stock Pool — 20 S&P 500 stocks across 8 sectors.

All stocks meet:
  - Daily avg volume > 5M shares
  - Market cap > $100B
  - S&P 500 constituent
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StockInfo:
    ticker: str
    name: str
    sector: str


# 20 stocks, 8 sectors
STOCK_POOL: list[StockInfo] = [
    # Technology (4)
    StockInfo("AAPL", "Apple", "Technology"),
    StockInfo("MSFT", "Microsoft", "Technology"),
    StockInfo("GOOGL", "Alphabet", "Technology"),
    StockInfo("NVDA", "NVIDIA", "Technology"),
    # Financials (3)
    StockInfo("JPM", "JPMorgan Chase", "Financials"),
    StockInfo("BAC", "Bank of America", "Financials"),
    StockInfo("GS", "Goldman Sachs", "Financials"),
    # Healthcare (3)
    StockInfo("JNJ", "Johnson & Johnson", "Healthcare"),
    StockInfo("UNH", "UnitedHealth", "Healthcare"),
    StockInfo("PFE", "Pfizer", "Healthcare"),
    # Consumer Discretionary (3)
    StockInfo("AMZN", "Amazon", "Consumer Discretionary"),
    StockInfo("WMT", "Walmart", "Consumer Discretionary"),
    StockInfo("PG", "Procter & Gamble", "Consumer Discretionary"),
    # Energy (2)
    StockInfo("XOM", "Exxon Mobil", "Energy"),
    StockInfo("CVX", "Chevron", "Energy"),
    # Industrials (2)
    StockInfo("CAT", "Caterpillar", "Industrials"),
    StockInfo("HON", "Honeywell", "Industrials"),
    # Communication (2)
    StockInfo("META", "Meta Platforms", "Communication"),
    StockInfo("DIS", "Walt Disney", "Communication"),
    # Utilities (1)
    StockInfo("NEE", "NextEra Energy", "Utilities"),
]

TICKERS = [s.ticker for s in STOCK_POOL]
SECTOR_MAP = {s.ticker: s.sector for s in STOCK_POOL}
SECTORS = list(set(s.sector for s in STOCK_POOL))


def get_tickers() -> list[str]:
    return TICKERS


def get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker, "Unknown")


def get_stocks_by_sector(sector: str) -> list[str]:
    return [s.ticker for s in STOCK_POOL if s.sector == sector]
