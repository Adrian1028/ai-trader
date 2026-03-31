"""
ISIN Dynamic Mapping Middleware
===============================
Maintains a bidirectional map between Trading 212 internal ticker codes
and ISIN (the single source of truth for cross-system identity).

CRITICAL NOTE:
  Trading 212 may return European exchange tickers for US stocks
  (e.g., APCd_EQ for Apple on Deutsche Börse, MSFd_EQ for Microsoft).
  These are NOT valid for Polygon/AlphaVantage/Finnhub.

  We maintain a static ISIN → standard US ticker map for our target
  stocks, and fall back to T212 ticker parsing for others.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Static ISIN → Standard US Ticker Map ─────────────────────────
# ISIN is the universal security identifier. T212 may return arbitrary
# exchange codes (APCd_EQ, MSFd_EQ, TL0d_EQ etc.) that are useless
# for external data providers. This table maps ISINs to standard
# US market tickers that Polygon/AlphaVantage/Finnhub understand.
_ISIN_TO_US_TICKER: dict[str, str] = {
    # ── 科技 (Technology) ──
    "US0378331005": "AAPL",     # Apple
    "US5949181045": "MSFT",     # Microsoft
    "US67066G1040": "NVDA",     # NVIDIA
    "US88160R1014": "TSLA",     # Tesla
    "US0231351067": "AMZN",     # Amazon
    "US02079K3059": "GOOGL",    # Alphabet (Class A)
    "US02079K1079": "GOOG",     # Alphabet (Class C)
    "US30303M1027": "META",     # Meta Platforms
    "US11135F1012": "AVGO",     # Broadcom
    "US12652L1008": "CRM",      # Salesforce
    "US0079031078": "AMD",      # AMD
    "US68389X1054": "ORCL",     # Oracle
    "US4581401001": "INTC",     # Intel
    # ── 能源 (Energy) ──
    "US30231G1022": "XOM",      # Exxon Mobil
    "US1667641005": "CVX",      # Chevron
    "US20825C1045": "COP",      # ConocoPhillips
    "US8064071025": "SLB",      # Schlumberger
    "US26875P1012": "EOG",      # EOG Resources
    "US65339F1012": "NEE",      # NextEra Energy
    "US29355A1079": "ENPH",     # Enphase Energy
    "US6745991058": "OXY",      # Occidental Petroleum
    # ── 金融 (Financials) ──
    "US46625H1005": "JPM",      # JPMorgan Chase
    "US0605051046": "BAC",      # Bank of America
    "US9311421039": "WMT",      # Walmart
    "US7427181091": "PG",       # Procter & Gamble
    "US4781601046": "JNJ",      # Johnson & Johnson
    "US92826C8394": "V",        # Visa
    "US5801351017": "MCD",      # McDonald's
    "US2546871060": "DIS",      # Disney
    "US17275R1023": "CSCO",     # Cisco
    "US7170811035": "PFE",      # Pfizer
    "US0846707026": "BRK.B",    # Berkshire Hathaway B
    "US0258161092": "AXP",      # American Express
    # ── 其他熱門 ──
    "US0970231058": "BA",       # Boeing
    "US00724F1012": "ADBE",     # Adobe
    "US6541061031": "NFLX",     # Netflix
    "US79466L3024": "SBUX",     # Starbucks
    "US7960542030": "SHOP",     # Shopify
    "US55354G1004": "MRVL",     # Marvell Technology
    "US46120E6023": "INTU",     # Intuit
    "US5324571083": "LLY",      # Eli Lilly
    "US58933Y1055": "MRK",      # Merck
    "US0028241000": "ABBV",     # AbbVie
}



@dataclass
class Instrument:
    ticker: str          # Trading 212 internal code
    isin: str            # International Securities Identification Number
    currency: str
    name: str
    exchange: str = ""
    type: str = ""       # e.g. "STOCK", "ETF"


class ISINMapper:
    """
    Resolves between Trading 212 tickers and ISINs.
    Populated at startup from /equity/metadata/instruments.
    """

    def __init__(self) -> None:
        self._by_isin: dict[str, Instrument] = {}
        self._by_ticker: dict[str, Instrument] = {}

    @property
    def count(self) -> int:
        return len(self._by_isin)

    # ── bulk load ─────────────────────────────────────────────────────

    def load_instruments(self, raw_instruments: list[dict]) -> None:
        """Ingest the instrument list returned by the T212 metadata endpoint."""
        self._by_isin.clear()
        self._by_ticker.clear()

        for raw in raw_instruments:
            inst = Instrument(
                ticker=raw.get("ticker", ""),
                isin=raw.get("isin", ""),
                currency=raw.get("currencyCode", ""),
                name=raw.get("name", ""),
                exchange=raw.get("exchangeId", ""),
                type=raw.get("type", ""),
            )
            if inst.isin:
                # Prefer US exchange listing if multiple exist
                existing = self._by_isin.get(inst.isin)
                if existing is None or "_US_" in inst.ticker:
                    self._by_isin[inst.isin] = inst
            if inst.ticker:
                self._by_ticker[inst.ticker] = inst

        logger.info("ISINMapper loaded %d instruments", self.count)

    # ── lookups ───────────────────────────────────────────────────────

    def ticker_for_isin(self, isin: str) -> str | None:
        inst = self._by_isin.get(isin)
        return inst.ticker if inst else None

    def isin_for_ticker(self, ticker: str) -> str | None:
        inst = self._by_ticker.get(ticker)
        return inst.isin if inst else None

    def instrument_by_isin(self, isin: str) -> Instrument | None:
        return self._by_isin.get(isin)

    def instrument_by_ticker(self, ticker: str) -> Instrument | None:
        return self._by_ticker.get(ticker)

    def standard_ticker_for_isin(self, isin: str) -> str:
        """
        Get the standard US market ticker for an ISIN.

        Priority:
          1. Static ISIN → US ticker map (most reliable)
          2. Parse T212 ticker code (fallback)

        Returns
        -------
        Standard ticker like 'AAPL', 'MSFT', 'TSLA' etc.
        """
        # 1. Check static map first (always correct)
        if isin in _ISIN_TO_US_TICKER:
            return _ISIN_TO_US_TICKER[isin]

        # 2. Fallback: try to parse from T212 ticker
        t212_ticker = self.ticker_for_isin(isin)
        if t212_ticker:
            return self.to_standard_ticker(t212_ticker)

        return ""

    @staticmethod
    def to_standard_ticker(t212_ticker: str) -> str:
        """
        Convert Trading 212 internal ticker to standard market ticker.

        WARNING: This only works reliably for *_US_EQ format tickers.
        For European exchange codes (APCd_EQ, MSFd_EQ, TL0d_EQ),
        use standard_ticker_for_isin() instead which uses a static ISIN map.

        Examples:
            'AAPL_US_EQ'  → 'AAPL'
            'AMD_US_EQ'   → 'AMD'
            'NVDA_US_EQ'  → 'NVDA'
            'BRKb_US_EQ'  → 'BRK.B'

        T212 format: {SYMBOL}_{COUNTRY}_{TYPE}
        """
        if not t212_ticker:
            return t212_ticker

        # Strip the _XX_YY suffix (e.g., _US_EQ, _GB_EQ)
        parts = t212_ticker.split("_")
        if len(parts) >= 3:
            symbol = "_".join(parts[:-2])  # handle edge cases with _ in symbol
        elif len(parts) == 2:
            symbol = parts[0]
        else:
            symbol = t212_ticker

        # T212 uses lowercase letters for share class (e.g., BRKb → BRK.B)
        # Convert trailing lowercase to dot-notation
        if len(symbol) > 1 and symbol[-1].islower() and symbol[-2].isupper():
            symbol = symbol[:-1] + "." + symbol[-1].upper()

        return symbol
