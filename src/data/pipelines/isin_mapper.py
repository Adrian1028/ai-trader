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

# ── Market identifier ────────────────────────────────────────────
# Used to route tickers to the correct data provider and schedule.
MARKET_US = "US"
MARKET_UK = "UK"

# ── Static ISIN → Standard Ticker Map ────────────────────────────
# ISIN is the universal security identifier. T212 may return arbitrary
# exchange codes (APCd_EQ, MSFd_EQ, TL0d_EQ etc.) that are useless
# for external data providers. This table maps ISINs to standard
# market tickers that data providers understand.
#
# Format: ISIN → (ticker, market)
#   US: plain ticker e.g. "AAPL"
#   UK: LSE ticker e.g. "BARC.L" (Yahoo Finance format)
_ISIN_TO_TICKER: dict[str, tuple[str, str]] = {
    # ══════════════════════════════════════════════════════════════
    # US STOCKS (美股)
    # ══════════════════════════════════════════════════════════════
    # ── 科技 (Technology) ──
    "US0378331005": ("AAPL", MARKET_US),     # Apple
    "US5949181045": ("MSFT", MARKET_US),     # Microsoft
    "US67066G1040": ("NVDA", MARKET_US),     # NVIDIA
    "US88160R1014": ("TSLA", MARKET_US),     # Tesla
    "US0231351067": ("AMZN", MARKET_US),     # Amazon
    "US02079K3059": ("GOOGL", MARKET_US),    # Alphabet (Class A)
    "US02079K1079": ("GOOG", MARKET_US),     # Alphabet (Class C)
    "US30303M1027": ("META", MARKET_US),     # Meta Platforms
    "US11135F1012": ("AVGO", MARKET_US),     # Broadcom
    "US12652L1008": ("CRM", MARKET_US),      # Salesforce
    "US0079031078": ("AMD", MARKET_US),      # AMD
    "US68389X1054": ("ORCL", MARKET_US),     # Oracle
    "US4581401001": ("INTC", MARKET_US),     # Intel
    # ── 能源 (Energy) ──
    "US30231G1022": ("XOM", MARKET_US),      # Exxon Mobil
    "US1667641005": ("CVX", MARKET_US),      # Chevron
    "US20825C1045": ("COP", MARKET_US),      # ConocoPhillips
    "US8064071025": ("SLB", MARKET_US),      # Schlumberger
    "US26875P1012": ("EOG", MARKET_US),      # EOG Resources
    "US65339F1012": ("NEE", MARKET_US),      # NextEra Energy
    "US29355A1079": ("ENPH", MARKET_US),     # Enphase Energy
    "US6745991058": ("OXY", MARKET_US),      # Occidental Petroleum
    # ── 金融 (Financials) ──
    "US46625H1005": ("JPM", MARKET_US),      # JPMorgan Chase
    "US0605051046": ("BAC", MARKET_US),      # Bank of America
    "US9311421039": ("WMT", MARKET_US),      # Walmart
    "US7427181091": ("PG", MARKET_US),       # Procter & Gamble
    "US4781601046": ("JNJ", MARKET_US),      # Johnson & Johnson
    "US92826C8394": ("V", MARKET_US),        # Visa
    "US5801351017": ("MCD", MARKET_US),      # McDonald's
    "US2546871060": ("DIS", MARKET_US),      # Disney
    "US17275R1023": ("CSCO", MARKET_US),     # Cisco
    "US7170811035": ("PFE", MARKET_US),      # Pfizer
    "US0846707026": ("BRK.B", MARKET_US),    # Berkshire Hathaway B
    "US0258161092": ("AXP", MARKET_US),      # American Express
    # ── 其他熱門 ──
    "US0970231058": ("BA", MARKET_US),       # Boeing
    "US00724F1012": ("ADBE", MARKET_US),     # Adobe
    "US6541061031": ("NFLX", MARKET_US),     # Netflix
    "US79466L3024": ("SBUX", MARKET_US),     # Starbucks
    "US7960542030": ("SHOP", MARKET_US),     # Shopify
    "US55354G1004": ("MRVL", MARKET_US),     # Marvell Technology
    "US46120E6023": ("INTU", MARKET_US),     # Intuit
    "US5324571083": ("LLY", MARKET_US),      # Eli Lilly
    "US58933Y1055": ("MRK", MARKET_US),      # Merck
    "US0028241000": ("ABBV", MARKET_US),     # AbbVie
    # ══════════════════════════════════════════════════════════════
    # UK STOCKS (英股) — London Stock Exchange
    # ══════════════════════════════════════════════════════════════
    # Yahoo Finance format: TICKER.L
    # ── 銀行 & 金融 ──
    "GB0031348658": ("BARC.L", MARKET_UK),   # Barclays
    "GB0005405286": ("HSBA.L", MARKET_UK),   # HSBC Holdings
    "GB00B16GWD56": ("LLOY.L", MARKET_UK),   # Lloyds Banking Group
    "GB0008706128": ("NWG.L", MARKET_UK),    # NatWest Group
    "GB00BN7SWP63": ("LSEG.L", MARKET_UK),  # London Stock Exchange Group
    # ── 能源 & 石油 ──
    "GB0007980591": ("BP.L", MARKET_UK),     # BP
    "GB00B03MLX29": ("SHEL.L", MARKET_UK),   # Shell
    # ── 消費品 ──
    "GB00B24CGK77": ("RECKITT.L", MARKET_UK),# Reckitt Benckiser → actual ticker RKT.L
    "GB0006731235": ("AZN.L", MARKET_UK),    # AstraZeneca
    "GB00BN7SWP63": ("LSEG.L", MARKET_UK),  # LSE Group
    "GB0009895292": ("DGE.L", MARKET_UK),    # Diageo
    "GB00B10RZP78": ("ULVR.L", MARKET_UK),   # Unilever
    # ── 礦業 & 原材料 ──
    "GB0000566504": ("RIO.L", MARKET_UK),    # Rio Tinto
    "AU000000BHP4": ("BHP.L", MARKET_UK),    # BHP Group
    "GB00B2B0DG97": ("GLEN.L", MARKET_UK),   # Glencore
    "JE00B4T3BW64": ("AAL.L", MARKET_UK),   # Anglo American
    # ── 電信 & 科技 ──
    "GB00BH4HKS39": ("VOD.L", MARKET_UK),   # Vodafone
    # ── 保險 & 其他金融 ──
    "GB00B0SWJX34": ("PHNX.L", MARKET_UK),  # Phoenix Group
    "GB0001383545": ("AV.L", MARKET_UK),     # Aviva
    "GB0002162385": ("STAN.L", MARKET_UK),   # Standard Chartered
    # ── 其他 ──
    "GB00B0744B38": ("TSCO.L", MARKET_UK),   # Tesco
    "GB00BDCPN049": ("RELX.L", MARKET_UK),   # RELX (Reed Elsevier)
    "IE000S9YS762": ("EXPN.L", MARKET_UK),   # Experian
    "GB00B1XZS820": ("RR.L", MARKET_UK),     # Rolls-Royce
    "GB0007188757": ("RTO.L", MARKET_UK),    # Rentokil Initial
}

# ── Backward-compatible flat map: ISIN → ticker string ───────────
_ISIN_TO_US_TICKER: dict[str, str] = {
    isin: ticker for isin, (ticker, market) in _ISIN_TO_TICKER.items()
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
        Get the standard market ticker for an ISIN.

        Priority:
          1. Static ISIN → ticker map (most reliable)
          2. Parse T212 ticker code (fallback)

        Returns
        -------
        Standard ticker like 'AAPL', 'BARC.L', etc.
        """
        # 1. Check static map first (always correct)
        if isin in _ISIN_TO_TICKER:
            return _ISIN_TO_TICKER[isin][0]

        # 2. Backward compat: check flat US map
        if isin in _ISIN_TO_US_TICKER:
            return _ISIN_TO_US_TICKER[isin]

        # 3. Fallback: try to parse from T212 ticker
        t212_ticker = self.ticker_for_isin(isin)
        if t212_ticker:
            return self.to_standard_ticker(t212_ticker)

        return ""

    def market_for_isin(self, isin: str) -> str:
        """
        Get the market identifier for an ISIN.

        Returns
        -------
        'US', 'UK', or '' if unknown.
        """
        if isin in _ISIN_TO_TICKER:
            return _ISIN_TO_TICKER[isin][1]
        # Default: assume US if ISIN starts with US
        if isin.startswith("US"):
            return MARKET_US
        if isin.startswith("GB") or isin.startswith("IE") or isin.startswith("JE"):
            return MARKET_UK
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
