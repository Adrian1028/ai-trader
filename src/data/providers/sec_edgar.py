"""
SEC EDGAR Provider — 內部人交易數據源
======================================
從 SEC EDGAR 的 EFTS (Full-Text Search) API 取得 Form 4 內部人交易數據。

免費 API，無需 API key，但需要設定 User-Agent header。
https://efts.sec.gov/LATEST/

使用者：
  - InsiderTradingAgent：內部人交易信號分析
"""
from __future__ import annotations

import logging
from typing import Any

import aiohttp

from src.data.providers.base_provider import BaseDataProvider

logger = logging.getLogger(__name__)

# SEC EDGAR company tickers endpoint (maps ticker → CIK)
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


class SECEdgarProvider(BaseDataProvider):
    """
    SEC EDGAR 內部人交易數據提供者。

    使用 EDGAR EFTS API 取得 Form 4 (內部人交易報告)。
    免費，不需 API key，但需設定合規的 User-Agent header。

    Parameters
    ----------
    user_agent : SEC 要求的 User-Agent（格式：公司名/聯繫方式）
    cache_ttl : 快取 TTL（內部人交易數據更新較慢，預設 10 分鐘）
    """

    def __init__(
        self,
        user_agent: str = "TradingBot/1.0 (contact@example.com)",
        cache_ttl: float = 600.0,
    ) -> None:
        super().__init__(
            name="sec_edgar",
            api_key="",
            base_url="https://efts.sec.gov/LATEST",
            cache_ttl_seconds=cache_ttl,
        )
        self._user_agent = user_agent
        self._ticker_to_cik: dict[str, str] = {}

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Override to set required User-Agent header."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "User-Agent": self._user_agent,
                    "Accept": "application/json",
                },
            )
        return self._session

    async def _load_ticker_map(self) -> None:
        """載入 ticker → CIK 映射表。"""
        if self._ticker_to_cik:
            return
        try:
            session = await self._ensure_session()
            async with session.get(_COMPANY_TICKERS_URL) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    for entry in data.values():
                        ticker = entry.get("ticker", "").upper()
                        cik = str(entry.get("cik_str", ""))
                        if ticker and cik:
                            self._ticker_to_cik[ticker] = cik.zfill(10)
                    self.logger.info(
                        "[SEC EDGAR] Loaded %d ticker→CIK mappings",
                        len(self._ticker_to_cik),
                    )
        except Exception:
            self.logger.warning(
                "[SEC EDGAR] Failed to load ticker map", exc_info=True,
            )

    async def insider_transactions(
        self,
        ticker: str,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """
        取得指定股票的內部人交易記錄。

        使用 EDGAR full-text search API 搜尋 Form 4 filing。
        返回正規化的交易記錄列表。

        Parameters
        ----------
        ticker : 股票代碼 (e.g. "AAPL")
        limit : 最多回傳筆數
        """
        await self._load_ticker_map()

        clean_ticker = ticker.replace(".L", "").upper()
        cik = self._ticker_to_cik.get(clean_ticker)
        if not cik:
            self.logger.debug("[SEC EDGAR] No CIK for ticker %s", ticker)
            return []

        url = f"{self._base_url}/search-index"
        params = {
            "q": f'"form 4"',
            "dateRange": "custom",
            "startdt": "",  # will be overridden below
            "enddt": "",
            "forms": "4",
            "from": "0",
            "size": str(limit),
        }

        # Use company filings API instead (more reliable)
        filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"

        try:
            session = await self._ensure_session()
            async with session.get(filings_url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)

            recent = data.get("filings", {}).get("recent", {})
            if not recent:
                return []

            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])

            transactions: list[dict[str, Any]] = []
            for i, form in enumerate(forms):
                if form != "4":
                    continue
                if len(transactions) >= limit:
                    break

                transactions.append({
                    "form": form,
                    "filing_date": dates[i] if i < len(dates) else "",
                    "accession": accessions[i] if i < len(accessions) else "",
                    "ticker": clean_ticker,
                })

            return transactions

        except Exception:
            self.logger.warning(
                "[SEC EDGAR] Failed to fetch insider data for %s",
                ticker, exc_info=True,
            )
            return []

    async def insider_summary(
        self,
        ticker: str,
        days: int = 90,
    ) -> dict[str, Any]:
        """
        取得內部人交易摘要統計。

        Returns
        -------
        {
            "total_filings": int,
            "recent_form4_count": int,
            "ticker": str,
            "activity_level": "high" | "moderate" | "low" | "none"
        }
        """
        transactions = await self.insider_transactions(ticker, limit=50)

        count = len(transactions)
        if count >= 10:
            level = "high"
        elif count >= 5:
            level = "moderate"
        elif count >= 1:
            level = "low"
        else:
            level = "none"

        return {
            "total_filings": count,
            "recent_form4_count": count,
            "ticker": ticker.replace(".L", "").upper(),
            "activity_level": level,
        }

    async def health_check(self) -> bool:
        """驗證 SEC EDGAR API 連線。"""
        try:
            await self._load_ticker_map()
            return len(self._ticker_to_cik) > 0
        except Exception:
            return False
