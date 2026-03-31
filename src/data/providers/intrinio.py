"""
Intrinio Provider
=================
Deep fundamental data + SEC filing retrieval.
Used by: Fundamental Agent for in-depth balance sheet / filing analysis.
"""
from __future__ import annotations

import base64
from typing import Any

from .base_provider import BaseDataProvider

_BASE = "https://api-v2.intrinio.com"


class IntrinioProvider(BaseDataProvider):

    def __init__(self, api_key: str) -> None:
        super().__init__("intrinio", api_key, _BASE)

    async def health_check(self) -> bool:
        try:
            data = await self._authed_get("/companies", page_size="1")
            return "companies" in data
        except Exception:
            return False

    # ── company & fundamentals ────────────────────────────────────────

    async def company(self, identifier: str) -> dict[str, Any]:
        return await self._authed_get(f"/companies/{identifier}")

    async def securities(self, identifier: str) -> dict[str, Any]:
        return await self._authed_get(f"/companies/{identifier}/securities")

    async def fundamentals(
        self,
        identifier: str,
        statement_code: str = "income_statement",
        fiscal_year: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, str] = {"statement_code": statement_code}
        if fiscal_year:
            params["fiscal_year"] = str(fiscal_year)
        return await self._authed_get(
            f"/companies/{identifier}/fundamentals", **params,
        )

    async def standardized_financials(self, fundamental_id: str) -> dict[str, Any]:
        return await self._authed_get(f"/fundamentals/{fundamental_id}/standardized_financials")

    async def historical_data(
        self,
        identifier: str,
        tag: str,
        frequency: str = "yearly",
    ) -> dict[str, Any]:
        return await self._authed_get(
            f"/companies/{identifier}/historical_data/{tag}",
            frequency=frequency,
        )

    # ── SEC filings ───────────────────────────────────────────────────

    async def filings(
        self,
        identifier: str,
        report_type: str = "10-K",
        page_size: int = 10,
    ) -> dict[str, Any]:
        return await self._authed_get(
            f"/companies/{identifier}/filings",
            report_type=report_type,
            page_size=str(page_size),
        )

    # ── stock prices ──────────────────────────────────────────────────

    async def stock_prices(
        self,
        identifier: str,
        start_date: str = "",
        end_date: str = "",
        frequency: str = "daily",
        page_size: int = 100,
    ) -> dict[str, Any]:
        params: dict[str, str] = {
            "frequency": frequency,
            "page_size": str(page_size),
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return await self._authed_get(
            f"/securities/{identifier}/prices", **params,
        )

    # ── internal ──────────────────────────────────────────────────────

    async def _authed_get(self, path: str, **params: str) -> Any:
        params["api_key"] = self._api_key
        return await self._get(f"{_BASE}{path}", params=params)
