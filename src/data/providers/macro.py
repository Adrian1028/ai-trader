"""
Macro Data Provider — 宏觀經濟數據源 (FRED API)
================================================
從聖路易斯聯儲 FRED API 取得宏觀經濟指標：
  - Fed Funds Rate（聯邦基金利率）
  - CPI（消費者物價指數）
  - PMI（製造業採購經理人指數）
  - 失業率
  - 國債殖利率曲線 (10Y-2Y spread)

FRED API：免費，120 requests/min，需申請 API key。
https://fred.stlouisfed.org/docs/api/fred/

使用者：
  - MacroAgent：宏觀因子分析
"""
from __future__ import annotations

import logging
from typing import Any

from src.data.providers.base_provider import BaseDataProvider

logger = logging.getLogger(__name__)

# FRED series IDs
_SERIES = {
    "fed_funds_rate": "FEDFUNDS",         # Federal Funds Effective Rate
    "cpi": "CPIAUCSL",                    # CPI All Urban Consumers
    "core_cpi": "CPILFESL",              # Core CPI (excl. food & energy)
    "pmi": "MANEMP",                      # ISM Manufacturing Employment (proxy)
    "unemployment": "UNRATE",             # Unemployment Rate
    "treasury_10y": "DGS10",             # 10-Year Treasury
    "treasury_2y": "DGS2",              # 2-Year Treasury
    "treasury_3m": "DTB3",              # 3-Month Treasury
    "vix": "VIXCLS",                     # CBOE VIX
}


class MacroDataProvider(BaseDataProvider):
    """
    FRED API 數據提供者。

    提供宏觀經濟指標的取得介面，供 MacroAgent 使用。
    免費 API，需申請 key：https://fred.stlouisfed.org/docs/api/api_key.html

    Parameters
    ----------
    api_key : FRED API key
    cache_ttl : 快取 TTL（宏觀數據更新慢，預設 1 小時）
    """

    def __init__(
        self,
        api_key: str,
        cache_ttl: float = 3600.0,
    ) -> None:
        super().__init__(
            name="macro_fred",
            api_key=api_key,
            base_url="https://api.stlouisfed.org/fred",
            cache_ttl_seconds=cache_ttl,
        )

    async def _fred_series(
        self,
        series_id: str,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """
        取得 FRED 時間序列的最近 N 筆觀測值。

        Returns list of {"date": "YYYY-MM-DD", "value": float}
        """
        if not self._api_key:
            return []

        url = f"{self._base_url}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }

        try:
            data = await self._get(url, params)
            observations = data.get("observations", [])
            result = []
            for obs in observations:
                val = obs.get("value", ".")
                if val == ".":
                    continue
                try:
                    result.append({
                        "date": obs.get("date", ""),
                        "value": float(val),
                    })
                except (ValueError, TypeError):
                    continue
            return result
        except Exception:
            self.logger.warning(
                "Failed to fetch FRED series %s", series_id, exc_info=True,
            )
            return []

    async def fed_funds_rate(self, months: int = 12) -> list[dict[str, Any]]:
        """取得聯邦基金利率（最近 N 個月）。"""
        return await self._fred_series(_SERIES["fed_funds_rate"], limit=months)

    async def cpi(self, months: int = 12) -> list[dict[str, Any]]:
        """取得 CPI 指數（最近 N 個月）。"""
        return await self._fred_series(_SERIES["cpi"], limit=months)

    async def core_cpi(self, months: int = 12) -> list[dict[str, Any]]:
        """取得核心 CPI（排除食品與能源）。"""
        return await self._fred_series(_SERIES["core_cpi"], limit=months)

    async def unemployment(self, months: int = 12) -> list[dict[str, Any]]:
        """取得失業率。"""
        return await self._fred_series(_SERIES["unemployment"], limit=months)

    async def treasury_yields(self) -> dict[str, float | None]:
        """
        取得國債殖利率（最新值）。

        Returns {"10y": float, "2y": float, "3m": float, "spread_10y_2y": float}
        """
        results: dict[str, float | None] = {}
        for key in ("treasury_10y", "treasury_2y", "treasury_3m"):
            data = await self._fred_series(_SERIES[key], limit=1)
            short_key = key.replace("treasury_", "")
            results[short_key] = data[0]["value"] if data else None

        # Calculate yield curve spread
        y10 = results.get("10y")
        y2 = results.get("2y")
        results["spread_10y_2y"] = (y10 - y2) if (y10 is not None and y2 is not None) else None

        return results

    async def vix(self) -> float | None:
        """取得 VIX 恐慌指數（最新值）。"""
        data = await self._fred_series(_SERIES["vix"], limit=1)
        return data[0]["value"] if data else None

    async def health_check(self) -> bool:
        """驗證 FRED API 連線。"""
        if not self._api_key:
            self.logger.warning("[FRED] No API key configured — degraded mode")
            return False
        try:
            data = await self._fred_series(_SERIES["fed_funds_rate"], limit=1)
            return len(data) > 0
        except Exception:
            return False
