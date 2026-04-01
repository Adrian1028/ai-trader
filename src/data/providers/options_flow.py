"""
Options Flow Provider — 期權異常流量數據源
==========================================
從 Unusual Whales API 取得期權市場異常活動數據。

追蹤指標：
  - 異常期權交易量（大戶動向）
  - Put/Call Ratio（市場恐慌/貪婪指標）
  - 高額期權金交易（Smart Money 流向）

使用者：
  - OptionsFlowAgent：期權流量信號分析
"""
from __future__ import annotations

import logging
from typing import Any

from src.data.providers.base_provider import BaseDataProvider

logger = logging.getLogger(__name__)


class OptionsFlowProvider(BaseDataProvider):
    """
    期權異常流量數據提供者。

    使用 Unusual Whales API (或相容替代) 取得期權市場數據。

    Parameters
    ----------
    api_key : Unusual Whales API key
    cache_ttl : 快取 TTL（期權數據變化快，預設 1 分鐘）
    """

    def __init__(
        self,
        api_key: str,
        cache_ttl: float = 60.0,
    ) -> None:
        super().__init__(
            name="options_flow",
            api_key=api_key,
            base_url="https://api.unusualwhales.com/api",
            cache_ttl_seconds=cache_ttl,
        )

    async def _authed_get(
        self, endpoint: str, params: dict[str, Any] | None = None,
    ) -> Any:
        """帶認證的 GET 請求。"""
        url = f"{self._base_url}/{endpoint}"
        params = params or {}
        params["token"] = self._api_key
        return await self._get(url, params)

    async def unusual_activity(
        self, ticker: str, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        取得異常期權交易活動。

        Returns list of unusual options trades:
        [{
            "ticker": str,
            "strike": float,
            "expiry": str,
            "type": "call"|"put",
            "volume": int,
            "open_interest": int,
            "premium": float,
            "sentiment": "bullish"|"bearish"|"neutral"
        }]
        """
        if not self._api_key:
            return []

        try:
            data = await self._authed_get(
                f"stock/{ticker}/options-flow",
                {"limit": str(limit)},
            )
            flows = data.get("data", []) if isinstance(data, dict) else []

            result = []
            for flow in flows[:limit]:
                sentiment = "neutral"
                opt_type = str(flow.get("put_call", "")).lower()
                if opt_type == "call":
                    sentiment = "bullish"
                elif opt_type == "put":
                    sentiment = "bearish"

                result.append({
                    "ticker": ticker,
                    "strike": float(flow.get("strike_price", 0)),
                    "expiry": flow.get("expiration_date", ""),
                    "type": opt_type or "unknown",
                    "volume": int(flow.get("volume", 0)),
                    "open_interest": int(flow.get("open_interest", 0)),
                    "premium": float(flow.get("premium", 0)),
                    "sentiment": sentiment,
                })
            return result

        except Exception:
            self.logger.warning(
                "[OptionsFlow] Failed to fetch unusual activity for %s",
                ticker, exc_info=True,
            )
            return []

    async def put_call_ratio(self, ticker: str) -> dict[str, Any]:
        """
        取得 Put/Call Ratio。

        Returns {"ratio": float, "put_volume": int, "call_volume": int}
        """
        if not self._api_key:
            return {}

        try:
            data = await self._authed_get(f"stock/{ticker}/options-volume")
            if not isinstance(data, dict):
                return {}

            vol_data = data.get("data", {})
            call_vol = int(vol_data.get("call_volume", 0))
            put_vol = int(vol_data.get("put_volume", 0))

            ratio = put_vol / call_vol if call_vol > 0 else 0.0

            return {
                "ratio": ratio,
                "put_volume": put_vol,
                "call_volume": call_vol,
            }

        except Exception:
            self.logger.warning(
                "[OptionsFlow] Failed to fetch P/C ratio for %s",
                ticker, exc_info=True,
            )
            return {}

    async def smart_money_flow(
        self, ticker: str, min_premium: float = 100_000.0,
    ) -> dict[str, Any]:
        """
        分析大額期權交易（Smart Money 流向）。

        篩選 premium > min_premium 的交易，統計看多/看空比例。

        Returns {
            "bullish_premium": float,
            "bearish_premium": float,
            "net_flow": float,
            "large_trades_count": int
        }
        """
        flows = await self.unusual_activity(ticker, limit=50)

        bullish_premium = 0.0
        bearish_premium = 0.0
        large_count = 0

        for flow in flows:
            premium = flow.get("premium", 0)
            if premium < min_premium:
                continue

            large_count += 1
            if flow.get("sentiment") == "bullish":
                bullish_premium += premium
            elif flow.get("sentiment") == "bearish":
                bearish_premium += premium

        return {
            "bullish_premium": bullish_premium,
            "bearish_premium": bearish_premium,
            "net_flow": bullish_premium - bearish_premium,
            "large_trades_count": large_count,
        }

    async def health_check(self) -> bool:
        """驗證 Unusual Whales API 連線。"""
        if not self._api_key:
            self.logger.warning(
                "[OptionsFlow] No API key — degraded mode",
            )
            return False
        try:
            data = await self._authed_get("market/overview")
            return isinstance(data, dict)
        except Exception:
            return False
