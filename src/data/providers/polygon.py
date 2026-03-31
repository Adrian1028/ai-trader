"""
Polygon.io 市場情報感測器 (Data Scout)
=======================================
即時行情 + REST K 線 + WebSocket 串流。

職責：
  - 歷史 K 線 (日/分/時) → 標準化 OHLCVBar 清單
  - 前一收盤 + 即時快照
  - 標的詳細資訊 (市值, 上市地, 行業)
  - WebSocket 即時交易串流

使用者：
  - TechnicalAgent：日內/日線技術分析
  - ExecutionAgent：滑價偵測
  - RiskAgent：VaR 歷史回報數據

API 限制：
  - 免費版：5 次/分
  - Starter+：無限 REST (delayed), 1 WS connection
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

import aiohttp

from .base_provider import BaseDataProvider, OHLCVBar, DataProviderError

_REST_BASE = "https://api.polygon.io"
_WS_URL = "wss://socket.polygon.io/stocks"

logger = logging.getLogger(__name__)


class PolygonProvider(BaseDataProvider):
    """
    Polygon.io REST + WebSocket 非同步客戶端。
    所有 REST 方法都返回標準化的 Python 資料結構。
    """

    def __init__(self, api_key: str, cache_ttl: float = 120.0) -> None:
        super().__init__("polygon", api_key, _REST_BASE, cache_ttl_seconds=cache_ttl)
        self._ws: aiohttp.ClientWebSocketResponse | None = None

    async def health_check(self) -> bool:
        try:
            data = await self._authed_get(
                "/v3/reference/tickers", limit="1",
            )
            return data.get("status") == "OK"
        except Exception:
            return False

    # ══════════════════════════════════════════════════════════════
    # K 線資料 (OHLCV) — 標準化輸出
    # ══════════════════════════════════════════════════════════════

    async def aggregates(
        self,
        ticker: str,
        multiplier: int = 1,
        timespan: str = "day",
        from_date: str = "",
        to_date: str = "",
        limit: int = 500,
    ) -> list[OHLCVBar]:
        """
        取得歷史 K 線資料，返回標準化 OHLCVBar 清單。

        Parameters
        ----------
        ticker : 股票代號 (e.g. "AAPL")
        multiplier : 時間倍數 (e.g. 1, 5, 15)
        timespan : "minute", "hour", "day", "week", "month"
        from_date : 起始日期 "YYYY-MM-DD"
        to_date : 結束日期 "YYYY-MM-DD"
        limit : 最多回傳幾根 K 線

        Returns
        -------
        list[OHLCVBar] : 標準化 K 線資料（由舊到新）
        """
        if not from_date:
            from_date = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.utcnow().strftime("%Y-%m-%d")

        url = (
            f"{_REST_BASE}/v2/aggs/ticker/{ticker}/range/"
            f"{multiplier}/{timespan}/{from_date}/{to_date}"
        )
        raw = await self._get(url, params={
            "apiKey": self._api_key,
            "limit": str(limit),
            "adjusted": "true",
            "sort": "asc",
        })

        return self._parse_aggregates(raw)

    async def aggregates_raw(
        self,
        ticker: str,
        multiplier: int = 1,
        timespan: str = "day",
        from_date: str = "",
        to_date: str = "",
        limit: int = 500,
    ) -> dict[str, Any]:
        """取得歷史 K 線（原始 JSON，向後相容）"""
        if not from_date:
            from_date = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.utcnow().strftime("%Y-%m-%d")

        url = (
            f"{_REST_BASE}/v2/aggs/ticker/{ticker}/range/"
            f"{multiplier}/{timespan}/{from_date}/{to_date}"
        )
        return await self._get(url, params={
            "apiKey": self._api_key,
            "limit": str(limit),
            "adjusted": "true",
        })

    async def daily_bars(
        self,
        ticker: str,
        days: int = 365,
    ) -> list[OHLCVBar]:
        """
        便利方法：取得過去 N 天的日 K 線。
        """
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.utcnow().strftime("%Y-%m-%d")
        return await self.aggregates(
            ticker, multiplier=1, timespan="day",
            from_date=from_date, to_date=to_date, limit=days + 10,
        )

    # ══════════════════════════════════════════════════════════════
    # 即時/延遲行情
    # ══════════════════════════════════════════════════════════════

    async def previous_close(self, ticker: str) -> dict[str, Any]:
        """
        取得前一交易日收盤資料。
        返回：{"close": 185.5, "high": 186.2, "low": 184.1, "open": 185.0, "volume": 50000000}
        """
        raw = await self._authed_get(
            f"/v2/aggs/ticker/{ticker}/prev", adjusted="true",
        )
        results = raw.get("results", [])
        if not results:
            return {}

        bar = results[0]
        return {
            "ticker": ticker,
            "close": bar.get("c", 0),
            "high": bar.get("h", 0),
            "low": bar.get("l", 0),
            "open": bar.get("o", 0),
            "volume": bar.get("v", 0),
            "vwap": bar.get("vw", 0),
            "timestamp": bar.get("t", 0),
        }

    async def snapshot(self, ticker: str) -> dict[str, Any]:
        """
        取得即時快照（需 Starter+ 方案）。
        返回完整的 day/min/prevDay 快照。
        """
        raw = await self._authed_get(
            f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}",
        )
        snapshot_data = raw.get("ticker", {})
        if not snapshot_data:
            return {}

        day = snapshot_data.get("day", {})
        prev = snapshot_data.get("prevDay", {})
        last_quote = snapshot_data.get("lastQuote", {})
        last_trade = snapshot_data.get("lastTrade", {})

        return {
            "ticker": ticker,
            "day": {
                "open": day.get("o", 0),
                "high": day.get("h", 0),
                "low": day.get("l", 0),
                "close": day.get("c", 0),
                "volume": day.get("v", 0),
                "vwap": day.get("vw", 0),
            },
            "prev_day": {
                "close": prev.get("c", 0),
                "volume": prev.get("v", 0),
            },
            "last_trade_price": last_trade.get("p", 0),
            "last_trade_size": last_trade.get("s", 0),
            "bid": last_quote.get("p", 0),
            "ask": last_quote.get("P", 0),
            "change_pct": snapshot_data.get("todaysChangePerc", 0),
        }

    async def current_price(self, ticker: str) -> float:
        """
        便利方法：取得當前/最近價格。
        優先用 snapshot，降級用 previous_close。
        """
        try:
            snap = await self.snapshot(ticker)
            if snap and snap.get("last_trade_price", 0) > 0:
                return snap["last_trade_price"]
        except DataProviderError:
            pass

        try:
            prev = await self.previous_close(ticker)
            return prev.get("close", 0)
        except DataProviderError:
            return 0.0

    # ══════════════════════════════════════════════════════════════
    # 標的參考資料
    # ══════════════════════════════════════════════════════════════

    async def ticker_details(self, ticker: str) -> dict[str, Any]:
        """
        取得標的詳細資訊。
        返回：市值, SIC 行業, 上市地, 員工數等。
        """
        raw = await self._authed_get(f"/v3/reference/tickers/{ticker}")
        results = raw.get("results", {})
        if not results:
            return {}

        return {
            "ticker": results.get("ticker", ""),
            "name": results.get("name", ""),
            "market_cap": results.get("market_cap", 0),
            "locale": results.get("locale", ""),
            "primary_exchange": results.get("primary_exchange", ""),
            "type": results.get("type", ""),
            "currency": results.get("currency_name", ""),
            "sic_code": results.get("sic_code", ""),
            "sic_description": results.get("sic_description", ""),
            "total_employees": results.get("total_employees", 0),
            "list_date": results.get("list_date", ""),
            "homepage_url": results.get("homepage_url", ""),
        }

    async def search_tickers(
        self,
        query: str,
        market: str = "stocks",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """搜尋標的"""
        raw = await self._authed_get(
            "/v3/reference/tickers",
            search=query, market=market, limit=str(limit), active="true",
        )
        results = raw.get("results", [])
        return [
            {
                "ticker": r.get("ticker", ""),
                "name": r.get("name", ""),
                "market": r.get("market", ""),
                "primary_exchange": r.get("primary_exchange", ""),
            }
            for r in results
        ]

    # ══════════════════════════════════════════════════════════════
    # WebSocket: 即時交易串流
    # ══════════════════════════════════════════════════════════════

    async def stream_trades(self, tickers: list[str]) -> AsyncIterator[dict[str, Any]]:
        """
        即時交易串流 (WebSocket)。
        Yields 標準化交易訊息：
          {"ticker": "AAPL", "price": 185.5, "size": 100, "timestamp": 1705312800000}
        """
        session = await self._ensure_session()
        async with session.ws_connect(_WS_URL) as ws:
            self._ws = ws

            # 認證
            await ws.send_json({"action": "auth", "params": self._api_key})
            auth_resp = await ws.receive_json()
            if not any(
                m.get("status") == "auth_success"
                for m in auth_resp
                if isinstance(m, dict)
            ):
                self.logger.warning("Polygon WS auth response: %s", auth_resp)

            # 訂閱
            channels = ",".join(f"T.{t}" for t in tickers)
            await ws.send_json({"action": "subscribe", "params": channels})
            self.logger.info("已訂閱即時串流: %s", channels)

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    for record in json.loads(msg.data):
                        if record.get("ev") == "T":
                            yield {
                                "ticker": record.get("sym", ""),
                                "price": record.get("p", 0),
                                "size": record.get("s", 0),
                                "timestamp": record.get("t", 0),
                                "conditions": record.get("c", []),
                            }
                        else:
                            yield record
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

        self._ws = None

    # ══════════════════════════════════════════════════════════════
    # 內部輔助方法
    # ══════════════════════════════════════════════════════════════

    async def _authed_get(self, path: str, **params: str) -> Any:
        """附帶 API Key 的 GET 請求"""
        params["apiKey"] = self._api_key
        return await self._get(f"{_REST_BASE}{path}", params=params)

    def _parse_aggregates(self, raw: dict[str, Any]) -> list[OHLCVBar]:
        """將 Polygon aggregates JSON 轉換為標準化 OHLCVBar 清單"""
        results = raw.get("results", [])
        if not results:
            self.logger.warning(
                "K 線資料為空, status=%s, count=%s",
                raw.get("status"), raw.get("resultsCount"),
            )
            return []

        bars: list[OHLCVBar] = []
        for bar in results:
            # Polygon 的 timestamp 是 Unix ms
            ts_ms = bar.get("t", 0)
            if ts_ms > 0:
                date_str = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            else:
                date_str = ""

            bars.append(OHLCVBar(
                timestamp=date_str,
                open=float(bar.get("o", 0)),
                high=float(bar.get("h", 0)),
                low=float(bar.get("l", 0)),
                close=float(bar.get("c", 0)),
                volume=float(bar.get("v", 0)),
                adjusted_close=None,   # Polygon 不分開提供 adjusted close
            ))

        self.logger.debug(
            "解析 %d 根 K 線 (%s → %s)",
            len(bars),
            bars[0].timestamp if bars else "?",
            bars[-1].timestamp if bars else "?",
        )
        return bars

    async def close(self) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.close()
        await super().close()
