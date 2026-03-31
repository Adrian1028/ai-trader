"""
Finnhub Provider
================
Global news stream + sentiment analysis + basic quotes.
Used by: Sentiment Agent, Data Scout real-time news pipeline.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, AsyncIterator

import aiohttp

from .base_provider import BaseDataProvider

_REST_BASE = "https://finnhub.io/api/v1"
_WS_URL = "wss://ws.finnhub.io"

logger = logging.getLogger(__name__)


class FinnhubProvider(BaseDataProvider):

    def __init__(self, api_key: str) -> None:
        super().__init__("finnhub", api_key, _REST_BASE)
        self._ws: aiohttp.ClientWebSocketResponse | None = None

    async def health_check(self) -> bool:
        try:
            data = await self._authed_get("/stock/symbol", exchange="US", mic="XNYS")
            return isinstance(data, list)
        except Exception:
            return False

    # ── news & sentiment ──────────────────────────────────────────────

    async def company_news(
        self,
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        return await self._authed_get(
            "/company-news",
            symbol=symbol,
            **{"from": from_date or week_ago, "to": to_date or today},
        )

    async def general_news(self, category: str = "general") -> list[dict[str, Any]]:
        return await self._authed_get("/news", category=category)

    async def news_sentiment(self, symbol: str) -> dict[str, Any]:
        return await self._authed_get("/news-sentiment", symbol=symbol)

    # ── fundamentals ──────────────────────────────────────────────────

    async def basic_financials(self, symbol: str, metric: str = "all") -> dict[str, Any]:
        return await self._authed_get("/stock/metric", symbol=symbol, metric=metric)

    async def recommendation_trends(self, symbol: str) -> list[dict[str, Any]]:
        return await self._authed_get("/stock/recommendation", symbol=symbol)

    async def earnings_surprises(self, symbol: str) -> list[dict[str, Any]]:
        return await self._authed_get("/stock/earnings", symbol=symbol)

    # ── quote ─────────────────────────────────────────────────────────

    async def quote(self, symbol: str) -> dict[str, Any]:
        return await self._authed_get("/quote", symbol=symbol)

    # ── WebSocket: real-time trades ───────────────────────────────────

    async def stream_trades(self, symbols: list[str]) -> AsyncIterator[dict[str, Any]]:
        session = await self._ensure_session()
        async with session.ws_connect(f"{_WS_URL}?token={self._api_key}") as ws:
            self._ws = ws
            for sym in symbols:
                await ws.send_json({"type": "subscribe", "symbol": sym})

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    payload = json.loads(msg.data)
                    if payload.get("type") == "trade":
                        for trade in payload.get("data", []):
                            yield trade
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        self._ws = None

    # ── internal ──────────────────────────────────────────────────────

    async def _authed_get(self, path: str, **params: str) -> Any:
        params["token"] = self._api_key
        return await self._get(f"{_REST_BASE}{path}", params=params)

    async def close(self) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.close()
        await super().close()
