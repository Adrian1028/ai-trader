"""
Trading 212 API 非同步客戶端 (Async Client)
============================================
負責：環境路由、身分驗證、速率限制整合、指數退避與游標分頁處理。

Usage
-----
```python
from config.settings import config

client = Trading212Client()
await client.get_account_info()
await client.close()
```
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import aiohttp

from config.settings import config, Environment, T212Config
from src.core.rate_limiter import (
    RateLimiter,
    ExponentialBackoff,
    RateLimitGuard,
)

logger = logging.getLogger(__name__)


class Trading212APIError(Exception):
    """Non-retryable API error."""

    def __init__(self, status: int, body: str, url: str) -> None:
        self.status = status
        self.body = body
        self.url = url
        super().__init__(f"HTTP {status} from {url}: {body}")


class Trading212Client:
    """
    Trading 212 API 基礎非同步客戶端。
    負責：環境路由、身分驗證、速率限制整合、指數退避與游標分頁處理。

    同時支援兩種初始化方式：
      1. 無參數：使用全域 ``config`` 單例
      2. 傳入 ``T212Config``：向後相容舊有上層模組
    """

    def __init__(self, t212_config: T212Config | None = None) -> None:
        # 優先使用傳入的 T212Config (向後相容)，否則從全域 config 取值
        if t212_config is not None:
            self._auth_header = t212_config.auth_header
            self._base_url = t212_config.base_url
            self._max_retries = 5
            self._initial_backoff = t212_config.backoff_base_seconds
            self._max_backoff = t212_config.backoff_max_seconds
            # 保留舊 guard 以支持上層模組
            self._guard = RateLimitGuard(
                buffer=t212_config.rate_limit_buffer,
                backoff=ExponentialBackoff(
                    base=t212_config.backoff_base_seconds,
                    multiplier=t212_config.backoff_multiplier,
                    ceiling=t212_config.backoff_max_seconds,
                ),
            )
            self._config = t212_config
        else:
            self._auth_header = config.AUTH_HEADER
            self._base_url = config.BASE_URL
            self._max_retries = config.MAX_RETRIES
            self._initial_backoff = config.INITIAL_BACKOFF
            self._max_backoff = config.MAX_BACKOFF
            self._guard = None
            self._config = None

        self.headers = {
            "Authorization": self._auth_header,
            "Content-Type": "application/json",
        }
        self.rate_limiter = RateLimiter()
        self._session: Optional[aiohttp.ClientSession] = None

    # ── lifecycle ─────────────────────────────────────────────────────

    async def __aenter__(self) -> "Trading212Client":
        await self.get_session()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def get_session(self) -> aiohttp.ClientSession:
        """獲取或建立 aiohttp session (Singleton 模式)"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self.headers)
        return self._session

    async def close(self):
        """關閉連線 (優雅停機時呼叫)"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── 核心請求邏輯 ──────────────────────────────────────────────────

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """核心請求邏輯：包含速率限制檢查與指數退避演算法"""
        url = f"{self._base_url}{endpoint}"
        session = await self.get_session()

        for attempt in range(self._max_retries):
            # 1. 執行前先確認是否有 API 額度
            if self._guard:
                await self._guard.wait_if_needed()
            else:
                await self.rate_limiter.acquire()

            try:
                async with session.request(method, url, **kwargs) as response:
                    # 2. 每次請求後更新速率狀態
                    headers_dict = dict(response.headers)
                    self.rate_limiter.update_from_headers(headers_dict)
                    if self._guard:
                        self._guard.record_response(headers_dict)

                    if response.status in (200, 201):
                        if self._guard:
                            self._guard.on_success()
                        body_text = await response.text()
                        if not body_text:
                            return {}
                        return await response.json(content_type=None)

                    error_text = await response.text()

                    # 429 Too Many Requests
                    if response.status == 429:
                        if self._guard:
                            await self._guard.backoff_on_429()
                        else:
                            backoff = min(
                                self._initial_backoff * (2 ** attempt),
                                self._max_backoff,
                            )
                            logger.warning(
                                "觸發 429 限流，啟動指數退避。等待 %.1f 秒 (嘗試 %d/%d)",
                                backoff, attempt + 1, self._max_retries,
                            )
                            await asyncio.sleep(backoff)
                        continue

                    # 5xx Server errors — retry with backoff
                    if response.status >= 500:
                        backoff = min(
                            self._initial_backoff * (2 ** attempt),
                            self._max_backoff,
                        )
                        logger.warning(
                            "Server error %d — retry %d/%d in %.1fs",
                            response.status, attempt + 1, self._max_retries, backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue

                    # 4xx Client errors (non-429) — non-retryable
                    raise Trading212APIError(
                        response.status, error_text, f"{method} {endpoint}"
                    )

            except aiohttp.ClientError as e:
                backoff = min(
                    self._initial_backoff * (2 ** attempt),
                    self._max_backoff,
                )
                logger.error("網路連線錯誤: %s。等待 %.1f 秒重試...", e, backoff)
                await asyncio.sleep(backoff)

        raise Trading212APIError(
            429,
            f"API 請求失敗，已達最大重試次數 ({self._max_retries})",
            f"{method} {endpoint}",
        )

    # ── public HTTP verbs ─────────────────────────────────────────────

    async def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        return await self._request("GET", path, params=params)

    async def post(
        self,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> Any:
        return await self._request("POST", path, json=json)

    async def delete(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        return await self._request("DELETE", path, params=params)

    # ── pagination helper ─────────────────────────────────────────────

    async def get_all_pages(
        self,
        path: str,
        *,
        cursor_param: str = "cursor",
        items_key: str = "items",
        next_cursor_key: str = "nextPageCursor",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Exhaustive cursor-based pagination.
        Returns a flat list of all items across every page.
        """
        collected: list[dict[str, Any]] = []
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit

        while True:
            page = await self.get(path, params=params)
            items = page.get(items_key, [])
            collected.extend(items)

            next_cursor = page.get(next_cursor_key)
            if not next_cursor:
                break
            params[cursor_param] = next_cursor

        return collected

    # ── 實用 API 端點封裝 ─────────────────────────────────────────────

    async def get_account_info(self) -> Dict[str, Any]:
        """取得帳戶基本資訊與淨值"""
        return await self._request("GET", "/equity/account/info")

    async def account_info(self) -> Dict[str, Any]:
        """Alias for get_account_info (向後相容)"""
        return await self.get_account_info()

    async def account_cash(self) -> Dict[str, Any]:
        return await self.get("/equity/account/cash")

    async def portfolio(self) -> list[dict[str, Any]]:
        return await self.get("/equity/portfolio")

    async def exchange_instruments(self) -> list[dict[str, Any]]:
        """Fetch the full instrument list (used for ISIN mapping)."""
        return await self.get("/equity/metadata/instruments")

    async def place_market_order(self, ticker: str, quantity: float) -> Dict[str, Any]:
        """
        發送市價單。
        注意：依據系統設計，若要賣出，quantity 必須為負數。此邏輯會在 ExecutionAgent 進行驗證。
        """
        payload = {
            "instrumentCode": ticker,
            "quantity": quantity,
        }
        return await self._request("POST", "/equity/orders/market", json=payload)

    async def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Place an equity order (generic payload)."""
        return await self.post("/equity/orders/market", json=payload)

    async def place_limit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.post("/equity/orders/limit", json=payload)

    async def place_stop_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.post("/equity/orders/stop", json=payload)

    async def cancel_order(self, order_id: int) -> Any:
        return await self.delete(f"/equity/orders/{order_id}")

    async def cancel_all_orders(self) -> Any:
        return await self.delete("/equity/orders")

    async def order_history(
        self, cursor: str | None = None, limit: int = 50,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self.get("/equity/history/orders", params=params)

    # ── environment helpers ───────────────────────────────────────────

    @property
    def environment(self) -> str:
        if self._config:
            return self._config.environment
        return Environment.DEMO if config.ENV == "demo" else Environment.LIVE

    @property
    def is_live(self) -> bool:
        return config.ENV == "live"

    @property
    def base_url(self) -> str:
        return self._base_url
