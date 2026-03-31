"""
Base Data Provider — 市場情報感測器基礎架構
============================================
所有外部數據源的共享基礎設施：
  - aiohttp session 管理 (Singleton)
  - 自動重試 + 指數退避
  - 記憶體快取 (TTL-based)
  - 標準化 OHLCV 輸出格式
  - 健康檢查介面
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import aiohttp


class DataProviderError(Exception):
    """數據源錯誤（含 provider 名稱）"""
    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


@dataclass
class OHLCVBar:
    """標準化 K 線資料結構 — 所有 provider 都輸出此格式"""
    timestamp: str        # ISO 8601 date string (e.g. "2024-01-15")
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: float | None = None

    @property
    def as_dict(self) -> dict[str, Any]:
        d = {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }
        if self.adjusted_close is not None:
            d["adjusted_close"] = self.adjusted_close
        return d


@dataclass
class CacheEntry:
    """帶 TTL 的快取條目"""
    data: Any
    expires_at: float


class BaseDataProvider(ABC):
    """
    所有 REST-based 數據源的共享基礎設施。
    提供：
      1. aiohttp session 管理 (Singleton)
      2. 自動重試 + 指數退避 (3 次重試)
      3. 記憶體快取 (可設定 TTL)
      4. 健康檢查介面
    """

    MAX_RETRIES = 3
    INITIAL_BACKOFF = 1.0

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        cache_ttl_seconds: float = 300.0,   # 預設快取 5 分鐘
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None
        self._cache: dict[str, CacheEntry] = {}
        self._cache_ttl = cache_ttl_seconds
        self.logger = logging.getLogger(f"provider.{name}")

        # 統計計數器
        self._request_count = 0
        self._cache_hits = 0
        self._error_count = 0

    # ── lifecycle ─────────────────────────────────────────────────

    async def __aenter__(self) -> "BaseDataProvider":
        await self._ensure_session()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── core HTTP with retry ──────────────────────────────────────

    async def _get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> Any:
        """
        帶快取和重試的 GET 請求。
        - 快取 key 由 URL + params 組成
        - 429/5xx 觸發指數退避重試
        """
        # 快取檢查
        if use_cache:
            cache_key = self._make_cache_key(url, params)
            cached = self._get_cached(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached

        session = await self._ensure_session()
        last_error: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                self._request_count += 1
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)

                        # 存入快取
                        if use_cache:
                            self._set_cached(cache_key, data)

                        return data

                    # 429 Too Many Requests → 退避重試
                    if resp.status == 429:
                        backoff = self.INITIAL_BACKOFF * (2 ** attempt)
                        self.logger.warning(
                            "[%s] 429 限流，退避 %.1fs (嘗試 %d/%d)",
                            self.name, backoff, attempt + 1, self.MAX_RETRIES,
                        )
                        await asyncio.sleep(backoff)
                        continue

                    # 5xx Server error → 退避重試
                    if resp.status >= 500:
                        backoff = self.INITIAL_BACKOFF * (2 ** attempt)
                        self.logger.warning(
                            "[%s] Server error %d，退避 %.1fs",
                            self.name, resp.status, backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue

                    # 4xx Client error → 不重試
                    body = await resp.text()
                    self._error_count += 1
                    raise DataProviderError(
                        self.name,
                        f"HTTP {resp.status} from {url}: {body[:300]}",
                    )

            except aiohttp.ClientError as e:
                last_error = e
                backoff = self.INITIAL_BACKOFF * (2 ** attempt)
                self.logger.error(
                    "[%s] 網路錯誤: %s，退避 %.1fs", self.name, e, backoff,
                )
                await asyncio.sleep(backoff)

        self._error_count += 1
        raise DataProviderError(
            self.name,
            f"已達最大重試次數 ({self.MAX_RETRIES}): {last_error}",
        )

    # ── cache ─────────────────────────────────────────────────────

    def _make_cache_key(self, url: str, params: dict[str, Any] | None) -> str:
        param_str = "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
        return f"{url}?{param_str}"

    def _get_cached(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() > entry.expires_at:
            del self._cache[key]
            return None
        return entry.data

    def _set_cached(self, key: str, data: Any) -> None:
        self._cache[key] = CacheEntry(
            data=data,
            expires_at=time.time() + self._cache_ttl,
        )

    def clear_cache(self) -> None:
        """手動清除所有快取"""
        self._cache.clear()

    # ── stats ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "requests": self._request_count,
            "cache_hits": self._cache_hits,
            "errors": self._error_count,
            "cache_size": len(self._cache),
        }

    # ── abstract ──────────────────────────────────────────────────

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and authenticated."""
