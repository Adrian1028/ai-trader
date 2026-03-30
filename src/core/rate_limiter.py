"""
非同步速率限制器 (Async Rate Limiter)
=====================================
專門處理 Trading 212 基於帳戶的 x-ratelimit 標頭，
並在逼近閾值時觸發退避，保護帳戶不被封鎖。
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    非同步速率限制器 (Rate Limiter)
    專門處理 Trading 212 基於帳戶的 x-ratelimit 標頭，並在逼近閾值時觸發退避。
    """

    def __init__(self, safe_margin: int = 2):
        self.remaining: int = 50   # 預設剩餘額度
        self.reset_time: float = time.time()
        self.safe_margin = safe_margin  # 安全邊際，保留幾次請求不發送以防萬一
        self._lock = asyncio.Lock()

    async def acquire(self):
        """獲取請求許可，若額度不足則非同步等待至重置時間"""
        async with self._lock:
            if self.remaining <= self.safe_margin:
                now = time.time()
                sleep_duration = self.reset_time - now

                if sleep_duration > 0:
                    logger.warning(
                        "[RateLimiter] 逼近 API 速率限制 (剩餘: %d)。"
                        "暫停執行 %.2f 秒等待重置...",
                        self.remaining,
                        sleep_duration,
                    )
                    await asyncio.sleep(sleep_duration)

                # 等待結束後，預設重置額度 (實際額度會由下一次 API 回應更新)
                self.remaining = 50

    def update_from_headers(self, headers: dict):
        """從 API 回應的標頭中提取並更新速率限制狀態"""
        try:
            if "x-ratelimit-remaining" in headers:
                self.remaining = int(headers["x-ratelimit-remaining"])

            if "x-ratelimit-reset" in headers:
                # T212 回傳的是 Unix Timestamp
                self.reset_time = float(headers["x-ratelimit-reset"])

            logger.debug(
                "[RateLimiter] 更新狀態: 剩餘=%d, 重置時間=%s",
                self.remaining,
                self.reset_time,
            )
        except ValueError as e:
            logger.error("[RateLimiter] 解析標頭失敗: %s", e)


# ── 向後相容別名 (Backward-compatible aliases) ───────────────────
# 已存在的模組仍可使用舊的類別名稱

@dataclass
class RateLimitState:
    """Tracks the current rate-limit window from API response headers."""
    remaining: int | None = None
    reset_timestamp: float | None = None
    last_updated: float = field(default_factory=time.monotonic)

    def update_from_headers(self, headers: dict[str, str]) -> None:
        raw_remaining = headers.get("x-ratelimit-remaining")
        raw_reset = headers.get("x-ratelimit-reset")
        if raw_remaining is not None:
            self.remaining = int(raw_remaining)
        if raw_reset is not None:
            self.reset_timestamp = float(raw_reset)
        self.last_updated = time.monotonic()


class ExponentialBackoff:
    """
    Stateful backoff calculator.
    Doubles the wait on each consecutive retry, up to a ceiling.
    Resets on a successful call.
    """

    def __init__(
        self,
        base: float = 1.0,
        multiplier: float = 2.0,
        ceiling: float = 60.0,
    ) -> None:
        self._base = base
        self._multiplier = multiplier
        self._ceiling = ceiling
        self._attempt = 0

    @property
    def delay(self) -> float:
        d = self._base * (self._multiplier ** self._attempt)
        return min(d, self._ceiling)

    def escalate(self) -> float:
        """Return current delay and move to the next level."""
        d = self.delay
        self._attempt += 1
        logger.warning("Backoff escalated to %.2fs (attempt %d)", d, self._attempt)
        return d

    def reset(self) -> None:
        self._attempt = 0


class RateLimitGuard:
    """
    Async gate that blocks callers when the API rate limit is nearly exhausted,
    then releases them once the window resets.
    """

    def __init__(self, buffer: int = 5, backoff: ExponentialBackoff | None = None) -> None:
        self._buffer = buffer
        self._state = RateLimitState()
        self._backoff = backoff or ExponentialBackoff()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> RateLimitState:
        return self._state

    def record_response(self, headers: dict[str, str]) -> None:
        self._state.update_from_headers(headers)

    async def wait_if_needed(self) -> None:
        if self._state.remaining is not None and self._state.remaining <= self._buffer:
            async with self._lock:
                if self._state.remaining is not None and self._state.remaining <= self._buffer:
                    wait = self._compute_wait()
                    logger.warning(
                        "Rate limit nearly exhausted (remaining=%s). "
                        "Sleeping %.2fs before next request.",
                        self._state.remaining,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    self._state.remaining = None

    async def backoff_on_429(self) -> None:
        delay = self._backoff.escalate()
        logger.warning("HTTP 429 received — backing off for %.2fs", delay)
        await asyncio.sleep(delay)

    def on_success(self) -> None:
        self._backoff.reset()

    def _compute_wait(self) -> float:
        if self._state.reset_timestamp is not None:
            delta = self._state.reset_timestamp - time.time()
            if delta > 0:
                return delta
        return self._backoff.escalate()
