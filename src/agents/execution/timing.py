"""
Smart Timing — 智能下單時機選擇
================================
分析當前時間和市場狀態，判斷是否應立即執行訂單或延遲。

避免時段：
  1. 開盤前 30 分鐘（流動性差、波動大）
  2. 收盤前 15 分鐘（價格波動加劇）
  3. 重大宏觀數據發布前後（FOMC、CPI、非農）
  4. 個股盈餘公告前後

使用者：
  - ExecutionAgent：下單前檢查是否應延遲執行
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TimingDecision:
    """下單時機判斷結果。"""
    should_execute_now: bool
    reason: str
    suggested_delay_seconds: float = 0.0
    volatility_score: float = 0.0


# 已知的美國重大宏觀事件排程（月份, 日份大約範圍, UTC 時間）
# 實際日期需要外部日曆，這裡用時段規避
_FOMC_RELEASE_HOUR_UTC = 18     # FOMC 公告通常 14:00 ET = 18:00 UTC
_CPI_RELEASE_HOUR_UTC = 12      # CPI 通常 08:30 ET = 12:30 UTC
_JOBS_RELEASE_HOUR_UTC = 12     # 非農通常 08:30 ET = 12:30 UTC


class SmartTiming:
    """
    智能下單時機分析器。

    根據市場時段、宏觀事件日曆和即時波動率，
    判斷當前是否適合執行訂單。

    Parameters
    ----------
    avoid_open_minutes : 開盤後避免下單的分鐘數
    avoid_close_minutes : 收盤前避免下單的分鐘數
    macro_calendar : 宏觀事件日曆 [{date, event, time_utc}]
    """

    def __init__(
        self,
        avoid_open_minutes: int = 30,
        avoid_close_minutes: int = 15,
        macro_calendar: list[dict[str, Any]] | None = None,
    ) -> None:
        self.avoid_open_minutes = avoid_open_minutes
        self.avoid_close_minutes = avoid_close_minutes
        self._calendar = macro_calendar or []

    def evaluate(
        self,
        ticker: str,
        current_time: datetime | None = None,
        market: str = "US",
        earnings_date: str | None = None,
        recent_atr: float | None = None,
        current_price: float | None = None,
    ) -> TimingDecision:
        """
        評估當前是否適合下單。

        Parameters
        ----------
        ticker : 股票代碼
        current_time : 當前時間（UTC）
        market : "US" 或 "UK"
        earnings_date : 盈餘公告日（ISO 格式）
        recent_atr : 近期 ATR
        current_price : 當前價格
        """
        now = current_time or datetime.now(timezone.utc)

        # Check 1: Market open/close windows
        window_check = self._check_open_close_window(now, market)
        if window_check is not None:
            return window_check

        # Check 2: Macro event proximity
        macro_check = self._check_macro_events(now)
        if macro_check is not None:
            return macro_check

        # Check 3: Earnings proximity
        if earnings_date:
            earnings_check = self._check_earnings_proximity(now, earnings_date)
            if earnings_check is not None:
                return earnings_check

        # Check 4: Intraday volatility spike
        if recent_atr is not None and current_price is not None and current_price > 0:
            vol_check = self._check_volatility_spike(recent_atr, current_price)
            if vol_check is not None:
                return vol_check

        return TimingDecision(
            should_execute_now=True,
            reason="All timing checks passed — clear to execute",
        )

    def _check_open_close_window(
        self, now: datetime, market: str,
    ) -> TimingDecision | None:
        """檢查是否處於開盤/收盤高波動時段。"""
        hour = now.hour
        minute = now.minute
        total_min = hour * 60 + minute

        if market == "UK":
            # LSE: 08:00–16:30 UTC
            market_open = 8 * 60       # 08:00 UTC
            market_close = 16 * 60 + 30  # 16:30 UTC
        else:
            # NYSE: 13:30–20:00 UTC (9:30–16:00 ET)
            market_open = 13 * 60 + 30  # 13:30 UTC
            market_close = 20 * 60      # 20:00 UTC

        # Before market open or after market close → don't block, scheduler handles this
        if total_min < market_open or total_min > market_close:
            return None

        # Open avoidance window
        open_end = market_open + self.avoid_open_minutes
        if total_min < open_end:
            delay = (open_end - total_min) * 60
            return TimingDecision(
                should_execute_now=False,
                reason=f"Within {self.avoid_open_minutes}min of {market} market open — "
                       f"high volatility window",
                suggested_delay_seconds=delay,
            )

        # Close avoidance window
        close_start = market_close - self.avoid_close_minutes
        if total_min > close_start:
            delay = (market_close - total_min + 5) * 60  # wait until after close + buffer
            return TimingDecision(
                should_execute_now=False,
                reason=f"Within {self.avoid_close_minutes}min of {market} market close — "
                       f"high volatility window",
                suggested_delay_seconds=delay,
            )

        return None

    def _check_macro_events(self, now: datetime) -> TimingDecision | None:
        """檢查是否接近重大宏觀數據發布時間。"""
        today_str = now.strftime("%Y-%m-%d")

        for event in self._calendar:
            event_date = event.get("date", "")
            if event_date != today_str:
                continue

            event_hour = event.get("hour_utc", 0)
            event_minute = event.get("minute_utc", 0)
            event_total = event_hour * 60 + event_minute
            now_total = now.hour * 60 + now.minute

            # Avoid ±30 min around macro events
            if abs(now_total - event_total) <= 30:
                delay = max(0, (event_total + 30 - now_total)) * 60
                return TimingDecision(
                    should_execute_now=False,
                    reason=f"Within 30min of macro event: {event.get('event', 'Unknown')} — "
                           f"market may be volatile",
                    suggested_delay_seconds=delay,
                )

        return None

    def _check_earnings_proximity(
        self, now: datetime, earnings_date: str,
    ) -> TimingDecision | None:
        """檢查是否接近個股盈餘公告日。"""
        try:
            earn_dt = datetime.fromisoformat(earnings_date).replace(
                tzinfo=timezone.utc,
            )
            delta = abs((earn_dt - now).total_seconds())

            # Avoid trading within 4 hours of earnings announcement
            if delta < 4 * 3600:
                return TimingDecision(
                    should_execute_now=False,
                    reason=f"Within 4 hours of earnings announcement — "
                           f"high uncertainty",
                    suggested_delay_seconds=delta + 3600,  # wait until 1hr after
                )
        except (ValueError, TypeError):
            pass

        return None

    def _check_volatility_spike(
        self, atr: float, current_price: float,
    ) -> TimingDecision | None:
        """檢查日內波動率是否異常偏高。"""
        atr_pct = (atr / current_price) * 100.0

        # ATR% > 5% considered extreme volatility
        if atr_pct > 5.0:
            return TimingDecision(
                should_execute_now=False,
                reason=f"Extreme intraday volatility (ATR={atr_pct:.1f}%) — "
                       f"consider waiting for stabilization",
                suggested_delay_seconds=900,  # wait 15 min
                volatility_score=atr_pct,
            )

        return None
