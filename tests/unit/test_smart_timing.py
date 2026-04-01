"""
Unit Tests for SmartTiming (智能下單時機)
=========================================
測試：
  1. 開盤/收盤避免時段
  2. 宏觀事件避免
  3. 盈餘公告避免
  4. 波動率檢查
  5. 正常時段放行
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.agents.execution.timing import SmartTiming, TimingDecision


class TestOpenCloseWindow:
    def test_us_market_open_blocked(self):
        timing = SmartTiming(avoid_open_minutes=30)
        # 9:35 AM ET = 13:35 UTC — within 30 min of US open
        now = datetime(2026, 1, 5, 13, 35, tzinfo=timezone.utc)
        result = timing.evaluate("AAPL", current_time=now, market="US")
        assert not result.should_execute_now
        assert "open" in result.reason.lower()

    def test_us_market_close_blocked(self):
        timing = SmartTiming(avoid_close_minutes=15)
        # 3:50 PM ET = 19:50 UTC — within 15 min of US close (20:00 UTC)
        now = datetime(2026, 1, 5, 19, 50, tzinfo=timezone.utc)
        result = timing.evaluate("AAPL", current_time=now, market="US")
        assert not result.should_execute_now
        assert "close" in result.reason.lower()

    def test_us_midday_allowed(self):
        timing = SmartTiming()
        # 12:00 PM ET = 17:00 UTC — middle of trading day
        now = datetime(2026, 1, 5, 17, 0, tzinfo=timezone.utc)
        result = timing.evaluate("AAPL", current_time=now, market="US")
        assert result.should_execute_now

    def test_uk_market_open_blocked(self):
        timing = SmartTiming(avoid_open_minutes=30)
        # 8:10 UTC — within 30 min of UK open (8:00 UTC)
        now = datetime(2026, 1, 5, 8, 10, tzinfo=timezone.utc)
        result = timing.evaluate("BARC.L", current_time=now, market="UK")
        assert not result.should_execute_now

    def test_uk_midday_allowed(self):
        timing = SmartTiming()
        now = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
        result = timing.evaluate("BARC.L", current_time=now, market="UK")
        assert result.should_execute_now


class TestMacroEvents:
    def test_near_macro_event_blocked(self):
        calendar = [
            {
                "date": "2026-01-05",
                "event": "FOMC Decision",
                "hour_utc": 18,
                "minute_utc": 0,
            },
        ]
        timing = SmartTiming(macro_calendar=calendar)
        # 17:45 UTC — within 30 min of FOMC
        now = datetime(2026, 1, 5, 17, 45, tzinfo=timezone.utc)
        result = timing.evaluate("AAPL", current_time=now, market="US")
        assert not result.should_execute_now
        assert "FOMC" in result.reason

    def test_far_from_macro_event_allowed(self):
        calendar = [
            {
                "date": "2026-01-05",
                "event": "FOMC Decision",
                "hour_utc": 18,
                "minute_utc": 0,
            },
        ]
        timing = SmartTiming(macro_calendar=calendar)
        # 15:00 UTC — 3 hours before FOMC
        now = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)
        result = timing.evaluate("AAPL", current_time=now, market="US")
        assert result.should_execute_now


class TestEarningsProximity:
    def test_near_earnings_blocked(self):
        timing = SmartTiming()
        now = datetime(2026, 1, 5, 16, 0, tzinfo=timezone.utc)
        result = timing.evaluate(
            "AAPL",
            current_time=now,
            market="US",
            earnings_date="2026-01-05T18:00:00",
        )
        assert not result.should_execute_now
        assert "earnings" in result.reason.lower()


class TestVolatilitySpike:
    def test_extreme_vol_blocked(self):
        timing = SmartTiming()
        now = datetime(2026, 1, 5, 17, 0, tzinfo=timezone.utc)
        result = timing.evaluate(
            "AAPL",
            current_time=now,
            market="US",
            recent_atr=8.0,
            current_price=100.0,
        )
        assert not result.should_execute_now
        assert "volatility" in result.reason.lower()

    def test_normal_vol_allowed(self):
        timing = SmartTiming()
        now = datetime(2026, 1, 5, 17, 0, tzinfo=timezone.utc)
        result = timing.evaluate(
            "AAPL",
            current_time=now,
            market="US",
            recent_atr=2.0,
            current_price=100.0,
        )
        assert result.should_execute_now
