"""
Unit tests for Phase 4: Telegram Notification Agent
=====================================================
Tests cover:
  - TelegramNotifier initialization and from_env
  - Graceful no-op when disabled (no token)
  - Message formatting for trade alerts, cycle summaries, daily reports
  - send_trade_alert skips HOLD actions
  - send_cycle_summary skips all-HOLD cycles
  - System event messages
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.notifications.telegram import TelegramNotifier


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestInit:
    def test_disabled_without_token(self):
        n = TelegramNotifier(bot_token="", chat_id="123")
        assert n.enabled is False

    def test_disabled_without_chat_id(self):
        n = TelegramNotifier(bot_token="tok", chat_id="")
        assert n.enabled is False

    def test_enabled_with_both(self):
        n = TelegramNotifier(bot_token="tok", chat_id="123")
        assert n.enabled is True

    def test_from_env(self):
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "c"}):
            n = TelegramNotifier.from_env()
            assert n.enabled is True

    def test_from_env_empty(self):
        with patch.dict("os.environ", {}, clear=True):
            n = TelegramNotifier.from_env()
            assert n.enabled is False


class TestDisabledNoOp:
    """When disabled, all send methods should return False silently."""

    def test_send_trade_alert(self):
        n = TelegramNotifier()
        assert run(n.send_trade_alert({"action": "BUY", "ticker": "AAPL"})) is False

    def test_send_cycle_summary(self):
        n = TelegramNotifier()
        assert run(n.send_cycle_summary([], 1.0)) is False

    def test_send_daily_summary(self):
        n = TelegramNotifier()
        assert run(n.send_daily_summary({})) is False

    def test_send_system_event(self):
        n = TelegramNotifier()
        assert run(n.send_system_event("START")) is False

    def test_send_watchdog_alert(self):
        n = TelegramNotifier()
        assert run(n.send_watchdog_alert("test")) is False


class TestTradeAlert:
    def test_hold_is_skipped(self):
        n = TelegramNotifier(bot_token="t", chat_id="c")
        result = run(n.send_trade_alert({"action": "HOLD", "ticker": "AAPL"}))
        assert result is False  # HOLDs should not send

    def test_buy_formats_correctly(self):
        n = TelegramNotifier(bot_token="t", chat_id="c")
        sent_text = None

        async def mock_send(text, parse_mode="HTML"):
            nonlocal sent_text
            sent_text = text
            return True

        n._send = mock_send

        result = run(n.send_trade_alert({
            "action": "BUY",
            "ticker": "AAPL",
            "order_status": "SUBMITTED",
            "quantity": 5.5,
            "value": 825.0,
            "reasoning": "Strong signal",
        }))

        assert result is True
        assert "BUY" in sent_text
        assert "AAPL" in sent_text
        assert "$825.00" in sent_text
        assert "Strong signal" in sent_text

    def test_sell_formats_correctly(self):
        n = TelegramNotifier(bot_token="t", chat_id="c")
        sent_text = None

        async def mock_send(text, parse_mode="HTML"):
            nonlocal sent_text
            sent_text = text
            return True

        n._send = mock_send

        run(n.send_trade_alert({
            "action": "SELL",
            "ticker": "TSLA",
            "order_status": "SUBMITTED",
            "quantity": 2.0,
            "value": 500.0,
            "reasoning": "Overbought",
        }))

        assert "SELL" in sent_text
        assert "TSLA" in sent_text


class TestCycleSummary:
    def test_all_holds_skipped(self):
        n = TelegramNotifier(bot_token="t", chat_id="c")
        results = [
            {"action": "HOLD", "ticker": "AAPL"},
            {"action": "HOLD", "ticker": "MSFT"},
        ]
        assert run(n.send_cycle_summary(results, 5.0)) is False

    def test_submitted_trades_sent(self):
        n = TelegramNotifier(bot_token="t", chat_id="c")
        sent_text = None

        async def mock_send(text, parse_mode="HTML"):
            nonlocal sent_text
            sent_text = text
            return True

        n._send = mock_send

        results = [
            {"action": "BUY", "ticker": "AAPL", "order_status": "SUBMITTED", "value": 500},
            {"action": "HOLD", "ticker": "MSFT"},
        ]
        result = run(n.send_cycle_summary(results, 10.5))
        assert result is True
        assert "Cycle Summary" in sent_text
        assert "AAPL" in sent_text
        assert "Submitted (1)" in sent_text


class TestDailySummary:
    def test_formats_audit_stats(self):
        n = TelegramNotifier(bot_token="t", chat_id="c")
        sent_text = None

        async def mock_send(text, parse_mode="HTML"):
            nonlocal sent_text
            sent_text = text
            return True

        n._send = mock_send

        summary = {
            "audit_stats": {"total_trades": 8, "win_rate": 0.625, "avg_roi": 0.0032},
            "opro_generation": 3,
            "episodes_stored": 42,
        }
        run(n.send_daily_summary(summary))
        assert "8" in sent_text
        assert "62.5%" in sent_text
        assert "OPRO generation: 3" in sent_text


class TestSystemEvents:
    def test_start_event(self):
        n = TelegramNotifier(bot_token="t", chat_id="c")
        sent_text = None

        async def mock_send(text, parse_mode="HTML"):
            nonlocal sent_text
            sent_text = text
            return True

        n._send = mock_send

        run(n.send_system_event("START", "Bot launched"))
        assert "START" in sent_text
        assert "Bot launched" in sent_text

    def test_watchdog_alert(self):
        n = TelegramNotifier(bot_token="t", chat_id="c")
        sent_text = None

        async def mock_send(text, parse_mode="HTML"):
            nonlocal sent_text
            sent_text = text
            return True

        n._send = mock_send

        run(n.send_watchdog_alert("Daily loss exceeds 5%"))
        assert "WATCHDOG" in sent_text
        assert "Daily loss" in sent_text
