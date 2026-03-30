"""
Telegram Notification Agent
============================
Async Telegram Bot API client for real-time trade notifications.

Sends alerts for:
  - Trade executions (BUY/SELL with details)
  - Daily performance summaries
  - Watchdog alerts (drawdown breaker, API failures)
  - System start/stop events

Configuration (via .env):
  TELEGRAM_BOT_TOKEN  — from @BotFather
  TELEGRAM_CHAT_ID    — your chat/group ID (use @userinfobot to find)

Graceful degradation:
  If no token is configured, all send methods are silent no-ops.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Telegram message character limit
_MAX_MSG_LEN = 4096


class TelegramNotifier:
    """
    Lightweight async Telegram notifier using the Bot API.

    Usage
    -----
    ```python
    notifier = TelegramNotifier.from_env()
    await notifier.send_trade_alert(result_dict)
    await notifier.send_daily_summary(summary_dict)
    ```
    """

    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
    ) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)
        self._session: Any = None  # lazy aiohttp session

        if self._enabled:
            logger.info(
                "Telegram notifier enabled (chat_id=%s)", chat_id,
            )
        else:
            logger.info(
                "Telegram notifier disabled — set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID in .env to enable",
            )

    @classmethod
    def from_env(cls) -> "TelegramNotifier":
        return cls(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ══════════════════════════════════════════════════════════════
    # Core send method
    # ══════════════════════════════════════════════════════════════

    async def _send(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message via Telegram Bot API.

        Returns True if sent successfully, False otherwise.
        """
        if not self._enabled:
            return False

        # Truncate if too long
        if len(text) > _MAX_MSG_LEN:
            text = text[: _MAX_MSG_LEN - 20] + "\n\n... (truncated)"

        try:
            import aiohttp

            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            payload = {
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }

            async with self._session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    logger.debug("Telegram message sent (%d chars)", len(text))
                    return True
                else:
                    body = await resp.text()
                    logger.warning(
                        "Telegram API error %d: %s", resp.status, body[:200],
                    )
                    return False

        except ImportError:
            logger.warning("aiohttp not installed — Telegram disabled")
            self._enabled = False
            return False
        except Exception:
            logger.warning("Failed to send Telegram message", exc_info=True)
            return False

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ══════════════════════════════════════════════════════════════
    # High-level notification methods
    # ══════════════════════════════════════════════════════════════

    async def send_trade_alert(self, result: dict[str, Any]) -> bool:
        """
        Send a trade execution notification.

        Parameters
        ----------
        result : dict from run_cycle() results, containing:
            ticker, action, order_status, quantity, value, reasoning, etc.
        """
        action = result.get("action", "?")
        ticker = result.get("ticker", "?")
        status = result.get("order_status", "?")

        # Only notify on actual trades, not HOLDs
        if action == "HOLD":
            return False

        emoji = {"BUY": "\U0001f7e2", "SELL": "\U0001f534"}.get(action, "\u26aa")
        status_emoji = {
            "SUBMITTED": "\u2705",
            "COMPLIANCE_VETOED": "\u26d4",
        }.get(status, "\u2753")

        qty = result.get("quantity", 0)
        value = result.get("value", 0)
        reasoning = result.get("reasoning", "")

        # Truncate reasoning to keep message clean
        if len(reasoning) > 300:
            reasoning = reasoning[:300] + "..."

        now = datetime.now(timezone.utc).strftime("%H:%M UTC")

        text = (
            f"{emoji} <b>{action} {ticker}</b> {status_emoji}\n"
            f"\n"
            f"<b>Status:</b> {status}\n"
            f"<b>Quantity:</b> {qty:.4f}\n"
            f"<b>Value:</b> ${value:,.2f}\n"
            f"\n"
            f"<i>{reasoning}</i>\n"
            f"\n"
            f"\U0001f552 {now}"
        )

        return await self._send(text)

    async def send_cycle_summary(
        self,
        results: list[dict[str, Any]],
        elapsed: float,
    ) -> bool:
        """
        Send a summary after each 15-min trading cycle.
        Only sends if there were actual trades (not all HOLDs).
        """
        submitted = [r for r in results if r.get("order_status") == "SUBMITTED"]
        vetoed = [r for r in results if r.get("order_status") == "COMPLIANCE_VETOED"]

        if not submitted and not vetoed:
            return False  # All HOLDs — don't spam

        now = datetime.now(timezone.utc).strftime("%H:%M UTC")

        lines = [
            f"\U0001f4ca <b>Cycle Summary</b> ({now})",
            f"Duration: {elapsed:.1f}s | Scanned: {len(results)} stocks",
            "",
        ]

        if submitted:
            lines.append(f"\u2705 <b>Submitted ({len(submitted)}):</b>")
            for r in submitted:
                action = r.get("action", "?")
                ticker = r.get("ticker", "?")
                value = r.get("value", 0)
                lines.append(f"  {action} {ticker} ${value:,.2f}")

        if vetoed:
            lines.append(f"\u26d4 <b>Vetoed ({len(vetoed)}):</b>")
            for r in vetoed:
                lines.append(f"  {r.get('ticker', '?')}: {r.get('reasoning', '')[:80]}")

        return await self._send("\n".join(lines))

    async def send_daily_summary(self, summary: dict[str, Any]) -> bool:
        """
        Send the daily AI learning report.

        Parameters
        ----------
        summary : dict from system.learning_summary()
        """
        audit = summary.get("audit_stats", {})
        total = audit.get("total_trades", 0)
        win_rate = audit.get("win_rate", 0)
        avg_roi = audit.get("avg_roi", 0)
        opro_gen = summary.get("opro_generation", 0)
        episodes = summary.get("episodes_stored", 0)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        text = (
            f"\U0001f4c8 <b>Daily Report</b> ({now})\n"
            f"\n"
            f"<b>Trading:</b>\n"
            f"  Trades today: {total}\n"
            f"  Win rate: {win_rate:.1%}\n"
            f"  Avg ROI: {avg_roi:.4f}\n"
            f"\n"
            f"<b>AI Learning:</b>\n"
            f"  OPRO generation: {opro_gen}\n"
            f"  Episodes stored: {episodes}\n"
            f"\n"
            f"\U0001f916 Bot is healthy and learning."
        )

        return await self._send(text)

    async def send_system_event(self, event: str, details: str = "") -> bool:
        """Send system-level events (startup, shutdown, errors)."""
        emoji_map = {
            "START": "\U0001f680",
            "STOP": "\U0001f6d1",
            "ERROR": "\u26a0\ufe0f",
            "WATCHDOG": "\U0001f6a8",
        }
        emoji = emoji_map.get(event.upper(), "\u2139\ufe0f")
        now = datetime.now(timezone.utc).strftime("%H:%M UTC")

        text = f"{emoji} <b>{event.upper()}</b>\n"
        if details:
            text += f"\n{details}\n"
        text += f"\n\U0001f552 {now}"

        return await self._send(text)

    async def send_watchdog_alert(self, alert: str) -> bool:
        """Send critical watchdog alerts (drawdown breaker, kill switch)."""
        return await self.send_system_event("WATCHDOG", alert)
