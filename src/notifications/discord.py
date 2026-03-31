"""
Discord Notification Agent
===========================
Async Discord Webhook client for real-time trade notifications.

Sends alerts for:
  - Trade executions (BUY/SELL with details)
  - Cycle summaries
  - Daily performance summaries
  - Watchdog alerts (drawdown breaker, API failures)
  - System start/stop events

Configuration (via .env):
  DISCORD_WEBHOOK_URL — from Discord channel settings > Integrations > Webhooks

Graceful degradation:
  If no webhook URL is configured, all send methods are silent no-ops.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Discord webhook message limit
_MAX_MSG_LEN = 2000


class DiscordNotifier:
    """
    Lightweight async Discord notifier using Webhooks.

    Usage
    -----
    ```python
    notifier = DiscordNotifier.from_env()
    await notifier.send_trade_alert(result_dict)
    await notifier.send_daily_summary(summary_dict)
    ```
    """

    def __init__(self, webhook_url: str = "") -> None:
        self._webhook_url = webhook_url
        self._enabled = bool(webhook_url)
        self._session: Any = None  # lazy aiohttp session

        if self._enabled:
            logger.info("Discord notifier enabled")
        else:
            logger.info(
                "Discord notifier disabled — set DISCORD_WEBHOOK_URL in .env to enable"
            )

    @classmethod
    def from_env(cls) -> "DiscordNotifier":
        return cls(webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""))

    @property
    def enabled(self) -> bool:
        return self._enabled

    # -- Core send method ------------------------------------------

    async def _send(self, content: str) -> bool:
        """Send a message via Discord Webhook. Returns True if successful."""
        if not self._enabled:
            return False

        if len(content) > _MAX_MSG_LEN:
            content = content[: _MAX_MSG_LEN - 20] + "\n\n... (truncated)"

        try:
            import aiohttp

            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            payload = {
                "content": content,
                "username": "AI Trading Bot",
            }

            async with self._session.post(
                self._webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status in (200, 204):
                    logger.debug("Discord message sent (%d chars)", len(content))
                    return True
                else:
                    body = await resp.text()
                    logger.warning("Discord webhook error %d: %s", resp.status, body[:200])
                    return False

        except ImportError:
            logger.warning("aiohttp not installed — Discord disabled")
            self._enabled = False
            return False
        except Exception:
            logger.warning("Failed to send Discord message", exc_info=True)
            return False

    async def _send_embed(self, embed: dict[str, Any]) -> bool:
        """Send a rich embed via Discord Webhook."""
        if not self._enabled:
            return False

        try:
            import aiohttp

            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            payload = {
                "username": "AI Trading Bot",
                "embeds": [embed],
            }

            async with self._session.post(
                self._webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status in (200, 204):
                    logger.debug("Discord embed sent")
                    return True
                else:
                    body = await resp.text()
                    logger.warning("Discord webhook error %d: %s", resp.status, body[:200])
                    return False

        except Exception:
            logger.warning("Failed to send Discord embed", exc_info=True)
            return False

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # -- High-level notification methods ---------------------------

    async def send_trade_alert(self, result: dict[str, Any]) -> bool:
        """Send a trade execution notification with a rich embed."""
        action = result.get("action", "?")
        ticker = result.get("ticker", "?")
        status = result.get("order_status", "?")

        # Only notify on actual trades, not HOLDs
        if action == "HOLD":
            return False

        color = {"BUY": 0x00FF00, "SELL": 0xFF0000}.get(action, 0x808080)
        status_icon = {"SUBMITTED": "Approved", "COMPLIANCE_VETOED": "Vetoed"}.get(status, status)

        qty = result.get("quantity", 0)
        value = result.get("value", 0)
        reasoning = result.get("reasoning", "")
        if len(reasoning) > 300:
            reasoning = reasoning[:300] + "..."

        now = datetime.now(timezone.utc).strftime("%H:%M UTC")

        embed = {
            "title": f"{'BUY' if action == 'BUY' else 'SELL'} {ticker}",
            "color": color,
            "fields": [
                {"name": "Action", "value": action, "inline": True},
                {"name": "Status", "value": status_icon, "inline": True},
                {"name": "Quantity", "value": f"{qty:.4f}", "inline": True},
                {"name": "Value", "value": f"${value:,.2f}", "inline": True},
                {"name": "Reasoning", "value": reasoning or "N/A", "inline": False},
            ],
            "footer": {"text": f"AI Trading Bot | {now}"},
        }

        return await self._send_embed(embed)

    async def send_cycle_summary(
        self,
        results: list[dict[str, Any]],
        elapsed: float,
    ) -> bool:
        """Send a summary after each 15-min trading cycle."""
        submitted = [r for r in results if r.get("order_status") == "SUBMITTED"]
        vetoed = [r for r in results if r.get("order_status") == "COMPLIANCE_VETOED"]

        if not submitted and not vetoed:
            return False  # All HOLDs — don't spam

        now = datetime.now(timezone.utc).strftime("%H:%M UTC")

        fields = [
            {"name": "Scanned", "value": str(len(results)), "inline": True},
            {"name": "Duration", "value": f"{elapsed:.1f}s", "inline": True},
            {"name": "Submitted", "value": str(len(submitted)), "inline": True},
        ]

        if submitted:
            trade_lines = []
            for r in submitted:
                action = r.get("action", "?")
                ticker = r.get("ticker", "?")
                value = r.get("value", 0)
                trade_lines.append(f"{action} **{ticker}** ${value:,.2f}")
            fields.append({
                "name": "Trades",
                "value": "\n".join(trade_lines[:10]),
                "inline": False,
            })

        if vetoed:
            veto_lines = [f"**{r.get('ticker', '?')}**: {r.get('reasoning', '')[:60]}" for r in vetoed[:5]]
            fields.append({
                "name": "Vetoed",
                "value": "\n".join(veto_lines),
                "inline": False,
            })

        embed = {
            "title": f"Cycle Summary ({now})",
            "color": 0x3498DB,
            "fields": fields,
            "footer": {"text": "AI Trading Bot"},
        }

        return await self._send_embed(embed)

    async def send_daily_summary(self, summary: dict[str, Any]) -> bool:
        """Send the daily AI learning report as a rich embed."""
        audit = summary.get("audit_stats", {})
        total = audit.get("total_trades", 0)
        win_rate = audit.get("win_rate", 0)
        avg_roi = audit.get("avg_roi", 0)
        opro_gen = summary.get("opro_generation", 0)
        episodes = summary.get("episodes_stored", 0)
        regime = summary.get("regime_distribution", {})

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        regime_text = "\n".join(f"{k}: {v}" for k, v in regime.items()) if regime else "N/A"

        embed = {
            "title": f"Daily AI Learning Report ({now})",
            "color": 0x9B59B6,
            "fields": [
                {"name": "Trades Today", "value": str(total), "inline": True},
                {"name": "Win Rate", "value": f"{win_rate:.1%}", "inline": True},
                {"name": "Avg ROI", "value": f"{avg_roi:.4f}", "inline": True},
                {"name": "OPRO Generation", "value": str(opro_gen), "inline": True},
                {"name": "Episodes Stored", "value": str(episodes), "inline": True},
                {"name": "Market Regime", "value": regime_text, "inline": False},
            ],
            "footer": {"text": "AI Trading Bot | Learning in progress"},
        }

        return await self._send_embed(embed)

    async def send_system_event(self, event: str, details: str = "") -> bool:
        """Send system-level events (startup, shutdown, errors)."""
        color_map = {
            "START": 0x00FF00,
            "STOP": 0xFF6600,
            "ERROR": 0xFF0000,
            "WATCHDOG": 0xFF0000,
        }
        icon_map = {
            "START": "Startup",
            "STOP": "Shutdown",
            "ERROR": "Error",
            "WATCHDOG": "Watchdog Alert",
        }

        now = datetime.now(timezone.utc).strftime("%H:%M UTC")

        embed = {
            "title": icon_map.get(event.upper(), event.upper()),
            "description": details or "No details",
            "color": color_map.get(event.upper(), 0x808080),
            "footer": {"text": f"AI Trading Bot | {now}"},
        }

        return await self._send_embed(embed)

    async def send_watchdog_alert(self, alert: str) -> bool:
        """Send critical watchdog alerts."""
        return await self.send_system_event("WATCHDOG", alert)
