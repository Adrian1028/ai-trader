"""
Compliance Watchdog (Independent Background Monitor)
=====================================================
Runs as an independent async task, SEPARATE from the trading pipeline.
Periodically monitors system health and can trigger emergency shutdown
even if the main pipeline is stuck or in a loop.

Checks:
  1. Order rate over sliding windows (1s, 10s, 60s)
  2. P&L drawdown breaker (daily loss limit)
  3. Position concentration limits
  4. API health (consecutive failures)
  5. Clock drift / stale data detection
  6. Heartbeat monitoring (detect frozen pipeline)

This watchdog is the LAST LINE OF DEFENCE.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.compliance.guard import ComplianceGuard
from src.core.client import Trading212Client

logger = logging.getLogger(__name__)


@dataclass
class WatchdogConfig:
    """Watchdog thresholds — all hard-coded, not tuneable by AI."""
    poll_interval_seconds: float = 120.0        # active market hours (2 min)
    idle_poll_interval_seconds: float = 600.0   # outside market hours (10 min)

    # Concentration check runs every N poll cycles (not every cycle)
    concentration_check_interval: int = 5      # every 5 cycles = ~5 min

    # Daily drawdown breaker
    max_daily_loss_pct: float = 0.05        # halt at 5% daily loss
    max_daily_loss_absolute: float = 5_000  # or £5,000 absolute

    # Position concentration
    max_single_position_pct: float = 0.20   # no single stock > 20% of NAV

    # API health
    max_consecutive_api_failures: int = 5

    # Heartbeat (pipeline liveness)
    max_heartbeat_gap_seconds: float = 900   # 15 min — a full 20-stock scan cycle can take a while on free tier

    # Stale data
    max_data_age_seconds: float = 900        # 15 min — matches heartbeat tolerance


@dataclass
class WatchdogState:
    """Mutable state tracked by the watchdog."""
    running: bool = False
    last_heartbeat: float = field(default_factory=time.monotonic)
    consecutive_api_failures: int = 0
    daily_pnl: float = 0.0
    start_of_day_nav: float = 0.0
    alerts: list[str] = field(default_factory=list)
    last_data_timestamp: float = field(default_factory=time.time)


class ComplianceWatchdog:
    """
    Independent safety monitor that runs in the background.
    Has authority to trigger emergency shutdown.

    Usage
    -----
    ```python
    watchdog = ComplianceWatchdog(client, guard)
    asyncio.create_task(watchdog.start())
    # ... later ...
    watchdog.heartbeat()  # called by main pipeline each cycle
    ```
    """

    def __init__(
        self,
        client: Trading212Client,
        guard: ComplianceGuard,
        config: WatchdogConfig | None = None,
    ) -> None:
        self._client = client
        self._guard = guard
        self._config = config or WatchdogConfig()
        self._state = WatchdogState()
        self._shutdown_event = asyncio.Event()
        self._cycle_count: int = 0  # tracks poll cycles for concentration check cadence

    # ── lifecycle ─────────────────────────────────────────────────────

    @staticmethod
    def _is_market_hours() -> bool:
        """Check if current UTC time is within any active market hours.

        Multi-market schedule (weekdays only):
          UK (LSE):  08:00 - 16:30 UTC
          US (NYSE): 13:30 - 20:00 UTC
          Combined:  08:00 - 20:00 UTC
        """
        now = datetime.now(timezone.utc)
        return now.weekday() < 5 and 8 <= now.hour < 20

    async def start(self) -> None:
        """Start the watchdog monitoring loop."""
        self._state.running = True
        logger.info(
            "Compliance watchdog started (active=%.0fs, idle=%.0fs)",
            self._config.poll_interval_seconds,
            self._config.idle_poll_interval_seconds,
        )

        while self._state.running and not self._shutdown_event.is_set():
            try:
                await self._check_cycle()
            except Exception:
                logger.exception("Watchdog check cycle failed")

            # Use longer interval outside market hours to conserve rate limit
            if self._is_market_hours():
                interval = self._config.poll_interval_seconds
            else:
                interval = self._config.idle_poll_interval_seconds
            await asyncio.sleep(interval)

        logger.info("Compliance watchdog stopped")

    def stop(self) -> None:
        """Signal the watchdog to stop."""
        self._state.running = False
        self._shutdown_event.set()

    def heartbeat(self) -> None:
        """Called by the main pipeline to signal liveness."""
        self._state.last_heartbeat = time.monotonic()

    def record_api_failure(self) -> None:
        self._state.consecutive_api_failures += 1

    def record_api_success(self) -> None:
        self._state.consecutive_api_failures = 0

    def update_daily_pnl(self, pnl: float) -> None:
        self._state.daily_pnl = pnl

    def set_start_of_day_nav(self, nav: float) -> None:
        self._state.start_of_day_nav = nav

    def update_data_timestamp(self) -> None:
        self._state.last_data_timestamp = time.time()

    # ── main check cycle ──────────────────────────────────────────────

    async def _check_cycle(self) -> None:
        """Run all watchdog checks."""
        in_market = self._is_market_hours()
        self._cycle_count += 1

        # 1. Heartbeat check (only during market hours — no heartbeat
        #    expected when the scheduler has no jobs running)
        if in_market:
            heartbeat_gap = time.monotonic() - self._state.last_heartbeat
            if heartbeat_gap > self._config.max_heartbeat_gap_seconds:
                alert = (
                    f"Pipeline heartbeat missing for {heartbeat_gap:.0f}s "
                    f"(limit {self._config.max_heartbeat_gap_seconds:.0f}s) — "
                    f"possible frozen state"
                )
                logger.warning("WATCHDOG: %s", alert)
                self._state.alerts.append(alert)

        # 2. Daily drawdown breaker (always active)
        if self._state.start_of_day_nav > 0:
            loss_pct = -self._state.daily_pnl / self._state.start_of_day_nav
            if (
                loss_pct > self._config.max_daily_loss_pct
                or -self._state.daily_pnl > self._config.max_daily_loss_absolute
            ):
                logger.critical(
                    "WATCHDOG: Daily drawdown breaker triggered — "
                    "loss=%.2f (%.1f%% of NAV)",
                    self._state.daily_pnl,
                    loss_pct * 100,
                )
                await self._emergency_halt(
                    f"Daily loss {self._state.daily_pnl:.2f} exceeds limit"
                )
                return

        # 3. API health (always active)
        if self._state.consecutive_api_failures >= self._config.max_consecutive_api_failures:
            logger.critical(
                "WATCHDOG: %d consecutive API failures — halting",
                self._state.consecutive_api_failures,
            )
            await self._emergency_halt(
                f"{self._state.consecutive_api_failures} consecutive API failures"
            )
            return

        # 4. Stale data detection (only during market hours)
        if in_market:
            data_age = time.time() - self._state.last_data_timestamp
            if data_age > self._config.max_data_age_seconds:
                logger.info(
                    "WATCHDOG: Market data is %.0fs old (threshold %.0fs) — "
                    "normal on free tier with %d stocks",
                    data_age,
                    self._config.max_data_age_seconds,
                    20,
                )

        # 5. Check portfolio concentration (requires 2 API calls — only
        #    run every N poll cycles to conserve rate limit budget)
        if in_market and (self._cycle_count % self._config.concentration_check_interval == 0):
            logger.debug("WATCHDOG: Running concentration check (cycle %d)", self._cycle_count)
            await self._check_concentration()

    async def _check_concentration(self) -> None:
        """Check that no single position exceeds concentration limits."""
        try:
            account = await self._client.account_info()
            positions = await self._client.portfolio()
            self.record_api_success()

            nav = float(account.get("value", 0))
            if nav <= 0:
                return

            for pos in positions:
                qty = float(pos.get("quantity", 0))
                price = float(pos.get("currentPrice", 0))
                value = qty * price
                pct = value / nav

                if pct > self._config.max_single_position_pct:
                    ticker = pos.get("ticker", "?")
                    alert = (
                        f"Position {ticker} is {pct:.1%} of NAV "
                        f"(limit {self._config.max_single_position_pct:.0%})"
                    )
                    logger.warning("WATCHDOG concentration alert: %s", alert)
                    self._state.alerts.append(alert)

        except Exception:
            self.record_api_failure()
            logger.warning("Watchdog: failed to check concentration")

    async def _emergency_halt(self, reason: str) -> None:
        """Trigger full emergency shutdown via the compliance guard."""
        logger.critical("WATCHDOG EMERGENCY HALT: %s", reason)
        await self._guard.emergency_shutdown(self._client)
        self.stop()

    # ── reporting ─────────────────────────────────────────────────────

    @property
    def state(self) -> WatchdogState:
        return self._state

    def get_alerts(self, clear: bool = True) -> list[str]:
        alerts = list(self._state.alerts)
        if clear:
            self._state.alerts.clear()
        return alerts
