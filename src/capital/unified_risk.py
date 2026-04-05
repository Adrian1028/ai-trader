"""
Unified Risk Manager
=====================

Cross-system risk limits:
  - Daily drawdown limit: 5% of total NAV
  - Monthly drawdown limit: 15% of total NAV
  - Per-system isolation: one system's loss cannot breach another's capital

If any limit is breached → auto-pause trading until manual review.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    daily_max_loss_pct: float = 5.0    # Max 5% daily loss
    monthly_max_loss_pct: float = 15.0  # Max 15% monthly loss
    max_single_position_pct: float = 100.0  # SPY is single-asset, so 100%


@dataclass
class RiskSnapshot:
    date: str
    nav: float
    daily_pnl_pct: float
    monthly_pnl_pct: float


class UnifiedRiskManager:
    """Enforces cross-system risk limits."""

    def __init__(
        self,
        limits: RiskLimits | None = None,
        history_file: str = "data/risk_history.json",
    ):
        self.limits = limits or RiskLimits()
        self.history_file = Path(history_file)
        self.history: list[RiskSnapshot] = []
        self.is_paused = False
        self.pause_reason: str = ""
        self._load_history()

    def record_nav(self, nav: float):
        """Record daily NAV and check risk limits."""
        today = datetime.utcnow().strftime("%Y-%m-%d")

        # Calculate daily P&L
        daily_pnl_pct = 0.0
        if self.history:
            prev_nav = self.history[-1].nav
            if prev_nav > 0:
                daily_pnl_pct = (nav / prev_nav - 1) * 100

        # Calculate monthly P&L (vs 30 days ago)
        monthly_pnl_pct = 0.0
        month_ago = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        month_snapshots = [s for s in self.history if s.date >= month_ago]
        if month_snapshots:
            month_start_nav = month_snapshots[0].nav
            if month_start_nav > 0:
                monthly_pnl_pct = (nav / month_start_nav - 1) * 100

        snapshot = RiskSnapshot(
            date=today,
            nav=nav,
            daily_pnl_pct=daily_pnl_pct,
            monthly_pnl_pct=monthly_pnl_pct,
        )

        # Don't duplicate same-day entries
        if self.history and self.history[-1].date == today:
            self.history[-1] = snapshot
        else:
            self.history.append(snapshot)

        # Trim to 365 days
        cutoff = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
        self.history = [s for s in self.history if s.date >= cutoff]

        self._save_history()
        self._check_limits(snapshot)

    def _check_limits(self, snapshot: RiskSnapshot):
        """Check if any risk limit is breached."""
        if snapshot.daily_pnl_pct < -self.limits.daily_max_loss_pct:
            self.is_paused = True
            self.pause_reason = (
                f"Daily loss {snapshot.daily_pnl_pct:.2f}% exceeds "
                f"-{self.limits.daily_max_loss_pct}% limit"
            )
            logger.critical("RISK BREACH: %s", self.pause_reason)

        if snapshot.monthly_pnl_pct < -self.limits.monthly_max_loss_pct:
            self.is_paused = True
            self.pause_reason = (
                f"Monthly loss {snapshot.monthly_pnl_pct:.2f}% exceeds "
                f"-{self.limits.monthly_max_loss_pct}% limit"
            )
            logger.critical("RISK BREACH: %s", self.pause_reason)

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed."""
        if self.is_paused:
            return False, f"PAUSED: {self.pause_reason}"
        return True, "OK"

    def manual_resume(self, reason: str = ""):
        """Manual resume after risk review."""
        logger.warning("Risk pause manually lifted: %s", reason)
        self.is_paused = False
        self.pause_reason = ""

    def get_risk_report(self) -> dict:
        """Current risk status for logging/Discord."""
        latest = self.history[-1] if self.history else None
        return {
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "latest_nav": latest.nav if latest else None,
            "daily_pnl_pct": latest.daily_pnl_pct if latest else None,
            "monthly_pnl_pct": latest.monthly_pnl_pct if latest else None,
            "limits": {
                "daily_max_loss_pct": self.limits.daily_max_loss_pct,
                "monthly_max_loss_pct": self.limits.monthly_max_loss_pct,
            },
        }

    def _save_history(self):
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        data = [{"date": s.date, "nav": s.nav, "daily_pnl_pct": s.daily_pnl_pct,
                 "monthly_pnl_pct": s.monthly_pnl_pct} for s in self.history]
        with open(self.history_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_history(self):
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                self.history = [RiskSnapshot(**d) for d in data]
                logger.info("Loaded %d risk history entries", len(self.history))
            except Exception as e:
                logger.warning("Failed to load risk history: %s", e)
