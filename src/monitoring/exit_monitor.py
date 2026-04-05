"""
Exit Monitor — Automated Strategy Health Checks
=================================================

Monitors:
  1. Rolling 6-month Sharpe ratio (auto-pause if < 0.30)
  2. Max drawdown breach (auto-pause if > 25%)
  3. Signal stagnation (same regime for > 90 trading days without review)
  4. Underperformance vs SPY buy-and-hold (6-month rolling)

All checks run daily after market close.
If any check triggers, trading is paused and Discord alert sent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExitCheck:
    name: str
    triggered: bool
    value: float
    threshold: float
    message: str


class ExitMonitor:
    """Monitor strategy health and trigger auto-pause."""

    # Thresholds
    MIN_SHARPE_6M = 0.30
    MAX_DRAWDOWN_PCT = 25.0
    MAX_SAME_REGIME_DAYS = 90
    MIN_EXCESS_RETURN_6M = -10.0  # Must not underperform SPY by >10% over 6 months

    def __init__(self, nav_history_file: str = "data/nav_history.json"):
        self.nav_file = Path(nav_history_file)
        self.nav_history: list[dict] = []  # [{date, nav, spy_close, regime}]
        self._load_history()

    def record_daily(self, date: str, nav: float, spy_close: float, regime: str):
        """Record daily NAV, SPY close, and current regime."""
        entry = {
            "date": date,
            "nav": nav,
            "spy_close": spy_close,
            "regime": regime,
        }
        # Avoid duplicates
        if self.nav_history and self.nav_history[-1]["date"] == date:
            self.nav_history[-1] = entry
        else:
            self.nav_history.append(entry)

        # Keep 2 years max
        cutoff = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-%m-%d")
        self.nav_history = [e for e in self.nav_history if e["date"] >= cutoff]
        self._save_history()

    def run_checks(self) -> list[ExitCheck]:
        """Run all exit checks. Returns list of check results."""
        checks = []

        if len(self.nav_history) < 30:
            logger.info("Exit monitor: not enough history (%d days), skipping", len(self.nav_history))
            return checks

        df = pd.DataFrame(self.nav_history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        checks.append(self._check_rolling_sharpe(df))
        checks.append(self._check_max_drawdown(df))
        checks.append(self._check_signal_stagnation(df))
        checks.append(self._check_excess_return(df))

        triggered = [c for c in checks if c.triggered]
        if triggered:
            for c in triggered:
                logger.warning("EXIT CHECK TRIGGERED: %s", c.message)

        return checks

    def should_pause(self) -> tuple[bool, list[str]]:
        """Check if trading should be paused."""
        checks = self.run_checks()
        triggered = [c for c in checks if c.triggered]
        if triggered:
            reasons = [c.message for c in triggered]
            return True, reasons
        return False, []

    def _check_rolling_sharpe(self, df: pd.DataFrame) -> ExitCheck:
        """6-month rolling Sharpe ratio."""
        nav = df["nav"]
        # Use last ~126 trading days (6 months)
        recent = nav.tail(126)
        if len(recent) < 60:
            return ExitCheck("rolling_sharpe_6m", False, 0, self.MIN_SHARPE_6M,
                             "Not enough data for 6-month Sharpe")

        returns = recent.pct_change().dropna()
        if returns.std() == 0:
            sharpe = 0.0
        else:
            sharpe = float((returns.mean() / returns.std()) * np.sqrt(252))

        triggered = sharpe < self.MIN_SHARPE_6M
        return ExitCheck(
            "rolling_sharpe_6m", triggered, sharpe, self.MIN_SHARPE_6M,
            f"6-month Sharpe {sharpe:.3f} {'< ' + str(self.MIN_SHARPE_6M) + ' BREACH' if triggered else 'OK'}"
        )

    def _check_max_drawdown(self, df: pd.DataFrame) -> ExitCheck:
        """Current drawdown from peak."""
        nav = df["nav"]
        peak = nav.cummax()
        dd = float((nav.iloc[-1] / peak.iloc[-1] - 1) * 100)
        dd_abs = abs(dd)

        triggered = dd_abs > self.MAX_DRAWDOWN_PCT
        return ExitCheck(
            "max_drawdown", triggered, dd_abs, self.MAX_DRAWDOWN_PCT,
            f"Current DD {dd_abs:.2f}% {'> ' + str(self.MAX_DRAWDOWN_PCT) + '% BREACH' if triggered else 'OK'}"
        )

    def _check_signal_stagnation(self, df: pd.DataFrame) -> ExitCheck:
        """Check if regime hasn't changed in too long."""
        regimes = df["regime"]
        if regimes.empty:
            return ExitCheck("signal_stagnation", False, 0, self.MAX_SAME_REGIME_DAYS, "No regime data")

        # Count consecutive same-regime days from the end
        current = regimes.iloc[-1]
        count = 0
        for r in reversed(regimes.values):
            if r == current:
                count += 1
            else:
                break

        triggered = count > self.MAX_SAME_REGIME_DAYS
        return ExitCheck(
            "signal_stagnation", triggered, count, self.MAX_SAME_REGIME_DAYS,
            f"Regime '{current}' for {count} days {'> ' + str(self.MAX_SAME_REGIME_DAYS) + ' REVIEW NEEDED' if triggered else 'OK'}"
        )

    def _check_excess_return(self, df: pd.DataFrame) -> ExitCheck:
        """6-month excess return vs SPY buy-and-hold."""
        recent = df.tail(126)
        if len(recent) < 60:
            return ExitCheck("excess_return_6m", False, 0, self.MIN_EXCESS_RETURN_6M,
                             "Not enough data for 6-month comparison")

        strat_ret = (recent["nav"].iloc[-1] / recent["nav"].iloc[0] - 1) * 100
        spy_ret = (recent["spy_close"].iloc[-1] / recent["spy_close"].iloc[0] - 1) * 100
        excess = strat_ret - spy_ret

        triggered = excess < self.MIN_EXCESS_RETURN_6M
        return ExitCheck(
            "excess_return_6m", triggered, excess, self.MIN_EXCESS_RETURN_6M,
            f"6-month excess {excess:.2f}% {'< ' + str(self.MIN_EXCESS_RETURN_6M) + '% BREACH' if triggered else 'OK'}"
        )

    def get_status_report(self) -> dict:
        """Full status report for logging/Discord."""
        checks = self.run_checks()
        return {
            "checks": [
                {"name": c.name, "triggered": c.triggered, "value": c.value,
                 "threshold": c.threshold, "message": c.message}
                for c in checks
            ],
            "any_triggered": any(c.triggered for c in checks),
            "history_days": len(self.nav_history),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _save_history(self):
        self.nav_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.nav_file, "w") as f:
            json.dump(self.nav_history, f, indent=2)

    def _load_history(self):
        if self.nav_file.exists():
            try:
                with open(self.nav_file) as f:
                    self.nav_history = json.load(f)
                logger.info("Loaded %d NAV history entries", len(self.nav_history))
            except Exception as e:
                logger.warning("Failed to load NAV history: %s", e)
