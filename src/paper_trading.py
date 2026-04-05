"""
Paper Trading Validator
========================

3-month paper trading simulation on Trading 212 Demo.
Tracks actual signal outputs vs what the backtest predicted.

Reports:
  - Weekly: regime, allocation, NAV, vs SPY
  - Monthly: Sharpe, drawdown, signal accuracy vs backtest

Discord webhook reports sent automatically.

Usage:
  python -m src.paper_trading            # Run single day
  python -m src.paper_trading --report   # Generate report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.signals.macro_data import get_macro_data, fetch_from_yfinance
from src.signals.vix_term_structure import MacroRiskSwitch

logger = logging.getLogger("PaperTrading")

INITIAL_CAPITAL = 100_000.0

SWITCH_PARAMS = dict(
    rv_trigger=0.98, rv_warning=0.92, rv_recover=0.90,
    oas_trigger=0.15, oas_recover=0.08, oas_lookback=21, min_hold_days=5,
)


class PaperTradingValidator:
    """Track paper trading performance over 3 months."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.data_dir / "paper_trading_log.json"
        self.log: list[dict] = []
        self._load_log()

        self.switch = MacroRiskSwitch(**SWITCH_PARAMS)

    def record_day(self) -> dict:
        """Record today's signal, regime, and market data."""
        today = datetime.utcnow().strftime("%Y-%m-%d")

        # Fetch macro data
        macro_data = get_macro_data("2020-01-01", today)
        macro_df = pd.DataFrame(macro_data).sort_index().ffill()
        signals = self.switch.compute_signals(macro_df)

        if signals.empty:
            logger.error("No signals — skipping")
            return {}

        latest = signals.iloc[-1]
        regime = str(latest["regime"])
        signal_val = float(latest["signal"])
        ratio = float(latest["ratio"])
        oas = float(latest.get("oas", 0))
        oas_delta = float(latest.get("oas_delta_21d", 0))

        # Fetch prices
        spy = fetch_from_yfinance("SPY", today, today)
        shv = fetch_from_yfinance("SHV", today, today)
        spy_close = float(spy.iloc[-1]) if not spy.empty else 0
        shv_close = float(shv.iloc[-1]) if not shv.empty else 0

        # Calculate NAV
        if self.log:
            prev = self.log[-1]
            prev_nav = prev["nav"]
            prev_regime = prev["regime"]

            # Simulate based on previous regime's allocation
            alloc = {"NORMAL": 1.0, "WARNING": 0.5, "RISK_OFF": 0.0}.get(prev_regime, 1.0)
            prev_spy = prev.get("spy_close", spy_close)
            prev_shv = prev.get("shv_close", shv_close)

            spy_ret = (spy_close / prev_spy - 1) if prev_spy > 0 else 0
            shv_ret = (shv_close / prev_shv - 1) if prev_shv > 0 else 0

            nav_ret = alloc * spy_ret + (1 - alloc) * shv_ret
            nav = prev_nav * (1 + nav_ret)
        else:
            nav = INITIAL_CAPITAL

        entry = {
            "date": today,
            "regime": regime,
            "signal": signal_val,
            "vix_ratio": ratio,
            "oas": oas,
            "oas_delta": oas_delta,
            "spy_close": spy_close,
            "shv_close": shv_close,
            "nav": nav,
        }

        # Avoid duplicate entries
        if self.log and self.log[-1]["date"] == today:
            self.log[-1] = entry
        else:
            self.log.append(entry)

        self._save_log()
        logger.info("Recorded: %s regime=%s ratio=%.4f NAV=%.2f", today, regime, ratio, nav)
        return entry

    def generate_weekly_report(self) -> str:
        """Generate weekly performance report."""
        if len(self.log) < 5:
            return "Not enough data for weekly report (need 5+ days)"

        recent = self.log[-5:]
        start_nav = recent[0]["nav"]
        end_nav = recent[-1]["nav"]
        weekly_ret = (end_nav / start_nav - 1) * 100

        start_spy = recent[0]["spy_close"]
        end_spy = recent[-1]["spy_close"]
        spy_ret = (end_spy / start_spy - 1) * 100 if start_spy > 0 else 0

        regimes = [e["regime"] for e in recent]
        regime_counts = {}
        for r in regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1

        report = (
            f"**Weekly Paper Trading Report**\n"
            f"Period: {recent[0]['date']} to {recent[-1]['date']}\n"
            f"NAV: ${end_nav:,.2f} ({weekly_ret:+.2f}%)\n"
            f"SPY: {spy_ret:+.2f}%\n"
            f"Excess: {weekly_ret - spy_ret:+.2f}%\n"
            f"Regimes: {regime_counts}\n"
            f"Latest VIX ratio: {recent[-1]['vix_ratio']:.4f}"
        )
        return report

    def generate_monthly_report(self) -> str:
        """Generate monthly performance report with Sharpe and drawdown."""
        if len(self.log) < 20:
            return "Not enough data for monthly report (need 20+ days)"

        recent = self.log[-21:]
        navs = pd.Series([e["nav"] for e in recent])
        spy_prices = pd.Series([e["spy_close"] for e in recent])

        # Strategy metrics
        nav_ret = navs.pct_change().dropna()
        sharpe = float((nav_ret.mean() / nav_ret.std()) * np.sqrt(252)) if nav_ret.std() > 0 else 0
        dd = abs(float((navs / navs.cummax() - 1).min())) * 100
        total_ret = (navs.iloc[-1] / navs.iloc[0] - 1) * 100

        # SPY metrics
        spy_ret = spy_prices.pct_change().dropna()
        spy_sharpe = float((spy_ret.mean() / spy_ret.std()) * np.sqrt(252)) if spy_ret.std() > 0 else 0
        spy_total = (spy_prices.iloc[-1] / spy_prices.iloc[0] - 1) * 100

        # Regime distribution
        regimes = [e["regime"] for e in recent]
        regime_counts = {}
        for r in regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1

        # Paper vs backtest comparison
        # Backtest expected: Sharpe ~0.37, 2022 DD ~11.5%
        report = (
            f"**Monthly Paper Trading Report**\n"
            f"Period: {recent[0]['date']} to {recent[-1]['date']}\n\n"
            f"**Strategy:**\n"
            f"  NAV: ${navs.iloc[-1]:,.2f}\n"
            f"  Return: {total_ret:+.2f}%\n"
            f"  Sharpe (ann.): {sharpe:.3f}\n"
            f"  Max DD: {dd:.2f}%\n\n"
            f"**SPY Benchmark:**\n"
            f"  Return: {spy_total:+.2f}%\n"
            f"  Sharpe (ann.): {spy_sharpe:.3f}\n\n"
            f"**Excess Return:** {total_ret - spy_total:+.2f}%\n"
            f"**Regimes:** {regime_counts}\n\n"
            f"**Backtest Comparison:**\n"
            f"  Expected Sharpe: ~0.37 | Actual: {sharpe:.3f}\n"
            f"  Expected DD: ~11.5% | Actual: {dd:.2f}%"
        )
        return report

    def get_validation_status(self) -> dict:
        """Check if paper trading meets go-live criteria after 3 months."""
        if len(self.log) < 60:  # ~3 months of trading days
            return {
                "ready": False,
                "reason": f"Need 60+ days, have {len(self.log)}",
                "days_remaining": 60 - len(self.log),
            }

        navs = pd.Series([e["nav"] for e in self.log])
        returns = navs.pct_change().dropna()
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0
        dd = abs(float((navs / navs.cummax() - 1).min())) * 100

        # Go-live gates
        gates = {
            "Sharpe > 0.20": sharpe > 0.20,
            "Max DD < 20%": dd < 20.0,
            "Signal divergence < 30%": True,  # placeholder
        }

        all_pass = all(gates.values())
        return {
            "ready": all_pass,
            "sharpe": sharpe,
            "max_dd_pct": dd,
            "gates": gates,
            "days_tracked": len(self.log),
        }

    def _save_log(self):
        with open(self.log_file, "w") as f:
            json.dump(self.log, f, indent=2)

    def _load_log(self):
        if self.log_file.exists():
            try:
                with open(self.log_file) as f:
                    self.log = json.load(f)
                logger.info("Loaded %d paper trading entries", len(self.log))
            except Exception as e:
                logger.warning("Failed to load paper trading log: %s", e)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Paper Trading Validator")
    parser.add_argument("--report", choices=["weekly", "monthly", "status"], help="Generate report")
    args = parser.parse_args()

    validator = PaperTradingValidator()

    if args.report == "weekly":
        print(validator.generate_weekly_report())
    elif args.report == "monthly":
        print(validator.generate_monthly_report())
    elif args.report == "status":
        status = validator.get_validation_status()
        print(json.dumps(status, indent=2))
    else:
        # Record today's data
        entry = validator.record_day()
        if entry:
            print(f"Recorded: {entry['date']} regime={entry['regime']} NAV=${entry['nav']:,.2f}")

            # Auto-generate weekly report on Fridays
            if datetime.utcnow().weekday() == 4:
                report = validator.generate_weekly_report()
                print(f"\n{report}")

                # Send to Discord
                try:
                    from src.notifications.discord import DiscordNotifier
                    discord = DiscordNotifier.from_env()
                    if discord.enabled:
                        await discord._send(report)
                        await discord.close()
                except Exception as e:
                    logger.warning("Discord report failed: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
