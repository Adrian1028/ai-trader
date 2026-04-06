"""
Production Macro Signal Trading System
========================================

Simple SPY/SHV toggle based on VIX term structure + OAS macro risk switch.

Architecture:
  - Daily execution at UTC 22:00 (after US market close)
  - Fetch macro data (VIX, VIX3M, HYG/TLT for synthetic OAS)
  - Compute regime: NORMAL → 100% SPY, WARNING → 50% SPY + 50% SHV, RISK_OFF → 100% SHV
  - Execute via Trading 212 Demo API (paper trading phase)
  - Log everything, send Discord notifications
  - Exit monitor checks run after each cycle

Schedule:
  UTC 22:00 Mon-Fri: Main trading cycle
  UTC 22:30 Mon-Fri: Exit monitor + risk check

Usage:
  python -m src.main_macro
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.signals.macro_data import get_macro_data, get_macro_dataframe, fetch_from_yfinance
from src.signals.vix_term_structure import MacroRiskSwitch, Regime
from src.resilience.failsafe import FailsafeManager, SystemMode
from src.capital.allocation_tracker import AllocationTracker
from src.capital.unified_risk import UnifiedRiskManager, RiskLimits
from src.monitoring.exit_monitor import ExitMonitor
from src.execution.trading212 import Trading212Executor, RebalanceResult

logger = logging.getLogger("MacroTrader")

# ── Configuration ──────────────────────────────────────────────────

INITIAL_CAPITAL = float(os.getenv("MACRO_INITIAL_CAPITAL", "100000.0"))

# Validated macro switch parameters (from D006)
SWITCH_PARAMS = dict(
    rv_trigger=0.98, rv_warning=0.92, rv_recover=0.90,
    oas_trigger=0.15, oas_recover=0.08, oas_lookback=21, min_hold_days=5,
)

# Allocation by regime
REGIME_ALLOCATION = {
    "NORMAL": {"SPY": 1.0, "SHV": 0.0},
    "WARNING": {"SPY": 0.5, "SHV": 0.5},
    "RISK_OFF": {"SPY": 0.0, "SHV": 1.0},  # SHV beats cash (Step 3 result)
}


def _configure_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    data_dir = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))
    os.makedirs(data_dir, exist_ok=True)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(data_dir, "macro_trader.log"),
            encoding="utf-8",
        ),
    ]
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


class MacroTrader:
    """
    Production macro signal trader.

    Daily workflow:
      1. Health check (failsafe manager)
      2. Fetch macro data
      3. Compute signal
      4. Check exit monitor
      5. Check risk limits
      6. Determine target allocation
      7. Execute trades (or log paper trades)
      8. Record NAV
      9. Send Discord notification
    """

    def __init__(self):
        data_dir = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))

        self.switch = MacroRiskSwitch(**SWITCH_PARAMS)
        self.failsafe = FailsafeManager(cache_dir=Path(data_dir) / "failsafe_cache")
        self.capital = AllocationTracker(state_file=os.path.join(data_dir, "capital_state.json"))
        self.risk = UnifiedRiskManager(
            limits=RiskLimits(daily_max_loss_pct=5.0, monthly_max_loss_pct=15.0),
            history_file=os.path.join(data_dir, "risk_history.json"),
        )
        self.exit_monitor = ExitMonitor(nav_history_file=os.path.join(data_dir, "nav_history.json"))

        # Register capital
        self.capital.register_system("macro_spy", INITIAL_CAPITAL)

        # Executor
        self.executor = Trading212Executor()

        # State
        self.current_regime = "NORMAL"
        self.current_positions: dict[str, float] = {}  # ticker -> value
        self.state_file = Path(data_dir) / "macro_trader_state.json"
        self._load_state()

        # Discord (lazy import)
        self._discord = None

    async def _get_discord(self):
        if self._discord is None:
            try:
                from src.notifications.discord import DiscordNotifier
                self._discord = DiscordNotifier.from_env()
            except Exception:
                self._discord = None
        return self._discord

    def _register_data_sources(self):
        """Register macro data sources with fallback chains."""
        import yfinance as yf

        def _fetch_vix():
            return fetch_from_yfinance("^VIX", "2020-01-01", datetime.utcnow().strftime("%Y-%m-%d"))

        def _fetch_vix3m():
            return fetch_from_yfinance("^VIX3M", "2020-01-01", datetime.utcnow().strftime("%Y-%m-%d"))

        def _fetch_oas_proxy():
            end = datetime.utcnow().strftime("%Y-%m-%d")
            hyg = fetch_from_yfinance("HYG", "2020-01-01", end)
            tlt = fetch_from_yfinance("TLT", "2020-01-01", end)
            if hyg.empty or tlt.empty:
                return None
            from src.signals.macro_data import _compute_synthetic_oas
            return _compute_synthetic_oas(hyg, tlt)

        self.failsafe.register_source("vix", [_fetch_vix])
        self.failsafe.register_source("vix3m", [_fetch_vix3m])
        self.failsafe.register_source("oas", [_fetch_oas_proxy])

    async def run_daily_cycle(self) -> dict:
        """Run the full daily trading cycle."""
        cycle_start = time.monotonic()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        result = {"date": today, "status": "OK", "actions": []}

        logger.info("=" * 60)
        logger.info("  DAILY MACRO TRADING CYCLE — %s", today)
        logger.info("=" * 60)

        # Step 1: Health check
        logger.info("Step 1: Data health check...")
        self._register_data_sources()
        vix_data = self.failsafe.fetch_with_fallback("vix")
        vix3m_data = self.failsafe.fetch_with_fallback("vix3m")
        oas_data = self.failsafe.fetch_with_fallback("oas")

        health = self.failsafe.evaluate_system_health()
        result["health"] = self.failsafe.get_status_report()

        if self.failsafe.should_force_exit():
            logger.critical("EMERGENCY: Forcing all-cash mode — %s", health.reason)
            result["status"] = "EMERGENCY_CASH"
            result["actions"].append({"action": "FORCE_EXIT", "reason": health.reason})
            await self._notify(f"EMERGENCY: {health.reason}\nForcing all-cash mode.")
            self._save_state()
            return result

        # Step 2: Compute signal
        logger.info("Step 2: Computing macro signal...")
        try:
            macro_data = get_macro_data("2020-01-01", today)
            import pandas as pd
            macro_df = pd.DataFrame(macro_data).sort_index().ffill()
            signals = self.switch.compute_signals(macro_df)

            if signals.empty:
                logger.error("No signals computed!")
                result["status"] = "ERROR"
                return result

            latest = signals.iloc[-1]
            regime = str(latest["regime"])
            signal_value = float(latest["signal"])
            ratio = float(latest["ratio"])

            logger.info("  VIX/VIX3M ratio: %.4f", ratio)
            logger.info("  Regime: %s (signal: %.1f)", regime, signal_value)

            result["regime"] = regime
            result["signal"] = signal_value
            result["vix_ratio"] = ratio

        except Exception as e:
            logger.error("Signal computation failed: %s", e)
            result["status"] = "ERROR"
            result["error"] = str(e)
            return result

        # Step 3: Apply failsafe clamp
        clamped_signal = self.failsafe.clamp_signal(signal_value)
        if clamped_signal != signal_value:
            logger.warning("Signal clamped by failsafe: %.2f -> %.2f", signal_value, clamped_signal)

        # Step 4: Check risk limits
        can_trade, risk_msg = self.risk.can_trade()
        if not can_trade:
            logger.warning("Risk manager blocked trading: %s", risk_msg)
            result["status"] = "RISK_BLOCKED"
            result["risk_message"] = risk_msg
            await self._notify(f"Trading blocked by risk manager: {risk_msg}")
            return result

        # Step 5: Check exit monitor
        should_pause, pause_reasons = self.exit_monitor.should_pause()
        if should_pause:
            logger.warning("Exit monitor triggered pause: %s", "; ".join(pause_reasons))
            result["status"] = "EXIT_MONITOR_PAUSE"
            result["exit_reasons"] = pause_reasons
            await self._notify(f"Exit monitor pause:\n" + "\n".join(pause_reasons))
            return result

        # Step 6: Determine target allocation
        alloc = REGIME_ALLOCATION.get(regime, REGIME_ALLOCATION["NORMAL"])
        # Apply failsafe clamp to SPY portion
        if health.max_exposure_pct < 1.0:
            spy_pct = alloc["SPY"] * health.max_exposure_pct
            shv_pct = 1.0 - spy_pct
            alloc = {"SPY": spy_pct, "SHV": shv_pct}

        logger.info("Step 6: Target allocation — SPY: %.0f%%, SHV: %.0f%%",
                     alloc["SPY"] * 100, alloc["SHV"] * 100)

        # Step 7: Track regime change
        prev_regime = self.current_regime
        if regime != prev_regime:
            action = {
                "type": "REGIME_CHANGE",
                "from": prev_regime,
                "to": regime,
                "spy_pct": alloc["SPY"] * 100,
                "shv_pct": alloc["SHV"] * 100,
            }
            result["actions"].append(action)
            logger.info("  REGIME CHANGE: %s -> %s", prev_regime, regime)
            self.current_regime = regime
        else:
            logger.info("  No regime change (still %s)", regime)

        # Step 8: Execute trades via Trading 212 API
        rebalance_result = None
        if self.executor.enabled:
            logger.info("Step 8: Executing rebalance via Trading 212...")
            try:
                rebalance_result = await self.executor.rebalance(alloc, regime)
                result["execution"] = rebalance_result.to_dict()
                logger.info("  Execution status: %s, trades: %d",
                            rebalance_result.status, len(rebalance_result.trades))

                # Use real NAV from T212
                nav = rebalance_result.nav_after if rebalance_result.nav_after > 0 else rebalance_result.nav_before
            except Exception as e:
                logger.error("Execution failed: %s", e)
                result["execution"] = {"status": "ERROR", "error": str(e)}
                nav = self.capital.systems["macro_spy"].current_nav
        else:
            logger.info("Step 8: Executor disabled — paper trading mode")
            nav = self.capital.systems["macro_spy"].current_nav

        # Step 8b: Update NAV tracking
        try:
            spy_price = fetch_from_yfinance("SPY", today, today)
            spy_close = float(spy_price.iloc[-1]) if not spy_price.empty else 0

            # If executor gave us real NAV, use it; otherwise estimate
            if self.executor.enabled and rebalance_result and rebalance_result.nav_after > 0:
                nav = rebalance_result.nav_after

            spy_value = nav * alloc["SPY"]
            shv_value = nav * alloc["SHV"]

            result["nav"] = nav
            result["spy_close"] = spy_close

            self.exit_monitor.record_daily(today, nav, spy_close, regime)
            self.risk.record_nav(nav)
            self.capital.update_nav("macro_spy", nav, shv_value, spy_value)

        except Exception as e:
            logger.warning("NAV tracking failed: %s", e)

        # Step 9: Send Discord notification (signal + execution)
        elapsed = time.monotonic() - cycle_start
        result["elapsed_seconds"] = elapsed

        msg = (
            f"**Daily Macro Signal** ({today})\n"
            f"Regime: **{regime}** (VIX ratio: {ratio:.4f})\n"
            f"Allocation: SPY {alloc['SPY']*100:.0f}% / SHV {alloc['SHV']*100:.0f}%\n"
            f"System: {health.mode.value} | Elapsed: {elapsed:.1f}s"
        )
        if result["actions"]:
            msg += f"\nActions: {len(result['actions'])} regime change(s)"

        # Append execution details
        if rebalance_result:
            msg += f"\n\n{rebalance_result.discord_summary()}"
        elif not self.executor.enabled:
            msg += "\n\n*Paper trading mode — no orders executed*"

        await self._notify(msg)

        self._save_state()
        logger.info("Cycle complete in %.1fs", elapsed)
        return result

    async def run_exit_check(self) -> dict:
        """Run the exit monitor check (separate from trading cycle)."""
        logger.info("Running exit monitor check...")
        report = self.exit_monitor.get_status_report()
        risk_report = self.risk.get_risk_report()

        if report["any_triggered"]:
            triggered = [c for c in report["checks"] if c["triggered"]]
            msg = "**Exit Monitor Alert**\n"
            for c in triggered:
                msg += f"- {c['message']}\n"
            await self._notify(msg)

        return {"exit_monitor": report, "risk": risk_report}

    async def _notify(self, message: str):
        discord = await self._get_discord()
        if discord and discord.enabled:
            try:
                await discord._send(message)
            except Exception as e:
                logger.warning("Discord notification failed: %s", e)
        logger.info("Notification: %s", message[:200])

    def _save_state(self):
        state = {
            "current_regime": self.current_regime,
            "current_positions": self.current_positions,
            "last_updated": datetime.utcnow().isoformat(),
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                self.current_regime = state.get("current_regime", "NORMAL")
                self.current_positions = state.get("current_positions", {})
                logger.info("Loaded state: regime=%s", self.current_regime)
            except Exception as e:
                logger.warning("Failed to load state: %s", e)


async def main():
    _configure_logging()

    logger.info("=" * 60)
    logger.info("  MACRO SIGNAL TRADING SYSTEM")
    logger.info("  SPY/SHV toggle based on VIX term structure + OAS")
    logger.info("=" * 60)

    trader = MacroTrader()

    mode = os.getenv("MACRO_MODE", "once")  # "once" or "scheduled"

    if mode == "once":
        # Run single cycle (useful for testing / cron)
        result = await trader.run_daily_cycle()
        logger.info("Result: %s", json.dumps(result, indent=2, default=str))
        exit_result = await trader.run_exit_check()
        logger.info("Exit check: %s", json.dumps(exit_result, indent=2, default=str))

    elif mode == "scheduled":
        # APScheduler mode for continuous running
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.error("APScheduler not installed. Run: pip install apscheduler")
            return

        scheduler = AsyncIOScheduler()

        # Daily trading cycle at UTC 22:00 Mon-Fri
        scheduler.add_job(
            trader.run_daily_cycle,
            CronTrigger(day_of_week="mon-fri", hour=22, minute=0, timezone="UTC"),
            id="macro_daily",
            name="Daily macro signal cycle",
            misfire_grace_time=600,
        )

        # Exit check at UTC 22:30 Mon-Fri
        scheduler.add_job(
            trader.run_exit_check,
            CronTrigger(day_of_week="mon-fri", hour=22, minute=30, timezone="UTC"),
            id="exit_check",
            name="Exit monitor check",
            misfire_grace_time=600,
        )

        scheduler.start()
        logger.info("Scheduler started — 2 jobs registered")

        stop = asyncio.Event()

        def _shutdown(sig, _):
            logger.warning("Received %s — shutting down...", signal.Signals(sig).name)
            stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _shutdown)
            except (OSError, ValueError):
                pass

        logger.info("System running. Ctrl+C to stop.")
        await stop.wait()
        scheduler.shutdown(wait=True)
        logger.info("Shutdown complete.")

    # Cleanup sessions
    await trader.executor.close()
    discord = await trader._get_discord()
    if discord:
        await discord.close()


if __name__ == "__main__":
    asyncio.run(main())
