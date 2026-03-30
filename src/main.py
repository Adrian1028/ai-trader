"""
AI Trading System — Main Entry Point
=====================================
Production-ready scheduler that runs the full MAS trading pipeline
on a cron schedule using APScheduler.

Schedule:
  - Every 15 min during US market hours (Mon-Fri UTC 13:30-20:00):
    Intelligence -> Decision -> Execution -> Audit
  - Daily at UTC 20:30: Cognitive reflection, OPRO evolution, learning report
  - Daily at UTC 21:00: Fill reconciliation and virtual-account sync

Graceful shutdown:
  Captures SIGINT/SIGTERM, waits for the current job to finish,
  then persists all state (virtual accounts, OPRO, episodic memory)
  before exiting cleanly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

# Load .env file from project root (must happen before any config import)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.is_file():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed; rely on real env vars

logger = logging.getLogger("Main")

# ── Target symbols (ISIN-based for T212 compatibility) ────────────
# Override via environment variable TARGET_ISINS (comma-separated)
_DEFAULT_TARGETS = [
    # ── 科技 (12) ──
    "US0378331005",   # AAPL  — Apple
    "US5949181045",   # MSFT  — Microsoft
    "US67066G1040",   # NVDA  — NVIDIA
    "US88160R1014",   # TSLA  — Tesla
    "US0231351067",   # AMZN  — Amazon
    "US02079K3059",   # GOOG  — Alphabet
    "US30303M1027",   # META  — Meta
    "US11135F1012",   # AVGO  — Broadcom
    "US12652L1008",   # CRM   — Salesforce
    "US0079031078",   # AMD   — AMD
    "US68389X1054",   # ORCL  — Oracle
    "US4581401001",   # INTC  — Intel
    # ── 能源 (8) ──
    "US30231G1022",   # XOM   — Exxon Mobil
    "US1667641005",   # CVX   — Chevron
    "US20825C1045",   # COP   — ConocoPhillips
    "US8064071025",   # SLB   — Schlumberger
    "US26875P1012",   # EOG   — EOG Resources
    "US65339F1012",   # NEE   — NextEra Energy
    "US29355A1079",   # ENPH  — Enphase Energy
    "US6745991058",   # OXY   — Occidental
]


def _configure_logging() -> None:
    """Set up structured logging to console + rotating file."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    data_dir = os.getenv("DATA_DIR", "data")
    os.makedirs(data_dir, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(data_dir, "trading_bot.log"),
            encoding="utf-8",
        ),
    ]
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


async def main() -> None:
    # Deferred imports so logging is configured first
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    from src.core.orchestrator import TradingSystem
    from src.core.virtual_account import VirtualAccountManager
    from src.notifications.telegram import TelegramNotifier

    logger.info("=" * 60)
    logger.info("  AI Quantitative Trading System (Trading 212 Agentic MAS)")
    logger.info("=" * 60)

    # ── 1. Shared infrastructure ───────────────────────────────────
    data_dir = os.getenv("DATA_DIR", "data")
    os.makedirs(data_dir, exist_ok=True)

    account_manager = VirtualAccountManager(
        storage_file=os.path.join(data_dir, "virtual_accounts.json"),
    )

    # ── 2. Build TradingSystem from environment ────────────────────
    bot_id = os.getenv("BOT_ID", "AlphaBot_01")
    initial_capital = float(os.getenv("INITIAL_CAPITAL", "10000.0"))
    target_isins = os.getenv("TARGET_ISINS", "").split(",")
    target_isins = [t.strip() for t in target_isins if t.strip()] or _DEFAULT_TARGETS

    system = TradingSystem.from_env(
        bot_id=bot_id,
        initial_capital=initial_capital,
        account_manager=account_manager,
    )

    # ── 3. Telegram notifier ─────────────────────────────────────
    notifier = TelegramNotifier.from_env()

    # ── 4. Initialise with async context manager ──────────────────
    async with system:
        await system.initialise()
        env_name = os.getenv("T212_ENV", "demo").upper()
        logger.info(
            "System initialised — env=%s, bot=%s, capital=%.2f, targets=%d ISINs",
            env_name, bot_id, initial_capital, len(target_isins),
        )

        # Notify system startup
        await notifier.send_system_event(
            "START",
            f"Env: {env_name} | Bot: {bot_id}\n"
            f"Capital: ${initial_capital:,.2f}\n"
            f"Targets: {len(target_isins)} stocks",
        )

        # ── Cycle-level state ──────────────────────────────────────
        # Track audit record IDs from the last trading cycle
        # so we can reconcile fills and run reflection later.
        last_cycle_record_ids: list[str] = []

        # ── Job definitions ────────────────────────────────────────

        async def trading_job() -> None:
            """15-minute market scan → intelligence → decision → execution."""
            nonlocal last_cycle_record_ids
            logger.info("=== Trading cycle started ===")
            t0 = time.monotonic()

            try:
                results = await system.run_cycle(target_isins)
                elapsed = time.monotonic() - t0

                # Collect audit IDs for later reflection
                record_ids = [r["audit_id"] for r in results if "audit_id" in r]
                last_cycle_record_ids.extend(record_ids)

                submitted = sum(1 for r in results if r.get("order_status") == "SUBMITTED")
                vetoed = sum(1 for r in results if r.get("order_status") == "COMPLIANCE_VETOED")
                held = sum(1 for r in results if r.get("action") == "HOLD")

                logger.info(
                    "Trading cycle done in %.1fs — "
                    "%d submitted, %d vetoed, %d held, %d total",
                    elapsed, submitted, vetoed, held, len(results),
                )

                # ── Telegram notifications ─────────────────────
                # Send individual trade alerts
                for r in results:
                    await notifier.send_trade_alert(r)
                # Send cycle summary (only if trades were made)
                await notifier.send_cycle_summary(results, elapsed)

            except Exception:
                logger.exception("Trading cycle failed unexpectedly")

        async def sync_and_reflect_job() -> None:
            """
            Post-market fill reconciliation + cognitive reflection.
            Checks order fills, updates virtual accounts, runs failure
            attribution, feeds OPRO, and triggers evolution.
            """
            nonlocal last_cycle_record_ids
            logger.info("=== Fill sync & cognitive reflection ===")

            try:
                # Check fills for all submitted orders
                for ticket_id, ticket in list(system._executor.active_tickets.items()):
                    if ticket.api_order_id is not None:
                        await system._executor.check_fill(ticket)

                # Run cognitive reflection on today's trades
                if last_cycle_record_ids:
                    reflection_results = await system.reflect(last_cycle_record_ids)
                    logger.info(
                        "Reflected on %d trades — %d failure reports generated",
                        len(reflection_results),
                        sum(1 for r in reflection_results if "failure_report" in r),
                    )
                    last_cycle_record_ids.clear()

                # Persist virtual account state
                account_manager.save_state()
                logger.info("Virtual account state saved")

            except Exception:
                logger.exception("Sync & reflect job failed")

        async def learning_report_job() -> None:
            """Daily AI learning summary + OPRO state report."""
            logger.info("=== Daily AI Learning Report ===")
            try:
                summary = system.learning_summary()
                report_lines = [
                    f"  Episodes stored:     {summary.get('episodes_stored', 0)}",
                    f"  OPRO generation:     {summary.get('opro_generation', 0)}",
                    f"  OPRO active:         {summary.get('opro_active_candidate', 'N/A')}",
                    f"  OPRO score:          {summary.get('opro_active_score', 0):.1f}",
                    f"  Total reflections:   {summary.get('total_reflections', 0)}",
                    f"  Regime distribution: {summary.get('regime_distribution', {})}",
                    f"  Avg ROI by regime:   {summary.get('avg_roi_by_regime', {})}",
                ]
                audit_stats = summary.get("audit_stats", {})
                if audit_stats:
                    report_lines.extend([
                        f"  Total trades:        {audit_stats.get('total_trades', 0)}",
                        f"  Win rate:            {audit_stats.get('win_rate', 0):.1%}",
                        f"  Avg ROI:             {audit_stats.get('avg_roi', 0):.4f}",
                    ])

                logger.info("Learning Summary:\n%s", "\n".join(report_lines))

                # Also dump to JSON for external dashboards
                report_path = os.path.join(data_dir, "learning_report.json")
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

                # Send daily summary to Telegram
                await notifier.send_daily_summary(summary)

            except Exception:
                logger.exception("Learning report generation failed")

        # ── 4. Configure APScheduler ───────────────────────────────
        scheduler = AsyncIOScheduler()

        # Rule A: Every 15 min during US market hours (UTC 13:30-20:00)
        # Mon-Fri, hours 13-19, every 15 min
        scheduler.add_job(
            trading_job,
            CronTrigger(
                day_of_week="mon-fri",
                hour="13-19",
                minute="*/15",
                timezone="UTC",
            ),
            id="trading_cycle",
            name="15-min market scan",
            misfire_grace_time=300,
        )

        # Rule B: Daily post-market sync & reflection (UTC 20:30)
        scheduler.add_job(
            sync_and_reflect_job,
            CronTrigger(
                day_of_week="mon-fri",
                hour=20,
                minute=30,
                timezone="UTC",
            ),
            id="sync_and_reflect",
            name="Post-market sync & reflect",
            misfire_grace_time=600,
        )

        # Rule C: Daily learning report (UTC 21:00)
        scheduler.add_job(
            learning_report_job,
            CronTrigger(
                day_of_week="mon-fri",
                hour=21,
                minute=0,
                timezone="UTC",
            ),
            id="learning_report",
            name="Daily learning report",
            misfire_grace_time=600,
        )

        scheduler.start()
        logger.info(
            "Scheduler started — %d jobs registered", len(scheduler.get_jobs()),
        )
        for job in scheduler.get_jobs():
            logger.info("  [%s] %s — next run: %s", job.id, job.name, job.next_run_time)

        # ── 5. Graceful shutdown ───────────────────────────────────
        stop_event = asyncio.Event()

        def _handle_shutdown(sig: int, _frame: Any) -> None:
            sig_name = signal.Signals(sig).name
            logger.warning(
                "Received %s — initiating graceful shutdown...", sig_name,
            )
            stop_event.set()

        # Register signal handlers (works on Linux/macOS; Windows SIGTERM via Docker)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle_shutdown)
            except (OSError, ValueError):
                pass  # Some signals unavailable on Windows

        logger.info("System running. Waiting for scheduled jobs... (Ctrl+C to stop)")
        await stop_event.wait()

        # ── 6. Cleanup ─────────────────────────────────────────────
        logger.info("Shutting down scheduler...")
        scheduler.shutdown(wait=True)

        # Persist all state before exit
        logger.info("Persisting final state...")
        account_manager.save_state()
        system.watchdog.stop()

        # Notify shutdown via Telegram
        await notifier.send_system_event("STOP", "Graceful shutdown complete.")
        await notifier.close()

        logger.info("System shut down cleanly.")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure project root is on sys.path so "from src.xxx" imports work
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    _configure_logging()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user — exiting.")
