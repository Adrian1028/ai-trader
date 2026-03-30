"""
AI Trading System — Single Cycle Runner
=========================================
Runs ONE trading cycle then exits. Designed for:
  - Cloud Run Jobs (triggered by Cloud Scheduler)
  - AWS Lambda / GCP Cloud Functions
  - Cron jobs on VMs
  - Manual testing

Usage:
  python src/run_cycle.py                    # Run trading cycle
  python src/run_cycle.py --mode=reflect     # Run reflection only
  python src/run_cycle.py --mode=report      # Run learning report only

Environment variables (same as main.py):
  T212_ENV, T212_API_KEY, POLYGON_KEY, etc. (see .env.example)

Exit codes:
  0 = success
  1 = error
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Load .env file from project root
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.is_file():
        load_dotenv(_env_path)
except ImportError:
    pass

logger = logging.getLogger("RunCycle")

# ── Target symbols ──────────────────────────────────────────────
_DEFAULT_TARGETS = [
    # ── Tech (12) ──
    "US0378331005",   # AAPL
    "US5949181045",   # MSFT
    "US67066G1040",   # NVDA
    "US88160R1014",   # TSLA
    "US0231351067",   # AMZN
    "US02079K3059",   # GOOG
    "US30303M1027",   # META
    "US11135F1012",   # AVGO
    "US12652L1008",   # CRM
    "US0079031078",   # AMD
    "US68389X1054",   # ORCL
    "US4581401001",   # INTC
    # ── Energy (8) ──
    "US30231G1022",   # XOM
    "US1667641005",   # CVX
    "US20825C1045",   # COP
    "US8064071025",   # SLB
    "US26875P1012",   # EOG
    "US65339F1012",   # NEE
    "US29355A1079",   # ENPH
    "US6745991058",   # OXY
]


def _configure_logging() -> None:
    """Lightweight logging for single-cycle mode."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    data_dir = os.getenv("DATA_DIR", "data")
    os.makedirs(data_dir, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    # Only add file handler if data dir is writable (skip in Cloud Run)
    try:
        log_path = os.path.join(data_dir, "trading_bot.log")
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    except (OSError, PermissionError):
        pass  # Read-only filesystem (Cloud Run)

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


async def run_trading_cycle() -> dict:
    """
    Execute one full trading cycle.

    Returns a summary dict with results.
    """
    from src.core.orchestrator import TradingSystem
    from src.core.virtual_account import VirtualAccountManager
    from src.notifications.telegram import TelegramNotifier

    data_dir = os.getenv("DATA_DIR", "data")
    os.makedirs(data_dir, exist_ok=True)

    bot_id = os.getenv("BOT_ID", "AlphaBot_01")
    initial_capital = float(os.getenv("INITIAL_CAPITAL", "10000.0"))
    target_isins = os.getenv("TARGET_ISINS", "").split(",")
    target_isins = [t.strip() for t in target_isins if t.strip()] or _DEFAULT_TARGETS

    account_manager = VirtualAccountManager(
        storage_file=os.path.join(data_dir, "virtual_accounts.json"),
    )

    system = TradingSystem.from_env(
        bot_id=bot_id,
        initial_capital=initial_capital,
        account_manager=account_manager,
    )

    notifier = TelegramNotifier.from_env()

    async with system:
        await system.initialise()

        logger.info("=== Single Trading Cycle ===")
        t0 = time.monotonic()

        results = await system.run_cycle(target_isins)
        elapsed = time.monotonic() - t0

        submitted = sum(1 for r in results if r.get("order_status") == "SUBMITTED")
        vetoed = sum(1 for r in results if r.get("order_status") == "COMPLIANCE_VETOED")
        held = sum(1 for r in results if r.get("action") == "HOLD")

        logger.info(
            "Cycle done in %.1fs — %d submitted, %d vetoed, %d held, %d total",
            elapsed, submitted, vetoed, held, len(results),
        )

        # Send Telegram notifications
        for r in results:
            await notifier.send_trade_alert(r)
        await notifier.send_cycle_summary(results, elapsed)

        # Save state
        account_manager.save_state()
        system.watchdog.stop()
        await notifier.close()

        return {
            "status": "ok",
            "elapsed": elapsed,
            "submitted": submitted,
            "vetoed": vetoed,
            "held": held,
            "total": len(results),
            "results": results,
        }


async def run_reflection() -> dict:
    """Run post-market reflection and fill reconciliation."""
    from src.core.orchestrator import TradingSystem
    from src.core.virtual_account import VirtualAccountManager

    data_dir = os.getenv("DATA_DIR", "data")
    bot_id = os.getenv("BOT_ID", "AlphaBot_01")
    initial_capital = float(os.getenv("INITIAL_CAPITAL", "10000.0"))

    account_manager = VirtualAccountManager(
        storage_file=os.path.join(data_dir, "virtual_accounts.json"),
    )

    system = TradingSystem.from_env(
        bot_id=bot_id,
        initial_capital=initial_capital,
        account_manager=account_manager,
    )

    async with system:
        await system.initialise()

        logger.info("=== Post-Market Reflection ===")

        # Check fills
        for ticket_id, ticket in list(system._executor.active_tickets.items()):
            if ticket.api_order_id is not None:
                await system._executor.check_fill(ticket)

        account_manager.save_state()
        system.watchdog.stop()

        return {"status": "ok", "mode": "reflection"}


async def run_learning_report() -> dict:
    """Generate daily learning report."""
    from src.core.orchestrator import TradingSystem
    from src.core.virtual_account import VirtualAccountManager
    from src.notifications.telegram import TelegramNotifier

    data_dir = os.getenv("DATA_DIR", "data")
    bot_id = os.getenv("BOT_ID", "AlphaBot_01")
    initial_capital = float(os.getenv("INITIAL_CAPITAL", "10000.0"))

    account_manager = VirtualAccountManager(
        storage_file=os.path.join(data_dir, "virtual_accounts.json"),
    )

    system = TradingSystem.from_env(
        bot_id=bot_id,
        initial_capital=initial_capital,
        account_manager=account_manager,
    )

    notifier = TelegramNotifier.from_env()

    async with system:
        await system.initialise()

        logger.info("=== Daily Learning Report ===")
        summary = system.learning_summary()

        # Save to JSON
        report_path = os.path.join(data_dir, "learning_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

        # Telegram
        await notifier.send_daily_summary(summary)
        await notifier.close()
        system.watchdog.stop()

        logger.info("Learning report saved: %s", report_path)
        return {"status": "ok", "mode": "report", "summary": summary}


if __name__ == "__main__":
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    _configure_logging()

    # Parse mode from command line
    mode = "trade"
    for arg in sys.argv[1:]:
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]

    try:
        if mode == "reflect":
            result = asyncio.run(run_reflection())
        elif mode == "report":
            result = asyncio.run(run_learning_report())
        else:
            result = asyncio.run(run_trading_cycle())

        logger.info("Run complete: %s", result.get("status", "unknown"))
        sys.exit(0)

    except Exception:
        logger.exception("Run failed")
        sys.exit(1)
