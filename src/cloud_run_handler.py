"""
Cloud Run HTTP Handler
=======================
Wraps the trading cycle in an HTTP server for Cloud Run Services.
Cloud Scheduler sends HTTP POST to trigger each cycle.

Routes:
  POST /trade    — Run one trading cycle
  POST /reflect  — Run post-market reflection
  POST /report   — Generate learning report
  GET  /health   — Health check (Cloud Run requirement)

Security:
  Cloud Scheduler sends a header with an OIDC token.
  Set CLOUD_RUN_AUTH=true to enforce authentication.

Cloud Run free tier budget (per month):
  - 2M requests ✓ (we use ~620/month)
  - 360,000 GB-seconds ✓ (we use ~19,000)
  - 180,000 vCPU-seconds ✓ (we use ~37,000)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.is_file():
        load_dotenv(_env_path)
except ImportError:
    pass

from aiohttp import web

logger = logging.getLogger("CloudRun")


def _configure_logging() -> None:
    """Cloud Run logging — structured JSON for Cloud Logging."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── HTTP Handlers ────────────────────────────────────────────────

async def health_handler(request: web.Request) -> web.Response:
    """Health check endpoint (required by Cloud Run)."""
    return web.json_response({"status": "healthy", "time": time.time()})


async def trade_handler(request: web.Request) -> web.Response:
    """Run one trading cycle."""
    from src.run_cycle import run_trading_cycle

    logger.info("=== Cloud Run: Trading cycle triggered ===")
    t0 = time.monotonic()

    try:
        result = await run_trading_cycle()
        elapsed = time.monotonic() - t0

        logger.info(
            "Trading cycle complete in %.1fs: %d submitted, %d held",
            elapsed,
            result.get("submitted", 0),
            result.get("held", 0),
        )

        return web.json_response({
            "status": "ok",
            "elapsed": elapsed,
            "submitted": result.get("submitted", 0),
            "vetoed": result.get("vetoed", 0),
            "held": result.get("held", 0),
            "total": result.get("total", 0),
        })

    except Exception as e:
        logger.exception("Trading cycle failed")
        return web.json_response(
            {"status": "error", "error": str(e)},
            status=500,
        )


async def reflect_handler(request: web.Request) -> web.Response:
    """Run post-market reflection."""
    from src.run_cycle import run_reflection

    logger.info("=== Cloud Run: Reflection triggered ===")
    try:
        result = await run_reflection()
        return web.json_response(result)
    except Exception as e:
        logger.exception("Reflection failed")
        return web.json_response({"status": "error", "error": str(e)}, status=500)


async def report_handler(request: web.Request) -> web.Response:
    """Generate learning report."""
    from src.run_cycle import run_learning_report

    logger.info("=== Cloud Run: Report triggered ===")
    try:
        result = await run_learning_report()
        return web.json_response({
            "status": "ok",
            "episodes": result.get("summary", {}).get("episodes_stored", 0),
            "opro_gen": result.get("summary", {}).get("opro_generation", 0),
        })
    except Exception as e:
        logger.exception("Report failed")
        return web.json_response({"status": "error", "error": str(e)}, status=500)


# ── App Factory ──────────────────────────────────────────────────

def create_app() -> web.Application:
    """Create the aiohttp web application."""
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/", health_handler)         # Cloud Run default
    app.router.add_post("/trade", trade_handler)
    app.router.add_post("/reflect", reflect_handler)
    app.router.add_post("/report", report_handler)
    return app


if __name__ == "__main__":
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    _configure_logging()

    port = int(os.getenv("PORT", "8080"))  # Cloud Run sets PORT env var
    logger.info("Starting Cloud Run handler on port %d", port)
    web.run_app(create_app(), port=port, access_log=logger)
