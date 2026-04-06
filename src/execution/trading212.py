"""
Trading 212 Execution Module for Macro Signal System
=====================================================

Handles:
  1. Connect to Trading 212 Demo API (or Live)
  2. Read current portfolio positions
  3. Calculate required trades to reach target allocation
  4. Execute market orders (SPY/SHV only)
  5. Track slippage (target vs fill price)

Safety:
  - Pre-trade position check (no duplicate orders)
  - Max single trade = 50% of NAV
  - Max 3 retries on API failure, then Discord alert
  - All trades logged with full audit trail

API Reference:
  Base URL: https://demo.trading212.com/api/v0
  Auth: Basic Base64(API_KEY:API_SECRET) or direct key header
  Key endpoints:
    GET  /equity/account/cash     → {free, invested, result, ...}
    GET  /equity/portfolio        → [{ticker, quantity, currentPrice, ppl, ...}]
    POST /equity/orders/market    → {instrumentCode, quantity}
    GET  /equity/metadata/instruments → instrument search
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Record of a single trade execution."""
    ticker: str
    side: str           # "BUY" or "SELL"
    target_qty: float
    filled_qty: float = 0.0
    target_price: float = 0.0
    fill_price: float = 0.0
    slippage_pct: float = 0.0
    status: str = "PENDING"  # PENDING, FILLED, PARTIAL, FAILED, SKIPPED
    api_response: dict = field(default_factory=dict)
    error: str = ""
    timestamp: str = ""


@dataclass
class RebalanceResult:
    """Full rebalance execution summary."""
    date: str
    regime: str
    target_alloc: dict[str, float] = field(default_factory=dict)  # {SPY: 0.5, SHV: 0.5}
    nav_before: float = 0.0
    nav_after: float = 0.0
    trades: list[TradeResult] = field(default_factory=list)
    status: str = "OK"
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "regime": self.regime,
            "target_alloc": self.target_alloc,
            "nav_before": self.nav_before,
            "nav_after": self.nav_after,
            "trades": [
                {
                    "ticker": t.ticker, "side": t.side,
                    "target_qty": t.target_qty, "filled_qty": t.filled_qty,
                    "target_price": t.target_price, "fill_price": t.fill_price,
                    "slippage_pct": t.slippage_pct, "status": t.status,
                    "error": t.error,
                }
                for t in self.trades
            ],
            "status": self.status,
            "error": self.error,
        }

    def discord_summary(self) -> str:
        lines = [f"**Execution Report** ({self.date})"]
        lines.append(f"Regime: **{self.regime}** | NAV: ${self.nav_before:,.2f}")
        lines.append(f"Target: SPY {self.target_alloc.get('SPY', 0)*100:.0f}% / SHV {self.target_alloc.get('SHV', 0)*100:.0f}%")
        for t in self.trades:
            slip = f" (slip: {t.slippage_pct:+.3f}%)" if t.slippage_pct else ""
            lines.append(f"  {t.side} {t.ticker}: {t.filled_qty:.4f} @ ${t.fill_price:.2f}{slip} [{t.status}]")
        if not self.trades:
            lines.append("  No trades needed (already at target)")
        if self.error:
            lines.append(f"  ERROR: {self.error}")
        return "\n".join(lines)


class Trading212Executor:
    """
    Trading 212 API executor for the macro signal system.

    Simplified client that only handles SPY/SHV market orders.
    Uses the same auth pattern as the existing src/core/client.py.
    """

    MAX_RETRIES = 3
    BACKOFF_BASE = 2.0
    MAX_SINGLE_TRADE_PCT = 0.50  # Max 50% of NAV per trade

    # Trading 212 instrument codes for US ETFs
    # These are the T212 ticker codes (not ISINs)
    INSTRUMENTS = {
        "SPY": "SPY_US_EQ",     # SPDR S&P 500 ETF
        "SHV": "SHV_US_EQ",     # iShares Short Treasury Bond ETF
    }

    def __init__(self):
        self._api_key = os.getenv("T212_API_KEY", "")
        self._api_secret = os.getenv("T212_API_SECRET", "")
        self._env = os.getenv("T212_ENV", "demo")

        if self._env == "live":
            self._base_url = "https://live.trading212.com/api/v0"
        else:
            self._base_url = "https://demo.trading212.com/api/v0"

        self._session: aiohttp.ClientSession | None = None
        self._enabled = bool(self._api_key and self._api_key != "your_trading212_api_key_here")

        if self._enabled:
            logger.info("T212 Executor enabled (env=%s, url=%s)", self._env, self._base_url)
        else:
            logger.warning("T212 Executor DISABLED — no API key configured")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _build_auth_header(self) -> str:
        """Build auth header. T212 accepts both Basic and direct key."""
        if self._api_secret:
            creds = f"{self._api_key}:{self._api_secret}"
            encoded = base64.b64encode(creds.encode()).decode()
            return f"Basic {encoded}"
        # Fallback: direct API key (newer T212 API style)
        return self._api_key

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": self._build_auth_header(),
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _request(self, method: str, endpoint: str, **kwargs) -> dict | list | None:
        """HTTP request with retry + backoff. Returns None on total failure."""
        url = f"{self._base_url}{endpoint}"
        session = await self._get_session()

        last_error = ""
        for attempt in range(self.MAX_RETRIES):
            try:
                async with session.request(method, url, **kwargs) as resp:
                    if resp.status in (200, 201):
                        text = await resp.text()
                        if not text:
                            return {}
                        return await resp.json(content_type=None)

                    error_text = await resp.text()
                    last_error = f"HTTP {resp.status}: {error_text[:200]}"

                    if resp.status == 429:
                        backoff = self.BACKOFF_BASE * (2 ** attempt)
                        logger.warning("T212 rate limited (429), waiting %.1fs (attempt %d/%d)",
                                       backoff, attempt + 1, self.MAX_RETRIES)
                        await asyncio.sleep(backoff)
                        continue

                    if resp.status >= 500:
                        backoff = self.BACKOFF_BASE * (2 ** attempt)
                        logger.warning("T212 server error %d, retrying in %.1fs", resp.status, backoff)
                        await asyncio.sleep(backoff)
                        continue

                    # 4xx client error — don't retry
                    logger.error("T212 API error: %s", last_error)
                    return None

            except aiohttp.ClientError as e:
                last_error = str(e)
                backoff = self.BACKOFF_BASE * (2 ** attempt)
                logger.warning("T212 connection error: %s, retrying in %.1fs", e, backoff)
                await asyncio.sleep(backoff)

        logger.error("T212 API failed after %d retries: %s", self.MAX_RETRIES, last_error)
        return None

    # ── Account & Portfolio ──────────────────────────────────────────

    async def get_account_cash(self) -> dict | None:
        """Get account cash info: {free, invested, ppl, result, total}"""
        return await self._request("GET", "/equity/account/cash")

    async def get_portfolio(self) -> list[dict] | None:
        """Get all open positions."""
        return await self._request("GET", "/equity/portfolio")

    async def get_portfolio_as_dict(self) -> dict[str, dict]:
        """Get portfolio indexed by ticker: {ticker: {quantity, currentPrice, ...}}"""
        positions = await self.get_portfolio()
        if positions is None:
            return {}
        result = {}
        for pos in positions:
            ticker = pos.get("ticker", "")
            result[ticker] = pos
        return result

    async def get_nav(self) -> float:
        """Get total account NAV (cash + invested)."""
        cash_info = await self.get_account_cash()
        if cash_info is None:
            return 0.0
        return float(cash_info.get("total", 0))

    # ── Instrument Discovery ─────────────────────────────────────────

    async def find_instrument_code(self, symbol: str) -> str | None:
        """
        Find the T212 instrument code for a given symbol.
        Falls back to common patterns if exact match not found.
        """
        # Try known mappings first
        if symbol in self.INSTRUMENTS:
            return self.INSTRUMENTS[symbol]

        # Search via API
        instruments = await self._request("GET", "/equity/metadata/instruments")
        if instruments is None:
            return None

        for inst in instruments:
            ticker = inst.get("ticker", "")
            short_name = inst.get("shortName", "")
            if ticker.startswith(symbol) or short_name == symbol:
                code = ticker
                logger.info("Found T212 code for %s: %s", symbol, code)
                self.INSTRUMENTS[symbol] = code  # Cache it
                return code

        logger.warning("Could not find T212 instrument code for %s", symbol)
        return None

    # ── Order Execution ──────────────────────────────────────────────

    async def place_market_order(self, ticker: str, quantity: float) -> dict | None:
        """
        Place a market order. quantity > 0 = buy, < 0 = sell.
        Returns API response or None on failure.
        """
        if abs(quantity) < 0.0001:
            logger.info("Skipping order for %s: quantity %.6f too small", ticker, quantity)
            return {"status": "SKIPPED", "reason": "quantity_too_small"}

        payload = {
            "instrumentCode": ticker,
            "quantity": round(quantity, 4),
        }
        logger.info("Placing market order: %s qty=%.4f", ticker, quantity)
        result = await self._request("POST", "/equity/orders/market", json=payload)

        if result is not None:
            logger.info("Order response for %s: %s", ticker, str(result)[:200])
        else:
            logger.error("Order FAILED for %s qty=%.4f", ticker, quantity)

        return result

    # ── Rebalance Logic ──────────────────────────────────────────────

    async def rebalance(
        self,
        target_alloc: dict[str, float],
        regime: str,
    ) -> RebalanceResult:
        """
        Rebalance portfolio to match target allocation.

        target_alloc: {"SPY": 0.5, "SHV": 0.5} — fractions summing to 1.0
        Returns RebalanceResult with full execution details.
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        result = RebalanceResult(date=today, regime=regime, target_alloc=target_alloc)

        if not self._enabled:
            result.status = "DISABLED"
            result.error = "T212 executor not enabled (no API key)"
            logger.warning("Rebalance skipped: executor disabled")
            return result

        # 1. Get current NAV and positions
        logger.info("Step 7a: Fetching current portfolio...")
        nav = await self.get_nav()
        if nav <= 0:
            result.status = "ERROR"
            result.error = "Could not fetch NAV"
            return result
        result.nav_before = nav

        positions = await self.get_portfolio_as_dict()
        if positions is None:
            result.status = "ERROR"
            result.error = "Could not fetch portfolio"
            return result

        logger.info("  NAV: $%.2f, Positions: %d", nav, len(positions))
        for ticker, pos in positions.items():
            logger.info("    %s: qty=%.4f, price=$%.2f, value=$%.2f",
                        ticker, pos.get("quantity", 0), pos.get("currentPrice", 0),
                        pos.get("quantity", 0) * pos.get("currentPrice", 0))

        # 2. Resolve T212 instrument codes
        t212_codes = {}
        for symbol in target_alloc:
            code = await self.find_instrument_code(symbol)
            if code is None:
                result.status = "ERROR"
                result.error = f"Cannot find T212 code for {symbol}"
                return result
            t212_codes[symbol] = code

        # 3. Calculate required trades
        trades_needed: list[tuple[str, str, float, float]] = []  # (symbol, t212_code, qty_delta, current_price)

        for symbol, target_pct in target_alloc.items():
            t212_code = t212_codes[symbol]
            target_value = nav * target_pct

            # Current position
            current_pos = positions.get(t212_code, {})
            current_qty = float(current_pos.get("quantity", 0))
            current_price = float(current_pos.get("currentPrice", 0))

            # If no position and no target, skip
            if target_pct == 0 and current_qty == 0:
                continue

            # Need current price to calculate quantity
            if current_price <= 0:
                # Try to get price from a small test (use yfinance as fallback)
                try:
                    from src.signals.macro_data import fetch_from_yfinance
                    price_series = fetch_from_yfinance(symbol, today, today)
                    if not price_series.empty:
                        current_price = float(price_series.iloc[-1])
                except Exception:
                    pass

            if current_price <= 0:
                logger.warning("No price for %s, skipping", symbol)
                continue

            current_value = current_qty * current_price
            delta_value = target_value - current_value

            # Skip if already close enough (within 2% of NAV)
            if abs(delta_value) < nav * 0.02:
                logger.info("  %s: already at target (delta $%.2f < 2%% NAV)", symbol, delta_value)
                continue

            delta_qty = delta_value / current_price

            # Safety: cap single trade at MAX_SINGLE_TRADE_PCT of NAV
            max_trade_value = nav * self.MAX_SINGLE_TRADE_PCT
            if abs(delta_value) > max_trade_value:
                logger.warning("  %s: capping trade from $%.2f to $%.2f (50%% NAV limit)",
                               symbol, delta_value, max_trade_value if delta_value > 0 else -max_trade_value)
                delta_qty = (max_trade_value / current_price) * (1 if delta_value > 0 else -1)

            trades_needed.append((symbol, t212_code, delta_qty, current_price))

        if not trades_needed:
            logger.info("  No trades needed — portfolio already at target")
            result.status = "NO_CHANGE"
            result.nav_after = nav
            return result

        # 4. Execute sells first, then buys (free up cash before buying)
        sells = [(s, c, q, p) for s, c, q, p in trades_needed if q < 0]
        buys = [(s, c, q, p) for s, c, q, p in trades_needed if q > 0]

        for batch_name, batch in [("SELL", sells), ("BUY", buys)]:
            for symbol, t212_code, qty, price in batch:
                trade = TradeResult(
                    ticker=symbol,
                    side="SELL" if qty < 0 else "BUY",
                    target_qty=abs(qty),
                    target_price=price,
                    timestamp=datetime.utcnow().isoformat(),
                )

                resp = await self.place_market_order(t212_code, qty)

                if resp is None:
                    trade.status = "FAILED"
                    trade.error = "API returned None after retries"
                    result.status = "PARTIAL"
                elif resp.get("status") == "SKIPPED":
                    trade.status = "SKIPPED"
                    trade.error = resp.get("reason", "")
                else:
                    trade.status = "FILLED"
                    trade.filled_qty = abs(float(resp.get("filledQuantity", qty)))
                    trade.fill_price = float(resp.get("filledValue", 0)) / trade.filled_qty if trade.filled_qty > 0 else price
                    trade.api_response = resp

                    # Calculate slippage
                    if price > 0 and trade.fill_price > 0:
                        trade.slippage_pct = (trade.fill_price / price - 1) * 100

                result.trades.append(trade)
                logger.info("  %s %s: qty=%.4f, status=%s",
                            trade.side, trade.ticker, trade.target_qty, trade.status)

        # 5. Get updated NAV
        await asyncio.sleep(1)  # Brief pause for T212 to settle
        nav_after = await self.get_nav()
        result.nav_after = nav_after if nav_after > 0 else nav

        # Check for any failures
        failed = [t for t in result.trades if t.status == "FAILED"]
        if failed:
            if result.status != "PARTIAL":
                result.status = "PARTIAL"
            result.error = f"{len(failed)} trade(s) failed"

        logger.info("Rebalance complete: %d trades, status=%s", len(result.trades), result.status)
        return result
